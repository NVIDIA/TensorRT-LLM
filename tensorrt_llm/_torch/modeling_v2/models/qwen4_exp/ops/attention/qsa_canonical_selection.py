# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA block selection, made canonical: one answer per row, whichever tied columns the
selection engine happened to keep.

Tier 5 (Triton). No registered op rewrites a selection, and none
of the top-k engines (`indexer_topk_prefill`, the CuTe DSL kernels, FlashInfer's top-k)
takes a tie-break rule or fixes the order of its output. A row's k-th value is read off
the engine's picks; the window is counted against it once and compacted into the row in
a second, early-exiting pass. Only the window's columns are read, and nothing
`[rows, cols]`-sized is allocated.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's block budget: `indexer_budget` 2048 tokens over
#: `indexer_compress_ratio` 4.
_BLOCK_TOPK = 512
#: Score-row width the target hands over: a 260-page block table at 32 tokens
#: per block, compressed 4:1.
_SCORE_COLUMNS = 260 * 32 // 4
#: The score-row width at the 256k-token context: 65536 compressed blocks.
_LONG_SCORE_COLUMNS = 262144 // 4
#: From this many rows on, one program per row keeps the GPU busy and rereads its window
#: from L2 (`_canonical_rows_kernel`). Below it a row's window is cut into segments of
#: `_SEGMENT` columns, each its own program, so a decode-sized launch is not one program
#: streaming a 256 KiB row.
_SPLIT_ROWS = 1024
_SEGMENT = 4096
#: Columns per step of the counting pass, of the writing pass (whose in-block prefix sum
#: costs more per column), and warps per program, tuned on GB300 for both launch shapes.
_ROWS_COUNT_BLOCK = 2048
_SPLIT_COUNT_BLOCK = 4096
_WRITE_BLOCK = 1024
_NUM_WARPS = 4


def _cell(why: str, rows, windows, logits: str, tail: str = "nan", **extra) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            rows=rows,
            cols=extra.pop("cols", _SCORE_COLUMNS),
            index_topk=_BLOCK_TOPK,
            windows=windows,
            logits=logits,
            tail=tail,
            selection=extra.pop("selection", "random"),
            **extra,
        ),
    )


@triton.jit
def _kth_value(
    row_logits,
    row_indices,
    start,
    num_cols,
    stride_col,
    stride_index,
    K: tl.constexpr,
    K_P2: tl.constexpr,
):
    """The smallest score the engine kept, read at the clamped column `start + pick`: an
    empty slot (`pick < 0`) reads +inf, and a NaN pick makes it NaN, which no score
    compares to, so the row comes out empty."""
    slots = tl.arange(0, K_P2)
    in_k = slots < K
    picks = tl.load(row_indices + slots * stride_index, mask=in_k, other=-1)
    columns = tl.minimum(tl.maximum(picks + start, 0), num_cols - 1)
    picked = tl.load(
        row_logits + columns.to(tl.int64) * stride_col,
        mask=in_k & (picks >= 0),
        other=float("inf"),
    ).to(tl.float32)
    kth = tl.min(picked, axis=0)
    picked_nan = tl.max((picked != picked).to(tl.int32), axis=0)
    return tl.where(picked_nan > 0, float("nan"), kth)


@triton.jit
def _load_window(row_logits, cols, lo, hi, stride_col):
    # NaN past the window: it compares false, so it is never counted or kept.
    return tl.load(
        row_logits + cols.to(tl.int64) * stride_col,
        mask=(cols >= lo) & (cols < hi),
        other=float("nan"),
    ).to(tl.float32)


@triton.jit
def _count(x, kth):
    """Columns of `x` above the k-th value and tied with it (float equality: -0.0 == +0.0,
    NaN equals nothing)."""
    return tl.sum((x > kth).to(tl.int32), axis=0), tl.sum((x == kth).to(tl.int32), axis=0)


@triton.jit
def _write_block(
    row_indices,
    stride_index,
    cols,
    start,
    x,
    kth,
    drop,
    kept,
    ties_seen,
    K: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
):
    """Compact one block of the window into the row; returns the updated running counts.

    Kept: every column above the k-th value, and every tie whose 0-based rank among the
    window's ties is at least `drop` -- the lowest-index `drop` ties are the surplus. The
    j-th kept column, ascending, goes to slot j as a window-local offset; slots from k on
    are not written. One prefix sum carries both counts, above in the high half-word."""
    tl.static_assert(WRITE_BLOCK < 65536)
    is_above = (x > kth).to(tl.int32)
    is_tied = (x == kth).to(tl.int32)
    packed = is_above * 65536 + is_tied
    before = tl.cumsum(packed, axis=0) - packed
    ties_before = ties_seen + (before & 65535)
    kept_ties = tl.maximum(ties_seen - drop, 0)
    slot = kept + (before >> 16) + tl.maximum(ties_before - drop, 0) - kept_ties
    keep = (is_above != 0) | ((is_tied != 0) & (ties_before >= drop))
    tl.store(row_indices + slot * stride_index, cols - start, mask=keep & (slot < K))
    block = tl.sum(packed, axis=0)
    ties_seen += block & 65535
    kept += (block >> 16) + tl.maximum(ties_seen - drop, 0) - kept_ties
    return kept, ties_seen


@triton.jit
def _canonical_rows_kernel(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    indices_ptr,
    num_cols,
    stride_starts,
    stride_ends,
    stride_logits_row,
    stride_logits_col,
    stride_indices_row,
    stride_indices_col,
    K: tl.constexpr,
    K_P2: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
):
    # One program per row: both passes, the second reading the window back from L2. Each
    # loop loads its next block before reducing the current one. Every pick is read
    # before the k-th value's reduction completes and every store depends on it, so the
    # in-place rewrite never overwrites a pick another thread has yet to read.
    row = tl.program_id(0).to(tl.int64)
    row_logits = logits_ptr + row * stride_logits_row
    row_indices = indices_ptr + row * stride_indices_row
    start = tl.load(starts_ptr + row * stride_starts).to(tl.int32)
    end = tl.load(ends_ptr + row * stride_ends).to(tl.int32)
    kth = _kth_value(
        row_logits, row_indices, start, num_cols, stride_logits_col, stride_indices_col, K, K_P2
    )
    # The window's columns inside the matrix, in blocks aligned to the block size so the
    # loads vectorize.
    lo = tl.maximum(start, 0)
    hi = tl.minimum(end, num_cols)

    above = tl.zeros((), dtype=tl.int32)
    tied = tl.zeros((), dtype=tl.int32)
    first = (lo // COUNT_BLOCK) * COUNT_BLOCK
    x = _load_window(row_logits, first + tl.arange(0, COUNT_BLOCK), lo, hi, stride_logits_col)
    for c0 in range(first, hi, COUNT_BLOCK):
        c0 = tl.multiple_of(c0, COUNT_BLOCK)
        cols = c0 + COUNT_BLOCK + tl.arange(0, COUNT_BLOCK)
        x_next = _load_window(row_logits, cols, lo, hi, stride_logits_col)
        block_above, block_tied = _count(x, kth)
        above += block_above
        tied += block_tied
        x = x_next
    # A window of at most k has no surplus.
    drop = tl.maximum(above + tied - K, 0)

    kept = tl.zeros((), dtype=tl.int32)
    ties_seen = tl.zeros((), dtype=tl.int32)
    c0 = (lo // WRITE_BLOCK) * WRITE_BLOCK
    x = _load_window(row_logits, c0 + tl.arange(0, WRITE_BLOCK), lo, hi, stride_logits_col)
    while (c0 < hi) & (kept < K):
        c0 = tl.multiple_of(c0, WRITE_BLOCK)
        cols = c0 + tl.arange(0, WRITE_BLOCK)
        x_next = _load_window(row_logits, cols + WRITE_BLOCK, lo, hi, stride_logits_col)
        kept, ties_seen = _write_block(
            row_indices,
            stride_indices_col,
            cols,
            start,
            x,
            kth,
            drop,
            kept,
            ties_seen,
            K,
            WRITE_BLOCK,
        )
        x = x_next
        c0 += WRITE_BLOCK
    slots = tl.arange(0, K_P2)
    tl.store(row_indices + slots * stride_indices_col, -1, mask=(slots < K) & (slots >= kept))


@triton.jit
def _split_kth_kernel(
    logits_ptr,
    starts_ptr,
    indices_ptr,
    kth_ptr,
    num_cols,
    stride_starts,
    stride_logits_row,
    stride_logits_col,
    stride_indices_row,
    stride_indices_col,
    K: tl.constexpr,
    K_P2: tl.constexpr,
):
    # Its own launch: the segments' programs that write the row come after every pick of
    # it has been read.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts_ptr + row * stride_starts).to(tl.int32)
    kth = _kth_value(
        logits_ptr + row * stride_logits_row,
        indices_ptr + row * stride_indices_row,
        start,
        num_cols,
        stride_logits_col,
        stride_indices_col,
        K,
        K_P2,
    )
    tl.store(kth_ptr + row, kth)


@triton.jit
def _split_count_kernel(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    kth_ptr,
    counts_ptr,
    num_rows,
    num_cols,
    num_segments,
    stride_starts,
    stride_ends,
    stride_logits_row,
    stride_logits_col,
    SEGMENT: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
):
    # One program per (row, segment): the segment's window columns above and tied with
    # the k-th value, into `counts[0 | 1, row, segment]`.
    row = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(1)
    row_logits = logits_ptr + row * stride_logits_row
    seg_lo = segment * SEGMENT
    start = tl.load(starts_ptr + row * stride_starts).to(tl.int32)
    end = tl.load(ends_ptr + row * stride_ends).to(tl.int32)
    lo = tl.maximum(tl.maximum(start, 0), seg_lo)
    hi = tl.minimum(tl.minimum(end, num_cols), seg_lo + SEGMENT)
    kth = tl.load(kth_ptr + row)
    above = tl.zeros((), dtype=tl.int32)
    tied = tl.zeros((), dtype=tl.int32)
    for c0 in range(seg_lo, hi, COUNT_BLOCK):
        c0 = tl.multiple_of(c0, COUNT_BLOCK)
        x = _load_window(row_logits, c0 + tl.arange(0, COUNT_BLOCK), lo, hi, stride_logits_col)
        block_above, block_tied = _count(x, kth)
        above += block_above
        tied += block_tied
    tl.store(counts_ptr + row * num_segments + segment, above)
    tl.store(counts_ptr + (num_rows + row) * num_segments + segment, tied)


@triton.jit
def _split_write_kernel(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    kth_ptr,
    counts_ptr,
    indices_ptr,
    num_rows,
    num_cols,
    num_segments,
    stride_starts,
    stride_ends,
    stride_logits_row,
    stride_logits_col,
    stride_indices_row,
    stride_indices_col,
    K: tl.constexpr,
    K_P2: tl.constexpr,
    SEGMENT: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    COUNTS_BLOCK: tl.constexpr,
):
    # One program per (row, segment): the row's totals and the counts of the segments
    # before this one place the segment's kept columns; segment 0 also writes the `-1`s.
    row = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(1)
    row_logits = logits_ptr + row * stride_logits_row
    row_indices = indices_ptr + row * stride_indices_row
    start = tl.load(starts_ptr + row * stride_starts).to(tl.int32)
    end = tl.load(ends_ptr + row * stride_ends).to(tl.int32)
    seg_lo = segment * SEGMENT
    lo = tl.maximum(tl.maximum(start, 0), seg_lo)
    hi = tl.minimum(tl.minimum(end, num_cols), seg_lo + SEGMENT)

    above = tl.zeros((), dtype=tl.int32)
    tied = tl.zeros((), dtype=tl.int32)
    above_before = tl.zeros((), dtype=tl.int32)
    ties_before = tl.zeros((), dtype=tl.int32)
    for j0 in range(0, num_segments, COUNTS_BLOCK):
        j = j0 + tl.arange(0, COUNTS_BLOCK)
        seg_above = tl.load(counts_ptr + row * num_segments + j, mask=j < num_segments, other=0)
        seg_tied = tl.load(
            counts_ptr + (num_rows + row) * num_segments + j, mask=j < num_segments, other=0
        )
        above += tl.sum(seg_above, axis=0)
        tied += tl.sum(seg_tied, axis=0)
        above_before += tl.sum(tl.where(j < segment, seg_above, 0), axis=0)
        ties_before += tl.sum(tl.where(j < segment, seg_tied, 0), axis=0)
    drop = tl.maximum(above + tied - K, 0)
    if segment == 0:
        total = above + tl.maximum(tied - drop, 0)
        slots = tl.arange(0, K_P2)
        tl.store(row_indices + slots * stride_indices_col, -1, mask=(slots < K) & (slots >= total))

    kth = tl.load(kth_ptr + row)
    kept = above_before + tl.maximum(ties_before - drop, 0)
    ties_seen = ties_before
    c0 = seg_lo
    while (c0 < hi) & (kept < K):
        c0 = tl.multiple_of(c0, WRITE_BLOCK)
        cols = c0 + tl.arange(0, WRITE_BLOCK)
        x = _load_window(row_logits, cols, lo, hi, stride_logits_col)
        kept, ties_seen = _write_block(
            row_indices,
            stride_indices_col,
            cols,
            start,
            x,
            kth,
            drop,
            kept,
            ties_seen,
            K,
            WRITE_BLOCK,
        )
        c0 += WRITE_BLOCK


class QsaCanonicalSelection(OpWrapper):
    """Rewrite an exact top-k selection, in place, into the window's canonical selection.

    For row `r` with window `[s, e) = [row_starts[r], row_ends[r])` of `logits` and
    `k = indices.shape[1]`, `indices[r]` holds on entry what a top-k engine wrote:
    window-local offsets of an exact top-k of the window, in any order, `-1` past
    `min(k, e - s)`. Let `v` be the smallest selected score. On return the row is

    * every window column scoring strictly above `v`, then the highest-index window
      columns scoring exactly `v` until there are `k` -- the k largest
      `(score, column)` pairs; or the whole window when `e - s <= k`;
    * as window-local offsets in ascending order, `-1` in every remaining slot.

    So the row is a function of `(logits, row_starts, row_ends, k)` alone: two engines,
    or two launches of one engine, that broke a tie at the k-th value differently produce
    the same row. On a row with no tie at the k-th value this is the engine's own set,
    sorted. Ties are float equality: `-0.0` and `+0.0` are one tie class. Returns
    `indices`.

    Fusion boundary: the rewrite only. Selecting is the engine's call before it; the
    engine must be exact -- `v` is read off its picks, and a pick below the true k-th
    value would admit more than `k` columns above it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls this right after `qsa_topk_select`, at `k = 512` over the
    # `[rows, cols]` score matrix whose columns past each `row_end` are allocator
    # garbage (the cells poison them). `selection` names what produced the input row:
    # `random` is an exact top-k that keeps a random subset of the tie class at the k-th
    # value, in a random order (the adversary); `radix` is the engine the target calls.
    CELLS: tuple[Cell, ...] = (
        _cell(
            "decode: 1, 128 and 256 rows (MTP3 and MTP7 at batch 32)",
            (1, 128, 256),
            ("positions_random", 2051, 8191),
            "relu",
        ),
        _cell(
            "prefill: 8192 rows from position 0",
            8192,
            ("chunk", 0),
            "relu",
        ),
        _cell(
            "input from the radix engine (qsa_topk_select), zero ties",
            64,
            ("positions_random", 2051, 8191),
            "relu_sparse",
            selection="radix",
        ),
        _cell(
            "decode at up to 256k context: 65536-wide table, relu-zero ties",
            (1, 16, 64),
            ("positions_random", 2051, 262143),
            "relu_sparse",
            cols=_LONG_SCORE_COLUMNS,
        ),
        _cell(
            "prefill: a 2048-row chunk ending at the 256k context",
            2048,
            ("chunk", 262144 - 2048),
            "relu",
            cols=_LONG_SCORE_COLUMNS,
        ),
    )

    note = """
    Metadata consumed: none. Stateless; every input is an argument, and `indices` is
    both the engine's selection read and the canonical row written.

    Preconditions (none checked):
      - `indices` holds an EXACT top-k of each window: the smallest selected score is
        the window's k-th largest. `qsa_topk_select` is exact; an approximate selection
        would be rewritten into a row with more than `k` columns at or above its k-th
        pick, truncated to the lowest `k` column indices -- wrong, silently.
      - `0 <= row_starts[r] <= row_ends[r] <= logits.shape[1]`. Anything outside the
        window is never read for a decision, so garbage, NaN and +inf there are fine
        (the cells poison it).
      - Scores inside the window are not NaN: a NaN column is never kept, and a NaN pick
        empties the row. (The engine returns the identity for a window of at most k
        without reading it; this rewrite reads it.)

    Cost: no host read, no data-dependent shape, so it can be captured into a CUDA graph
    (the entry's test replays one). A row reads its k picks, its window once to count
    and again, up to its k-th kept column, to write; columns outside the window are not
    read, so narrowing `logits` to a bound on the windows (`logits[:, :bound]`) gives
    the same rows and saves nothing. From 1024 rows: one launch, one program per row.
    Below: three launches -- the k-th values, then counts and writes over 4096-column
    segments, each its own program -- through `[rows]` fp32 and `[2, rows, cols / 4096]`
    int32 scratch.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        num_rows, num_cols = logits.shape
        index_topk = indices.shape[1]
        if num_rows == 0 or index_topk == 0:
            return indices
        # A one-element window broadcasts over every row -- the hazard `is_valid`
        # refuses -- as a stride-0 view rather than a read past the tensor's end.
        row_starts = row_starts.expand(num_rows)
        row_ends = row_ends.expand(num_rows)
        k_p2 = triton.next_power_of_2(index_topk)
        if num_rows >= _SPLIT_ROWS:
            _canonical_rows_kernel[(num_rows,)](
                logits,
                row_starts,
                row_ends,
                indices,
                num_cols,
                row_starts.stride(0),
                row_ends.stride(0),
                logits.stride(0),
                logits.stride(1),
                indices.stride(0),
                indices.stride(1),
                K=index_topk,
                K_P2=k_p2,
                COUNT_BLOCK=_ROWS_COUNT_BLOCK,
                WRITE_BLOCK=_WRITE_BLOCK,
                num_warps=_NUM_WARPS,
            )
            return indices
        num_segments = triton.cdiv(num_cols, _SEGMENT)
        kth = torch.empty(num_rows, dtype=torch.float32, device=logits.device)
        counts = torch.empty(2, num_rows, num_segments, dtype=torch.int32, device=logits.device)
        _split_kth_kernel[(num_rows,)](
            logits,
            row_starts,
            indices,
            kth,
            num_cols,
            row_starts.stride(0),
            logits.stride(0),
            logits.stride(1),
            indices.stride(0),
            indices.stride(1),
            K=index_topk,
            K_P2=k_p2,
            num_warps=_NUM_WARPS,
        )
        _split_count_kernel[(num_rows, num_segments)](
            logits,
            row_starts,
            row_ends,
            kth,
            counts,
            num_rows,
            num_cols,
            num_segments,
            row_starts.stride(0),
            row_ends.stride(0),
            logits.stride(0),
            logits.stride(1),
            SEGMENT=_SEGMENT,
            COUNT_BLOCK=_SPLIT_COUNT_BLOCK,
            num_warps=_NUM_WARPS,
        )
        _split_write_kernel[(num_rows, num_segments)](
            logits,
            row_starts,
            row_ends,
            kth,
            counts,
            indices,
            num_rows,
            num_cols,
            num_segments,
            row_starts.stride(0),
            row_ends.stride(0),
            logits.stride(0),
            logits.stride(1),
            indices.stride(0),
            indices.stride(1),
            K=index_topk,
            K_P2=k_p2,
            SEGMENT=_SEGMENT,
            WRITE_BLOCK=_WRITE_BLOCK,
            COUNTS_BLOCK=64,
            num_warps=_NUM_WARPS,
        )
        return indices

    def reference(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """The k largest `(score, column)` pairs of each window, ascending by column,
        `-1`-padded: a top-k over one int64 key per column whose high word orders the
        scores as fp32 does and whose low word is the column. `indices` supplies only the
        width `k`; its contents are ignored."""
        num_rows, num_cols = logits.shape
        index_topk = indices.shape[1]
        columns = torch.arange(num_cols, device=logits.device)[None, :]
        starts, ends = row_starts.long()[:, None], row_ends.long()[:, None]
        in_window = (columns >= starts) & (columns < ends)
        # `+ 0.0` folds -0.0 into +0.0, so the two are one tie class.
        bits = (logits.float() + 0.0).view(torch.int32).long()
        order = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        key = torch.where(in_window, (order << 32) | columns, torch.iinfo(torch.int64).min)
        width = min(index_topk, num_cols)
        top = torch.topk(key, width, dim=1).indices
        taken = torch.gather(in_window, 1, top)
        pad = torch.iinfo(torch.int64).max
        ordered = torch.sort(torch.where(taken, top, pad), dim=1).values
        out = torch.where(ordered == pad, -1, ordered - starts).to(torch.int32)
        if width < index_topk:
            out = torch.nn.functional.pad(out, (0, index_topk - width), value=-1)
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        # A one-element `row_starts` / `row_ends`, or a one-row `indices`, broadcasts over
        # every row of `logits` without an error: a [1] `row_starts` against 4 rows
        # returns every row framed by row 0's start.
        num_rows = logits.shape[0]
        assert row_starts.shape == (num_rows,) and row_ends.shape == (num_rows,), (
            f"row_starts and row_ends need one entry per logits row ({num_rows}); got "
            f"{tuple(row_starts.shape)} and {tuple(row_ends.shape)}"
        )
        assert indices.dim() == 2 and indices.shape[0] == num_rows, (
            f"indices must be [{num_rows}, k]; got {tuple(indices.shape)}"
        )


qsa_canonical_selection = QsaCanonicalSelection()
