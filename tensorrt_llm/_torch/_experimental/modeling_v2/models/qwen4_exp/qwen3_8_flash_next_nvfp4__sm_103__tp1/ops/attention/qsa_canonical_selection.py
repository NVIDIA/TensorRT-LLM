# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA block selection, made canonical: one answer per row, whichever tied columns the
selection engine happened to keep.

Composed from native torch ops: no registered op rewrites a selection, and none of the
top-k engines (`indexer_topk_prefill`, the CuTe DSL kernels, FlashInfer's top-k) takes a
tie-break rule or fixes the order of its output.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's block budget: `indexer_budget` 2048 tokens over
#: `indexer_compress_ratio` 4.
_BLOCK_TOPK = 512
#: Score-row width the target hands over: a 260-page block table at 32 tokens
#: per block, compressed 4:1.
_SCORE_COLUMNS = 260 * 32 // 4
#: The most `[rows, cols]` elements one rewrite pass holds at once; a taller matrix is
#: rewritten in row tiles so the fp32 / int32 temporaries stay a few hundred MiB.
_TILE_ELEMENTS = 1 << 24


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


def _rewrite_tile(logits, row_starts, row_ends, indices) -> None:
    num_rows, num_cols = logits.shape
    index_topk = indices.shape[1]
    starts = row_starts[:, None]
    columns = torch.arange(num_cols, dtype=torch.int32, device=logits.device)[None, :]
    # The window's scores, NaN elsewhere: NaN compares false, so nothing outside the
    # window is ever counted or kept.
    windowed = torch.where(
        (columns >= starts) & (columns < row_ends[:, None]), logits, float("nan")
    )
    # The k-th value: the smallest score the engine kept.
    picked = torch.gather(logits, 1, (indices + starts).clamp_(0, num_cols - 1).long())
    kth = picked.masked_fill_(indices < 0, float("inf")).amin(1, keepdim=True)
    at_least = windowed >= kth
    tied = windowed == kth
    # Keep every column at or above the k-th value except the lowest-index ties past
    # k: a tied column is dropped while no more than `surplus` ties lie at or before
    # it. A window of at most k has no surplus.
    surplus = at_least.sum(1, keepdim=True, dtype=torch.int32) - index_topk
    dropped = tied & (torch.cumsum(tied, 1, dtype=torch.int32) <= surplus)
    kept_seen = torch.cumsum(at_least ^ dropped, 1, dtype=torch.int32)
    # The j-th kept column is the first whose running count reaches j.
    slots = torch.arange(1, index_topk + 1, dtype=torch.int32, device=logits.device)
    slots = slots.repeat(num_rows, 1)
    torch.searchsorted(kept_seen, slots, out_int32=True, out=indices)
    indices.sub_(starts)
    indices.masked_fill_(slots > kept_seen[:, -1:], -1)


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
    # `[rows, 2080]` score matrix whose columns past each `row_end` are allocator
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
            "prefill: 8192 rows from position 0, rewritten in row tiles",
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

    Cost: about 30 kernels -- elementwise passes, two int32 prefix sums over
    `[rows, cols]` and one sorted search over `[rows, k]` -- with no host read and no
    data-dependent shape, so it can be captured into a CUDA graph (not certified: no
    cell replays one). It reads the whole `cols`-wide score row, not the live window: a
    caller that knows a bound on the windows passes `logits[:, :bound]`, which gives the
    same rows. A matrix of more than 2^24 elements is rewritten in row tiles.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        num_rows, num_cols = logits.shape
        tile = max(1, _TILE_ELEMENTS // max(num_cols, 1))
        for lo in range(0, num_rows, tile):
            hi = min(lo + tile, num_rows)
            _rewrite_tile(logits[lo:hi], row_starts[lo:hi], row_ends[lo:hi], indices[lo:hi])
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
