# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate block pre-filter (level one of DeepSeek-V4.1's two-level index top-k), one Triton launch.

Per row of index scores: max-pool the row's visible positions in blocks of ``block_size``, pin
the block that holds the newest visible position, keep the ``topk_blocks`` best blocks, and
return a bool mask over the score columns that is True on every visible position of a kept
block.

The kernel is defined here (``_candidate_block_select_kernel``). No engine kernel implements
this step, so this entry is the implementation, not a wrapper around one.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Launch configuration: (candidate blocks scored per loop iteration, warps). One program owns
# one row, so few rows (decode) take wide iterations and many rows (prefill) narrow ones. The
# result does not depend on the choice.
_FEW_ROWS = 256
_FEW_ROWS_CONFIG = (1024, 8)
_MANY_ROWS_CONFIG = (256, 4)
_FILL_COLS = 1024


@triton.jit
def _block_keys(
    row_ptr,
    col_stride,
    start,
    length,
    num_cols,
    bidx,
    bvalid,
    BLOCK_SIZE: tl.constexpr,
    UNIT_COL_STRIDE: tl.constexpr,
):
    """(amax, order-preserving uint32 key) of the candidate blocks ``bidx`` of one row.

    A block's positions past ``length`` count as -inf. The key orders like the float,
    with -0.0 folded onto +0.0 so the two tie.
    """
    pos = bidx[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    col = start + pos
    live = bvalid[:, None] & (pos < length) & (col >= 0) & (col < num_cols)
    if UNIT_COL_STRIDE:
        ptrs = row_ptr + col
    else:
        ptrs = row_ptr + col.to(tl.int64) * col_stride
    x = tl.load(ptrs, mask=live, other=float("-inf")).to(tl.float32)
    amax = tl.max(x, axis=1)
    bits = amax.to(tl.int32, bitcast=True)
    bits = tl.where(bits == -2147483648, 0, bits)
    u = bits.to(tl.uint32, bitcast=True)
    key = tl.where(bits < 0, ~u, u | 0x80000000)
    return amax, key


@triton.jit
def _candidate_block_select_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    row_starts_ptr,
    row_ends_ptr,
    mask_ptr,
    mask_row_stride,
    num_cols,
    topk_blocks,
    NEXT_N: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNIT_COL_STRIDE: tl.constexpr,
    CHUNK_BLOCKS: tl.constexpr,
    FILL_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    end = tl.load(row_ends_ptr + row // NEXT_N).to(tl.int32) - (NEXT_N - 1 - row % NEXT_N)
    if HAS_ROW_STARTS:
        start = tl.load(row_starts_ptr + row).to(tl.int32)
    else:
        start = row * 0
    length = tl.maximum(end - start, 0)
    end = start + length
    num_blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    newest = num_blocks - 1  # the pinned block; -1 (none) for an empty row
    binding = num_blocks > topk_blocks
    row_ptr = logits_ptr + row.to(tl.int64) * logits_row_stride
    out_ptr = mask_ptr + row.to(tl.int64) * mask_row_stride

    # Level-one selection when the row has more blocks than topk_blocks: the pinned block
    # takes one slot, the other topk_blocks - 1 go to the best of blocks [0, newest).
    # thr = the (topk_blocks - 1)-th largest key among them (radix select, 4 bits a pass);
    # above = how many keys are strictly greater; n_tie = how many keys equal to thr to keep.
    need = topk_blocks - 1
    thr = tl.zeros((), dtype=tl.uint32)
    n_tie = row * 0
    if binding:
        digits = tl.arange(0, 16)
        above = row * 0
        prefix = tl.zeros((), dtype=tl.uint32)
        for p in tl.static_range(8):
            hist = tl.zeros((16,), dtype=tl.int32)
            for b0 in range(0, newest, CHUNK_BLOCKS):
                bidx = b0 + tl.arange(0, CHUNK_BLOCKS)
                bvalid = bidx < newest
                _, key = _block_keys(
                    row_ptr,
                    logits_col_stride,
                    start,
                    length,
                    num_cols,
                    bidx,
                    bvalid,
                    BLOCK_SIZE,
                    UNIT_COL_STRIDE,
                )
                if p == 0:
                    sel = bvalid
                else:  # the digits above this one must equal the prefix found so far
                    sel = bvalid & ((key >> (32 - 4 * p)) == (prefix >> (32 - 4 * p)))
                d = ((key >> (28 - 4 * p)) & 15).to(tl.int32)
                onehot = (d[:, None] == digits[None, :]) & sel[:, None]
                hist += tl.sum(onehot.to(tl.int32), axis=0)
            ge = tl.sum(tl.where(digits[None, :] >= digits[:, None], hist[None, :], 0), axis=1)
            ok = (above + ge) >= need
            dsel = tl.max(tl.where(ok, digits, -1), axis=0)
            above += tl.sum(tl.where(digits > dsel, hist, 0), axis=0)
            prefix = prefix | (dsel.to(tl.uint32) << (28 - 4 * p))
        thr = prefix
        n_tie = need - above

    # Expand the kept blocks over the row's visible columns. Ties at thr go to the lowest
    # block indices; a block whose visible scores are all -inf is dropped unless pinned.
    tie_seen = row * 0
    for b0 in range(0, num_blocks, CHUNK_BLOCKS):
        bidx = b0 + tl.arange(0, CHUNK_BLOCKS)
        bvalid = bidx < num_blocks
        amax, key = _block_keys(
            row_ptr,
            logits_col_stride,
            start,
            length,
            num_cols,
            bidx,
            bvalid,
            BLOCK_SIZE,
            UNIT_COL_STRIDE,
        )
        older = bvalid & (bidx < newest)
        is_tie = (older & (key == thr) & binding).to(tl.int32)
        tie_rank = tie_seen + tl.cumsum(is_tie, axis=0) - is_tie
        chosen = (key > thr) | ((is_tie != 0) & (tie_rank < n_tie)) | (num_blocks <= topk_blocks)
        keep = (bidx == newest) | (older & (amax > float("-inf")) & chosen)
        tie_seen += tl.sum(is_tie, axis=0)
        pos = bidx[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        col = start + pos
        live = bvalid[:, None] & (pos < length) & (col >= 0) & (col < num_cols)
        val = tl.broadcast_to(keep[:, None], (CHUNK_BLOCKS, BLOCK_SIZE)).to(tl.uint8)
        tl.store(out_ptr + col, val, mask=live)

    # Every column outside [start, end) is False.
    zeros = tl.zeros((FILL_COLS,), dtype=tl.uint8)
    for c0 in range(0, start, FILL_COLS):
        col = c0 + tl.arange(0, FILL_COLS)
        tl.store(out_ptr + col, zeros, mask=(col < start) & (col < num_cols))
    for c0 in range(end, num_cols, FILL_COLS):
        col = c0 + tl.arange(0, FILL_COLS)
        tl.store(out_ptr + col, zeros, mask=(col >= end) & (col >= 0) & (col < num_cols))


_TOPK_BLOCKS = 2048  # the checkpoint's candidate_topk_blocks
_BLOCK = 8  # the checkpoint's candidate_block_size
_ENV_KV = 16384
_ENVELOPE_LM = 1048576 + 14  # the longest request with DSpark: 1,048,590 keys

#: Decode rows' visible counts: tiny, both sides of the 16384 binding edge, and far past it.
_DECODE_LENS = (1, 5, 8, 9, 16383, 16384, 16385, 16391, 16392, 16393, 16400, 20000, 32768)
_DECODE_LENS += (65536, 131072, 262144)

#: Packed prefill calls made untiled (max_num_tokens 8192, max_batch_size 32, max_seq_len 16384):
#: (what, query rows per request, keys per request). Request b's last q_b positions are the rows.
_UNTILED_STEPS = (
    ("maximal: 32 requests x 16384 keys, 256 rows each", (256,) * 32, (_ENV_KV,) * 32),
    ("widest step, one request's rows only", (8192 - 31,) + (1,) * 31, (_ENV_KV,) * 32),
    ("W 16384 (the target skips it: exact)", (4096, 4096), (8192, 8192)),
    ("W 16384, one request, 8192 rows", (8192,), (16384,)),
    ("W 16385, 2 rows", (1, 1), (8193, 8192)),
    ("W 16385, 256 rows", (128, 128), (8193, 8192)),
    ("W 16385, 257 rows", (129, 128), (8193, 8192)),
    ("W 16385, 8192 rows", (4096, 4096), (8193, 8192)),
    ("W 16385, 32 requests", (256,) * 32, (512,) * 31 + (513,)),
    ("dumped (7674, 24058)", (7674,), (24058,)),
    ("dumped (6523, 24058)", (6523,), (24058,)),
    ("dumped (8192, 17535)", (8192,), (17535,)),
)

#: Packed prefill calls in the target's row tiles (max_num_tokens 16384, max_batch_size 256,
#: max_seq_len up to 1,048,576): one call per tile of at most 2^28 fp32 logits elements. A request
#: with 0 rows only holds keys in the gather (the decoder-replay tail's neighbours).
_TILED_STEPS = (
    ("L 32768 DSpark: one request at Lm 32782, 16384 rows", (16384,), (32782,)),
    ("L 32768 no DSpark: one request at 32768, 16384 rows", (16384,), (32768,)),
    ("L 65536 DSpark: one request at Lm 65550, 16384 rows", (16384,), (65550,)),
    ("L 131072 DSpark: one request at Lm 131086, 16384 rows", (16384,), (131086,)),
    ("L 262144 DSpark: one request at Lm 262158, 16384 rows", (16384,), (262158,)),
    ("L 1048576 DSpark: one request at Lm 1048590, 1024 rows", (1024,), (_ENVELOPE_LM,)),
    ("L 1048576 no DSpark: one request at 1048576, 1024 rows", (1024,), (1048576,)),
    ("tail (128, 1048219): keys of another request first", (0, 128), (448219, 600000)),
    ("tail (128, 523951)", (0, 128), (23951, 500000)),
    ("tail (128, 261860)", (128, 0), (200000, 61860)),
    ("tail (128, 130773)", (0, 0, 128), (30000, 773, 100000)),
    (
        "tail 3 requests x 128 rows at S_1 1048219 (tiles 256 + 128)",
        (128, 128, 0, 128),
        (300000, 200000, 48219, 500000),
    ),
    ("256 requests x 1 row, W 1048576", (1,) * 256, (4096,) * 256),
    (
        "two requests at Lm, 256 rows, S_1 2097180 (tiles of 64)",
        (128, 128),
        (_ENVELOPE_LM, _ENVELOPE_LM),
    ),
    ("four requests at Lm, 128 rows, S_1 4194360 (tiles of 63)", (32,) * 4, (_ENVELOPE_LM,) * 4),
    (
        "sixteen requests at Lm, 64 rows, S_1 16777440 (tiles of 15)",
        (4,) * 16,
        (_ENVELOPE_LM,) * 16,
    ),
    ("W 16384 at 16384 rows (the target skips it)", (16384,), (16384,)),
    ("W 16385 at 16384 rows", (8192, 8192), (8193, 8192)),
    ("W 16385, 256 requests, 16384 rows", (64,) * 256, (65,) + (64,) * 255),
    ("dumped (16292, 16625)", (10441, 5851), (10441, 6184)),
    ("dumped (16378, 16695)", (10451, 5927), (10451, 6244)),
)


class CandidateBlockSelect(OpWrapper):
    """Return a fresh bool ``[N, W]`` candidate mask for index scores ``logits`` ``[N, W]``.

    Row ``i`` sees columns ``[s, e)``::

        s = row_starts[i]                                   (0 when row_starts is None)
        e = row_ends[i // next_n] - (next_n - 1 - i % next_n)
        L = max(e - s, 0)                                   visible columns s .. s + L - 1

    Its visible positions are cut into blocks of ``block_size`` starting at ``s``; each block scores
    the fp32 max of its visible scores; the block holding the newest position (``e - 1``) is always
    kept; the other ``topk_blocks - 1`` places go to the highest-scoring remaining blocks (all of
    them when there are at most ``topk_blocks``), ties to the lower block index; a block whose
    visible scores are all -inf is dropped unless pinned. ``mask[i, c]`` is True exactly for the
    visible columns of kept blocks; every other element is written False.

    Fusion boundary: block max-pooling, the pin, the block top-k, the -inf drop, the expansion to
    columns and clearing the columns outside each row's range are inside the call. Producing the
    scores, and applying the keep mask to a consumer layer's scores (``torch.where(mask, s, -inf)``
    before its own top-512), are the caller's. Returns the mask (fresh, contiguous).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"prefill one request seq={seq} starts={'explicit' if explicit else 'None'}: "
                "every query row of a fresh prompt, visible 1..seq",
                spec=dict(kind="prefill_one", seq=seq, explicit_starts=explicit),
            )
            for seq in (4096, 20480)
            for explicit in (True, False)
        )
        + (
            Cell(
                why="prefill chunk 16000..20095 row stride W+13: a later chunk of a long context, "
                "past the binding edge",
                spec=dict(kind="prefill_chunk", first=16000, q=4096, pad=13),
            ),
            Cell(
                why="prefill 3 requests sharing the columns: ragged row_starts, 5 spare columns",
                spec=dict(
                    kind="prefill_ragged",
                    keys=(18000, 700, 21000),
                    queries=((16300, 1700), (0, 700), (19000, 2000)),
                    spare=5,
                ),
            ),
            Cell(
                why="decode visible 1..262144, NaN/+inf unread: the binding edge and far past it",
                spec=dict(kind="decode", lens=_DECODE_LENS),
            ),
            Cell(
                why="decode next_n=6 DSpark verify rows, 3 empty: row j sees one more than j - 1",
                spec=dict(kind="decode_next_n", visible=(16386, 16390, 20000, 50000, 3), next_n=6),
            ),
            Cell(
                why="decode bf16 scores: frequent ties, cut by the lowest block index",
                spec=dict(kind="bf16", lens=(16385, 20000, 40000)),
            ),
            Cell(
                why="decode heavy ties {-1,-0,0,1,2}: the cut inside a +-0 tie",
                spec=dict(kind="ties", lens=(16392, 20000, 40001, 30000)),
            ),
            Cell(
                why="decode -inf blocks: dropped though < 2048 remain, the pin held",
                spec=dict(kind="minus_inf", lens=(20000, 20000, 30000)),
            ),
            Cell(
                why="empty rows L=0 around a full row: all False",
                spec=dict(kind="empty_rows", lens=(0, 64, 0), width=64),
            ),
            Cell(
                why="no rows N=0: an empty [0, W] mask, no launch",
                spec=dict(kind="no_rows", width=64),
            ),
            Cell(
                why="column-strided logits (stride 8): visible 20000 and 16000",
                spec=dict(kind="col_strided", lens=(20000,) * 4 + (16000,) * 4),
            ),
        )
        + tuple(
            Cell(
                why=f"untiled {what}",
                spec=dict(kind="packed", q_lens=q, kv_lens=kv, tiled=False),
            )
            for what, q, kv in _UNTILED_STEPS
        )
        + tuple(
            Cell(
                why=f"tiled {what}",
                spec=dict(kind="packed", q_lens=q, kv_lens=kv, tiled=True),
            )
            for what, q, kv in _TILED_STEPS
        )
    )

    note = """
    Every cell runs at the checkpoint's `topk_blocks = 2048`, `block_size = 8`: the pre-filter
    binds only on rows with more than 16384 visible positions, and at or below that the mask is
    the row's range minus its non-pinned all--inf blocks. The packed cells fill every element
    outside a row's range with NaN and set 1 in 32 whole blocks to -inf, so each row's addressing
    shows in its mask; each fresh mask lands on a block pre-filled with 1s. Call sites: prefill
    with `row_starts = ks`, `row_ends = ke` on DeepGEMM's `[rows, S_1]` logits (row stride
    `align(S_1 + 256, 256)`), whole or per row tile; decode with `row_starts = None` on the paged
    `[G * next_n, max_ctx]` logits (row stride `ceil256(max_ctx)`) and a row-offset view of the
    step's visible-length buffer, `next_n` 1 (each verify row carries its own length) or 6.

    Relation to the checkpoint's `select_candidate_blocks`: equal on every visible position on
    distinct scores; on tied block scores the kept block-score multiset is equal (its
    `torch.topk` keeps an unspecified member of the tie). It also marks the invisible tail of the
    pinned block in prefill, which this op leaves False; consumers hold -inf there either way.

    Metadata consumed: none; the row ranges are the caller's.

    Preconditions: `0 <= s <= e <= W` per row (reads and writes are masked to `[0, W)`, so a
    violation touches nothing outside the two tensors, but is not certified); `block_size` a power
    of two (12 fails to compile); CUDA tensors (a CPU tensor raises). Columns outside a row's range
    are never read (NaN and +inf there are certified). A row's mask depends only on its own scores
    and range: the same row alone, inside a larger call, or with its keys moved within the gather
    gets the same mask.

    The kernel JIT-compiles on the first call of each specialization (input dtype, `next_n`,
    `row_starts` given, `block_size`, unit column stride, launch configuration, Triton's
    divisibility and pointer-alignment classes); capture in a CUDA graph after an eager call of
    the same specialization is certified, a first-ever call inside capture is not. The launch
    configuration follows the row count (at most 256 rows: 1024 blocks an iteration and 8 warps,
    else 256 and 4); the mask does not depend on it.

    Measured on sm_103: one program per row runs an 8-pass radix select over the block scores, so
    decode rows above 16384 visible are slower than the checkpoint's torch composition (0.89x at
    20480, 0.32x at 131072); prefill is faster.

    Not certified: `topk_blocks` other than 2048 or `block_size` other than 8; `next_n` other than
    1 and 6, or `next_n > 1` with `row_starts`; decode calls of more than 1536 rows; CUDA-graph
    capture at a batch outside 1..32, 40..128 step 8, 192, 256, or at a width other than 16389,
    16390 and L, L + 14, L + 19 for L in 16384, 32768, 65536, 131072, 262144, 1048576;
    `W > 33,329,476`; more than 20480 rows in one call; more than 2^32 logits elements
    or mask bytes in one call; NaN or +inf inside a visible range; float16 logits and int64 lengths
    (observed working only).
    """

    def raw_call(
        self,
        logits: torch.Tensor,
        row_starts: Optional[torch.Tensor],
        row_ends: torch.Tensor,
        topk_blocks: int,
        block_size: int,
        next_n: int,
    ) -> torch.Tensor:
        num_rows, num_cols = logits.shape
        mask = torch.empty((num_rows, num_cols), dtype=torch.bool, device=logits.device)
        if num_rows == 0:
            return mask
        chunk_blocks, num_warps = _FEW_ROWS_CONFIG if num_rows <= _FEW_ROWS else _MANY_ROWS_CONFIG
        _candidate_block_select_kernel[(num_rows,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            row_ends if row_starts is None else row_starts,
            row_ends,
            mask.view(torch.uint8),
            mask.stride(0),
            num_cols,
            topk_blocks,
            NEXT_N=next_n,
            HAS_ROW_STARTS=row_starts is not None,
            BLOCK_SIZE=block_size,
            UNIT_COL_STRIDE=logits.stride(1) == 1,
            CHUNK_BLOCKS=chunk_blocks,
            FILL_COLS=_FILL_COLS,
            num_warps=num_warps,
        )
        return mask

    def is_valid(
        self,
        logits: torch.Tensor,
        row_starts: Optional[torch.Tensor],
        row_ends: torch.Tensor,
        topk_blocks: int,
        block_size: int,
        next_n: int,
    ) -> None:
        num_rows, num_cols = logits.shape
        # The kernel reads row_ends[i // next_n] and row_starts[i] at unit stride with no bound: a
        # shorter or strided tensor is read wrongly with no error. Measured with the kernel called
        # directly on 8 rows x 20000: a 4-element row_ends got rows 4-7 wrong; an 8-element
        # row_ends with next_n = 6 gave rows 0-4, 6 and 7 a shorter range; a 4-element row_starts
        # got rows 4 and 6 wrong; a stride-2 view of either got rows 1, 3, 5 and 7 wrong.
        assert row_ends.is_contiguous() and row_ends.numel() * next_n == num_rows, (
            "row_ends must be contiguous with row_ends.numel() * next_n == logits.size(0)"
        )
        assert row_starts is None or (
            row_starts.is_contiguous() and row_starts.numel() == num_rows
        ), "row_starts must be None or contiguous with row_starts.numel() == logits.size(0)"

    def reference(
        self,
        logits: torch.Tensor,
        row_starts: Optional[torch.Tensor],
        row_ends: torch.Tensor,
        topk_blocks: int,
        block_size: int,
        next_n: int,
    ) -> torch.Tensor:
        """The checkpoint's `select_candidate_blocks` with this op's tie rule, from torch ops.

        Per row: gather the visible scores `[s, s + L)` (everything else -inf, so NaN outside the
        range is never read), -inf-pad to whole blocks, take the block amax in fp32 (-0.0 folded
        onto +0.0), pin the newest block to +inf, keep the first `topk_blocks` of a stable
        descending sort (ties to the lower block index) whose value is above -inf, expand to
        positions and scatter back to the row's columns. Rows are processed in chunks of at most
        2^27 gathered elements, so the reference runs at the widths the cells reach. On distinct
        scores this is the checkpoint's `torch.topk` selection exactly.
        """
        num_rows, num_cols = logits.shape
        dev = logits.device
        mask = torch.zeros((num_rows, num_cols), dtype=torch.bool, device=dev)
        if num_rows == 0:
            return mask
        i = torch.arange(num_rows, device=dev)
        ends = row_ends.long().repeat_interleave(next_n) - (next_n - 1 - i % next_n)
        starts = torch.zeros_like(ends) if row_starts is None else row_starts.long()
        lens = (ends - starts).clamp(min=0)
        lens_host = lens.tolist()
        r0 = 0
        while r0 < num_rows:
            r1, widest = r0, 1
            while r1 < num_rows:
                w = max(widest, lens_host[r1])
                padded = -(-w // block_size) * block_size
                if r1 > r0 and (
                    (r1 + 1 - r0) * padded > 1 << 27 or (r1 + 1 - r0) * (num_cols + 1) > 1 << 30
                ):
                    break
                widest, r1 = w, r1 + 1
            width = -(-widest // block_size) * block_size
            pos = torch.arange(width, device=dev)
            cols = starts[r0:r1, None] + pos
            live = (pos < lens[r0:r1, None]) & (cols >= 0) & (cols < num_cols)
            vals = logits[r0:r1].gather(1, cols.clamp(0, num_cols - 1)).float()
            vals = vals.masked_fill(~live, float("-inf"))
            blk = vals.view(r1 - r0, -1, block_size).amax(dim=-1)
            blk = torch.where(blk == 0, torch.zeros_like(blk), blk)
            newest = (lens[r0:r1, None] + block_size - 1) // block_size - 1
            blk = blk.masked_fill(torch.arange(blk.size(1), device=dev) == newest, float("inf"))
            k = min(topk_blocks, blk.size(1))
            order = torch.sort(blk, dim=1, descending=True, stable=True)
            keep_blk = torch.zeros_like(blk, dtype=torch.bool).scatter_(
                1, order.indices[:, :k], order.values[:, :k] > float("-inf")
            )
            keep = keep_blk.repeat_interleave(block_size, dim=1) & live
            part = torch.zeros((r1 - r0, num_cols + 1), dtype=torch.bool, device=dev)
            part.scatter_(1, torch.where(live, cols, num_cols), keep)
            mask[r0:r1] = part[:, :num_cols]
            r0 = r1
        return mask

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bitwise on the whole `[N, W]` mask, every byte 0 or 1.

        The selection is discrete -- block maxima of the same fp32 values, one order, one tie rule
        -- so there is nothing to round differently: any differing element is a wrong block, range
        or pin. Controls measured on sm_103 (decode rows above 16384 visible / every row of a
        20480-token prefill): no pin of the newest block changes 8 / 1443 rows, a block size of 16
        13 / 4096, a top-16384 over positions instead of blocks 13 / 4096; each is identical to
        this op on every row with at most 16384 visible positions.
        """
        assert out.dtype == ref.dtype == torch.bool, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        if out.numel():
            assert int(out.view(torch.uint8).max()) <= 1, "a mask byte is neither 0 nor 1"
        if not torch.equal(out, ref):
            bad = (out != ref).any(dim=1).nonzero().flatten()
            raise AssertionError(
                f"{bad.numel()} of {out.size(0)} rows differ from the reference, first {int(bad[0])}"
            )


candidate_block_select = CandidateBlockSelect()
