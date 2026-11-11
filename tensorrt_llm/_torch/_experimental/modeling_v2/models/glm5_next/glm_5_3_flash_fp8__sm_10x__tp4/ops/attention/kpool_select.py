# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool top-k of a sparse MLA decode step, expanded to packed latent-cache rows, in one launch.

Per row: the ``select_k`` highest pool scores among the row's ``kv_len // kpool`` complete
pools (an exact radix select on the FP32 bits), each selected pool expanded to its ``kpool``
latent-cache row ids, then the rows of the incomplete trailing pool, packed from column 0 in
pool order and padded with -1. A row with no visible key gets one dummy id 0, length 1 and
``nonempty`` False. The module path runs a radix top-k, an expand kernel and a compaction
kernel; the selected set is the same (up to the choice among pools of equal score at the
threshold) and the packed order is ascending pool index instead of the top-k's order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .compact_sparse_rows import compact_sparse_rows
from .kpool_expand import kpool_expand


@triton.jit
def _order_key(scores):
    """FP32 -> uint32 with the same order (larger float, larger key)."""
    bits = scores.to(tl.uint32, bitcast=True)
    return tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)


@triton.jit
def _in_prefix(key, thresh, shift):
    """Whether ``key`` agrees with ``thresh`` on the bits above ``shift + 8`` (all of them in the
    first pass; two shifts keep every shift amount below 32)."""
    return ((key >> (shift + 7)) >> 1) == ((thresh >> (shift + 7)) >> 1)


@triton.jit
def _radix_step(hist, remaining, thresh, shift):
    """Fix the 8 key bits at ``shift`` of the ``remaining``-th largest key from their histogram."""
    bins = tl.arange(0, 256)
    at_or_above = tl.cumsum(hist, 0, reverse=True)
    b = tl.max(tl.where(at_or_above >= remaining, bins, -1), 0)
    remaining -= tl.sum(tl.where(bins > b, hist, 0), 0)
    return remaining, thresh | (b.to(tl.uint32) << shift)


@triton.jit
def _select_in_registers(srow, sel_row, num_cand, SELECT_K: tl.constexpr, WIDTH: tl.constexpr):
    """Rows with at most ``WIDTH`` candidates: one load, the radix passes and the emit on the
    register tile. Writes the selected pool ids ascending to ``sel_row``; returns their count."""
    idx = tl.arange(0, WIDTH)
    m = idx < num_cand
    key = _order_key(tl.load(srow + idx, mask=m, other=0.0))
    thresh = tl.zeros([], tl.uint32)
    ties = tl.full([], 1 << 30, tl.int32)
    if num_cand > SELECT_K:
        remaining = tl.full([], SELECT_K, tl.int32)
        # Runtime pass loop: an unrolled one quadruples the code, which is fetched cold on
        # every decode layer.
        for p in range(4):
            shift = (24 - 8 * p).to(tl.uint32)
            match = m & _in_prefix(key, thresh, shift)
            hist = tl.histogram(((key >> shift) & 255).to(tl.int32), 256, mask=match)
            remaining, thresh = _radix_step(hist, remaining, thresh, shift)
        ties = remaining
    eq = m & (key == thresh)
    sel = (m & (key > thresh)) | (eq & (tl.cumsum(eq.to(tl.int32), 0) <= ties))
    dest = tl.cumsum(sel.to(tl.int32), 0) - 1
    tl.store(sel_row + dest, idx.to(tl.int32), mask=sel)
    return tl.sum(sel.to(tl.int32), 0)


@triton.jit
def _select_chunked(srow, sel_row, num_cand, SELECT_K: tl.constexpr, CHUNK: tl.constexpr):
    """Longer rows: every radix pass and the emit stream the scores in ``CHUNK``-wide pieces."""
    lane = tl.arange(0, CHUNK)
    thresh = tl.zeros([], tl.uint32)
    remaining = tl.full([], SELECT_K, tl.int32)
    for p in range(4):
        shift = (24 - 8 * p).to(tl.uint32)
        hist = tl.zeros([256], tl.int32)
        for c0 in range(0, num_cand, CHUNK):
            idx = c0 + lane
            m = idx < num_cand
            key = _order_key(tl.load(srow + idx, mask=m, other=0.0))
            m = m & _in_prefix(key, thresh, shift)
            hist += tl.histogram(((key >> shift) & 255).to(tl.int32), 256, mask=m)
        remaining, thresh = _radix_step(hist, remaining, thresh, shift)
    n_sel = tl.zeros([], tl.int32)
    ties_seen = tl.zeros([], tl.int32)
    for c0 in range(0, num_cand, CHUNK):
        idx = c0 + lane
        m = idx < num_cand
        key = _order_key(tl.load(srow + idx, mask=m, other=0.0))
        eq = m & (key == thresh)
        eq_rank = ties_seen + tl.cumsum(eq.to(tl.int32), 0)
        sel = (m & (key > thresh)) | (eq & (eq_rank <= remaining))
        dest = n_sel + tl.cumsum(sel.to(tl.int32), 0) - 1
        tl.store(sel_row + dest, idx.to(tl.int32), mask=sel)
        ties_seen += tl.sum(eq.to(tl.int32), 0)
        n_sel += tl.sum(sel.to(tl.int32), 0)
    return n_sel


@triton.jit
def _kpool_select_kernel(
    SCORES,
    KV_LENS,
    BT,
    SEL,
    PACKED,
    LENGTHS,
    NONEMPTY,
    score_stride,
    bt_stride,
    tpb,
    base_row,
    rows_per_slot,
    SELECT_K: tl.constexpr,
    KPOOL: tl.constexpr,
    KPOOL_P2: tl.constexpr,
    OUT_W: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    kv_len = tl.load(KV_LENS + r).to(tl.int64)
    num_cand = kv_len // KPOOL
    srow = SCORES + r * score_stride
    sel_row = SEL + r * SELECT_K
    # Selected pool ids, ascending, into SEL[r, :n_sel]: on one register tile up to 4096
    # candidates (a 16K-token context at kpool 4), streamed in chunks past that.
    if num_cand <= 4096:
        n_sel = _select_in_registers(srow, sel_row, num_cand, SELECT_K, 4096)
    else:
        n_sel = _select_chunked(srow, sel_row, num_cand, SELECT_K, 4096)
    tl.debug_barrier()

    # Expand: selected pool i -> columns [i * KPOOL, (i + 1) * KPOOL).
    i = tl.arange(0, SELECT_K)
    mm = tl.arange(0, KPOOL_P2)
    ok = (i < n_sel)[:, None] & (mm < KPOOL)[None, :]
    j = tl.load(SEL + r * SELECT_K + i, mask=i < n_sel, other=0).to(tl.int64)
    pos = j[:, None] * KPOOL + mm[None, :]
    page = pos // tpb
    slot = tl.load(BT + r * bt_stride + page, mask=ok, other=0).to(tl.int64)
    rows = base_row + slot * rows_per_slot + (pos - page * tpb)
    tl.store(PACKED + r * OUT_W + i[:, None] * KPOOL + mm[None, :], rows.to(tl.int32), mask=ok)

    # The incomplete trailing pool, always visible to its own query.
    tail_count = kv_len - num_cand * KPOOL
    t = tl.arange(0, KPOOL_P2)
    tvalid = t < tail_count
    tpos = num_cand * KPOOL + t
    tpage = tpos // tpb
    tslot = tl.load(BT + r * bt_stride + tpage, mask=tvalid, other=0).to(tl.int64)
    trows = base_row + tslot * rows_per_slot + (tpos - tpage * tpb)
    length = n_sel * KPOOL + tail_count.to(tl.int32)
    tl.store(PACKED + r * OUT_W + n_sel * KPOOL + t, trows.to(tl.int32), mask=tvalid)

    # -1 after the packed ids; a row with none gets the dummy id 0.
    start = tl.maximum(length, 1)
    for c0 in range(0, OUT_W, PAD_BLOCK):
        col = c0 + tl.arange(0, PAD_BLOCK)
        tl.store(
            PACKED + r * OUT_W + col,
            tl.full([PAD_BLOCK], -1, tl.int32),
            mask=(col >= start) & (col < OUT_W),
        )
    if length == 0:
        tl.store(PACKED + r * OUT_W, 0)
    tl.store(LENGTHS + r, tl.maximum(length, 1))
    tl.store(NONEMPTY + r, length > 0)


class KpoolSelect(OpWrapper):
    """``scores`` FP32 [R, >= max candidates] (``kpool_score``), ``kv_lens`` [R], ``block_tables``
    [R, blocks] -> (``packed`` int32 [R, W], ``lengths`` int32 [R], ``nonempty`` bool [R]) with
    ``W = cdiv(out_width, 64) * 64``, the layout ``compact_sparse_rows`` returns."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP-3 verify of one 13K-token request", spec=dict(rows=4, kv=13312, pools=32800)),
        Cell(
            why="MTP-3 verify of eight 13K-token requests",
            spec=dict(rows=32, kv=13312, pools=32800),
        ),
        Cell(
            why="short contexts: fewer complete pools than select_k",
            spec=dict(rows=4, kv=1500, pools=32800),
        ),
        Cell(
            why="20K-token context: chunked radix passes", spec=dict(rows=4, kv=20000, pools=32800)
        ),
        Cell(why="128K-token context", spec=dict(rows=2, kv=131072, pools=32800)),
    )

    def __call__(
        self,
        scores,
        kv_lens,
        block_tables,
        tokens_per_block,
        *,
        select_k,
        kpool,
        base_row,
        rows_per_slot,
        out_width,
    ):
        rows = scores.shape[0]
        width = triton.cdiv(out_width, 64) * 64
        packed = torch.empty((rows, width), dtype=torch.int32, device=scores.device)
        lengths = torch.empty(rows, dtype=torch.int32, device=scores.device)
        nonempty = torch.empty(rows, dtype=torch.bool, device=scores.device)
        selected = torch.empty((rows, select_k), dtype=torch.int32, device=scores.device)
        _kpool_select_kernel[(rows,)](
            scores,
            kv_lens,
            block_tables,
            selected,
            packed,
            lengths,
            nonempty,
            scores.stride(0),
            block_tables.stride(0),
            tokens_per_block,
            base_row,
            rows_per_slot,
            SELECT_K=select_k,
            KPOOL=kpool,
            KPOOL_P2=triton.next_power_of_2(kpool),
            OUT_W=width,
            PAD_BLOCK=512,
            num_warps=16,
        )
        return packed, lengths, nonempty

    def reference(
        self,
        scores,
        kv_lens,
        block_tables,
        tokens_per_block,
        *,
        select_k,
        kpool,
        base_row,
        rows_per_slot,
        out_width,
    ):
        num_cand = (kv_lens.long() // kpool).clamp(max=scores.shape[1])
        col = torch.arange(scores.shape[1], device=scores.device)
        masked = torch.where(col[None, :] < num_cand[:, None], scores, float("-inf"))
        top = torch.topk(masked, select_k, dim=1).indices
        top = torch.where(torch.gather(masked, 1, top) > float("-inf"), top, -1)
        top = torch.sort(torch.where(top >= 0, top, scores.shape[1]), dim=1).values
        top = torch.where(top < scores.shape[1], top, -1).int()
        expanded = kpool_expand(
            top,
            kv_lens,
            block_tables,
            tokens_per_block,
            base_row=base_row,
            rows_per_slot=rows_per_slot,
            kpool=kpool,
            out_width=out_width,
        )
        return compact_sparse_rows(expanded)

    def is_valid(
        self, scores, kv_lens, block_tables, tokens_per_block, *, select_k, kpool, **kwargs
    ):
        assert scores.dtype == torch.float32 and scores.stride(1) == 1
        assert kv_lens.shape == (scores.shape[0],) and block_tables.shape[0] == scores.shape[0]
        assert select_k & (select_k - 1) == 0, f"select_k {select_k} must be a power of two"
        assert kwargs["out_width"] >= select_k * kpool + kpool - 1


kpool_select = KpoolSelect()
