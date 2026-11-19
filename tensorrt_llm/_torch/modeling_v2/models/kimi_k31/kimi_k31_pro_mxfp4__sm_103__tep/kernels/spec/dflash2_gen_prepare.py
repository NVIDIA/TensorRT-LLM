# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernel of the DFlash2 worker's per-step drafter input bookkeeping (generation requests).

``K31DFlashWorker.prepare_1st_drafter_inputs`` and the head of the MLA drafter's ``dflash_forward`` build
the drafter's inputs with about forty small torch launches: the anchor (bonus) token, the block query
positions, the context positions and paged store rows of the accepted rows, the context length
advance (clamped, the dummy slot emptied), the advertised context lengths, the block's paged store
addresses, the page-table rows and the noise block embedding. :class:`GenPrepareKernel` does all of it in
one launch over a private paged context arena (optionally a ring of ``ring_rows`` rows per slot), with
the same integer arithmetic, so every output is bitwise the torch path's.

CTA 0 does the bookkeeping in four barrier-separated phases (reads of the old context lengths, the
advance, the clamp of every slot, the dummy slot emptied and the reads of the new lengths); CTA ``1 + r``
writes request ``r``'s noise block: row 0 the embedding of its anchor token, rows ``1 ..`` the mask row.
"""

from __future__ import annotations

import functools

import cutlass
import torch
from cutlass import Int32, Int64, cute

THREADS = 256
#: Elements per 16-byte BF16 vector.
VEC = 8

_TORCH_TO_CUTLASS = {
    torch.bfloat16: cutlass.BFloat16,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.uint8: cutlass.Uint8,
}


def _row_major(dtype, shape, align=None):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align
    )


class GenPrepareKernel:
    """One launch of the generation requests' drafter inputs; see the module docstring.

    Args:
        width: target rows per request (``K + 1``), the accepted-row store width.
        block: the drafter block width (query rows per request).
        ring: the arena is a ring of ``ring_rows`` rows per slot (positions taken modulo it).
    """

    def __init__(self, width: int, block: int, ring: bool):
        self.width = width
        self.block = block
        self.ring = ring

    @cute.jit
    def __call__(
        self,
        mNa: cute.Tensor,  # [n] int32 accepted counts of the generation requests
        mAcc: cute.Tensor,  # [n, width] int32 accepted tokens
        mSlots: cute.Tensor,  # [n] int64 arena slots
        mCtxLen: cute.Tensor,  # [slots] int64, advanced in place
        mTable: cute.Tensor,  # [slots, pages] int32 page table
        mEmbed: cute.Tensor,  # [vocab, hidden / 8, 8] BF16 embedding rows
        mMaskRow: cute.Tensor,  # [hidden / 8, 8] BF16 mask row
        mNoise: cute.Tensor,  # [n * block, hidden / 8, 8] BF16 noise block embedding
        mBonus: cute.Tensor,  # [n] int64
        mQPos: cute.Tensor,  # [n, block] int64 block query positions
        mCtxPos: cute.Tensor,  # [n * width] int64 context positions of the accepted rows
        mStoreRow: cute.Tensor,  # [n * width] int64 pool rows (page * page_size + offset)
        mStoreMask: cute.Tensor,  # [n * width] uint8
        mNumCtx: cute.Tensor,  # [n] int64 advertised context lengths
        mPageRows: cute.Tensor,  # [n, pages] int64 the requests' page-table rows
        mBlkPage: cute.Tensor,  # [n * block] int64 block store pages
        mBlkOff: cute.Tensor,  # [n * block] int64 block store offsets
        mSeqLens: cute.Tensor,  # [n] seq_lens of the generation requests (set to width)
        max_ctx: Int64,
        dummy: Int64,
        ring_rows: Int64,
        page_size: Int64,
        stream,
    ):
        n = mNa.shape[0]
        self.kernel(
            mNa,
            mAcc,
            mSlots,
            mCtxLen,
            mTable,
            mEmbed,
            mMaskRow,
            mNoise,
            mBonus,
            mQPos,
            mCtxPos,
            mStoreRow,
            mStoreMask,
            mNumCtx,
            mPageRows,
            mBlkPage,
            mBlkOff,
            mSeqLens,
            max_ctx,
            dummy,
            ring_rows,
            page_size,
        ).launch(grid=[n + 1, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.jit
    def ring_pos(self, pos: Int64, ring_rows: Int64) -> Int64:
        """Arena row of ``pos`` (``torch.remainder`` of a non-negative position)."""
        out = pos
        if cutlass.const_expr(self.ring):
            out = pos % ring_rows
        return out

    @cute.kernel
    def kernel(
        self,
        mNa: cute.Tensor,
        mAcc: cute.Tensor,
        mSlots: cute.Tensor,
        mCtxLen: cute.Tensor,
        mTable: cute.Tensor,
        mEmbed: cute.Tensor,
        mMaskRow: cute.Tensor,
        mNoise: cute.Tensor,
        mBonus: cute.Tensor,
        mQPos: cute.Tensor,
        mCtxPos: cute.Tensor,
        mStoreRow: cute.Tensor,
        mStoreMask: cute.Tensor,
        mNumCtx: cute.Tensor,
        mPageRows: cute.Tensor,
        mBlkPage: cute.Tensor,
        mBlkOff: cute.Tensor,
        mSeqLens: cute.Tensor,
        max_ctx: Int64,
        dummy: Int64,
        ring_rows: Int64,
        page_size: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        n = mNa.shape[0]
        W: cutlass.Constexpr = self.width
        B: cutlass.Constexpr = self.block
        if bid > 0:
            # The noise block of request bid - 1: [anchor embedding | mask row x (block - 1)].
            r = bid - 1
            na = Int32(mNa[r])
            idx = na - 1
            if idx < 0:
                idx = Int32(0)
            tok = Int64(mAcc[r, idx])
            # The embedding's lookup of an id outside the vocabulary is a zero row.
            ok = Int32(1)
            if tok < Int64(0):
                ok = Int32(0)
            if tok >= Int64(mEmbed.shape[0]):
                ok = Int32(0)
            safe = tok
            if ok == 0:
                safe = Int64(0)
            vecs = mNoise.shape[1]
            for v in cutlass.range(tid, vecs, THREADS):
                row = mEmbed[safe, v, None].load()
                if ok == 0:
                    row = cute.zeros_like(row)
                mNoise[r * B, v, None].store(row)
                mrow = mMaskRow[v, None].load()
                for j in cutlass.range_constexpr(1, B):
                    mNoise[r * B + j, v, None].store(mrow)
        else:
            # Phase 1: everything read from the old context lengths.
            for item in cutlass.range(tid, n * W, THREADS):
                r = item // W
                j = item - r * W
                slot = Int64(mSlots[r])
                old = Int64(mCtxLen[slot])
                pos = old + Int64(j)
                mCtxPos[item] = pos
                col = pos
                if col > max_ctx - Int64(1):
                    col = max_ctx - Int64(1)
                col = self.ring_pos(col, ring_rows)
                mStoreRow[item] = (
                    Int64(mTable[slot, col // page_size]) * page_size + col % page_size
                )
                keep = cutlass.Uint8(0)
                if Int64(j) < Int64(mNa[r]):
                    keep = cutlass.Uint8(1)
                mStoreMask[item] = keep
            for item in cutlass.range(tid, n * B, THREADS):
                r = item // B
                j = item - r * B
                slot = Int64(mSlots[r])
                q = Int64(mCtxLen[slot]) + Int64(mNa[r])
                if q > max_ctx:
                    q = max_ctx
                mQPos[r, j] = q + Int64(j)
            for r in cutlass.range(tid, n, THREADS):
                na = Int32(mNa[r])
                idx = na - 1
                if idx < 0:
                    idx = Int32(0)
                mBonus[r] = Int64(mAcc[r, idx])
                mSeqLens[r] = mSeqLens.element_type(W)
            cute.arch.sync_threads()
            # Phase 2: the advance (only the dummy slot repeats, and it is emptied below).
            for r in cutlass.range(tid, n, THREADS):
                slot = Int64(mSlots[r])
                mCtxLen[slot] = Int64(mCtxLen[slot]) + Int64(mNa[r])
            cute.arch.sync_threads()
            # Phase 3: every slot clamped to max_ctx.
            for s in cutlass.range(tid, mCtxLen.shape[0], THREADS):
                v = Int64(mCtxLen[s])
                if v > max_ctx:
                    mCtxLen[s] = max_ctx
            cute.arch.sync_threads()
            if tid == 0:
                mCtxLen[dummy] = Int64(0)
            cute.arch.sync_threads()
            # Phase 4: the new lengths, the page-table rows and the block's store addresses.
            for r in cutlass.range(tid, n, THREADS):
                mNumCtx[r] = Int64(mCtxLen[Int64(mSlots[r])])
            pages = mTable.shape[1]
            for item in cutlass.range(tid, n * pages, THREADS):
                r = item // pages
                p = item - r * pages
                mPageRows[r, p] = Int64(mTable[Int64(mSlots[r]), p])
            for item in cutlass.range(tid, n * B, THREADS):
                r = item // B
                j = item - r * B
                slot = Int64(mSlots[r])
                pos = self.ring_pos(Int64(mCtxLen[slot]) + Int64(j), ring_rows)
                mBlkPage[item] = Int64(mTable[slot, pos // page_size])
                mBlkOff[item] = pos % page_size


@functools.cache
def _compiled(width, block, ring, seq_lens_dtype):
    kernel = GenPrepareKernel(width, block, ring)
    n = cute.sym_int()
    slots = cute.sym_int()
    pages = cute.sym_int()
    vocab = cute.sym_int()
    vecs = cute.sym_int()
    nw = cute.sym_int()
    nb = cute.sym_int()
    i32, i64 = cutlass.Int32, cutlass.Int64
    bf16 = cutlass.BFloat16
    return cute.compile(
        kernel,
        _row_major(i32, (n,)),
        _row_major(i32, (n, width)),
        _row_major(i64, (n,)),
        _row_major(i64, (slots,)),
        _row_major(i32, (slots, pages)),
        _row_major(bf16, (vocab, vecs, VEC), 16),
        _row_major(bf16, (vecs, VEC), 16),
        _row_major(bf16, (nb, vecs, VEC), 16),
        _row_major(i64, (n,)),
        _row_major(i64, (n, block)),
        _row_major(i64, (nw,)),
        _row_major(i64, (nw,)),
        _row_major(cutlass.Uint8, (nw,)),
        _row_major(i64, (n,)),
        _row_major(i64, (n, pages)),
        _row_major(i64, (nb,)),
        _row_major(i64, (nb,)),
        _row_major(_TORCH_TO_CUTLASS[seq_lens_dtype], (n,)),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def gen_prepare(
    num_accepted: torch.Tensor,
    accepted_tokens: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    page_table: torch.Tensor,
    embed_weight: torch.Tensor,
    mask_row: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block: int,
    max_ctx: int,
    dummy_slot: int,
    ring_rows: int,
    page_size: int,
) -> dict:
    """Launch :class:`GenPrepareKernel`; returns its outputs by name (see the kernel's arguments).

    ``seq_lens`` (the generation requests' slice) and ``ctx_len`` are written in place.
    """
    n, width = accepted_tokens.shape
    hidden = embed_weight.shape[1]
    device = accepted_tokens.device
    i64 = dict(dtype=torch.int64, device=device)
    out = dict(
        noise=torch.empty((n, block, hidden), dtype=embed_weight.dtype, device=device),
        bonus=torch.empty((n,), **i64),
        query_positions=torch.empty((n, block), **i64),
        ctx_positions=torch.empty((n * width,), **i64),
        store_rows=torch.empty((n * width,), **i64),
        store_mask=torch.empty((n * width,), dtype=torch.uint8, device=device),
        num_ctx=torch.empty((n,), **i64),
        page_rows=torch.empty((n, page_table.shape[1]), **i64),
        block_pages=torch.empty((n * block,), **i64),
        block_offsets=torch.empty((n * block,), **i64),
    )
    if n == 0:
        return out
    _compiled(width, block, bool(ring_rows), seq_lens.dtype)(
        num_accepted,
        accepted_tokens,
        slots,
        ctx_len,
        page_table,
        embed_weight.view(embed_weight.shape[0], hidden // VEC, VEC),
        mask_row.view(hidden // VEC, VEC),
        out["noise"].view(n * block, hidden // VEC, VEC),
        out["bonus"],
        out["query_positions"],
        out["ctx_positions"],
        out["store_rows"],
        out["store_mask"],
        out["num_ctx"],
        out["page_rows"],
        out["block_pages"],
        out["block_offsets"],
        seq_lens,
        max_ctx,
        dummy_slot,
        int(ring_rows or 0),
        page_size,
    )
    return out


__all__ = ["GenPrepareKernel", "gen_prepare"]
