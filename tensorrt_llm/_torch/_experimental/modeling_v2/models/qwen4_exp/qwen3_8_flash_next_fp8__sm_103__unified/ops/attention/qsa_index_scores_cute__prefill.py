# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA index scores of a context chunk's rows on SM103 tensor cores (CuTe DSL, tcgen05).

Bitwise equal to the family's `_qsa_paged_index_scores_qtile_kernel` (`only_visible_blocks`):
columns at and past a row's visible block count are unspecified. A persistent CTA takes
(64-row tile, `CHUNK_TILES` x 128-block chunk) items; per tile one tcgen05 MMA chain
`D[block, 4 r + h]` (M 128, N 256, K 128, bf16 in, fp32 in tensor memory, two accumulators).
Warps 9-10 gather q rows and keys with `cp.async` into a `STAGES`-slot SW128 ring, warp 8 issues
the MMAs, warps 0-7 store the scores. A tile spanning several requests runs once per request.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op

try:
    from cutlass.memory import SmemAllocator
except ImportError:  # DSL releases before the memory namespace split
    from cutlass.utils import SmemAllocator

BF16 = cutlass.BFloat16
HEADS = 4
DIM = 128
#: Query rows a tile (the MMA's N = rows x heads).
QROWS = 64
BN = QROWS * HEADS
#: Compressed blocks a tile (the MMA's M).
BM = 128
#: Key ring slots (32 KB each); with the 64 KB query operand, ~192 KB of shared memory.
STAGES = 4
CHUNK_TILES = 8
#: Epilogue warps: two per tensor-memory lane quadrant, each taking half the accumulator.
EPI_WARPS = 8
MMA_WARP = 8
LOAD_WARP0 = 9
LOAD_WARPS = 2
#: Accumulator columns an epilogue thread loads at a time (8 rows x 4 heads).
EPI_COLS = 32
#: Items a CTA's metadata table holds (more: each role derives an item's metadata itself).
MAX_ITEMS = 64
THREADS = 32 * (EPI_WARPS + 1 + LOAD_WARPS)
LOAD_THREADS = 32 * LOAD_WARPS
#: Keys a page holds: one load thread's two token rows (`__init__`'s assert).
KEYS_PER_PAGE = 2 * (LOAD_THREADS // 16)


@dsl_user_op
def _fence_before(*, loc=None, ip=None):
    _nvvm.tcgen05_fence(_nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def _fence_after(*, loc=None, ip=None):
    _nvvm.tcgen05_fence(_nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


class _IndexScoresKernel:
    def __init__(self, *, tokens_per_block, compress_ratio):
        assert tokens_per_block % compress_ratio == 0
        self.tpb = tokens_per_block
        self.cr = compress_ratio
        self.kpp = tokens_per_block // compress_ratio  # keys a page
        assert BM % self.kpp == 0 and BM // self.kpp <= 32
        assert self.kpp == KEYS_PER_PAGE, "a thread's keys: two token rows of each page"
        self.pages_per_tile = BM // self.kpp

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        cache: cute.Pointer,
        table: cute.Pointer,
        pos: cute.Pointer,
        req: cute.Pointer,
        out_t: cute.Tensor,
        rows: Int32,
        width: Int32,
        q_stride_row: Int32,
        cache_stride_page: Int32,
        cache_stride_token: Int32,
        table_stride: Int32,
        out_stride: Int32,
        scale: Float32,
        num_ctas: Int32,
        stream,
    ):
        # The output arrives as a tensor, as the TVM-FFI entry takes its stream from a GPU
        # tensor; the kernel addresses it through its pointer and row stride.
        out = out_t.iterator

        def make_mma():
            return sm100_utils.make_trivial_tiled_mma(
                BF16,
                BF16,
                OperandMajorMode.K,
                OperandMajorMode.K,
                Float32,
                tcgen05.CtaGroup.ONE,
                (BM, BN),
                tcgen05.OperandSource.SMEM,
            )

        mma, mma_acc = make_mma(), make_mma()
        a_layout = sm100_utils.make_smem_layout_a(mma, (BM, BN, DIM), BF16, STAGES)
        b_layout = sm100_utils.make_smem_layout_b(mma, (BM, BN, DIM), BF16, 1)

        @cute.struct
        class SharedStorage:
            k_full: cute.struct.MemRange[Int64, STAGES]
            k_empty: cute.struct.MemRange[Int64, STAGES]
            q_full: cute.struct.MemRange[Int64, 1]
            q_empty: cute.struct.MemRange[Int64, 1]
            acc_full: cute.struct.MemRange[Int64, 2]
            acc_empty: cute.struct.MemRange[Int64, 2]
            tmem_holding: Int32
            vis: cute.struct.MemRange[Int32, QROWS]
            meta_starts: cute.struct.MemRange[Int64, MAX_ITEMS]
            meta_tot: cute.struct.MemRange[Int32, MAX_ITEMS]
            meta_hi: cute.struct.MemRange[Int32, MAX_ITEMS]
            meta_req: cute.struct.MemRange[Int32, MAX_ITEMS]
            sA: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(a_layout)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(b_layout)], 1024]

        self.shared_storage = SharedStorage
        self.kernel(
            mma,
            mma_acc,
            a_layout,
            b_layout,
            q,
            cache,
            table,
            pos,
            req,
            out,
            rows,
            width,
            q_stride_row,
            cache_stride_page,
            cache_stride_token,
            table_stride,
            out_stride,
            scale,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def item_geometry(self, item: Int32, nrt: Int32, rows: Int32, width: Int32):
        """Items chunk-major, so a wave's CTAs share a chunk's keys in L2."""
        ch = item // nrt
        rt = item - ch * nrt
        row0 = rt * QROWS
        rows_in = cutlass.min(rows - row0, Int32(QROWS))
        c0 = ch * (BM * CHUNK_TILES)
        c1 = cutlass.min(c0 + BM * CHUNK_TILES, width)
        return row0, rows_in, c0, c1

    @cute.jit
    def item_meta(self, meta, li: Int32, nmine: Int32, mReq, mPos, row0, rows_in, c0, c1, lane):
        """(run starts, tiles, first run's window bound, first run's request) of the CTA's
        item `li`: from the table, or derived here past it."""
        mS = cute.make_tensor(meta[0], cute.make_layout((MAX_ITEMS,)))
        mT = cute.make_tensor(meta[1], cute.make_layout((MAX_ITEMS,)))
        mH = cute.make_tensor(meta[2], cute.make_layout((MAX_ITEMS,)))
        mR = cute.make_tensor(meta[3], cute.make_layout((MAX_ITEMS,)))
        starts = Int64(0)
        total = Int32(0)
        hi1 = Int32(0)
        req1 = Int32(0)
        if nmine <= MAX_ITEMS:
            starts = mS[li]
            total = mT[li]
            hi1 = mH[li]
            req1 = mR[li]
        else:
            starts = self.group_starts(mReq, row0, rows_in, lane)
            s = Int32(0)
            while s < rows_in:
                e = self.group_end(starts, s, rows_in)
                nt, vis = self.group_tiles(mPos, row0, e, c0, c1)
                if s == 0:
                    hi1 = cutlass.min(vis, c1)
                total = total + nt
                s = e
            req1 = Int32(mReq[row0])
        return starts, total, hi1, req1

    @cute.jit
    def group_starts(self, mReq, row0: Int32, rows_in: Int32, lane: Int32):
        """Bit j set where tile row j starts a run of one request (warp-uniform)."""
        starts = Int64(0)
        for half in cutlass.range_constexpr(2):
            j = lane + 32 * half
            flag = cutlass.Boolean(False)
            if j < rows_in:
                if j == 0:
                    flag = True
                else:
                    flag = Int64(mReq[row0 + j]) != Int64(mReq[row0 + j - 1])
            bal = cute.arch.vote_ballot_sync(flag)
            starts = starts | ((Int64(bal) & Int64(0xFFFFFFFF)) << Int64(32 * half))
        return starts

    @cute.jit
    def group_end(self, starts: Int64, s: Int32, rows_in: Int32):
        """The first row after `s` that starts another run, or `rows_in`."""
        rest = starts & ~((Int64(2) << Int64(s)) - Int64(1))
        e = rows_in
        if rest != Int64(0):
            low = rest & (Int64(0) - rest)
            m = low - Int64(1)
            e = Int32(cute.arch.popc(Int32(m & Int64(0xFFFFFFFF)))) + Int32(
                cute.arch.popc(Int32(m >> Int64(32)))
            )
        return e

    @cute.jit
    def group_tiles(self, mPos, row0: Int32, e: Int32, c0: Int32, c1: Int32):
        """Tiles a run ending before row `e` walks in the chunk, and its widest window."""
        vis = (Int32(mPos[row0 + e - 1]) + 1) // self.cr
        hi = cutlass.min(vis, c1)
        ntiles = Int32(0)
        if hi > c0:
            ntiles = (hi - c0 + (BM - 1)) // BM
        return ntiles, vis

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(
        self,
        mma: cute.TiledMma,
        mma_acc: cute.TiledMma,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        q: cute.Pointer,
        cache: cute.Pointer,
        table: cute.Pointer,
        pos: cute.Pointer,
        req: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        width: Int32,
        q_stride_row: Int32,
        cache_stride_page: Int32,
        cache_stride_token: Int32,
        table_stride: Int32,
        out_stride: Int32,
        scale: Float32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        mma_acc.set(tcgen05.Field.ACCUMULATE, True)

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        k_full = storage.k_full.data_ptr()
        k_empty = storage.k_empty.data_ptr()
        q_full = storage.q_full.data_ptr()
        q_empty = storage.q_empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        acc_empty = storage.acc_empty.data_ptr()
        if tidx == 0:
            for i in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(k_full + i, LOAD_THREADS)
                cute.arch.mbarrier_init(k_empty + i, 1)
            cute.arch.mbarrier_init(q_full, LOAD_THREADS)
            cute.arch.mbarrier_init(q_empty, 1)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(acc_full + i, 1)
                cute.arch.mbarrier_init(acc_empty + i, 32 * EPI_WARPS)
        cute.arch.mbarrier_init_fence()

        tmem_users = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (EPI_WARPS + 1))
        if warp_idx == MMA_WARP:
            cute.arch.alloc_tmem(512, storage.tmem_holding.ptr)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()

        thr = mma.get_slice(0)
        sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        per_row = cute.make_layout((1 << 30,))
        mReq = cute.make_tensor(req, per_row)
        mPos = cute.make_tensor(pos, per_row)
        # The widest visible window of the call bounds the chunks (the table is max_seq_len wide).
        sMax = cute.make_tensor(storage.vis.data_ptr(), cute.make_layout((QROWS,)))
        vmax = Int32(0)
        for i in cutlass.range(tidx, rows, THREADS, unroll=4):
            vmax = cutlass.max(vmax, (Int32(mPos[i]) + 1) // self.cr)
        vmax = cute.arch.warp_redux_sync(vmax, "max")
        if lane == 0:
            sMax[warp_idx] = vmax
        cute.arch.sync_threads()
        vmax = Int32(0)
        if lane < THREADS // 32:
            vmax = sMax[lane]
        vmax = cute.arch.warp_redux_sync(vmax, "max")
        table_width = width
        width = cutlass.min(width, vmax)
        cute.arch.sync_threads()
        nrt = (rows + (QROWS - 1)) // QROWS
        nch = (width + (BM * CHUNK_TILES - 1)) // (BM * CHUNK_TILES)
        items = nrt * nch
        # The CTA's items' metadata (see `item_meta`), once, a warp an item.
        meta = (
            storage.meta_starts.data_ptr(),
            storage.meta_tot.data_ptr(),
            storage.meta_hi.data_ptr(),
            storage.meta_req.data_ptr(),
        )
        mS = cute.make_tensor(meta[0], cute.make_layout((MAX_ITEMS,)))
        mT = cute.make_tensor(meta[1], cute.make_layout((MAX_ITEMS,)))
        mH = cute.make_tensor(meta[2], cute.make_layout((MAX_ITEMS,)))
        mR = cute.make_tensor(meta[3], cute.make_layout((MAX_ITEMS,)))
        nmine = Int32(0)
        if bid < items:
            nmine = (items - bid + nctas - 1) // nctas
        if nmine <= MAX_ITEMS:
            for li in cutlass.range(warp_idx, nmine, THREADS // 32, unroll=1):
                row0, rows_in, c0, c1 = self.item_geometry(bid + li * nctas, nrt, rows, width)
                starts = self.group_starts(mReq, row0, rows_in, lane)
                total = Int32(0)
                hi1 = Int32(0)
                s = Int32(0)
                while s < rows_in:
                    e = self.group_end(starts, s, rows_in)
                    nt, vis = self.group_tiles(mPos, row0, e, c0, c1)
                    if s == 0:
                        hi1 = cutlass.min(vis, c1)
                    total = total + nt
                    s = e
                if lane == 0:
                    mS[li] = starts
                    mT[li] = total
                    mH[li] = hi1
                    mR[li] = Int32(mReq[row0])
        cute.arch.sync_threads()

        if warp_idx >= LOAD_WARP0:
            self.load_role(
                tidx - 32 * LOAD_WARP0,
                lane,
                bid,
                nctas,
                items,
                nrt,
                meta,
                sA,
                storage.sA.data_ptr(),
                sB,
                q,
                cache,
                table,
                mReq,
                mPos,
                rows,
                width,
                q_stride_row,
                cache_stride_page,
                cache_stride_token,
                table_stride,
                k_full,
                k_empty,
                q_full,
                q_empty,
            )
        elif warp_idx == MMA_WARP:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding.ptr)
            self.mma_role(
                mma,
                mma_acc,
                thr,
                sA,
                sB,
                tmem_ptr,
                lane,
                bid,
                nctas,
                items,
                nrt,
                meta,
                mReq,
                mPos,
                rows,
                width,
                k_full,
                k_empty,
                q_full,
                q_empty,
                acc_full,
                acc_empty,
            )
            tmem_users.arrive_and_wait()
            cute.arch.dealloc_tmem(tmem_ptr, 512)
        else:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding.ptr)
            self.epilogue_role(
                thr,
                tmem_ptr,
                tidx,
                lane,
                bid,
                nctas,
                items,
                nrt,
                meta,
                mReq,
                mPos,
                storage.vis.data_ptr(),
                out,
                rows,
                width,
                table_width,
                out_stride,
                scale,
                acc_full,
                acc_empty,
            )
            tmem_users.arrive_and_wait()

    # ------------------------------------------------------------------ loaders (warps 9-10)
    @cute.jit
    def load_role(
        self,
        lt: Int32,
        lane: Int32,
        bid: Int32,
        nctas: Int32,
        items: Int32,
        nrt: Int32,
        meta,
        sA,
        sA_raw: cute.Pointer,
        sB,
        q: cute.Pointer,
        cache: cute.Pointer,
        table: cute.Pointer,
        mReq,
        mPos,
        rows: Int32,
        width: Int32,
        q_stride_row: Int32,
        cache_stride_page: Int32,
        cache_stride_token: Int32,
        table_stride: Int32,
        k_full,
        k_empty,
        q_full,
        q_empty,
    ):
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BF16, num_bits_per_copy=128
        )
        # The query operand's element-coordinate view: (8 elements, (chunk, 64-dim block), row).
        sB_c = cute.make_tensor(
            sB.iterator,
            cute.make_layout((8, (8, DIM // 64), BN), stride=(1, (8, BN * 64), 64)),
        )
        chunk = lt % 16
        kq = lt // 16
        kstep: cutlass.Constexpr = LOAD_THREADS // 16
        toff = []
        dbase = []
        for v in cutlass.range_constexpr(self.kpp // kstep):
            toff.append(((kq + kstep * v) * self.cr + self.cr - 1) * cache_stride_token + chunk * 8)
            # SW128 K-major: 16-byte chunk c of row m at chunk c ^ (m % 8) of its 128-byte row,
            # each 64-dim half a BM x 128-byte block.
            kr = kq + kstep * v
            dbase.append((chunk // 8) * (BM * 64) + kr * 64 + ((chunk % 8) ^ kr) * 8)
        ring = Int32(0)
        ring_phase = Int32(1)
        q_phase = Int32(1)
        nmine = Int32(0)
        if bid < items:
            nmine = (items - bid + nctas - 1) // nctas
        for li in cutlass.range(0, nmine, 1, unroll=1):
            row0, rows_in, c0, c1 = self.item_geometry(bid + li * nctas, nrt, rows, width)
            starts, total, hi1, req1 = self.item_meta(
                meta, li, nmine, mReq, mPos, row0, rows_in, c0, c1, lane
            )
            if total > 0:
                # The item's query rows: B row n = 4 r + h, 16 chunks of 16 bytes a row.
                cute.arch.mbarrier_wait(q_empty, q_phase)
                q_phase = q_phase ^ 1
                for it in cutlass.range_constexpr(BN * 16 // LOAD_THREADS):
                    x = lt + LOAD_THREADS * it
                    n = x // 16
                    c = x % 16
                    r = n // HEADS
                    h = n % HEADS
                    if r < rows_in:
                        src = cute.make_tensor(
                            q
                            + cute.assume(
                                Int64(row0 + r) * q_stride_row + h * DIM + c * 8, divby=8
                            ),
                            cute.make_layout((8,)),
                        )
                        cute.copy(atom, src, sB_c[(None, (c % 8, c // 8), n)])
                cute.arch.cp_async_mbarrier_arrive_noinc(q_full)
                s = Int32(0)
                while s < rows_in:
                    e = self.group_end(starts, s, rows_in)
                    nt = total
                    hi = hi1
                    r_id = Int64(req1)
                    if starts != Int64(1):
                        # Several runs: each its own tiles, window and request.
                        nt, vis = self.group_tiles(mPos, row0, e, c0, c1)
                        hi = cutlass.min(vis, c1)
                        r_id = Int64(mReq[row0 + s])
                    mTable = cute.make_tensor(table, cute.make_layout((1 << 30,)))
                    # Lane p holds the tile's page p; the next tile's are read during its copies.
                    page_next = self.tile_page(mTable, r_id, table_stride, c0, hi, lane)
                    for j in cutlass.range(0, nt, 1, unroll=1):
                        b0 = c0 + j * BM
                        page = page_next
                        if j + 1 < nt:
                            page_next = self.tile_page(
                                mTable, r_id, table_stride, b0 + BM, hi, lane
                            )
                        cute.arch.mbarrier_wait(k_empty + ring, ring_phase)
                        full = b0 + BM <= hi
                        # Key m = kq + kstep * it: page it // 2 (a broadcast shuffle), token row
                        # kq + kstep * (it % 2); only a partial tile tests the bound per key.
                        sbase = ring * (BM * DIM)
                        if full:
                            self.copy_keys(
                                atom,
                                cache,
                                sA_raw,
                                page,
                                cache_stride_page,
                                toff,
                                dbase,
                                sbase,
                                kq,
                                b0,
                                hi,
                                False,
                            )
                        else:
                            self.copy_keys(
                                atom,
                                cache,
                                sA_raw,
                                page,
                                cache_stride_page,
                                toff,
                                dbase,
                                sbase,
                                kq,
                                b0,
                                hi,
                                True,
                            )
                        cute.arch.cp_async_mbarrier_arrive_noinc(k_full + ring)
                        ring = ring + 1
                        if ring == STAGES:
                            ring = Int32(0)
                            ring_phase = ring_phase ^ 1
                    s = e
        # Tail: every slot released, so no load is in flight when the CTA retires.
        for i in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_wait(k_empty + ring, ring_phase)
            ring = ring + 1
            if ring == STAGES:
                ring = Int32(0)
                ring_phase = ring_phase ^ 1

    @cute.jit
    def copy_keys(
        self,
        atom,
        cache,
        sA_raw,
        page,
        cache_stride_page,
        toff,
        dbase,
        sbase,
        kq,
        b0,
        hi,
        bounded: cutlass.Constexpr,
    ):
        kstep: cutlass.Constexpr = LOAD_THREADS // 16
        for pp in cutlass.range_constexpr(BM // self.kpp):
            pg = cute.arch.shuffle_sync(page, offset=pp)
            pbase = Int64(pg) * cache_stride_page
            for par in cutlass.range_constexpr(2):
                it = 2 * pp + par
                src = cute.make_tensor(
                    cache + cute.assume(pbase + toff[par], divby=8), cute.make_layout((8,))
                )
                dst = cute.make_tensor(
                    sA_raw + cute.assume(dbase[par] + sbase + pp * (self.kpp * 64), divby=8),
                    cute.make_layout((8,)),
                )
                if cutlass.const_expr(bounded):
                    if b0 + kq + kstep * it < hi:
                        cute.copy(atom, src, dst)
                else:
                    cute.copy(atom, src, dst)

    @cute.jit
    def tile_page(self, mTable, r_id, table_stride: Int32, b0: Int32, hi: Int32, lane: Int32):
        lp0 = (b0 * self.cr + self.cr - 1) // self.tpb
        page = Int32(0)
        if lane < self.pages_per_tile:
            if b0 + lane * self.kpp < hi:
                page = mTable[r_id * table_stride + lp0 + lane]
        return page

    # ------------------------------------------------------------------ MMA (warp 8)
    @cute.jit
    def mma_role(
        self,
        mma: cute.TiledMma,
        mma_acc: cute.TiledMma,
        thr,
        sA,
        sB,
        tmem_ptr,
        lane: Int32,
        bid: Int32,
        nctas: Int32,
        items: Int32,
        nrt: Int32,
        meta,
        mReq,
        mPos,
        rows: Int32,
        width: Int32,
        k_full,
        k_empty,
        q_full,
        q_empty,
        acc_full,
        acc_empty,
    ):
        tCrA = thr.make_fragment_A(sA)
        tCrB = thr.make_fragment_B(sB)
        tAcc_f = thr.make_fragment_C(thr.partition_shape_C((BM, BN)))
        ring = Int32(0)
        ring_phase = Int32(0)
        q_phase = Int32(0)
        t = Int32(0)  # tiles so far: accumulator t % 2, its phase (t // 2) % 2
        nmine = Int32(0)
        if bid < items:
            nmine = (items - bid + nctas - 1) // nctas
        for li in cutlass.range(0, nmine, 1, unroll=1):
            row0, rows_in, c0, c1 = self.item_geometry(bid + li * nctas, nrt, rows, width)
            starts, total, hi1, req1 = self.item_meta(
                meta, li, nmine, mReq, mPos, row0, rows_in, c0, c1, lane
            )
            if total > 0:
                cute.arch.mbarrier_wait(q_full, q_phase)
                q_phase = q_phase ^ 1
                for j in cutlass.range(0, total, 1, unroll=1):
                    ab = t % 2
                    cute.arch.mbarrier_wait(k_full + ring, ring_phase)
                    cute.arch.mbarrier_wait(acc_empty + ab, ((t // 2) % 2) ^ 1)
                    _fence_after()
                    tAcc = cute.make_tensor(tmem_ptr + ab * BN, tAcc_f.layout)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        if cutlass.const_expr(kb == 0):
                            cute.gemm(
                                mma, tAcc, tCrA[None, None, kb, ring], tCrB[None, None, kb, 0], tAcc
                            )
                        else:
                            cute.gemm(
                                mma_acc,
                                tAcc,
                                tCrA[None, None, kb, ring],
                                tCrB[None, None, kb, 0],
                                tAcc,
                            )
                    with cute.arch.elect_one():
                        tcgen05.commit(k_empty + ring)
                        tcgen05.commit(acc_full + ab)
                    ring = ring + 1
                    if ring == STAGES:
                        ring = Int32(0)
                        ring_phase = ring_phase ^ 1
                    t = t + 1
                with cute.arch.elect_one():
                    tcgen05.commit(q_empty)

    # ------------------------------------------------------------------ epilogue (warps 0-7)
    @cute.jit
    def head_sum(self, v0, v1, v2, v3):
        """The family's head reduction order, bit for bit."""
        z = Float32(0.0)
        r0 = cute.arch.fmax(v0, z)
        r1 = cute.arch.fmax(v1, z)
        r2 = cute.arch.fmax(v2, z)
        r3 = cute.arch.fmax(v3, z)
        return (r0 + r2) + (r1 + r3)

    @cute.jit
    def epilogue_role(
        self,
        thr,
        tmem_ptr,
        tidx: Int32,
        lane: Int32,
        bid: Int32,
        nctas: Int32,
        items: Int32,
        nrt: Int32,
        meta,
        mReq,
        mPos,
        vis_ptr: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        width: Int32,
        table_width: Int32,
        out_stride: Int32,
        scale: Float32,
        acc_full,
        acc_empty,
    ):
        epi_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * EPI_WARPS)
        sVis = cute.make_tensor(vis_ptr, cute.make_layout((QROWS,)))
        tAcc_f = thr.make_fragment_C(thr.partition_shape_C((BM, BN)))
        cols = EPI_COLS
        rpc = cols // HEADS  # rows a load
        half = tidx // 128  # the accumulator half (rows 0-31 / 32-63) this warp reads
        tAcc0 = cute.make_tensor(tmem_ptr, tAcc_f.layout)
        tAcc_c = cute.make_tensor(
            tAcc0.iterator, cute.composition(tAcc0.layout, cute.make_layout((BM, cols)))
        )
        cAcc = cute.make_identity_tensor((BM, BN))
        tCcAcc = thr.partition_C(cAcc)
        tCcAcc_c = cute.make_tensor(
            tCcAcc.iterator, cute.composition(tCcAcc.layout, cute.make_layout((BM, cols)))
        )
        load = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(cols)), Float32),
            tAcc_c,
        )
        thr_load = load.get_slice(tidx % 128)
        tL = thr_load.partition_S(tAcc_c)
        tLc = thr_load.partition_D(tCcAcc_c)
        mOut = cute.make_tensor(out, cute.make_layout((1 << 30,)))
        m = tLc[0][0]
        t = Int32(0)
        nmine = Int32(0)
        if bid < items:
            nmine = (items - bid + nctas - 1) // nctas
        for li in cutlass.range(0, nmine, 1, unroll=1):
            row0, rows_in, c0, c1 = self.item_geometry(bid + li * nctas, nrt, rows, width)
            starts, total, hi1, req1 = self.item_meta(
                meta, li, nmine, mReq, mPos, row0, rows_in, c0, c1, lane
            )
            if total > 0:
                # One request's 64 rows, every tile inside the table: stores unmasked.
                whole = (rows_in == QROWS) & (starts == Int64(1)) & (c0 + total * BM <= table_width)
                if not whole:
                    # The rows' visible block counts (bounded by the table width).
                    epi_bar.arrive_and_wait()
                    if tidx < QROWS:
                        v = Int32(0)
                        if tidx < rows_in:
                            v = cutlass.min((Int32(mPos[row0 + tidx]) + 1) // self.cr, table_width)
                        sVis[tidx] = v
                    epi_bar.arrive_and_wait()
                s = Int32(0)
                while s < rows_in:
                    e = self.group_end(starts, s, rows_in)
                    nt = total
                    if starts != Int64(1):
                        nt, _ = self.group_tiles(mPos, row0, e, c0, c1)
                    for j in cutlass.range(0, nt, 1, unroll=1):
                        ab = t % 2
                        b = c0 + j * BM + m
                        base = Int64(row0) * out_stride + b
                        cute.arch.mbarrier_wait(acc_full + ab, (t // 2) % 2)
                        _fence_after()
                        for cc in cutlass.range_constexpr(BN // 2 // cols):
                            rv = cute.make_rmem_tensor(tLc.shape, Float32)
                            col0 = half * (BN // 2) + cc * cols
                            cute.copy(
                                load,
                                cute.make_tensor(tL.iterator + ab * BN + col0, tL.layout),
                                rv,
                            )
                            cute.arch.fence_view_async_tmem_load()
                            if cutlass.const_expr(cc == BN // 2 // cols - 1):
                                _fence_before()
                                cute.arch.mbarrier_arrive(acc_empty + ab)
                            r0 = half * (QROWS // 2) + cc * rpc
                            for rr in cutlass.range_constexpr(rpc):
                                val = (
                                    self.head_sum(
                                        rv[HEADS * rr],
                                        rv[HEADS * rr + 1],
                                        rv[HEADS * rr + 2],
                                        rv[HEADS * rr + 3],
                                    )
                                    * scale
                                )
                                r = r0 + rr
                                if whole:
                                    mOut[base + Int64(r) * out_stride] = val
                                else:
                                    if r >= s:
                                        if r < e:
                                            if b < sVis[r]:
                                                mOut[base + Int64(r) * out_stride] = val
                        t = t + 1
                    s = e


_COMPILED: dict = {}
_SMS: dict = {}
_INDEX = {torch.int32: Int32, torch.int64: Int64}


def _compiled(tpb, cr, pos_dtype, req_dtype, out):
    key = (tpb, cr, pos_dtype, req_dtype)
    fn = _COMPILED.get(key)
    if fn is None:
        kernel = _IndexScoresKernel(tokens_per_block=tpb, compress_ratio=cr)
        gmem = cute.AddressSpace.gmem

        def fake(dtype, align):
            return make_ptr(dtype, 16 * 1024, gmem, assumed_align=align)

        fn = _COMPILED[key] = cute.compile(
            kernel,
            fake(BF16, 16),
            fake(BF16, 16),
            fake(Int32, 4),
            fake(_INDEX[pos_dtype], 4),
            fake(_INDEX[req_dtype], 4),
            from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            1,
            1,
            8,
            8,
            8,
            8,
            8,
            1.0,
            1,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return fn


def supports(q, index_cache, tokens_per_block, compress_ratio) -> bool:
    """A context-row call this kernel serves: bf16 4 x 128 index heads (unit dim stride, heads
    128 apart, 16-byte rows), a `[pages, tpb, 1, 128]` bf16 pool with 256-byte token rows."""
    return (
        q.dtype == torch.bfloat16
        and index_cache.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[1] == HEADS
        and q.shape[2] == DIM
        and q.stride(2) == 1
        and q.stride(1) == DIM
        and q.stride(0) % 8 == 0
        and q.data_ptr() % 16 == 0
        and index_cache.ndim == 4
        and index_cache.shape[2] == 1
        and index_cache.shape[3] == DIM
        and index_cache.stride(3) == 1
        and index_cache.stride(1) % 8 == 0
        and index_cache.stride(0) % 8 == 0
        and index_cache.data_ptr() % 16 == 0
        and tokens_per_block % compress_ratio == 0
        and tokens_per_block // compress_ratio == KEYS_PER_PAGE
        and torch.cuda.get_device_capability(q.device) == (10, 3)
    )


def qsa_index_scores_prefill(
    q,
    index_cache,
    block_table,
    query_positions,
    request_indices,
    tokens_per_block,
    compress_ratio,
):
    """The family's context-row index scores (`only_visible_blocks`): `[rows, max_blocks]` fp32,
    columns at and past each row's visible count unspecified."""
    rows = q.shape[0]
    max_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    out = torch.empty((rows, max_blocks), dtype=torch.float32, device=q.device)
    if rows == 0:
        return out
    fn = _compiled(
        tokens_per_block,
        compress_ratio,
        query_positions.dtype,
        request_indices.dtype,
        out,
    )
    dev = q.device.index if q.device.index is not None else 0
    sms = _SMS.get(dev)
    if sms is None:
        sms = _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    nch = -(-max_blocks // (BM * CHUNK_TILES))
    items = -(-rows // QROWS) * nch
    fn(
        q.data_ptr(),
        index_cache.data_ptr(),
        block_table.data_ptr(),
        query_positions.data_ptr(),
        request_indices.data_ptr(),
        out,
        rows,
        max_blocks,
        q.stride(0),
        index_cache.stride(0),
        index_cache.stride(1),
        block_table.stride(0),
        out.stride(0),
        float(q.shape[2] ** -0.5),
        max(1, min(items, sms)),
    )
    return out
