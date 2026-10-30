# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernel behind ``multi_request_split_kv``: decode attention partials of a batch (tcgen05, sm_103).

Only ``multi_request_split_kv`` imports this module, and only when the op is called. The op
module states the contract, the ownership and progress argument and the PDL memory-order
record; the comments here say what each role does.

The per-tile machinery is ``full_attention_split_kv_kernel``'s (the one-sequence kernel): a CTA
serves one pair of KV heads (``blockIdx.y``) with 384 threads -- two softmax warpgroups that
take alternating 128-key tiles, one MMA warp, one TMA loader warp, two other warps -- over a
ring of stages 32-key pages (K in two 2 KB boxes, one per head, V in one 4 KB box). Q' is
block-diagonal over the pair's two heads (row = head * 64 + token * 8 + q-head-in-group), so
one M=128 MMA per Q term and K block scores a 128-key tile; P goes to TMEM in E4M3 in two
halves; each warpgroup keeps its own running max / sum and TMEM O accumulator.

What is new is the work split. The grid's ``num_work`` CTAs per pair share the batch's pages
evenly: the pages each request attends (``[lo_b, ceil(L_b / 32))``, ``lo_b`` 0 for a full
layer, the window's first page for a sliding-window layer) are concatenated in request order
(``N`` pages in all), and CTA ``c`` of ``W = min(num_work, ceil(N / min_pages))`` takes
concatenated pages ``[c N / W, (c + 1) N / W)``. That range is cut at request boundaries into
segments; the CTA runs its segments one after the other with the ring, the barriers and the
TMEM buffers continuing across them, and stores one partial per segment (normalized O in FP16
and a log2-sum-exp per (token, query head)) at slot ``c + b`` (``b`` the request): along the
concatenation, consecutive segments step ``c`` or ``b`` (or both), so ``c + b`` is unique and
request ``b``'s slots are the contiguous ``[c_first(b) + b, c_last(b) + b]``.

Per segment the softmax warps build that request's Q' in one of two Q buffers (segment parity;
the next-but-one segment's Q is built right after a segment's epilogue, when every MMA that
read that buffer has completed), the MMA warp restarts the O accumulators (the first P.V of
each warpgroup does not accumulate) only after both warpgroups have read the previous
segment's O (``o_free``), and the warpgroups exchange their (max, sum) through a shared
buffer indexed by segment parity.

Every tensor is built here from a raw pointer and explicit extents and strides.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.typing import Float32, Int32

from .full_attention_split_kv_kernel import (
    BF16,
    EVICT_FIRST_POLICY,
    F8,
    F16,
    HKV,
    HQ,
    PAGE,
    PAGES_PER_TILE,
    RANK_KV_HEADS,
    RESCALE_COLS,
    ROWS,
    TILE_KEYS,
    D,
    G,
    _ab_layout,
    _fence_after,
    _fence_before,
    _private_mma,
    fmax3,
    log2_approx,
    pack_e4m3x4,
    rcp_approx,
    split_hi,
)

#: Query rows per request the row map is written for: one accepted token and up to 7 drafts.
MAX_ROWS = 8
#: Requests one launch can carry, per build: the 64-request build (512 rows of 8) serves every
#: batch of up to 64 requests; the 128-request build (1024 rows) serves 65..128, the decode steps
#: of a generation worker of ``max_batch_size`` up to 128. The builds differ only in the size of
#: the shared-memory partition tables and in how many requests each lane of the partition warp
#: scans (two or four); every other instruction is the same.
REQUEST_BUILDS = (64, 128)
MAX_REQUESTS = REQUEST_BUILDS[-1]
#: Page ids the loader stages per pass: one per lane.
ID_CHUNK = 32
#: log2(e): a device bmm1 scale times this is the softmax's exp2 scale.
LOG2E = 1.4426950408889634


class MultiRequestSplitKvKernel:
    """Balanced page-split partials of a batch's decode attention, one CTA per SM per pair.

    ``nq`` is the number of E4M3 terms Q enters as: 2 (``Q_hi + Q_lo``, BF16-class Q) or 1 (one
    E4M3 rounding). ``q_fp8`` takes Q as E4M3 bytes (the served build: the generation
    preprocess's E4M3 Q, used as is; needs ``nq`` 1). ``device_scale`` reads the QK scale from
    ``qk_scale_ptr[0]`` (fp32, times log2 e) after the dependency wait instead of taking
    ``qk_scale_log2`` as a launch argument. ``window`` 0 builds the full-attention
    kernel; ``window > 0`` the sliding-window one (row t of a request attends keys
    ``(L - M + t - window, L - M + t]``). ``rescale_cols`` is how many O columns the rare
    rescale holds in registers at once (16: the spill-free build). ``max_requests`` is the
    build's request capacity (``REQUEST_BUILDS``).
    """

    def __init__(
        self,
        enable_pdl: bool,
        kv_heads: int = 4,
        nq: int = 2,
        window: int = 0,
        stages: int = 16,
        rescale_cols: int = 16,
        rescale_log2: float = 4.0,
        p_pre_log2: float = 4.0,
        regs_softmax: int = 208,
        regs_other: int = 64,
        min_pages: int = 4,
        evict_first: bool = True,
        q_fp8: bool = False,
        device_scale: bool = False,
        max_requests: int = REQUEST_BUILDS[0],
    ):
        assert kv_heads in RANK_KV_HEADS, kv_heads
        assert max_requests in REQUEST_BUILDS, max_requests
        assert nq in (1, 2), nq
        assert not q_fp8 or nq == 1, "an E4M3 Q is one term"
        assert stages % PAGES_PER_TILE == 0, stages
        assert rescale_cols in RESCALE_COLS, rescale_cols
        self.enable_pdl = enable_pdl
        self.kv_heads = kv_heads
        self.pairs = kv_heads // HKV
        self.hq = kv_heads * G
        self.nq = nq
        self.q_fp8 = q_fp8
        self.device_scale = device_scale
        self.window = window
        self.stages = stages
        self.rescale_cols = rescale_cols
        self.tau = rescale_log2
        self.pre = p_pre_log2
        self.regs_softmax = regs_softmax
        self.regs_other = regs_other
        self.min_pages = min_pages
        self.evict_first = evict_first
        self.mma_warp = 8
        self.load_warp = 9
        self.part_warp = 10
        self.threads = 32 * 12
        self.tmem_warps = 9
        self.tmem_cols = 512
        self.max_requests = max_requests

    def pair_index(self):
        if self.pairs == 1:
            return 0
        _, pair, _ = cute.arch.block_idx()
        return pair

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,  # BF16 (F8 when q_fp8) [requests * rows, hq * 64], row stride q_stride
        pool_ptr: cute.Pointer,  # F8 [num_blocks, kv_heads, 32, 64], contiguous
        table_ptr: cute.Pointer,  # Int32 [requests, 2, table_width]
        seq_lens_ptr: cute.Pointer,  # Int32 [requests]
        part_o_ptr: cute.Pointer,  # F16 [slots, MAX_ROWS * hq, 64]
        part_lse_ptr: cute.Pointer,  # Float32 [slots, MAX_ROWS * hq]
        num_requests: Int32,
        rows: Int32,
        q_stride: Int32,
        num_blocks: Int32,
        table_width: Int32,
        table_stride_b: Int32,
        table_stride_kv: Int32,
        num_work: Int32,
        slots: Int32,
        qk_scale_log2: Float32,
        qk_scale_ptr: cute.Pointer,  # Float32 [1]: bmm1 scale (device_scale builds)
        stream,
    ):
        NKV = self.kv_heads
        q = cute.make_tensor(
            q_ptr,
            cute.make_layout((num_requests * rows, self.hq * D), stride=(q_stride, 1)),
        )
        pool = cute.make_tensor(
            pool_ptr,
            cute.make_layout((num_blocks, NKV, PAGE, D), stride=(NKV * PAGE * D, PAGE * D, D, 1)),
        )
        table = cute.make_tensor(
            table_ptr,
            cute.make_layout(
                (num_requests, 2, table_width), stride=(table_stride_b, table_stride_kv, 1)
            ),
        )
        seq_lens = cute.make_tensor(seq_lens_ptr, cute.make_layout(num_requests))
        qk_scale = cute.make_tensor(qk_scale_ptr, cute.make_layout(1))
        part_o = cute.make_tensor(part_o_ptr, cute.make_layout(slots * MAX_ROWS * self.hq * D))
        part_lse = cute.make_tensor(part_lse_ptr, cute.make_layout(slots * MAX_ROWS * self.hq))

        one = tcgen05.CtaGroup.ONE
        K = OperandMajorMode.K
        MN = OperandMajorMode.MN
        qk_mma = sm100_utils.make_trivial_tiled_mma(F8, F8, K, K, Float32, one, (128, PAGE))
        qk_mma_tile = sm100_utils.make_trivial_tiled_mma(
            F8, F8, K, K, Float32, one, (128, TILE_KEYS)
        )
        pv_mma = sm100_utils.make_trivial_tiled_mma(
            F8, F8, K, MN, Float32, one, (128, ROWS), tcgen05.OperandSource.TMEM
        )
        qk_tiler = (128, PAGE, ROWS)
        pv_tiler = (128, ROWS, PAGE)
        kinds = sm100_utils.SmemLayoutAtomKind
        # Q stages: (Q buffer of the segment's parity) x (Q term).
        sQ_layout = _ab_layout(qk_mma, qk_tiler, "A", kinds.K_SW64, 2 * self.nq)
        qk_mma_box = sm100_utils.make_trivial_tiled_mma(F8, F8, K, K, Float32, one, (128, PAGE))
        box_tiler = (128, PAGE, D)
        # K of a 128-key tile as one N=128 operand: per tile slot [head][128 keys][64 B], written
        # as 2 KB boxes, one per page and head, at slot * 16 KB + head * 8 KB + page * 2 KB.
        sK_layout = _ab_layout(
            qk_mma_tile, (128, TILE_KEYS, ROWS), "B", kinds.K_SW64,
            self.stages // PAGES_PER_TILE,
        )  # fmt: skip
        sKb_layout = _ab_layout(qk_mma_box, box_tiler, "B", kinds.K_SW64, HKV * self.stages)
        sV_layout = _ab_layout(pv_mma, pv_tiler, "B", kinds.MN_SW64, self.stages)
        p_layout = sm100_utils.make_smem_layout_a(pv_mma, pv_tiler, F8, 1)

        # V pool as the PV B operand: (dims (64, head 2), keys 32, pair + pairs * block)
        mV = cute.make_tensor(
            pool.iterator,
            cute.make_layout(
                ((D, HKV), PAGE, self.pairs * num_blocks),
                stride=((1, PAGE * D), D, HKV * PAGE * D),
            ),
        )
        # K pool as one head's B operand: (keys 32, dims 64, head + kv_heads * block)
        mKb = cute.make_tensor(
            pool.iterator,
            cute.make_layout((PAGE, D, NKV * num_blocks), stride=(D, 1, PAGE * D)),
        )
        tma_op = cpasync.CopyBulkTensorTileG2SOp(one)
        sV_one = cute.select(sV_layout, mode=[0, 1, 2])
        assert cute.cosize(sKb_layout) == cute.cosize(sK_layout)
        sKb_one = cute.select(sKb_layout, mode=[0, 1, 2])
        tma_atom_k, tma_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, mKb, sKb_one, box_tiler, qk_mma_box, (1, 1, 1)
        )
        k_bytes = HKV * cute.size_in_bytes(F8, sKb_one)
        tma_atom_v, tma_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, mV, sV_one, pv_tiler, pv_mma, (1, 1, 1)
        )
        self.page_bytes = k_bytes + cute.size_in_bytes(F8, sV_one)

        stages = self.stages
        nreq = self.max_requests

        @cute.struct
        class SharedStorage:
            kv_full: cute.struct.MemRange[cutlass.Int64, stages]
            kv_empty: cute.struct.MemRange[cutlass.Int64, stages]
            s_full: cute.struct.MemRange[cutlass.Int64, 2]
            p_full: cute.struct.MemRange[cutlass.Int64, 2]
            p_full2: cute.struct.MemRange[cutlass.Int64, 2]
            o_done: cute.struct.MemRange[cutlass.Int64, 2]
            q_full: cute.struct.MemRange[cutlass.Int64, 2]
            o_free: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: Int32
            # The batch's page partition (``partition``): per request its context length,
            # its first attended page and the exclusive prefix of attended pages (one more
            # entry: the total), then this CTA's [b0, b1) and [g0, g1).
            lens: cute.struct.MemRange[Int32, nreq]
            lo: cute.struct.MemRange[Int32, nreq]
            pref: cute.struct.MemRange[Int32, nreq + 4]
            info: cute.struct.MemRange[Int32, 4]
            # (segment parity, warpgroup, row, (max, sum))
            sml: cute.struct.MemRange[Float32, 2 * 2 * ROWS * 2]
            sQ: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(sV_layout)], 1024]

        self.shared_storage = SharedStorage

        self.kernel(
            qk_mma_tile,
            qk_mma_box,
            pv_mma,
            tma_atom_k,
            tma_k,
            tma_atom_v,
            tma_v,
            q,
            table,
            seq_lens,
            part_o,
            part_lse,
            num_requests,
            rows,
            qk_scale_log2,
            qk_scale,
            sQ_layout,
            sK_layout,
            sKb_layout,
            sV_layout,
            p_layout,
        ).launch(
            grid=(num_work, self.pairs, 1),
            block=[self.threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        qk_mma_tile: cute.TiledMma,
        qk_mma_box: cute.TiledMma,
        pv_mma: cute.TiledMma,
        tma_atom_k: cute.CopyAtom,
        tma_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_v: cute.Tensor,
        q: cute.Tensor,
        table: cute.Tensor,
        seq_lens: cute.Tensor,
        part_o: cute.Tensor,
        part_lse: cute.Tensor,
        num_requests: Int32,
        rows: Int32,
        qk_scale_log2: Float32,
        qk_scale: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKb_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        nwork, _, _ = cute.arch.grid_dim()

        if warp == self.load_warp:
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        kv_full = storage.kv_full.data_ptr()
        kv_empty = storage.kv_empty.data_ptr()
        s_full = storage.s_full.data_ptr()
        p_full = storage.p_full.data_ptr()
        p_full2 = storage.p_full2.data_ptr()
        o_done = storage.o_done.data_ptr()
        q_full = storage.q_full.data_ptr()
        o_free = storage.o_free.data_ptr()

        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(kv_full + i, 1)
                    cute.arch.mbarrier_init(kv_empty + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(s_full + i, 1)
                    cute.arch.mbarrier_init(p_full + i, 128)
                    cute.arch.mbarrier_init(p_full2 + i, 128)
                    cute.arch.mbarrier_init(o_done + i, 1)
                    cute.arch.mbarrier_init(q_full + i, 256)
                    cute.arch.mbarrier_init(o_free + i, 256)
            cute.arch.mbarrier_init_fence()

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * self.tmem_warps)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=self.mma_warp,
        )

        lens = cute.make_tensor(storage.lens.data_ptr(), cute.make_layout(self.max_requests))
        lo = cute.make_tensor(storage.lo.data_ptr(), cute.make_layout(self.max_requests))
        pref = cute.make_tensor(storage.pref.data_ptr(), cute.make_layout(self.max_requests + 4))
        info = cute.make_tensor(storage.info.data_ptr(), cute.make_layout(4))
        # Before the dependency wait: the batch's lengths (no kernel of the step writes them).
        if warp == self.part_warp:
            if cutlass.const_expr(self.max_requests == REQUEST_BUILDS[0]):
                self.partition(seq_lens, num_requests, rows, lens, lo, pref, info, cta, nwork)
            else:
                self.partition_wide(seq_lens, num_requests, rows, lens, lo, pref, info, cta, nwork)
        cute.arch.barrier()  # barrier inits and the partition visible to all warps

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)

        b0 = info[0]
        b1 = info[1]
        segs = (lens, lo, pref, info, rows)
        bars = (kv_full, kv_empty, s_full, p_full, o_done, q_full, p_full2, o_free)

        # Everything derived from the tiled MMAs is built here, before any role region: the
        # MMA role mutates the tiled MMAs (ACCUMULATE field) and later regions must not use
        # values defined inside it.
        pv_thr = pv_mma.get_slice(0)
        qk_tile_thr = qk_mma_tile.get_slice(0)
        sKb = storage.sK.get_tensor(sKb_layout.outer, swizzle=sKb_layout.inner)
        gK = cute.flat_divide(tma_k, (PAGE, D))
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sKb, 0, 3),
            cute.group_modes(qk_mma_box.get_slice(0).partition_B(gK), 0, 3),
        )
        gV = cute.flat_divide(tma_v, (ROWS, PAGE))
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(pv_thr.partition_B(gV), 0, 3),
        )
        tOrV = pv_thr.make_fragment_B(sV)
        o_acc_layout = pv_thr.make_fragment_C(pv_thr.partition_shape_C((128, ROWS))).layout
        s_tile_layout = qk_tile_thr.make_fragment_C(
            qk_tile_thr.partition_shape_C((128, TILE_KEYS))
        ).layout
        tSrQ = qk_tile_thr.make_fragment_A(sQ)
        tSrK = qk_tile_thr.make_fragment_B(sK)
        tScS = qk_tile_thr.partition_C(cute.make_identity_tensor((128, TILE_KEYS)))
        tOcO = pv_thr.partition_C(cute.make_identity_tensor((128, ROWS)))
        frags = (tSrQ, tSrK, tOrV, s_tile_layout, o_acc_layout, p_layout)

        if b1 > b0:
            # setmaxnreg sits inside each role's branch so ptxas allocates per role.
            if warp == self.load_warp:
                cute.arch.setmaxregister_decrease(self.regs_other)
                self.load_role(tma_atom_k, tKsK, tKgK, tma_atom_v, tVsV, tVgV, table, segs, bars)
            if warp >= self.load_warp + 1:
                cute.arch.setmaxregister_decrease(self.regs_other)
            # TMEM release: the MMA warp and the softmax warps meet (tmem_bar) before the free.
            if warp == self.mma_warp:
                cute.arch.setmaxregister_decrease(self.regs_other)
                self.mma_role(qk_mma_tile, pv_mma, frags, tmem, storage.sV.data_ptr(), bars, segs)
                tmem_bar.arrive_and_wait()
                tmem_end = tmem.retrieve_ptr(Float32)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_end)
            if warp < self.mma_warp:
                cute.arch.setmaxregister_increase(self.regs_softmax)
                self.softmax_role(
                    q,
                    s_tile_layout,
                    tScS,
                    o_acc_layout,
                    tOcO,
                    storage.sQ.data_ptr(),
                    storage.sml.data_ptr(),
                    tmem,
                    part_o,
                    part_lse,
                    qk_scale_log2,
                    qk_scale,
                    tidx,
                    warp,
                    cta,
                    bars,
                    segs,
                )
                tmem_bar.arrive_and_wait()

    # ------------------------------------------------------------------ partition
    @cute.jit
    def partition(self, seq_lens, num_requests, rows, lens, lo, pref, info, cta, nwork):
        """One warp: per request its length and first attended page, the exclusive prefix of
        the attended pages, and this CTA's requests ``[b0, b1)`` and pages ``[g0, g1)``."""
        lane = cute.arch.lane_idx()
        b_hi = lane + 32
        ok0 = lane < num_requests
        ok1 = b_hi < num_requests
        L0 = seq_lens[cutlass.min(lane, num_requests - 1)]
        L1 = seq_lens[cutlass.min(b_hi, num_requests - 1)]
        L0 = cutlass.select_(ok0, L0, Int32(0))
        L1 = cutlass.select_(ok1, L1, Int32(0))
        f0 = Int32(0)
        f1 = Int32(0)
        if cutlass.const_expr(self.window > 0):
            f0 = cutlass.max(L0 - rows - self.window + 1, Int32(0)) // PAGE
            f1 = cutlass.max(L1 - rows - self.window + 1, Int32(0)) // PAGE
        n0 = cutlass.select_(ok0, (L0 + PAGE - 1) // PAGE - f0, Int32(0))
        n1 = cutlass.select_(ok1, (L1 + PAGE - 1) // PAGE - f1, Int32(0))
        lens[lane] = L0
        lens[b_hi] = L1
        lo[lane] = f0
        lo[b_hi] = f1
        # Inclusive scans of requests 0..31 and 32..63; the second on top of the first's total.
        x0 = n0
        x1 = n1
        for d in cutlass.range_constexpr(5):
            y0 = cute.arch.shuffle_sync_up(x0, 1 << d)
            y1 = cute.arch.shuffle_sync_up(x1, 1 << d)
            x0 = x0 + cutlass.select_(lane >= (1 << d), y0, Int32(0))
            x1 = x1 + cutlass.select_(lane >= (1 << d), y1, Int32(0))
        total0 = cute.arch.shuffle_sync(x0, 31)
        x1 = x1 + total0
        total = cute.arch.shuffle_sync(x1, 31)
        pref[lane + 1] = x0
        pref[b_hi + 1] = x1
        if lane == 0:
            pref[0] = Int32(0)
        weff = cutlass.min(nwork, (total + self.min_pages - 1) // self.min_pages)
        weff_safe = cutlass.max(weff, Int32(1))
        g0 = cta * total // weff_safe
        g1 = (cta + 1) * total // weff_safe
        # b0: requests whose pages all lie before g0; b1: requests whose pages start before g1.
        e0 = x0 - n0
        e1 = x1 - n1
        b0 = cute.arch.popc(cute.arch.vote_ballot_sync(ok0 & (x0 <= g0))) + cute.arch.popc(
            cute.arch.vote_ballot_sync(ok1 & (x1 <= g0))
        )
        b1 = cute.arch.popc(cute.arch.vote_ballot_sync(ok0 & (e0 < g1))) + cute.arch.popc(
            cute.arch.vote_ballot_sync(ok1 & (e1 < g1))
        )
        keyed = cta < weff
        if lane == 0:
            info[0] = cutlass.select_(keyed, b0, Int32(0))
            info[1] = cutlass.select_(keyed, b1, Int32(0))
            info[2] = g0
            info[3] = g1

    @cute.jit
    def _request_pages(self, seq_lens, num_requests, rows, b):
        """``(ok, L, first, n)`` of request ``b`` (zeros past the batch): its context length, its
        first attended page and its number of attended pages."""
        ok = b < num_requests
        L = seq_lens[cutlass.min(b, num_requests - 1)]
        L = cutlass.select_(ok, L, Int32(0))
        f = Int32(0)
        if cutlass.const_expr(self.window > 0):
            f = cutlass.max(L - rows - self.window + 1, Int32(0)) // PAGE
        n = cutlass.select_(ok, (L + PAGE - 1) // PAGE - f, Int32(0))
        return ok, L, f, n

    @cute.jit
    def _warp_scan(self, x, lane):
        """Inclusive warp scan of ``x`` over the lanes."""
        for d in cutlass.range_constexpr(5):
            y = cute.arch.shuffle_sync_up(x, 1 << d)
            x = x + cutlass.select_(lane >= (1 << d), y, Int32(0))
        return x

    @cute.jit
    def partition_wide(self, seq_lens, num_requests, rows, lens, lo, pref, info, cta, nwork):
        """``partition`` of the 128-request build: lane ``l`` scans requests ``l``, ``l + 32``,
        ``l + 64`` and ``l + 96``, each 32-request chunk's inclusive scan on top of the totals of
        the chunks before it. The same integers as ``partition`` for every batch both take."""
        lane = cute.arch.lane_idx()
        b0_ = lane
        b1_ = lane + 32
        b2_ = lane + 64
        b3_ = lane + 96
        ok0, L0, f0, n0 = self._request_pages(seq_lens, num_requests, rows, b0_)
        ok1, L1, f1, n1 = self._request_pages(seq_lens, num_requests, rows, b1_)
        ok2, L2, f2, n2 = self._request_pages(seq_lens, num_requests, rows, b2_)
        ok3, L3, f3, n3 = self._request_pages(seq_lens, num_requests, rows, b3_)
        lens[b0_] = L0
        lens[b1_] = L1
        lens[b2_] = L2
        lens[b3_] = L3
        lo[b0_] = f0
        lo[b1_] = f1
        lo[b2_] = f2
        lo[b3_] = f3
        x0 = self._warp_scan(n0, lane)
        x1 = self._warp_scan(n1, lane) + cute.arch.shuffle_sync(x0, 31)
        x2 = self._warp_scan(n2, lane) + cute.arch.shuffle_sync(x1, 31)
        x3 = self._warp_scan(n3, lane) + cute.arch.shuffle_sync(x2, 31)
        total = cute.arch.shuffle_sync(x3, 31)
        pref[b0_ + 1] = x0
        pref[b1_ + 1] = x1
        pref[b2_ + 1] = x2
        pref[b3_ + 1] = x3
        if lane == 0:
            pref[0] = Int32(0)
        weff = cutlass.min(nwork, (total + self.min_pages - 1) // self.min_pages)
        weff_safe = cutlass.max(weff, Int32(1))
        g0 = cta * total // weff_safe
        g1 = (cta + 1) * total // weff_safe
        # b0: requests whose pages all lie before g0; b1: requests whose pages start before g1.
        b0 = (
            cute.arch.popc(cute.arch.vote_ballot_sync(ok0 & (x0 <= g0)))
            + cute.arch.popc(cute.arch.vote_ballot_sync(ok1 & (x1 <= g0)))
            + cute.arch.popc(cute.arch.vote_ballot_sync(ok2 & (x2 <= g0)))
            + cute.arch.popc(cute.arch.vote_ballot_sync(ok3 & (x3 <= g0)))
        )
        b1 = (
            cute.arch.popc(cute.arch.vote_ballot_sync(ok0 & (x0 - n0 < g1)))
            + cute.arch.popc(cute.arch.vote_ballot_sync(ok1 & (x1 - n1 < g1)))
            + cute.arch.popc(cute.arch.vote_ballot_sync(ok2 & (x2 - n2 < g1)))
            + cute.arch.popc(cute.arch.vote_ballot_sync(ok3 & (x3 - n3 < g1)))
        )
        keyed = cta < weff
        if lane == 0:
            info[0] = cutlass.select_(keyed, b0, Int32(0))
            info[1] = cutlass.select_(keyed, b1, Int32(0))
            info[2] = g0
            info[3] = g1

    @cute.jit
    def segment(self, b, segs):
        """Request ``b``'s segment of this CTA: ``(L, past, p_lo, p_hi, key_lo, key_hi,
        n_tiles, last_page_tail, last_idx)`` with pages request-relative."""
        lens, lo, pref, info, rows = segs
        g0 = info[2]
        g1 = info[3]
        L = lens[b]
        first = lo[b]
        start = pref[b]
        end = pref[b + 1]
        p_lo = cutlass.max(g0, start) - start + first
        p_hi = cutlass.min(g1, end) - start + first
        key_lo = p_lo * PAGE
        key_hi = cutlass.min(p_hi * PAGE, L)
        n_tiles = (p_hi - p_lo + PAGES_PER_TILE - 1) // PAGES_PER_TILE
        last_page_tail = (key_hi == L) & ((L % PAGE) != 0)
        last_idx = (L - 1) // PAGE - p_lo
        return L, L - rows, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail, last_idx

    # ------------------------------------------------------------------ load warp
    @cute.jit
    def load_role(self, tma_atom_k, tKsK, tKgK, tma_atom_v, tVsV, tVgV, table, segs, bars):
        S = self.stages
        kv_full, kv_empty, s_full, p_full, o_done, q_full, p_full2, o_free = bars
        lens, lo, pref, info, rows = segs
        lane = cute.arch.lane_idx()
        pair = self.pair_index()
        b0 = info[0]
        b1 = info[1]
        policy = cutlass.Int64(EVICT_FIRST_POLICY)
        # Pages may hold the step's own keys, which the predecessor writes.
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        gslot = Int32(0)
        for b in cutlass.range(b0, b1, unroll=1):
            (L, past, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail,
             last_idx) = self.segment(b, segs)  # fmt: skip
            n_mine = p_hi - p_lo
            n_slots = n_tiles * PAGES_PER_TILE
            for ch in cutlass.range((n_slots + ID_CHUNK - 1) // ID_CHUNK, unroll=1):
                base = ch * ID_CHUNK
                # This chunk's page ids, one per lane (the block table is not written by any
                # kernel of the step).
                src = cutlass.min(p_lo + base + lane, p_hi - 1)
                rk = table[b, 0, src]
                rv = table[b, 1, src]
                cnt = cutlass.min(Int32(ID_CHUNK), n_slots - base)
                for jj in cutlass.range(cnt, unroll=1):
                    idx = base + jj
                    g = gslot + idx
                    st = g % S
                    cute.arch.mbarrier_wait(kv_empty + st, (g // S + 1) % 2)
                    kid = cute.arch.shuffle_sync(rk, jj)
                    vid = cute.arch.shuffle_sync(rv, jj)
                    valid = idx < n_mine
                    kid = cutlass.select_(valid, kid, Int32(-1))
                    vid = cutlass.select_(valid, vid, Int32(-1))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(kv_full + st, self.page_bytes)
                    # This CTA's pair of the page's K and V heads; a padding slot (id -1) lies
                    # outside the tensor, so TMA fills it with zeros.
                    kpair = self.pairs * kid + pair
                    vpair = self.pairs * vid + pair
                    box = (st // PAGES_PER_TILE) * (HKV * PAGES_PER_TILE) + st % PAGES_PER_TILE
                    for h in cutlass.range_constexpr(HKV):
                        if cutlass.const_expr(self.evict_first):
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, 0, 0, HKV * kpair + h],
                                tKsK[None, box + h * PAGES_PER_TILE],
                                tma_bar_ptr=kv_full + st,
                                cache_policy=policy,
                            )
                        else:
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, 0, 0, HKV * kpair + h],
                                tKsK[None, box + h * PAGES_PER_TILE],
                                tma_bar_ptr=kv_full + st,
                            )
                    if cutlass.const_expr(self.evict_first):
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, 0, 0, vpair],
                            tVsV[None, st],
                            tma_bar_ptr=kv_full + st,
                            cache_policy=policy,
                        )
                    else:
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, 0, 0, vpair],
                            tVsV[None, st],
                            tma_bar_ptr=kv_full + st,
                        )
            gslot = gslot + n_slots

    # ------------------------------------------------------------------ MMA warp
    @cute.jit
    def qk_tile(self, qk_mma_in, tmem_ptr, buf, tSrQ, tSrK, kv_full, gt, qb, s_acc_layout):
        S = self.stages
        # A private copy: setting ACCUMULATE must not leak this region's value to the caller.
        qk_mma = _private_mma(qk_mma_in)
        for p in cutlass.range_constexpr(PAGES_PER_TILE):
            g = gt * PAGES_PER_TILE + p
            cute.arch.mbarrier_wait(kv_full + g % S, (g // S) % 2)
        slot = gt % (S // PAGES_PER_TILE)
        acc = cute.make_tensor(tmem_ptr + buf * 128, s_acc_layout)
        for qt in cutlass.range_constexpr(self.nq):
            for k in cutlass.range_constexpr(4):
                qk_mma.set(tcgen05.Field.ACCUMULATE, (qt + k) != 0)
                cute.gemm(
                    qk_mma,
                    acc,
                    tSrQ[None, None, k, qb * self.nq + qt],
                    tSrK[None, None, k, slot],
                    acc,
                )

    @cute.jit
    def pv_tile(self, pv_mma_in, tmem_ptr, buf, tOrP0, tOrV, o_acc_layout, t, gt, p0, p1):
        S = self.stages
        pv_mma = _private_mma(pv_mma_in)
        tOtO = cute.make_tensor(tmem_ptr + 256 + buf * 128, o_acc_layout)
        for p in cutlass.range_constexpr(p0, p1):
            st = (gt * PAGES_PER_TILE + p) % S
            a = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + buf * 128 + 8 * p, dtype=F8), tOrP0.layout
            )
            pv_mma.set(tcgen05.Field.ACCUMULATE, (t >= 2) | (p != 0))
            cute.gemm(pv_mma, tOtO, a[None, None, 0], tOrV[None, None, 0, st], tOtO)

    @cute.jit
    def mma_role(self, qk_mma, pv_mma, frags, tmem, sv_ptr, bars, segs):
        S = self.stages
        kv_full, kv_empty, s_full, p_full, o_done, q_full, p_full2, o_free = bars
        lens, lo, pref, info, rows = segs
        tSrQ, tSrK, tOrV, s_acc_layout, o_acc_layout, p_layout = frags
        tmem.allocate(self.tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tP = cute.make_tensor(tmem_ptr, p_layout.outer)
        tOrP0 = pv_mma.get_slice(0).make_fragment_A(tP)[None, None, None, 0]
        lane = cute.arch.lane_idx()
        sv_i32 = cute.make_tensor(cute.recast_ptr(sv_ptr, dtype=Int32), cute.make_layout(S * 1024))

        b0 = info[0]
        b1 = info[1]
        gt = Int32(0)  # tiles of earlier segments
        base0 = Int32(0)  # tiles of warpgroup 0 in earlier segments
        base1 = Int32(0)
        for b in cutlass.range(b0, b1, unroll=1):
            j = b - b0
            qb = j % 2
            (L, past, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail,
             last_idx) = self.segment(b, segs)  # fmt: skip
            vfix_tile = last_idx // PAGES_PER_TILE
            cute.arch.mbarrier_wait(q_full + qb, (j // 2) % 2)
            _fence_after()
            if n_tiles > 0:
                self.qk_tile(qk_mma, tmem_ptr, 0, tSrQ, tSrK, kv_full, gt, qb, s_acc_layout)
                with cute.arch.elect_one():
                    tcgen05.commit(s_full)
            if n_tiles > 1:
                self.qk_tile(qk_mma, tmem_ptr, 1, tSrQ, tSrK, kv_full, gt + 1, qb, s_acc_layout)
                with cute.arch.elect_one():
                    tcgen05.commit(s_full + 1)
            # Both warpgroups have read the previous segment's O before it is overwritten.
            if j > 0:
                cute.arch.mbarrier_wait(o_free, (j - 1) % 2)
            for t in cutlass.range(n_tiles, unroll=1):
                buf = t % 2
                u = t // 2
                base_b = cutlass.select_(buf == 0, base0, base1)
                cute.arch.mbarrier_wait(p_full + buf, (base_b + u) % 2)
                if last_page_tail & (t == vfix_tile):
                    # V rows at or past L in the last page may hold any bytes (an E4M3 NaN
                    # times P = 0 is NaN): zero them. This warp waited for that page's load in
                    # QK(t), and the stage cannot be refilled before PV(t)'s kv_empty commit.
                    st = (gt * PAGES_PER_TILE + last_idx) % S
                    r0 = L % PAGE
                    n_words = (PAGE - r0) * (D // 4)  # per head
                    for i in cutlass.range(lane, 2 * n_words, 32, unroll=1):
                        sv_i32[st * 1024 + (i // n_words) * 512 + r0 * 16 + i % n_words] = Int32(0)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_warp()
                _fence_after()
                # Pages 0-1 as soon as their P is in TMEM, pages 2-3 after the second half.
                self.pv_tile(pv_mma, tmem_ptr, buf, tOrP0, tOrV, o_acc_layout, t, gt + t, 0, 2)
                cute.arch.mbarrier_wait(p_full2 + buf, (base_b + u) % 2)
                _fence_after()
                self.pv_tile(pv_mma, tmem_ptr, buf, tOrP0, tOrV, o_acc_layout, t, gt + t, 2, 4)
                with cute.arch.elect_one():
                    tcgen05.commit(o_done + buf)
                    for p in cutlass.range_constexpr(PAGES_PER_TILE):
                        tcgen05.commit(kv_empty + ((gt + t) * PAGES_PER_TILE + p) % S)
                if t + 2 < n_tiles:
                    self.qk_tile(
                        qk_mma, tmem_ptr, buf, tSrQ, tSrK, kv_full, gt + t + 2, qb, s_acc_layout
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(s_full + buf)
            gt = gt + n_tiles
            base0 = base0 + (n_tiles + 1) // 2
            base1 = base1 + n_tiles // 2

    # ------------------------------------------------------------------ softmax warpgroups
    @cute.jit
    def build_q(self, q, b, rows, sq_ptr, qb, r, h, wg, tok, qh, row_valid):
        """This thread's 32 dims of row r of request b as E4M3 term(s), block-diagonal Q' in
        Q buffer ``qb``."""
        tok_c = cutlass.select_(row_valid, tok, 0)
        sq_i32 = cute.make_tensor(
            cute.recast_ptr(sq_ptr, dtype=Int32), cute.make_layout(2 * self.nq * 4096)
        )
        rw = cute.make_rmem_tensor((self.nq, 8), Int32)
        if cutlass.const_expr(self.q_fp8):
            # The row slice's 32 E4M3 bytes as 8 words (byte 0 = dim 0, the low byte).
            qoff = cute.assume((b * rows + tok_c) * q.stride[0] + qh * D + wg * (D // 2), divby=16)
            src = cute.make_tensor(
                cute.recast_ptr(q.iterator + qoff, dtype=Int32), cute.make_layout(8)
            )
            rqw = cute.make_rmem_tensor((8,), Int32)
            rqw.store(src.load())
            for e4 in cutlass.range_constexpr(8):
                rw[0, e4] = cutlass.select_(row_valid, rqw[e4], Int32(0))
        else:
            qoff = cute.assume((b * rows + tok_c) * q.stride[0] + qh * D + wg * (D // 2), divby=8)
            rqb = cute.make_rmem_tensor((D // 2,), BF16)
            rqb.store(cute.make_tensor(q.iterator + qoff, cute.make_layout(D // 2)).load())
            rq = cute.make_rmem_tensor((D // 2,), Float32)
            rq.store(rqb.load().to(Float32))
            zf = Float32(0.0)
            for e4 in cutlass.range_constexpr(8):
                e = 4 * e4
                v0 = cutlass.select_(row_valid, rq[e], zf)
                v1 = cutlass.select_(row_valid, rq[e + 1], zf)
                v2 = cutlass.select_(row_valid, rq[e + 2], zf)
                v3 = cutlass.select_(row_valid, rq[e + 3], zf)
                if cutlass.const_expr(self.nq == 2):
                    h0, h1, h2, h3 = split_hi(v0), split_hi(v1), split_hi(v2), split_hi(v3)
                    rw[0, e4] = pack_e4m3x4(h0, h1, h2, h3)
                    rw[1, e4] = pack_e4m3x4(v0 - h0, v1 - h1, v2 - h2, v3 - h3)
                else:
                    rw[0, e4] = pack_e4m3x4(v0, v1, v2, v3)
        zero4 = cute.make_rmem_tensor((4,), Int32)
        zero4.fill(0)
        for cl in cutlass.range_constexpr(2):  # this thread's two 16-byte chunks of the row slice
            cs = (2 * wg + cl) ^ ((r >> 1) & 3)
            for qt in cutlass.range_constexpr(self.nq):
                base = cute.assume((qb * self.nq + qt) * 4096 + r * 16 + cs * 4, divby=4)
                own = cute.make_tensor(sq_i32.iterator + base + h * 2048, cute.make_layout(4))
                oth = cute.make_tensor(sq_i32.iterator + base + (1 - h) * 2048, cute.make_layout(4))
                vals = cute.make_rmem_tensor((4,), Int32)
                for w in cutlass.range_constexpr(4):
                    vals[w] = rw[qt, 4 * cl + w]
                own.store(vals.load())
                oth.store(zero4.load())

    @cute.jit
    def softmax_role(
        self,
        q,
        s_tile_layout,
        tScS,
        o_acc_layout,
        tOcO,
        sq_ptr,
        sml_ptr,
        tmem,
        part_o,
        part_lse,
        qk_scale_log2,
        qk_scale,
        tidx,
        warp,
        cta,
        bars,
        segs,
    ):
        kv_full, kv_empty, s_full, p_full, o_done, q_full, p_full2, o_free = bars
        lens, lo, pref, info, rows = segs
        wg = warp // 4  # 0 or 1: takes tiles wg, wg + 2, ... of every segment
        pair = self.pair_index()
        r = tidx % 128  # row 0..127 = TMEM lane
        h = r // D
        ri = r % D
        tok = ri // G
        gq = ri % G
        qh = pair * HQ + h * G + gq
        row_valid = tok < rows
        b0 = info[0]
        b1 = info[1]

        # Q is the predecessor's output: read only after the dependency wait.
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        self.build_q(q, b0, rows, sq_ptr, 0, r, h, wg, tok, qh, row_valid)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.mbarrier_arrive(q_full)
        if b1 - b0 > 1:
            self.build_q(q, b0 + 1, rows, sq_ptr, 1, r, h, wg, tok, qh, row_valid)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(q_full + 1)

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        s_base = tmem_ptr + wg * 128
        # S tile (128 x 128 fp32): one row per thread
        tStS = cute.make_tensor(s_base, s_tile_layout)
        ld_s = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32), tStS
        )
        thr_ld_s = ld_s.get_slice(r)
        tLcS = thr_ld_s.partition_D(tScS)
        tLtS = thr_ld_s.partition_S(tStS)
        rS = cute.make_rmem_tensor(tLcS.shape, Float32)

        # The two 16-column halves of P (keys 0-63 and 64-127 of the tile), E4M3 in the S buffer.
        tStPh = cute.make_tensor(
            s_base, cute.composition(s_tile_layout, cute.make_layout((128, 16)))
        )
        st_ph = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32),
            tStPh,
        )
        thr_st_ph = st_ph.get_slice(r)
        tPtP0 = thr_st_ph.partition_D(tStPh)
        tPtP1 = thr_st_ph.partition_D(
            cute.make_tensor(
                s_base + 16, cute.composition(s_tile_layout, cute.make_layout((128, 16)))
            )
        )
        tScPh = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((128, 16)))
        )
        tPcPh = thr_st_ph.partition_S(tScPh)
        rPh0 = cute.make_rmem_tensor(tPcPh.shape, Float32)
        rPh0_i32 = cute.recast_tensor(rPh0, Int32)
        rPh1 = cute.make_rmem_tensor(tPcPh.shape, Float32)
        rPh1_i32 = cute.recast_tensor(rPh1, Int32)

        # O of this row's head in chunks of RC columns (the rare rescale).
        RC = self.rescale_cols
        o_chunk_layout = cute.composition(o_acc_layout, cute.make_layout((128, RC)))
        tOtO_ref = cute.make_tensor(tmem_ptr + 256, o_chunk_layout)
        ld_o = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(RC)), Float32),
            tOtO_ref,
        )
        st_o = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(RC)), Float32),
            tOtO_ref,
        )
        thr_ld_o = ld_o.get_slice(r)
        thr_st_o = st_o.get_slice(r)
        o_col0 = 256 + wg * 128 + h * D
        tOcO_h = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((128, RC)))
        )
        tLcO = thr_ld_o.partition_D(tOcO_h)
        rO = cute.make_rmem_tensor(tLcO.shape, Float32)

        # The epilogue's view: this thread's 32 output columns of its head from an accumulator.
        o_q_layout = cute.composition(o_acc_layout, cute.make_layout((128, D // 2)))
        tOq_ref = cute.make_tensor(tmem_ptr + 256, o_q_layout)
        ld_q = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32),
            tOq_ref,
        )
        thr_ld_q = ld_q.get_slice(r)
        tOcq = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((128, D // 2)))
        )
        tLcq = thr_ld_q.partition_D(tOcq)
        rO0 = cute.make_rmem_tensor(tLcq.shape, Float32)
        rO1 = cute.make_rmem_tensor(tLcq.shape, Float32)

        sc = qk_scale_log2
        if cutlass.const_expr(self.device_scale):
            # Written by the step's preprocess: read after the dependency wait (above).
            sc = qk_scale[0] * Float32(LOG2E)
        neg_inf = -Float32.inf
        bar2 = pipeline.NamedBarrier(barrier_id=2, num_threads=256)
        sml = cute.make_tensor(
            sml_ptr, cute.make_layout((2, 2, ROWS, 2), stride=(4 * ROWS, 2 * ROWS, 2, 1))
        )
        slot_rows = MAX_ROWS * self.hq
        R = tok * self.hq + qh
        base0 = Int32(0)
        base1 = Int32(0)
        for b in cutlass.range(b0, b1, unroll=1):
            j = b - b0
            (L, past, p_lo, p_hi, key_lo, key_hi, n_tiles, last_page_tail,
             last_idx) = self.segment(b, segs)  # fmt: skip
            qpos = past + tok
            lim = cutlass.min(key_hi - 1, qpos)
            low = Int32(0)
            if cutlass.const_expr(self.window > 0):
                low = qpos - self.window + 1
            base_w = cutlass.select_(wg == 0, base0, base1)
            m = neg_inf
            l = Float32(0.0)  # noqa: E741 - the softmax denominator
            my_tiles = (n_tiles - wg + 1) // 2
            for u in cutlass.range(my_tiles, unroll=1):
                t = wg + 2 * u
                cute.arch.mbarrier_wait(s_full + wg, (base_w + u) % 2)
                _fence_after()
                cute.copy(ld_s, tLtS, rS)
                cute.arch.fence_view_async_tmem_load()
                key0 = key_lo + t * TILE_KEYS
                if cutlass.const_expr(self.window > 0):
                    if (key0 + TILE_KEYS - 1 > lim) | (key0 < low):
                        for jj in cutlass.range_constexpr(TILE_KEYS):
                            rS[jj] = cutlass.select_(
                                (key0 + jj <= lim) & (key0 + jj >= low), rS[jj], neg_inf
                            )
                else:
                    if key0 + TILE_KEYS - 1 > lim:
                        for jj in cutlass.range_constexpr(TILE_KEYS):
                            rS[jj] = cutlass.select_(key0 + jj <= lim, rS[jj], neg_inf)
                mx0 = fmax3(rS[0], rS[1], rS[2])
                mx1 = fmax3(rS[3], rS[4], rS[5])
                mx2 = fmax3(rS[6], rS[7], rS[8])
                mx3 = fmax3(rS[9], rS[10], rS[11])
                for jj in cutlass.range_constexpr(12, TILE_KEYS - 4, 8):
                    mx0 = fmax3(mx0, rS[jj], rS[jj + 1])
                    mx1 = fmax3(mx1, rS[jj + 2], rS[jj + 3])
                    mx2 = fmax3(mx2, rS[jj + 4], rS[jj + 5])
                    mx3 = fmax3(mx3, rS[jj + 6], rS[jj + 7])
                mx0 = fmax3(mx0, rS[TILE_KEYS - 4], rS[TILE_KEYS - 3])
                mx1 = fmax3(mx1, rS[TILE_KEYS - 2], rS[TILE_KEYS - 1])
                m_new = fmax3(fmax3(mx0, mx1, mx2), mx3, m)
                need = (m_new - m) * sc > Float32(self.tau)
                do_rescale = need & (u > 0) & (m != neg_inf)
                # This warpgroup's previous P.V completed before S(t) was committed.
                if cute.arch.vote_any_sync(do_rescale):
                    factor = cutlass.select_(
                        do_rescale,
                        cute.math.exp2((m - m_new) * sc, fastmath=True),
                        Float32(1.0),
                    )
                    for c in cutlass.range_constexpr(D // RC):
                        tOc = cute.make_tensor(tmem_ptr + o_col0 + c * RC, o_chunk_layout)
                        cute.copy(ld_o, thr_ld_o.partition_S(tOc), rO)
                        cute.arch.fence_view_async_tmem_load()
                        for jj in cutlass.range_constexpr(0, RC, 2):
                            rO[jj], rO[jj + 1] = cute.arch.mul_packed_f32x2(
                                (rO[jj], rO[jj + 1]), (factor, factor)
                            )
                        cute.copy(st_o, rO, thr_st_o.partition_D(tOc))
                    cute.arch.fence_view_async_tmem_store()
                    l = l * factor  # noqa: E741
                m = cutlass.select_(need, m_new, m)
                m_safe = cutlass.select_(m == neg_inf, Float32(0.0), m)
                bias = Float32(self.pre) - m_safe * sc
                for hh in cutlass.range_constexpr(2):
                    for jj in cutlass.range_constexpr(64 * hh, 64 * hh + 64, 2):
                        rS[jj], rS[jj + 1] = cute.arch.fma_packed_f32x2(
                            (rS[jj], rS[jj + 1]), (sc, sc), (bias, bias)
                        )
                        rS[jj] = cute.math.exp2(rS[jj], fastmath=True)
                        rS[jj + 1] = cute.math.exp2(rS[jj + 1], fastmath=True)
                    if cutlass.const_expr(hh == 0):
                        for w in cutlass.range_constexpr(16):
                            rPh0_i32[w] = pack_e4m3x4(
                                rS[4 * w], rS[4 * w + 1], rS[4 * w + 2], rS[4 * w + 3]
                            )
                        cute.copy(st_ph, rPh0, tPtP0)
                        cute.arch.fence_view_async_tmem_store()
                        _fence_before()
                        cute.arch.mbarrier_arrive(p_full + wg)
                    else:
                        for w in cutlass.range_constexpr(16):
                            rPh1_i32[w] = pack_e4m3x4(
                                rS[64 + 4 * w],
                                rS[64 + 4 * w + 1],
                                rS[64 + 4 * w + 2],
                                rS[64 + 4 * w + 3],
                            )
                        cute.copy(st_ph, rPh1, tPtP1)
                        cute.arch.fence_view_async_tmem_store()
                        _fence_before()
                        cute.arch.mbarrier_arrive(p_full2 + wg)
                acc0 = (Float32(0.0), Float32(0.0))
                acc1 = (Float32(0.0), Float32(0.0))
                for jj in cutlass.range_constexpr(0, TILE_KEYS, 4):
                    acc0 = cute.arch.add_packed_f32x2(acc0, (rS[jj], rS[jj + 1]))
                    acc1 = cute.arch.add_packed_f32x2(acc1, (rS[jj + 2], rS[jj + 3]))
                l = l + ((acc0[0] + acc0[1]) + (acc1[0] + acc1[1]))  # noqa: E741

            # ---- epilogue: merge the two warpgroups' (O, m, l) of this row into one partial ----
            if cutlass.const_expr(self.enable_pdl):
                if b == b1 - 1:
                    # The dependent grid (the combine) waits for this grid's completion.
                    cute.arch.griddepcontrol_launch_dependents()
            n0 = (n_tiles + 1) // 2  # tiles of warpgroup 0
            n1 = n_tiles // 2
            my_n = cutlass.select_(wg == 0, n0, n1)
            ot_n = cutlass.select_(wg == 0, n1, n0)
            base_o = cutlass.select_(wg == 0, base1, base0)
            # Own last P.V first: it completes after the other warpgroup's P.V of the previous
            # tile, so the second wait is never more than one phase behind.
            if my_n > 0:
                cute.arch.mbarrier_wait(o_done + wg, (base_w + my_n - 1) % 2)
            if ot_n > 0:
                cute.arch.mbarrier_wait(o_done + (1 - wg), (base_o + ot_n - 1) % 2)
            _fence_after()
            sml[j % 2, wg, r, 0] = m
            sml[j % 2, wg, r, 1] = l
            bar2.arrive_and_wait()
            cute.copy(
                ld_q,
                thr_ld_q.partition_S(
                    cute.make_tensor(tmem_ptr + 256 + h * D + wg * (D // 2), o_q_layout)
                ),
                rO0,
            )
            cute.copy(
                ld_q,
                thr_ld_q.partition_S(
                    cute.make_tensor(tmem_ptr + 384 + h * D + wg * (D // 2), o_q_layout)
                ),
                rO1,
            )
            cute.arch.fence_view_async_tmem_load()
            # Both accumulators read: the next segment's first P.V may overwrite them.
            _fence_before()
            cute.arch.mbarrier_arrive(o_free)
            m0 = sml[j % 2, 0, r, 0]
            l0 = sml[j % 2, 0, r, 1]
            m1 = sml[j % 2, 1, r, 0]
            l1 = sml[j % 2, 1, r, 1]
            mx = cutlass.max(m0, m1)
            mx_safe = cutlass.select_(mx == neg_inf, Float32(0.0), mx)
            w0 = cutlass.select_(
                (n0 > 0) & (m0 != neg_inf),
                cute.math.exp2((m0 - mx_safe) * sc, fastmath=True),
                Float32(0.0),
            )
            w1 = cutlass.select_(
                (n1 > 0) & (m1 != neg_inf),
                cute.math.exp2((m1 - mx_safe) * sc, fastmath=True),
                Float32(0.0),
            )
            lsum = w0 * l0 + w1 * l1
            inv = cutlass.select_(lsum > Float32(0.0), rcp_approx(lsum), Float32(0.0))
            a0 = w0 * inv
            a1 = w1 * inv
            res = cute.make_rmem_tensor((D // 2,), Float32)
            for jj in cutlass.range_constexpr(D // 2):
                x0 = cutlass.select_(n0 > 0, rO0[jj], Float32(0.0))
                x1 = cutlass.select_(n1 > 0, rO1[jj], Float32(0.0))
                res[jj] = a0 * x0 + a1 * x1
            slot = cta + b
            if row_valid:
                poff = cute.assume((slot * slot_rows + R) * D + wg * (D // 2), divby=8)
                cute.make_tensor(part_o.iterator + poff, cute.make_layout(D // 2)).store(
                    res.load().to(F16)
                )
                if wg == 0:
                    part_lse[slot * slot_rows + R] = cutlass.select_(
                        lsum > Float32(0.0),
                        mx_safe * sc - Float32(self.pre) + log2_approx(lsum),
                        neg_inf,
                    )
            # The Q of the segment after next goes to this segment's buffer: every MMA of this
            # segment (each QK read that buffer) completed before the o_done waits above.
            if b + 2 < b1:
                self.build_q(q, b + 2, rows, sq_ptr, j % 2, r, h, wg, tok, qh, row_valid)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(q_full + j % 2)
            base0 = base0 + n0
            base1 = base1 + n1
        _fence_before()
