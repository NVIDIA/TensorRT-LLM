# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode MoE engine for BF16 experts routed from the router logits: Renormalize routing and one rank's routed experts
in one persistent launch, returning what ``trtllm::bf16_moe_runner`` returns un-finalized (per (token, k) a gemm2 row
and the BF16 top-k weight; the local picks' rows packed).

Every CTA routes the step's rows in its prologue, then runs FC1 -> SwiGLU -> FC2 with ``kind::f16`` MMAs over the
whole K in order on the BlockMajorK weights in place. Scheduling: ``moe_persistent_engine``.

Parity with ``bf16_moe_runner``: the same picks and gemm2 rows bit for bit; the BF16 top-k weights within 1 ulp. The
prologue's softmax is the fast-math sequence (``ex2.approx``, ``rcp.approx``) while the runner's routing
(RoutingKernel.cuh, built without fast math) uses ``expf`` and an IEEE divide, so a weight can round to the
neighbouring BF16 value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Int64
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineUmmaAsync,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
)

if TYPE_CHECKING:
    import torch

from ...cute_dsl_utils import gmem_ptr
from .moe_decode_front import (
    _below,
    _bit,
    _bitonic_pairs,
    _key1,
    _key_val,
    _ld_v4,
    _max_u32,
    _min_u32,
    _popc,
    _red_or_shared,
    _redux_max_u32,
    _sel_eq,
    _sel_eq_f,
    _sel_lt,
    _st_b16,
    _st_b32,
)
from .moe_persistent_engine import (
    EPI_BAR,
    EPI_GROUPS,
    EPI_WARP0,
    LOG2E,
    NUM_ACC,
    QD,
    THREADS,
    TILE_M,
    PersistentMoeEngine,
    launch,
    new_state,
    ring_smem_bytes,
)
from .ptx_helpers import (
    _add_rn,
    _cp_async_16,
    _cp_async_arrive_noinc,
    _cp_async_commit,
    _ex2,
    _mul_rn,
    _pack_bf16x2_rn,
    _policy_evict_first,
    _policy_evict_last,
    _rcp,
    _st_global_b32,
    _st_global_b32_hint,
    sum_tree,
)

TILE_K = 128  # bf16 elements per ring stage (256 bytes per row)
KB = 64  # BlockMajorK block (128 bytes of bf16)
M_MAX = 32  # rows routed in the prologue (one token bit per row)
N_MAX = (16, 32)  # MMA N (rows per segment; an expert holds at most m rows)

__all__ = [
    "M_MAX",
    "MoeBf16Engine",
    "moe_bf16_engine",
    "new_state",
    "supports_local_experts",
    "supports_routing",
]


# ----------------------------------------------------------------------------------------------------------------
# PTX helpers
# ----------------------------------------------------------------------------------------------------------------
def _sub_rn_ftz(a, b):
    return cute.arch.inline_ptx(
        "sub.rn.ftz.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


def _mul_rn_ftz(a, b):
    return cute.arch.inline_ptx(
        "mul.rn.ftz.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


def _pack_bf16x2_satfinite(lo, hi):
    """{bf16(hi), bf16(lo)} by cvt.rn.satfinite.bf16x2.f32 (lo in the low half)."""
    return cute.arch.inline_ptx(
        "cvt.rn.satfinite.bf16x2.f32 {$w0}, {$r1}, {$r0};",
        write_only_types=[Int32],
        read_only_args=[lo, hi],
    )


def supports_routing(num_experts: int, top_k: int) -> bool:
    """Router shapes the prologue takes: one warp per row, ``num_experts / 32`` logits per lane in whole 16-byte
    loads (8 or 16, bitonic-sorted), and every pick within one lane's sorted keys."""
    per_lane = num_experts // 32
    return num_experts % 32 == 0 and per_lane in (8, 16) and 1 <= top_k <= per_lane


def supports_local_experts(num_experts: int, num_local: int) -> bool:
    """Local expert counts the kernel takes: whole lanes of warp 0's per-expert scan (``segments``)."""
    return num_local % 32 == 0 and 32 <= num_local <= num_experts


def _butterfly_sum(top_k: int):
    """The routing kernels' softmax sum over the top-k values held by lanes 0 .. top_k - 1 (the others hold 0): the
    warp butterfly (xor 16, 8, 4, 2, 1) as lane 0 sees it, as nested pairs of value indices (adding 0 is exact)."""
    lanes = [i if i < top_k else None for i in range(32)]
    for off in (16, 8, 4, 2, 1):
        lanes = [
            lanes[i]
            if lanes[i ^ off] is None
            else (lanes[i ^ off] if lanes[i] is None else (lanes[i], lanes[i ^ off]))
            for i in range(32)
        ]
    return lanes[0]


def smem_views(storage, num_local: int, top_k: int):
    """Tensor views of the engine's shared-memory int32 arrays: nseg, seg_e, seg_r0, seg_n, mask (token bits per
    local expert), eoff (first permuted row per local expert, then the total), pick (local expert of pick
    row * top_k + k, or -1), p2t (permuted row -> token)."""

    def view(mr, n):
        return mr.get_tensor(cute.make_layout(n))

    return (
        view(storage.nseg, 4),
        view(storage.seg_e, M_MAX * top_k),
        view(storage.seg_r0, M_MAX * top_k),
        view(storage.seg_n, M_MAX * top_k),
        view(storage.mask, num_local),
        view(storage.eoff, num_local + 1),
        view(storage.pick, M_MAX * top_k),
        view(storage.p2t, M_MAX * top_k),
    )


# ----------------------------------------------------------------------------------------------------------------
class MoeBf16Engine(PersistentMoeEngine):
    """Compile-time configuration: ``num_experts`` / ``top_k`` (routing, :func:`supports_routing`), ``num_local``
    (local experts), ``n_max`` (MMA N, rows per segment), ``num_stages`` (ring depth). Epilogues: FC1
    ``h = bf16_rn(u (g sigmoid(g)))`` with sigmoid ``rcp.approx(ex2.approx(-g log2e) + 1)``, FC2
    ``bf16_rn_satfinite(acc)``. Every segment holds >= 1 local pick, so a call has at most ``M_MAX * top_k``.

    The activation loader acquires an FC2 unit's segment counter, so the weight loader streams the unit's weights
    while FC1 completes; the dependents are released as each CTA exits."""

    fc2_wait_in_scheduler = False

    def __init__(
        self,
        hidden: int,
        inter: int,
        num_experts: int,
        top_k: int,
        num_local: int,
        n_max: int,
        num_stages: int,
    ):
        assert supports_routing(num_experts, top_k)
        assert supports_local_experts(num_experts, num_local)
        assert n_max in N_MAX
        super().__init__(hidden, inter, n_max, num_stages, TILE_K)
        self.num_experts, self.top_k, self.num_local = num_experts, top_k, num_local
        self.b_stage_bytes = n_max * TILE_K * 2
        self.kchunks = TILE_K * 2 // 16  # 16-byte chunks of one activation row per stage

    def _setup(self):
        self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            (TILE_M, self.n_max),
        )
        total = NUM_ACC * self.n_max
        cols = 32
        while cols < total:
            cols *= 2
        self.num_tmem_cols = cols
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (self.tiled_mma.thr_id.shape,)
        )
        tiler = (TILE_M, self.n_max, self.tile_k)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, tiler, self.a_dtype, self.num_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, tiler, self.b_dtype, self.num_stages
        )

    def _a_atoms(self, w_ptr, rows, k, e):
        """TMA atom over the BlockMajorK weights: element (row r, k kk, expert x) at x rows k + (kk / 64) rows 64 +
        r 64 + kk % 64 (the K mode is (64, K / 64))."""
        w = cute.make_tensor(
            w_ptr, cute.make_layout((rows, (KB, k // KB), e), stride=(KB, (1, rows * KB), rows * k))
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), self.tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        atom, tens = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w,
            a_smem_layout,
            (TILE_M, self.n_max, self.tile_k),
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        self.tma_bytes_a = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        return atom, tens

    @cute.jit
    def __call__(
        self,
        w1_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        b_dummy_ptr: cute.Pointer,
        x_addr: Int64,
        x_row_bytes: Int64,
        lg_addr: Int64,
        m: Int32,
        local_start: Int32,
        wts_addr: Int64,
        e2p_addr: Int64,
        h_addr: Int64,
        out_addr: Int64,
        state_addr: Int64,
        num_local: Int32,
        grid_ctas: Int32,
        stream: cuda.CUstream,
    ):
        e = num_local
        self.a_dtype = w1_ptr.value_type
        self.b_dtype = b_dummy_ptr.value_type
        w_probe = cute.make_tensor(
            w1_ptr, cute.make_ordered_layout((2 * self.inter, self.hidden, e), order=(1, 0, 2))
        )
        b_probe = cute.make_tensor(
            b_dummy_ptr, cute.make_ordered_layout((self.n_max, self.tile_k, 1), order=(1, 0, 2))
        )
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(w_probe).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(b_probe).mma_major_mode()
        self._setup()
        a1, t1 = self._a_atoms(w1_ptr, 2 * self.inter, self.hidden, e)
        a2, t2 = self._a_atoms(w2_ptr, self.hidden, self.inter, e)
        self.kernel(
            self.tiled_mma,
            a1,
            t1,
            a2,
            t2,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            x_addr,
            x_row_bytes,
            lg_addr,
            m,
            local_start,
            wts_addr,
            e2p_addr,
            h_addr,
            out_addr,
            state_addr,
        ).launch(
            grid=[grid_ctas, 1, 1],
            block=[THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        x_addr: Int64,
        x_row_bytes: Int64,
        lg_addr: Int64,
        m: Int32,
        local_start: Int32,
        wts_addr: Int64,
        e2p_addr: Int64,
        h_addr: Int64,
        out_addr: Int64,
        state_addr: Int64,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        Stages = self.num_stages
        NT = THREADS
        NLOC = self.num_local
        SEGS = M_MAX * self.top_k

        @cute.struct
        class SharedStorage:
            ab_bars: cute.struct.MemRange[cutlass.Int64, Stages * 2]
            acc_bars: cute.struct.MemRange[cutlass.Int64, NUM_ACC * 2]
            tmem_bar: cutlass.Int64
            tmem_base: cutlass.Int32
            nseg: cute.struct.MemRange[cutlass.Int32, 4]
            seg_e: cute.struct.MemRange[cutlass.Int32, SEGS]
            seg_r0: cute.struct.MemRange[cutlass.Int32, SEGS]
            seg_n: cute.struct.MemRange[cutlass.Int32, SEGS]
            mask: cute.struct.MemRange[cutlass.Int32, NLOC]
            eoff: cute.struct.MemRange[cutlass.Int32, NLOC + 1]
            pick: cute.struct.MemRange[cutlass.Int32, SEGS]
            p2t: cute.struct.MemRange[cutlass.Int32, SEGS]
            uq: cute.struct.MemRange[cutlass.Int32, QD]
            q_bars: cute.struct.MemRange[cutlass.Int64, 3 * QD]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_nseg, s_seg_e, s_seg_r0, s_seg_n, s_mask, s_eoff, s_pick, s_p2t = smem_views(
            storage, NLOC, self.top_k
        )
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a1)
            cpasync.prefetch_descriptor(tma_a2)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_bars.data_ptr(),
            num_stages=Stages,
            producer_group=CooperativeGroup(Agent.Thread, 1 + 32),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full = ab_pipeline.sync_object_full.barrier_storage
        ab_empty = ab_pipeline.sync_object_empty.barrier_storage
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_bars.data_ptr(),
            num_stages=NUM_ACC,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        acc_full = acc_pipeline.sync_object_full.barrier_storage
        acc_empty = acc_pipeline.sync_object_empty.barrier_storage
        tmem_bar = storage.tmem_bar.ptr
        tmem_base_smem_ptr = storage.tmem_base.ptr
        s_q = storage.uq.get_tensor(cute.make_layout(QD))
        q_full = storage.q_bars.data_ptr()  # full [QD], empty [QD], taken [QD]
        self.init_barriers(warp_idx, tmem_bar, q_full)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        sb_base = sB.iterator.toint()
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)

        # pre-wait: the token masks start empty (shared memory only)
        for i in cutlass.range(tidx, NLOC, NT, unroll=1):
            s_mask[i] = Int32(0)
        gen = self.wait_dependency(state_addr)
        cute.arch.barrier()  # the masks are zero before any warp sets bits in them
        # ---- routing: every CTA routes all m rows (warp w: rows w, w + 12, ...), trtllm-gen Renormalize (same picks,
        # weights within 1 BF16 ulp)
        self.route(lg_addr, m, local_start, wts_addr, s_mask, s_pick, warp_idx, lane, bidx)
        cute.arch.barrier()
        # ---- segments (warp 0): per local expert its rows (token bits), first permuted row, segments of <= n_max
        if warp_idx == 0:
            self.segments(s_mask, s_eoff, s_seg_e, s_seg_r0, s_seg_n, s_nseg, lane)
        cute.arch.barrier()
        # ---- permuted row -> token (every CTA); expanded -> permuted for the fold (CTA 0)
        self.rows(m, s_mask, s_eoff, s_pick, s_p2t, e2p_addr, tidx, bidx)
        cute.arch.barrier()
        nseg = s_nseg[0]
        cnt_base, pool_addr, u1, u2, n1 = self.begin_call(state_addr, gen, nseg, bidx, tidx, gdx)

        if warp_idx == 0:
            self.load_a(
                ab_full,
                ab_empty,
                tma_a1,
                mA1,
                tma_a2,
                mA2,
                sA,
                tiled_mma,
                cluster_layout_vmnk,
                s_seg_e,
                s_q,
                q_full,
                u1,
                bidx,
            )
        elif warp_idx == 1:
            self.load_b(
                ab_full,
                ab_empty,
                s_seg_r0,
                s_seg_n,
                s_p2t,
                sb_base,
                x_addr,
                x_row_bytes,
                h_addr,
                s_q,
                q_full,
                u1,
                lane,
                bidx,
                cnt_base,
            )
        elif warp_idx == 2:
            self.mma(
                ab_full,
                ab_empty,
                acc_full,
                acc_empty,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                sA,
                sB,
                s_q,
                q_full,
                u1,
            )
        elif warp_idx == 3:
            self.scheduler(s_q, q_full, n1, u1, u2, bidx, gdx, pool_addr, cnt_base, lane)
        elif warp_idx >= EPI_WARP0:
            self.epilogue(
                acc_full,
                acc_empty,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                s_seg_r0,
                s_seg_n,
                h_addr,
                out_addr,
                cnt_base,
                (tidx - EPI_WARP0 * 32) % 128,
                (tidx - EPI_WARP0 * 32) // 128,
                s_q,
                q_full,
                u1,
            )
        self.end_call(warp_idx, tidx, gdx, tmem_base_smem_ptr, state_addr, gen)
        return

    # ------------------------------------------------------------------------------------------------------------
    @cute.jit
    def route(self, lg_addr, m, local_start, wts_addr, s_mask, s_pick, warp_idx, lane, bidx):
        """Top-k Renormalize of the BF16 router logits [m, E], MoeDecodeFront's selection: lane l holds experts
        E/32 l .. E/32 (l + 1) - 1, bitonic-sorted per lane, the warp max popped top_k times (keys unique); the
        softmax in fast math (ex2.approx, rcp.approx: within 1 BF16 ulp of the runner's expf / IEEE divide), its sum
        tree the routing kernels' warp butterfly.
        Lane k < top_k records pick k: the token bit in its local expert's mask, its local expert (or -1), and in CTA
        0 its BF16 weight."""
        NLOC = self.num_local
        TOPK = self.top_k
        EPL = self.num_experts // 32  # logits (keys) per lane
        SORT = _bitonic_pairs(EPL)
        mask_addr = s_mask.iterator.toint()
        for row in cutlass.range(warp_idx, m, THREADS // 32, unroll=1):
            src = lg_addr + Int64(row) * Int64(self.num_experts * 2) + Int64(lane) * Int64(EPL * 2)
            ws = []
            for v in cutlass.range_constexpr(EPL // 8):
                ws.extend(_ld_v4(src + Int64(16 * v)))
            e0 = lane * EPL
            keys = cute.make_rmem_tensor(cute.make_layout((EPL,)), Int32)
            for j in cutlass.range_constexpr(EPL // 2):
                keys[2 * j] = _key1(ws[j], e0 + 2 * j)
                keys[2 * j + 1] = _key1(ws[j] >> 16, e0 + 2 * j + 1)
            for pi in cutlass.range_constexpr(len(SORT)):
                i, jj, imax = SORT[pi]
                x = keys[i]
                y = keys[jj]
                hi = _max_u32(x, y)
                lo = _min_u32(x, y)
                if cutlass.const_expr(imax):
                    keys[i] = hi
                    keys[jj] = lo
                else:
                    keys[i] = lo
                    keys[jj] = hi
            sel = cute.make_rmem_tensor(cute.make_layout((TOPK,)), Int32)
            val = cute.make_rmem_tensor(cute.make_layout((TOPK,)), cutlass.Float32)
            for k in cutlass.range_constexpr(TOPK):
                head = keys[0]
                g = _redux_max_u32(head)
                sel[k] = Int32(65535) - (g & Int32(65535))
                val[k] = _key_val(g)
                if cutlass.const_expr(k < TOPK - 1):
                    # pop: the owner lane's sorted keys move up by one
                    for j in cutlass.range_constexpr(TOPK - 1):
                        keys[j] = _sel_eq(head, g, keys[j + 1], keys[j])
            vmax = val[0]
            ex = cute.make_rmem_tensor(cute.make_layout((TOPK,)), cutlass.Float32)
            for k in cutlass.range_constexpr(TOPK):
                # fast-math exp: FADD.FTZ, FMUL.FTZ by log2e, MUFU.EX2 (the runner: expf)
                ex[k] = _ex2(_mul_rn_ftz(_sub_rn_ftz(val[k], vmax), cutlass.Float32(LOG2E)))
            ssum = sum_tree(ex, _butterfly_sum(TOPK))
            my_sel = sel[0]
            my_e = ex[0]
            for k in cutlass.range_constexpr(TOPK - 1):
                my_sel = _sel_eq(lane, Int32(k + 1), sel[k + 1], my_sel)
                my_e = _sel_eq_f(lane, Int32(k + 1), ex[k + 1], my_e)
            if lane < TOPK:
                o = Int64(row) * Int64(TOPK) + Int64(lane)
                if bidx == 0:
                    # MUFU.RCP of the sum, FMUL.FTZ, F2FP.BF16 (round to nearest even); the runner: IEEE divide
                    q = _mul_rn_ftz(my_e, _rcp(ssum))
                    _st_b16(wts_addr + o * Int64(2), _pack_bf16x2_rn(q, q) & Int32(65535))
                lid = my_sel - local_start
                pick = _sel_lt(lid, Int32(NLOC), lid, Int32(-1))
                if pick >= Int32(0):
                    _red_or_shared(mask_addr + pick * 4, _bit(row))
                s_pick[row * TOPK + lane] = pick

    @cute.jit
    def segments(self, s_mask, s_eoff, s_seg_e, s_seg_r0, s_seg_n, s_nseg, lane):
        """Warp 0, num_local / 32 local experts per lane: rows (token bits), first permuted row (warp scan),
        segments of <= n_max rows in expert order."""
        NLOC = self.num_local
        cnt = cute.make_rmem_tensor(cute.make_layout((NLOC // 32,)), Int32)
        nsg = cute.make_rmem_tensor(cute.make_layout((NLOC // 32,)), Int32)
        rows_l = Int32(0)
        segs_l = Int32(0)
        for q in cutlass.range_constexpr(NLOC // 32):
            c = _popc(s_mask[lane * (NLOC // 32) + q])
            ns = (c + (self.n_max - 1)) // self.n_max
            cnt[q] = c
            nsg[q] = ns
            rows_l = rows_l + c
            segs_l = segs_l + ns
        ri = rows_l
        si = segs_l
        for off in (1, 2, 4, 8, 16):
            t1 = cute.arch.shuffle_sync_up(ri, off)
            t2 = cute.arch.shuffle_sync_up(si, off)
            if lane >= off:
                ri = ri + t1
                si = si + t2
        roff = ri - rows_l
        soff = si - segs_l
        for q in cutlass.range_constexpr(NLOC // 32):
            e = lane * (NLOC // 32) + q
            s_eoff[e] = roff
            for j in cutlass.range(nsg[q], unroll=1):
                s_seg_e[soff] = e
                s_seg_r0[soff] = roff + j * self.n_max
                s_seg_n[soff] = cutlass.min(cnt[q] - j * self.n_max, self.n_max)
                soff = soff + 1
            roff = roff + cnt[q]
        if lane == 31:
            s_nseg[0] = si
            s_eoff[NLOC] = ri

    @cute.jit
    def rows(self, m, s_mask, s_eoff, s_pick, s_p2t, e2p_addr, tidx, bidx):
        """Permuted row -> token (an expert's rows in token order) in every CTA; CTA 0 writes the expanded -> permuted
        index of every pick (-1 off this rank)."""
        NLOC = self.num_local
        for e in cutlass.range(tidx, NLOC, THREADS, unroll=1):
            mk = s_mask[e]
            r = s_eoff[e]
            while mk != Int32(0):
                low = mk & (Int32(0) - mk)
                s_p2t[r] = _popc(low - Int32(1))
                r = r + 1
                mk = mk ^ low
        if bidx == 0:
            for i in cutlass.range(tidx, m * self.top_k, THREADS, unroll=1):
                lid = s_pick[i]
                v = Int32(-1)
                if lid >= Int32(0):
                    v = s_eoff[lid] + _popc(s_mask[lid] & _below(i // self.top_k))
                _st_b32(e2p_addr + Int64(i) * Int64(4), v)

    @cute.jit
    def load_a(
        self,
        ab_full,
        ab_empty,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        sA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        s_seg_e: cute.Tensor,
        s_q: cute.Tensor,
        q_full,
        u1,
        bidx,
    ):
        """Weights of every unit, back to back through the ring (constant data)."""
        thr_mma = tiled_mma.get_slice(0)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        g1 = cute.local_tile(mA1, (TILE_M, self.tile_k), (None, None, None))
        t1s, t1g = cpasync.tma_partition(
            tma_a1,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g1), 0, 3),
        )
        g2 = cute.local_tile(mA2, (TILE_M, self.tile_k), (None, None, None))
        t2s, t2g = cpasync.tma_partition(
            tma_a2,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g2), 0, 3),
        )
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        wp = _policy_evict_first()
        k = Int32(0)
        g = self.first_unit(s_q, q_full, u1, bidx)
        self.take(q_full, 0)
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            ex = s_seg_e[seg]
            if is1 != 0:
                for kt in cutlass.range(self.k1_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full + st.index, self.tma_bytes_a
                        )
                    cute.copy(
                        tma_a1,
                        t1g[(None, mt, kt, ex)],
                        t1s[(None, st.index)],
                        tma_bar_ptr=ab_full + st.index,
                        cache_policy=wp,
                    )
                    st.advance()
            else:
                for kt in cutlass.range(self.k2_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full + st.index, self.tma_bytes_a
                        )
                    cute.copy(
                        tma_a2,
                        t2g[(None, mt, kt, ex)],
                        t2s[(None, st.index)],
                        tma_bar_ptr=ab_full + st.index,
                        cache_policy=wp,
                    )
                    st.advance()
            k = k + 1
            g, slot = self.unit(s_q, q_full, k)
            self.take(q_full, slot)

    @cute.jit
    def _b_stage(self, is1, kt, rows, r0, sb_stage, x_addr, x_row_bytes, h_addr, s_p2t, lane):
        """Issue one stage's activation rows (cp.async; 128-byte swizzle: K column j of 64 elements at j n_max 128,
        row n at 1024 (n / 8) + 128 (n % 8), 16-byte chunk c at c ^ (n % 8))."""
        kc = self.kchunks
        rb = (self.n_max // 8) * 1024  # bytes of one 128-byte-wide K column of swizzle atoms
        if is1 != 0:
            for n in cutlass.range(lane, rows, 32, unroll=1):
                t = s_p2t[r0 + n]
                src = x_addr + Int64(t) * x_row_bytes + Int64(kt * (kc * 16))
                dbase = sb_stage + (n >> 3) * 1024 + (n & 7) * 128
                n7 = n & 7
                for c in cutlass.range_constexpr(kc):
                    _cp_async_16(
                        dbase + ((c >> 3) * rb) + (((c & 7) ^ n7) << 4), src + Int64(c * 16)
                    )
        else:
            for n in cutlass.range(lane, rows, 32, unroll=1):
                r = r0 + n
                src = h_addr + Int64(r) * Int64(self.inter * 2) + Int64(kt * (kc * 16))
                dbase = sb_stage + (n >> 3) * 1024 + (n & 7) * 128
                n7 = n & 7
                for c in cutlass.range_constexpr(kc):
                    _cp_async_16(
                        dbase + ((c >> 3) * rb) + (((c & 7) ^ n7) << 4), src + Int64(c * 16)
                    )
        _cp_async_commit()

    @cute.jit
    def load_b(
        self,
        ab_full,
        ab_empty,
        s_seg_r0,
        s_seg_n,
        s_p2t,
        sb_base,
        x_addr,
        x_row_bytes,
        h_addr,
        s_q,
        q_full,
        u1,
        lane,
        bidx,
        cnt_base,
    ):
        """Activation rows of every unit: FC1 gathers x rows by permuted index -> token, FC2 reads the segment's
        intermediate rows after this warp's own acquire of the segment's FC1 counter. Every lane arms an
        asynchronous arrival on the stage's full barrier after issuing its copies (cp.async.mbarrier.arrive.noinc)."""
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        k = Int32(0)
        g = self.first_unit(s_q, q_full, u1, bidx)
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            rows = s_seg_n[seg]
            r0 = s_seg_r0[seg]
            kts = self.k2_tiles
            if is1 != 0:
                kts = self.k1_tiles
            else:
                # every lane acquires the counter, so its own copies below are ordered after the FC1 rows
                self.wait_fc1_done(cnt_base, seg)
            for kt in cutlass.range(kts, unroll=1):
                cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                self._b_stage(
                    is1,
                    kt,
                    rows,
                    r0,
                    sb_base + st.index * self.b_stage_bytes,
                    x_addr,
                    x_row_bytes,
                    h_addr,
                    s_p2t,
                    lane,
                )
                _cp_async_arrive_noinc((ab_full + st.index).toint())
                st.advance()
            k = k + 1
            g, slot = self.unit(s_q, q_full, k)

    @cute.jit
    def mma(
        self,
        ab_full,
        ab_empty,
        acc_full,
        acc_empty,
        tmem_bar,
        tmem_base_smem_ptr,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        s_q: cute.Tensor,
        q_full,
        u1,
    ):
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((TILE_M, self.n_max))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        cute.arch.alloc_tmem(self.num_tmem_cols, tmem_base_smem_ptr)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.relinquish_tmem_alloc_permit()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        num_kphases = cute.size(tCrA, mode=[2])
        ab_st = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)
        acc_st = make_pipeline_state(PipelineUserType.Producer, NUM_ACC)
        k = Int32(0)
        g, slot = self.unit(s_q, q_full, k)
        while g >= 0:
            kts = self.k2_tiles
            if g < u1:
                kts = self.k1_tiles
            cute.arch.mbarrier_wait(acc_empty + acc_st.index, acc_st.phase)
            tCtAcc = cute.make_tensor(tmem_ptr + acc_st.index * self.n_max, tCtAcc_fake.layout)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(kts, unroll=1):
                cute.arch.mbarrier_wait(ab_full + ab_st.index, ab_st.phase)
                for kp in cutlass.range(num_kphases, unroll_full=True):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kp, ab_st.index)],
                        tCrB[(None, None, kp, ab_st.index)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(ab_empty + ab_st.index, None, self.cta_group)
                ab_st.advance()
            with cute.arch.elect_one():
                tcgen05.commit(acc_full + acc_st.index, None, self.cta_group)
            acc_st.advance()
            k = k + 1
            g, slot = self.unit(s_q, q_full, k)

    @cute.jit
    def epilogue(
        self,
        acc_full,
        acc_empty,
        tmem_bar,
        tmem_base_smem_ptr,
        tiled_mma: cute.TiledMma,
        s_seg_r0,
        s_seg_n,
        h_addr,
        out_addr,
        cnt_base,
        epi_tid,
        grp,
        s_q: cute.Tensor,
        q_full,
        u1,
    ):
        tmem_ptr, tCtAcc_fake, tiled_copy_t2r, thr_copy_t2r, rAcc, row0 = self.epilogue_tmem(
            tiled_mma, tmem_bar, tmem_base_smem_ptr, epi_tid
        )
        lane = cute.arch.lane_idx()
        wq = row0 // 32  # epilogue warp = 32-row block of the weight tile
        q = row0 % 32
        hcol = (
            wq * 32 + (q % 8) * 4 + q // 8
        )  # FC2: hidden column of this TMEM lane (32-row block shuffle)
        lead = (
            q // 8
        ) % 2 == 0  # FC2: owns the even column of a pair (partner lane q + 8 has hcol + 1)
        bar_id = Int32(EPI_BAR) + grp
        k = grp
        g, slot = self.unit(s_q, q_full, k)
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            rows = s_seg_n[seg]
            r0 = s_seg_r0[seg]
            buf = k % NUM_ACC
            cute.arch.mbarrier_wait(acc_full + buf, (k // NUM_ACC) & 1)
            nch = (rows + 7) // 8
            if is1 != 0:
                # up lanes q in [0, 8) and [16, 24) (gate partner q ^ 8); lane j < 8 stores the pair of intermediate
                # columns 64 mt + 16 wq + 2 j (its own) and + 1 (lane j + 16's)
                hc = Int64(mt * 64 + wq * 16 + lane * 2)
                for ch in cutlass.range(nch, unroll=1):
                    self.load_acc(
                        tmem_ptr, tCtAcc_fake, tiled_copy_t2r, thr_copy_t2r, rAcc, buf, ch
                    )
                    for t in cutlass.range_constexpr(8):
                        nrow = ch * 8 + t
                        xv = rAcc[t]
                        gate = cute.arch.shuffle_sync_bfly(xv, 8)
                        # u * (g * sigmoid(g)), sigmoid = rcp.approx(ex2.approx(-g log2e) + 1)
                        sgm = _rcp(
                            _add_rn(
                                _ex2(-_mul_rn(gate, cutlass.Float32(LOG2E))),
                                cutlass.Float32(1.0),
                            )
                        )
                        hv = _mul_rn(xv, _mul_rn(gate, sgm))
                        hi = cute.arch.shuffle_sync_down(hv, 16)
                        if lane < 8:
                            if nrow < rows:
                                r = r0 + nrow
                                _st_global_b32(
                                    h_addr + (Int64(r) * Int64(self.inter) + hc) * 2,
                                    _pack_bf16x2_rn(hv, hi),
                                )
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + buf)
                self.free_unit(bar_id, epi_tid, q_full, slot, cnt_base, seg, True)
            else:
                col = Int64(mt * TILE_M + hcol)
                op = _policy_evict_last()
                for ch in cutlass.range(nch, unroll=1):
                    self.load_acc(
                        tmem_ptr, tCtAcc_fake, tiled_copy_t2r, thr_copy_t2r, rAcc, buf, ch
                    )
                    for t in cutlass.range_constexpr(8):
                        partner = cute.arch.shuffle_sync_down(rAcc[t], 8)
                        nrow = ch * 8 + t
                        if lead:
                            if nrow < rows:
                                r = r0 + nrow
                                _st_global_b32_hint(
                                    out_addr + (Int64(r) * Int64(self.hidden) + col) * 2,
                                    _pack_bf16x2_satfinite(rAcc[t], partner),
                                    op,
                                )
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + buf)
                self.free_unit(bar_id, epi_tid, q_full, slot, cnt_base, seg, False)
            k = k + EPI_GROUPS
            g, slot = self.unit(s_q, q_full, k)


# ----------------------------------------------------------------------------------------------------------------
def stages_for(n_max: int, budget: int) -> int:
    """Ring depth: (128 weight rows + n_max activation rows) x TILE_K bf16 per stage within the budget."""
    return min(10, budget // ((128 + n_max) * TILE_K * 2))


def moe_bf16_engine(
    x: torch.Tensor,
    logits: torch.Tensor,
    w31: torch.Tensor,
    w2: torch.Tensor,
    h: torch.Tensor,
    out: torch.Tensor,
    wts: torch.Tensor,
    e2p: torch.Tensor,
    state: torch.Tensor,
    *,
    top_k: int,
    local_start: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route and run the experts. ``x`` [m, H] bf16 (unit column stride), ``logits`` [m, E] bf16 (contiguous),
    m <= M_MAX; ``w31`` [L, H/64, 2I, 64] / ``w2`` [L, I/64, H, 64] bf16 (BlockMajorK, trtllm-gen row shuffle),
    local experts [local_start, local_start + L); ``h`` [>= m top_k, I] bf16 (scratch), ``out`` [>= m top_k, H]
    bf16, ``wts`` [m, top_k] bf16 and ``e2p`` [m, top_k] int32 (outputs), ``state`` from :func:`new_state`.
    Returns (out, wts, e2p): what bf16_moe_runner returns with do_finalize=False."""
    import torch

    e, hb, two_i, kb = w31.shape
    hidden, inter = hb * kb, two_i // 2
    m, num_experts = logits.shape
    assert kb == KB and tuple(w2.shape) == (e, inter // KB, hidden, KB), "BlockMajorK weights"
    assert w31.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    assert 1 <= m <= M_MAX, "m <= M_MAX rows routed in the prologue"
    assert supports_routing(num_experts, top_k) and supports_local_experts(num_experts, e)
    assert (
        x.dim() == 2
        and x.shape == (m, hidden)
        and x.stride(1) == 1
        and (x.stride(0) * 2) % 16 == 0
        and x.data_ptr() % 16 == 0
    )
    assert logits.dtype == torch.bfloat16 and logits.is_contiguous() and logits.data_ptr() % 16 == 0
    assert (
        h.dtype == torch.bfloat16
        and h.shape[0] >= m * top_k
        and h.shape[1] == inter
        and h.is_contiguous()
    )
    assert (
        out.dtype == torch.bfloat16
        and out.shape[0] >= m * top_k
        and out.shape[1] == hidden
        and out.is_contiguous()
    )
    assert wts.dtype == torch.bfloat16 and wts.numel() == m * top_k and wts.is_contiguous()
    assert e2p.dtype == torch.int32 and e2p.numel() == m * top_k and e2p.is_contiguous()
    # one segment per expert: an expert holds at most m rows
    n_max = N_MAX[0] if m <= N_MAX[0] else N_MAX[1]
    num_stages = stages_for(n_max, ring_smem_bytes(x.device))
    args = (
        gmem_ptr(w31, cutlass.BFloat16, 128),
        gmem_ptr(w2, cutlass.BFloat16, 128),
        gmem_ptr(w2, cutlass.BFloat16, 128),
        Int64(x.data_ptr()),
        Int64(x.stride(0) * 2),
        Int64(logits.data_ptr()),
        Int32(m),
        Int32(local_start),
        Int64(wts.data_ptr()),
        Int64(e2p.data_ptr()),
        Int64(h.data_ptr()),
        Int64(out.data_ptr()),
        Int64(state.data_ptr()),
        Int32(e),
    )
    config = (hidden, inter, num_experts, top_k, e, n_max, num_stages)
    launch(lambda: MoeBf16Engine(*config), (MoeBf16Engine, *config), args, x.device)
    return out, wts, e2p
