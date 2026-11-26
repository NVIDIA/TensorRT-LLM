# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode MoE engine for NVFP4 experts: the routed experts of one rank in one persistent launch, FC1 -> SwiGLU ->
NVFP4 -> FC2 on the decode front's pre-routed tables, bit for bit the trtllm-gen FP4 MoE runner's un-finalized gemm2
rows: every MMA is block-scaled tcgen05 M128 x N n_max x K64 over the whole K in order, FC1's epilogue is trtllm-gen's
SwiGLU E2m1 epilogue operation for operation and FC2's is ``moe_lowm_fc2``'s (:func:`moe_nvfp4_engine`). Scheduling:
``moe_persistent_engine``.

``route_top_k`` / ``route_lists`` (compile time): the engine routes the call itself after its dependency wait, from
each token's top ``route_top_k`` keys of each of its ``route_lists`` 128-expert tiles (``moe_router_gemm``'s
``route_top_k`` epilogue), and writes the top-k ids, the BF16 weights and the pre-routed tables bit for bit as the
route front (``moe_route_front``) would.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
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

from ...cute_dsl_utils import gmem_ptr
from .moe_bf16_engine import _butterfly_sum
from .moe_decode_front import (
    _bit,
    _div_bf16,
    _exp_tr,
    _key_val,
    _ld_shared,
    _ld_v4,
    _max_u32,
    _popc,
    _red_or_shared,
    _sel_eq,
    _sel_lt,
    _smem_addr,
    _st_b16,
    _st_shared,
    _sub,
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
    Release,
    launch,
    new_state,
    ring_smem_bytes,
)
from .ptx_helpers import (
    _add_rn,
    _bar_sync,
    _cp_async_16,
    _cp_async_arrive_noinc,
    _cp_async_commit,
    _ex2,
    _ld_nc_f32,
    _ld_s32,
    _mul_rn,
    _policy_evict_first,
    _policy_evict_last,
    _rcp,
    _scale_pack_bf16x2,
    _st_global_b32,
    _st_global_b32_hint,
    s2t_copy,
    sum_tree,
)

SF_VEC = 16
SEG_CAP = 512  # segments per call (experts x row chunks); >= 128 + m * top_k / n_max for every supported m
RCP6 = 0.16666667163372039795  # fp32(1 / 6)
N_MAX = (16, 32, 64)  # MMA N (rows per segment) the engine compiles for
MAX_LOCAL_EXPERTS = (
    128  # per-expert scalars and segment tables are staged for this many local experts
)
# Tokens of a routing call: two token-mask words per expert.
ROUTE_MAX_ROWS = 64
_ROUTE_BAR = EPI_BAR + EPI_GROUPS  # named barrier: the epilogue warps release the activation loader


def route_error(top_k: int, lists: int) -> Optional[str]:
    """Why the engine cannot route from ``lists`` candidate lists of ``top_k`` keys per token, or None."""
    if top_k < 1 or lists < 1:
        return "route_top_k and route_lists must be positive"
    if top_k > 32:
        return "the weights' sum runs over the lanes of one warp: route_top_k <= 32"
    if (lists * top_k) % 4:
        return "a token's candidates load as 16-byte vectors: route_lists x route_top_k must be a multiple of 4"
    if ROUTE_MAX_ROWS * lists * top_k > 8 * THREADS:
        return f"the candidates of {ROUTE_MAX_ROWS} tokens load in two 16-byte vectors per thread"
    return None


def _route_scratch(top_k: int, lists: int) -> tuple:
    """Byte offsets of the routing scratch at the start of the activation ring, after the candidates: per local
    expert its token mask (two words), its first tile, then a sink word for the mask ORs of other ranks' picks."""
    masks = ROUTE_MAX_ROWS * lists * top_k * 4
    starts = masks + MAX_LOCAL_EXPERTS * 8
    return masks, starts, starts + MAX_LOCAL_EXPERTS * 4


__all__ = [
    "MAX_LOCAL_EXPERTS",
    "N_MAX",
    "ROUTE_MAX_ROWS",
    "MoeNvfp4Engine",
    "moe_nvfp4_engine",
    "new_state",
    "route_error",
]


# ----------------------------------------------------------------------------------------------------------------
# PTX helpers
# ----------------------------------------------------------------------------------------------------------------
def _st_global_u8(addr: Int64, v) -> None:
    cute.arch.inline_ptx(
        "{ .reg .b16 t; cvt.u16.u32 t, {$r1}; st.global.u8 [{$r0}], t; }", read_only_args=[addr, v]
    )


def _cp_async_4(dst: Int32, src: Int64) -> None:
    cute.arch.inline_ptx(
        "cp.async.ca.shared.global [{$r0}], [{$r1}], 4;", read_only_args=[dst, src]
    )


def _redux_max_abs_f32(x):
    """max over the warp of |x| (redux.sync.max.abs.f32, sm_100a+); NaN inputs are ignored like fmax."""
    return cute.arch.inline_ptx(
        "redux.sync.max.abs.f32 {$w0}, {$r0}, 0xffffffff;",
        write_only_types=[cutlass.Float32],
        read_only_args=[x],
    )


def _ballot(p):
    return cute.arch.inline_ptx(
        "{ .reg .pred q; setp.ne.s32 q, {$r0}, 0; vote.sync.ballot.b32 {$w0}, q, 0xffffffff; }",
        write_only_types=[Int32],
        read_only_args=[p],
    )


def _fma_rn(a, b, c):
    return cute.arch.inline_ptx(
        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b, c],
    )


def _e4m3_bits(x):
    """e4m3 (nearest even, saturating) of x as 0..255."""
    return cute.arch.inline_ptx(
        "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, 0f00000000, {$r0}; cvt.u32.u16 {$w0}, t; }",
        write_only_types=[Int32],
        read_only_args=[x],
    ) & Int32(0xFF)


def _e4m3_to_f32(b):
    return cute.arch.inline_ptx(
        "{ .reg .b16 t, l, u; .reg .b32 h; cvt.u16.u32 t, {$r0}; cvt.rn.f16x2.e4m3x2 h, t; mov.b32 {l, u}, h;"
        " cvt.f32.f16 {$w0}, l; }",
        write_only_types=[cutlass.Float32],
        read_only_args=[b],
    )


def _e2m1x2(hi, lo):
    """One byte of two E2M1 codes (lo in the low nibble), nearest even, saturating."""
    return cute.arch.inline_ptx(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, {$r0}, {$r1}; cvt.u32.u8 {$w0}, t; }",
        write_only_types=[Int32],
        read_only_args=[hi, lo],
    )


def _sel_f32(p, a, b):
    """p != 0 ? a : b."""
    return cute.arch.inline_ptx(
        "{ .reg .pred q; setp.ne.s32 q, {$r0}, 0; selp.f32 {$w0}, {$r1}, {$r2}, q; }",
        write_only_types=[cutlass.Float32],
        read_only_args=[p, a, b],
    )


def _st_shared_v4(addr, a, b, c, d) -> None:
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};", read_only_args=[addr, a, b, c, d]
    )


def _bar_arrive(bar_id, n: int) -> None:
    """Arrive on named barrier bar_id (a register) over n threads without waiting."""
    cute.arch.inline_ptx(f"bar.arrive {{$r0}}, {n};", read_only_args=[bar_id])


def _fence_gpu() -> None:
    cute.arch.inline_ptx("fence.acq_rel.gpu;")


def _max_tree(keys: list):
    """The largest of ``keys`` (unsigned) as a pairwise tree: max(max(k0, k1), max(k2, k3)) for four."""
    if len(keys) == 1:
        return keys[0]
    half = (len(keys) + 1) // 2
    return _max_u32(_max_tree(keys[:half]), _max_tree(keys[half:]))


def _fence_cta_after(dep) -> None:
    """fence.acq_rel.cta, issued once dep is computed."""
    cute.arch.inline_ptx(
        "{ .reg .b32 d; mov.b32 d, {$r0}; fence.acq_rel.cta; }", read_only_args=[dep]
    )


def smem_views(storage):
    """Tensor views of the engine's shared-memory int32 arrays (dynamic indexing): nseg, wsum, exp_e, exp_y0,
    exp_lim, seg_e, seg_r0, seg_n, tok."""

    def view(mr, n):
        return mr.get_tensor(cute.make_layout(n))

    return (
        view(storage.nseg, 4),
        view(storage.wsum, 12),
        view(storage.exp_e, 128),
        view(storage.exp_y0, 129),
        view(storage.exp_lim, 128),
        view(storage.seg_e, SEG_CAP),
        view(storage.seg_r0, SEG_CAP),
        view(storage.seg_n, SEG_CAP),
        view(storage.tok, 128),
    )


# ----------------------------------------------------------------------------------------------------------------
class MoeNvfp4Engine(PersistentMoeEngine):
    """Compile-time configuration: ``n_max`` (MMA N, rows per segment), ``num_stages`` (ring depth), ``tile`` (the
    routing tables' token tile), ``tile_k`` (K per ring stage), ``release_after_wait`` (PDL release point:
    ``Release.AFTER_WAIT``, else ``Release.POOL_SPENT``), ``route_top_k`` / ``route_lists`` (0: tables given; else the
    engine routes the call, module doc).

    The scheduler queues an FC2 unit once its segment's FC1 counter is complete. Warps: 0 weight TMA, 1 activation
    loader, 2 MMA, 3 unit scheduler, 4-11 two epilogue groups taking alternate units.

    Routing: every CTA merges each token's candidate lists and derives the experts' rows and segments
    (:meth:`_route_prologue`); its epilogue warps then write their CTA's share of the routing outputs and the CTA's
    own permuted index -> token copy, which its activation loader waits for (:meth:`_route_deferred`)."""

    fc2_wait_in_scheduler = True

    def __init__(
        self,
        hidden: int,
        inter: int,
        n_max: int,
        num_stages: int,
        tile: int,
        tile_k: int,
        release_after_wait: bool,
        route_top_k: int = 0,
        route_lists: int = 0,
    ):
        assert tile_k in (256, 512) and n_max in N_MAX
        super().__init__(hidden, inter, n_max, num_stages, tile_k)
        self.release = Release.AFTER_WAIT if release_after_wait else Release.POOL_SPENT
        self.route = route_top_k > 0
        self.route_top_k = route_top_k
        self.route_lists = route_lists
        self.tile = tile
        self.groups = inter // 64  # 4-scale groups per intermediate row (8x4 layout)
        self.sfb_n = 128
        self.b_stage_bytes = n_max * tile_k // 2
        self.sfb_stage_bytes = self.sfb_n * (tile_k // SF_VEC)
        self.kchunks = tile_k // 32  # 16-byte chunks of one activation row per stage
        self.kgroups = tile_k // 64  # 4-scale groups per row per stage
        if self.route:
            err = route_error(route_top_k, route_lists)
            assert err is None, err
            self.route_masks, self.route_starts, self.route_sink = _route_scratch(
                route_top_k, route_lists
            )
            assert num_stages * self.b_stage_bytes >= self.route_sink + 4, "route scratch"

    def _setup(self):
        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            (TILE_M, self.n_max),
        )
        mma_inst_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.tile_k // mma_inst_k
        self.num_sfa_tmem_cols = (TILE_M // 32) * self.mma_inst_tile_k
        self.num_sfb_tmem_cols = self.mma_inst_tile_k * (self.sfb_n // 32)
        total = NUM_ACC * self.n_max + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        assert total <= 512, f"TMEM: {NUM_ACC} x {self.n_max} accumulator columns + scales > 512"
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
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, tiler, SF_VEC, self.num_stages
        )
        self.tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            (TILE_M, self.sfb_n),
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma_sfb, (TILE_M, self.sfb_n, self.tile_k), SF_VEC, self.num_stages
        )

    def _a_atoms(self, w_ptr, wsf_ptr, rows, k, e):
        w = cute.make_tensor(w_ptr, cute.make_ordered_layout((rows, k, e), order=(1, 0, 2)))
        wsf = cute.make_tensor(wsf_ptr, blockscaled_utils.tile_atom_to_shape_SF(w.shape, SF_VEC))
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
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        atom_sf, tens_sf = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            wsf,
            sfa_smem_layout,
            (TILE_M, self.n_max, self.tile_k),
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.tma_bytes_a = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.sf_dtype, sfa_smem_layout
        )
        return atom, tens, atom_sf, tens_sf

    @cute.jit
    def __call__(
        self,
        w1_ptr: cute.Pointer,
        w1sf_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2sf_ptr: cute.Pointer,
        b_dummy_ptr: cute.Pointer,
        xq_addr: Int64,
        xsf_addr: Int64,
        p2t_addr: Int64,
        batch_addr: Int64,
        limit_addr: Int64,
        nctas_addr: Int64,
        max_ctas: Int32,
        scale_c_addr: Int64,
        alpha1_addr: Int64,
        alpha2_addr: Int64,
        h_addr: Int64,
        hsf_addr: Int64,
        out_addr: Int64,
        state_addr: Int64,
        num_experts: Int32,
        grid_ctas: Int32,
        stream: cuda.CUstream,
        cand_addr: Optional[Int64] = None,
        ids_addr: Optional[Int64] = None,
        wts_addr: Optional[Int64] = None,
        e2p_addr: Optional[Int64] = None,
        npad_addr: Optional[Int64] = None,
        p2tp_addr: Optional[Int64] = None,
        m_rows: Optional[Int32] = None,
        local_start: Optional[Int32] = None,
    ):
        # route only: the candidates, the routing outputs (ids, weights, e2p, padded row count), the CTAs' permuted
        # index -> token copies [grid, p2t rows], the call's tokens and the rank's first expert
        e = num_experts
        self.a_dtype = w1_ptr.value_type
        self.b_dtype = b_dummy_ptr.value_type
        self.sf_dtype = w1sf_ptr.value_type
        w_probe = cute.make_tensor(
            w1_ptr, cute.make_ordered_layout((2 * self.inter, self.hidden, e), order=(1, 0, 2))
        )
        b_probe = cute.make_tensor(
            b_dummy_ptr, cute.make_ordered_layout((self.n_max, self.tile_k, 1), order=(1, 0, 2))
        )
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(w_probe).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(b_probe).mma_major_mode()
        self._setup()
        a1, t1, a1s, t1s = self._a_atoms(w1_ptr, w1sf_ptr, 2 * self.inter, self.hidden, e)
        a2, t2, a2s, t2s = self._a_atoms(w2_ptr, w2sf_ptr, self.hidden, self.inter, e)
        self.kernel(
            self.tiled_mma,
            a1,
            t1,
            a1s,
            t1s,
            a2,
            t2,
            a2s,
            t2s,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            xq_addr,
            xsf_addr,
            p2t_addr,
            batch_addr,
            limit_addr,
            nctas_addr,
            max_ctas,
            scale_c_addr,
            alpha1_addr,
            alpha2_addr,
            h_addr,
            hsf_addr,
            out_addr,
            state_addr,
            num_experts,
            cand_addr,
            ids_addr,
            wts_addr,
            e2p_addr,
            npad_addr,
            p2tp_addr,
            m_rows,
            local_start,
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
        tma_a1s: cute.CopyAtom,
        mA1s: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_a2s: cute.CopyAtom,
        mA2s: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        xq_addr: Int64,
        xsf_addr: Int64,
        p2t_addr: Int64,
        batch_addr: Int64,
        limit_addr: Int64,
        nctas_addr: Int64,
        max_ctas: Int32,
        scale_c_addr: Int64,
        alpha1_addr: Int64,
        alpha2_addr: Int64,
        h_addr: Int64,
        hsf_addr: Int64,
        out_addr: Int64,
        state_addr: Int64,
        n_exp: Int32,
        cand_addr: Optional[Int64] = None,
        ids_addr: Optional[Int64] = None,
        wts_addr: Optional[Int64] = None,
        e2p_addr: Optional[Int64] = None,
        npad_addr: Optional[Int64] = None,
        p2tp_addr: Optional[Int64] = None,
        m_rows: Optional[Int32] = None,
        local_start: Optional[Int32] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        Stages = self.num_stages
        NT = THREADS

        @cute.struct
        class SharedStorage:
            ab_bars: cute.struct.MemRange[cutlass.Int64, Stages * 2]
            acc_bars: cute.struct.MemRange[cutlass.Int64, NUM_ACC * 2]
            tmem_bar: cutlass.Int64
            tmem_base: cutlass.Int32
            nseg: cute.struct.MemRange[cutlass.Int32, 4]
            wsum: cute.struct.MemRange[cutlass.Int32, 12]
            wsplit: cute.struct.MemRange[cutlass.Int32, 12]
            exp_e: cute.struct.MemRange[cutlass.Int32, 128]
            exp_y0: cute.struct.MemRange[cutlass.Int32, 129]
            exp_lim: cute.struct.MemRange[cutlass.Int32, 128]
            seg_e: cute.struct.MemRange[cutlass.Int32, SEG_CAP]
            seg_r0: cute.struct.MemRange[cutlass.Int32, SEG_CAP]
            seg_n: cute.struct.MemRange[cutlass.Int32, SEG_CAP]
            tok: cute.struct.MemRange[cutlass.Int32, 128]
            uq: cute.struct.MemRange[cutlass.Int32, QD]
            q_bars: cute.struct.MemRange[cutlass.Int64, 3 * QD]
            ex_a1: cute.struct.MemRange[cutlass.Float32, 128]
            ex_c: cute.struct.MemRange[cutlass.Float32, 128]
            ex_a2: cute.struct.MemRange[cutlass.Float32, 128]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_nseg, s_wsum, s_exp_e, s_exp_y0, s_exp_lim, s_seg_e, s_seg_r0, s_seg_n, s_tok = (
            smem_views(storage)
        )
        s_wsplit = storage.wsplit.get_tensor(cute.make_layout(12))
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a1)
            cpasync.prefetch_descriptor(tma_a1s)
            cpasync.prefetch_descriptor(tma_a2)
            cpasync.prefetch_descriptor(tma_a2s)
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
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype, layout=sfa_smem_layout_staged, byte_alignment=128
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype, layout=sfb_smem_layout_staged, byte_alignment=128
        )
        if cutlass.const_expr(self.route):
            # each token's merged keys, descending
            s_pick = smem.allocate_tensor(
                Int32, cute.make_layout(ROUTE_MAX_ROWS * self.route_top_k), byte_alignment=16
            )
        sb_base = sB.iterator.toint()
        sfb_base = sSFB.iterator.toint()
        route_done = None
        if cutlass.const_expr(self.route):
            # Completes once every epilogue thread wrote its routing outputs, which the MoE-output all-reduce reads
            # before (Lamport FC2) or without its dependency wait: the scheduler releases the dependents after it.
            route_done = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout(1), byte_alignment=8
            ).iterator
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(route_done, 128 * EPI_GROUPS)
                cute.arch.mbarrier_init_fence()
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)

        s_a1 = storage.ex_a1.get_tensor(cute.make_layout(128))
        s_c = storage.ex_c.get_tensor(cute.make_layout(128))
        s_a2 = storage.ex_a2.get_tensor(cute.make_layout(128))
        if tidx >= 128:
            if tidx < 256:
                ei = tidx - 128
                if ei < n_exp:
                    # weights' constants: not written by the previous kernel, so read before the dependency wait
                    s_a1[ei] = _ld_nc_f32(alpha1_addr + Int64(ei) * 4)
                    s_c[ei] = _ld_nc_f32(scale_c_addr + Int64(ei) * 4)
                    s_a2[ei] = _ld_nc_f32(alpha2_addr + Int64(ei) * 4)
        # gen is loaded here and used after the table scan, so its latency overlaps the scan's loads
        gen = self.wait_dependency(state_addr, route_done)

        # tiles per n_max rows: an expert splits iff the tile RT after its first is still its own
        RT = (
            self.n_max // self.tile
            if self.n_max % self.tile == 0 and self.n_max >= self.tile
            else 0
        )
        if cutlass.const_expr(self.route):
            nexp, anysplit = self._route_prologue(
                cand_addr,
                m_rows,
                local_start,
                n_exp,
                s_exp_e,
                s_exp_y0,
                s_exp_lim,
                s_wsum,
                s_pick,
                tidx,
                lane,
                sb_base,
            )
        else:
            # ---- expert segments from the tile tables (whole CTA). Per 256 tiles one round of independent loads: the
            # tile's expert, its neighbours' and its row limit; a tile starts an expert if its expert differs from the
            # previous tile's and ends one if it differs from the next tile's (the end tile's limit closes the expert).
            n_tiles = _ld_s32(nctas_addr)
            warp_in = tidx // 32
            lane_lt = (Int32(1) << lane) - Int32(1)
            nexp = Int32(0)
            base = Int32(0)
            more = Int32(1)
            anysplit = Int32(0)  # some expert has more than n_max rows
            while more != 0:
                y = base + tidx
                bc = Int32(-1)
                bp = Int32(-1)
                bn = Int32(-1)
                br = Int32(-2)
                lim = Int32(0)
                if y < max_ctas:
                    bc = _ld_s32(batch_addr + Int64(y) * 4)
                    lim = _ld_s32(limit_addr + Int64(y) * 4)
                    if y > 0:
                        bp = _ld_s32(batch_addr + Int64(y - 1) * 4)
                    if y + 1 < max_ctas:
                        bn = _ld_s32(batch_addr + Int64(y + 1) * 4)
                    if cutlass.const_expr(RT > 0):
                        if y + RT < max_ctas:
                            br = _ld_s32(batch_addr + Int64(y + RT) * 4)
                nt = cutlass.min(n_tiles, max_ctas)
                start = Int32(0)
                end = Int32(0)
                spl = Int32(0)
                if y < nt:
                    if bc != bp:
                        start = Int32(1)
                        if cutlass.const_expr(RT > 0):
                            if y + RT < nt:
                                if br == bc:
                                    spl = Int32(1)
                    if bn != bc:
                        end = Int32(1)
                    if y + 1 == nt:
                        end = Int32(1)
                bal = _ballot(start)
                sbal = Int32(0)
                if cutlass.const_expr(RT > 0):
                    sbal = _ballot(spl)
                pre = _popc(bal & lane_lt)
                if lane == 0:
                    s_wsum[warp_in] = _popc(bal)
                    if cutlass.const_expr(RT > 0):
                        s_wsplit[warp_in] = sbal
                cute.arch.barrier()
                off = Int32(0)
                tot = Int32(0)
                for w in cutlass.range_constexpr(NT // 32):
                    v = s_wsum[w]
                    if w < warp_in:
                        off = off + v
                    tot = tot + v
                    if cutlass.const_expr(RT > 0):
                        anysplit = anysplit | s_wsplit[w]
                incl = nexp + off + pre + start  # experts started at or before tile y
                if start != 0:
                    s_exp_e[incl - 1] = bc
                    s_exp_y0[incl - 1] = y
                if end != 0:
                    s_exp_lim[incl - 1] = lim
                nexp = nexp + tot
                base = base + NT
                more = Int32(0)
                if base < nt:
                    more = Int32(1)
                cute.arch.barrier()
        # split experts into segments of <= n_max rows: with no expert above n_max rows the segments are the experts,
        # written by every thread in one step; else warp 0 (4 experts per lane, warp prefix of the chunk counts)
        slow = Int32(1)
        if cutlass.const_expr(RT > 0):
            if anysplit == 0:
                slow = Int32(0)
                for xi in cutlass.range(tidx, nexp, NT, unroll=1):
                    r0f = s_exp_y0[xi] * self.tile
                    s_seg_e[xi] = s_exp_e[xi]
                    s_seg_r0[xi] = r0f
                    s_seg_n[xi] = s_exp_lim[xi] - r0f
                if tidx == 0:
                    s_nseg[0] = nexp
        if warp_idx + Int32(32) * (Int32(1) - slow) == 0:
            cnts = cute.make_rmem_tensor((4,), Int32)
            mysum = Int32(0)
            for qi in cutlass.range_constexpr(4):
                xi = lane * 4 + qi
                c = Int32(0)
                if xi < nexp:
                    c = (s_exp_lim[xi] - s_exp_y0[xi] * self.tile + (self.n_max - 1)) // self.n_max
                cnts[qi] = c
                mysum = mysum + c
            incl = mysum
            for off2 in (1, 2, 4, 8, 16):
                t = cute.arch.shuffle_sync_up(incl, off2)
                if lane >= off2:
                    incl = incl + t
            pos = incl - mysum
            for qi in cutlass.range_constexpr(4):
                xi = lane * 4 + qi
                if xi < nexp:
                    r0 = s_exp_y0[xi] * self.tile
                    rows = s_exp_lim[xi] - r0
                    for ch in cutlass.range(cnts[qi], unroll=1):
                        s_seg_e[pos] = s_exp_e[xi]
                        s_seg_r0[pos] = r0 + ch * self.n_max
                        s_seg_n[pos] = cutlass.min(rows - ch * self.n_max, self.n_max)
                        pos = pos + 1
            if lane == 31:
                s_nseg[0] = incl
        cute.arch.barrier()
        nseg = s_nseg[0]
        cnt_base, pool_addr, u1, u2, n1 = self.begin_call(state_addr, gen, nseg, bidx, tidx, gdx)
        p2t_ld = p2t_addr
        if cutlass.const_expr(self.route):
            # the activation loader reads this CTA's own copy, which this CTA's epilogue warps write
            p2t_rows = (max_ctas + Int32(1)) * Int32(self.tile)
            p2t_ld = p2tp_addr + Int64(bidx) * Int64(p2t_rows) * Int64(4)

        if warp_idx == 0:
            self.load_a(
                ab_full,
                ab_empty,
                tma_a1,
                mA1,
                tma_a1s,
                mA1s,
                tma_a2,
                mA2,
                tma_a2s,
                mA2s,
                sA,
                sSFA,
                tiled_mma,
                cluster_layout_vmnk,
                s_seg_e,
                s_q,
                q_full,
                u1,
                bidx,
            )
        elif warp_idx == 1:
            if cutlass.const_expr(self.route):
                _bar_sync(Int32(_ROUTE_BAR), 32 + 128 * EPI_GROUPS)
            self.load_b(
                ab_full,
                ab_empty,
                s_seg_r0,
                s_seg_n,
                s_tok,
                sb_base,
                sfb_base,
                xq_addr,
                xsf_addr,
                p2t_ld,
                h_addr,
                hsf_addr,
                s_q,
                q_full,
                u1,
                lane,
                bidx,
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
                sSFA,
                sSFB,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                s_q,
                q_full,
                u1,
            )
        elif warp_idx == 3:
            self.scheduler(
                s_q, q_full, n1, u1, u2, bidx, gdx, pool_addr, cnt_base, lane, route_done
            )
        elif warp_idx >= EPI_WARP0:
            if cutlass.const_expr(self.route):
                self._route_deferred(
                    s_pick,
                    s_wsum,
                    sb_base,
                    ids_addr,
                    wts_addr,
                    e2p_addr,
                    p2t_addr,
                    p2t_ld,
                    batch_addr,
                    limit_addr,
                    nctas_addr,
                    npad_addr,
                    m_rows,
                    local_start,
                    n_exp,
                    tidx - Int32(EPI_WARP0 * 32),
                    bidx,
                    gdx,
                )
                _fence_gpu()
                cute.arch.mbarrier_arrive(route_done)
            self.epilogue(
                acc_full,
                acc_empty,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                s_seg_e,
                s_seg_r0,
                s_seg_n,
                s_a1,
                s_c,
                s_a2,
                h_addr,
                hsf_addr,
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
    def _route_prologue(
        self,
        cand_addr: Int64,
        m_rows: Int32,
        local_start: Int32,
        n_exp: Int32,
        s_exp_e,
        s_exp_y0,
        s_exp_lim,
        s_wsum,
        s_pick,
        tidx,
        lane,
        sb_base,
    ):
        """Routing, whole CTA, after the dependency wait: each token's top ``route_top_k`` keys merged from its
        ``route_lists`` descending lists (s_pick), every local expert's token mask and first tile (sB scratch), and the
        experts with rows in exp_e / exp_y0 / exp_lim as the tile-table scan leaves them. Returns (experts with rows,
        some expert above n_max rows); s_wsum[0] holds the call's tiles."""
        NT: cutlass.Constexpr = THREADS
        TK: cutlass.Constexpr = self.route_top_k
        NL: cutlass.Constexpr = self.route_lists
        tile: cutlass.Constexpr = self.tile
        cbase = Int32(sb_base)
        mask_base = cbase + Int32(self.route_masks)
        start_base = cbase + Int32(self.route_starts)
        sink = cbase + Int32(self.route_sink)
        key_base = _smem_addr(s_pick.iterator)
        warp_in = tidx // Int32(32)
        # one round of loads: the m x NL x TK keys as 16-byte vectors (at most two per thread), both issued before
        # the first store, so the copy pays one L2 round trip; the masks are zeroed meanwhile
        nv = m_rows * Int32(NL * TK // 4)
        i0 = tidx
        i1 = tidx + Int32(NT)
        a0 = Int32(0)
        b0 = Int32(0)
        c0 = Int32(0)
        d0 = Int32(0)
        a1 = Int32(0)
        b1 = Int32(0)
        c1 = Int32(0)
        d1 = Int32(0)
        if i0 < nv:
            a0, b0, c0, d0 = _ld_v4(cand_addr + Int64(i0) * Int64(16))
        if i1 < nv:
            a1, b1, c1, d1 = _ld_v4(cand_addr + Int64(i1) * Int64(16))
        for i in cutlass.range(tidx, 2 * MAX_LOCAL_EXPERTS, NT, unroll=1):
            _st_shared(mask_base + i * Int32(4), Int32(0))
        if i0 < nv:
            _st_shared_v4(cbase + i0 * Int32(16), a0, b0, c0, d0)
        if i1 < nv:
            _st_shared_v4(cbase + i1 * Int32(16), a1, b1, c1, d1)
        cute.arch.barrier()
        # one thread per token merges its NL lists: each keeps its head and next key in registers; the list whose
        # head is picked advances. A list whose keys are all used reads past its end, a value that could only become
        # a head after pick TK + 1. Picks stay in registers until their stores and mask ORs after the loop.
        t = tidx
        if t < m_rows:
            p = [cbase + t * Int32(NL * TK * 4)]
            for j in cutlass.range_constexpr(1, NL):
                p.append(p[0] + Int32(j * TK * 4))
            h = [_ld_shared(p[j]) for j in range(NL)]
            n = [_ld_shared(p[j] + Int32(4)) for j in range(NL)]
            gk = []
            for k in cutlass.range_constexpr(TK):
                g = _max_tree(h)
                gk.append(g)
                if cutlass.const_expr(k < TK - 1):
                    p = [_sel_eq(h[j], g, p[j] + Int32(4), p[j]) for j in range(NL)]
                    h = [_sel_eq(h[j], g, n[j], h[j]) for j in range(NL)]
                    n = [_ld_shared(p[j] + Int32(4)) for j in range(NL)]
            kb = key_base + t * Int32(TK * 4)
            mrow = mask_base + (t // Int32(32)) * Int32(4)
            mbit = _bit(t % Int32(32))
            for k in cutlass.range_constexpr(TK):
                _st_shared(kb + Int32(4 * k), gk[k])
            for k in cutlass.range_constexpr(TK):
                lid = Int32(65535) - (gk[k] & Int32(65535)) - local_start
                _red_or_shared(_sel_lt(lid, n_exp, mrow + lid * Int32(8), sink), mbit)
        cute.arch.barrier()
        # warp 0, local experts 4 lane .. 4 lane + 3: rows, tiles, first tile, and the experts with rows in order, from
        # one warp scan of (tiles << 16 | experts with rows)
        if warp_in == Int32(0):
            cn = cute.make_rmem_tensor(cute.make_layout((4,)), Int32)
            pk = Int32(0)
            bigl = Int32(0)
            for qi in cutlass.range_constexpr(4):
                e = lane * Int32(4) + Int32(qi)
                c = Int32(0)
                if e < n_exp:
                    c = _popc(_ld_shared(mask_base + e * Int32(8))) + _popc(
                        _ld_shared(mask_base + e * Int32(8) + Int32(4))
                    )
                cn[qi] = c
                pk = (
                    pk
                    + (((c + Int32(tile - 1)) // Int32(tile)) << Int32(16))
                    + _sel_lt(Int32(0), c, Int32(1), Int32(0))
                )
                bigl = bigl | _sel_lt(Int32(self.n_max), c, Int32(1), Int32(0))
            incl = pk
            for off2 in (1, 2, 4, 8, 16):
                u = cute.arch.shuffle_sync_up(incl, off2)
                if lane >= off2:
                    incl = incl + u
            exl = incl - pk
            toff = exl >> Int32(16)
            aoff = exl & Int32(65535)
            for qi in cutlass.range_constexpr(4):
                e = lane * Int32(4) + Int32(qi)
                c = cn[qi]
                if e < n_exp:
                    _st_shared(start_base + e * Int32(4), toff)
                if c > Int32(0):
                    s_exp_e[aoff] = e
                    s_exp_y0[aoff] = toff
                    s_exp_lim[aoff] = toff * Int32(tile) + c
                aoff = aoff + _sel_lt(Int32(0), c, Int32(1), Int32(0))
                toff = toff + (c + Int32(tile - 1)) // Int32(tile)
            bbal = _ballot(bigl)
            if lane == Int32(31):
                s_wsum[0] = incl >> Int32(16)  # tiles
                s_wsum[1] = incl & Int32(65535)  # experts with rows
                s_wsum[2] = bbal
        cute.arch.barrier()
        return s_wsum[1], s_wsum[2]

    @cute.jit
    def _route_deferred(
        self,
        s_pick,
        s_wsum,
        sb_base,
        ids_addr: Int64,
        wts_addr: Int64,
        e2p_addr: Int64,
        p2t_addr: Int64,
        p2t_cta: Int64,
        batch_addr: Int64,
        limit_addr: Int64,
        nctas_addr: Int64,
        npad_addr: Int64,
        m_rows: Int32,
        local_start: Int32,
        n_exp: Int32,
        q,
        bidx,
        gdx,
    ):
        """Routing, epilogue thread q of 128 x EPI_GROUPS, before its first accumulator:
        1. each local pick (expanded index token * TK + k) into this CTA's p2t copy; the token's owner CTA (token %
           grid) also writes the call's p2t and e2p (-1: another rank's expert), one writer per word;
        2. release the activation loader (the copy's only reader) once the masks and first tiles are in registers;
        3. the tile tables (the expert's owner CTA) and their counts (CTA 0);
        4. ids and BF16 weights of the tokens this CTA owns, with the route front's arithmetic."""
        TK: cutlass.Constexpr = self.route_top_k
        NE: cutlass.Constexpr = 128 * EPI_GROUPS
        tile: cutlass.Constexpr = self.tile
        key_base = _smem_addr(s_pick.iterator)
        mask_base = Int32(sb_base) + Int32(self.route_masks)
        start_base = Int32(sb_base) + Int32(self.route_starts)
        for x in cutlass.range(q, m_rows * Int32(TK), NE, unroll=1):
            tt = x // Int32(TK)
            key = _ld_shared(key_base + x * Int32(4))
            lid = Int32(65535) - (key & Int32(65535)) - local_start
            own = Int32(0)
            if tt % gdx == bidx:
                own = Int32(1)
            row = Int32(-1)
            if _sel_lt(lid, n_exp, Int32(1), Int32(0)) != Int32(0):
                lo = _ld_shared(mask_base + lid * Int32(8))
                rank = Int32(0)
                if tt < Int32(32):
                    rank = _popc(lo & ((Int32(1) << tt) - Int32(1)))
                else:
                    hi = _ld_shared(mask_base + lid * Int32(8) + Int32(4))
                    rank = _popc(lo) + _popc(hi & ((Int32(1) << (tt - Int32(32))) - Int32(1)))
                row = _ld_shared(start_base + lid * Int32(4)) * Int32(tile) + rank
                _st_global_b32(p2t_cta + Int64(row) * Int64(4), tt)
                if own != Int32(0):
                    _st_global_b32(p2t_addr + Int64(row) * Int64(4), tt)
            if own != Int32(0):
                _st_global_b32(e2p_addr + Int64(x) * Int64(4), row)
        e = q
        cnt = Int32(0)
        st0 = Int32(0)
        if e < n_exp:
            cnt = _popc(_ld_shared(mask_base + e * Int32(8))) + _popc(
                _ld_shared(mask_base + e * Int32(8) + Int32(4))
            )
            st0 = _ld_shared(start_base + e * Int32(4))
        _fence_cta_after(cnt)
        _bar_arrive(Int32(_ROUTE_BAR), 32 + NE)
        if cnt > Int32(0):
            if e % gdx == bidx:
                for c in cutlass.range((cnt + Int32(tile - 1)) // Int32(tile), unroll=1):
                    _st_global_b32(batch_addr + Int64(st0 + c) * Int64(4), e)
                    lim1 = (st0 + c + Int32(1)) * Int32(tile)
                    lim2 = st0 * Int32(tile) + cnt
                    _st_global_b32(limit_addr + Int64(st0 + c) * Int64(4), cutlass.min(lim1, lim2))
        if bidx == Int32(0):
            if q == Int32(0):
                ntot = s_wsum[0]
                _st_global_b32(nctas_addr, ntot)
                _st_global_b32(npad_addr, ntot * Int32(tile))
        t = bidx + q * gdx
        if t < m_rows:
            kb = key_base + t * Int32(TK * 4)
            val = []
            for k in cutlass.range_constexpr(TK):
                g = _ld_shared(kb + Int32(4 * k))
                o = Int64(t) * Int64(TK) + Int64(k)
                _st_global_b32(ids_addr + o * Int64(4), Int32(65535) - (g & Int32(65535)))
                val.append(_key_val(g))
            ex = []
            for k in cutlass.range_constexpr(TK):
                ex.append(_exp_tr(_sub(val[k], val[0])))
            ssum = sum_tree(ex, _butterfly_sum(TK))
            for k in cutlass.range_constexpr(TK):
                o = Int64(t) * Int64(TK) + Int64(k)
                _st_b16(wts_addr + o * Int64(2), _div_bf16(ex[k], ssum))

    @cute.jit
    def load_a(
        self,
        ab_full,
        ab_empty,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_a1s: cute.CopyAtom,
        mA1s: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_a2s: cute.CopyAtom,
        mA2s: cute.Tensor,
        sA: cute.Tensor,
        sSFA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        s_seg_e: cute.Tensor,
        s_q: cute.Tensor,
        q_full,
        u1,
        bidx,
    ):
        """Weights + scales of every unit, back to back through the ring (constant data)."""
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
        g1s = cute.local_tile(mA1s, (TILE_M, self.tile_k), (None, None, None))
        t1ss, t1sg = cpasync.tma_partition(
            tma_a1s,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g1s), 0, 3),
        )
        t1ss = cute.filter_zeros(t1ss)
        t1sg = cute.filter_zeros(t1sg)
        g2 = cute.local_tile(mA2, (TILE_M, self.tile_k), (None, None, None))
        t2s, t2g = cpasync.tma_partition(
            tma_a2,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g2), 0, 3),
        )
        g2s = cute.local_tile(mA2s, (TILE_M, self.tile_k), (None, None, None))
        t2ss, t2sg = cpasync.tma_partition(
            tma_a2s,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(g2s), 0, 3),
        )
        t2ss = cute.filter_zeros(t2ss)
        t2sg = cute.filter_zeros(t2sg)
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
                    cute.copy(
                        tma_a1s,
                        t1sg[(None, mt, kt, ex)],
                        t1ss[(None, st.index)],
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
                    cute.copy(
                        tma_a2s,
                        t2sg[(None, mt, kt, ex)],
                        t2ss[(None, st.index)],
                        tma_bar_ptr=ab_full + st.index,
                        cache_policy=wp,
                    )
                    st.advance()
            k = k + 1
            g, slot = self.unit(s_q, q_full, k)
            self.take(q_full, slot)

    @cute.jit
    def _b_stage(
        self,
        is1,
        kt,
        rows,
        r0,
        sb_stage,
        sfb_stage,
        xq_addr,
        xsf_addr,
        h_addr,
        hsf_addr,
        s_tok,
        lane,
    ):
        """Issue one stage's activation rows (cp.async, 128-byte swizzle: row n at 128 n, chunk c at c ^ (n % 8))
        and their scale words into the stage's 128x4 SFB atom (word g of row n at 512 g + 16 (n % 32) + 4 (n / 32))."""
        kc = self.kchunks
        kg = self.kgroups
        rb = (self.n_max // 8) * 1024  # bytes of one 128-byte-wide K column of swizzle atoms
        if is1 != 0:
            for n in cutlass.range(lane, rows, 32, unroll=1):
                t = s_tok[n]
                src = xq_addr + Int64(t) * Int64(self.hidden // 2) + Int64(kt * (kc * 16))
                ssrc = xsf_addr + Int64(t) * Int64(self.hidden // SF_VEC) + Int64(kt * (kg * 4))
                dbase = sb_stage + (n >> 3) * 1024 + (n & 7) * 128
                n7 = n & 7
                for c in cutlass.range_constexpr(kc):
                    _cp_async_16(
                        dbase + ((c >> 3) * rb) + (((c & 7) ^ n7) << 4), src + Int64(c * 16)
                    )
                sbase = sfb_stage + (n & 31) * 16 + (n >> 5) * 4
                for g in cutlass.range_constexpr(kg):
                    _cp_async_4(sbase + g * 512, ssrc + Int64(g * 4))
        else:
            # the segment's intermediate scales: final once its counter read 8 (lines of one 8-row tile belong to one
            # segment, so this SM has not cached them earlier in the call; L1 starts each kernel clean)
            for n in cutlass.range(lane, rows, 32, unroll=1):
                r = r0 + n
                src = h_addr + Int64(r) * Int64(self.inter // 2) + Int64(kt * (kc * 16))
                ssrc = (
                    hsf_addr
                    + (Int64(r >> 3) * Int64(self.groups) + Int64(kt * kg)) * 32
                    + Int64(r & 7) * 4
                )
                dbase = sb_stage + (n >> 3) * 1024 + (n & 7) * 128
                n7 = n & 7
                for c in cutlass.range_constexpr(kc):
                    _cp_async_16(
                        dbase + ((c >> 3) * rb) + (((c & 7) ^ n7) << 4), src + Int64(c * 16)
                    )
                sbase = sfb_stage + (n & 31) * 16 + (n >> 5) * 4
                for g in cutlass.range_constexpr(kg):
                    _cp_async_4(sbase + g * 512, ssrc + Int64(g * 32))
        _cp_async_commit()

    @cute.jit
    def load_b(
        self,
        ab_full,
        ab_empty,
        s_seg_r0,
        s_seg_n,
        s_tok,
        sb_base,
        sfb_base,
        xq_addr,
        xsf_addr,
        p2t_addr,
        h_addr,
        hsf_addr,
        s_q,
        q_full,
        u1,
        lane,
        bidx,
    ):
        """Activation rows of every unit: FC1 gathers x rows by permuted index -> token, FC2 reads the segment's
        intermediate rows (ready: the scheduler acquired its counter before queueing the unit). Every lane arms an
        asynchronous arrival on the stage's full barrier after issuing its copies (cp.async.mbarrier.arrive.noinc),
        so the loader never waits for L2 and keeps the whole ring in flight."""
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
                # token of each row (read only while issuing: in-flight copies already hold their addresses)
                cute.arch.sync_warp()
                for n in cutlass.range(lane, rows, 32, unroll=1):
                    s_tok[n] = _ld_s32(p2t_addr + Int64(r0 + n) * 4)
                cute.arch.sync_warp()
            for kt in cutlass.range(kts, unroll=1):
                cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                self._b_stage(
                    is1,
                    kt,
                    rows,
                    r0,
                    sb_base + st.index * self.b_stage_bytes,
                    sfb_base + st.index * self.sfb_stage_bytes,
                    xq_addr,
                    xsf_addr,
                    h_addr,
                    hsf_addr,
                    s_tok,
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
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
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
        sfa_col = NUM_ACC * self.n_max
        sfb_col = sfa_col + self.num_sfa_tmem_cols
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            tiled_mma.shape_mnk,
            SF_VEC,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            tiled_mma.shape_mnk,
            SF_VEC,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + sfa_col, dtype=self.sf_dtype), tCtSFA_layout
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + sfb_col, dtype=self.sf_dtype), tCtSFB_layout
        )
        copy_sfa, tCsSFA_s2t, tCtSFA_s2t = s2t_copy(sSFA, tCtSFA, self.cta_group, self.sf_dtype)
        copy_sfb, tCsSFB_s2t, tCtSFB_s2t = s2t_copy(sSFB, tCtSFB, self.cta_group, self.sf_dtype)
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
                cute.copy(copy_sfa, tCsSFA_s2t[(None, None, None, None, ab_st.index)], tCtSFA_s2t)
                cute.copy(copy_sfb, tCsSFB_s2t[(None, None, None, None, ab_st.index)], tCtSFB_s2t)
                for kp in cutlass.range(num_kphases, unroll_full=True):
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
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
        s_seg_e,
        s_seg_r0,
        s_seg_n,
        s_a1: cute.Tensor,
        s_c: cute.Tensor,
        s_a2: cute.Tensor,
        h_addr,
        hsf_addr,
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
        is_up = Int32(0)
        if ((q >> 3) & 1) == 0:
            is_up = Int32(1)  # FC1: up lanes (gate partner at q ^ 8)
        # this group's units: k = grp, grp + EPI_GROUPS, ...; unit k's accumulator is buffer k % NUM_ACC
        bar_id = Int32(EPI_BAR) + grp
        k = grp
        g, slot = self.unit(s_q, q_full, k)
        while g >= 0:
            is1, seg, mt = self.decode(g, u1)
            ex = s_seg_e[seg]
            rows = s_seg_n[seg]
            r0 = s_seg_r0[seg]
            buf = k % NUM_ACC
            cute.arch.mbarrier_wait(acc_full + buf, (k // NUM_ACC) & 1)
            nch = (rows + 7) // 8
            if is1 != 0:
                a = s_a1[ex]
                c = s_c[ex]
                c1 = _mul_rn(a, cutlass.Float32(LOG2E))  # a * (alpha * log2e), alpha = 1
                for ch in cutlass.range(nch, unroll=1):
                    self.load_acc(
                        tmem_ptr, tCtAcc_fake, tiled_copy_t2r, thr_copy_t2r, rAcc, buf, ch
                    )
                    for t in cutlass.range_constexpr(8):
                        nrow = ch * 8 + t
                        xv = rAcc[t]
                        gate = cute.arch.shuffle_sync_bfly(xv, 8)
                        # trtllm-gen swiGlu E2m1 epilogue (no clamp: the selects of the clamped variant are identities)
                        sgm = _rcp(_add_rn(_ex2(-_mul_rn(gate, c1)), cutlass.Float32(1.0)))
                        gs = _mul_rn(gate, sgm)
                        hv = _mul_rn(_fma_rn(xv, a, _mul_rn(a, cutlass.Float32(0.0))), gs)
                        hv = _sel_f32(is_up, hv, cutlass.Float32(0.0))
                        # |h c| = |h| c (c >= 0); gate lanes 0
                        amax = _redux_max_abs_f32(_mul_rn(hv, c))
                        sfb = _e4m3_bits(_mul_rn(amax, cutlass.Float32(RCP6)))
                        sff = _e4m3_to_f32(sfb)
                        osc = _sel_f32(
                            sfb, _rcp(sff), cutlass.Float32(0.0)
                        )  # sfb != 0 <=> scale != 0
                        qv = _mul_rn(_mul_rn(hv, osc), c)
                        byte = _e2m1x2(cute.arch.shuffle_sync_down(qv, 16), qv)
                        # lane j < 8 holds code byte j of the row's 8 (columns 16 wq + 2 j, + 1)
                        if lane < 8:
                            if nrow < rows:
                                r = r0 + nrow
                                _st_global_u8(
                                    h_addr
                                    + Int64(r) * Int64(self.inter // 2)
                                    + Int64(mt * 32 + wq * 8 + lane),
                                    byte,
                                )
                                if lane == 0:
                                    saddr = (
                                        hsf_addr
                                        + (Int64(r >> 3) * Int64(self.groups) + Int64(mt)) * 32
                                        + Int64(r & 7) * 4
                                        + Int64(wq)
                                    )
                                    _st_global_u8(saddr, sfb)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + buf)
                self.free_unit(bar_id, epi_tid, q_full, slot, cnt_base, seg, True)
            else:
                s = s_a2[ex]
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
                                    _scale_pack_bf16x2(rAcc[t], partner, s),
                                    op,
                                )
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + buf)
                self.free_unit(bar_id, epi_tid, q_full, slot, cnt_base, seg, False)
            k = k + EPI_GROUPS
            g, slot = self.unit(s_q, q_full, k)


# ----------------------------------------------------------------------------------------------------------------
def moe_nvfp4_engine(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    p2t: torch.Tensor,
    cta_batch: torch.Tensor,
    cta_limit: torch.Tensor,
    num_ctas: torch.Tensor,
    w31: torch.Tensor,
    w31_sf: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    scale_c: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    h: torch.Tensor,
    hsf: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    *,
    tile: int,
    n_max: int,
    tile_k: int,
    release_after_wait: bool = False,
    spare_sms: int = 0,
    candidates: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    e2p: Optional[torch.Tensor] = None,
    num_padded: Optional[torch.Tensor] = None,
    local_start: int = 0,
) -> torch.Tensor:
    """Run the engine. ``xq`` [m, H/2] u8, ``xsf`` [m, H/16] u8 (linear), tables int32 for token tile ``tile``,
    ``w31`` [E, 2I, H/2] u8, ``w31_sf`` [E, 2I, H/16] u8 (128x4), ``w2`` [E, H, I/2] u8, ``w2_sf`` [E, H, I/16] u8,
    ``scale_c`` / ``alpha1`` / ``alpha2`` [E] fp32 (output1_scale_scalar, output1_scale_gate_scalar,
    output2_scale_scalar), ``h`` [rows, I/2] u8 and ``hsf`` [rows * I/16] u8 (scratch), ``out`` [rows, H] bf16,
    ``state`` from :func:`new_state`. Rows above n_max per expert split into segments. One CTA per SM but
    ``spare_sms``.

    ``candidates`` [m, lists, top_k] int32 (m <= ROUTE_MAX_ROWS, ``moe_router_gemm``'s ``route_top_k`` epilogue over
    lists x 128 experts): the engine routes the call for the local experts from ``local_start`` and writes
    ``topk_ids`` [m, top_k] int32, ``topk_weights`` [m, top_k] bf16, ``e2p`` [m * top_k], ``p2t``, the tile tables
    and ``num_ctas`` / ``num_padded`` (``prerouted_table_views`` of one buffer) before it reads them."""
    e, two_i, half_h = w31.shape
    hidden, inter = 2 * half_h, two_i // 2
    assert e <= MAX_LOCAL_EXPERTS, "per-expert scalars are staged for <= 128 local experts"
    per_stage = (128 + n_max) * tile_k // 2 + 2 * 128 * tile_k // SF_VEC
    num_stages = min(10, ring_smem_bytes(xq.device) // per_stage)
    args = (
        gmem_ptr(w31, cutlass.Float4E2M1FN, 32),
        gmem_ptr(w31_sf, cutlass.Float8E4M3FN, 16),
        gmem_ptr(w2, cutlass.Float4E2M1FN, 32),
        gmem_ptr(w2_sf, cutlass.Float8E4M3FN, 16),
        gmem_ptr(w2, cutlass.Float4E2M1FN, 32),
        Int64(xq.data_ptr()),
        Int64(xsf.data_ptr()),
        Int64(p2t.data_ptr()),
        Int64(cta_batch.data_ptr()),
        Int64(cta_limit.data_ptr()),
        Int64(num_ctas.data_ptr()),
        Int32(cta_batch.numel()),
        Int64(scale_c.data_ptr()),
        Int64(alpha1.data_ptr()),
        Int64(alpha2.data_ptr()),
        Int64(h.data_ptr()),
        Int64(hsf.data_ptr()),
        Int64(out.data_ptr()),
        Int64(state.data_ptr()),
        Int32(e),
    )
    route_lists, route_top_k = 0, 0
    route_args = ()
    if candidates is not None:
        m = xq.shape[0]
        assert candidates.dim() == 3 and candidates.shape[0] == m <= ROUTE_MAX_ROWS
        _, route_lists, route_top_k = candidates.shape
        assert route_error(route_top_k, route_lists) is None, route_error(route_top_k, route_lists)
        assert candidates.dtype == torch.int32 and candidates.is_contiguous()
        assert topk_ids.shape == (m, route_top_k) and topk_ids.dtype == torch.int32
        assert topk_weights.shape == (m, route_top_k) and topk_weights.dtype == torch.bfloat16
        assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
        assert e2p.numel() == m * route_top_k and num_padded.numel() == 1
        assert p2t.numel() == (cta_batch.numel() + 1) * tile
        sms = torch.cuda.get_device_properties(xq.device).multi_processor_count
        p2t_copies = torch.empty((sms, p2t.numel()), dtype=torch.int32, device=xq.device)
        route_args = (
            Int64(candidates.data_ptr()),
            Int64(topk_ids.data_ptr()),
            Int64(topk_weights.data_ptr()),
            Int64(e2p.data_ptr()),
            Int64(num_padded.data_ptr()),
            Int64(p2t_copies.data_ptr()),
            Int32(m),
            Int32(local_start),
        )
    config = (
        hidden,
        inter,
        n_max,
        num_stages,
        tile,
        tile_k,
        release_after_wait,
        route_top_k,
        route_lists,
    )
    launch(
        lambda: MoeNvfp4Engine(*config),
        (MoeNvfp4Engine, e, *config),
        args,
        xq.device,
        spare_sms,
        tail_args=route_args,
    )
    return out
