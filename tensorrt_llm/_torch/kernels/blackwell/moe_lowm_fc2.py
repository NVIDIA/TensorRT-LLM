# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routed NVFP4 MoE FC2 for decode-sized batches, bit-identical to the trtllm-gen FC2 cubin
``bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x8x256_*_dynB`` (Gemm2 of ``moe/trtllmGen/runner.cu``), with a shared-memory
and register footprint that fits ``CTAS_PER_SM`` CTAs per SM.

``out[p, h] = bf16_satfinite(scale[e] * sum_k act[p, k] * w2[e, h, k])`` for every permuted row ``p`` of a token tile
(8 rows) of local expert ``e``. Operands and routing tables are the trtllm-gen MoE runner's buffers (``moe_lowm_fc2``):
``w2`` rows shuffled in 32-row blocks with 128x4 E4M3 scales, the FC1 output with 8x4 scales.

Bit-exactness: the cubin's MMA (``tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X``, M 128, N 8, K 64) in its K
order with K tiles of 256 (the zero-filled K tail issues the same MMAs), one FP32 ``mul.rn`` by ``scale[e]`` and
``cvt.rn.satfinite.bf16x2.f32``.

Work: a persistent grid walks the (m tile, token tile) pairs. Warp 0 streams weights by TMA before the PDL wait, warp 1
activations after it, warp 2 issues the MMAs into two TMEM accumulators, warp 3 L2-prefetches later tiles' weights and
warps 4-7 scale, convert and store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
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
from ...flashinfer_utils import get_env_enable_pdl

if TYPE_CHECKING:
    import torch
from .moe_decode_front import _st_shared
from .ptx_helpers import _ld_nc_f32, _ld_s32, _scale_pack_bf16x2, _st_global_b32, s2t_copy

TILE_M = 128
TILE_N = 8
TILE_K = 256  # the cubin's K tile (K order and zero-filled tail MMAs are defined by it)
SF_VEC = 16
EPI_WARP0 = 4
NUM_ACC = 2
NUM_STAGES = 4  # K stages per CTA
CTAS_PER_SM = 2


def _ld_b32(addr: Int64):
    return cute.arch.inline_ptx(
        "ld.global.b32 {$w0}, [{$r0}];", write_only_types=[Int32], read_only_args=[addr]
    )


def _fence_proxy_async_smem() -> None:
    cute.arch.inline_ptx("fence.proxy.async.shared::cta;")


def _shfl_down8_f32(v):
    return cute.arch.inline_ptx(
        "shfl.sync.down.b32 {$w0}, {$r0}, 8, 0x1f, 0xffffffff;",
        write_only_types=[cutlass.Float32],
        read_only_args=[v],
    )


def _mbar_expect_tx(addr: Int32, nbytes: Int32) -> None:
    cute.arch.inline_ptx(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [{$r0}], {$r1};",
        read_only_args=[addr, nbytes],
    )


def _prefetch_bulk_l2(addr: Int64, nbytes: Int32) -> None:
    cute.arch.inline_ptx(
        "cp.async.bulk.prefetch.L2.global [{$r0}], {$r1};", read_only_args=[addr, nbytes]
    )


class MoeLowMFc2Nvfp4:
    """Routed NVFP4 FC2, tile 128 (hidden) x 8 (rows) x 256 (K), ``NUM_STAGES`` K stages per CTA."""

    def __init__(self, k_tiles: int, k_groups: int, use_pdl: bool = True):
        self.use_pdl = use_pdl
        self.k_tiles = k_tiles
        self.k_groups = k_groups  # 4-scale groups per activation row (I / 64)
        self.tile_k = TILE_K
        self.gps = TILE_K // (4 * SF_VEC)  # 4-scale groups per K tile
        self.num_stages = NUM_STAGES
        self.cta_group = tcgen05.CtaGroup.ONE
        self.acc_dtype = cutlass.Float32
        self.threads_per_cta = 256
        self.sfb_n = 128  # SFB rows per stage: one whole 128-row scale atom, rows 8.. stay zero

    @cute.jit
    def _unit(self, t, m_tiles):
        """(token tile, m tile) of unit t, token-tile major."""
        y = t // m_tiles
        mt = t % m_tiles
        return y, mt

    def _setup_attributes(self):
        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            (TILE_M, TILE_N),
        )
        mma_inst_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.tile_k // mma_inst_k
        self.num_sfa_tmem_cols = (TILE_M // 32) * self.mma_inst_tile_k
        self.num_sfb_tmem_cols = self.mma_inst_tile_k * (self.sfb_n // 32)
        total = NUM_ACC * TILE_N + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        cols = 32
        while cols < total:
            cols *= 2
        self.num_tmem_cols = cols
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (self.tiled_mma.thr_id.shape,)
        )
        tiler = (TILE_M, TILE_N, self.tile_k)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, tiler, self.a_dtype, self.num_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, tiler, self.b_dtype, self.num_stages
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, tiler, SF_VEC, self.num_stages
        )
        self.mma_tiler_sfb = (TILE_M, self.sfb_n, self.tile_k)
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
            self.tiled_mma_sfb, self.mma_tiler_sfb, SF_VEC, self.num_stages
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_addr: Int64,
        c_addr: Int64,
        scale_addr: Int64,
        cta_batch_addr: Int64,
        cta_limit_addr: Int64,
        num_ctas_addr: Int64,
        max_ctas: Int32,
        problem: tuple,
        grid_ctas: Int32,
        stream: cuda.CUstream,
    ):
        """problem = (H, I, E, padded rows); the grid is ``grid_ctas`` persistent CTAs; the tables hold ``max_ctas``
        entries."""
        h, k, e, rows = problem
        a = cute.make_tensor(a_ptr, cute.make_ordered_layout((h, k, e), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, cute.make_ordered_layout((rows, k, 1), order=(1, 0, 2)))
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(b).mma_major_mode()
        sfa = cute.make_tensor(sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
        self._setup_attributes()
        tiled_mma = self.tiled_mma
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            (TILE_M, TILE_N, self.tile_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            (TILE_M, TILE_N, self.tile_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            sfa,
            sfa_smem_layout,
            (TILE_M, TILE_N, self.tile_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.tma_bytes_a = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.sf_dtype, sfa_smem_layout
        )
        self.tma_bytes_b = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.kernel(
            tiled_mma,
            self.tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_b,
            tma_tensor_b,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            sfb_addr,
            c_addr,
            scale_addr,
            cta_batch_addr,
            cta_limit_addr,
            num_ctas_addr,
            max_ctas,
            h,
            k,
            a_ptr.toint(),
            sfa_ptr.toint(),
        ).launch(
            grid=[grid_ctas, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=CTAS_PER_SM,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfb_addr: Int64,
        c_addr: Int64,
        scale_addr: Int64,
        cta_batch_addr: Int64,
        cta_limit_addr: Int64,
        num_ctas_addr: Int64,
        max_ctas: Int32,
        h_total: Int32,
        k_total: Int32,
        a_base: Int64,
        sfa_base: Int64,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        m_tiles = h_total // TILE_M
        # Routing tables before the dependency wait (tile count; first tile's expert): the trtllm-gen FC1 cubin
        # triggers this launch only after its own wait for the routing kernel (GemmOptions.h: a load task that
        # triggers the secondary also waits for the primary).
        n_ctas = _ld_s32(num_ctas_addr)
        y0f, _mt0 = self._unit(bidx, m_tiles)
        y0 = cutlass.min(y0f, max_ctas - 1)
        e0 = _ld_s32(cta_batch_addr + Int64(y0) * 4)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)

        Stages = self.num_stages

        @cute.struct
        class SharedStorage:
            ab_bars: cute.struct.MemRange[cutlass.Int64, Stages * 2]
            acc_bars: cute.struct.MemRange[cutlass.Int64, NUM_ACC * 2]
            tmem_bar: cutlass.Int64
            tmem_base: cutlass.Int32

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_bars.data_ptr(),
            num_stages=Stages,
            producer_group=CooperativeGroup(Agent.Thread, 2),
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
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_bar, 32 + 128)

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
        # SFB: only rows 0..7 of each 128-row atom are ever written; zero the rest once.
        sfb_base = sSFB.iterator.toint()
        sfb_words = Stages * (self.sfb_n * (self.tile_k // SF_VEC)) // 4
        for i in cutlass.range(tidx, sfb_words, self.threads_per_cta, unroll=1):
            _st_shared(sfb_base + i * 4, Int32(0))
        _fence_proxy_async_smem()
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)
        cute.arch.barrier()

        num_tiles = n_ctas * m_tiles

        if warp_idx == 0:
            self.load_a(
                ab_full,
                ab_empty,
                tma_atom_a,
                mA_mkl,
                sA,
                tma_atom_sfa,
                mSFA_mkl,
                sSFA,
                tiled_mma,
                cluster_layout_vmnk,
                cta_batch_addr,
                e0,
                bidx,
                gdx,
                num_tiles,
                m_tiles,
            )
        elif warp_idx == 1:
            self.load_b(
                ab_full,
                ab_empty,
                tma_atom_b,
                mB_nkl,
                sB,
                sfb_base,
                sfb_addr,
                tiled_mma,
                cluster_layout_vmnk,
                bidx,
                gdx,
                num_tiles,
                m_tiles,
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
                bidx,
                gdx,
                num_tiles,
            )
        elif warp_idx == 3:
            # L2 prefetch of this CTA's second tile's weights and of a late-launching CTA's first tile's
            lane = cute.arch.lane_idx()
            t = bidx + gdx  # this CTA's second tile
            if lane >= 2:
                t = (
                    gdx - 1 - bidx
                )  # the first tile of a CTA that launches late (CTAs launch in index order)
            if lane < 4:
                if t < num_tiles:
                    if lane < 2 or bidx * 2 < gdx:
                        yq, mq = self._unit(t, m_tiles)
                        ey = Int64(_ld_s32(cta_batch_addr + Int64(yq) * 4))
                        mt = Int64(mq)
                        a_tile = Int64(TILE_M) * Int64(k_total // 2)
                        s_tile = Int64(TILE_M) * Int64(k_total // SF_VEC)
                        if lane % 2 == 0:
                            _prefetch_bulk_l2(
                                a_base + (ey * Int64(m_tiles) + mt) * a_tile, Int32(a_tile)
                            )
                        else:
                            _prefetch_bulk_l2(
                                sfa_base + (ey * Int64(m_tiles) + mt) * s_tile, Int32(s_tile)
                            )
        elif warp_idx >= EPI_WARP0:
            self.epilogue(
                acc_full,
                acc_empty,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                c_addr,
                scale_addr,
                cta_batch_addr,
                cta_limit_addr,
                tidx - EPI_WARP0 * 32,
                bidx,
                gdx,
                num_tiles,
                m_tiles,
                h_total,
                e0,
            )
        cute.arch.barrier()
        if warp_idx == 2:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
            cute.arch.dealloc_tmem(tmem_ptr, self.num_tmem_cols)
        return

    @cute.jit
    def load_a(
        self,
        ab_full,
        ab_empty,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        sA: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        sSFA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        cta_batch_addr: Int64,
        e0: Int32,
        bidx: Int32,
        gdx: Int32,
        num_tiles: Int32,
        m_tiles: Int32,
    ):
        """Weights + scales (constant): issued before griddepcontrol.wait, like the cubin (gridWaitForPrimaryA off);
        the routing tables were complete before the previous kernel triggered this launch."""
        thr_mma = tiled_mma.get_slice(0)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        gA = cute.local_tile(mA_mkl, (TILE_M, self.tile_k), (None, None, None))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        gSFA = cute.local_tile(mSFA_mkl, (TILE_M, self.tile_k), (None, None, None))
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gSFA), 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        ey = e0
        for t in cutlass.range(bidx, num_tiles, gdx, unroll=1):
            ya, mt = self._unit(t, m_tiles)
            if t != bidx:
                ey = _ld_s32(cta_batch_addr + Int64(ya) * 4)
            for kt in cutlass.range_constexpr(self.k_tiles):
                cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ab_full + st.index, self.tma_bytes_a)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, mt, kt, ey)],
                    tAsA[(None, st.index)],
                    tma_bar_ptr=ab_full + st.index,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, mt, kt, ey)],
                    tAsSFA[(None, st.index)],
                    tma_bar_ptr=ab_full + st.index,
                )
                st.advance()

    @cute.jit
    def load_b(
        self,
        ab_full,
        ab_empty,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        sB: cute.Tensor,
        sfb_base: Int32,
        sfb_addr: Int64,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        bidx: Int32,
        gdx: Int32,
        num_tiles: Int32,
        m_tiles: Int32,
    ):
        """FC1 output + its 8x4 scales, after griddepcontrol.wait. Per tile: the activation TMA of every K tile first
        (expect_tx only), then lane l loads the 4 scales of row l % 8, group 4 kt + l / 8 for every K tile at once,
        writes them into the stage's 128x4 atom l / 8 (16-byte rows) and the warp arrives (the second producer)."""
        thr_mma = tiled_mma.get_slice(0)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        gB = cute.local_tile(mB_nkl, (TILE_N, self.tile_k), (None, None, None))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        lane = cute.arch.lane_idx()
        groups = self.k_groups
        stage_bytes = self.sfb_n * (self.tile_k // SF_VEC)
        gps = self.gps
        own = lane // 8 < gps  # lanes of the stage's 128x4 atoms
        sf = cute.make_rmem_tensor((self.k_tiles,), Int32)
        st = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        for t in cutlass.range(bidx, num_tiles, gdx, unroll=1):
            y, _mtb = self._unit(t, m_tiles)
            if cutlass.const_expr(self.num_stages >= self.k_tiles):
                # every K tile has its own stage: all activation TMAs, then the scales
                i0 = st.index
                for kt in cutlass.range_constexpr(self.k_tiles):
                    cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                    with cute.arch.elect_one():
                        _mbar_expect_tx((ab_full + st.index).toint(), Int32(self.tma_bytes_b))
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, y, kt, 0)],
                        tBsB[(None, st.index)],
                        tma_bar_ptr=ab_full + st.index,
                    )
                    st.advance()
                for kt in cutlass.range_constexpr(self.k_tiles):
                    g = kt * gps + lane // 8
                    sf[kt] = Int32(0)
                    if own:
                        if g < groups:
                            sf[kt] = _ld_b32(
                                sfb_addr + Int64(y * groups + g) * 32 + Int64(lane % 8) * 4
                            )
                for kt in cutlass.range_constexpr(self.k_tiles):
                    idx = (i0 + kt) % self.num_stages
                    if own:
                        _st_shared(
                            sfb_base + idx * stage_bytes + (lane // 8) * 512 + (lane % 8) * 16,
                            sf[kt],
                        )
                    _fence_proxy_async_smem()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(ab_full + idx)
            else:
                # fewer stages than K tiles: scales loaded up front, one stage at a time
                for kt in cutlass.range_constexpr(self.k_tiles):
                    g = kt * gps + lane // 8
                    sf[kt] = Int32(0)
                    if own:
                        if g < groups:
                            sf[kt] = _ld_b32(
                                sfb_addr + Int64(y * groups + g) * 32 + Int64(lane % 8) * 4
                            )
                for kt in cutlass.range_constexpr(self.k_tiles):
                    cute.arch.mbarrier_wait(ab_empty + st.index, st.phase)
                    with cute.arch.elect_one():
                        _mbar_expect_tx((ab_full + st.index).toint(), Int32(self.tma_bytes_b))
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, y, kt, 0)],
                        tBsB[(None, st.index)],
                        tma_bar_ptr=ab_full + st.index,
                    )
                    if own:
                        _st_shared(
                            sfb_base + st.index * stage_bytes + (lane // 8) * 512 + (lane % 8) * 16,
                            sf[kt],
                        )
                    _fence_proxy_async_smem()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(ab_full + st.index)
                    st.advance()

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
        bidx: Int32,
        gdx: Int32,
        num_tiles: Int32,
    ):
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((TILE_M, TILE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        cute.arch.alloc_tmem(self.num_tmem_cols, tmem_base_smem_ptr)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.relinquish_tmem_alloc_permit()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        sfa_col = NUM_ACC * TILE_N
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
        for _t in cutlass.range(bidx, num_tiles, gdx, unroll=1):
            cute.arch.mbarrier_wait(acc_empty + acc_st.index, acc_st.phase)
            tCtAcc = cute.make_tensor(tmem_ptr + acc_st.index * TILE_N, tCtAcc_fake.layout)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kt in cutlass.range_constexpr(self.k_tiles):
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

    @cute.jit
    def epilogue(
        self,
        acc_full,
        acc_empty,
        tmem_bar,
        tmem_base_smem_ptr,
        tiled_mma: cute.TiledMma,
        c_addr: Int64,
        scale_addr: Int64,
        cta_batch_addr: Int64,
        cta_limit_addr: Int64,
        epi_tid: Int32,
        bidx: Int32,
        gdx: Int32,
        num_tiles: Int32,
        m_tiles: Int32,
        h_total: Int32,
        e0: Int32,
    ):
        thr_mma = tiled_mma.get_slice(0)
        acc_shape = tiled_mma.partition_shape_C((TILE_M, TILE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tCcL = thr_mma.partition_C(cute.make_identity_tensor((TILE_M, TILE_N)))
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(TILE_N), tcgen05.Pack.NONE), self.acc_dtype
        )
        epi_tile = (TILE_M, TILE_N)
        tAcc0 = cute.flat_divide(
            cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[((None, None), 0, 0)], epi_tile
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc0[(None, None, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
        tTR_cL = thr_copy_t2r.partition_D(cute.flat_divide(tCcL[((None, None), 0, 0)], epi_tile))
        rAcc = cute.make_rmem_tensor(tTR_cL[(None, None, None, 0, 0)].shape, self.acc_dtype)
        crd = tTR_cL[(None, None, None, 0, 0)]
        row0, _c0 = crd[0]
        q = row0 % 32
        hcol = (
            (row0 // 32) * 32 + (q % 8) * 4 + q // 8
        )  # hidden column of this TMEM lane (32-row block shuffle)
        lead = (
            q // 8
        ) % 2 == 0  # owns the even hidden column of a pair (partner lane q + 8 has hcol + 1)
        acc_st = make_pipeline_state(PipelineUserType.Consumer, NUM_ACC)
        for t in cutlass.range(bidx, num_tiles, gdx, unroll=1):
            y, mt = self._unit(t, m_tiles)
            ey = e0
            if t != bidx:
                ey = _ld_s32(cta_batch_addr + Int64(y) * 4)
            limit = _ld_s32(cta_limit_addr + Int64(y) * 4)
            s = _ld_nc_f32(scale_addr + Int64(ey) * 4)
            cute.arch.mbarrier_wait(acc_full + acc_st.index, acc_st.phase)
            tAcc = cute.flat_divide(
                cute.make_tensor(tmem_ptr + acc_st.index * TILE_N, tCtAcc_fake.layout)[
                    ((None, None), 0, 0)
                ],
                epi_tile,
            )
            cute.copy(
                tiled_copy_t2r, thr_copy_t2r.partition_S(tAcc)[(None, None, None, 0, 0)], rAcc
            )
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(acc_empty + acc_st.index)
            acc_st.advance()
            col = Int64(mt * TILE_M + hcol)
            for j in cutlass.range_constexpr(TILE_N):
                partner = _shfl_down8_f32(rAcc[j])
                r = y * TILE_N + j
                if lead:
                    if r < limit:
                        _st_global_b32(
                            c_addr + (Int64(r) * Int64(h_total) + col) * 2,
                            _scale_pack_bf16x2(rAcc[j], partner, s),
                        )


_COMPILED = {}


def num_sms(device: Optional[torch.device] = None) -> int:
    """SMs of ``device`` the persistent grid spans: all but those ``cute_dsl_utils.sm_reserve()`` keeps out."""
    import torch

    from ...cute_dsl_utils import sm_reserve

    return torch.cuda.get_device_properties(device).multi_processor_count - sm_reserve()


def moe_lowm_fc2(
    act: torch.Tensor,
    act_sf: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    scale: torch.Tensor,
    cta_batch: torch.Tensor,
    cta_limit: torch.Tensor,
    num_non_exiting: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """FC2 of the trtllm-gen MoE path. ``act`` [rows, I/2] uint8 (E2M1 x2), ``act_sf`` 8x4 E4M3 (uint8),
    ``w2`` [E, H, I/2] uint8, ``w2_sf`` [E, H, I/16] (128x4), ``scale`` [E] fp32, tables int32 (``cta_batch`` /
    ``cta_limit`` per token tile, ``num_non_exiting`` [1]), ``out`` [padded rows, H] bf16 with at least the total padded
    rows (rows of a tile past its limit are not written)."""
    import torch

    rows, half_k = act.shape
    e, h, half_k2 = w2.shape
    k = 2 * half_k
    if half_k2 != half_k or h % TILE_M or k % (4 * SF_VEC):
        raise ValueError(f"unsupported FC2 shape rows {rows} H {h} I {k} E {e}")
    if (
        out.dim() != 2
        or out.shape[1] != h
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise ValueError("out must be a contiguous bf16 [padded rows, H] tensor")
    grid_ctas = num_sms(act.device) * CTAS_PER_SM
    ptrs = (
        gmem_ptr(w2, cutlass.Float4E2M1FN, 32),
        gmem_ptr(w2_sf, cutlass.Float8E4M3FN, 16),
        gmem_ptr(act, cutlass.Float4E2M1FN, 32),
    )
    addrs = (
        Int64(act_sf.data_ptr()),
        Int64(out.data_ptr()),
        Int64(scale.data_ptr()),
        Int64(cta_batch.data_ptr()),
        Int64(cta_limit.data_ptr()),
        Int64(num_non_exiting.data_ptr()),
        Int32(cta_batch.numel()),
    )
    problem = (Int32(h), Int32(k), Int32(e), Int32(rows))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_pdl = get_env_enable_pdl()
    fn = _COMPILED.get((k, use_pdl))
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile the low-M MoE FC2 before CUDA graph capture")
        kern = MoeLowMFc2Nvfp4(-(-k // TILE_K), k // (4 * SF_VEC), use_pdl)
        fn = cute.compile(kern, *ptrs, *addrs, problem, Int32(grid_ctas), stream)
        _COMPILED[(k, use_pdl)] = fn
    fn(*ptrs, *addrs, problem, Int32(grid_ctas), stream)
    return out
