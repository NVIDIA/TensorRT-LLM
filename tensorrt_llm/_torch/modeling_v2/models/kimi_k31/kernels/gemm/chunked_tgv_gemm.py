# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer's TGV CuTe DSL GEMM (Apache-2.0,
# ``flashinfer/gemm/kernels/tgv_gemm_cute_ext.py``) through ``TgvGemmKernel``: its DMA warps, shared-memory staging
# and MMA atoms are reused; the tile mapping, the chunked MMA K loop and the epilogue are this kernel's.
"""TGV GEMM with the K dimension summed in given chunks: the bits of a split-K GEMM.

``x [M, K]`` BF16 times one or two weights ("kinds", each ``[rows, K]`` BF16, K-major), swap-AB like FlashInfer's TGV
kernel: one 64-row (or 128-row) weight tile and one token tile per CTA, 2-CTA clusters. Kind 0's tiles come first, then
kind 1's, each kind in whole clusters (the cluster's leader runs the MMA K loop of both CTAs, so a cluster's two tiles
must share the chunk bounds); a tile's rows past its weight are TMA zero fill and its output rows past the weight are
written with those zero products. Each kind accumulates K tiles (128 wide) in order, split at its K-tile bounds into
chunks in separate TMEM accumulators; the epilogue adds the chunk sums in order ((c0 + c1) + c2 ...) and rounds once to
the kind's output dtype (BF16 or FP32). Those are the bits of a split-K GEMM over the same chunks whose reduction adds
the partials in order (cuBLAS's split-K kernels with their splitKreduce, FlashInfer's split-K), and with no bound the
bits of any GEMM that accumulates K in order in FP32 and rounds once.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from flashinfer.gemm.kernels import tgv_gemm_cute_ext as _tgv

from .tgv_gemm import TgvGemmKernel

CTA_M = 64
CTA_K = _tgv._TGV_CUTE_EXT_CTA_K
#: TMEM columns the chunk accumulators may use (``acc stages x cta_n``).
TMEM_COLS = 256


def kind_tiles(rows: int, cta_m: int = CTA_M) -> int:
    """``cta_m``-row weight tiles of one kind: whole 2-CTA clusters."""
    return -(-rows // (2 * cta_m)) * 2


class ChunkedTgvGemm(TgvGemmKernel):
    """Kind 0's weight tiles (``tiles0`` of them), then kind 1's; each kind's K split into chunks at its K-tile
    bounds."""

    def __init__(
        self,
        *,
        cta_n: int,
        num_ab_stage: int,
        use_pdl: bool,
        tiles0: int,
        bounds0: Tuple[int, ...],
        bounds1: Tuple[int, ...],
        cta_m: int = CTA_M,
    ):
        super().__init__(
            acc_dtype=cutlass.Float32,
            cta_m=cta_m,
            cta_n=cta_n,
            cta_k=CTA_K,
            num_ab_stage=num_ab_stage,
            use_2cta=True,
            use_pdl=use_pdl,
            has_bias=False,
        )
        if tiles0 % 2:
            raise ValueError("kind 0 must fill whole 2-CTA clusters")
        self.tiles0 = tiles0
        self.bounds0 = tuple(bounds0)
        self.bounds1 = tuple(bounds1)
        self.acc_stages = 1 + max(len(self.bounds0), len(self.bounds1))
        if self.acc_stages * cta_n > 256:
            raise ValueError("the chunk accumulators need more than 256 TMEM columns")

    @cute.experimental.jit
    def __call__(
        self,
        a0: cute.Tensor,  # (kind 0 rows, K, 1), K-major
        a1: cute.Tensor,  # (kind 1 rows, K, 1), K-major
        b: cute.Tensor,  # (M, K, 1), K-major: the tokens
        c0: cute.Tensor,  # (tiles0 * 64, M, 1)
        c1: cute.Tensor,  # (kind 1 tiles * 64, M, 1); kind 1 absent: c0 again
        tiles1: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        grid = (self.tiles0 + tiles1, cute.ceil_div(b.layout.shape[0], self.cta_n), 1)
        self.kernel(a0, a1, b, c0, c1).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA0: cute.Tensor,
        mA1: cute.Tensor,
        mB: cute.Tensor,
        mC0: cute.Tensor,
        mC1: cute.Tensor,
    ):
        DMA_Stage = self.num_ab_stage
        a_major = utils.LayoutEnum.from_tensor(mA0).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA0.element_type
        d0_layout = utils.LayoutEnum.from_tensor(mC0)
        d1_layout = utils.LayoutEnum.from_tensor(mC1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)
        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        leader_rank = cutlass.Int32(0)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader = cta_rank_in_cluster == 0
        is_kind1 = bidx >= self.tiles0
        m0 = bidx
        m1 = cutlass.Int32(0)
        if is_kind1:
            m0 = cutlass.Int32(0)
            m1 = bidx - self.tiles0

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, mnk_tiler, ab_dtype, DMA_Stage
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, mnk_tiler, ab_dtype, DMA_Stage
        )
        sA = cute_ext.allocate(
            ab_dtype, cute.AddressSpace.smem, a_smem_layout_staged, alignment=1024
        )
        sB = cute_ext.allocate(
            ab_dtype, cute.AddressSpace.smem, b_smem_layout_staged, alignment=1024
        )
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=self.acc_stages
        )
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(DMA_Stage), alignment=8
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(DMA_Stage), alignment=8
        )
        bar_tma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator
        bar_tma_epilog = bar_tma_epilog_arr.iterator
        bar_mma_epilog = bar_mma_epilog_arr.iterator
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator
        tmem_base_ptr = tmem_base_arr.iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + 128)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

        k_tile_count = cute.ceil_div(cute.size(mA0, mode=[1]), self.cta_k)
        a0_cta_v_map = cute_ext.get_cta_v_map_ab(mA0, mnk_tiler, tiled_mma, "A")
        a1_cta_v_map = cute_ext.get_cta_v_map_ab(mA1, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
        gA0_tile = cute.local_tile(mA0, a_tiler_mk, (m0, None, 0))
        gA1_tile = cute.local_tile(mA1, a_tiler_mk, (m1, None, 0))
        gB_n_idx = bidy * num_mma_ctas + cta_rank_in_cluster
        gB_tile = cute.local_tile(mB, b_tiler_nk, (gB_n_idx, None, 0))
        gD0_tile = cute.local_tile(mC0, c_tiler_mn, (m0, bidy, 0))
        gD1_tile = cute.local_tile(mC1, c_tiler_mn, (m1, bidy, 0))

        cute.arch.cluster_wait()

        if warp_idx == 0:
            if is_kind1:
                self.dma_a_warp(
                    bar_full, bar_empty, leader_rank, gA1_tile, sA, a1_cta_v_map, k_tile_count
                )
            else:
                self.dma_a_warp(
                    bar_full, bar_empty, leader_rank, gA0_tile, sA, a0_cta_v_map, k_tile_count
                )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.chunk_mma_warp(
                is_leader,
                is_kind1,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.chunk_epilog_warp(
                is_kind1,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD0_tile,
                gD1_tile,
                tidx - 128,
                d0_layout,
                d1_layout,
            )

        self.pair_teardown(warp_idx, tmem_base_ptr)

    @cute.experimental.jit
    def chunk_mma_warp(
        self,
        is_leader: cutlass.Boolean,
        is_kind1: cutlass.Boolean,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        """The TGV MMA warp with chunk ``i`` of the K tiles in accumulator ``i``."""
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(tmem_ptr, acc_layout)

        if is_leader:
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            commit_mask = 0b11
            full_phase = cutlass.Int32(0)
            if is_kind1:
                full_phase = self._chunks(
                    mma_atom,
                    commit_mask,
                    bar_full,
                    bar_empty,
                    sA,
                    sB,
                    tAcc,
                    mma_inst_tile_k,
                    k_tile_count,
                    self.bounds1,
                    full_phase,
                )
            else:
                full_phase = self._chunks(
                    mma_atom,
                    commit_mask,
                    bar_full,
                    bar_empty,
                    sA,
                    sB,
                    tAcc,
                    mma_inst_tile_k,
                    k_tile_count,
                    self.bounds0,
                    full_phase,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)

        # Phase 0 (the allocation) completes before this warp arrives on phase 1: the non-leader CTA's MMA warp
        # skips the K loop and could otherwise arrive twice before the epilogue's first arrivals.
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)

    @cute.experimental.jit
    def _chunks(
        self,
        mma_atom,
        commit_mask: cutlass.Constexpr,
        bar_full,
        bar_empty,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tAcc: cute.Tensor,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
        bounds: cutlass.Constexpr,
        full_phase: cutlass.Int32,
    ) -> cutlass.Int32:
        lo = cutlass.Int32(0)
        for i in cutlass.range_constexpr(len(bounds) + 1):
            hi = k_tile_count
            if cutlass.const_expr(i < len(bounds)):
                hi = cutlass.Int32(bounds[i])
            full_phase = self._mma_k_loop(
                mma_atom,
                commit_mask,
                bar_full,
                bar_empty,
                sA,
                sB,
                tAcc[None, None, None, i],
                mma_inst_tile_k,
                lo,
                hi,
                full_phase,
            )
            lo = hi
        return full_phase

    @cute.experimental.jit
    def _mma_k_loop(
        self,
        mma_atom,
        commit_mask: cutlass.Constexpr,
        bar_full,
        bar_empty,
        sA: cute.Tensor,
        sB: cute.Tensor,
        acc: cute.Tensor,
        mma_inst_tile_k: cutlass.Constexpr,
        k_begin: cutlass.Int32,
        k_end: cutlass.Int32,
        full_phase: cutlass.Int32,
    ) -> cutlass.Int32:
        """K tiles ``[k_begin, k_end)`` into ``acc``, the first one overwriting it (the TGV MMA loop)."""
        DMA_Stage = self.num_ab_stage
        for k_tile in cutlass.range(k_begin, k_end, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != k_begin)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                a_frag = cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3)
                b_frag = cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3)
                cute_ext.dot(mma_atom, a_frag, b_frag, acc)
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, commit_mask, self.cta_group)
            if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                full_phase = full_phase ^ 1
        return full_phase

    @cute.experimental.jit
    def chunk_epilog_warp(
        self,
        is_kind1: cutlass.Boolean,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD0_tile: cute.Tensor,
        gD1_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        d0_layout: cutlass.Constexpr,
        d1_layout: cutlass.Constexpr,
    ):
        """TMEM -> registers -> st.global: the chunk sums added in order, rounded once to the output's dtype."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(tmem_ptr, acc_layout)
        epi_tile = (self.cta_m, self.cta_n)
        mnk = (self.cta_m, self.cta_n, self.cta_k)
        if is_kind1:
            self._epilog_kind(
                tCtAcc,
                bar_mma_epilog,
                bar_tmem_alloc,
                gD1_tile,
                epi_tile,
                mnk,
                epi_tid,
                d1_layout,
                gD1_tile.element_type,
                self.bounds1,
            )
        else:
            self._epilog_kind(
                tCtAcc,
                bar_mma_epilog,
                bar_tmem_alloc,
                gD0_tile,
                epi_tile,
                mnk,
                epi_tid,
                d0_layout,
                gD0_tile.element_type,
                self.bounds0,
            )

    @cute.experimental.jit
    def _epilog_kind(
        self,
        tCtAcc: cute.Tensor,
        bar_mma_epilog,
        bar_tmem_alloc,
        gD_tile: cute.Tensor,
        epi_tile: cutlass.Constexpr,
        mnk: cutlass.Constexpr,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
        out_dtype: cutlass.Constexpr,
        bounds: cutlass.Constexpr,
    ):
        atom = sm100_utils.get_tmem_load_op(
            mnk, d_layout, out_dtype, self.acc_dtype, epi_tile, True
        )
        acc0 = tCtAcc[((None, None), 0, 0, 0)]
        copy0 = tcgen05.make_tmem_copy(atom, acc0)
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(copy0, gD_epi, epi_tid)
        thr0 = copy0.get_slice(epi_tid)
        r0 = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        ri = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr0, acc0, r0)
        for i in cutlass.range_constexpr(1, len(bounds) + 1):
            acci = tCtAcc[((None, None), 0, 0, i)]
            copyi = tcgen05.make_tmem_copy(atom, acci)
            cute_ext.partition_and_copy(copyi.get_slice(epi_tid), acci, ri)
            cute.arch.fence_view_async_tmem_load()
            r0.store(r0.load() + ri.load())
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        if cutlass.const_expr(out_dtype == cutlass.Float32):
            cute_ext.partition_and_copy(thr0, r0, gD_epi[None, None, 0, 0])
        else:
            rD = cute_ext.allocate(out_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
            rD.store(r0.load().to(out_dtype))
            cute_ext.partition_and_copy(thr0, rD, gD_epi[None, None, 0, 0])


class ChunkedTgvGemmSplit(ChunkedTgvGemm):
    """One CTA (pair) per K chunk: grid z = chunk; a kind's CTA z accumulates the K tiles of its chunk z in order into
    one accumulator and stores it into plane z of its output (FP32 partials for a kind with bounds, the final dtype
    for a kind without). The caller adds the planes in order and rounds once: the bits of the multi-accumulator
    kernel, with each CTA streaming only its chunk of the weight tile."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.acc_stages = 1
        self.planes = 1 + max(len(self.bounds0), len(self.bounds1))

    @cute.experimental.jit
    def __call__(self, a0, a1, b, c0, c1, tiles1: cutlass.Constexpr, stream: cuda.CUstream):
        grid = (self.tiles0 + tiles1, cute.ceil_div(b.layout.shape[0], self.cta_n), self.planes)
        self.kernel(a0, a1, b, c0, c1).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.jit
    def _edge(
        self,
        bounds: cutlass.Constexpr,
        z: cutlass.Int32,
        k_tiles: cutlass.Int32,
        upper: cutlass.Constexpr,
    ):
        """Edge z (upper: z + 1) of the chunks [0, b0, ..., K)."""
        e = cutlass.Int32(0)
        if cutlass.const_expr(upper):
            e = k_tiles
        for i in cutlass.range_constexpr(len(bounds)):
            if cutlass.const_expr(upper):
                if z == i:
                    e = cutlass.Int32(bounds[i])
            else:
                if z == i + 1:
                    e = cutlass.Int32(bounds[i])
        return e

    @cute.experimental.kernel
    def kernel(
        self,
        mA0: cute.Tensor,
        mA1: cute.Tensor,
        mB: cute.Tensor,
        mC0: cute.Tensor,
        mC1: cute.Tensor,
    ):
        DMA_Stage = self.num_ab_stage
        bidx, bidy, bidz = cute.arch.block_idx()
        is_kind1 = bidx >= self.tiles0
        nch = cutlass.Int32(1 + len(self.bounds0))
        if is_kind1:
            nch = cutlass.Int32(1 + len(self.bounds1))
        active = bidz < nch
        a_major = utils.LayoutEnum.from_tensor(mA0).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA0.element_type
        d0_layout = utils.LayoutEnum.from_tensor(mC0)
        d1_layout = utils.LayoutEnum.from_tensor(mC1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)
        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        leader_rank = cutlass.Int32(0)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader = cta_rank_in_cluster == 0
        m0 = bidx
        m1 = cutlass.Int32(0)
        if is_kind1:
            m0 = cutlass.Int32(0)
            m1 = bidx - self.tiles0
        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, DMA_Stage),
            alignment=1024,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, DMA_Stage),
            alignment=1024,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(DMA_Stage), alignment=8
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(DMA_Stage), alignment=8
        )
        bar_tma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator
        bar_tma_epilog = bar_tma_epilog_arr.iterator
        bar_mma_epilog = bar_mma_epilog_arr.iterator
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator
        tmem_base_ptr = tmem_base_arr.iterator
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + 128)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        k_tiles = cute.ceil_div(cute.size(mA0, mode=[1]), self.cta_k)
        lo = self._edge(self.bounds0, bidz, k_tiles, False)
        hi = self._edge(self.bounds0, bidz, k_tiles, True)
        if is_kind1:
            lo = self._edge(self.bounds1, bidz, k_tiles, False)
            hi = self._edge(self.bounds1, bidz, k_tiles, True)
        count = hi - lo
        if not active:
            count = cutlass.Int32(0)
        a0_map = cute_ext.get_cta_v_map_ab(mA0, mnk_tiler, tiled_mma, "A")
        a1_map = cute_ext.get_cta_v_map_ab(mA1, mnk_tiler, tiled_mma, "A")
        b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
        gA0 = cute.local_tile(mA0, a_tiler_mk, (m0, None, 0))
        gA1 = cute.local_tile(mA1, a_tiler_mk, (m1, None, 0))
        gB = cute.local_tile(mB, b_tiler_nk, (bidy * num_mma_ctas + cta_rank_in_cluster, None, 0))
        gD0 = cute.local_tile(mC0, c_tiler_mn, (m0, bidy, bidz))
        gD1 = cute.local_tile(mC1, c_tiler_mn, (m1, bidy, bidz))
        cute.arch.cluster_wait()
        if warp_idx == 0:
            if is_kind1:
                self.dma_k(bar_full, bar_empty, leader_rank, gA1, sA, a1_map, lo, count, False)
            else:
                self.dma_k(bar_full, bar_empty, leader_rank, gA0, sA, a0_map, lo, count, False)
        elif warp_idx == 1:
            self.dma_k(bar_full, bar_empty, leader_rank, gB, sB, b_map, lo, count, True)
        elif warp_idx == 2:
            num_tmem_cols = 256
            cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=True)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
            tAcc = cute.make_tensor(tmem_ptr, acc_layout)
            if is_leader:
                mma_atom = cute.make_mma_atom(tiled_mma.op)
                self._mma_k_loop(
                    mma_atom,
                    0b11,
                    bar_full,
                    bar_empty,
                    sA,
                    sB,
                    tAcc[None, None, None, 0],
                    mma_inst_tile_k,
                    cutlass.Int32(0),
                    count,
                    cutlass.Int32(0),
                )
                with cute.arch.elect_one():
                    tcgen05.commit(bar_mma_epilog, 0b11, self.cta_group)
            # phase 0 completes before this warp arrives on phase 1 (see chunk_mma_warp)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        elif warp_idx >= 4:
            if is_kind1:
                self.split_epilog(
                    active,
                    bar_mma_epilog,
                    bar_tmem_alloc,
                    tmem_base_ptr,
                    acc_layout,
                    gD1,
                    tidx - 128,
                    d1_layout,
                )
            else:
                self.split_epilog(
                    active,
                    bar_mma_epilog,
                    bar_tmem_alloc,
                    tmem_base_ptr,
                    acc_layout,
                    gD0,
                    tidx - 128,
                    d0_layout,
                )
        self.pair_teardown(warp_idx, tmem_base_ptr)

    @cute.experimental.jit
    def split_epilog(
        self,
        active: cutlass.Boolean,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
    ):
        """TMEM -> registers -> st.global of the CTA's one accumulator in the output's dtype (skipped when inactive)."""

        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(tmem_ptr, acc_layout)
        epi_tile = (self.cta_m, self.cta_n)
        mnk = (self.cta_m, self.cta_n, self.cta_k)
        out_dtype = gD_tile.element_type
        atom = sm100_utils.get_tmem_load_op(
            mnk, d_layout, out_dtype, self.acc_dtype, epi_tile, True
        )
        acc0 = tCtAcc[((None, None), 0, 0, 0)]
        copy0 = tcgen05.make_tmem_copy(atom, acc0)
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(copy0, gD_epi, epi_tid)
        thr0 = copy0.get_slice(epi_tid)
        r0 = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr0, acc0, r0)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        if active:
            if cutlass.const_expr(out_dtype == cutlass.Float32):
                cute_ext.partition_and_copy(thr0, r0, gD_epi[None, None, 0, 0])
            else:
                rD = cute_ext.allocate(out_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
                rD.store(r0.load().to(out_dtype))
                cute_ext.partition_and_copy(thr0, rD, gD_epi[None, None, 0, 0])

    @cute.experimental.jit
    def dma_k(
        self,
        bar_full,
        bar_empty,
        leader_rank: cutlass.Int32,
        g_tile: cute.Tensor,
        s: cute.Tensor,
        cta_v_map: cute.Layout,
        k_begin: cutlass.Int32,
        count: cutlass.Int32,
        activation: cutlass.Constexpr,
    ):
        """FlashInfer's TGV DMA warp over K tiles [k_begin, k_begin + count); the activation operand waits on the
        previous grid first, the weight operand releases the dependents once its loads are issued."""
        DMA_Stage = self.num_ab_stage
        if cutlass.const_expr(activation and self.use_pdl):
            cute.arch.griddepcontrol_wait()
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s.element_type, cute.slice_(s.layout, (None, None, None, 0))
                    ),
                    peer_cta_rank_in_cluster=leader_rank,
                )
            cute_ext.tma_load(
                g_tile[None, None, k_begin + k_tile],
                s[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1
        if cutlass.const_expr((not activation) and self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1


@cute.experimental.jit
def _entry(
    gemm_op: cutlass.Constexpr,
    a0: cute.Tensor,  # (1, kind 0 rows, K)
    a1: cute.Tensor,  # (1, kind 1 rows, K)
    b: cute.Tensor,  # (1, K, M)
    c0: cute.Tensor,  # (1, tiles0 * 64, M)
    c1: cute.Tensor,  # (1, kind 1 tiles * 64, M)
    tiles1: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    a0 = cute.make_tensor(a0.iterator, cute.select(a0.layout, mode=[1, 2, 0]))
    a1 = cute.make_tensor(a1.iterator, cute.select(a1.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c0 = cute.make_tensor(c0.iterator, cute.select(c0.layout, mode=[1, 2, 0]))
    c1 = cute.make_tensor(c1.iterator, cute.select(c1.layout, mode=[1, 2, 0]))
    gemm_op(a0, a1, b, c0, c1, tiles1, stream)


_COMPILED: Dict[Tuple, object] = {}


def _views(x, w0, w1, out0, out1):
    """The swap-AB cute views: A = the kinds' weights, B = the tokens, C = the padded outputs transposed."""
    return (
        from_dlpack(w0.unsqueeze(0), assumed_align=32).mark_layout_dynamic(leading_dim=2),
        from_dlpack(w1.unsqueeze(0), assumed_align=32).mark_layout_dynamic(leading_dim=2),
        from_dlpack(x.t().unsqueeze(0), assumed_align=32).mark_layout_dynamic(leading_dim=1),
        from_dlpack(out0.t().unsqueeze(0), assumed_align=32).mark_layout_dynamic(leading_dim=1),
        from_dlpack(out1.t().unsqueeze(0), assumed_align=32).mark_layout_dynamic(leading_dim=1),
    )


def _key(x, kinds, cta_n, stages, pdl, cta_m=CTA_M) -> Tuple:
    return (
        cta_m,
        x.device.index,
        x.shape[1],
        cta_n,
        stages,
        bool(pdl),
        tuple((w.shape[0], out.shape[1], out.dtype, tuple(bounds)) for w, out, bounds in kinds),
    )


def run_chunked_tgv(
    x: torch.Tensor,
    kinds: Sequence[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]],
    cta_n: int,
    stages: int,
    pdl: bool,
    cta_m: int = CTA_M,
) -> None:
    """For each kind ``(w [rows, K], out [M, kind_tiles(rows, cta_m) * cta_m], bounds)``: ``out[:, :rows] = x @ w.T``, K
    summed in the chunks between the K-tile ``bounds`` and rounded once to ``out.dtype`` (BF16 or FP32); one launch for
    one or two kinds, on ``cta_m``-row weight tiles (64 or 128). ``x`` and the weights K-contiguous and 32-byte aligned
    (views of a larger weight are fine)."""
    if not 1 <= len(kinds) <= 2:
        raise ValueError("one or two kinds")
    for w, out, bounds in kinds:
        if out.shape[1] != kind_tiles(w.shape[0], cta_m) * cta_m or out.shape[0] != x.shape[0]:
            raise ValueError(
                f"output {tuple(out.shape)} for {w.shape[0]} weight rows and {x.shape[0]} tokens"
            )
        if (1 + len(bounds)) * cta_n > TMEM_COLS:
            raise ValueError(f"{1 + len(bounds)} chunk accumulators of {cta_n} columns exceed TMEM")
    (w0, out0, bounds0), (w1, out1, bounds1) = kinds[0], kinds[-1]
    tiles1 = out1.shape[1] // cta_m if len(kinds) == 2 else 0
    key = _key(x, kinds, cta_n, stages, pdl, cta_m)
    compiled = _COMPILED.get(key)
    if compiled is None:
        gemm = ChunkedTgvGemm(
            cta_n=cta_n,
            num_ab_stage=stages,
            use_pdl=bool(pdl),
            tiles0=out0.shape[1] // cta_m,
            bounds0=bounds0,
            bounds1=bounds1,
            cta_m=cta_m,
        )
        dev, k = x.device, x.shape[1]
        rx = torch.empty(8, k, dtype=x.dtype, device=dev)
        r0 = torch.empty(w0.shape[0], k, dtype=w0.dtype, device=dev)
        r1 = torch.empty(w1.shape[0], k, dtype=w1.dtype, device=dev)
        o0 = torch.empty(8, out0.shape[1], dtype=out0.dtype, device=dev)
        o1 = torch.empty(8, out1.shape[1], dtype=out1.dtype, device=dev)
        compiled = cute_ext.compile(
            _entry, gemm, *_views(rx, r0, r1, o0, o1), tiles1, make_fake_stream()
        )
        _COMPILED[key] = compiled
    compiled(*_views(x, w0, w1, out0, out1), cuda.CUstream(torch.cuda.current_stream().cuda_stream))


def _split_views(x, w0, w1, out0, out1):
    """The swap-AB cute views of the split launch: C = the planes ``[planes, M, width]`` as ``(planes, width, M)``."""
    return (
        from_dlpack(w0.unsqueeze(0), assumed_align=32).mark_layout_dynamic(leading_dim=2),
        from_dlpack(w1.unsqueeze(0), assumed_align=32).mark_layout_dynamic(leading_dim=2),
        from_dlpack(x.t().unsqueeze(0), assumed_align=32).mark_layout_dynamic(leading_dim=1),
        from_dlpack(out0.permute(0, 2, 1), assumed_align=32).mark_layout_dynamic(leading_dim=1),
        from_dlpack(out1.permute(0, 2, 1), assumed_align=32).mark_layout_dynamic(leading_dim=1),
    )


def _split_key(x, kinds, cta_n, stages, pdl) -> Tuple:
    return (
        "split",
        x.device.index,
        x.shape[1],
        cta_n,
        stages,
        bool(pdl),
        tuple(
            (w.shape[0], out.shape[0], out.shape[2], out.dtype, tuple(bounds))
            for w, out, bounds in kinds
        ),
    )


def run_chunked_tgv_split(
    x: torch.Tensor,
    kinds: Sequence[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]],
    cta_n: int,
    stages: int,
    pdl: bool,
) -> None:
    """:func:`run_chunked_tgv` with one CTA per K chunk. For each kind ``(w [rows, K], out [1 + len(bounds), M,
    kind_tiles(rows) * 64], bounds)``: plane ``i`` of ``out`` gets chunk ``i``'s product (FP32; a kind without bounds
    gets its whole product in ``out.dtype`` in plane 0). Adding the planes in order in FP32 and rounding once gives
    :func:`run_chunked_tgv`'s bits."""
    if not 1 <= len(kinds) <= 2:
        raise ValueError("one or two kinds")
    for w, out, bounds in kinds:
        if out.dim() != 3 or out.shape[0] != 1 + len(bounds) or out.shape[1] != x.shape[0]:
            raise ValueError(
                f"planes {tuple(out.shape)} for {len(bounds) + 1} chunks and {x.shape[0]} tokens"
            )
        if out.shape[2] != kind_tiles(w.shape[0]) * CTA_M:
            raise ValueError(f"plane width {out.shape[2]} for {w.shape[0]} weight rows")
        if bounds and out.dtype != torch.float32:
            raise ValueError("a split kind's planes are FP32 partials")
    (w0, out0, bounds0), (w1, out1, bounds1) = kinds[0], kinds[-1]
    tiles1 = out1.shape[2] // CTA_M if len(kinds) == 2 else 0
    key = _split_key(x, kinds, cta_n, stages, pdl)
    compiled = _COMPILED.get(key)
    if compiled is None:
        gemm = ChunkedTgvGemmSplit(
            cta_n=cta_n,
            num_ab_stage=stages,
            use_pdl=bool(pdl),
            tiles0=out0.shape[2] // CTA_M,
            bounds0=bounds0,
            bounds1=bounds1,
        )
        dev, k = x.device, x.shape[1]
        rx = torch.empty(8, k, dtype=x.dtype, device=dev)
        r0 = torch.empty(w0.shape[0], k, dtype=w0.dtype, device=dev)
        r1 = torch.empty(w1.shape[0], k, dtype=w1.dtype, device=dev)
        o0 = torch.empty(out0.shape[0], 8, out0.shape[2], dtype=out0.dtype, device=dev)
        o1 = torch.empty(out1.shape[0], 8, out1.shape[2], dtype=out1.dtype, device=dev)
        compiled = cute_ext.compile(
            _entry, gemm, *_split_views(rx, r0, r1, o0, o1), tiles1, make_fake_stream()
        )
        _COMPILED[key] = compiled
    compiled(
        *_split_views(x, w0, w1, out0, out1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


__all__ = [
    "CTA_M",
    "ChunkedTgvGemm",
    "ChunkedTgvGemmSplit",
    "kind_tiles",
    "run_chunked_tgv",
    "run_chunked_tgv_split",
]
