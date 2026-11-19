# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer's TGV CuTe DSL GEMM (Apache-2.0,
# ``flashinfer/gemm/kernels/tgv_gemm_cute_ext.py``): its DMA warps, shared-memory staging and MMA atoms are reused
# through subclassing; the tile mapping, the MMA K loop and the epilogue are this kernel's.
"""Kimi K3.1 MoE head in one launch: the router logits and the routed latent of every row.

``x [M, 7168]`` BF16 times the stacked head weight ``[router 896 | latent down 3584] x 7168`` BF16, swap-AB like
FlashInfer's TGV kernel (the weight rows are the MMA M, the tokens the MMA N; a 2-CTA cluster per 64-row weight
tile and token tile, no persistence):

* latent tiles (weight rows 896..4479) accumulate the whole K in order in one TMEM accumulator and store BF16 into
  ``latent [M, 3584]``: the bits of any GEMM that accumulates K in order in FP32 and rounds once (cuBLAS without
  split-K, every TGV config);
* router tiles (weight rows 0..895) accumulate K tiles ``[0, k_split)`` and ``[k_split, K / 128)`` in two TMEM
  accumulators and store their FP32 sum into ``logits [M, 896]``: the bits of a two-way split-K GEMM over those K
  halves (cuBLAS's FP32 router GEMM at 65-176 rows splits K that way);
* with ``latent_split`` the latent tiles do the same and round the sum once to BF16: the bits of a two-way split-K
  latent GEMM (cuBLAS's at 65-96 rows).

The two halves of a router tile run in one CTA, so the sum needs no atomics and no zeroed output. Every CTA reads all
K tiles, router or latent, so the grid is balanced.
"""

from __future__ import annotations

from typing import Dict, Tuple

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

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.gemm.tgv_gemm import TgvGemmKernel

HIDDEN = 7168
ROUTER_ROWS = 896
LATENT_ROWS = 3584
HEAD_ROWS = ROUTER_ROWS + LATENT_ROWS
CTA_M = 64
CTA_K = _tgv._TGV_CUTE_EXT_CTA_K


class MoeHeadGemm(TgvGemmKernel):
    """Router logits (two exact K halves, FP32) and routed latent (whole K, BF16) of ``x`` in one launch."""

    def __init__(
        self,
        *,
        cta_n: int,
        num_ab_stage: int,
        use_pdl: bool,
        k_split: int,
        latent_split: bool = False,
    ):
        super().__init__(
            acc_dtype=cutlass.Float32,
            cta_m=CTA_M,
            cta_n=cta_n,
            cta_k=CTA_K,
            num_ab_stage=num_ab_stage,
            use_2cta=True,
            use_pdl=use_pdl,
            has_bias=False,
        )
        if 2 * cta_n > 256:
            raise ValueError("two TMEM accumulators need cta_n <= 128")
        self.router_tiles = ROUTER_ROWS // CTA_M
        self.k_split = k_split
        self.latent_split = latent_split

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,  # (HEAD_ROWS, K, 1), K-major: the stacked head weight
        b: cute.Tensor,  # (M, K, 1), K-major: the tokens
        c_router: cute.Tensor,  # (ROUTER_ROWS, M, 1), FP32, router-row-major
        c_latent: cute.Tensor,  # (LATENT_ROWS, M, 1), BF16, latent-row-major
        stream: cuda.CUstream,
    ):
        grid = cute.round_up(
            (
                cute.ceil_div(a.layout.shape[0], self.cta_m),
                cute.ceil_div(b.layout.shape[0], self.cta_n),
                1,
            ),
            self.cluster_shape,
        )
        self.kernel(a, b, c_router, c_latent).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC0: cute.Tensor, mC1: cute.Tensor):
        DMA_Stage = self.num_ab_stage
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type
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
        # Router weight rows come first; the router / latent boundary is a whole cluster of M tiles.
        is_router = bidx < self.router_tiles

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
        # Two accumulators: the latent uses stage 0; the router's K halves use stages 0 and 1.
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=2)

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

        k_tile_count = cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k)
        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
        gA_tile = cute.local_tile(mA, a_tiler_mk, (bidx, None, 0))
        gB_n_idx = bidy * num_mma_ctas + cta_rank_in_cluster
        gB_tile = cute.local_tile(mB, b_tiler_nk, (gB_n_idx, None, 0))
        # A router CTA writes logits tile bidx; a latent CTA latent tile bidx - router_tiles (each view is only
        # dereferenced by its own kind of CTA).
        router_m = bidx
        latent_m = bidx - self.router_tiles
        if is_router:
            router_m = bidx
            latent_m = cutlass.Int32(0)
        else:
            router_m = cutlass.Int32(0)
        gD0_tile = cute.local_tile(mC0, c_tiler_mn, (router_m, bidy, 0))
        gD1_tile = cute.local_tile(mC1, c_tiler_mn, (latent_m, bidy, 0))

        cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_a_warp(
                bar_full, bar_empty, leader_rank, gA_tile, sA, a_cta_v_map, k_tile_count
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
            self.head_mma_warp(
                is_leader,
                is_router,
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
            self.head_epilog_warp(
                is_router,
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

        # Both CTAs of the pair meet before the TMEM dealloc and again before exit.
        self.pair_teardown(warp_idx, tmem_base_ptr)

    @cute.experimental.jit
    def head_mma_warp(
        self,
        is_leader: cutlass.Boolean,
        is_router: cutlass.Boolean,
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
        """The TGV MMA warp with the router's second K half in the second accumulator."""
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(tmem_ptr, acc_layout)
        acc0 = tAcc[None, None, None, 0]
        acc1 = tAcc[None, None, None, 1]

        if is_leader:
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            commit_mask = 0b11
            # Router CTAs (and latent CTAs with latent_split): K tiles [0, k_split) into acc0, the rest into acc1.
            # Otherwise every K tile into acc0.
            two_halves = is_router
            if cutlass.const_expr(self.latent_split):
                two_halves = cutlass.Boolean(True)
            split = k_tile_count
            if two_halves:
                split = cutlass.Int32(self.k_split)
            full_phase = cutlass.Int32(0)
            full_phase = self._mma_k_loop(
                mma_atom,
                commit_mask,
                bar_full,
                bar_empty,
                sA,
                sB,
                acc0,
                mma_inst_tile_k,
                cutlass.Int32(0),
                split,
                full_phase,
            )
            if two_halves:
                full_phase = self._mma_k_loop(
                    mma_atom,
                    commit_mask,
                    bar_full,
                    bar_empty,
                    sA,
                    sB,
                    acc1,
                    mma_inst_tile_k,
                    split,
                    k_tile_count,
                    full_phase,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)

        # Phase 0 (the allocation, with the epilogue's first arrivals) completes before this warp's phase-1 arrival.
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        # The dealloc runs in pair_teardown, once both CTAs of the pair are there.

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
    def head_epilog_warp(
        self,
        is_router: cutlass.Boolean,
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
        """TMEM -> registers -> st.global: router tiles add their two FP32 halves and store FP32, latent tiles round
        their one accumulator to BF16."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(tmem_ptr, acc_layout)
        acc0 = tCtAcc[((None, None), 0, 0, 0)]
        acc1 = tCtAcc[((None, None), 0, 0, 1)]
        epi_tile = (self.cta_m, self.cta_n)
        mnk = (self.cta_m, self.cta_n, self.cta_k)
        if is_router:
            atom = sm100_utils.get_tmem_load_op(
                mnk, d0_layout, cutlass.Float32, self.acc_dtype, epi_tile, self.use_2cta
            )
            copy0 = tcgen05.make_tmem_copy(atom, acc0)
            copy1 = tcgen05.make_tmem_copy(atom, acc1)
            gD_epi = cute.flat_divide(gD0_tile, epi_tile)
            rmem_layout = cute_ext.make_t2r_rmem_layout(copy0, gD_epi, epi_tid)
            r0 = cute_ext.allocate(
                self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
            )
            r1 = cute_ext.allocate(
                self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
            )
            thr0 = copy0.get_slice(epi_tid)
            thr1 = copy1.get_slice(epi_tid)
            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            cute_ext.partition_and_copy(thr0, acc0, r0)
            cute_ext.partition_and_copy(thr1, acc1, r1)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            r0.store(r0.load() + r1.load())
            cute_ext.partition_and_copy(thr0, r0, gD_epi[None, None, 0, 0])
        else:
            atom = sm100_utils.get_tmem_load_op(
                mnk, d1_layout, cutlass.BFloat16, self.acc_dtype, epi_tile, self.use_2cta
            )
            copy0 = tcgen05.make_tmem_copy(atom, acc0)
            gD_epi = cute.flat_divide(gD1_tile, epi_tile)
            rmem_layout = cute_ext.make_t2r_rmem_layout(copy0, gD_epi, epi_tid)
            r0 = cute_ext.allocate(
                self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
            )
            rD = cute_ext.allocate(
                cutlass.BFloat16, cute.AddressSpace.rmem, rmem_layout, alignment=32
            )
            thr0 = copy0.get_slice(epi_tid)
            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            cute_ext.partition_and_copy(thr0, acc0, r0)
            if cutlass.const_expr(self.latent_split):
                copy1 = tcgen05.make_tmem_copy(atom, acc1)
                r1 = cute_ext.allocate(
                    self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
                )
                cute_ext.partition_and_copy(copy1.get_slice(epi_tid), acc1, r1)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            if cutlass.const_expr(self.latent_split):
                r0.store(r0.load() + r1.load())
            rD.store(r0.load().to(cutlass.BFloat16))
            cute_ext.partition_and_copy(thr0, rD, gD_epi[None, None, 0, 0])


@cute.experimental.jit
def _head_entry(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (1, HEAD_ROWS, K)
    b: cute.Tensor,  # (1, K, M)
    c0: cute.Tensor,  # (1, ROUTER_ROWS, M)
    c1: cute.Tensor,  # (1, LATENT_ROWS, M)
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c0 = cute.make_tensor(c0.iterator, cute.select(c0.layout, mode=[1, 2, 0]))
    c1 = cute.make_tensor(c1.iterator, cute.select(c1.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c0, c1, stream)


_COMPILED: Dict[Tuple, object] = {}


def _views(x: torch.Tensor, head_weight: torch.Tensor, logits: torch.Tensor, latent: torch.Tensor):
    """The swap-AB cute views: kernel A = the weight, B = the tokens, C = the outputs transposed."""
    a = head_weight.unsqueeze(0)  # (1, HEAD_ROWS, K): weight rows x K, K contiguous
    b = x.t().unsqueeze(0)  # (1, K, M): K contiguous
    c0 = logits.t().unsqueeze(0)  # (1, ROUTER_ROWS, M): router row contiguous
    c1 = latent.t().unsqueeze(0)  # (1, LATENT_ROWS, M): latent row contiguous
    return (
        from_dlpack(a, assumed_align=32).mark_layout_dynamic(leading_dim=2),
        from_dlpack(b, assumed_align=32).mark_layout_dynamic(leading_dim=1),
        from_dlpack(c0, assumed_align=32).mark_layout_dynamic(leading_dim=1),
        from_dlpack(c1, assumed_align=32).mark_layout_dynamic(leading_dim=1),
    )


def run_moe_head(
    x: torch.Tensor,
    head_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    config: Tuple[int, int, bool],
    pdl: bool,
    k_split: int,
) -> None:
    """``logits = x @ head_weight[:896].T`` (two K halves split at K tile ``k_split``, FP32) and ``latent = x @
    head_weight[896:].T`` (BF16; with ``latent_split`` the sum of the same two K halves) in one launch; ``config`` =
    (cta_n, ab_stages, latent_split)."""
    cta_n, stages, latent_split = config
    key = (torch.cuda.current_device(), cta_n, stages, latent_split, bool(pdl), k_split)
    compiled = _COMPILED.get(key)
    if compiled is None:
        gemm = MoeHeadGemm(
            cta_n=cta_n,
            num_ab_stage=stages,
            use_pdl=bool(pdl),
            k_split=k_split,
            latent_split=latent_split,
        )
        rx = torch.empty(8, HIDDEN, dtype=torch.bfloat16, device=x.device)
        rw = torch.empty(HEAD_ROWS, HIDDEN, dtype=torch.bfloat16, device=x.device)
        r0 = torch.empty(8, ROUTER_ROWS, dtype=torch.float32, device=x.device)
        r1 = torch.empty(8, LATENT_ROWS, dtype=torch.bfloat16, device=x.device)
        compiled = cute_ext.compile(_head_entry, gemm, *_views(rx, rw, r0, r1), make_fake_stream())
        _COMPILED[key] = compiled
    compiled(
        *_views(x, head_weight, logits, latent),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
