# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0): flashinfer/gemm/kernels/tgv_gemm_cute_ext.py. The
# DMA warp bodies of HeadShardPushGemm are FlashInfer's TGV DMA warps with the PDL release moved from the weight
# warp to the token warp.
"""Kimi K3.1 sharded MoE head, push variant: the slice GEMM's epilogue multicasts its outputs into the all-gather.

:mod:`.moe_head_shard` runs the slice GEMM into padded output planes, then a gather kernel (or the gather with the
MoE prologue folded in) reads this rank's planes back, multicasts them into the current Lamport buffer of the
workspace and polls every rank's. Where one CTA holds the final value of an output element -- the multi-accumulator
row counts, whose latent sums the whole K in one chunk -- the GEMM can push instead:

* :class:`HeadShardPushGemm`: the multi-accumulator chunked TGV GEMM (the same tiles, chunks, accumulation and
  rounding, so the same values bit for bit) whose epilogue, after staging its tile in shared memory, multicasts the
  tile's valid columns into the gather's slots of this rank (the rows of :class:`.moe_head_shard.MoeHeadGatherKernel`),
  ``-0.0`` words sent as ``+0.0`` as the gather sends them;
* :class:`MoeHeadPollRouteQuantKernel`: the fused gather + MoE prologue without the push: it polls the slots and
  routes and quantizes the rows as :class:`.moe_head_shard.MoeHeadRouteQuantKernel` does, so its outputs are
  that kernel's bits.

Ordering (the rules of :class:`.lamport_ar.LamportFinishKernel` for the AR-push GEMM): every CTA of the GEMM
reads the rotation state in its epilogue, after its token warp's ``griddepcontrol.wait``, and only the token warp
releases the dependent grid, right after that wait (the weight warp streams weights before it and releases nothing),
so the poll kernel becomes resident only once every earlier grid has completed and reads this call's state without
waiting. Its workers count themselves after reading the state; the updater CTA alone waits for the GEMM grid (every
GEMM CTA has read the state by then), then for the workers, and writes the next call's state; its wait also makes the
poll kernel's completion imply the GEMM's. The slots the GEMM writes were re-armed by the call before last, as for
every user of the workspace.
"""

from __future__ import annotations

import functools
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm.lamport_allgather import (
    sanitize_f32,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.gemm import chunked_tgv_gemm as _ctgv
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.route_tail import (
    _WIDE_BIAS_WORDS,
    NUM_EXPERTS,
    TOP_K,
    _ld_nc_v4_f32,
    _rcp_approx_ftz,
    _st_global_b16,
    _st_global_b32,
    _sts_v4_f32,
)

from .moe_head_shard import (
    _UNIT_SCALE_BITS,
    LATENT_ROWS,
    ROUTER_ROWS,
    MoeHeadGatherKernel,
    MoeHeadRouteQuantKernel,
    _gather_args,
    gather_bytes,
)

#: Named barrier of the push GEMM's four epilogue warps (the tile's stores before its reads).
_EPI_PUSH_BAR = 6


@dsl_user_op
def _lds_v4_b32(addr, *, loc=None, ip=None):
    """16-byte shared-memory load of four words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


class HeadShardPushGemm(_ctgv.ChunkedTgvGemm):
    """Multi-accumulator chunked TGV GEMM over kind 0 = this rank's latent rows (BF16 out) and kind 1 = its router
    rows (FP32 out) whose epilogue also multicasts each stored tile's valid columns into the gather's Lamport slots
    (see the module docstring)."""

    def __init__(
        self,
        *,
        world: int,
        cta_n: int,
        num_ab_stage: int,
        use_pdl: bool,
        tiles0: int,
        bounds0: Tuple[int, ...],
        bounds1: Tuple[int, ...],
    ):
        super().__init__(
            cta_n=cta_n,
            num_ab_stage=num_ab_stage,
            use_pdl=use_pdl,
            tiles0=tiles0,
            bounds0=bounds0,
            bounds1=bounds1,
        )
        if self.cta_m != 64:
            raise ValueError("the push epilogue serves 64-row weight tiles")
        # the gathered row layout (vectors per rank / row) only
        self.g = MoeHeadGatherKernel(world, 0, 0, use_pdl)
        self.row_bytes = self.g.vecs * 16

    @cute.experimental.jit
    def __call__(
        self,
        a0: cute.Tensor,
        a1: cute.Tensor,
        b: cute.Tensor,
        c0: cute.Tensor,
        c1: cute.Tensor,
        tiles1: cutlass.Constexpr,
        m: Int32,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        grid = (self.tiles0 + tiles1, cute.ceil_div(b.layout.shape[0], self.cta_n), 1)
        self.kernel(a0, a1, b, c0, c1, m, mc_base, flags, rank).launch(
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
        m: Int32,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
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
            self.stage_epilog_warp(
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
                sA,
            )
            self.push_staged(is_kind1, m0, m1, bidy, m, mc_base, flags, rank, sA, tidx - 128)

        self.pair_teardown(warp_idx, tmem_base_ptr)

    @cute.experimental.jit
    def stage_epilog_warp(
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
        sA: cute.Tensor,
    ):
        """The chunk epilogue (sums added in order, rounded once) into a ``[cta_n][cta_m]`` tile in shared memory:
        the A stages, drained once the MMA's last commit has landed."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(tmem_ptr, acc_layout)
        epi_tile = (self.cta_m, self.cta_n)
        mnk = (self.cta_m, self.cta_n, self.cta_k)
        if is_kind1:
            self._stage_kind(
                tCtAcc, bar_mma_epilog, bar_tmem_alloc, gD1_tile, epi_tile, mnk, epi_tid, d1_layout,
                gD1_tile.element_type, self.bounds1, sA,
            )  # fmt: skip
        else:
            self._stage_kind(
                tCtAcc, bar_mma_epilog, bar_tmem_alloc, gD0_tile, epi_tile, mnk, epi_tid, d0_layout,
                gD0_tile.element_type, self.bounds0, sA,
            )  # fmt: skip

    @cute.experimental.jit
    def _stage_kind(
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
        sA: cute.Tensor,
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
        sD = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=out_dtype),
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
        )
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
            cute_ext.partition_and_copy(thr0, r0, sD)
        else:
            rD = cute_ext.allocate(out_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
            rD.store(r0.load().to(out_dtype))
            cute_ext.partition_and_copy(thr0, rD, sD)

    @cute.experimental.jit
    def push_staged(
        self,
        is_kind1: cutlass.Boolean,
        m0: Int32,
        m1: Int32,
        bidy: Int32,
        m: Int32,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        sA: cute.Tensor,
        epi_tid: Int32,
    ):
        """Multicast the valid columns of the tile :meth:`stage_epilog_warp` left in shared memory: 16-byte vectors
        of 8 BF16 latent or 4 FP32 router columns of one token, read after the epilogue warps' barrier, into this
        rank's slots of the gathered rows, in the current Lamport buffer (the state read here, after the token warp's
        wait). Padding columns and padding tokens are not sent."""
        cute.arch.barrier(barrier_id=_EPI_PUSH_BAR, number_of_threads=128)
        # (cur, dirty, bytes per buffer, ...) of the rotation state; buffer cur's single stage
        state = _lam.ld_volatile_v4(flags)
        mc = mc_base + Int64(state[0]) * Int64(state[2])
        n0 = bidy * self.cta_n
        sbase = sA.iterator.toint()
        if is_kind1:
            self._push_kind_smem(sbase, 4, 4, self.g.rw, 0, m1, n0, m, mc, rank, epi_tid)
        else:
            self._push_kind_smem(sbase, 8, 2, self.g.lw, self.g.rv, m0, n0, m, mc, rank, epi_tid)

    @cute.experimental.jit
    def _push_kind_smem(
        self,
        sbase: Int32,
        per_vec: cutlass.Constexpr,
        elem_bytes: cutlass.Constexpr,
        kind_rows: cutlass.Constexpr,
        j0: cutlass.Constexpr,
        mt: Int32,
        n0: Int32,
        m: Int32,
        mc: Int64,
        rank: Int32,
        epi_tid: Int32,
    ):
        vpt = self.cta_m // per_vec
        c_base = mt * self.cta_m
        for i in cutlass.range(epi_tid, self.cta_n * vpt, 128, unroll=1):
            tl = i // vpt
            cl = (i - tl * vpt) * per_vec
            c = c_base + cl
            t = n0 + tl
            if t < m:
                if c < kind_rows:
                    w = _lds_v4_b32(sbase + (tl * self.cta_m + cl) * elem_bytes)
                    dst = (
                        mc
                        + Int64(t) * self.row_bytes
                        + Int64(rank * self.g.own_vecs + j0 + c // per_vec) * 16
                    )
                    _lam.multimem_st_v4(
                        dst,
                        sanitize_f32(w[0]),
                        sanitize_f32(w[1]),
                        sanitize_f32(w[2]),
                        sanitize_f32(w[3]),
                    )

    # FlashInfer's TGV DMA warps, the PDL release moved to the token warp (after its wait).
    @cute.experimental.jit
    def dma_a_warp(
        self,
        bar_full,
        bar_empty,
        leader_rank: cutlass.Int32,
        gA_tile: cute.Tensor,
        sA: cute.Tensor,
        a_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        """The weight warp: the weights stream without waiting for the previous grid and release nothing."""
        DMA_Stage = self.num_ab_stage
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        sA.element_type, cute.slice_(sA.layout, (None, None, None, 0))
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gA_tile[None, None, k_tile],
                sA[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=a_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def dma_b_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        leader_rank: cutlass.Int32,
        gB_tile: cute.Tensor,
        sB: cute.Tensor,
        b_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        """The token warp: waits for the previous grid, then releases the dependent grid (the poll kernel reads the
        rotation state once resident), then loads the tokens."""
        DMA_Stage = self.num_ab_stage
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        sB.element_type, cute.slice_(sB.layout, (None, None, None, 0))
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gB_tile[None, None, k_tile],
                sB[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=b_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1


@cute.experimental.jit
def _push_entry(
    gemm_op: cutlass.Constexpr,
    a0: cute.Tensor,
    a1: cute.Tensor,
    b: cute.Tensor,
    c0: cute.Tensor,
    c1: cute.Tensor,
    tiles1: cutlass.Constexpr,
    m: Int32,
    mc_base: Int64,
    flags: Int64,
    rank: Int32,
    stream: cuda.CUstream,
):
    a0 = cute.make_tensor(a0.iterator, cute.select(a0.layout, mode=[1, 2, 0]))
    a1 = cute.make_tensor(a1.iterator, cute.select(a1.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c0 = cute.make_tensor(c0.iterator, cute.select(c0.layout, mode=[1, 2, 0]))
    c1 = cute.make_tensor(c1.iterator, cute.select(c1.layout, mode=[1, 2, 0]))
    gemm_op(a0, a1, b, c0, c1, tiles1, m, mc_base, flags, rank, stream)


_PUSH_COMPILED: Dict[Tuple, object] = {}


def _push_compiled(
    world: int,
    x,
    w_latent,
    w_router,
    out_latent,
    out_router,
    config,
    pdl: bool,
):
    cta_n, stages, latent_bounds, router_bounds = config
    kinds = ((w_latent, out_latent, latent_bounds), (w_router, out_router, router_bounds))
    key = (world,) + _ctgv._key(x, kinds, cta_n, stages, pdl)
    compiled = _PUSH_COMPILED.get(key)
    if compiled is None:
        gemm = HeadShardPushGemm(
            world=world,
            cta_n=cta_n,
            num_ab_stage=stages,
            use_pdl=bool(pdl),
            tiles0=out_latent.shape[1] // _ctgv.CTA_M,
            bounds0=latent_bounds,
            bounds1=router_bounds,
        )
        dev, k = x.device, x.shape[1]
        rx = torch.empty(8, k, dtype=x.dtype, device=dev)
        r0 = torch.empty(w_latent.shape[0], k, dtype=w_latent.dtype, device=dev)
        r1 = torch.empty(w_router.shape[0], k, dtype=w_router.dtype, device=dev)
        o0 = torch.empty(8, out_latent.shape[1], dtype=out_latent.dtype, device=dev)
        o1 = torch.empty(8, out_router.shape[1], dtype=out_router.dtype, device=dev)
        a = Int64(1 << 20)
        compiled = cute_ext.compile(
            _push_entry,
            gemm,
            *_ctgv._views(rx, r0, r1, o0, o1),
            out_router.shape[1] // _ctgv.CTA_M,
            Int32(1),
            a,
            a,
            Int32(0),
            make_fake_stream(),
        )
        _PUSH_COMPILED[key] = compiled
    return compiled


def push_serves(config) -> bool:
    """Whether the push GEMM serves this shard config: multi-accumulator rows (the latent sums the whole K in one
    chunk, so one CTA holds every final value)."""
    return not config[2]


def run_head_push_gemm(
    ws: "_lam.LamportWorkspace",
    x: torch.Tensor,
    w_latent: torch.Tensor,
    w_router: torch.Tensor,
    out_latent: torch.Tensor,
    out_router: torch.Tensor,
    config: Tuple[int, int, Tuple[int, ...], Tuple[int, ...]],
    pdl: bool = True,
) -> None:
    """The slice GEMM of :func:`.moe_head_shard.run_head_shard_gemm` (multi-accumulator configs, one plane per
    kind) that also multicasts this rank's slices into the gather's slots of the workspace's current Lamport buffer;
    :func:`run_head_poll_route_quant` must follow it on the stream."""
    if not push_serves(config):
        raise ValueError("the push GEMM serves the multi-accumulator configs only")
    m = x.shape[0]
    if gather_bytes(m, ws.world) > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} head rows")
    lat, rou = out_latent[0], out_router[0]
    compiled = _push_compiled(ws.world, x, w_latent, w_router, lat, rou, config, pdl)
    compiled(
        *_ctgv._views(x, w_latent, w_router, lat, rou),
        Int32(m),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


class MoeHeadPollRouteQuantKernel(MoeHeadRouteQuantKernel):
    """:class:`.moe_head_shard.MoeHeadRouteQuantKernel` for slices the GEMM pushed itself
    (:class:`HeadShardPushGemm`): the same workers, routing and quantization on the polled rows, without the
    push, and the ordering of the module docstring (workers start without waiting; the updater waits for the GEMM)."""

    @cute.kernel
    def kernel(
        self,
        router_in: Int64,
        latent_in: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        xq_addr: Int64,
        xsf_addr: Int64,
        m: Int32,
        rank: Int32,
        route_ctas: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        workers = route_ctas + (m + (self.quant_rows - 1)) // self.quant_rows
        arrivals = _lam.arrivals_addr(flags)
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bias_smem = storage.scratch.data_ptr().toint()
        stage = storage.stage.data_ptr().toint()
        routes = cta < route_ctas
        if routes:
            for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
                v = _ld_nc_v4_f32(bias_addr + Int64(e4 * 16))
                _sts_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        if cta == workers:
            # The updater alone waits for the GEMM: its CTAs read the state after their own wait, so the state may
            # only change once it has completed.
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
                _lam.griddepcontrol_wait()
            if tidx == 0:
                cur, _, nbytes, _, _, _, _, _ = _lam.lamport_state(flags)
                arrived = _lam.ld_acquire_u32(arrivals)
                while arrived < workers:
                    arrived = _lam.ld_acquire_u32(arrivals)
                _lam.st_v4(flags, (cur + 1) % _lam.NUM_BUFFERS, cur, nbytes, Int32(1))
                _lam.st_v4(flags + Int64(16), m * (self.g.vecs * 16), Int32(0), Int32(0), Int32(0))
                _lam.st_u32(arrivals, Int32(0))
        else:
            # Resident only once the GEMM's token warps have passed their wait: the state is this call's.
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(flags)
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            cute.arch.barrier()
            if tidx == 0:
                _lam.red_release_add(arrivals, Int32(1))
            # Re-arm the previous call's buffer (every stage it used) while the GEMM still runs.
            _lam.clear_dirty_stages(
                uc_local,
                dirty,
                nbytes,
                dstages,
                c0,
                c1,
                c2,
                c3,
                cta * self.threads + tidx,
                workers * self.threads,
            )
            uc = _lam.stage_addr(uc_local, cur, Int32(0), 1, nbytes)
            if routes:
                t_begin = cta * self.tokens
                t_end = t_begin + self.tokens
                if t_end > m:
                    t_end = m
                self.gather_logits(uc, t_begin, t_end, stage, tidx)
                cute.arch.barrier()
                cand_all = bias_smem + _WIDE_BIAS_WORDS * 4
                top_all = cand_all + self.cand_words * 4
                team = warp
                cand_smem = cand_all + team * (self.region_slots * 16)
                top_smem = top_all + team * (TOP_K * 8)
                logits = Int64(stage) - Int64(t_begin) * Int64(NUM_EXPERTS * 4)
                for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
                    expert, weight = self.select_token(
                        token,
                        lane,
                        logits,
                        bias_smem,
                        Int64(_UNIT_SCALE_BITS),
                        cand_smem,
                        top_smem,
                    )
                    if lane < TOP_K:
                        out = token * TOP_K + lane
                        _st_global_b32(ids_addr + Int64(out * 4), expert)
                        _st_global_b16(vals_addr + Int64(out * 2), weight)
            else:
                r_begin = (cta - route_ctas) * self.quant_rows
                r_end = r_begin + self.quant_rows
                if r_end > m:
                    r_end = m
                rcp448 = _rcp_approx_ftz(cutlass.Float32(448.0))
                self.quant_block(uc, r_begin, r_end, xq_addr, xsf_addr, rcp448, tidx)


@functools.cache
def _compiled_poll(world: int, router_pad: int, latent_pad: int, use_pdl: bool):
    kernel = MoeHeadPollRouteQuantKernel(world, router_pad, latent_pad, use_pdl, 1, 1)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        *([a] * 10),
        Int32(1),
        Int32(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def compile_head_push(
    world: int,
    x: torch.Tensor,
    w_latent: torch.Tensor,
    w_router: torch.Tensor,
    out_latent: torch.Tensor,
    out_router: torch.Tensor,
    config,
    use_pdl: bool,
) -> None:
    """Compile the push GEMM and its poll kernel for this config and these planes (before CUDA-graph capture),
    without launching them."""
    _push_compiled(
        world, x, w_latent, w_router, out_latent[0], out_router[0], config, bool(use_pdl)
    )
    rp, lp, _, _ = _gather_args(out_router, out_latent)
    _compiled_poll(world, rp, lp, bool(use_pdl))


def run_head_poll_route_quant(
    ws: "_lam.LamportWorkspace",
    out_router: torch.Tensor,
    out_latent: torch.Tensor,
    bias: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    use_pdl: bool = True,
) -> None:
    """:func:`.moe_head_shard.run_head_route_quant`'s outputs from the slices :func:`run_head_push_gemm` pushed
    (right before it on the stream)."""
    m = topk_ids.shape[0]
    rp, lp, _, _ = _gather_args(out_router, out_latent)
    _compiled_poll(ws.world, rp, lp, bool(use_pdl))(
        Int64(0),
        Int64(0),
        Int64(ws.uc_local),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int64(bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(x_fp8.data_ptr()),
        Int64(x_sf.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


__all__ = [
    "HeadShardPushGemm",
    "MoeHeadPollRouteQuantKernel",
    "compile_head_push",
    "push_serves",
    "run_head_poll_route_quant",
    "run_head_push_gemm",
    "LATENT_ROWS",
    "ROUTER_ROWS",
]
