# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0; upstream notice: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.):
# https://github.com/flashinfer-ai/flashinfer/blob/711d3bc94e6d2130af8607320eec612899600084/flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
"""Low-M BF16 projection whose epilogue pushes its output into the next MNNVL all-reduce (SM100 / SM103).

``out[M, N] = x[M, K] @ W[N, K]^T`` for a row-parallel BF16 projection followed by an MNNVL all-reduce (the Kimi K3.1
attention ``o_proj``): instead of storing ``out``, the owner CTA of each output tile stores its BF16 rows into the
Lamport slots of the all-reduce (:mod:`k3_lamport_ar`), each token row to its owner's scatter slot (two-shot) or to
every rank's slot (one-shot, multicast), with ``-0`` sent as ``+0`` as the C++ all-reduce sends it. The consumer then
finishes the all-reduce (``k3_lamport_ar.run_lamport_finish`` or the attention-residual mix with ``lamport_finish``):
the GEMM output reaches the network without a store, a reload and a copy kernel.

The GEMM is the swap-AB tcgen05 split-K kernel of the fused MoE tail (``kimi_k3_moe_tail.splitk_multicast_gemm``):
kernel-M carries the output columns (weight rows), kernel-N the tokens, each cluster rank accumulates an exact K slice
in FP32 and the peers publish their partials to rank 0 through DSMEM; rank 0 adds them in rank order and rounds once
to BF16. The weight loads do not wait for the producer grid (PDL); the token loads do. The tactic table covers the
row counts where this GEMM + the finish beats cuBLAS + the C++ all-reduce (TP8 up to 72 rows, TP16 up to 128).

Ordering (the rules of :mod:`k3_lamport_ar`): the epilogue reads the workspace's rotation state after its own
``griddepcontrol.wait``, and the grid releases its dependent only from the token producer, right after that warp's
wait, so a consumer that reads the state as soon as it is resident sees this call's state.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

from ...modules.kimi_k3_moe_tail.splitk_multicast_gemm import (
    _AB_BUFFER_ALIGN_BYTES,
    _CTA_K,
    _EPILOG_BARRIER_ID,
    _FP32_BYTES,
    _MAILBOX_ALIGN_BYTES,
    _MBARRIER_BYTES,
    _SMEM_CAPACITY_BYTES,
    _TMEM_POINTER_BYTES,
    _TRANSPOSE_ALIGN_BYTES,
    OWNER_RANK,
    SplitKTactic,
    _align_up,
    _load_shared_u32x4,
    _max_ab_stages_for,
    _smem_bytes,
    _store_shared_remote_v4,
)
from . import k3_lamport_ar as _lam

#: Largest token count the kernel serves.
MAX_M = 128


class Bf16ArPushGemmKernel:
    """BF16 GEMM with a cluster-local split-K reduction whose epilogue pushes the BF16 output
    into the Lamport slots of the next MNNVL all-reduce (see the module docstring)."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        hidden_dim: int,
        world: int,
        one_shot: bool,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.world = world
        self.one_shot = one_shot
        self.smem_bytes = min(
            _align_up(_smem_bytes(tactic, tactic.ab_stages), 1024) + 1024, _SMEM_CAPACITY_BYTES
        )
        self.acc_dtype = cutlass.Float32
        self.cta_m = tactic.mma_m
        self.cta_n = tactic.mma_n
        self.cta_k = _CTA_K
        self.num_ab_stage = tactic.ab_stages
        self.split_k = tactic.split_k
        self.use_pdl = use_pdl
        self.has_bias = False

        self.threads_per_cta = 256
        self.epilog_threads = 128
        self.mma_tiler_mn = (tactic.mma_m, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.cluster_shape = (1, tactic.split_k, 1)

        values_per_thread = (tactic.mma_m * tactic.mma_n) // self.epilog_threads
        if values_per_thread % 4:
            raise ValueError(
                f"CTA tile ({tactic.mma_m}, {tactic.mma_n}) gives "
                f"{values_per_thread} "
                "values per epilogue thread; remote stores require a multiple of 4"
            )
        self.mailbox_elements = (tactic.split_k - 1) * self.epilog_threads * values_per_thread
        self.expected_transaction_bytes = self.mailbox_elements * _FP32_BYTES

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        peer_ptrs: cutlass.Int64,
        mc_base: cutlass.Int64,
        flags: cutlass.Int64,
        rank: cutlass.Int32,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, peer_ptrs, mc_base, flags, rank).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            # The tactic's own footprint (not the whole SMEM) so small tiles
            # can run two CTAs per SM.
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major: only its tiling is used
        peer_ptrs: cutlass.Int64,
        mc_base: cutlass.Int64,
        flags: cutlass.Int64,
        rank: cutlass.Int32,
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator
        # Column-contiguous output tile: lets the owner store 16-byte vectors
        # (the TMEM partition gives each thread one column of many tokens).
        staging = cute_ext.allocate(
            mC.element_type,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
            alignment=_TRANSPOSE_ALIGN_BYTES,
        )

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                staging,
                cute.size(mC, mode=[1]),
                n_idx,
                bidx,
                peer_ptrs,
                mc_base,
                flags,
                rank,
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):
            cute.arch.griddepcontrol_wait()
            # The dependent (the consumer that finishes the all-reduce) reads the workspace's
            # rotation state as soon as it is resident: release it only once this grid has
            # passed its own wait, i.e. every earlier grid (the previous call included) is done.
            cute.arch.griddepcontrol_launch_dependents()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type,
                        cute.slice_(s_tile.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        self._drain_producer(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def _drain_producer(
        self,
        bar_empty,
        empty_phase: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self,
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
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        staging: cute.Tensor,
        rows: cutlass.Int32,
        n_idx: cutlass.Int32,
        m_idx: cutlass.Int32,
        peer_ptrs: cutlass.Int64,
        mc_base: cutlass.Int64,
        flags: cutlass.Int64,
        rank: cutlass.Int32,
    ):
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1):
            assert cute.size(rmem_layout) == self.mailbox_elements // (
                (self.split_k - 1) * self.epilog_threads
            )
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            # Round to BF16 (push_v4 sends -0 as +0, as the MNNVL all-reduce's copy phase does).
            rD.store(rAcc.load().to(c_dtype))
            cute_ext.partition_and_copy(thr_t2r, rD, staging)
            cute.arch.barrier(barrier_id=_EPILOG_BARRIER_ID, number_of_threads=self.epilog_threads)
            # The rotation state of the workspace is this call's once every earlier grid is done.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            cur, _d, nbytes, _s, _c0, _c1, _c2, _c3 = _lam.lamport_state(flags)
            chunks_per_token = self.cta_m // 8
            total_chunks = self.cta_n * chunks_per_token
            for i in cutlass.range_constexpr(
                (total_chunks + self.epilog_threads - 1) // self.epilog_threads
            ):
                chunk = epi_tid + Int32(i * self.epilog_threads)
                token = chunk // Int32(chunks_per_token)
                column = (chunk - token * Int32(chunks_per_token)) * Int32(8)
                tok = n_idx * Int32(self.cta_n) + token
                if chunk < Int32(total_chunks) and tok < rows:
                    packed = _load_shared_u32x4(
                        staging.iterator + token * Int32(self.cta_m) + column
                    )
                    dst = _lam.push_slot(
                        peer_ptrs,
                        mc_base,
                        cur,
                        nbytes,
                        tok,
                        rank,
                        m_idx * Int32(self.cta_m) + column,
                        self.world,
                        self.hidden_dim,
                        self.one_shot,
                    )
                    _lam.push_v4(dst, packed[0], packed[1], packed[2], packed[3], self.one_shot)

        # The reduction mbarrier covers remote stores; no cluster barrier needed.


#: Measured tactic per (N, K) and token count (``(max_m, tactic)`` rows, first row whose bound covers M), GB300,
#: 24-layer graph of GEMM + all-reduce with distinct weights, us per layer (cuBLAS + C++ MNNVL all-reduce -> this GEMM
#: + the Lamport finish). Past the last bound cuBLAS + the copy-kernel push stays faster.
_TACTICS = {
    # Kimi K3.1 KDA / MLA o_proj, TP8 (K = 12288 / 8): M=7 9.33 -> 8.07, M=28 11.59 -> 9.58, M=48 13.32 -> 10.73,
    # M=64 13.41 -> 11.19, M=72 13.90 -> 12.89; M=84 13.20 vs 13.43 (loses from there).
    (7168, 1536): (
        (8, SplitKTactic(128, 8, 2, 6)),
        (16, SplitKTactic(128, 16, 2, 6)),
        (32, SplitKTactic(64, 32, 1, 9)),
        (64, SplitKTactic(64, 64, 1, 6)),
        (72, SplitKTactic(64, 128, 1, 4)),
    ),
    # TP16 (K = 12288 / 16): M=7 9.20 -> 7.43, M=28 10.45 -> 8.44, M=48 11.10 -> 9.24, M=64 11.43 -> 9.82,
    # M=84 12.33 -> 11.70, M=112 15.97 -> 12.48, M=128 16.41 -> 13.22.
    (7168, 768): (
        (8, SplitKTactic(128, 8, 1, 6)),
        (16, SplitKTactic(128, 16, 1, 6)),
        (32, SplitKTactic(64, 32, 1, 7)),
        (64, SplitKTactic(64, 64, 1, 6)),
        (128, SplitKTactic(128, 64, 1, 3)),
    ),
    # TP32 (K = 12288 / 32: three K tiles), GEMM alone with cold weights (us per call, cuBLAS -> this tactic): M=6
    # 2.87 -> 2.39, M=16 3.00 -> 2.48, M=32 3.21 -> 2.69; cuBLAS is faster from 48 rows.
    (7168, 384): (
        (8, SplitKTactic(64, 8, 1, 3)),
        (16, SplitKTactic(64, 16, 1, 3)),
        (32, SplitKTactic(64, 32, 1, 3)),
    ),
}


def tactic_for(m: int, n: int, k: int) -> Optional[SplitKTactic]:
    """The tactic for ``m`` rows of an ``[N, K]`` weight, or ``None`` where the kernel is not tuned."""
    rows = _TACTICS.get((n, k))
    if rows is None or not 1 <= m <= min(MAX_M, rows[-1][0]):
        return None
    tactic = next(t for bound, t in rows if m <= bound)
    stages = min(tactic.ab_stages, _max_ab_stages_for(tactic, _SMEM_CAPACITY_BYTES))
    tactic = dataclasses.replace(tactic, ab_stages=stages)
    if stages < 1 or (k // _CTA_K) % tactic.split_k or k % _CTA_K:
        return None
    return tactic


@cute.experimental.jit
def _launch(
    gemm_op: cutlass.Constexpr,
    weight: cute.Tensor,
    tokens: cute.Tensor,
    out: cute.Tensor,
    rows: Int32,
    peer_ptrs: cutlass.Int64,
    mc_base: cutlass.Int64,
    flags: cutlass.Int64,
    rank: Int32,
    hidden_dim: cutlass.Constexpr,
    stream: _cuda.CUstream,
):
    """``weight`` [N, K] and ``tokens`` [M, K] K-major; ``out`` [M, N] gives the output tiling only."""
    k = cute.size(weight, mode=[1])
    m_a = cute.make_tensor(weight.iterator, cute.make_layout((hidden_dim, k, 1), stride=(k, 1, 0)))
    m_b = cute.make_tensor(tokens.iterator, cute.make_layout((rows, k, 1), stride=(k, 1, 0)))
    m_c = cute.make_tensor(
        out.iterator, cute.make_layout((hidden_dim, rows, 1), stride=(1, hidden_dim, 0))
    )
    gemm_op(m_a, m_b, m_c, peer_ptrs, mc_base, flags, rank, stream)


@functools.cache
def _compiled(
    tactic: SplitKTactic,
    n: int,
    k: int,
    world: int,
    one_shot: bool,
    use_pdl: bool,
    device_index: int,
):
    device = torch.device("cuda", device_index)
    kernel = Bf16ArPushGemmKernel(
        tactic=tactic, use_pdl=use_pdl, hidden_dim=n, world=world, one_shot=one_shot
    )
    weight = torch.empty((n, k), dtype=torch.bfloat16, device=device)
    tokens = torch.empty((MAX_M, k), dtype=torch.bfloat16, device=device)
    out = torch.empty((MAX_M, n), dtype=torch.bfloat16, device=device)
    return cute_ext.compile(
        _launch,
        kernel,
        from_dlpack(weight, assumed_align=32),
        from_dlpack(tokens, assumed_align=32).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        ),
        from_dlpack(out, assumed_align=32).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1)),
        Int32(MAX_M),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        Int32(0),
        n,
        _cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )


def can_push(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`run_bf16_gemm_ar_push` serves ``x [M, K] @ weight[N, K]^T`` (BF16, contiguous, a tuned shape)."""
    if not (x.is_cuda and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16):
        return False
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if torch.cuda.get_device_capability(x.device) not in ((10, 0), (10, 3)):
        return False
    return tactic_for(x.shape[0], weight.shape[0], weight.shape[1]) is not None


def run_bf16_gemm_ar_push(
    x: torch.Tensor,
    weight: torch.Tensor,
    workspace: "_lam.LamportWorkspace",
    one_shot: bool,
    out: torch.Tensor,
    use_pdl: bool = True,
    tactic: Optional[SplitKTactic] = None,
) -> None:
    """Push ``x @ weight^T`` (BF16 ``[M, N]``) into the Lamport slots of the next all-reduce on
    ``workspace`` (two-shot or ``one_shot`` transport); ``out`` (``[M, N]`` BF16, not written) is the
    all-reduce's output tensor, whose shape the kernel tiles. Check :func:`can_push` first.
    ``tactic`` overrides the table (for tuning)."""
    m, k = x.shape
    n = weight.shape[0]
    if tactic is None:
        tactic = tactic_for(m, n, k)
    if tactic is None or not 1 <= m <= MAX_M:
        raise ValueError(f"no tactic for M={m} N={n} K={k}")
    _lam._check(workspace, m, n, one_shot)
    compiled = _compiled(
        tactic, n, k, workspace.world, bool(one_shot), bool(use_pdl), x.device.index
    )
    compiled(
        from_dlpack(weight.detach(), assumed_align=32),
        from_dlpack(x, assumed_align=32).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1)),
        from_dlpack(out, assumed_align=32).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1)),
        Int32(m),
        cutlass.Int64(workspace.peer_ptrs.data_ptr()),
        cutlass.Int64(workspace.mc_base),
        cutlass.Int64(workspace.flags.data_ptr()),
        Int32(workspace.rank),
        _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


__all__ = ["MAX_M", "can_push", "run_bf16_gemm_ar_push", "tactic_for"]
