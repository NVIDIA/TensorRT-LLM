# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fused Sol-Attn forward kernel for GeForce Blackwell SM120.

The warp-MMA/TMA execution skeleton and online-softmax helpers are adapted
from NVIDIA cuDNN Frontend's SM120 block-sparse-attention kernel.  Sol-specific
routing, CTA-local exact-index compaction, approximate block mass, and the
mixed approximate/exact mainloop are implemented here.
"""

from __future__ import annotations

import operator

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

from sol_attn._vendor.flash_attn.cute import utils as kernel_utils
from sol_attn.common import layout_utils
from sol_attn.common.selector import (
    sol_attn_popc_b32,
    sol_attn_route_is_exact,
)


M = 64
N = 64
D = 128
DV = 128
THREADS = 128
STAGES = 1


class SolAttnForwardSm120:
    """M64/N64 warp-MMA Sol-Attn kernel for BF16 D128 inputs."""

    def __init__(
        self,
        *,
        debug_route_trace: bool = False,
        prefetch_first_exact_k: bool = True,
        prefetch_next_route_k: bool = True,
    ):
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.tile_shape_qk = (M, N, D)
        self.tile_shape_pv = (M, DV, N)
        self.num_threads = THREADS
        self.q_stage = 1
        self.kv_stage = STAGES
        self.debug_route_trace = debug_route_trace
        self.prefetch_first_exact_k = prefetch_first_exact_k
        self.prefetch_next_route_k = prefetch_next_route_k

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mKC: cute.Tensor,
        mVC: cute.Tensor,
        mThreshold: cute.Tensor,
        mLSE: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_KC: cute.CopyAtom,
        tma_atom_VC: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        O_smem_layout: cute.ComposedLayout,
        scale_softmax_log2e: cutlass.Float32,
        sink_start_block: cutlass.Int32,
        sink_end_block: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_tile_idx, head_idx, batch_idx = cute.arch.block_idx()
        q_tile_idx = cute.arch.make_warp_uniform(q_tile_idx)
        head_idx = cute.arch.make_warp_uniform(head_idx)
        batch_idx = cute.arch.make_warp_uniform(batch_idx)

        token_count = mK.shape[0]
        num_blocks = mKC.shape[0]
        num_route_groups = cute.ceil_div(num_blocks, N)
        q_start = q_tile_idx * M
        q_len = token_count - q_start
        if q_len > M:
            q_len = cutlass.Int32(M)
        threshold = cutlass.Float32(
            mThreshold[batch_idx, q_tile_idx, head_idx]
        )

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage_t)
        if warp == 0 and lane == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_KC)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_VC)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_threads // 32
        )
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        Q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stage,
            producer_group=cg,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self.Q_dtype, cute.select(Q_smem_layout, mode=[0, 1])
            ),
            barrier_storage=storage.Q_barrier.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        K_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self.K_dtype, cute.select(K_smem_layout, mode=[0, 1])
            ),
            barrier_storage=storage.K_barrier.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        V_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self.V_dtype, cute.select(V_smem_layout, mode=[0, 1])
            ),
            barrier_storage=storage.V_barrier.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        Q_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        Q_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        K_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        K_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        V_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        V_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )

        sQ = storage.Q_smem.get_tensor(
            Q_smem_layout.outer, swizzle=Q_smem_layout.inner
        )
        sK = storage.K_smem.get_tensor(
            K_smem_layout.outer, swizzle=K_smem_layout.inner
        )
        sV = storage.V_smem.get_tensor(
            V_smem_layout.outer, swizzle=V_smem_layout.inner
        )
        # Q is register-resident after the prologue.  Reuse its 16 KiB SMEM
        # allocation for route scratch until the same allocation becomes sO
        # in the epilogue.  This drops the CTA below the 2-block/SM threshold
        # on SM120 without changing any route reduction or synchronization.
        route_f32_ptr = cute.recast_ptr(
            storage.Q_smem.data_ptr(), dtype=cutlass.Float32
        )
        route_i32_ptr = cute.recast_ptr(
            storage.Q_smem.data_ptr(), dtype=cutlass.Int32
        )
        route_sums = cute.make_tensor(
            route_f32_ptr, cute.make_layout((4, N))
        )
        column_masks = cute.make_tensor(
            route_f32_ptr + 4 * N, cute.make_layout(N)
        )
        route_indices = cute.make_tensor(
            route_i32_ptr + 5 * N, cute.make_layout(N)
        )
        route_meta = cute.make_tensor(
            route_i32_ptr + 6 * N, cute.make_layout(2)
        )

        mQ_slice = mQ[None, None, head_idx, batch_idx]
        mK_slice = mK[None, None, head_idx, batch_idx]
        mV_slice = mV[None, None, head_idx, batch_idx]
        mO_slice = mO[None, None, head_idx, batch_idx]
        mKC_slice = mKC[None, None, head_idx, batch_idx]
        mVC_slice = mVC[None, None, head_idx, batch_idx]
        if cutlass.const_expr(not self.debug_route_trace):
            mLSE_slice = mLSE[None, head_idx, batch_idx]

        gQ = cute.local_tile(
            mQ_slice, (M, D), coord=(q_tile_idx, 0)
        )
        gK = cute.local_tile(mK_slice, (N, D), coord=(None, 0))
        gV = cute.local_tile(mV_slice, (DV, N), coord=(0, None))
        gKC = cute.local_tile(mKC_slice, (N, D), coord=(None, 0))
        gVC = cute.local_tile(mVC_slice, (DV, N), coord=(0, None))
        gO = cute.local_tile(
            mO_slice, (M, DV), coord=(q_tile_idx, 0)
        )

        cta_coord_layout = (0, cute.make_layout(1))
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            *cta_coord_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            *cta_coord_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            *cta_coord_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )
        tKCsK, tKCgKC = cute.nvgpu.cpasync.tma_partition(
            tma_atom_KC,
            *cta_coord_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gKC, 0, 2),
        )
        tVCsV, tVCgVC = cute.nvgpu.cpasync.tma_partition(
            tma_atom_VC,
            *cta_coord_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gVC, 0, 2),
        )

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSsQ = thr_mma_qk.partition_A(sQ)
        tSsK = thr_mma_qk.partition_B(sK)
        tSrQ = tiled_mma_qk.make_fragment_A(tSsQ[None, None, None, 0])
        tSrK = tiled_mma_qk.make_fragment_B(tSsK[None, None, None, 0])
        tSrS = cute.make_rmem_tensor(
            thr_mma_qk.partition_shape_C((M, N)), self.acc_dtype
        )
        tScS = thr_mma_qk.partition_C(cS)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOsV = thr_mma_pv.partition_B(sV)
        tOrV = tiled_mma_pv.make_fragment_B(tOsV[None, None, None, 0])
        tOrO = cute.make_rmem_tensor(
            thr_mma_pv.partition_shape_C((M, DV)), self.acc_dtype
        )

        atom_copy_Q = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.Q_layout.is_m_major_a(), 4
            ),
            self.Q_dtype,
        )
        atom_copy_K = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.K_layout.is_n_major_b(), 4
            ),
            self.K_dtype,
        )
        atom_copy_V = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.V_layout.is_n_major_b(), 4
            ),
            self.V_dtype,
        )
        smem_copy_Q = cute.make_tiled_copy_A(atom_copy_Q, tiled_mma_qk)
        smem_copy_K = cute.make_tiled_copy_B(atom_copy_K, tiled_mma_qk)
        smem_copy_V = cute.make_tiled_copy_B(atom_copy_V, tiled_mma_pv)
        thr_copy_Q = smem_copy_Q.get_slice(tidx)
        thr_copy_K = smem_copy_K.get_slice(tidx)
        thr_copy_V = smem_copy_V.get_slice(tidx)
        tSsQ_copy = thr_copy_Q.partition_S(sQ)
        tSrQ_copy = thr_copy_Q.retile(tSrQ)
        tSsK_copy = thr_copy_K.partition_S(sK)
        tOsV_copy = thr_copy_V.partition_S(sV)

        max_m_layout = cute.make_layout(
            cute.size(
                layout_utils.reshape_acc_to_mn(tOrO).layout,
                mode=[0],
            )
        )
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        if warp == 0:
            Q_pipeline.producer_acquire(Q_producer)
            cute.copy(
                tma_atom_Q,
                tQgQ,
                tQsQ[None, Q_producer.index],
                tma_bar_ptr=Q_pipeline.producer_get_barrier(Q_producer),
            )
            Q_pipeline.producer_commit(Q_producer)
            Q_producer.advance()
        cute.arch.sync_threads()
        q_wait = Q_pipeline.consumer_try_wait(Q_consumer)
        Q_pipeline.consumer_wait(Q_consumer, q_wait)
        q_stage = Q_consumer.index
        for k_block in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
            cute.copy(
                smem_copy_Q,
                tSsQ_copy[None, None, k_block, q_stage],
                tSrQ_copy[None, None, k_block],
            )
        Q_pipeline.consumer_release(Q_consumer)
        Q_consumer.advance()

        for route_group in cutlass.range(
            0, num_route_groups, 1, unroll=1
        ):
            group_start = route_group * cutlass.Int32(N)
            valid_blocks = num_blocks - group_start
            if valid_blocks > N:
                valid_blocks = cutlass.Int32(N)

            if warp == 0:
                if cutlass.const_expr(self.prefetch_next_route_k):
                    # P19-style terminal handoff: when the previous route
                    # group had an exact block, its final exact QK already
                    # refilled this K stage with the current group's KC.
                    if route_group == 0:
                        K_pipeline.producer_acquire(K_producer)
                        cute.copy(
                            tma_atom_KC,
                            tKCgKC[None, route_group],
                            tKCsK[None, K_producer.index],
                            tma_bar_ptr=K_pipeline.producer_get_barrier(
                                K_producer
                            ),
                        )
                        K_pipeline.producer_commit(K_producer)
                        K_producer.advance()
                    else:
                        previous_group_exact_count = cutlass.Int32(
                            route_meta[0]
                        )
                        if previous_group_exact_count == 0:
                            K_pipeline.producer_acquire(K_producer)
                            cute.copy(
                                tma_atom_KC,
                                tKCgKC[None, route_group],
                                tKCsK[None, K_producer.index],
                                tma_bar_ptr=K_pipeline.producer_get_barrier(
                                    K_producer
                                ),
                            )
                            K_pipeline.producer_commit(K_producer)
                            K_producer.advance()
                else:
                    K_pipeline.producer_acquire(K_producer)
                    cute.copy(
                        tma_atom_KC,
                        tKCgKC[None, route_group],
                        tKCsK[None, K_producer.index],
                        tma_bar_ptr=K_pipeline.producer_get_barrier(
                            K_producer
                        ),
                    )
                    K_pipeline.producer_commit(K_producer)
                    K_producer.advance()
                V_pipeline.producer_acquire(V_producer)
                cute.copy(
                    tma_atom_VC,
                    tVCgVC[None, route_group],
                    tVCsV[None, V_producer.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer),
                )
                V_pipeline.producer_commit(V_producer)
                V_producer.advance()

            k_wait = K_pipeline.consumer_try_wait(K_consumer)
            K_pipeline.consumer_wait(K_consumer, k_wait)
            gemm_smem_zero_acc(
                tiled_mma_qk,
                tSrS,
                tSrQ,
                tSrK,
                tSsK_copy[None, None, None, K_consumer.index],
                smem_copy_K,
            )
            K_pipeline.consumer_release(K_consumer)
            K_consumer.advance()

            reduce_route_columns(
                tSrS,
                tScS,
                route_sums,
                warp,
                lane,
                q_len,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            if warp == 0:
                preceding = cutlass.Int32(0)
                lane_mask_lt = cutlass.Int32(0x7FFFFFFF) >> (
                    cutlass.Int32(31) - lane
                )
                for word in cutlass.range_constexpr(2):
                    off = cutlass.Int32(word * 32) + lane
                    valid = off < valid_blocks
                    exact = False
                    if valid:
                        col_sum = (
                            cutlass.Float32(route_sums[0, off])
                            + cutlass.Float32(route_sums[1, off])
                            + cutlass.Float32(route_sums[2, off])
                            + cutlass.Float32(route_sums[3, off])
                        )
                        col_mean = (
                            col_sum
                            * scale_softmax_log2e
                            / cutlass.Float32(q_len)
                        )
                        kv_block = group_start + off
                        exact = sol_attn_route_is_exact(
                            q_tile_idx,
                            kv_block,
                            col_mean,
                            threshold,
                            valid,
                        )
                        exact = exact or (
                            kv_block >= sink_start_block
                            and kv_block < sink_end_block
                        )
                    ballot = cutlass.Int32(
                        cute.arch.vote_ballot_sync(exact)
                    )
                    column_masks[off] = (
                        -cutlass.Float32.inf
                        if (exact or not valid)
                        else cutlass.Float32(0.0)
                    )
                    rank = preceding + sol_attn_popc_b32(
                        ballot & lane_mask_lt
                    )
                    if exact:
                        route_indices[rank] = group_start + off
                    preceding += sol_attn_popc_b32(ballot)
                    if cutlass.const_expr(self.debug_route_trace):
                        if lane == 0:
                            mLSE[
                                batch_idx,
                                q_tile_idx,
                                head_idx,
                                route_group,
                                word,
                            ] = ballot
                if lane == 0:
                    route_meta[0] = preceding
                    route_meta[1] = valid_blocks
                    cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            exact_count = cutlass.Int32(route_meta[0])
            has_approx = exact_count < valid_blocks
            if cutlass.const_expr(self.prefetch_first_exact_k):
                # Once routing identifies the first exact block, the route KC
                # stage is free.  Refill it before the approximate softmax/PV
                # so the first exact K transfer overlaps that work.
                if warp == 0 and exact_count > 0:
                    first_exact = cutlass.Int32(route_indices[0])
                    K_pipeline.producer_acquire(K_producer)
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, first_exact],
                        tKsK[None, K_producer.index],
                        tma_bar_ptr=K_pipeline.producer_get_barrier(
                            K_producer
                        ),
                    )
                    K_pipeline.producer_commit(K_producer)
                    K_producer.advance()
            v_wait = V_pipeline.consumer_try_wait(V_consumer)
            V_pipeline.consumer_wait(V_consumer, v_wait)
            if has_approx:
                apply_route_mask(tSrS, tScS, column_masks, q_len)
                row_scale = online_softmax_route(
                    tSrS,
                    tScS,
                    max_m,
                    sum_m,
                    scale_softmax_log2e,
                    group_start,
                    token_count,
                )
                rescale_o_for_next_acc(tOrO, row_scale)
                tOrP_frg = cute.make_rmem_tensor_like(
                    tSrS, self.K_dtype
                )
                tOrP_frg.store(tSrS.load().to(self.K_dtype))
                tOrP = layout_utils.reshape_acc_to_frgA(tOrP_frg)
                gemm_rs_smem(
                    tiled_mma_pv,
                    tOrO,
                    tOrP,
                    tOrV,
                    tOsV_copy[None, None, None, V_consumer.index],
                    smem_copy_V,
                )
            V_pipeline.consumer_release(V_consumer)
            V_consumer.advance()

            if warp == 0 and exact_count > 0:
                first_exact = cutlass.Int32(route_indices[0])
                if cutlass.const_expr(not self.prefetch_first_exact_k):
                    K_pipeline.producer_acquire(K_producer)
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, first_exact],
                        tKsK[None, K_producer.index],
                        tma_bar_ptr=K_pipeline.producer_get_barrier(
                            K_producer
                        ),
                    )
                    K_pipeline.producer_commit(K_producer)
                    K_producer.advance()
                V_pipeline.producer_acquire(V_producer)
                cute.copy(
                    tma_atom_V,
                    tVgV[None, first_exact],
                    tVsV[None, V_producer.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer),
                )
                V_pipeline.producer_commit(V_producer)
                V_producer.advance()

            for ordinal in cutlass.range(0, exact_count, 1, unroll=1):
                exact_block = cutlass.Int32(route_indices[ordinal])
                k_wait = K_pipeline.consumer_try_wait(K_consumer)
                K_pipeline.consumer_wait(K_consumer, k_wait)
                gemm_smem_zero_acc(
                    tiled_mma_qk,
                    tSrS,
                    tSrQ,
                    tSrK,
                    tSsK_copy[None, None, None, K_consumer.index],
                    smem_copy_K,
                )
                K_pipeline.consumer_release(K_consumer)
                K_consumer.advance()
                next_ordinal = ordinal + cutlass.Int32(1)
                if warp == 0:
                    if next_ordinal < exact_count:
                        next_exact = cutlass.Int32(
                            route_indices[next_ordinal]
                        )
                        K_pipeline.producer_acquire(K_producer)
                        cute.copy(
                            tma_atom_K,
                            tKgK[None, next_exact],
                            tKsK[None, K_producer.index],
                            tma_bar_ptr=K_pipeline.producer_get_barrier(
                                K_producer
                            ),
                        )
                        K_pipeline.producer_commit(K_producer)
                        K_producer.advance()
                    else:
                        if cutlass.const_expr(
                            self.prefetch_next_route_k
                        ):
                            next_route_group = route_group + cutlass.Int32(1)
                            if next_route_group < num_route_groups:
                                # Reuse the K stage released by the final
                                # exact QK.  The next outer prologue supplies
                                # VC, matching the SM90 P19 partial handoff.
                                K_pipeline.producer_acquire(K_producer)
                                cute.copy(
                                    tma_atom_KC,
                                    tKCgKC[None, next_route_group],
                                    tKCsK[None, K_producer.index],
                                    tma_bar_ptr=(
                                        K_pipeline.producer_get_barrier(
                                            K_producer
                                        )
                                    ),
                                )
                                K_pipeline.producer_commit(K_producer)
                                K_producer.advance()
                block_len = token_count - exact_block * cutlass.Int32(N)
                if block_len > N:
                    block_len = cutlass.Int32(N)
                mask_exact_scores(tSrS, tScS, block_len, q_len)
                row_scale = online_softmax(
                    tSrS, max_m, sum_m, scale_softmax_log2e
                )
                rescale_o_for_next_acc(tOrO, row_scale)
                tOrP_frg = cute.make_rmem_tensor_like(
                    tSrS, self.K_dtype
                )
                tOrP_frg.store(tSrS.load().to(self.K_dtype))
                tOrP = layout_utils.reshape_acc_to_frgA(tOrP_frg)

                v_wait = V_pipeline.consumer_try_wait(V_consumer)
                V_pipeline.consumer_wait(V_consumer, v_wait)
                gemm_rs_smem(
                    tiled_mma_pv,
                    tOrO,
                    tOrP,
                    tOrV,
                    tOsV_copy[None, None, None, V_consumer.index],
                    smem_copy_V,
                )
                V_pipeline.consumer_release(V_consumer)
                V_consumer.advance()
                if warp == 0 and next_ordinal < exact_count:
                    next_exact = cutlass.Int32(route_indices[next_ordinal])
                    V_pipeline.producer_acquire(V_producer)
                    cute.copy(
                        tma_atom_V,
                        tVgV[None, next_exact],
                        tVsV[None, V_producer.index],
                        tma_bar_ptr=V_pipeline.producer_get_barrier(
                            V_producer
                        ),
                    )
                    V_pipeline.producer_commit(V_producer)
                    V_producer.advance()

        final_ratio, lse = finalize_softmax(
            max_m, sum_m, scale_softmax_log2e
        )
        rescale_o_for_next_acc(tOrO, final_ratio)
        if cutlass.const_expr(not self.debug_route_trace):
            tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
            for m in cutlass.range_constexpr(cute.size(lse)):
                row = tScS_mn[m, 0][0]
                if tScS_mn[m, 0][1] == 0 and row < q_len:
                    mLSE_slice[q_start + row] = lse[m]

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))
        sO = storage.Q_smem.get_tensor(
            O_smem_layout.outer, swizzle=O_smem_layout.inner
        )
        tiled_copy_O = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.O_layout.is_m_major_c(), 4
                ),
                self.O_dtype,
            ),
            tiled_mma_pv,
        )
        tOrO_cv = tiled_copy_O.retile(tOrO_cvt)
        tOsO = tiled_copy_O.get_slice(tidx).partition_D(sO)
        cute.copy(tiled_copy_O, tOrO_cv, tOsO)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_O,
            *cta_coord_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )
        if warp == 0:
            cute.copy(tma_atom_O, tOsO, tOgO)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        kc: cute.Tensor,
        vc: cute.Tensor,
        threshold: cute.Tensor,
        lse: cute.Tensor,
        softmax_scale: cutlass.Float32,
        sink_start_block: cutlass.Int32,
        sink_end_block: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        q_mkl, k_nkl, kc_nkl = [
            layout_utils.select(t, [1, 3, 2, 0])
            for t in (q, k, kc)
        ]
        v_nkl, vc_nkl = [
            layout_utils.select(t, [3, 1, 2, 0]) for t in (v, vc)
        ]
        o_mkl = layout_utils.select(o, [1, 3, 2, 0])
        if cutlass.const_expr(self.debug_route_trace):
            lse_target = lse
        else:
            lse_target = layout_utils.select(lse, [1, 2, 0])

        self.Q_dtype = q_mkl.element_type
        self.K_dtype = k_nkl.element_type
        self.V_dtype = v_nkl.element_type
        self.O_dtype = o_mkl.element_type
        self.Q_layout = utils.LayoutEnum.from_tensor(q_mkl)
        self.K_layout = utils.LayoutEnum.from_tensor(k_nkl)
        self.V_layout = utils.LayoutEnum.from_tensor(v_nkl)
        self.O_layout = utils.LayoutEnum.from_tensor(o_mkl)
        assert self.Q_dtype == cutlass.BFloat16
        assert self.K_dtype == cutlass.BFloat16
        assert self.V_dtype == cutlass.BFloat16

        self.Q_smem_layout = sm90_utils.make_smem_layout_a(
            self.Q_layout,
            self.tile_shape_qk,
            self.Q_dtype,
            self.q_stage,
        )
        self.K_smem_layout = sm90_utils.make_smem_layout_b(
            self.K_layout,
            self.tile_shape_qk,
            self.K_dtype,
            self.kv_stage,
        )
        self.V_smem_layout = sm90_utils.make_smem_layout_b(
            self.V_layout,
            self.tile_shape_pv,
            self.V_dtype,
            self.kv_stage,
        )
        O_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.O_dtype,
            self.O_layout,
            self.tile_shape_pv[:2],
            1,
        )
        self.O_smem_layout = cute.select(
            O_smem_layout_staged, mode=[0, 1]
        )

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[
                cutlass.Int64, self.q_stage * 2
            ]
            K_barrier: cute.struct.MemRange[
                cutlass.Int64, self.kv_stage * 2
            ]
            V_barrier: cute.struct.MemRange[
                cutlass.Int64, self.kv_stage * 2
            ]
            Q_smem: cute.struct.Align[
                cute.struct.MemRange[
                    self.Q_dtype, cute.cosize(self.Q_smem_layout)
                ],
                128,
            ]
            K_smem: cute.struct.Align[
                cute.struct.MemRange[
                    self.K_dtype, cute.cosize(self.K_smem_layout)
                ],
                128,
            ]
            V_smem: cute.struct.Align[
                cute.struct.MemRange[
                    self.V_dtype, cute.cosize(self.V_smem_layout)
                ],
                128,
            ]
        self.shared_storage_t = SharedStorage

        tiled_mma_qk = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.Q_dtype,
                self.acc_dtype,
                (16, 8, 16),
            ),
            cute.make_layout((4, 1, 1)),
            permutation_mnk=(64, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.K_dtype,
                self.acc_dtype,
                (16, 8, 16),
            ),
            cute.make_layout((4, 1, 1)),
            permutation_mnk=(64, 16, 16),
        )

        g2s_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_Q, tma_tensor_Q = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                g2s_op,
                q_mkl,
                self.Q_smem_layout,
                (M, D),
                num_multicast=1,
            )
        )
        tma_atom_K, tma_tensor_K = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                g2s_op,
                k_nkl,
                self.K_smem_layout,
                (N, D),
                num_multicast=1,
            )
        )
        tma_atom_V, tma_tensor_V = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                g2s_op,
                v_nkl,
                self.V_smem_layout,
                (DV, N),
                num_multicast=1,
            )
        )
        tma_atom_KC, tma_tensor_KC = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                g2s_op,
                kc_nkl,
                self.K_smem_layout,
                (N, D),
                num_multicast=1,
            )
        )
        tma_atom_VC, tma_tensor_VC = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                g2s_op,
                vc_nkl,
                self.V_smem_layout,
                (DV, N),
                num_multicast=1,
            )
        )
        s2g_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_O, tma_tensor_O = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                s2g_op,
                o_mkl,
                self.O_smem_layout,
                (M, DV),
                num_multicast=1,
            )
        )

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            tma_tensor_KC,
            tma_tensor_VC,
            threshold,
            lse_target,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_KC,
            tma_atom_VC,
            tma_atom_O,
            tiled_mma_qk,
            tiled_mma_pv,
            self.Q_smem_layout,
            self.K_smem_layout,
            self.V_smem_layout,
            self.O_smem_layout,
            softmax_scale * 1.4426950408889634,
            sink_start_block,
            sink_end_block,
        ).launch(
            grid=(cute.ceil_div(q_mkl.shape[0], M), q_mkl.shape[2], q_mkl.shape[3]),
            block=(self.num_threads, 1, 1),
            cluster=(1, 1, 1),
            smem=self.shared_storage_t.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )


@cute.jit
def gemm_smem_zero_acc(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_tiled_copy_B: cute.TiledCopy,
):
    acc.fill(0.0)
    tCrB_copy = smem_tiled_copy_B.retile(tCrB)
    cute.copy(
        smem_tiled_copy_B,
        tCsB[None, None, 0],
        tCrB_copy[None, None, 0],
    )
    for k_block in cutlass.range_constexpr(cute.size(tCsB.shape[2])):
        if k_block < cute.size(tCsB.shape[2]) - 1:
            cute.copy(
                smem_tiled_copy_B,
                tCsB[None, None, k_block + 1],
                tCrB_copy[None, None, k_block + 1],
            )
        cute.gemm(
            tiled_mma,
            acc,
            tCrA[None, None, k_block],
            tCrB[None, None, k_block],
            acc,
        )


@cute.jit
def gemm_rs_smem(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_tiled_copy_B: cute.TiledCopy,
):
    tCrB_copy = smem_tiled_copy_B.retile(tCrB)
    cute.copy(
        smem_tiled_copy_B,
        tCsB[None, None, 0],
        tCrB_copy[None, None, 0],
    )
    for k_block in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if k_block < cute.size(tCrA.shape[2]) - 1:
            cute.copy(
                smem_tiled_copy_B,
                tCsB[None, None, k_block + 1],
                tCrB_copy[None, None, k_block + 1],
            )
        cute.gemm(
            tiled_mma,
            acc,
            tCrA[None, None, k_block],
            tCrB[None, None, k_block],
            acc,
        )


@cute.jit
def reduce_route_columns(
    scores: cute.Tensor,
    coords: cute.Tensor,
    route_sums: cute.Tensor,
    warp: cutlass.Int32,
    lane: cutlass.Int32,
    q_len: cutlass.Int32,
):
    """Reduce M64 score columns using the measured SM120 lane layout."""

    scores_mn = layout_utils.reshape_acc_to_mn(scores)
    coords_mn = layout_utils.reshape_acc_to_mn(coords)
    row0 = coords_mn[0, 0][0]
    row1 = coords_mn[1, 0][0]
    valid0 = row0 < q_len
    valid1 = row1 < q_len
    for group in cutlass.range_constexpr(8):
        n0 = group * 2
        partial0 = cutlass.Float32(0.0)
        partial1 = cutlass.Float32(0.0)
        if valid0:
            partial0 += cutlass.Float32(scores_mn[0, n0])
            partial1 += cutlass.Float32(scores_mn[0, n0 + 1])
        if valid1:
            partial0 += cutlass.Float32(scores_mn[1, n0])
            partial1 += cutlass.Float32(scores_mn[1, n0 + 1])
        for offset in (4, 8, 16):
            partial0 += cute.arch.shuffle_sync_bfly(partial0, offset=offset)
            partial1 += cute.arch.shuffle_sync_bfly(partial1, offset=offset)
        if lane < 4:
            column = cutlass.Int32(group * 8) + lane * cutlass.Int32(2)
            route_sums[warp, column] = partial0
            route_sums[warp, column + 1] = partial1


@cute.jit
def apply_route_mask(
    scores: cute.Tensor,
    coords: cute.Tensor,
    column_masks: cute.Tensor,
    q_len: cutlass.Int32,
):
    scores_mn = layout_utils.reshape_acc_to_mn(scores)
    coords_mn = layout_utils.reshape_acc_to_mn(coords)
    for m in cutlass.range_constexpr(cute.size(scores_mn, mode=[0])):
        valid_row = coords_mn[m, 0][0] < q_len
        for n in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
            column = coords_mn[m, n][1]
            scores_mn[m, n] = (
                cutlass.Float32(scores_mn[m, n])
                + cutlass.Float32(column_masks[column])
                if valid_row
                else -cutlass.Float32.inf
            )


@cute.jit
def mask_exact_scores(
    scores: cute.Tensor,
    coords: cute.Tensor,
    block_len: cutlass.Int32,
    q_len: cutlass.Int32,
):
    scores_mn = layout_utils.reshape_acc_to_mn(scores)
    coords_mn = layout_utils.reshape_acc_to_mn(coords)
    for m in cutlass.range_constexpr(cute.size(scores_mn, mode=[0])):
        valid_row = coords_mn[m, 0][0] < q_len
        for n in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
            if (not valid_row) or coords_mn[m, n][1] >= block_len:
                scores_mn[m, n] = -cutlass.Float32.inf


@cute.jit
def online_softmax(
    scores: cute.Tensor,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    scale_log2e: cutlass.Float32,
):
    scores_mn = layout_utils.reshape_acc_to_mn(scores)
    row_scale = cute.make_rmem_tensor_like(row_max, cutlass.Float32)
    for m in cutlass.range_constexpr(cute.size(row_max)):
        score_row = scores_mn[m, None].load()
        current_max = kernel_utils.fmax_reduce(
            score_row, init_val=row_max[m], arch=80
        )
        current_max = cute.arch.warp_reduction_max(
            current_max, threads_in_group=4
        )
        previous_max = row_max[m]
        row_max[m] = current_max
        safe_max = (
            cutlass.Float32(0.0)
            if current_max == -cutlass.Float32.inf
            else current_max
        )
        scaled_max = safe_max * scale_log2e
        probabilities = cute.math.exp2(
            score_row * scale_log2e - scaled_max, fastmath=True
        )
        row_scale[m] = cute.math.exp2(
            (previous_max - safe_max) * scale_log2e, fastmath=True
        )
        row_sum[m] = kernel_utils.fadd_reduce(
            probabilities,
            init_val=row_sum[m] * row_scale[m],
            arch=80,
        )
        scores_mn[m, None].store(probabilities)
    return row_scale


@cute.jit
def online_softmax_route(
    scores: cute.Tensor,
    coords: cute.Tensor,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    scale_log2e: cutlass.Float32,
    group_start: cutlass.Int32,
    token_count: cutlass.Int32,
):
    scores_mn = layout_utils.reshape_acc_to_mn(scores)
    coords_mn = layout_utils.reshape_acc_to_mn(coords)
    row_scale = cute.make_rmem_tensor_like(row_max, cutlass.Float32)
    for m in cutlass.range_constexpr(cute.size(row_max)):
        score_row = scores_mn[m, None].load()
        current_max = kernel_utils.fmax_reduce(
            score_row, init_val=row_max[m], arch=80
        )
        current_max = cute.arch.warp_reduction_max(
            current_max, threads_in_group=4
        )
        previous_max = row_max[m]
        row_max[m] = current_max
        safe_max = (
            cutlass.Float32(0.0)
            if current_max == -cutlass.Float32.inf
            else current_max
        )
        probabilities = cute.math.exp2(
            score_row * scale_log2e - safe_max * scale_log2e,
            fastmath=True,
        )
        row_scale[m] = cute.math.exp2(
            (previous_max - safe_max) * scale_log2e, fastmath=True
        )
        masses = cute.make_rmem_tensor_like(
            scores_mn[m, None], cutlass.Float32
        )
        for n in cutlass.range_constexpr(cute.size(masses)):
            block = group_start + coords_mn[m, n][1]
            length = token_count - block * cutlass.Int32(N)
            if length > N:
                length = cutlass.Int32(N)
            if length < 0:
                length = cutlass.Int32(0)
            masses[n] = cutlass.Float32(probabilities[n]) * cutlass.Float32(
                length
            )
        row_sum[m] = kernel_utils.fadd_reduce(
            masses.load(),
            init_val=row_sum[m] * row_scale[m],
            arch=80,
        )
        scores_mn[m, None].store(probabilities)
    return row_scale


@cute.jit
def finalize_softmax(
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    scale_log2e: cutlass.Float32,
):
    row_sum.store(
        kernel_utils.warp_reduce(row_sum.load(), operator.add, width=4)
    )
    ratio = cute.make_rmem_tensor_like(row_sum, cutlass.Float32)
    lse = cute.make_rmem_tensor_like(row_sum, cutlass.Float32)
    for m in cutlass.range_constexpr(cute.size(row_sum)):
        total = row_sum[m]
        invalid = total == 0.0 or total != total
        ratio[m] = cute.arch.rcp_approx(total if not invalid else 1.0)
        lse[m] = (
            -cutlass.Float32.inf
            if invalid
            else (
                row_max[m] * scale_log2e
                + cute.math.log2(total, fastmath=True)
            )
            * 0.6931471805599453
        )
    return ratio, lse


@cute.jit
def rescale_o_for_next_acc(
    output: cute.Tensor,
    row_scale: cute.Tensor,
):
    output_mn = layout_utils.reshape_acc_to_mn(output)
    for m in cutlass.range_constexpr(cute.size(row_scale)):
        output_mn[m, None].store(
            output_mn[m, None].load() * row_scale[m]
        )


__all__ = ["SolAttnForwardSm120"]
