# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fused Sol-Attn forward kernel for Ada SM89.

The synchronous warp-MMA, cp.async load, and universal-store skeleton comes
from the vendored FlashAttention SM80 CuTe DSL path.  Sol routing, approximate
and exact contributions, and FP32 online-softmax math match the released
SM120 kernel.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp

from sol_attn._vendor.flash_attn.cute import utils as fa_utils
from sol_attn._vendor.flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from sol_attn._vendor.flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from sol_attn._vendor.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sol_attn.common import layout_utils
from sol_attn.common.selector import sol_attn_popc_b32, sol_attn_route_is_exact
from sol_attn.sm120.mainloop import (
    apply_route_mask,
    finalize_softmax,
    gemm_rs_smem,
    gemm_smem_zero_acc,
    mask_exact_scores,
    online_softmax,
    online_softmax_route,
    reduce_route_columns,
    rescale_o_for_next_acc,
)


M = 64
N = 64
D = 128
DV = 128
THREADS = 128
STAGES = 1
ROUTE_WORDS_F32 = 6 * N + 2


@cute.jit
def clear_smem_partition(tensor: cute.Tensor):
    """Cooperatively zero one thread's partition of an SMEM tile."""

    zeros = cute.make_fragment_like(tensor)
    zeros.fill(0.0)
    cute.basic_copy(zeros, tensor)


@cute.jit
def store_route_matrix_trace(
    matrix: cute.Tensor,
    coords: cute.Tensor,
    target: cute.Tensor,
    batch_idx: Int32,
    q_tile_idx: Int32,
    head_idx: Int32,
    route_group: Int32,
    q_len: Int32,
    valid_blocks: Int32,
):
    matrix_mn = layout_utils.reshape_acc_to_mn(matrix)
    coords_mn = layout_utils.reshape_acc_to_mn(coords)
    for m in cutlass.range_constexpr(cute.size(matrix_mn, mode=[0])):
        row = coords_mn[m, 0][0]
        if row < q_len:
            for n in cutlass.range_constexpr(cute.size(matrix_mn, mode=[1])):
                column = coords_mn[m, n][1]
                if column < valid_blocks:
                    target[
                        batch_idx,
                        q_tile_idx,
                        head_idx,
                        route_group,
                        row,
                        column,
                    ] = Float32(matrix_mn[m, n])


class SolAttnForwardSm89(FlashAttentionForwardSm80):
    """M64/N64 warp-MMA Sol-Attn kernel for BF16 D128 inputs."""

    def __init__(
        self,
        *,
        route_sum_order: int = 3,
        debug_route_trace: bool = False,
        debug_index_trace: bool = False,
        debug_route_group_limit: int = 0,
        debug_score_trace: bool = False,
        debug_probability_trace: bool = False,
    ):
        super().__init__(
            dtype=cutlass.BFloat16,
            head_dim=D,
            head_dim_v=DV,
            qhead_per_kvhead=1,
            is_causal=False,
            is_local=False,
            pack_gqa=False,
            tile_m=M,
            tile_n=N,
            num_stages=STAGES,
            num_threads=THREADS,
            Q_in_regs=True,
        )
        if route_sum_order not in (0, 1, 2, 3):
            raise ValueError("route_sum_order must be 0, 1, 2, or 3")
        self.route_sum_order = route_sum_order
        self.debug_route_trace = debug_route_trace
        self.debug_index_trace = debug_index_trace
        self.debug_route_group_limit = debug_route_group_limit
        self.debug_score_trace = debug_score_trace
        self.debug_probability_trace = debug_probability_trace

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
        softmax_scale: Float32,
        sink_start_block: Int32,
        sink_end_block: Int32,
        stream: cuda.CUstream,
    ):
        assert q.element_type == cutlass.BFloat16
        assert k.element_type == cutlass.BFloat16
        assert v.element_type == cutlass.BFloat16
        assert o.element_type == cutlass.BFloat16

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        self.use_tma_O = False
        self._setup_attributes()

        cosize_s_qv = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        s_qv_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_s_qv], 1024
        ]
        s_k_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)], 1024
        ]
        route_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, ROUTE_WORDS_F32], 128
        ]

        @cute.struct
        class SharedStorage:
            sQV: s_qv_struct
            sK: s_k_struct
            route: route_struct

        q, k, v, o, kc, vc = [
            assume_tensor_aligned(t) for t in (q, k, v, o, kc, vc)
        ]
        q_mkl, k_nkl, v_nkl, o_mkl, kc_nkl, vc_nkl = [
            layout_utils.select(t, [1, 3, 2, 0])
            for t in (q, k, v, o, kc, vc)
        ]
        lse_target = (
            lse
            if const_expr(
                self.debug_route_trace
                or self.debug_index_trace
                or self.debug_score_trace
                or self.debug_probability_trace
            )
            else layout_utils.select(lse, [1, 2, 0])
        )

        self.kernel(
            q_mkl,
            k_nkl,
            v_nkl,
            o_mkl,
            kc_nkl,
            vc_nkl,
            threshold,
            lse_target,
            softmax_scale * 1.4426950408889634,
            sink_start_block,
            sink_end_block,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=(cute.ceil_div(q_mkl.shape[0], M), q_mkl.shape[2], q_mkl.shape[3]),
            block=(self.num_threads, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
        )

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
        scale_softmax_log2e: Float32,
        sink_start_block: Int32,
        sink_end_block: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_tile_idx, head_idx, batch_idx = cute.arch.block_idx()
        q_tile_idx = cute.arch.make_warp_uniform(q_tile_idx)
        head_idx = cute.arch.make_warp_uniform(head_idx)
        batch_idx = cute.arch.make_warp_uniform(batch_idx)

        token_count = mK.shape[0]
        num_blocks = mKC.shape[0]
        num_route_groups = cute.ceil_div(num_blocks, N)
        if const_expr(self.debug_route_group_limit > 0):
            num_route_groups = Int32(self.debug_route_group_limit)
        q_start = q_tile_idx * M
        q_len = token_count - q_start
        if q_len > M:
            q_len = Int32(M)
        threshold = Float32(mThreshold[batch_idx, q_tile_idx, head_idx])

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        sQ = storage.sQV.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout
        )
        sVt = layout_utils.transpose_view(sV)

        route_f32_ptr = storage.route.data_ptr()
        route_i32_ptr = cute.recast_ptr(route_f32_ptr, dtype=Int32)
        route_sums = cute.make_tensor(route_f32_ptr, cute.make_layout((4, N)))
        column_masks = cute.make_tensor(route_f32_ptr + 4 * N, cute.make_layout(N))
        route_indices = cute.make_tensor(route_i32_ptr + 5 * N, cute.make_layout(N))
        route_meta = cute.make_tensor(route_i32_ptr + 6 * N, cute.make_layout(2))

        mQ_slice = mQ[None, None, head_idx, batch_idx]
        mK_slice = mK[None, None, head_idx, batch_idx]
        mV_slice = mV[None, None, head_idx, batch_idx]
        mKC_slice = mKC[None, None, head_idx, batch_idx]
        mVC_slice = mVC[None, None, head_idx, batch_idx]
        gQ = cute.local_tile(mQ_slice, (M, D), (q_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (N, D), (None, 0))
        gV = cute.local_tile(mV_slice, (N, DV), (None, 0))
        gKC = cute.local_tile(mKC_slice, (N, D), (None, 0))
        gVC = cute.local_tile(mVC_slice, (N, DV), (None, 0))

        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        tKsK = gmem_thr_copy_K.partition_D(sK)
        tKgK = gmem_thr_copy_K.partition_S(gK)
        tKCgKC = gmem_thr_copy_K.partition_S(gKC)
        tVsV = gmem_thr_copy_V.partition_D(sV)
        tVgV = gmem_thr_copy_V.partition_S(gV)
        tVCgVC = gmem_thr_copy_V.partition_S(gVC)

        cK = cute.make_identity_tensor((N, D))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_tiled_copy_K.get_slice(0).partition_S(cK)
        tKpK = fa_utils.predicate_k(tKcK, limit=D)
        cV = cute.make_identity_tensor((N, DV))
        tVcV = gmem_thr_copy_V.partition_S(cV)
        t0VcV = gmem_tiled_copy_V.get_slice(0).partition_S(cV)
        tVpV = fa_utils.predicate_k(tVcV, limit=DV)

        cS = cute.make_identity_tensor((M, N))
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrV = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((M, N)), Float32)
        tScS = thr_mma_qk.partition_C(cS)
        tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((M, DV)), Float32)

        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_Q = fa_utils.make_tiled_copy_A(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_K = fa_utils.make_tiled_copy_B(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_V = fa_utils.make_tiled_copy_B(
            smem_copy_atom_V, tiled_mma_pv
        ).get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsV = smem_thr_copy_V.partition_S(sVt)

        max_m_layout = cute.make_layout(
            cute.size(layout_utils.reshape_acc_to_mn(tOrO).layout, mode=[0])
        )
        max_m = cute.make_rmem_tensor_like(max_m_layout, Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, Float32)
        tOrO.fill(0.0)
        max_m.fill(float("-inf"))
        sum_m.fill(0.0)

        self.load_Q(gmem_thr_copy_Q, gQ, sQ, q_tile_idx, token_count, D)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)
        cute.arch.barrier()

        for route_group in cutlass.range(0, num_route_groups, 1, unroll=1):
            group_start = route_group * Int32(N)
            valid_blocks = num_blocks - group_start
            if valid_blocks > N:
                valid_blocks = Int32(N)

            if valid_blocks < N:
                # Predicated cp.async leaves skipped rows untouched.  The
                # approximate PV consumes every route-group row, so stale
                # BF16 NaNs from an earlier kernel would otherwise survive a
                # zero probability as 0 * NaN.  Clear only a ragged summary
                # group, then overwrite every valid KC/VC row asynchronously.
                clear_smem_partition(tKsK[None, None, None, 0])
                clear_smem_partition(tVsV[None, None, None, 0])
                cute.arch.sync_threads()

            self.load_K(
                gmem_tiled_copy_K,
                tKCgKC,
                tKsK,
                tKcK,
                t0KcK,
                tKpK,
                route_group,
                Int32(0),
                num_blocks,
                True,
            )
            cute.arch.cp_async_commit_group()
            self.load_V(
                gmem_tiled_copy_V,
                tVCgVC,
                tVsV,
                tVcV,
                t0VcV,
                tVpV,
                route_group,
                Int32(0),
                num_blocks,
                True,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier()

            gemm_smem_zero_acc(
                tiled_mma_qk,
                tSrS,
                tSrQ,
                tSrK,
                tSsK[None, None, None, 0],
                smem_thr_copy_K,
            )
            if const_expr(self.debug_score_trace):
                store_route_matrix_trace(
                    tSrS,
                    tScS,
                    mLSE,
                    batch_idx,
                    q_tile_idx,
                    head_idx,
                    route_group,
                    q_len,
                    valid_blocks,
                )
            reduce_route_columns(tSrS, tScS, route_sums, warp_idx, lane, q_len)
            cute.arch.sync_threads()

            if warp_idx == 0:
                preceding = Int32(0)
                lane_mask_lt = Int32(0x7FFFFFFF) >> (Int32(31) - lane)
                for word in cutlass.range_constexpr(2):
                    off = Int32(word * 32) + lane
                    valid = off < valid_blocks
                    exact = False
                    if valid:
                        r0 = Float32(route_sums[0, off])
                        r1 = Float32(route_sums[1, off])
                        r2 = Float32(route_sums[2, off])
                        r3 = Float32(route_sums[3, off])
                        col_sum = r0 + r1 + r2 + r3
                        if const_expr(self.route_sum_order == 1):
                            col_sum = (r0 + r1) + (r2 + r3)
                        if const_expr(self.route_sum_order == 2):
                            col_sum = (r0 + r2) + (r1 + r3)
                        if const_expr(self.route_sum_order == 3):
                            col_sum = (r0 + r3) + (r1 + r2)
                        col_mean = col_sum * scale_softmax_log2e / Float32(q_len)
                        kv_block = group_start + off
                        exact = sol_attn_route_is_exact(
                            q_tile_idx, kv_block, col_mean, threshold, valid
                        )
                        exact = exact or (
                            kv_block >= sink_start_block
                            and kv_block < sink_end_block
                        )
                    ballot = Int32(cute.arch.vote_ballot_sync(exact))
                    column_masks[off] = (
                        -Float32.inf if (exact or not valid) else Float32(0.0)
                    )
                    rank = preceding + sol_attn_popc_b32(ballot & lane_mask_lt)
                    if exact:
                        route_indices[rank] = group_start + off
                    preceding += sol_attn_popc_b32(ballot)
                    if const_expr(self.debug_route_trace):
                        if lane == 0:
                            mLSE[
                                batch_idx,
                                q_tile_idx,
                                head_idx,
                                route_group,
                                word,
                            ] = ballot
                    if const_expr(self.debug_index_trace):
                        if lane == 0:
                            mLSE[
                                batch_idx,
                                q_tile_idx,
                                head_idx,
                                route_group,
                                66 + word,
                            ] = ballot
                if lane == 0:
                    route_meta[0] = preceding
                    route_meta[1] = valid_blocks
            cute.arch.sync_threads()

            exact_count = Int32(route_meta[0])
            has_approx = exact_count < valid_blocks
            if const_expr(self.debug_index_trace):
                if warp_idx == 0:
                    if lane == 0:
                        mLSE[
                            batch_idx,
                            q_tile_idx,
                            head_idx,
                            route_group,
                            0,
                        ] = exact_count
                    if lane < exact_count:
                        mLSE[
                            batch_idx,
                            q_tile_idx,
                            head_idx,
                            route_group,
                            lane + 1,
                        ] = route_indices[lane]
                    upper = lane + Int32(32)
                    if upper < exact_count:
                        mLSE[
                            batch_idx,
                            q_tile_idx,
                            head_idx,
                            route_group,
                            upper + 1,
                        ] = route_indices[upper]
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
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
                if const_expr(self.debug_probability_trace):
                    store_route_matrix_trace(
                        tSrS,
                        tScS,
                        mLSE,
                        batch_idx,
                        q_tile_idx,
                        head_idx,
                        route_group,
                        q_len,
                        valid_blocks,
                    )
                rescale_o_for_next_acc(tOrO, row_scale)
                tOrP_frg = cute.make_rmem_tensor_like(tSrS, self.dtype)
                tOrP_frg.store(tSrS.load().to(self.dtype))
                tOrP = layout_utils.reshape_acc_to_frgA(tOrP_frg)
                gemm_rs_smem(
                    tiled_mma_pv,
                    tOrO,
                    tOrP,
                    tOrV,
                    tOsV[None, None, None, 0],
                    smem_thr_copy_V,
                )
            cute.arch.barrier()

            for ordinal in cutlass.range(0, exact_count, 1, unroll=1):
                exact_block = Int32(route_indices[ordinal])
                self.load_K(
                    gmem_tiled_copy_K,
                    tKgK,
                    tKsK,
                    tKcK,
                    t0KcK,
                    tKpK,
                    exact_block,
                    Int32(0),
                    token_count,
                    True,
                )
                cute.arch.cp_async_commit_group()
                self.load_V(
                    gmem_tiled_copy_V,
                    tVgV,
                    tVsV,
                    tVcV,
                    t0VcV,
                    tVpV,
                    exact_block,
                    Int32(0),
                    token_count,
                    True,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
                cute.arch.barrier()

                gemm_smem_zero_acc(
                    tiled_mma_qk,
                    tSrS,
                    tSrQ,
                    tSrK,
                    tSsK[None, None, None, 0],
                    smem_thr_copy_K,
                )
                block_len = token_count - exact_block * Int32(N)
                if block_len > N:
                    block_len = Int32(N)
                mask_exact_scores(tSrS, tScS, block_len, q_len)
                row_scale = online_softmax(
                    tSrS, max_m, sum_m, scale_softmax_log2e
                )
                rescale_o_for_next_acc(tOrO, row_scale)
                tOrP_frg = cute.make_rmem_tensor_like(tSrS, self.dtype)
                tOrP_frg.store(tSrS.load().to(self.dtype))
                tOrP = layout_utils.reshape_acc_to_frgA(tOrP_frg)

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                gemm_rs_smem(
                    tiled_mma_pv,
                    tOrO,
                    tOrP,
                    tOrV,
                    tOsV[None, None, None, 0],
                    smem_thr_copy_V,
                )
                cute.arch.barrier()

        final_ratio, lse = finalize_softmax(max_m, sum_m, scale_softmax_log2e)
        rescale_o_for_next_acc(tOrO, final_ratio)
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=token_count,
            seqlen_k_static=token_count,
            tile_m=M,
            tile_n=N,
        )
        self.epilogue(
            tOrO,
            lse,
            mO,
            None
            if const_expr(
                self.debug_route_trace
                or self.debug_index_trace
                or self.debug_score_trace
                or self.debug_probability_trace
            )
            else mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            None,
            tiled_mma_pv,
            tidx,
            q_tile_idx,
            head_idx,
            batch_idx,
        )


__all__ = ["SolAttnForwardSm89"]
