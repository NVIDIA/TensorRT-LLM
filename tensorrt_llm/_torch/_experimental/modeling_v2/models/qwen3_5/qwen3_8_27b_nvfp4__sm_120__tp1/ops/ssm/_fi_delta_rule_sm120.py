# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SM120 chunked gated-delta-rule prefill, non-context-parallel: FlashInfer's fully
fused kernel, or its V-split x2 variant when one long sequence leaves the GPU idle.

``_VSplitDeltaRuleSm120`` subclasses FlashInfer's ``_FullyFusedDeltaRuleSm120``
(flashinfer-ai/flashinfer commit 69ff11fc4954396d98326656dc85debd2223f637,
flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py, Apache-2.0; imported from
the installed package, unmodified). It runs 2 CTAs per (sequence, head), each owning
64 of the state's 128 V rows: the aux warp group computes KK^T, its inverse and QK^T
into staged smem, the state warp group runs O1 = S Q^T, SK = S K^T, NewV, O2 and
S += NewV K on its 64 rows, 4 warps x 16 whole state rows. Every MMA keeps the fully
fused kernel's per-warp shape (16 V rows x the full reduction) and V is never a
reduction dimension, so ``o`` and the final state are bitwise the fully fused
kernel's. KK^T / inverse / QK^T are recomputed identically by both halves.

``run`` launches it when ``vsplit_pays_off`` (the long sequences' doubled CTAs still
fit one wave), else the installed fully fused launcher. fp32 packed initial and
final states, no checkpoints.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from flashinfer.gdn_kernels.delta_rule_dsl.alpha import AlphaProcessor
from flashinfer.gdn_kernels.delta_rule_dsl.collective_store_tma import CollectiveStoreTma
from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import (
    cached_compile,
    sm12x_compile_options,
)
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import (
    LoadStoreWarpRole,
    WarpGroupRole,
    _FullyFusedDeltaRuleSm120,
)
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import (
    delta_rule_prefill_dsl as _fully_fused_prefill,
)
from flashinfer.gdn_kernels.delta_rule_dsl.helpers import SM80
from flashinfer.gdn_kernels.delta_rule_dsl.schedule import WorkDesc
from flashinfer.utils import _get_cache_buf, get_device_sm_count

#: The fully fused kernel's compile-cache key attributes (its ``__init__``'s list).
_BASE_CACHE_KEY = (
    "needs_alpha",
    "needs_beta",
    "needs_init_state",
    "needs_checkpointing",
    "dtype",
    "acc_dtype",
    "inverse_dtype",
    "BLK_Q",
    "BLK_KV",
    "D",
    "q_stage",
    "k_stage",
    "v_stage",
    "o_stage",
    "alpha_beta_stage",
)


# ─── V-split x2 variant ────────────────────────────────────────────────────────
# Grid: (num_seqs * num_sab_heads * 2, 1, 1); bx -> (seq, head, v_half).
# Each CTA owns 64 of the 128 V rows of one (seq, head) state.
#   WG1 (math WG0, "aux"):   KK^T, inverse (+beta), QK^T -> staged smem (kk/qk pipes)
#   WG2 (math WG1, "state"): O1 = S Q^T, SK = S K^T, NewV, O2, S += NewV K on its
#                            64 V rows, 4 warps x 16 whole state rows each.
# Every MMA keeps the stock per-warp shape (16 V rows x full reduction), so o and
# the final state are bitwise equal to the stock kernel. KK^T / inverse / QK^T
# are recomputed identically by both halves.


class _VSplitDeltaRuleSm120(_FullyFusedDeltaRuleSm120):
    def __init__(
        self,
        needs_alpha: bool,
        needs_beta: bool,
        needs_init_state: bool,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.v_split = 2
        self.DV = 64
        self.kk_stage = 2
        self.qk_stage = 2
        super().__init__(needs_alpha, needs_beta, needs_init_state, False, dtype, acc_dtype)
        self.manual_cache_key(*_BASE_CACHE_KEY, "v_split", "DV", "kk_stage", "qk_stage")

    def get_next_work_vs(
        self,
        cu_seqlens: cute.Tensor,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        per_seq = num_sab_heads * cutlass.Int32(self.v_split)
        seq_idx = bx // per_seq
        rem = bx % per_seq
        o_head_idx = rem // cutlass.Int32(self.v_split)
        v_half = rem % cutlass.Int32(self.v_split)
        q_head_idx = o_head_idx * num_q_heads // num_sab_heads
        v_head_idx = o_head_idx * num_v_heads // num_sab_heads
        tok_start = cutlass.Int32(cu_seqlens[seq_idx])
        tok_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
        return (
            WorkDesc(
                seq_idx=seq_idx,
                private_q_head_idx=q_head_idx,
                private_v_head_idx=v_head_idx,
                tok_offset=tok_start,
                seq_len=tok_end - tok_start,
                tile_idx=cutlass.Int32(0),
            ),
            v_half,
        )

    # ── state I/O on the CTA's 64 V rows ([N,H,V,K] layout, gKV is (K, V)) ──

    @cute.jit
    def kv_load_vs(self, tKVrKV: cute.Tensor, gKV: cute.Tensor, kv_thr_mma):
        c_kv = cute.make_identity_tensor((self.DV, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            tKVrKV[i] = gKV[k_idx, v_idx]

    @cute.jit
    def kv_store_vs(self, tKVrKV: cute.Tensor, gKV: cute.Tensor, kv_thr_mma):
        c_kv = cute.make_identity_tensor((self.DV, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            gKV[k_idx, v_idx] = tKVrKV[i]

    # ── TMA loads: full Q/K tiles, the CTA's 64-row V tile ──

    @cute.jit
    def load_qkv_tma_vs(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        blk: cutlass.Int32,
        tok_start: cutlass.Int32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
        v_off: cutlass.Int32,
    ):
        blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)

        sK = sK_DS[None, None, k_producer_state.index]
        mK = cute.domain_offset(
            (cutlass.Int32(0), blk_tok),
            tma_tensor_k[None, None, k_head_idx],
        )
        gK = cute.zipped_divide(mK, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        k_pipeline.producer_acquire(k_producer_state)
        cute.copy(
            tma_atom_k,
            tKgK,
            tKsK,
            tma_bar_ptr=k_pipeline.producer_get_barrier(k_producer_state),
        )
        k_pipeline.producer_commit(k_producer_state)
        k_producer_state.advance()

        sQ = sQ_SD[None, None, q_producer_state.index]
        mQ = cute.domain_offset(
            (blk_tok, cutlass.Int32(0)),
            tma_tensor_q[None, None, q_head_idx],
        )
        gQ = cute.zipped_divide(mQ, (self.BLK_Q, self.D))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        q_pipeline.producer_acquire(q_producer_state)
        cute.copy(
            tma_atom_q,
            tQgQ,
            tQsQ,
            tma_bar_ptr=q_pipeline.producer_get_barrier(q_producer_state),
        )
        q_pipeline.producer_commit(q_producer_state)
        q_producer_state.advance()

        sV = sV_DS[None, None, v_producer_state.index]
        mV = cute.domain_offset(
            (v_off, blk_tok),
            tma_tensor_v[None, None, v_head_idx],
        )
        gV = cute.zipped_divide(mV, (self.DV, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )
        v_pipeline.producer_acquire(v_producer_state)
        cute.copy(
            tma_atom_v,
            tVgV,
            tVsV,
            tma_bar_ptr=v_pipeline.producer_get_barrier(v_producer_state),
        )
        v_pipeline.producer_commit(v_producer_state)
        v_producer_state.advance()
        return q_producer_state, k_producer_state, v_producer_state

    @cute.jit
    def run_load_qkv_role_vs(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        num_blocks: cutlass.Int32,
        tok_start: cutlass.Int32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
        v_off: cutlass.Int32,
    ):
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stage
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.v_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
            ) = self.load_qkv_tma_vs(
                sQ_SD,
                sK_DS,
                sV_DS,
                tma_atom_q,
                tma_tensor_q,
                tma_atom_k,
                tma_tensor_k,
                tma_atom_v,
                tma_tensor_v,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                blk,
                tok_start,
                q_head_idx,
                k_head_idx,
                v_head_idx,
                v_off,
            )

    # ── aux WG: KK^T + inverse (+beta) -> sKK_opd[stage], QK^T -> sQK[stage] ──

    @cute.jit
    def vs_aux_body(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sQK: cute.Tensor,  # (BlkQ, BlkKV, qk_stage)
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV, kk_stage) inverse dtype
        sKK_opd: cute.Tensor,  # same storage as Element
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        q_pipeline,
        q_consumer_state,
        k_pipeline,
        k_consumer_state,
        kk_pipeline,
        kk_producer_state,
        qk_pipeline,
        qk_producer_state,
        alpha_pipeline,
        alpha_consumer_state,
        beta_pipeline,
        beta_consumer_state,
        is_final_block: bool,
        B: cutlass.Int32,
        scale: cutlass.Float32,
        aux_tidx: cutlass.Int32,
    ):
        blk_q = self.BLK_Q
        blk_kv = self.BLK_KV
        d = self.D
        tile_shape_qk = (blk_q, blk_kv, d)
        k_stage = k_consumer_state.index
        q_stage = q_consumer_state.index
        alpha_stage = alpha_consumer_state.index
        beta_stage = beta_consumer_state.index
        kk_stage = kk_producer_state.index
        qk_stage = qk_producer_state.index

        mma_atom_4w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))
        qk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_qk
        )
        kk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_qk
        )
        qk_thr_mma = qk_tiled_mma.get_slice(aux_tidx)
        kk_thr_mma = kk_tiled_mma.get_slice(aux_tidx)

        ldsm_n4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        sQ_k = sQ_SD[None, None, q_stage]
        sK_SD_k = sK_SD[None, None, k_stage]

        qk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, qk_tiled_mma)
        qk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, qk_tiled_mma)
        qk_thr_copy_A = qk_tiled_copy_A.get_slice(aux_tidx)
        qk_thr_copy_B = qk_tiled_copy_B.get_slice(aux_tidx)
        tQKrQ = qk_thr_mma.make_fragment_A(qk_thr_mma.partition_A(sQ_k))
        tQKrQ_cv = qk_thr_copy_A.retile(tQKrQ)
        tQKsQ = qk_thr_copy_A.partition_S(sQ_SD)
        tQKrK = qk_thr_mma.make_fragment_B(qk_thr_mma.partition_B(sK_SD_k))
        tQKrK_cv = qk_thr_copy_B.retile(tQKrK)
        tQKsK = qk_thr_copy_B.partition_S(sK_SD)

        kk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, kk_tiled_mma)
        kk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, kk_tiled_mma)
        kk_thr_copy_A = kk_tiled_copy_A.get_slice(aux_tidx)
        kk_thr_copy_B = kk_tiled_copy_B.get_slice(aux_tidx)
        tKKrA = kk_thr_mma.make_fragment_A(kk_thr_mma.partition_A(sK_SD_k))
        tKKrA_cv = kk_thr_copy_A.retile(tKKrA)
        tKKsA = kk_thr_copy_A.partition_S(sK_SD)
        tKKrB = kk_thr_mma.make_fragment_B(kk_thr_mma.partition_B(sK_SD_k))
        tKKrB_cv = kk_thr_copy_B.retile(tKKrB)
        tKKsB = kk_thr_copy_B.partition_S(sK_SD)

        cMqk = cute.make_identity_tensor((blk_q, blk_kv))
        tQKcMqk = qk_thr_mma.partition_C(cMqk)
        tKKcMkk = kk_thr_mma.partition_C(cMqk)

        # ── KK^T, inverse ──
        k_pipeline.consumer_wait(k_consumer_state)
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_wait(alpha_consumer_state)
            cute.arch.fence_view_async_shared()
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.consumer_wait(beta_consumer_state)
            cute.arch.fence_view_async_shared()
        cute.copy(kk_tiled_copy_A, tKKsA[None, None, None, k_stage], tKKrA_cv)
        cute.copy(kk_tiled_copy_B, tKKsB[None, None, None, k_stage], tKKrB_cv)
        tKKrKK = cute.make_rmem_tensor(
            kk_thr_mma.partition_shape_C((blk_kv, blk_kv)), self.acc_dtype
        )
        tKKrKK.fill(self.acc_dtype(0.0))
        cute.gemm(kk_tiled_mma, tKKrKK, tKKrA, tKKrB, tKKrKK)
        self.kk_epi(tKKrKK, tKKcMkk, sAlpha, sBeta, alpha_stage, beta_stage)
        self.qk_or_kk_mask(tKKrKK, tKKcMkk, is_final_block, B)
        kk_pipeline.producer_acquire(kk_producer_state)
        self._kk_store_and_inv(
            tKKrKK,
            kk_tiled_mma,
            aux_tidx,
            sKK_inv[None, None, kk_stage],
            sKK_opd[None, None, kk_stage],
            sBeta,
            beta_stage,
            tKKcMkk,
        )
        cute.arch.fence_view_async_shared()
        kk_pipeline.producer_commit(kk_producer_state)
        kk_producer_state.advance()
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.consumer_release(beta_consumer_state)
            beta_consumer_state.advance()

        # ── QK^T ──
        q_pipeline.consumer_wait(q_consumer_state)
        cute.copy(qk_tiled_copy_A, tQKsQ[None, None, None, q_stage], tQKrQ_cv)
        cute.copy(qk_tiled_copy_B, tQKsK[None, None, None, k_stage], tQKrK_cv)
        tQKrQK = cute.make_rmem_tensor(
            qk_thr_mma.partition_shape_C((blk_q, blk_kv)), self.acc_dtype
        )
        tQKrQK.fill(self.acc_dtype(0.0))
        cute.gemm(qk_tiled_mma, tQKrQK, tQKrQ, tQKrK, tQKrQK)
        q_pipeline.consumer_release(q_consumer_state)
        q_consumer_state.advance()
        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        self.qk_epi(tQKrQK, tQKcMqk, sAlpha, alpha_stage, scale)
        self.qk_or_kk_mask(tQKrQK, tQKcMqk, is_final_block, B)
        qk_pipeline.producer_acquire(qk_producer_state)
        self.qk_store(tQKrQK, sQK[None, None, qk_stage], qk_tiled_mma, aux_tidx)
        cute.arch.fence_view_async_shared()
        qk_pipeline.producer_commit(qk_producer_state)
        qk_producer_state.advance()
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_release(alpha_consumer_state)
            alpha_consumer_state.advance()
        return (
            q_consumer_state,
            k_consumer_state,
            kk_producer_state,
            qk_producer_state,
            alpha_consumer_state,
            beta_consumer_state,
        )

    @cute.jit
    def run_aux_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        q_pipeline,
        k_pipeline,
        kk_pipeline,
        qk_pipeline,
        alpha_pipeline,
        beta_pipeline,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        aux_tidx: cutlass.Int32,
        num_blocks: cutlass.Int32,
    ):
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        kk_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kk_stage
        )
        qk_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.qk_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        beta_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        # Same per-block flags as the stock kernel: the first block is masked as
        # a final block with B = min(seq_len, BLK).
        first_B = work_desc.seq_len
        if first_B > cutlass.Int32(self.BLK_KV):
            first_B = cutlass.Int32(self.BLK_KV)
        (
            q_consumer_state,
            k_consumer_state,
            kk_producer_state,
            qk_producer_state,
            alpha_consumer_state,
            beta_consumer_state,
        ) = self.vs_aux_body(
            sQ_SD,
            sK_SD,
            sQK,
            sKK_inv,
            sKK_opd,
            sAlpha,
            sBeta,
            q_pipeline,
            q_consumer_state,
            k_pipeline,
            k_consumer_state,
            kk_pipeline,
            kk_producer_state,
            qk_pipeline,
            qk_producer_state,
            alpha_pipeline,
            alpha_consumer_state,
            beta_pipeline,
            beta_consumer_state,
            True,
            first_B,
            scale,
            aux_tidx,
        )
        for blk in cutlass.range(
            cutlass.Int32(1), num_blocks - cutlass.Int32(1), cutlass.Int32(1), unroll=1
        ):
            (
                q_consumer_state,
                k_consumer_state,
                kk_producer_state,
                qk_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.vs_aux_body(
                sQ_SD,
                sK_SD,
                sQK,
                sKK_inv,
                sKK_opd,
                sAlpha,
                sBeta,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                kk_pipeline,
                kk_producer_state,
                qk_pipeline,
                qk_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                False,
                cutlass.Int32(self.BLK_KV),
                scale,
                aux_tidx,
            )
        if num_blocks != cutlass.Int32(1):
            last_blk = num_blocks - cutlass.Int32(1)
            last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
            (
                q_consumer_state,
                k_consumer_state,
                kk_producer_state,
                qk_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.vs_aux_body(
                sQ_SD,
                sK_SD,
                sQK,
                sKK_inv,
                sKK_opd,
                sAlpha,
                sBeta,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                kk_pipeline,
                kk_producer_state,
                qk_pipeline,
                qk_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                True,
                last_B,
                scale,
                aux_tidx,
            )

    # ── state WG: V-dependent MMAs on the CTA's 64 V rows ──

    @cute.jit
    def vs_state_body(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,  # (DV, BlkKV, StagesV)
        sQK: cute.Tensor,  # (BlkQ, BlkKV, qk_stage)
        sKK_opd: cute.Tensor,  # (BlkKV, BlkKV, kk_stage)
        sO: cute.Tensor,  # (DV, BlkQ, StagesO)
        sAlpha: cute.Tensor,
        kv_tiled_mma,
        q_pipeline,
        q_consumer_state,
        k_pipeline,
        k_consumer_state,
        v_pipeline,
        v_consumer_state,
        o_pipeline,
        o_producer_state,
        kk_pipeline,
        kk_consumer_state,
        qk_pipeline,
        qk_consumer_state,
        alpha_pipeline,
        alpha_consumer_state,
        is_first_block: bool,
        is_final_block: bool,
        B: cutlass.Int32,
        tKVrKV: cute.Tensor,
        scale: cutlass.Float32,
        thread_idx: cutlass.Int32,
    ):
        blk_q = self.BLK_Q
        blk_kv = self.BLK_KV
        d = self.D
        dv = self.DV
        tile_shape_o1 = (dv, blk_q, d)
        tile_shape_o2 = (dv, blk_q, blk_kv)
        tile_shape_sk = (dv, blk_kv, d)
        tile_shape_newv = (dv, blk_kv, blk_kv)
        k_stage = k_consumer_state.index
        q_stage = q_consumer_state.index
        v_stage = v_consumer_state.index
        o_stage = cutlass.Int32(0)
        alpha_stage = alpha_consumer_state.index
        kk_stage = kk_consumer_state.index
        qk_stage = qk_consumer_state.index

        mma_atom = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))
        # 4 warps x 16M = 64 V rows; same per-warp tile as the stock 8-warp MMAs.
        o1_tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_o1
        )
        o2_tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_o2
        )
        sk_tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_sk
        )
        newv_tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_newv
        )

        sk_thr_mma = sk_tiled_mma.get_slice(thread_idx)
        newv_thr_mma = newv_tiled_mma.get_slice(thread_idx)
        o1_thr_mma = o1_tiled_mma.get_slice(thread_idx)
        o2_thr_mma = o2_tiled_mma.get_slice(thread_idx)
        kv_thr_mma = kv_tiled_mma.get_slice(thread_idx)

        ldsm_n4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        ldsm_t4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )

        sK_DS_k = sK_DS[None, None, k_stage]
        sKK_opd_k = sKK_opd[None, None, kk_stage]
        sQK_k = sQK[None, None, qk_stage]

        # SK B: K from sK_SD with LDSM_N; SK C: V from sV_DS with LDSM_T
        sk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, sk_tiled_mma)
        sk_tiled_copy_C = cute.make_tiled_copy_C(ldsm_t4, sk_tiled_mma)
        sk_thr_copy_B = sk_tiled_copy_B.get_slice(thread_idx)
        sk_thr_copy_C = sk_tiled_copy_C.get_slice(thread_idx)
        tSKrK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_B(cute.slice_(tile_shape_sk, (0, None, None))),
            self.dtype,
        )
        tSKrK_cv = sk_thr_copy_B.retile(tSKrK)
        tSKsK = sk_thr_copy_B.partition_S(sK_SD)

        newv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, newv_tiled_mma)
        newv_thr_copy_B = newv_tiled_copy_B.get_slice(thread_idx)
        tNewVrB = newv_thr_mma.make_fragment_B(newv_thr_mma.partition_B(sKK_opd_k))
        tNewVrB_cv = newv_thr_copy_B.retile(tNewVrB)
        tNewVsB = newv_thr_copy_B.partition_S(sKK_opd_k)

        kv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_t4, kv_tiled_mma)
        kv_thr_copy_B = kv_tiled_copy_B.get_slice(thread_idx)
        tKVrK = kv_thr_mma.make_fragment_B(kv_thr_mma.partition_B(sK_DS_k))
        tKVrK_cv = kv_thr_copy_B.retile(tKVrK)
        tKVsK = kv_thr_copy_B.partition_S(sK_DS)

        o1_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o1_tiled_mma)
        o2_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o2_tiled_mma)
        o1_thr_copy_B = o1_tiled_copy_B.get_slice(thread_idx)
        o2_thr_copy_B = o2_tiled_copy_B.get_slice(thread_idx)
        tOrQ = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_B(cute.slice_(tile_shape_o1, (0, None, None))),
            self.dtype,
        )
        tOrQ_cv = o1_thr_copy_B.retile(tOrQ)
        tOsQ = o1_thr_copy_B.partition_S(sQ_SD)
        tOrQK = o2_thr_mma.make_fragment_B(o2_thr_mma.partition_B(sQK_k))
        tOrQK_cv = o2_thr_copy_B.retile(tOrQK)
        tOsQK = o2_thr_copy_B.partition_S(sQK_k)

        o_stsm = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        o_tiled_copy_r2s = cute.make_tiled_copy_C(o_stsm, o1_tiled_mma)
        o_thr_copy_r2s = o_tiled_copy_r2s.get_slice(thread_idx)
        tOsO = o_thr_copy_r2s.partition_D(sO)

        cO = cute.make_identity_tensor((dv, blk_q))
        tOcO = o1_thr_mma.partition_C(cO)
        cSK = cute.make_identity_tensor((dv, blk_kv))
        tSKcSK = sk_thr_mma.partition_C(cSK)
        cV = cute.make_identity_tensor((dv, blk_kv))
        tKVcV = kv_thr_mma.partition_A(cV)

        # ── O1 = S @ Q^T ──
        q_pipeline.consumer_wait(q_consumer_state)
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_wait(alpha_consumer_state)
            cute.arch.fence_view_async_shared()
        tOrO = cute.make_rmem_tensor(o1_thr_mma.partition_shape_C((dv, blk_q)), self.acc_dtype)
        tOrO.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            cute.copy(o1_tiled_copy_B, tOsQ[None, None, None, q_stage], tOrQ_cv)
            tOrKV = SM80.make_acc_into_op(tKVrKV, o1_tiled_mma, self.dtype)
            cute.gemm(o1_tiled_mma, tOrO, tOrKV, tOrQ, tOrO)
            self.o1_epi(tOrO, tOcO, sAlpha, alpha_stage, scale)
        q_pipeline.consumer_release(q_consumer_state)
        q_consumer_state.advance()

        # ── SK = S @ K^T ──
        k_pipeline.consumer_wait(k_consumer_state)
        tSKrSK = cute.make_rmem_tensor(sk_thr_mma.partition_shape_C((dv, blk_kv)), self.acc_dtype)
        tSKrSK.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            tSKrS = SM80.make_acc_into_op(tKVrKV, sk_tiled_mma, self.dtype)
            cute.copy(sk_tiled_copy_B, tSKsK[None, None, None, k_stage], tSKrK_cv)
            cute.gemm(sk_tiled_mma, tSKrSK, tSKrS, tSKrK, tSKrSK)

        v_pipeline.consumer_wait(v_consumer_state)
        tSKrV = self.sk_load_v(tSKrSK, sV_DS, sk_tiled_copy_C, sk_thr_copy_C, v_stage)
        if cutlass.const_expr(not is_first_block):
            self.sk_epi(tSKrSK, tSKcSK, sAlpha, alpha_stage)
            for i in cutlass.range_constexpr(cute.size(tSKrV)):
                tSKrV[i] = tSKrV[i] - self.dtype(tSKrSK[i])

        # ── NewV = (V - SK) @ T^T ──
        tNewVrA = SM80.make_acc_into_op(tSKrV, newv_tiled_mma, self.dtype)
        tNewVrC = cute.make_rmem_tensor(
            newv_thr_mma.partition_shape_C((dv, blk_kv)), self.acc_dtype
        )
        kk_pipeline.consumer_wait(kk_consumer_state)
        cute.copy(newv_tiled_copy_B, tNewVsB, tNewVrB_cv)
        tNewVrC.fill(self.acc_dtype(0.0))
        cute.gemm(newv_tiled_mma, tNewVrC, tNewVrA, tNewVrB, tNewVrC)
        kk_pipeline.consumer_release(kk_consumer_state)
        kk_consumer_state.advance()
        v_pipeline.consumer_release(v_consumer_state)
        v_consumer_state.advance()

        # ── O2 = O1 + NewV @ QK ──
        tOrNewV = SM80.make_acc_into_op(tNewVrC, o2_tiled_mma, self.dtype)
        qk_pipeline.consumer_wait(qk_consumer_state)
        cute.copy(o2_tiled_copy_B, tOsQK, tOrQK_cv)
        cute.gemm(o2_tiled_mma, tOrO, tOrNewV, tOrQK, tOrO)
        qk_pipeline.consumer_release(qk_consumer_state)
        qk_consumer_state.advance()

        # ── O store to smem ──
        o_pipeline.producer_acquire(o_producer_state)
        self.o_store(
            tOrO,
            tOsO[None, None, None, o_stage],
            o_tiled_copy_r2s,
            o_thr_copy_r2s,
        )
        o_pipeline.producer_commit(o_producer_state)
        o_producer_state.advance()

        # ── KV state update ──
        block_coeff = cutlass.Float32(1.0)
        if cutlass.const_expr(self.needs_alpha):
            block_coeff = cutlass.Float32(
                sAlpha[B - cutlass.Int32(1), AlphaProcessor.CUMPROD, alpha_stage]
            )
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            tKVrKV[i] = block_coeff * tKVrKV[i]

        self.kv_decay_v(tOrNewV, tKVcV, sAlpha, alpha_stage, is_final_block, B)

        cute.copy(kv_tiled_copy_B, tKVsK[None, None, None, k_stage], tKVrK_cv)
        cute.gemm(kv_tiled_mma, tKVrKV, tOrNewV, tKVrK, tKVrKV)
        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_release(alpha_consumer_state)
            alpha_consumer_state.advance()
        return (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            o_producer_state,
            kk_consumer_state,
            qk_consumer_state,
            alpha_consumer_state,
        )

    @cute.jit
    def run_state_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sQK: cute.Tensor,
        sKK_opd: cute.Tensor,
        sO: cute.Tensor,
        sAlpha: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        o_pipeline,
        kk_pipeline,
        qk_pipeline,
        alpha_pipeline,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        work_desc: WorkDesc,
        v_half: cutlass.Int32,
        scale: cutlass.Float32,
        state_tidx: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.v_stage
        )
        o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.o_stage
        )
        kk_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kk_stage
        )
        qk_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.qk_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((4, 1, 1)),
            permutation_mnk=(self.DV, self.D, self.BLK_KV),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(state_tidx)
        tKVrKV = cute.make_rmem_tensor(
            kv_thr_mma.partition_shape_C((self.DV, self.D)), self.acc_dtype
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        state_layout = cute.make_ordered_layout(
            (self.D, self.D, num_sab_heads, num_seqs), order=(0, 1, 2, 3)
        )
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        mState = cute.make_tensor(g_state.iterator, state_layout)
        gStateKV = cute.flat_divide(
            mState[None, None, o_head_idx, work_desc.seq_idx], (self.D, self.DV)
        )[None, None, 0, v_half]
        if cutlass.const_expr(self.needs_init_state):
            mInitState = cute.make_tensor(g_init_state.iterator, state_layout)
            gInitKV = cute.flat_divide(
                mInitState[None, None, o_head_idx, work_desc.seq_idx],
                (self.D, self.DV),
            )[None, None, 0, v_half]
            self.kv_load_vs(tKVrKV, gInitKV, kv_thr_mma)

        first_B = work_desc.seq_len
        if first_B > cutlass.Int32(self.BLK_KV):
            first_B = cutlass.Int32(self.BLK_KV)
        (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            o_producer_state,
            kk_consumer_state,
            qk_consumer_state,
            alpha_consumer_state,
        ) = self.vs_state_body(
            sQ_SD,
            sK_SD,
            sK_DS,
            sV_DS,
            sQK,
            sKK_opd,
            sO,
            sAlpha,
            kv_tiled_mma,
            q_pipeline,
            q_consumer_state,
            k_pipeline,
            k_consumer_state,
            v_pipeline,
            v_consumer_state,
            o_pipeline,
            o_producer_state,
            kk_pipeline,
            kk_consumer_state,
            qk_pipeline,
            qk_consumer_state,
            alpha_pipeline,
            alpha_consumer_state,
            not self.needs_init_state,
            True,
            first_B,
            tKVrKV,
            scale,
            state_tidx,
        )
        for blk in cutlass.range(
            cutlass.Int32(1), num_blocks - cutlass.Int32(1), cutlass.Int32(1), unroll=1
        ):
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                kk_consumer_state,
                qk_consumer_state,
                alpha_consumer_state,
            ) = self.vs_state_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_opd,
                sO,
                sAlpha,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                kk_pipeline,
                kk_consumer_state,
                qk_pipeline,
                qk_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                False,
                False,
                cutlass.Int32(self.BLK_KV),
                tKVrKV,
                scale,
                state_tidx,
            )
        if num_blocks != cutlass.Int32(1):
            last_blk = num_blocks - cutlass.Int32(1)
            last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                kk_consumer_state,
                qk_consumer_state,
                alpha_consumer_state,
            ) = self.vs_state_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_opd,
                sO,
                sAlpha,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                kk_pipeline,
                kk_consumer_state,
                qk_pipeline,
                qk_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                False,
                True,
                last_B,
                tKVrKV,
                scale,
                state_tidx,
            )
        self.kv_store_vs(tKVrKV, gStateKV, kv_thr_mma)

    # ── Kernel entry point ──

    @cute.jit
    def __call__(
        self,
        g_q: cute.Tensor,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_o: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        g_tensormaps: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
        grid_x: int,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_storage_layout = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        q_smem_layout = cute.slice_(q_storage_layout, (None, None, 0))
        k_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0, 2])
        v_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.DV, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_storage_layout_ds = cute.select(v_storage_layout_sd, [1, 0, 2])
        k_smem_layout = cute.slice_(k_storage_layout_ds, (None, None, 0))
        v_smem_layout = cute.slice_(v_storage_layout_ds, (None, None, 0))
        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_storage_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.DV, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        o_smem_layout = cute.slice_(o_storage_layout, (None, None, 0))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op, g_q, q_smem_layout, (self.BLK_Q, self.D)
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op, g_k, k_smem_layout, (self.D, self.BLK_KV)
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op, g_v, v_smem_layout, (self.DV, self.BLK_KV)
        )

        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            tma_store_op, g_o, o_smem_layout, (self.DV, self.BLK_Q)
        )

        dtype_bytes = self.dtype.width // 8
        self.tma_load_q_bytes = cute.size(q_smem_layout) * dtype_bytes
        self.tma_load_k_bytes = cute.size(k_smem_layout) * dtype_bytes
        self.tma_load_v_bytes = cute.size(v_smem_layout) * dtype_bytes

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV, self.qk_stage), order=(1, 0, 2)
        )
        kk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.kk_stage), order=(1, 0, 2)
        )
        alpha_storage_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        beta_storage_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))

        @cute.struct
        class SharedStorageVS:
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.o_stage * 2]
            alpha_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.alpha_beta_stage * 2]
            beta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.alpha_beta_stage * 2]
            kk_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kk_stage * 2]
            qk_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.qk_stage * 2]

            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(q_storage_layout)],
                128,
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)],
                128,
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_storage_layout_sd)],
                128,
            ]
            smem_qk: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(qk_storage_layout)],
                16,
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[self.inverse_dtype, cute.cosize(kk_storage_layout)],
                16,
            ]
            smem_o: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(o_storage_layout)],
                128,
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_storage_layout)],
                16,
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_storage_layout)],
                16,
            ]

        self.shared_storage = SharedStorageVS

        self.kernel(
            g_alpha,
            g_beta,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            g_state,
            g_init_state,
            g_state_checkpoints,
            checkpoint_cu_starts,
            g_tensormaps,
            cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            num_seqs,
            total_checkpoints,
            checkpoint_every_n_tokens,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(384, 1, 1),
            max_number_threads=(384, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        g_tensormaps: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
    ):
        NUM_LOAD_WARP_GROUPS = 1
        NUM_MMA_WARP_GROUPS = 2
        THREADS_PER_WARP_GROUP = 128
        WARPS_PER_WARP_GROUP = 4
        MIN_BLOCKS_PER_MP = 1
        MAX_THREADS_PER_BLOCK = (
            NUM_LOAD_WARP_GROUPS + NUM_MMA_WARP_GROUPS
        ) * THREADS_PER_WARP_GROUP
        load_registers, mma_registers = self.get_register_requirements(
            MAX_THREADS_PER_BLOCK,
            MIN_BLOCKS_PER_MP,
            NUM_MMA_WARP_GROUPS,
            THREADS_PER_WARP_GROUP,
        )

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // cutlass.Int32(THREADS_PER_WARP_GROUP))
        ldst_warp_role = cute.arch.make_warp_uniform(warp_idx % cutlass.Int32(WARPS_PER_WARP_GROUP))

        if warp_idx == LoadStoreWarpRole.LOAD_QKV:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_o)

        work_desc, v_half = self.get_next_work_vs(
            cu_seqlens,
            num_q_heads,
            num_v_heads,
            num_sab_heads,
        )
        v_off = v_half * cutlass.Int32(self.DV)
        tok_end = work_desc.tok_offset + work_desc.seq_len
        num_blocks = (
            work_desc.seq_len + cutlass.Int32(self.BLK_KV) - cutlass.Int32(1)
        ) // cutlass.Int32(self.BLK_KV)

        math_tidx = tidx - cutlass.Int32(THREADS_PER_WARP_GROUP)
        wg_tidx = math_tidx % cutlass.Int32(THREADS_PER_WARP_GROUP)

        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)

        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        sQ_SD = storage.smem_q.get_tensor(q_layout_sd.outer, swizzle=q_layout_sd.inner)

        k_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_layout_ds = cute.select(k_layout_sd, [1, 0, 2])
        sK_SD = storage.smem_k.get_tensor(k_layout_sd.outer, swizzle=k_layout_sd.inner)
        sK_DS = storage.smem_k.get_tensor(k_layout_ds.outer, swizzle=k_layout_ds.inner)

        v_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.DV, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_layout_ds = cute.select(v_layout_sd, [1, 0, 2])
        sV_DS = storage.smem_v.get_tensor(v_layout_ds.outer, swizzle=v_layout_ds.inner)

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV, self.qk_stage), order=(1, 0, 2)
        )
        sQK = storage.smem_qk.get_tensor(qk_layout)

        kk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.kk_stage), order=(1, 0, 2)
        )
        sKK_inv = storage.smem_kk.get_tensor(kk_layout)
        kk_opd_ptr = cute.recast_ptr(storage.smem_kk.data_ptr(), dtype=self.dtype)
        sKK_opd = cute.make_tensor(kk_opd_ptr, kk_layout)

        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.DV, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        sO = storage.smem_o.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        alpha_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        beta_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))
        sBeta = storage.smem_beta.get_tensor(beta_layout)

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # q/k: both math WGs (8 warps); v: the state WG only (4 warps).
        qk_load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP
        )
        v_load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, WARPS_PER_WARP_GROUP
        )
        vector_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        alpha_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        beta_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS_PER_WARP_GROUP
        )
        wg_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS_PER_WARP_GROUP)
        o_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=load_producer_group,
            consumer_group=qk_load_consumer_group,
            tx_count=self.tma_load_q_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        k_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            num_stages=self.k_stage,
            producer_group=load_producer_group,
            consumer_group=qk_load_consumer_group,
            tx_count=self.tma_load_k_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_mbar_ptr.data_ptr(),
            num_stages=self.v_stage,
            producer_group=load_producer_group,
            consumer_group=v_load_consumer_group,
            tx_count=self.tma_load_v_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        o_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.o_mbar_ptr.data_ptr(),
            num_stages=self.o_stage,
            producer_group=wg_group,
            consumer_group=o_consumer_group,
        )
        alpha_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.alpha_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=alpha_consumer_group,
        )
        beta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.beta_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=beta_consumer_group,
        )
        kk_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.kk_mbar_ptr.data_ptr(),
            num_stages=self.kk_stage,
            producer_group=wg_group,
            consumer_group=wg_group,
        )
        qk_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.qk_mbar_ptr.data_ptr(),
            num_stages=self.qk_stage,
            producer_group=wg_group,
            consumer_group=wg_group,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if work_desc.seq_len != cutlass.Int32(0) and warp_group_idx == WarpGroupRole.LDST:
            cute.arch.setmaxregister_decrease(load_registers)
            if ldst_warp_role == LoadStoreWarpRole.LOAD_QKV:
                self.run_load_qkv_role_vs(
                    sQ_SD,
                    sK_DS,
                    sV_DS,
                    tma_atom_q,
                    tma_tensor_q,
                    tma_atom_k,
                    tma_tensor_k,
                    tma_atom_v,
                    tma_tensor_v,
                    q_pipeline,
                    k_pipeline,
                    v_pipeline,
                    num_blocks,
                    work_desc.tok_offset,
                    work_desc.q_head_idx(),
                    work_desc.k_head_idx(num_q_heads, num_v_heads),
                    work_desc.v_head_idx(),
                    v_off,
                )
            elif ldst_warp_role == LoadStoreWarpRole.STORE_O:
                CollectiveStoreTma(self.BLK_Q, self.DV).run(
                    sO,
                    tma_atom_o,
                    cute.domain_offset((v_off, cutlass.Int32(0), cutlass.Int32(0)), tma_tensor_o),
                    g_tensormaps,
                    o_pipeline,
                    num_blocks,
                    work_desc,
                    num_seqs,
                    self.o_stage,
                    num_q_heads,
                    num_v_heads,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:
                self.run_load_beta_role(
                    sBeta,
                    g_beta,
                    beta_pipeline,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_ALPHA:
                self.run_load_alpha_role(
                    sAlpha,
                    g_alpha,
                    alpha_pipeline,
                    scale,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                )
        elif work_desc.seq_len != cutlass.Int32(0):
            cute.arch.setmaxregister_increase(mma_registers)
            if warp_group_idx == WarpGroupRole.MATH_KK:
                self.run_aux_role(
                    sQ_SD,
                    sK_SD,
                    sQK,
                    sKK_inv,
                    sKK_opd,
                    sAlpha,
                    sBeta,
                    q_pipeline,
                    k_pipeline,
                    kk_pipeline,
                    qk_pipeline,
                    alpha_pipeline,
                    beta_pipeline,
                    work_desc,
                    scale,
                    wg_tidx,
                    num_blocks,
                )
            else:
                self.run_state_role(
                    sQ_SD,
                    sK_SD,
                    sK_DS,
                    sV_DS,
                    sQK,
                    sKK_opd,
                    sO,
                    sAlpha,
                    q_pipeline,
                    k_pipeline,
                    v_pipeline,
                    o_pipeline,
                    kk_pipeline,
                    qk_pipeline,
                    alpha_pipeline,
                    g_state,
                    g_init_state,
                    work_desc,
                    v_half,
                    scale,
                    wg_tidx,
                    num_blocks,
                    num_q_heads,
                    num_v_heads,
                    num_sab_heads,
                    num_seqs,
                )


def vsplit_pays_off(num_seqs: int, num_long_seqs: int, num_sab_heads: int, sm_count: int) -> bool:
    """Whether the V-split x2 kernel runs: it doubles the grid to 2 CTAs per (sequence,
    head) and pays only while the long sequences' CTAs still fit in one wave. Sequences
    of one token are short CTAs and are not counted (``num_long_seqs`` bounds the
    rest). At 48 heads on 188 SMs: exactly one long sequence."""
    n_long = min(int(num_long_seqs), num_seqs)
    if n_long < 1:
        return False
    return n_long * num_sab_heads * 2 <= sm_count


def run(
    o: torch.Tensor,  # (total_seqlen, num_o_heads, D) fp16/bf16, output
    state: torch.Tensor,  # (num_seqs, num_sab_heads, D, D) fp32, output
    q: torch.Tensor,  # (total_seqlen, num_q_heads, D)
    k: torch.Tensor,  # (total_seqlen, num_k_heads, D)
    v: torch.Tensor,  # (total_seqlen, num_v_heads, D)
    init_state: torch.Tensor,  # (num_seqs, num_sab_heads, D, D) fp32
    alpha: torch.Tensor,  # (total_seqlen, num_sab_heads) fp32, linear-space decay
    beta: torch.Tensor,  # (total_seqlen, num_sab_heads) fp32
    cu_seqlens: torch.Tensor,  # (num_seqs + 1,) int64
    scale: float,
    num_long_seqs: int,
) -> None:
    """The non-CP prefill: V-split x2 when ``vsplit_pays_off``, else the installed
    fully fused launcher (``delta_rule_prefill_dsl``), with the same arguments."""
    import cuda.bindings.driver as cuda_driver

    D = q.shape[-1]
    num_seqs = cu_seqlens.shape[0] - 1
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)

    if D != 128 or not vsplit_pays_off(
        num_seqs, num_long_seqs, num_sab_heads, get_device_sm_count(q.device)
    ):
        _fully_fused_prefill(o, state, q, k, v, init_state, alpha, beta, cu_seqlens, scale)
        return

    if not _FullyFusedDeltaRuleSm120.can_implement(
        num_q_heads, num_k_heads, num_v_heads, D, q.element_size()
    ):
        raise RuntimeError("can_implement failed")

    needs_alpha = alpha is not None
    needs_beta = beta is not None
    needs_init_state = init_state is not None
    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }.get(q.dtype)
    if kernel_dtype is None:
        raise RuntimeError(f"DSL kernel only supports fp16/bf16 inputs, got {q.dtype}")

    if k.dtype != q.dtype or v.dtype != q.dtype or o.dtype != q.dtype:
        raise RuntimeError(
            f"q/k/v/o dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}, {o.dtype}"
        )
    if alpha is not None and alpha.dtype != torch.float32:
        raise RuntimeError(f"alpha must have dtype torch.float32, got {alpha.dtype}")
    if beta is not None and beta.dtype != torch.float32:
        raise RuntimeError(f"beta must have dtype torch.float32, got {beta.dtype}")
    if init_state is not None and init_state.dtype != torch.float32:
        raise RuntimeError(f"init_state must have dtype torch.float32, got {init_state.dtype}")
    if state.dtype != torch.float32:
        raise RuntimeError(f"state must have dtype torch.float32, got {state.dtype}")
    if cu_seqlens.dtype != torch.int64:
        raise RuntimeError(f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}")

    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("o", o),
        ("state", state),
        ("cu_seqlens", cu_seqlens),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (("alpha", alpha), ("beta", beta), ("init_state", init_state)):
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    total_seqlen = q.shape[0]
    num_o_heads = o.shape[1]
    q_tma = q.as_strided(
        (total_seqlen, D, num_q_heads),
        (num_q_heads * D, 1, D),
    )
    k_tma = k.as_strided(
        (D, total_seqlen, num_k_heads),
        (1, num_k_heads * D, D),
    )
    v_tma = v.as_strided(
        (D, total_seqlen, num_v_heads),
        (1, num_v_heads * D, D),
    )
    o_tma = o.as_strided(
        (D, total_seqlen, num_o_heads),
        (1, num_o_heads * D, D),
    )
    total_checkpoints = 1

    workspace_size = get_device_sm_count(q.device) * 128
    tensormaps_t = _get_cache_buf("gdn_prefill_tensormaps", workspace_size, q.device)

    stream_val = torch.cuda.current_stream().cuda_stream
    stream = cuda_driver.CUstream(stream_val)

    def from_dlpack(*args, **kwargs):  # the TVM-FFI entry, as the fully fused launcher's
        return cute.runtime.from_dlpack(*args, **{**kwargs, "enable_tvm_ffi": True})

    # Keep head counts and varlen extents runtime values across cached compiles.
    q_cute = from_dlpack(q_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    k_cute = from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    v_cute = from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    o_cute = from_dlpack(o_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    alpha_cute = (
        from_dlpack(alpha.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_alpha
        else None
    )
    beta_cute = (
        from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_beta
        else None
    )
    state_cute = from_dlpack(state.reshape(-1), assumed_align=16).mark_layout_dynamic()
    init_state_cute = (
        from_dlpack(init_state.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_init_state
        else None
    )
    tensormaps_cute = from_dlpack(tensormaps_t, assumed_align=128).mark_layout_dynamic()
    cu_cute = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()

    delta_rule_kernel = _VSplitDeltaRuleSm120(
        needs_alpha, needs_beta, needs_init_state, kernel_dtype
    )
    grid_x = num_seqs * num_sab_heads * delta_rule_kernel.v_split

    kernel_args = (
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        alpha_cute,
        beta_cute,
        state_cute,
        init_state_cute,
        None,
        None,
        tensormaps_cute,
        cu_cute,
        cutlass.Float32(scale),
        cutlass.Int32(num_q_heads),
        cutlass.Int32(num_k_heads),
        cutlass.Int32(num_v_heads),
        cutlass.Int32(num_sab_heads),
        cutlass.Int32(num_seqs),
        cutlass.Int32(total_checkpoints),
        cutlass.Int32(0),
        grid_x,
        stream,
    )
    compiled_delta_rule_kernel = cached_compile(
        delta_rule_kernel,
        *kernel_args,
        compile_options=sm12x_compile_options(q.device),
    )
    compiled_delta_rule_kernel(*kernel_args)
