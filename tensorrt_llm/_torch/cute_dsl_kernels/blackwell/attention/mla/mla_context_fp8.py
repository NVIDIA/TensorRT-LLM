# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ruff: noqa: I001, E501, F841

"""FP8 MLA context FMHA for Blackwell (SM100 family incl. SM103/GB300), written for Kimi-K3 chunked prefill.

Built on the public CuTe DSL FMHA that already ships in this tree
(``visual_gen/cute_dsl_kernels/blackwell/attention/fmha.py``, BSD-3-Clause): the softmax, correction and MMA
machinery is inherited unchanged. What this file adds, designed from scratch for MLA chunked prefill:

* **Split K operand.** MLA keys are ``[k_nope (128) | k_pe (64)]`` where ``k_pe`` is shared by all heads. The key
  tile is staged in shared memory as two independent TMA-loaded pieces: ``k_nope`` read straight from the
  ``kv_b_proj`` output (strided ``[tokens, heads, 128]`` view) and ``k_pe`` read once per token through a
  broadcast (stride-0) head mode. No per-head K tensor (``cat``) is ever materialized.
* **Epilogue merge.** Each launch handles one KV partition (one cached-KV chunk, or the new tokens). Its epilogue
  folds the partition into the running ``(O, m, l)`` of the output rows (``merge_op``: 1 = merge, 2 = copy), so the
  separate softmax-stats merge kernel disappears. Rows are written with ``st.global`` (predicated), which also
  makes ragged (varlen) query lengths safe.
* Sequences with an empty KV partition are skipped by every warp role.
"""

from functools import partial
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.attention.fmha import (
    BlackwellFusedMultiHeadAttentionForward,
    make_thread_cooperative_group,
)
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.helpers import fmha_helpers as fmha_utils

LOG2_E = 1.4426950408889634


class MlaContextFp8Fmha(BlackwellFusedMultiHeadAttentionForward):
    """FP8 (E4M3) MLA context FMHA: Q[T,H,192] x (K_nope[S,H,128] | K_pe[S,64]) x V[S,H,128] -> O[T,H,128] (bf16)."""

    NOPE_DIM = 128
    ROPE_DIM = 64
    HEAD_DIM = 192
    HEAD_DIM_V = 128

    def __init__(
        self,
        mask_type: fmha_utils.MaskEnum,
        enable_skip_correction: bool = True,
        kv_stage: int = 4,
        persistent: bool = True,
    ):
        super().__init__(
            cutlass.Float32,
            cutlass.Float32,
            (128, 128),
            (self.HEAD_DIM, self.HEAD_DIM_V),
            persistent,
            mask_type,
            False,
            enable_skip_correction,
            use_tma_store=False,
        )
        self._kv_stage_cfg = kv_stage
        self.is_causal = mask_type == fmha_utils.MaskEnum.WINDOW_MASK

    def _setup_attributes(self, enable_skip_softmax: bool = False) -> None:
        super()._setup_attributes(enable_skip_softmax)
        self.kv_stage = self._kv_stage_cfg

    # ------------------------------------------------------------------------------------------
    # host side
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,  # [T, H, 192]  fp8
        kn_tensor: cute.Tensor,  # [S, H, 128]  fp8 (token/head strides arbitrary)
        kp_tensor: cute.Tensor,  # [S, 64]      fp8
        v_tensor: cute.Tensor,  # [S, H, 128]  fp8 (token/head strides arbitrary)
        o_tensor: cute.Tensor,  # [T, H, 128]  bf16, updated in place
        stats_tensor: cute.Tensor,  # [T, H, 2]    fp32 (running max in log2 domain, running sum), updated in place
        cum_seqlen_q: cute.Tensor,  # [B+1] int32
        cum_seqlen_k: cute.Tensor,  # [B+1] int32
        merge_op: cute.Tensor,  # [B]   int64: 1 merge into the running state, 2 overwrite (first partition)
        problem_size: Tuple[Int32, Int32, Int32],  # (b, s_q_max, s_k_max)
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        stream: cuda.CUstream,
    ):
        b, s_q_max, s_k_max = problem_size
        h = q_tensor.shape[1]
        d = self.HEAD_DIM
        dv = self.HEAD_DIM_V
        self.q_dtype = q_tensor.element_type
        self.k_dtype = kn_tensor.element_type
        self.v_dtype = v_tensor.element_type
        self.o_dtype = o_tensor.element_type
        s_q = q_tensor.shape[0]
        s_k = kn_tensor.shape[0]

        q_tok = cute.assume(q_tensor.stride[0], 16)
        q_head = cute.assume(q_tensor.stride[1], 16)
        kn_tok = cute.assume(kn_tensor.stride[0], 16)
        kn_head = cute.assume(kn_tensor.stride[1], 16)
        kp_tok = cute.assume(kp_tensor.stride[0], 16)
        v_tok = cute.assume(v_tensor.stride[0], 16)
        v_head = cute.assume(v_tensor.stride[1], 16)
        o_tok = cute.assume(o_tensor.stride[0], 16)
        o_head = cute.assume(o_tensor.stride[1], 16)

        # Hierarchical (h_r=1, h_k=h) head mode so the inherited scheduler / partition code applies unchanged.
        q = cute.make_tensor(
            q_tensor.iterator,
            cute.make_layout((s_q, d, ((1, h), b)), stride=(q_tok, 1, ((d, q_head), 0))),
        )
        kn = cute.make_tensor(
            kn_tensor.iterator,
            cute.make_layout(
                (s_k, self.NOPE_DIM, ((1, h), b)), stride=(kn_tok, 1, ((0, kn_head), 0))
            ),
        )
        # k_pe has one head: every head coordinate maps to the same rows (stride 0).
        kp = cute.make_tensor(
            kp_tensor.iterator,
            cute.make_layout((s_k, self.ROPE_DIM, ((1, h), b)), stride=(kp_tok, 1, ((0, 0), 0))),
        )
        v = cute.make_tensor(
            v_tensor.iterator,
            cute.make_layout((dv, s_k, ((1, h), b)), stride=(1, v_tok, ((0, v_head), 0))),
        )
        o = cute.make_tensor(
            o_tensor.iterator,
            cute.make_layout((s_q, dv, ((1, h), b)), stride=(o_tok, 1, ((dv, o_head), 0))),
        )

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            cute.shape((s_q_max, d, ((1, h), b))),
            self.cta_tiler,
            self.is_persistent,
        )
        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(kn).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)
        if cutlass.const_expr(self.q_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        self._setup_attributes(False)

        cta_group = tcgen05.CtaGroup.ONE
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = cute.nvgpu.OperandMajorMode.K
        qk_tiled_mma = self._make_qk_tiled_mma(cta_group)
        pv_tiled_mma = self._make_pv_tiled_mma(cta_group, p_major_mode, p_source)

        self.epi_tile = self.pv_mma_tiler[:2]
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        kn_tiler = (self.qk_mma_tiler[0], self.qk_mma_tiler[1], self.NOPE_DIM)
        kp_tiler = (self.qk_mma_tiler[0], self.qk_mma_tiler[1], self.ROPE_DIM)
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage
        )
        kn_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, kn_tiler, self.k_dtype, self.kv_stage
        )
        kp_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, kp_tiler, self.k_dtype, self.kv_stage
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.acc_stage
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.kv_stage
        )
        # V reuses the k_nope buffer of each stage: both tiles are 128x128 fp8 (16 KiB).
        if cutlass.const_expr(
            cute.cosize(cute.select(v_smem_layout_staged, mode=[0, 1, 2]))
            != cute.cosize(cute.select(kn_smem_layout_staged, mode=[0, 1, 2]))
        ):
            raise RuntimeError("V tile does not match the k_nope tile footprint")

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        kn_smem_layout = cute.select(kn_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_kn, tma_tensor_kn = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kn,
            kn_smem_layout,
            kn_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        kp_smem_layout = cute.select(kp_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_kp, tma_tensor_kp = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kp,
            kp_smem_layout,
            kp_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        self.tma_copy_k_bytes = cute.size_in_bytes(
            self.k_dtype, kn_smem_layout
        ) + cute.size_in_bytes(self.k_dtype, kp_smem_layout)
        self.tma_copy_v_bytes = cute.size_in_bytes(self.v_dtype, v_smem_layout)

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            p0_mma_mbar_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            p1_mma_mbar_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            s0_p1_inplace_barrier_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            s1_p0_inplace_barrier_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            tmem_holding_buf: Int32
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sKn: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(kn_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sKp: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(kp_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_kn,
            tma_tensor_kn,
            tma_atom_kp,
            tma_tensor_kp,
            tma_atom_v,
            tma_tensor_v,
            o,
            stats_tensor,
            cum_seqlen_q,
            cum_seqlen_k,
            merge_op,
            scale_softmax_log2,
            scale_softmax,
            q_smem_layout_staged,
            kn_smem_layout_staged,
            kp_smem_layout_staged,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------------------------------
    # device side
    # ------------------------------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_kn: cute.CopyAtom,
        mKn_kdl: cute.Tensor,
        tma_atom_kp: cute.CopyAtom,
        mKp_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO: cute.Tensor,
        mStats: cute.Tensor,
        cum_seqlen_q: cute.Tensor,
        cum_seqlen_k: cute.Tensor,
        merge_op: cute.Tensor,
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        kn_smem_layout_staged: cute.ComposedLayout,
        kp_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        # causal partitions (new tokens, kv_len == q_len): window mask with right = 0; cached partitions: no window
        if cutlass.const_expr(self.is_causal):
            window_size_right = Int32(0)
        else:
            window_size_right = None

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kn)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kp)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_kv_full_mbar_ptr = storage.load_kv_mbar_ptr.data_ptr()
        load_kv_empty_mbar_ptr = load_kv_full_mbar_ptr + self.kv_stage
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            barrier_storage=storage.mma_s0_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_s1_producer, mma_s1_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.mma_s1_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        p0_mma_producer, p0_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p0_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        p1_mma_producer, p1_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p1_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s0_corr_producer, s0_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len((*self.softmax0_warp_ids, self.mma_warp_id))
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s0_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s1_corr_producer, s1_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len((*self.softmax1_warp_ids, self.mma_warp_id))
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s1_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s0_p1_inplace_producer, s0_p1_inplace_consumer = pipeline.PipelineAsync.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.s0_p1_inplace_barrier_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s1_p0_inplace_producer, s1_p0_inplace_consumer = pipeline.PipelineAsync.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.s1_p0_inplace_barrier_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            arch=self.arch_str,
        )
        pipeline_init_arrive(is_relaxed=True)

        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sKn = storage.sKn.get_tensor(
            kn_smem_layout_staged.outer, swizzle=kn_smem_layout_staged.inner
        )
        sKp = storage.sKp.get_tensor(
            kp_smem_layout_staged.outer, swizzle=kp_smem_layout_staged.inner
        )
        # V shares the k_nope buffer of its stage.
        sV_ptr = cute.recast_ptr(
            cute.recast_ptr(sKn.iterator, dtype=self.v_dtype),
            v_smem_layout_staged.inner,
        )
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)

        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrKn = qk_thr_mma.make_fragment_B(sKn)
        tSrKp = qk_thr_mma.make_fragment_B(sKp)
        tOrV = pv_thr_mma.make_fragment_B(sV)

        def make_tmem_tensors(
            self,
            qk_thr_mma: cute.TiledMma,
            pv_thr_mma: cute.TiledMma,
            p_tmem_layout_staged: cute.Layout,
            tmem_ptr: cute.Pointer,
        ):
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
            )
            tStS_fake = qk_thr_mma.make_fragment_C(qk_acc_shape)
            tStS = cute.make_tensor(tmem_ptr + self.tmem_s0_offset, tStS_fake.layout)
            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
            tStS0 = cute.make_tensor(tmem_ptr + self.tmem_s0_offset, tStS.layout)
            tStS1 = cute.make_tensor(tmem_ptr + self.tmem_s1_offset, tStS.layout)
            tOtO0 = cute.make_tensor(tmem_ptr + self.tmem_o0_offset, tOtO.layout)
            tOtO1 = cute.make_tensor(tmem_ptr + self.tmem_o1_offset, tOtO.layout)
            tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
            tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
            tOrP0 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_p0_offset, dtype=tOrP.dtype), tOrP.layout
            )
            tOrP1 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_p1_offset, dtype=tOrP.dtype), tOrP.layout
            )
            return tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1

        tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        pipeline_init_wait()
        # The inherited softmax(): positional value_args = (seqlen_k_total, seqlen_q_total, scale_log2, skip_thr)
        softmax_fn = partial(
            self.softmax,
            qk_thr_mma=qk_thr_mma,
            value_args=(mKn_kdl.shape[0], mQ_qdl.shape[0], scale_softmax_log2, None),
            mask_args=(None, window_size_right),
            sched_args=(tile_sched, work_tile),
        )

        # EMPTY / EPILOGUE warps: nothing to do (st.global epilogue lives in the correction warps)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                seqlen_k = cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]
                continue_cond = (
                    not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0], curr_block_coord[0], seqlen_q
                    )
                ) or (seqlen_k <= 0)
                if not continue_cond:
                    mQ_qdl_ = cute.domain_offset((cuseqlen_q, 0, ((0, 0), 0)), mQ_qdl)
                    mKn_kdl_ = cute.domain_offset(
                        (cum_seqlen_k[batch_coord], 0, ((0, 0), 0)), mKn_kdl
                    )
                    mKp_kdl_ = cute.domain_offset(
                        (cum_seqlen_k[batch_coord], 0, ((0, 0), 0)), mKp_kdl
                    )
                    mV_dkl_ = cute.domain_offset(
                        (0, cum_seqlen_k[batch_coord], ((0, 0), 0)), mV_dkl
                    )
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord[2]]
                    gKn_kdl = cute.flat_divide(mKn_kdl_, (self.qk_mma_tiler[1], self.NOPE_DIM))
                    tSgKn_kdl = qk_thr_mma.partition_B(gKn_kdl)
                    tKnsK, tKngK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_kn,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sKn, 0, 3),
                        cute.group_modes(tSgKn_kdl, 0, 3),
                    )
                    tKngK = tKngK_kdl[None, None, 0, curr_block_coord[2]]
                    gKp_kdl = cute.flat_divide(mKp_kdl_, (self.qk_mma_tiler[1], self.ROPE_DIM))
                    tSgKp_kdl = qk_thr_mma.partition_B(gKp_kdl)
                    tKpsK, tKpgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_kp,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sKp, 0, 3),
                        cute.group_modes(tSgKp_kdl, 0, 3),
                    )
                    tKpgK = tKpgK_kdl[None, None, 0, curr_block_coord[2]]
                    gV_dkl = cute.flat_divide(mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2]))
                    tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_dkl, 0, 3),
                    )
                    tVgV = tVgV_dkl[None, 0, None, curr_block_coord[2]]
                    seqlen_kv_loop_start = fmha_utils.FusedMask.get_trip_start(
                        self.mask_type, curr_block_coord, self.cta_tiler, seqlen_q, seqlen_k, None
                    )
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q,
                        seqlen_k,
                        None,
                        window_size_right,
                    )
                    # Q0
                    q0_coord = 2 * curr_block_coord[0]
                    q0_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q0_coord],
                        tQsQ[None, q0_handle.index],
                        tma_bar_ptr=q0_handle.barrier,
                    )
                    # K0
                    kv_coord = seqlen_kv_loop_start
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_kn,
                        tKngK[None, kv_coord],
                        tKnsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    cute.copy(
                        tma_atom_kp,
                        tKpgK[None, kv_coord],
                        tKpsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    # Q1
                    q1_coord = q0_coord + 1
                    q1_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q1_coord],
                        tQsQ[None, q1_handle.index],
                        tma_bar_ptr=q1_handle.barrier,
                    )
                    kv_coord += 1
                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_kn,
                            tKngK[None, kv_coord],
                            tKnsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        cute.copy(
                            tma_atom_kp,
                            tKpgK[None, kv_coord],
                            tKpsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        # Vi-1
                        v_handle, load_kv_producer = self.kv_producer_update_tx_acquire_and_advance(
                            load_kv_producer,
                            load_kv_empty_mbar_ptr,
                            load_kv_full_mbar_ptr,
                            self.tma_copy_v_bytes,
                        )
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, kv_coord - 1],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=load_kv_full_mbar_ptr + v_handle.index,
                        )
                        kv_coord += 1
                    # last V
                    v_handle, load_kv_producer = self.kv_producer_update_tx_acquire_and_advance(
                        load_kv_producer,
                        load_kv_empty_mbar_ptr,
                        load_kv_full_mbar_ptr,
                        self.tma_copy_v_bytes,
                    )
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, kv_coord - 1],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=load_kv_full_mbar_ptr + v_handle.index,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            no_skip = (False, None, None, None)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                seqlen_k = cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]
                continue_cond = (
                    not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0], curr_block_coord[0], seqlen_q
                    )
                ) or (seqlen_k <= 0)
                if not continue_cond:
                    q0_handle = load_q_consumer.wait_and_advance()
                    tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrKn0 = tSrKn[None, None, None, k_handle.index]
                    tSrKp0 = tSrKp[None, None, None, k_handle.index]
                    mma_s0_producer, s0_corr_producer = self.mma_qk_split(
                        qk_tiled_mma,
                        (tSrQ0, tSrKn0, tSrKp0, tStS0),
                        (mma_s0_producer, s0_corr_producer),
                    )
                    q1_handle = load_q_consumer.wait_and_advance()
                    tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                    mma_s1_producer, s1_corr_producer = self.mma_qk_split(
                        qk_tiled_mma,
                        (tSrQ1, tSrKn0, tSrKp0, tStS1),
                        (mma_s1_producer, s1_corr_producer),
                    )
                    k_handle.release()
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q,
                        seqlen_k,
                        None,
                        window_size_right,
                    )
                    pv_whether_acc = False
                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        k_handle = load_kv_consumer.wait_and_advance()
                        tSrKni = tSrKn[None, None, None, k_handle.index]
                        tSrKpi = tSrKp[None, None, None, k_handle.index]
                        mma_s0_producer, s0_corr_producer = self.mma_qk_split(
                            qk_tiled_mma,
                            (tSrQ0, tSrKni, tSrKpi, tStS0),
                            (mma_s0_producer, s0_corr_producer),
                        )
                        v_handle = load_kv_consumer.wait_and_advance()
                        tOrVi = tOrV[None, None, None, v_handle.index]
                        mma_corr_producer, p0_mma_consumer = self.mma_pv(
                            pv_tiled_mma,
                            pv_whether_acc,
                            (tOrP0, tOrVi, tOtO0),
                            (mma_corr_producer, p0_mma_consumer),
                            no_skip,
                        )
                        mma_s1_producer, s1_corr_producer = self.mma_qk_split(
                            qk_tiled_mma,
                            (tSrQ1, tSrKni, tSrKpi, tStS1),
                            (mma_s1_producer, s1_corr_producer),
                        )
                        k_handle.release()
                        mma_corr_producer, p1_mma_consumer = self.mma_pv(
                            pv_tiled_mma,
                            pv_whether_acc,
                            (tOrP1, tOrVi, tOtO1),
                            (mma_corr_producer, p1_mma_consumer),
                            no_skip,
                        )
                        pv_whether_acc = True
                        v_handle.release()
                    q0_handle.release()
                    q1_handle.release()
                    v_handle = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle.index]
                    mma_corr_producer, p0_mma_consumer = self.mma_pv(
                        pv_tiled_mma,
                        pv_whether_acc,
                        (tOrP0, tOrVi, tOtO0),
                        (mma_corr_producer, p0_mma_consumer),
                        no_skip,
                    )
                    mma_corr_producer, p1_mma_consumer = self.mma_pv(
                        pv_tiled_mma,
                        pv_whether_acc,
                        (tOrP1, tOrVi, tOtO1),
                        (mma_corr_producer, p1_mma_consumer),
                        no_skip,
                    )
                    v_handle.release()
                    vec0_handle = s0_corr_producer.acquire_and_advance()
                    vec0_handle.commit()
                    vec1_handle = s1_corr_producer.acquire_and_advance()
                    vec1_handle.commit()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax0 / Softmax1 (inherited)
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.softmax1_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            softmax_fn(
                stage=0,
                tensor_args=(tStS, tStS0, cum_seqlen_k, cum_seqlen_q, None, None, None),
                pipeline_args=(mma_s0_consumer, s0_corr_producer, p0_mma_producer),
                inplace_args=(s0_p1_inplace_producer, s1_p0_inplace_consumer),
            )
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            softmax_fn(
                stage=1,
                tensor_args=(tStS, tStS1, cum_seqlen_k, cum_seqlen_q, None, None, None),
                pipeline_args=(mma_s1_consumer, s1_corr_producer, p1_mma_producer),
                inplace_args=(s1_p0_inplace_producer, s0_p1_inplace_consumer),
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction + merge epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
            tScS = qk_thr_mma.partition_C(cS)
            tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
            tStS_vec0 = cute.make_tensor(tStS.iterator + self.tmem_vec0_offset, tStS_vec_layout)
            tStS_vec1 = cute.make_tensor(tStS.iterator + self.tmem_vec1_offset, tStS_vec_layout)
            tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
            tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )
            tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_vec0)
            thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
            thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)
            tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
            tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
            tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                row_idx = Int32(0)
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                seqlen_k = cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]
                continue_cond = (
                    not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0], curr_block_coord[0], seqlen_q
                    )
                )
                if not continue_cond:
                    row_idx = curr_block_coord[0] * self.cta_tiler[0] + tTMEM_LOAD_VECcS[0][0]
                    continue_cond = seqlen_k <= 0
                if not continue_cond:
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    vec0_handle.release()
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    vec1_handle.release()
                    oi_peek_status = cutlass.Boolean(False)
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q,
                        seqlen_k,
                        None,
                        window_size_right,
                    )
                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        (
                            (s0_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                        ) = self.correction_rescale(
                            pv_thr_mma,
                            tiled_tmem_load_vec,
                            scale_softmax_log2,
                            (tOtO0, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECcS),
                            (s0_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                        )
                        (
                            (s1_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                        ) = self.correction_rescale(
                            pv_thr_mma,
                            tiled_tmem_load_vec,
                            scale_softmax_log2,
                            (tOtO1, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECcS),
                            (s1_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                        )
                    op = merge_op[batch_coord]
                    value_args = (cuseqlen_q, seqlen_q, curr_block_coord, scale_softmax_log2, op)
                    mO_ = cute.make_tensor(
                        mO.iterator + cuseqlen_q * mO.stride[0],
                        cute.make_layout((seqlen_q, mO.shape[1], mO.shape[2]), stride=mO.stride),
                    )
                    o0_coord = 2 * curr_block_coord[0]
                    o1_coord = o0_coord + 1
                    gO_stg = cute.local_tile(
                        mO_, (self.pv_mma_tiler[0], self.pv_mma_tiler[1]), (None, None, None)
                    )
                    gO0 = gO_stg[None, None, o0_coord, 0, curr_block_coord[2]]
                    gO1 = gO_stg[None, None, o1_coord, 0, curr_block_coord[2]]
                    s0_corr_consumer, mma_corr_consumer = self.correction_epilog_merge(
                        pv_thr_mma,
                        tiled_tmem_load_vec,
                        (tOtO0, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECcS, gO0, mStats),
                        (s0_corr_consumer, mma_corr_consumer),
                        (row_idx, *value_args),
                    )
                    row_idx += self.qk_mma_tiler[0]
                    s1_corr_consumer, mma_corr_consumer = self.correction_epilog_merge(
                        pv_thr_mma,
                        tiled_tmem_load_vec,
                        (tOtO1, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECcS, gO1, mStats),
                        (s1_corr_consumer, mma_corr_consumer),
                        (row_idx, *value_args),
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tmem.relinquish_alloc_permit()
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
        return

    # ------------------------------------------------------------------------------------------
    # MMA: QK^T over the split K operand (4 phases of k_nope, then 2 of k_pe), accumulating into one S tile
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def mma_qk_split(self, tiled_mma, tensor_args, pipeline_args):
        tSrQi, tSrKn, tSrKp, tStSi = tensor_args
        mma_si_producer, si_corr_producer = pipeline_args
        si_handle = mma_si_producer.acquire_and_advance()
        veci_handle = si_corr_producer.acquire_and_advance()
        veci_handle.commit()
        nk_nope = cute.size(tSrKn, mode=[2])
        nk_rope = cute.size(tSrKp, mode=[2])
        for kphase_idx in cutlass.range(nk_nope, unroll_full=True):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma,
                tStSi,
                tSrQi[(None, None, kphase_idx)],
                tSrKn[(None, None, kphase_idx)],
                tStSi,
            )
        for kphase_idx in cutlass.range(nk_rope, unroll_full=True):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(
                tiled_mma,
                tStSi,
                tSrQi[(None, None, nk_nope + kphase_idx)],
                tSrKp[(None, None, kphase_idx)],
                tStSi,
            )
        si_handle.commit()
        return mma_si_producer, si_corr_producer

    # ------------------------------------------------------------------------------------------
    # Epilogue: fold this partition's (O_acc, m, l) into the running state of the output rows
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def correction_epilog_merge(
        self,
        thr_mma: cute.ThrMma,
        tiled_tmem_load_vec: cute.TiledCopy,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        value_args: Tuple,
    ):
        tOtO, tTMEM_LOAD_VECtSi, tTMEM_LOAD_VECcS, dest_O, mStats = tensor_args
        row_idx, cuseqlen_q, seqlen_q, blk_coord, scale_softmax_log2, op = value_args
        si_corr_consumer, mma_corr_consumer = pipeline_args

        pv_tiled_mma_shape = (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOdO = thr_mma.partition_C(dest_O)
        tOcO = thr_mma.partition_C(cO)
        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, corr_tile_size)))
        tOdO_i = cute.logical_divide(tOdO, cute.make_layout((128, corr_tile_size)))

        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADdO = thr_tmem_load.partition_D(tOdO_i[(None, None), None])
        tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        vec_i_handle = si_corr_consumer.wait_and_advance()
        tTMEM_LOAD_VECrS = cute.make_rmem_tensor(tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtSi, tTMEM_LOAD_VECrS)
        cute.arch.fence_view_async_tmem_load()
        vec_i_handle.release()
        oi_handle = mma_corr_consumer.wait_and_advance()

        row_sum = tTMEM_LOAD_VECrS[0]
        row_valid = row_idx < seqlen_q
        tok = row_idx + cuseqlen_q
        head = blk_coord[2][0]
        # this partition's log2-domain max and its softmax denominator (undo the fp8 P pre-scale)
        m_cur = scale_softmax_log2 * tTMEM_LOAD_VECrS[1]
        l_cur = row_sum * Float32(2.0 ** (-self.p_fp8_prescale_log2))
        do_merge = op == 1
        do_write = op != 0
        m_new = m_cur
        l_new = l_cur
        w_prev = Float32(0.0)  # weight of the running (normalised) output
        w_cur = Float32(1.0) / row_sum  # weight of this partition's raw accumulator
        if row_valid and do_merge:
            m_prev = mStats[tok, head, 0]
            l_prev = mStats[tok, head, 1]
            m_new = cute.arch.fmax(m_prev, m_cur)
            a_prev = l_prev * cute.math.exp2(m_prev - m_new, fastmath=True)
            a_cur = l_cur * cute.math.exp2(m_cur - m_new, fastmath=True)
            l_new = a_prev + a_cur
            inv_l = Float32(1.0) / l_new
            w_prev = a_prev * inv_l
            w_cur = a_cur * inv_l / row_sum

        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADdO_i = tTMEM_LOADdO[None, 0, 0, i]
            tTMrO = cute.make_rmem_tensor(tTMEM_LOADoO[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            tDMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
            o_acc = tTMrO.load()
            if row_valid and do_write:
                if do_merge:
                    tPrev = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                    cute.autovec_copy(tTMEM_LOADdO_i, tPrev)
                    o_new = o_acc * w_cur + tPrev.load().to(Float32) * w_prev
                    tDMrO.store(o_new.to(self.o_dtype))
                else:
                    tDMrO.store((o_acc * w_cur).to(self.o_dtype))
                cute.autovec_copy(tDMrO, tTMEM_LOADdO_i)
        if row_valid and do_write:
            mStats[tok, head, 0] = m_new
            mStats[tok, head, 1] = l_new
        oi_handle.release()
        return (si_corr_consumer, mma_corr_consumer)
