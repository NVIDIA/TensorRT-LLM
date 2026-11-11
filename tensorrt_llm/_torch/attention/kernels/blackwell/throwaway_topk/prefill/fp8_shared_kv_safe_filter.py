# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""H64 native-FP16 FP8 filtering with shared KV and constant MMA scales.

Two query pairs share each KV tile. The constant 1/512 QK scaling occurs
in block-scaled MMA; native_fp8_pack balances the weights by 512. Per-key
FP32 scales follow the FP16 head reduction. Only threshold candidates are
written. The public dispatcher requires complete Q and K tiles."""

from .fp4_safe_filter import FP4PrefillMQALogitsKernel as _BaseKernel
from .native_lowp_math import native_fp8_pack
from .fp4_safe_filter import (
    Float8E8M0FNU,
    IndexerPipelineTmaAsync,
    _RND_RN,
    _subview,
    block_samples,
    cpasync,
    cuda,
    cute,
    cutlass,
    filter_boundary,
    finish_recovery,
    finite_integer_relu4,
    indexer_add_packed_f32x2,
    indexer_fma_packed_f32x2,
    indexer_memory,
    indexer_tensor_utils,
    integer_relu4,
    native_add,
    native_fma,
    native_pack_relu,
    native_sum_fp32,
    pdl_prologue,
    pipeline,
    pipeline_init_arrive,
    pipeline_init_wait,
    propose,
    publish_bank,
    runtime_samples,
    sample_tile,
    sm100_utils,
    subset_barrier,
    tcgen05,
    utccp_interleaved_qsf,
    utccp_required_smem_warp_transpose,
)


class FP8PrefillMQALogitsKernel(_BaseKernel):
    def __init__(self, **kwargs):
        if not kwargs.get("native_fp16") or kwargs.get("native_bf16"):
            raise ValueError("shared FP8 MMA requires explicit native_fp16")
        super().__init__(**kwargs)
        if self.num_heads != 64 or self.N != 128 or self.mode != "filter":
            raise ValueError("shared FP8 MMA requires H64 filtering")
        self.block_q = 4
        self.umma_inst_k = 32

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        tiled_mma = super()._setup_mma(a_dtype, b_dtype, a_major, b_major)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, b_dtype, self.num_q_stages * 2
        )
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # KV: [num_kv_tokens, head_dim] raw FP8 bytes
        key_scale: cute.Tensor,
        b: cute.Tensor,  # Q: flat [num_q_tokens * 64, head_dim] FP8 bytes
        weights: cute.Tensor,  # Flat FP32 weights; reshaped as a TMA view below.
        logits: cute.Tensor,  # [num_q_tokens, logits_stride]
        cu_seq_len_k_start: cute.Tensor,  # [num_q_tokens] int32
        cu_seq_len_k_end: cute.Tensor,  # [num_q_tokens] int32
        thresholds: cute.Tensor,
        counts: cute.Tensor,
        candidates: cute.Tensor,
        output: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # Reinterpret contiguous public FP8 tensors as two-query TMA tiles.
        # Dispatch requires complete four-query groups and 128-key tiles.
        b = cute.make_tensor(
            b.iterator,
            cute.make_layout((128, 128, num_q_tokens // 2), stride=(128, 1, 16384)),
        )
        weights = cute.make_tensor(
            weights.iterator,
            cute.make_layout((128, num_q_tokens // 2), stride=(1, 128)),
        )
        half_head_dim = self.head_dim  # FP8 bytes per row

        # Reinterpret each byte as one E4M3 element for MMA and TMA.
        kv_fp4 = cute.recast_tensor(kv, cutlass.Float8E4M3FN)

        # The public Q byte view uses the same E4M3 storage.
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)
        if cutlass.const_expr(self.interleaved_queries):
            # TMA interleaves query pairs directly; no Q copy or extra kernel.
            b = cute.make_tensor(
                b.iterator,
                cute.make_layout(
                    ((2, 32, 2), b.shape[1], b.shape[2]),
                    stride=(
                        (32 * b.stride[0], b.stride[0], 64 * b.stride[0]),
                        b.stride[1],
                        b.stride[2],
                    ),
                ),
            )

        # Contiguous KV tiles: one byte per head-dimension element.
        num_kv_blocks = num_kv_tokens // self.block_kv
        kv_layout = cute.make_layout(
            (self.block_kv, self.head_dim, num_kv_blocks),
            stride=(self.head_dim, 1, self.block_kv * self.head_dim),
        )
        a = cute.make_tensor(kv_fp4.iterator, kv_layout)

        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = indexer_tensor_utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = indexer_tensor_utils.LayoutEnum.ROW_MAJOR.mma_major_mode()

        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # ---- TMA for KV (A): contiguous [block_kv, head_dim, num_kv_blocks] ----
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_smem_layout = sm100_utils.make_smem_layout(
            cute.nvgpu.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        a_smem_per_stage = cute.slice_(self.a_tma_smem_layout, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            a,
            a_smem_per_stage,
            (self.block_kv, self.head_dim),
        )

        # TMA for Q (B) — full K=128, L dim = num_q_blocks
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA for Weights — [N, num_q_blocks], tile [N], L=num_q_blocks
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages * 2),
            stride=(1, self.w_stage_stride),
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            w_smem_per_stage,
            self.w_smem_layout_staged.shape[:1],
        )

        # Load block_kv packed scale words per stage in row-major order.
        # The UMMA warp transposes them to UTCCP chunk layout in SMEM.
        self.sf_kv_smem_layout_staged = cute.make_layout(
            (self.block_kv, self.num_kv_stages)
        )
        # TMA loads N Q-scale entries into an N_padded allocation aligned for UTCCP.
        # UTCCP copies the padded region, but MMA and the epilogue consume only N entries.
        N_padded = ((self.N + 127) // 128) * 128
        self.N_padded = N_padded
        # sf_q stage stride in int32 elem (= 4 bytes/elem). Pad to 128B (= 32 int32).
        sf_q_stage_bytes = N_padded * 4
        sf_q_stage_stride_int32 = ((sf_q_stage_bytes + 127) // 128 * 128) // 4
        self.sf_q_smem_layout_staged = cute.make_layout(
            (N_padded, self.num_q_stages * 2),
            stride=(1, sf_q_stage_stride_int32),
        )
        # Use an N-wide TMA view with the padded allocation's stage stride.
        # This satisfies the TMA layout symmetry check while preserving UTCCP alignment.
        self.sf_q_tma_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages * 2),
            stride=(1, sf_q_stage_stride_int32),
        )
        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.weight_bytes
        # Only the FP8 KV data is transferred; scales are constants in SMEM.
        self.num_kv_sf_tma_bytes = self.block_kv * half_head_dim
        # Q, Q scales and weights share one barrier. The scale transfer is
        # N int32 words; include N * 4 bytes in the barrier transaction count.
        sf_q_tma_bytes = 0
        self.num_q_tma_bytes = 2 * (
            b_copy_size * atom_thr_size + sf_q_tma_bytes + w_copy_size
        )

        num_q_blocks = (num_q_tokens + self.block_q - 1) // self.block_q
        num_ctas = num_q_blocks * self.splits
        if cutlass.const_expr(
            self.mode == "filter" or (self.mode == "sample" and self.persistent_sample)
        ):
            num_ctas = min(num_q_blocks, self.num_sms) * self.splits

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            # SFB WAR-closure mbarrier pair (1 stage: full + unused empty).
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            weights,
            key_scale,
            logits,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            thresholds,
            counts,
            candidates,
            output,
            num_q_tokens,
            num_q_blocks,
            num_kv_tokens,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.a_tma_smem_layout,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.sf_kv_smem_layout_staged,
            self.sf_q_smem_layout_staged,
            self.sf_q_tma_smem_layout_staged,
            self.tmem_sfa_layout,
            self.tmem_sfb_layout,
            self.sfa_chunk_smem_layout,
            self.sfb_chunk_smem_layout,
            self.copy_atom_s2t,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=2 if self.num_groups == 1 else 1,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV: [block_kv, head_dim, num_kv_blocks] (FP4)
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q (L dim = num_q_blocks, FP4)
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA coord tensor [N, num_q_blocks]
        mW_raw: cute.Tensor,
        mKeyScale: cute.Tensor,
        mLogits: cute.Tensor,  # [num_q_tokens, logits_stride]
        mCuSeqLenKStart: cute.Tensor,  # [num_q_tokens] int32
        mCuSeqLenKEnd: cute.Tensor,  # [num_q_tokens] int32
        thresholds: cute.Tensor,
        counts: cute.Tensor,
        candidates: cute.Tensor,
        output: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_q_blocks: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_tma_smem_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        sf_kv_smem_layout_staged: cute.Layout,
        sf_q_smem_layout_staged: cute.Layout,
        sf_q_tma_smem_layout_staged: cute.Layout,
        tmem_sfa_layout: cute.Layout,
        tmem_sfb_layout: cute.Layout,
        sfa_chunk_smem_layout: cute.Layout,
        sfb_chunk_smem_layout: cute.Layout,
        copy_atom_s2t: cute.CopyAtom,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        pdl_prologue(self.pdl)
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[2]
        row_base = (bid // self.splits) * self.block_q
        row = cutlass.Int32(0)
        required = cutlass.Int32(0)
        smem = indexer_memory.SmemAllocator()
        hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(1538), byte_alignment=128
        )
        scalar = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(2), byte_alignment=16
        )
        if cutlass.const_expr(self.mode == "sample" and not self.persistent_sample):
            for offset in cutlass.range_constexpr(self.block_q):
                row = row_base + offset
                if row < num_q_tokens:
                    for j in cutlass.range(tid, self.sample_size, 384):
                        mLogits[row, j] = self.output_dtype(float("-inf"))
                    if tid == 0:
                        counts[row] = 0
            cute.arch.barrier(barrier_id=14, number_of_threads=self.threads_per_cta)
        work = cutlass.Boolean(True)
        if cutlass.const_expr(self.mode == "recover"):
            work = cutlass.Boolean(False)
            for offset in cutlass.range_constexpr(self.block_q):
                row = row_base + offset
                if row < num_q_tokens:
                    required = min(
                        self.top_k, mCuSeqLenKEnd[row] - mCuSeqLenKStart[row]
                    )
                    work |= counts[row] < required
        if work:
            self._gemm_body(
                tiled_mma,
                tma_atom_a,
                mA_mkl,
                tma_atom_b,
                mB_nkl,
                tma_atom_w,
                mW_tma,
                mW_raw,
                mKeyScale,
                mLogits,
                mCuSeqLenKStart,
                mCuSeqLenKEnd,
                thresholds,
                counts,
                candidates,
                output,
                num_q_tokens,
                num_q_blocks,
                num_kv_tokens,
                cluster_layout_vmnk,
                a_smem_layout_staged,
                a_tma_smem_layout,
                b_smem_layout_staged,
                w_smem_layout_staged,
                sf_kv_smem_layout_staged,
                sf_q_smem_layout_staged,
                sf_q_tma_smem_layout_staged,
                tmem_sfa_layout,
                tmem_sfb_layout,
                sfa_chunk_smem_layout,
                sfb_chunk_smem_layout,
                copy_atom_s2t,
                epi_tile,
                SharedStorage,
            )
        if cutlass.const_expr(self.mode in ("sample", "recover")):
            cute.arch.barrier(barrier_id=14, number_of_threads=self.threads_per_cta)
            if cutlass.const_expr(self.mode != "sample"):
                cute.arch.fence_acq_rel_gpu()
                cute.arch.barrier(barrier_id=14, number_of_threads=self.threads_per_cta)
            if tid < 256:
                for offset in cutlass.range_constexpr(self.block_q):
                    row = row_base + offset
                    if row < num_q_tokens:
                        start = mCuSeqLenKStart[row]
                        end = mCuSeqLenKEnd[row]
                        required = min(self.top_k, end - start)
                        if cutlass.const_expr(
                            self.mode == "sample" and not self.persistent_sample
                        ):
                            active_samples = block_samples(
                                row // self.block_q,
                                mCuSeqLenKEnd,
                                num_q_tokens,
                                self.block_q,
                                self.top_k,
                            )
                            multiple = (
                                2
                                if active_samples == 512
                                else (4 if active_samples <= 2048 else 6)
                            )
                            rank = min(
                                active_samples,
                                max(
                                    1,
                                    (
                                        multiple * self.top_k * active_samples
                                        + max(1, end - start)
                                        - 1
                                    )
                                    // max(1, end - start),
                                ),
                            )
                            propose(
                                row,
                                tid,
                                mLogits,
                                hist,
                                scalar,
                                thresholds,
                                active_samples,
                                rank,
                            )
                        elif cutlass.const_expr(self.mode != "sample"):
                            if counts[row] < required:
                                finish_recovery(
                                    row,
                                    tid,
                                    candidates,
                                    counts,
                                    mCuSeqLenKStart,
                                    mCuSeqLenKEnd,
                                    mLogits,
                                    hist,
                                    scalar,
                                    output,
                                    thresholds,
                                    self.top_k,
                                    self.capacity,
                                    self.output_dtype,
                                    self.packed,
                                )

    @cute.jit
    def _gemm_body(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,
        mW_raw: cute.Tensor,
        mKeyScale: cute.Tensor,
        mLogits: cute.Tensor,
        mCuSeqLenKStart: cute.Tensor,
        mCuSeqLenKEnd: cute.Tensor,
        thresholds: cute.Tensor,
        counts: cute.Tensor,
        candidates: cute.Tensor,
        output: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_q_blocks: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_tma_smem_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        sf_kv_smem_layout_staged: cute.Layout,
        sf_q_smem_layout_staged: cute.Layout,
        sf_q_tma_smem_layout_staged: cute.Layout,
        tmem_sfa_layout: cute.Layout,
        tmem_sfb_layout: cute.Layout,
        sfa_chunk_smem_layout: cute.Layout,
        sfb_chunk_smem_layout: cute.Layout,
        copy_atom_s2t: cute.CopyAtom,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        smem = indexer_memory.SmemAllocator()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # Warp roles
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < self.num_math_warps
        is_tma_warp_0 = warp_idx == self.num_math_warps + 0
        is_tma_warp_1 = warp_idx == self.num_math_warps + 1
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == self.num_math_warps + 2
        is_umma_warp_1 = warp_idx == self.num_math_warps + 3

        # Grid-stride scheduler: this SM handles q_block_idx = sm_idx,
        # sm_idx+num_sms, ...
        NUM_MATH_WG = 1  # kNumMathWarpGroups
        block_kv_val = self.block_kv
        block_q_val = self.block_q
        sm_idx = bidz // self.splits

        # Prefetch the first Q block's KV length during prologue setup.
        # Clamp zero-work CTA indices; those CTAs never consume the loaded value.
        start_q_block = sm_idx
        start_q_block_clamped = min(start_q_block, num_q_blocks - 1)
        q_last_token_clamped = min(
            start_q_block_clamped * block_q_val + block_q_val - 1,
            num_q_tokens - 1,
        )
        current_num_kv = (
            mCuSeqLenKEnd[q_last_token_clamped] + block_kv_val - 1
        ) // block_kv_val

        # Start at the minimum visible K position, aligned down to SPLIT_KV to
        # preserve the two warp groups' tile ownership. Bind captured values as
        # default arguments because staged control flow cannot capture closures.
        def _kv_lo_for_block(
            qb,
            _ks=mCuSeqLenKStart,
            _nqt=num_q_tokens,
            _bq=block_q_val,
            _split=block_kv_val * NUM_MATH_WG,
            _nwg=NUM_MATH_WG,
            _part=bidz % self.splits,
            _sample=self.mode == "sample",
        ):
            if cutlass.const_expr(_sample):
                return cutlass.Int32(0)
            ks_min = _ks[min(qb * _bq, _nqt - 1)]
            for t_i in cutlass.range_constexpr(1, _bq):
                ks_min = min(ks_min, _ks[min(qb * _bq + t_i, _nqt - 1)])
            return (ks_min // _split) * _nwg + _part * _nwg

        current_kv_lo = _kv_lo_for_block(start_q_block_clamped)
        if cutlass.const_expr(self.mode == "sample"):
            current_num_kv = (
                runtime_samples(mCuSeqLenKEnd[q_last_token_clamped], self.top_k) // 128
            )
            current_kv_lo = cutlass.Int32(0)

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)

        storage = smem.allocate(SharedStorage)

        num_heads = self.num_heads
        num_epi_subtiles = self.num_epi_subtiles

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q pipeline: TMA producer → Math consumer (8 math warps)
        # PipelineTmaAsync: consumer_release uses is_signalling_thread
        # (lane 0 per warp). 8 math warps × 1 lane-0 = 8 arrives.
        q_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_math_warps
        )
        q_pipeline = IndexerPipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        # Both Math WGs share the same pipeline state (advance in lockstep)
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        # UMMA warps observe Q pipeline (wait only, no release)
        # to ensure Q is in SMEM before GEMM. Critical for UMMA warp 1
        # since TMA warp 1 only loads KV1 (not Q).
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        # UMMA waits on and releases the shared KV/scale pipeline.
        # Math consumes accumulators and does not release this pipeline.
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
        kv_pipeline_0 = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_sf_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        kv_pipeline_1 = kv_pipeline_0

        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        # UMMA consumer states (wait + release, owns the pipeline).
        # Math reads block-scaled accumulators and does not consume the KV+SF pipeline.
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )

        # UMMA pipelines (per-group)
        math_threads_per_group = self.num_math_threads // self.num_groups
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        umma_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )

        # Math and UMMA warps share the 320-thread TMEM allocation barrier.
        # Math warp 0 allocates; TMA warps load independently.
        tmem_alloc_num_threads = (
            self.num_math_threads + 32 * self.num_groups
        )  # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=tmem_alloc_num_threads
        )
        # Warp 0 copies Q scales into shared SFB TMEM; warp 1 reads that region.
        # Synchronize both UMMA warps after the copy: the Q pipeline orders TMA-to-SMEM,
        # not cross-warp TMEM visibility.
        sfb_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * self.num_groups
        )
        tmem = indexer_memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,  # math warp 0 does alloc+free (last TMEM consumer)
            is_two_cta=False,
            # Arch-aware: sm_107 exposes 576 cols and needs exclusive alloc for
            # >512 (next_n=4 -> 544). Defaults to sm_100 (512) otherwise.
            arch=self.arch_str,
        )

        # Persistent K1 keeps per-query scratch in the math groups. TMA/UMMA
        # continue independently; barrier 12 includes only the 256 math lanes.
        if cutlass.const_expr(self.mode == "sample" and self.persistent_sample):
            sample_hist = smem.allocate_tensor(
                cutlass.Int32, cute.make_layout(1538), byte_alignment=128
            )
            sample_scalar = smem.allocate_tensor(
                cutlass.Int32, cute.make_layout(2), byte_alignment=16
            )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = sKV_0
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # SF_Q SMEM (int32, packed UE8M0; flat layout, transposed
        # in-place by UMMA warp before UTCCP).
        sSF_Q = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_q_smem_layout_staged,
            byte_alignment=128,
        )
        # Weights SMEM: [N, num_q_stages], shared Q barrier
        sW = smem.allocate_tensor(
            element_type=self.weight_dtype,
            layout=w_smem_layout_staged,
            byte_alignment=128,
        )
        # SF_KV SMEM (int32, packed UE8M0; flat layout, transposed
        # in-place by UMMA warp before UTCCP).
        sSF_KV_0 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_kv_smem_layout_staged,
            byte_alignment=128,
        )
        sSF_KV_1 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_kv_smem_layout_staged,
            byte_alignment=128,
        )

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): tile [block_kv, head_dim], coord = (0, 0, kv_block_idx)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        gA = cute.local_tile(
            mA_mkl,
            (self.block_kv, self.head_dim),
            coord=(None, None, None),
        )
        # sKV SMEM view for TMA: group (block_kv, head_dim) into one mode to match
        # GMEM's ((block_kv,head_dim), ...) shape from group_modes(gA, 0, 2).
        # Use .outer (plain layout); swizzle is captured by sKV_0's iterator.
        sKV_0_for_tma = cute.make_tensor(
            sKV_0.iterator, cute.group_modes(a_tma_smem_layout.outer, 0, 2)
        )
        sKV_1_for_tma = cute.make_tensor(
            sKV_1.iterator, cute.group_modes(a_tma_smem_layout.outer, 0, 2)
        )
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_0_for_tma,
            cute.group_modes(gA, 0, 2),
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_1_for_tma,
            cute.group_modes(gA, 0, 2),
        )

        # Partition Q (B): shared SMEM, L dim = num_q_blocks
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, None)]  # [tma, K, L]

        # Partition Weights: standalone TMA, [N, num_q_blocks] → [N] per stage
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            w_cta_layout,
            cute.group_modes(sW, 0, 1),
            cute.group_modes(mW_tma, 0, 1),
        )

        # MMA fragments
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)  # shared
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # Staged acc (fp16_gemm_3 pattern): append UMMA stage dim
        # shape: (*acc_shape, STAGE) — dynamic index on last dim reduces rank
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # TMEM layout info (allocation deferred to UMMA/Math warp branches)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

        # Epilogue setup
        c_layout = indexer_tensor_utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (
            epi_tile[0],
            (
                (16 if self.branchless_publish else 32)
                if self.num_heads == 4
                else num_heads // num_epi_subtiles
            ),
        )
        # Both paired-query fragments already remain live until reduction.
        # One 64-value load replaces two 32-value loads without extra registers.
        if cutlass.const_expr(self.interleaved_queries):
            epi_sub_mn = (epi_tile[0], 64)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # Grid-stride Q scheduling: math warps nest Q/KV loops;
        # TMA and UMMA warps flatten them.
        q_block_idx = cutlass.Int32(sm_idx)
        kv_idx = cutlass.Int32(current_kv_lo)
        num_kv = current_num_kv
        # Sentinel: no previous Q block (forces initial Q load)
        prev_q_block_idx = num_q_blocks  # out-of-range sentinel

        next_q_block_idx = q_block_idx
        next_kv_idx = kv_idx
        next_num_kv = num_kv
        has_work = q_block_idx < num_q_blocks

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            # TMA warp 0: loads Q+SF_Q+Weights (on Q-block change) + KV group 0
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            # Prefetch first Q block before the loop
            q_pipeline.producer_acquire(q_prod_state)
            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
            for q_pair in cutlass.range_constexpr(2):
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, next_q_block_idx * 2 + q_pair)],
                    tBsB[(None, q_prod_state.index + q_pair * self.num_q_stages)],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                # SF_Q TMA load — under same q_bar as Q + W.
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, next_q_block_idx * 2 + q_pair)],
                    tWsW[(None, q_prod_state.index + q_pair * self.num_q_stages)],
                    tma_bar_ptr=q_bar,
                )
            q_prod_state.advance()

            while has_work:
                q_block_idx_old = prev_q_block_idx
                prev_q_block_idx = next_q_block_idx
                q_block_idx = next_q_block_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # On Q-block change: prefetch Q for the NEXT Q block
                if q_block_idx != q_block_idx_old:
                    prefetch_next_q = (
                        q_block_idx + cute.arch.grid_dim()[2] // self.splits
                    )
                    if cutlass.const_expr(self.num_q_stages == 1):
                        prefetch_next_q = q_block_idx
                    if prefetch_next_q < num_q_blocks and (
                        cutlass.const_expr(self.num_q_stages > 1)
                        or q_block_idx_old < num_q_blocks
                    ):
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        for q_pair in cutlass.range_constexpr(2):
                            cute.copy(
                                tma_atom_b,
                                tBgB[(None, 0, prefetch_next_q * 2 + q_pair)],
                                tBsB[
                                    (
                                        None,
                                        q_prod_state.index + q_pair * self.num_q_stages,
                                    )
                                ],
                                tma_bar_ptr=q_bar,
                                mcast_mask=b_mcast_mask,
                            )
                            # SF_Q TMA load — under same q_bar.
                            cute.copy(
                                tma_atom_w,
                                tWgW[(None, prefetch_next_q * 2 + q_pair)],
                                tWsW[
                                    (
                                        None,
                                        q_prod_state.index + q_pair * self.num_q_stages,
                                    )
                                ],
                                tma_bar_ptr=q_bar,
                            )
                        q_prod_state.advance()

                # Load KV + SF for group 0: block at kv_idx (contiguous,
                # no block_table)
                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                stage = kv_prod_state_0.index
                cute.copy(
                    tma_atom_a,
                    tAgA_0[
                        (
                            None,
                            0,
                            0,
                            (
                                sample_tile(
                                    kv_idx,
                                    q_block_idx,
                                    mCuSeqLenKEnd,
                                    num_q_tokens,
                                    self.block_q,
                                    self.sample_size,
                                    self.top_k,
                                )
                                if cutlass.const_expr(self.mode == "sample")
                                else kv_idx
                            ),
                        )
                    ],
                    tAsA_0[(None, stage)],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                kv_prod_state_0.advance()

                # Scheduler advance
                next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                if next_kv_idx >= num_kv:
                    next_q_block_idx = (
                        q_block_idx + cute.arch.grid_dim()[2] // self.splits
                    )  # grid-stride
                    next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                    if next_q_block_idx < num_q_blocks:
                        q_last_next = min(
                            next_q_block_idx * block_q_val + block_q_val - 1,
                            num_q_tokens - 1,
                        )
                        next_num_kv = (
                            runtime_samples(mCuSeqLenKEnd[q_last_next], self.top_k)
                            // 128
                            if cutlass.const_expr(self.mode == "sample")
                            else (
                                (mCuSeqLenKEnd[q_last_next] + block_kv_val - 1)
                                // block_kv_val
                            )
                        )
                has_work = next_q_block_idx < num_q_blocks

        elif is_umma_warp_0:
            # Wait for Q independently of KV: separate TMA barriers do not order
            # one another's SMEM writes.
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            # TMEM: wait for math warp 0's allocation, retrieve pointer
            subset_barrier(
                barrier_id=1,
                number_of_threads=self.num_math_threads + 32 * self.num_groups,
            )
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            # SF TMEM tensors. Per-WG SFA region (group 0 owns
            # cols [sf_base..sf_base+num_sfa_tmem_cols)). SFB is shared (Q SF)
            # and starts after BOTH groups' SFA regions.
            sf_base_offset = (
                self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            )
            sfa_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset, dtype=self.sf_dtype
            )
            sfb_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols * self.num_groups,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tmem_sfa_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tmem_sfb_layout)

            if is_leader_cta:
                num_k_blocks = cute.size(tCrA_0.shape[2])
                q_stage_0 = cutlass.Int32(0)

                while has_work:
                    q_block_idx_old = prev_q_block_idx
                    prev_q_block_idx = next_q_block_idx
                    q_block_idx = next_q_block_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.mode == "sample"):
                        num_kv = (
                            block_samples(
                                q_block_idx,
                                mCuSeqLenKEnd,
                                num_q_tokens,
                                block_q_val,
                                self.top_k,
                            )
                            // 128
                        )

                    # Wait for Q pipeline when the Q block changes
                    if q_block_idx != q_block_idx_old:
                        if q_block_idx_old < num_q_blocks:
                            # Commit MMAs before replacing Q scales. wait::st covers only the
                            # s2t store, so MMA completion also needs the mbarrier.
                            q_cons_state_umma_0.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_0)
                        q_stage_0 = q_cons_state_umma_0.index

                        if q_block_idx == sm_idx:
                            for sf_i in cutlass.range_constexpr(4):
                                sSF_Q[cute.arch.lane_idx() + sf_i * 32, 0] = (
                                    cutlass.Int32(0x76767676)
                                )
                            cute.arch.sync_warp()
                            cute.arch.fence_view_async_shared()
                            # One-time constant Q-scale publication.
                            # SMEM Q is N_padded tokens; loop over UTCCP atoms (128
                            # tokens each).
                            sf_q_atoms = self.N_padded // 128
                            for atom_idx in cutlass.range_constexpr(sf_q_atoms):
                                atom_offset = atom_idx * 128
                                stage_offset = q_stage_0 * sSF_Q.layout.stride[1]
                                if cutlass.const_expr(self.interleaved_queries):
                                    utccp_interleaved_qsf(
                                        sSF_Q.iterator + stage_offset + atom_offset
                                    )
                                else:
                                    utccp_required_smem_warp_transpose(
                                        sSF_Q.iterator + stage_offset + atom_offset
                                    )
                            cute.arch.fence_view_async_shared()
                            # Both UMMA commits must complete before SFB is overwritten.
                            # Recast packed int32 scales to UE8M0 for the UTCCP chunk layout.
                            sSF_Q_ue8m0 = cute.recast_tensor(sSF_Q, Float8E8M0FNU)
                            stage_off_ue8m0 = q_stage_0 * sSF_Q_ue8m0.layout.stride[1]
                            sSF_Q_chunk = cute.make_tensor(
                                sSF_Q_ue8m0.iterator + stage_off_ue8m0,
                                sfb_chunk_smem_layout,
                            )
                            tCsSFB_compact = cute.filter_zeros(sSF_Q_chunk)
                            tCtSFB_compact = cute.filter_zeros(tCtSFB)
                            tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(
                                copy_atom_s2t, tCtSFB_compact
                            )
                            thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                            tCsSFB_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
                            tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfb, tCsSFB_s2t_
                            )
                            tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)
                            cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t)
                            # Make SFB TMEM write visible to umma_warp_1 before its
                            # MMA reads the same SFB region. fence orders the async
                            # s2t; barrier crosses the warp boundary.
                            cute.arch.fence_view_async_tmem_store()
                            subset_barrier(
                                barrier_id=2, number_of_threads=32 * self.num_groups
                            )

                    # Wait for KV, then TMEM capacity. Padded iterations compute unused results.
                    kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                    umma_pipeline_0.producer_acquire(umma_prod_state_0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage = kv_cons_state_umma_0.index

                    if (q_block_idx == sm_idx) & (kv_idx == current_kv_lo):
                        for sf_i in cutlass.range_constexpr(4):
                            sSF_KV_0[cute.arch.lane_idx() + sf_i * 32, 0] = (
                                cutlass.Int32(0x7F7F7F7F)
                            )
                        cute.arch.sync_warp()
                        cute.arch.fence_view_async_shared()
                        # Transpose KV scales into UTCCP layout unless the host supplied that layout.
                        if cutlass.const_expr(not self.remove_online_sf_transpose):
                            sf_kv_atoms = self.block_kv // 128
                            for atom_idx in cutlass.range_constexpr(sf_kv_atoms):
                                atom_offset = atom_idx * 128
                                stage_offset = kv_stage * sSF_KV_0.layout.stride[1]
                                utccp_required_smem_warp_transpose(
                                    sSF_KV_0.iterator + stage_offset + atom_offset
                                )
                            cute.arch.fence_view_async_shared()
                        # int32 SMEM → UE8M0 view for UTCCP atom + chunk layout.
                        sSF_KV_0_ue8m0 = cute.recast_tensor(sSF_KV_0, Float8E8M0FNU)
                        stage_off_kv0_ue8m0 = kv_stage * sSF_KV_0_ue8m0.layout.stride[1]
                        sSF_KV_0_chunk = cute.make_tensor(
                            sSF_KV_0_ue8m0.iterator + stage_off_kv0_ue8m0,
                            sfa_chunk_smem_layout,
                        )
                        tCsSFA_compact = cute.filter_zeros(sSF_KV_0_chunk)
                        tCtSFA_compact = cute.filter_zeros(tCtSFA)
                        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(
                            copy_atom_s2t, tCtSFA_compact
                        )
                        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
                        tCsSFA_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
                        tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(
                            tiled_copy_s2t_sfa, tCsSFA_s2t_
                        )
                        tCtSFA_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t)

                    tCtAcc_0 = tCtAcc_base_0[
                        (None, None, None, umma_prod_state_0.index)
                    ]
                    # Slice the scale-factor TMEM K mode using the layout-derived sf_k_step.
                    # The last mode is the size-one stage dimension.
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        tCtSFA_k = cute.slice_(
                            tCtSFA, (None, None, k_block * self.sf_k_step, None)
                        )
                        tCtSFB_k = cute.slice_(
                            tCtSFB, (None, None, k_block * self.sf_k_step, None)
                        )
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_0,
                            tCrA_0[None, None, k_block, kv_stage],
                            tCrB[None, None, k_block, q_stage_0],
                            tCtAcc_0,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # UMMA releases the KV+SF stage after consuming its scales.
                    kv_pipeline_0.consumer_release(kv_cons_state_umma_0)
                    kv_cons_state_umma_0.advance()

                    umma_pipeline_0.producer_commit(umma_prod_state_0)
                    umma_prod_state_0.advance()

                    # Synchronize the UMMA warps on every tile so warp 0 cannot reach the next
                    # Q-scale overwrite before warp 1 has committed its MMA.

                    # Scheduler advance (grid-stride)
                    next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = (
                            q_block_idx + cute.arch.grid_dim()[2] // self.splits
                        )
                        next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                        if next_q_block_idx < num_q_blocks:
                            q_last_next = min(
                                next_q_block_idx * block_q_val + block_q_val - 1,
                                num_q_tokens - 1,
                            )
                            next_num_kv = (
                                runtime_samples(mCuSeqLenKEnd[q_last_next], self.top_k)
                                // 128
                                if cutlass.const_expr(self.mode == "sample")
                                else (
                                    (mCuSeqLenKEnd[q_last_next] + block_kv_val - 1)
                                    // block_kv_val
                                )
                            )
                    has_work = next_q_block_idx < num_q_blocks

        elif is_umma_warp_1 and cutlass.const_expr(self.num_groups == 2):
            # UMMA group 1 waits for Q loaded by TMA warp 0.
            cute.arch.warpgroup_reg_dealloc(self.store_registers)

            # TMEM: wait for umma_warp_0's allocation, retrieve pointer
            subset_barrier(
                barrier_id=1,
                number_of_threads=self.num_math_threads + 32 * self.num_groups,
            )
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            # Group 1 owns its SFA columns. SFB holds shared Q scales;
            # only warp 0 issues that UTCCP copy.
            sf_base_offset = (
                self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            )
            sfa_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            sfb_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols * self.num_groups,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tmem_sfa_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tmem_sfb_layout)

            if is_leader_cta:
                num_k_blocks_1 = cute.size(tCrA_1.shape[2])
                q_stage_1 = cutlass.Int32(0)

                while has_work:
                    q_block_idx_old = prev_q_block_idx
                    prev_q_block_idx = next_q_block_idx
                    q_block_idx = next_q_block_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.mode == "sample"):
                        num_kv = (
                            block_samples(
                                q_block_idx,
                                mCuSeqLenKEnd,
                                num_q_tokens,
                                block_q_val,
                                self.top_k,
                            )
                            // 128
                        )

                    # Wait for Q pipeline when the Q block changes
                    if q_block_idx != q_block_idx_old:
                        if q_block_idx_old < num_q_blocks:
                            q_cons_state_umma_1.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_1)
                        q_stage_1 = q_cons_state_umma_1.index + self.num_q_stages
                        # Warp 0 owns the Q-scale copy. Warp 1 waits for cross-warp TMEM visibility
                        # before reading SFB; the Q pipeline covers only TMA-to-SMEM ordering.
                        if q_block_idx == sm_idx:
                            subset_barrier(
                                barrier_id=2, number_of_threads=32 * self.num_groups
                            )

                    # Process KV block for group 1 (kv_idx + 1)
                    # Unconditional UMMA
                    # Wait KV first, then TMEM empty
                    kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                    umma_pipeline_1.producer_acquire(umma_prod_state_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage_1 = kv_cons_state_umma_1.index

                    if (q_block_idx == sm_idx) & (kv_idx == current_kv_lo):
                        for sf_i in cutlass.range_constexpr(4):
                            sSF_KV_1[cute.arch.lane_idx() + sf_i * 32, 0] = (
                                cutlass.Int32(0x7F7F7F7F)
                            )
                        cute.arch.sync_warp()
                        cute.arch.fence_view_async_shared()
                        # Host-prearranged scales already have UTCCP chunk layout;
                        # otherwise transpose group 1 scales in SMEM before the copy.
                        if cutlass.const_expr(not self.remove_online_sf_transpose):
                            sf_kv_atoms_1 = self.block_kv // 128
                            for atom_idx in cutlass.range_constexpr(sf_kv_atoms_1):
                                atom_offset = atom_idx * 128
                                stage_offset = kv_stage_1 * sSF_KV_1.layout.stride[1]
                                utccp_required_smem_warp_transpose(
                                    sSF_KV_1.iterator + stage_offset + atom_offset
                                )
                            cute.arch.fence_view_async_shared()
                        sSF_KV_1_ue8m0 = cute.recast_tensor(sSF_KV_1, Float8E8M0FNU)
                        stage_off_kv1_ue8m0 = (
                            kv_stage_1 * sSF_KV_1_ue8m0.layout.stride[1]
                        )
                        sSF_KV_1_chunk = cute.make_tensor(
                            sSF_KV_1_ue8m0.iterator + stage_off_kv1_ue8m0,
                            sfa_chunk_smem_layout,
                        )
                        tCsSFA_compact_1 = cute.filter_zeros(sSF_KV_1_chunk)
                        tCtSFA_compact_1 = cute.filter_zeros(tCtSFA)
                        tiled_copy_s2t_sfa_1 = tcgen05.make_s2t_copy(
                            copy_atom_s2t, tCtSFA_compact_1
                        )
                        thr_copy_s2t_sfa_1 = tiled_copy_s2t_sfa_1.get_slice(0)
                        tCsSFA_s2t_1_ = thr_copy_s2t_sfa_1.partition_S(tCsSFA_compact_1)
                        tCsSFA_s2t_1 = tcgen05.get_s2t_smem_desc_tensor(
                            tiled_copy_s2t_sfa_1, tCsSFA_s2t_1_
                        )
                        tCtSFA_s2t_1 = thr_copy_s2t_sfa_1.partition_D(tCtSFA_compact_1)
                        cute.copy(tiled_copy_s2t_sfa_1, tCsSFA_s2t_1, tCtSFA_s2t_1)

                    tCtAcc_1 = tCtAcc_base_1[
                        (None, None, None, umma_prod_state_1.index)
                    ]
                    # Rank-4 slice mode 2 at k_block*sf_k_step per K-instr.
                    # See umma_warp_0 for sf_k_step rationale.
                    for k_block in cutlass.range_constexpr(num_k_blocks_1):
                        tCtSFA_k = cute.slice_(
                            tCtSFA, (None, None, k_block * self.sf_k_step, None)
                        )
                        tCtSFB_k = cute.slice_(
                            tCtSFB, (None, None, k_block * self.sf_k_step, None)
                        )
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_1,
                            tCrA_1[None, None, k_block, kv_stage_1],
                            tCrB[None, None, k_block, q_stage_1],
                            tCtAcc_1,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # UMMA owns the KV+SF release.
                    kv_pipeline_1.consumer_release(kv_cons_state_umma_1)
                    kv_cons_state_umma_1.advance()

                    umma_pipeline_1.producer_commit(umma_prod_state_1)
                    umma_prod_state_1.advance()

                    # Synchronize both UMMA warps before warp 0 overwrites shared SFB.

                    # Scheduler advance (grid-stride)
                    next_kv_idx = kv_idx + NUM_MATH_WG * self.splits
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = (
                            q_block_idx + cute.arch.grid_dim()[2] // self.splits
                        )
                        next_kv_idx = _kv_lo_for_block(next_q_block_idx)
                        if next_q_block_idx < num_q_blocks:
                            q_last_next = min(
                                next_q_block_idx * block_q_val + block_q_val - 1,
                                num_q_tokens - 1,
                            )
                            next_num_kv = (
                                runtime_samples(mCuSeqLenKEnd[q_last_next], self.top_k)
                                // 128
                                if cutlass.const_expr(self.mode == "sample")
                                else (
                                    (mCuSeqLenKEnd[q_last_next] + block_kv_val - 1)
                                    // block_kv_val
                                )
                            )
                    has_work = next_q_block_idx < num_q_blocks

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(
                self.math_registers if self.mode == "filter" else 240
            )

            # TMEM: math warp 0 is the allocator; all math warps wait + retrieve
            tmem.allocate(num_tmem_alloc_cols_total)
            if cutlass.const_expr(self.num_heads == 32):
                tmem.relinquish_alloc_permit()
            subset_barrier(
                barrier_id=1,
                number_of_threads=self.num_math_threads + 32 * self.num_groups,
            )
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
            )

            math_group = warpgroup_idx // (
                self.num_math_threads // 128 // self.num_groups
            )
            math_barriers = storage.umma_mbar_0.data_ptr()
            if math_group == 1:
                math_barriers = storage.umma_mbar_1.data_ptr()
            tCtAcc_base_0 = cute.make_tensor(
                tmem_ptr + math_group * cols_per_group, tCtAcc_fake_staged.layout
            )
            local_tidx = tidx % 128
            query_slot = warpgroup_idx % (
                self.num_math_threads // 128 // self.num_groups
            )
            cC = cute.make_identity_tensor(epi_sub_mn)
            tAcc_ref = tCtAcc_base_0[(None, None, None, 0)][((None, None), 0, 0)]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref, epi_sub_mn)
            tiled_copy_ref = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_ref_epi[(None, None, 0, 0)]
            )
            thr_copy_ref = tiled_copy_ref.get_slice(local_tidx)
            tTR_cC = thr_copy_ref.partition_D(cC)
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
            w_cache = cute.make_rmem_tensor((self.max_w_in_reg,), cutlass.Float32)
            zero4 = cute.full((4,), 0.0, dtype=cutlass.Float32)
            # Producers issue one tile pair even for an empty query range.
            # Retire it before advancing the persistent consumer pipeline.
            while has_work:
                q_block_idx = next_q_block_idx
                num_kv = max(next_num_kv, _kv_lo_for_block(q_block_idx) + 1)
                q_pipeline.consumer_wait(q_cons_state)
                q_stage_local = q_cons_state.index + math_group * self.num_q_stages
                outrow = q_block_idx * 4 + math_group * 2 + query_slot
                token_idx = min(outrow, num_q_tokens - 1)
                kv_start = mCuSeqLenKStart[token_idx]
                kv_end = mCuSeqLenKEnd[token_idx]
                tau = (
                    filter_boundary(thresholds[token_idx], self.output_dtype)
                    if cutlass.const_expr(self.dsv4_cast)
                    else thresholds[token_idx]
                )
                # A NaN threshold conservatively accepts the whole interval.
                # K3/K4 still select exact top-k, including overflow recovery.
                if cutlass.const_expr(self.dsv4_unordered):
                    if tau != tau:
                        tau = cutlass.Float32(float("-inf"))
                if cutlass.const_expr(self.native_lowp):
                    native_weights = cute.make_rmem_tensor((32,), cutlass.Uint32)
                    for h in cutlass.range_constexpr(32):
                        wa = sW[(query_slot * 64 + h * 2, q_stage_local)]
                        wb = sW[(query_slot * 64 + h * 2 + 1, q_stage_local)]
                        if cutlass.const_expr(
                            self.use_relu_trick and self.weights_prepared
                        ):
                            wa = wa * 2.0
                            wb = wb * 2.0
                        native_weights[h] = native_fp8_pack(self.output_dtype, wa, wb)
                else:
                    for h in cutlass.range_constexpr(self.max_w_in_reg):
                        w = sW[(query_slot * 64 + h, q_stage_local)]
                        if cutlass.const_expr(
                            self.use_relu_trick and not self.weights_prepared
                        ):
                            w = w * 0.5
                        if cutlass.const_expr(
                            self.use_relu_trick
                            and ((self.dsv4_integer_head_mask >> h) & 1)
                        ):
                            w = w * 2.0
                        w_cache[h] = w
                bank_span = (candidates.shape[1] - 8) // 4
                bank_id = local_tidx // 32
                bank_rows = cute.make_tensor(
                    candidates.iterator
                    + cutlass.Int64(outrow) * candidates.stride[0]
                    + 8
                    + bank_id * bank_span,
                    cute.make_layout((1, bank_span), stride=(bank_span, 1)),
                )
                cursor = cutlass.Int32(0)
                kv_idx = _kv_lo_for_block(q_block_idx)
                # Every producer and both math WGs traverse the same KV tiles.
                if cutlass.const_expr(self.dsv4_full_tiles):
                    while (kv_idx < num_kv) & (kv_idx * 128 < kv_start):
                        kv_pos = kv_idx * 128 + m_coord
                        scale_val = mKeyScale[min(kv_pos, num_kv_tokens - 1)]
                        cute.arch.mbarrier_wait(
                            math_barriers + umma_cons_state_0.index,
                            umma_cons_state_0.phase,
                        )
                        tAcc = tCtAcc_base_0[
                            (None, None, None, umma_cons_state_0.index)
                        ][((None, None), 0, 0)]
                        tTR = thr_copy_ref.partition_S(
                            cute.flat_divide(tAcc, epi_sub_mn)
                        )
                        if cutlass.const_expr(self.native_lowp):
                            s_acc = cute.make_rmem_tensor((2,), cutlass.Uint32)
                            s_acc.fill(0)
                        else:
                            s_acc = cute.full((4,), 0.0, dtype=cutlass.Float32)
                        for i in cutlass.range_constexpr(num_epi_subtiles):
                            cute.copy(
                                tiled_copy_ref,
                                tTR[
                                    (
                                        None,
                                        None,
                                        None,
                                        0,
                                        query_slot * num_epi_subtiles + i,
                                    )
                                ],
                                tTR_rAcc,
                            )
                            cute.arch.fence_view_async_tmem_load()
                            if cutlass.const_expr(i == num_epi_subtiles - 1):
                                # 256 consumers release only after both query slices
                                # are in registers. Reuse also protects staged SFA.
                                cute.arch.mbarrier_arrive(
                                    math_barriers
                                    + self.num_umma_stages
                                    + umma_cons_state_0.index
                                )
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.native_lowp):
                                for h in cutlass.range_constexpr(
                                    0, 64 // num_epi_subtiles, 4
                                ):
                                    x4 = _subview(tTR_rAcc, h, 4)
                                    woff = (i * (64 // num_epi_subtiles) + h) // 2
                                    s_acc[0] = native_fma(
                                        self.output_dtype,
                                        native_pack_relu(
                                            self.output_dtype, x4[0], x4[1]
                                        ),
                                        native_weights[woff],
                                        s_acc[0],
                                    )
                                    s_acc[1] = native_fma(
                                        self.output_dtype,
                                        native_pack_relu(
                                            self.output_dtype, x4[2], x4[3]
                                        ),
                                        native_weights[woff + 1],
                                        s_acc[1],
                                    )
                            else:
                                for h in cutlass.range_constexpr(
                                    0, 64 // num_epi_subtiles, 4
                                ):
                                    x4 = _subview(tTR_rAcc, h, 4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        if cutlass.const_expr(
                                            (
                                                self.dsv4_integer_head_mask
                                                >> (i * (64 // num_epi_subtiles) + h)
                                            )
                                            & 1
                                        ):
                                            a4 = (
                                                finite_integer_relu4(x4)
                                                if cutlass.const_expr(
                                                    self.dsv4_finite_integer
                                                )
                                                else integer_relu4(x4)
                                            )
                                        else:
                                            a4 = indexer_add_packed_f32x2(
                                                x4, abs(x4), rnd=_RND_RN
                                            )
                                    else:
                                        a4 = cute.max(x4, zero4, propagate_nan=True)
                                    if cutlass.const_expr(
                                        i * (64 // num_epi_subtiles) + h
                                        < self.max_w_in_reg
                                    ):
                                        w4 = _subview(
                                            w_cache, i * (64 // num_epi_subtiles) + h, 4
                                        )
                                    else:
                                        w4 = cute.make_rmem_tensor(
                                            (4,), cutlass.Float32
                                        )
                                        for j in cutlass.range_constexpr(4):
                                            w = sW[
                                                (
                                                    query_slot * 64
                                                    + i * (64 // num_epi_subtiles)
                                                    + h
                                                    + j,
                                                    q_stage_local,
                                                )
                                            ]
                                            if cutlass.const_expr(
                                                self.use_relu_trick
                                                and not self.weights_prepared
                                            ):
                                                w = w * 0.5
                                            if cutlass.const_expr(
                                                self.use_relu_trick
                                                and (
                                                    (
                                                        self.dsv4_integer_head_mask
                                                        >> (
                                                            i * (64 // num_epi_subtiles)
                                                            + h
                                                            + j
                                                        )
                                                    )
                                                    & 1
                                                )
                                            ):
                                                w = w * 2.0
                                            w4[j] = w
                                        w4 = w4.load()
                                    s_acc = indexer_fma_packed_f32x2(
                                        a4, w4, s_acc, rnd=_RND_RN
                                    )
                        if cutlass.const_expr(self.native_lowp):
                            reduced = native_add(self.output_dtype, s_acc[0], s_acc[1])
                            score = (
                                native_sum_fp32(self.output_dtype, reduced) * scale_val
                            )
                        else:
                            score = s_acc[0] + s_acc[1] + s_acc[2] + s_acc[3]
                        if outrow < num_q_tokens:
                            valid = (kv_pos >= kv_start) & (kv_pos < kv_end)
                            cursor = publish_bank(
                                score,
                                valid,
                                tau,
                                0,
                                kv_pos,
                                cursor,
                                bank_rows,
                                self.output_dtype,
                                raw_half=self.dsv4_cast,
                                unordered=self.dsv4_unordered,
                                raw_packed=(
                                    self.dsv4_packed_store
                                    and bank_rows.element_type == cutlass.Int32
                                ),
                            )
                        kv_idx = kv_idx + 1
                while kv_idx + (6 if cutlass.const_expr(self.dsv4_unroll) else 30) < (
                    min(num_kv, kv_end // 128 - math_group)
                    if cutlass.const_expr(self.dsv4_full_tiles)
                    else num_kv
                ):
                    for kv_unroll in cutlass.range_constexpr(
                        4 if self.dsv4_unroll else 16
                    ):
                        kv_pos = kv_idx * 128 + m_coord
                        scale_val = mKeyScale[min(kv_pos, num_kv_tokens - 1)]
                        cute.arch.mbarrier_wait(
                            math_barriers + umma_cons_state_0.index,
                            umma_cons_state_0.phase,
                        )
                        tAcc = tCtAcc_base_0[
                            (None, None, None, umma_cons_state_0.index)
                        ][((None, None), 0, 0)]
                        tTR = thr_copy_ref.partition_S(
                            cute.flat_divide(tAcc, epi_sub_mn)
                        )
                        if cutlass.const_expr(self.native_lowp):
                            s_acc = cute.make_rmem_tensor((2,), cutlass.Uint32)
                            s_acc.fill(0)
                        else:
                            s_acc = cute.full((4,), 0.0, dtype=cutlass.Float32)
                        for i in cutlass.range_constexpr(num_epi_subtiles):
                            cute.copy(
                                tiled_copy_ref,
                                tTR[
                                    (
                                        None,
                                        None,
                                        None,
                                        0,
                                        query_slot * num_epi_subtiles + i,
                                    )
                                ],
                                tTR_rAcc,
                            )
                            cute.arch.fence_view_async_tmem_load()
                            if cutlass.const_expr(i == num_epi_subtiles - 1):
                                # 256 consumers release only after both query slices
                                # are in registers. Reuse also protects staged SFA.
                                cute.arch.mbarrier_arrive(
                                    math_barriers
                                    + self.num_umma_stages
                                    + umma_cons_state_0.index
                                )
                                umma_cons_state_0.advance()
                            if cutlass.const_expr(self.native_lowp):
                                for h in cutlass.range_constexpr(
                                    0, 64 // num_epi_subtiles, 4
                                ):
                                    x4 = _subview(tTR_rAcc, h, 4)
                                    woff = (i * (64 // num_epi_subtiles) + h) // 2
                                    s_acc[0] = native_fma(
                                        self.output_dtype,
                                        native_pack_relu(
                                            self.output_dtype, x4[0], x4[1]
                                        ),
                                        native_weights[woff],
                                        s_acc[0],
                                    )
                                    s_acc[1] = native_fma(
                                        self.output_dtype,
                                        native_pack_relu(
                                            self.output_dtype, x4[2], x4[3]
                                        ),
                                        native_weights[woff + 1],
                                        s_acc[1],
                                    )
                            else:
                                for h in cutlass.range_constexpr(
                                    0, 64 // num_epi_subtiles, 4
                                ):
                                    x4 = _subview(tTR_rAcc, h, 4)
                                    if cutlass.const_expr(self.use_relu_trick):
                                        if cutlass.const_expr(
                                            (
                                                self.dsv4_integer_head_mask
                                                >> (i * (64 // num_epi_subtiles) + h)
                                            )
                                            & 1
                                        ):
                                            a4 = (
                                                finite_integer_relu4(x4)
                                                if cutlass.const_expr(
                                                    self.dsv4_finite_integer
                                                )
                                                else integer_relu4(x4)
                                            )
                                        else:
                                            a4 = indexer_add_packed_f32x2(
                                                x4, abs(x4), rnd=_RND_RN
                                            )
                                    else:
                                        a4 = cute.max(x4, zero4, propagate_nan=True)
                                    if cutlass.const_expr(
                                        i * (64 // num_epi_subtiles) + h
                                        < self.max_w_in_reg
                                    ):
                                        w4 = _subview(
                                            w_cache, i * (64 // num_epi_subtiles) + h, 4
                                        )
                                    else:
                                        w4 = cute.make_rmem_tensor(
                                            (4,), cutlass.Float32
                                        )
                                        for j in cutlass.range_constexpr(4):
                                            w = sW[
                                                (
                                                    query_slot * 64
                                                    + i * (64 // num_epi_subtiles)
                                                    + h
                                                    + j,
                                                    q_stage_local,
                                                )
                                            ]
                                            if cutlass.const_expr(
                                                self.use_relu_trick
                                                and not self.weights_prepared
                                            ):
                                                w = w * 0.5
                                            if cutlass.const_expr(
                                                self.use_relu_trick
                                                and (
                                                    (
                                                        self.dsv4_integer_head_mask
                                                        >> (
                                                            i * (64 // num_epi_subtiles)
                                                            + h
                                                            + j
                                                        )
                                                    )
                                                    & 1
                                                )
                                            ):
                                                w = w * 2.0
                                            w4[j] = w
                                        w4 = w4.load()
                                    s_acc = indexer_fma_packed_f32x2(
                                        a4, w4, s_acc, rnd=_RND_RN
                                    )
                        if cutlass.const_expr(self.native_lowp):
                            reduced = native_add(self.output_dtype, s_acc[0], s_acc[1])
                            score = (
                                native_sum_fp32(self.output_dtype, reduced) * scale_val
                            )
                        else:
                            score = s_acc[0] + s_acc[1] + s_acc[2] + s_acc[3]
                        if outrow < num_q_tokens:
                            valid = (
                                True
                                if cutlass.const_expr(self.dsv4_full_tiles)
                                else (kv_pos >= kv_start) & (kv_pos < kv_end)
                            )
                            cursor = publish_bank(
                                score,
                                valid,
                                tau,
                                0,
                                kv_pos,
                                cursor,
                                bank_rows,
                                self.output_dtype,
                                raw_half=self.dsv4_cast,
                                unordered=self.dsv4_unordered,
                                raw_packed=(
                                    self.dsv4_packed_store
                                    and bank_rows.element_type == cutlass.Int32
                                ),
                            )
                        kv_idx = kv_idx + 1
                while kv_idx < num_kv:
                    kv_pos = kv_idx * 128 + m_coord
                    scale_val = mKeyScale[min(kv_pos, num_kv_tokens - 1)]
                    cute.arch.mbarrier_wait(
                        math_barriers + umma_cons_state_0.index, umma_cons_state_0.phase
                    )
                    tAcc = tCtAcc_base_0[(None, None, None, umma_cons_state_0.index)][
                        ((None, None), 0, 0)
                    ]
                    tTR = thr_copy_ref.partition_S(cute.flat_divide(tAcc, epi_sub_mn))
                    if cutlass.const_expr(self.native_lowp):
                        s_acc = cute.make_rmem_tensor((2,), cutlass.Uint32)
                        s_acc.fill(0)
                    else:
                        s_acc = cute.full((4,), 0.0, dtype=cutlass.Float32)
                    for i in cutlass.range_constexpr(num_epi_subtiles):
                        cute.copy(
                            tiled_copy_ref,
                            tTR[
                                (None, None, None, 0, query_slot * num_epi_subtiles + i)
                            ],
                            tTR_rAcc,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(i == num_epi_subtiles - 1):
                            # 256 consumers release only after both query slices
                            # are in registers. Reuse also protects staged SFA.
                            cute.arch.mbarrier_arrive(
                                math_barriers
                                + self.num_umma_stages
                                + umma_cons_state_0.index
                            )
                            umma_cons_state_0.advance()
                        if cutlass.const_expr(self.native_lowp):
                            for h in cutlass.range_constexpr(
                                0, 64 // num_epi_subtiles, 4
                            ):
                                x4 = _subview(tTR_rAcc, h, 4)
                                woff = (i * (64 // num_epi_subtiles) + h) // 2
                                s_acc[0] = native_fma(
                                    self.output_dtype,
                                    native_pack_relu(self.output_dtype, x4[0], x4[1]),
                                    native_weights[woff],
                                    s_acc[0],
                                )
                                s_acc[1] = native_fma(
                                    self.output_dtype,
                                    native_pack_relu(self.output_dtype, x4[2], x4[3]),
                                    native_weights[woff + 1],
                                    s_acc[1],
                                )
                        else:
                            for h in cutlass.range_constexpr(
                                0, 64 // num_epi_subtiles, 4
                            ):
                                x4 = _subview(tTR_rAcc, h, 4)
                                if cutlass.const_expr(self.use_relu_trick):
                                    if cutlass.const_expr(
                                        (
                                            self.dsv4_integer_head_mask
                                            >> (i * (64 // num_epi_subtiles) + h)
                                        )
                                        & 1
                                    ):
                                        a4 = (
                                            finite_integer_relu4(x4)
                                            if cutlass.const_expr(
                                                self.dsv4_finite_integer
                                            )
                                            else integer_relu4(x4)
                                        )
                                    else:
                                        a4 = indexer_add_packed_f32x2(
                                            x4, abs(x4), rnd=_RND_RN
                                        )
                                else:
                                    a4 = cute.max(x4, zero4, propagate_nan=True)
                                if cutlass.const_expr(
                                    i * (64 // num_epi_subtiles) + h < self.max_w_in_reg
                                ):
                                    w4 = _subview(
                                        w_cache, i * (64 // num_epi_subtiles) + h, 4
                                    )
                                else:
                                    w4 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    for j in cutlass.range_constexpr(4):
                                        w = sW[
                                            (
                                                query_slot * 64
                                                + i * (64 // num_epi_subtiles)
                                                + h
                                                + j,
                                                q_stage_local,
                                            )
                                        ]
                                        if cutlass.const_expr(
                                            self.use_relu_trick
                                            and not self.weights_prepared
                                        ):
                                            w = w * 0.5
                                        if cutlass.const_expr(
                                            self.use_relu_trick
                                            and (
                                                (
                                                    self.dsv4_integer_head_mask
                                                    >> (
                                                        i * (64 // num_epi_subtiles)
                                                        + h
                                                        + j
                                                    )
                                                )
                                                & 1
                                            )
                                        ):
                                            w = w * 2.0
                                        w4[j] = w
                                    w4 = w4.load()
                                s_acc = indexer_fma_packed_f32x2(
                                    a4, w4, s_acc, rnd=_RND_RN
                                )
                    if cutlass.const_expr(self.native_lowp):
                        reduced = native_add(self.output_dtype, s_acc[0], s_acc[1])
                        score = native_sum_fp32(self.output_dtype, reduced) * scale_val
                    else:
                        score = s_acc[0] + s_acc[1] + s_acc[2] + s_acc[3]
                    if outrow < num_q_tokens:
                        valid = (kv_pos >= kv_start) & (kv_pos < kv_end)
                        cursor = publish_bank(
                            score,
                            valid,
                            tau,
                            0,
                            kv_pos,
                            cursor,
                            bank_rows,
                            self.output_dtype,
                            raw_half=self.dsv4_cast,
                            unordered=self.dsv4_unordered,
                            raw_packed=(
                                self.dsv4_packed_store
                                and bank_rows.element_type == cutlass.Int32
                            ),
                        )
                    kv_idx = kv_idx + 1
                if outrow < num_q_tokens and cute.arch.lane_idx() == 0:
                    # Each bank has an even span. Publish initialized
                    # padding for K3's final 128-bit candidate pair.
                    if ((cursor & 1) != 0) & (cursor < bank_span):
                        bank_rows[0, cursor] = bank_rows.element_type(0)
                    candidates[outrow, bank_id] = candidates.element_type(cursor)
                    candidates[outrow, bank_id + 4] = candidates.element_type(0)
                    cute.arch.atomic_add(
                        (counts.iterator + outrow).llvm_ptr,
                        cursor - cutlass.Int32(cursor > bank_span) * 16777216,
                        sem="relaxed",
                        scope="gpu",
                    )
                q_pipeline.consumer_release(q_cons_state)
                q_cons_state.advance()
                next_q_block_idx = q_block_idx + cute.arch.grid_dim()[2]
                if next_q_block_idx < num_q_blocks:
                    q_last_next = min(next_q_block_idx * 4 + 3, num_q_tokens - 1)
                    next_num_kv = (mCuSeqLenKEnd[q_last_next] + 127) // 128
                has_work = next_q_block_idx < num_q_blocks
            if cutlass.const_expr(self.num_heads != 32):
                tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        else:
            if cutlass.const_expr(self.mode == "filter"):
                if cutlass.const_expr(self.store_registers >= 128):
                    cute.arch.warpgroup_reg_alloc(self.store_registers)
                else:
                    cute.arch.warpgroup_reg_dealloc(self.store_registers)
            else:
                cute.arch.warpgroup_reg_dealloc(self.store_registers)


def compile_shared_filter(k, capacity, sms, signature, pdl):
    """Compile against the existing flat FP8 K2 argument layout."""
    from ..api import DTYPES

    attrs = dict(
        pdl=pdl,
        num_heads=64,
        head_dim=128,
        num_sms=sms,
        epi_dtype=cutlass.Float32,
        weight_dtype=DTYPES[signature[3][0]],
        output_dtype=cutlass.Float16,
        mode="filter",
        top_k=k,
        capacity=capacity,
        use_relu_trick=False,
        use_batched_store=False,
        num_umma_stages=1,
        num_kv_stages=4,
        num_q_stages=2,
        max_w_in_reg=64,
        num_epi_subtiles=1,
        math_registers=112,
        store_registers=32,
        native_fp16=True,
        remove_online_sf_transpose=True,
        dsv4_unordered=True,
        dsv4_cast=True,
        dsv4_full_tiles=True,
        dsv4_packed_store=True,
    )
    tensors = []
    for i, (dtype, shape) in enumerate(signature):
        dims = tuple(cute.sym_int() if d < 0 else d for d in shape)
        if i in (4, 9):
            tensor = cute.runtime.make_fake_tensor(
                DTYPES[dtype], dims, stride=(cute.sym_int64(), 1), assumed_align=16
            )
        else:
            tensor = cute.runtime.make_fake_compact_tensor(
                DTYPES[dtype],
                dims,
                stride_order=tuple(reversed(range(len(shape)))),
                assumed_align=16,
            )
        tensors.append(tensor)
    return cute.compile(
        FP8PrefillMQALogitsKernel(**attrs),
        *tensors,
        cutlass.Int32(1),
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
