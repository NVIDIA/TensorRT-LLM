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
"""indexer_topk row-split FP8 paged MQA logits kernel for Blackwell (SM100 family).

Scores all ``next_n`` verify rows of a request against one stream of the
paged FP8 indexer K cache, derived from the native CuTe DSL kernel
``attention/kernels/blackwell/paged_mqa_logits/fp8_paged_mqa_logits``.

Differences from the native kernel (same TMA/UMMA building blocks, same
epilogue arithmetic per element):

- One KV stream instead of two KV groups: a single TMA warp loads every
  128-token compute tile into a deeper KV ring, a single UMMA warp issues
  ``M=128 x N=next_n*H x K=128`` per tile into a two-stage TMEM ring, so the
  MMA of tile ``j+1`` overlaps the epilogue of tile ``j``.
- All math warpgroups (2 or 3) consume the **same** tile: warpgroup ``w``
  owns verify rows ``[w*r, (w+1)*r)`` with ``r = next_n / num_math_wg`` (the
  TMEM columns of those rows). Each math thread still owns one token (TMEM
  lane) and keeps all weights of its rows in registers. Three warpgroups
  give the schedulers more math warps to hide latency with.

Per (row t, token j) the epilogue computes, exactly as the native kernel,
``(sum_h w[t,h] * relu(acc[t,h,j])) * k_scale[j]`` with four fp32 partial
sums over heads in steps of 4, so logits are bit-identical to the native
launch for any row grouping.

Requirements: ``next_n`` divisible by ``num_math_wg``, ``N = next_n *
num_heads`` a multiple of 16, ``2 * N <= 512`` TMEM columns,
``num_heads % 4 == 0``, fp32 everywhere.

Scheduler: the DeepGEMM paged-MQA schedule (``schedule_meta[sm] = (q_idx,
kv_split)`` with 256-token splits) is consumed as in the native kernel, with
tiles advanced one at a time instead of in pairs.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

_RND_RN = "rn"
_TMEM_COLS = 512
_REGS_PER_SM = 65536
_PRODUCER_REGS = 24
# Schedule splits are 256 tokens = 2 compute tiles of 128 tokens.
_TILES_PER_SPLIT = 2


class FP8MQALogitsRowSplitKernel:
    """Persistent paged MQA logits kernel with a row-split epilogue.

    Args:
        block_kv: Compute tile in tokens (MMA M), 128.
        phys_block_kv: Cache page in tokens; must divide ``block_kv``.
        num_heads: Indexer heads ``H``.
        head_dim: Indexer head dimension ``D`` (MMA K), 128.
        next_n: Verify rows per request (divisible by ``num_math_wg``).
        num_sms: Persistent grid size; must match the schedule.
        num_kv_stages: KV ring depth in compute tiles.
        num_q_stages: Query/weight ring depth.
        num_math_wg: Math warpgroups sharing each tile (2 or 3).
        debug: Performance-diagnosis bit flags, never for production (results
            are invalid when non-zero): 1 skips the epilogue reduction (stores
            one accumulator per row), 2 skips the MMA (the accumulator stage
            is committed unwritten), 4 skips the scale TMA loads (scale 1).
    """

    def __init__(
        self,
        block_kv: int = 128,
        phys_block_kv: int = 64,
        num_heads: int = 32,
        head_dim: int = 128,
        next_n: int = 6,
        num_sms: int = 148,
        num_kv_stages: int = 6,
        num_q_stages: int = 2,
        num_math_wg: int = 2,
        debug: int = 0,
    ):
        if block_kv % phys_block_kv != 0 or block_kv // phys_block_kv > 4:
            raise ValueError(f"phys_block_kv={phys_block_kv} must divide {block_kv} (<= 4 pages)")
        if num_math_wg not in (2, 3) or next_n % num_math_wg != 0:
            raise ValueError(f"next_n={next_n} must split over {num_math_wg} math warpgroups")
        if num_heads % 4 != 0:
            raise ValueError("num_heads must be a multiple of 4")
        n_cols = next_n * num_heads
        if n_cols % 16 != 0 or n_cols > 256 or 2 * n_cols > _TMEM_COLS:
            raise ValueError(f"next_n * num_heads = {n_cols} does not fit two TMEM stages")
        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        self.num_blocks_per_mma = block_kv // phys_block_kv
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.next_n = next_n
        self.num_math_wg = num_math_wg
        self.rows_per_wg = next_n // num_math_wg
        self.N = n_cols
        self.num_sms = num_sms
        self.num_kv_stages = num_kv_stages
        self.num_q_stages = num_q_stages
        self.num_umma_stages = 2
        self.debug = debug
        self.acc_dtype = cutlass.Float32
        self.w_stage_stride = (self.N * 4 + 127) // 128 * 128 // 4
        self.num_math_warps = 4 * num_math_wg
        self.num_math_threads = 128 * num_math_wg
        self.threads_per_cta = self.num_math_threads + 128
        self.tma_warp = self.num_math_warps
        self.umma_warp = self.num_math_warps + 2
        # Registers left for the math warpgroups once the producer warpgroup
        # drops to its minimum (setmaxnreg takes multiples of 8, at most 240).
        math_regs = (_REGS_PER_SM - 128 * _PRODUCER_REGS) // self.num_math_threads
        self.math_regs = min(240, math_regs // 8 * 8)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        full_k = mma_inst_shape_k * (self.head_dim // mma_inst_shape_k)
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], full_k)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, b_dtype, self.num_q_stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = self.num_tmem_alloc_cols * self.num_umma_stages
        if self.num_tmem_alloc_cols_total > _TMEM_COLS:
            raise ValueError("two TMEM stages exceed the 512 TMEM columns")
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,  # [num_phys_blocks, block_bytes] uint8
        b: cute.Tensor,  # Q: [N, head_dim, batch_size] uint8 (FP8 bytes)
        weights: cute.Tensor,  # [N, batch_size] fp32
        logits: cute.Tensor,  # [batch_size * next_n, max_context_len] fp32
        block_table: cute.Tensor,  # [batch_size, max_blocks_per_seq] int32
        context_lens: cute.Tensor,  # [batch_size] int32
        schedule_meta: cute.Tensor,  # [num_sms + 1, 2] int32
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        phys_block_kv = self.phys_block_kv
        scale_offset_elems = phys_block_kv * self.head_dim
        kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)
        kv_block_stride = kv_fused.layout.stride[0]
        kv_layout = cute.make_layout(
            (phys_block_kv, self.head_dim, num_phys_blocks),
            stride=(self.head_dim, 1, kv_block_stride),
        )
        a = cute.make_tensor(kv_fp8.iterator, kv_layout)
        scale_fp8_layout = cute.make_layout(
            (phys_block_kv * 4, num_phys_blocks), stride=(1, kv_block_stride)
        )
        scale_fp8 = cute.make_tensor(kv_fp8.iterator + scale_offset_elems, scale_fp8_layout)
        scales = cute.recast_tensor(scale_fp8, cutlass.Float32)

        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup_mma(a.element_type, b.element_type, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_view_layout = sm100_utils.make_smem_layout(
            tcgen05.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a.element_type,
            self.num_kv_stages,
        )
        self.a_tma_view_layout = cute.tiled_divide(
            self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
        )
        self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op, a, self.a_tma_view_layout[0], (self.phys_block_kv, self.head_dim)
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages), stride=(1, self.w_stage_stride)
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, weights, w_smem_per_stage, self.w_smem_layout_staged.shape[:1]
        )

        self.s_smem_layout_staged = cute.make_layout((self.block_kv, self.num_kv_stages))
        s_smem_per_subblock = cute.make_layout((phys_block_kv,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, scales, s_smem_per_subblock, (phys_block_kv,)
        )

        b_copy_size = cute.size_in_bytes(b.element_type, b_smem_layout)
        w_copy_size = self.N * 4
        scale_bytes = 0 if self.debug & 4 else phys_block_kv * 4
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            phys_block_kv * self.head_dim + scale_bytes
        )
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            batch_size,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            self.a_tma_view_layout,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, self.num_sms),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,
        mLogits: cute.Tensor,
        mBlockTable: cute.Tensor,
        mContextLens: cute.Tensor,
        mScheduleMeta: cute.Tensor,
        batch_size: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < self.num_math_warps
        is_tma_warp = warp_idx == self.tma_warp
        is_umma_warp = warp_idx == self.umma_warp

        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        sm_idx = bidz
        start_q = mScheduleMeta[(sm_idx, 0)]
        start_kv_split = mScheduleMeta[(sm_idx, 1)]
        end_q_idx = mScheduleMeta[(sm_idx + 1, 0)]
        end_kv_split = mScheduleMeta[(sm_idx + 1, 1)]
        start_q_clamped = min(start_q, batch_size - 1)
        current_num_kv = (mContextLens[start_q_clamped] + self.block_kv - 1) // self.block_kv

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        block_kv_val = self.block_kv
        num_heads = self.num_heads
        next_n = self.next_n
        rows_per_wg = self.rows_per_wg

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Q + weights: TMA producer, every math warp releases (lane 0 each).
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_math_warps),
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        # KV + scales: TMA producer; UMMA waits (no release); every math
        # warp reads the tile's scales and releases.
        kv_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_math_warps),
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_cons_state_umma = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        # Accumulator ring: UMMA producer, all math threads consume.
        umma_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_math_threads),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )

        # TMEM: math warps + UMMA warp; math warp 0 allocates.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_math_threads + 32
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sKV = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sW = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=w_smem_layout_staged, byte_alignment=128
        )
        sScales = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=s_smem_layout_staged, byte_alignment=128
        )

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        sKV_for_tma = cute.make_tensor(sKV.iterator, a_tma_view_layout.outer)
        gA = cute.local_tile(mA_mkl, (self.phys_block_kv, self.head_dim), coord=(None, None, None))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sKV_for_tma, cute.group_modes(gA, 0, 2)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, None)]
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout((1,)),
            cute.group_modes(sW, 0, 1),
            cute.group_modes(mW_tma, 0, 1),
        )
        s_tma_view_layout = cute.make_layout(
            (self.phys_block_kv, self.num_blocks_per_mma, self.num_kv_stages),
            stride=(1, self.phys_block_kv, self.block_kv),
        )
        sScales_for_tma = cute.make_tensor(sScales.iterator, s_tma_view_layout)
        gS = cute.local_tile(mS_tma, (self.phys_block_kv,), coord=(None, None))
        tSsS, tSgS = cpasync.tma_partition(tma_atom_s, 0, cute.make_layout(1), sScales_for_tma, gS)

        tCrA = tiled_mma.make_fragment_A(sKV)
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_shape_staged = cute.append(acc_shape, self.num_umma_stages)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # Epilogue: one TMEM load per warpgroup and tile covering its
        # rows_per_wg contiguous verify rows (rows_per_wg * num_heads columns).
        epi_sub_mn = (epi_tile[0], rows_per_wg * num_heads)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # Scheduler state (tiles of block_kv tokens, advanced one at a time).
        end_kv_idx = end_kv_split * _TILES_PER_SPLIT
        next_q_idx = start_q
        next_kv_idx = start_kv_split * _TILES_PER_SPLIT
        next_num_kv = current_num_kv
        q_idx = batch_size
        has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if is_tma_warp:
            cute.arch.warpgroup_reg_dealloc(_PRODUCER_REGS)
            lane_idx = tidx % 32
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)

            q_pipeline.producer_acquire(q_prod_state)
            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0, next_q_idx)],
                tBsB[(None, q_prod_state.index)],
                tma_bar_ptr=q_bar,
                mcast_mask=b_mcast_mask,
            )
            cute.copy(
                tma_atom_w,
                tWgW[(None, next_q_idx)],
                tWsW[(None, q_prod_state.index)],
                tma_bar_ptr=q_bar,
            )
            q_prod_state.advance()

            while has_work:
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                if q_idx != q_idx_old:
                    kv_blk_ptr = cutlass.Int32(32)
                    prefetch_next = q_idx + 1
                    if prefetch_next < end_q_idx:
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, prefetch_next)],
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[(None, prefetch_next)],
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()
                    elif prefetch_next == end_q_idx:
                        if end_kv_idx > 0:
                            q_pipeline.producer_acquire(q_prod_state)
                            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                            cute.copy(
                                tma_atom_b,
                                tBgB[(None, 0, prefetch_next)],
                                tBsB[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                                mcast_mask=b_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_w,
                                tWgW[(None, prefetch_next)],
                                tWsW[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                            )
                            q_prod_state.advance()

                # Page-table prefetch: lane l caches the pages of tile kv_idx + l.
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + lane_idx
                    if prefetch_kv < num_kv:
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[(q_idx, base_phys + i)]
                    else:
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = cutlass.Int32(0)
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1

                kv_pipeline.producer_acquire(kv_prod_state)
                bar = kv_pipeline.producer_get_barrier(kv_prod_state)
                stage = kv_prod_state.index
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, 0, 0, phys_blks[i])],
                        tAsA[(None, i, stage)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    if cutlass.const_expr(not self.debug & 4):
                        cute.copy(
                            tma_atom_s,
                            tSgS[(None, 0, phys_blks[i])],
                            tSsS[(None, i, stage)],
                            tma_bar_ptr=bar,
                        )
                kv_prod_state.advance()

                next_kv_idx = kv_idx + 1
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (mContextLens[next_q_idx] + block_kv_val - 1) // block_kv_val
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_umma_warp:
            cute.arch.warpgroup_reg_dealloc(_PRODUCER_REGS)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            if is_leader_cta:
                num_k_blocks = cute.size(tCrA.shape[2])
                q_stage = cutlass.Int32(0)
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma)
                        q_stage = q_cons_state_umma.index

                    kv_pipeline.consumer_wait(kv_cons_state_umma)
                    umma_pipeline.producer_acquire(umma_prod_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage = kv_cons_state_umma.index
                    tCtAcc = tCtAcc_base[(None, None, None, umma_prod_state.index)]
                    if cutlass.const_expr(not self.debug & 2):
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[None, None, k_block, kv_stage],
                                tCrB[None, None, k_block, q_stage],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_cons_state_umma.advance()
                    umma_pipeline.producer_commit(umma_prod_state)
                    umma_prod_state.advance()

                    next_kv_idx = kv_idx + 1
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(self.math_regs)
            tmem.allocate(self.num_tmem_alloc_cols_total)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            local_tidx = tidx % 128
            cC = cute.make_identity_tensor(epi_sub_mn)

            tAcc_ref = tCtAcc_base[(None, None, None, 0)][((None, None), 0, 0)]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref, epi_sub_mn)
            tiled_copy_ref = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_ref_epi[(None, None, 0, 0)])
            thr_copy_ref = tiled_copy_ref.get_slice(local_tidx)
            tTR_cC = thr_copy_ref.partition_D(cC)
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

            # This warpgroup's verify rows: [row0, row0 + rows_per_wg).
            row0 = warpgroup_idx * rows_per_wg
            w_cache = cute.make_rmem_tensor(rows_per_wg * num_heads, cutlass.Float32)
            q_stage_local = cutlass.Int32(0)

            while has_work:
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                if q_idx != q_idx_old:
                    if q_idx_old < batch_size:
                        q_pipeline.consumer_release(q_cons_state)
                        q_cons_state.advance()
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage_local = q_cons_state.index
                    for t_i in cutlass.range_constexpr(rows_per_wg):
                        for w_j in cutlass.range_constexpr(num_heads):
                            w_cache[t_i * num_heads + w_j] = sW[
                                ((row0 + t_i) * num_heads + w_j, q_stage_local)
                            ]

                kv_pos = kv_idx * block_kv_val + m_coord
                kv_pipeline.consumer_wait(kv_cons_state_math)
                scale_val = sScales[(m_coord, kv_cons_state_math.index)]
                umma_pipeline.consumer_wait(umma_cons_state)
                kv_pipeline.consumer_release(kv_cons_state_math)
                kv_cons_state_math.advance()

                tAcc = tCtAcc_base[(None, None, None, umma_cons_state.index)]
                tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0)], epi_sub_mn)
                tc = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
                tr = tc.get_slice(local_tidx)
                tTR = tr.partition_S(tAcc_epi)

                # One load of all this warpgroup's columns, one wait, then the
                # accumulator stage goes back to the UMMA warp.
                cute.copy(tc, tTR[(None, None, None, 0, warpgroup_idx)], tTR_rAcc)
                cute.arch.fence_view_async_tmem_load()
                umma_pipeline.consumer_release(umma_cons_state)
                umma_cons_state.advance()
                acc_vec = tTR_rAcc.load()

                if cutlass.const_expr(self.debug & 1):
                    for t in cutlass.range_constexpr(rows_per_wg):
                        mLogits[(q_idx * next_n + row0 + t, kv_pos)] = (
                            acc_vec[t * num_heads] * scale_val
                        )
                for t in cutlass.range_constexpr(0 if self.debug & 1 else rows_per_wg):
                    s0x = cutlass.Float32(0.0)
                    s0y = cutlass.Float32(0.0)
                    s1x = cutlass.Float32(0.0)
                    s1y = cutlass.Float32(0.0)
                    for h in cutlass.range_constexpr(0, num_heads, 4):
                        c0 = t * num_heads + h
                        a0 = cutlass.max(acc_vec[c0], cutlass.Float32(0.0))
                        a1 = cutlass.max(acc_vec[c0 + 1], cutlass.Float32(0.0))
                        a2 = cutlass.max(acc_vec[c0 + 2], cutlass.Float32(0.0))
                        a3 = cutlass.max(acc_vec[c0 + 3], cutlass.Float32(0.0))
                        r0 = t * num_heads + h
                        s0x, s0y = cute.arch.fma_packed_f32x2(
                            (a0, a1), (w_cache[r0], w_cache[r0 + 1]), (s0x, s0y), rnd=_RND_RN
                        )
                        s1x, s1y = cute.arch.fma_packed_f32x2(
                            (a2, a3), (w_cache[r0 + 2], w_cache[r0 + 3]), (s1x, s1y), rnd=_RND_RN
                        )
                    result_t = s0x + s0y + s1x + s1y
                    mLogits[(q_idx * next_n + row0 + t, kv_pos)] = result_t * scale_val

                next_kv_idx = kv_idx + 1
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (mContextLens[next_q_idx] + block_kv_val - 1) // block_kv_val
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            if q_idx < batch_size:
                q_pipeline.consumer_release(q_cons_state)
                q_cons_state.advance()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        else:
            cute.arch.warpgroup_reg_dealloc(_PRODUCER_REGS)
