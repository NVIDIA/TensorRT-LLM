# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""CuTe DSL Q8KV8 paged prefill indexer proxy scores for SM100/SM103."""

import enum
import os

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cuda.bindings.driver as cuda
from quack import copy_utils



# MiniMax-M3.1 vendoring (fmha_sm100_q8kv8, MSA; CTX handoff HANDOFF_CTX_IDX_SCORER_M2_v1 6a964784): the MMA instruction K of
# the SM107 FP8 path. 32 = EXACT mode (default): the generic F8F6F4 MMA, bitwise to the served fmha proxy scores. 64 = the
# upstream SM107 FP8 instruction (competitor numerics: a different accumulation order), only with
# TRTLLM_M31_IDX_SCORER_MMA_K=64 and behind MSA's parity gates. Read once, at import; part of every compile key.
MMA_K = 64 if os.environ.get("TRTLLM_M31_IDX_SCORER_MMA_K", "32") == "64" else 32

class NamedBarrierIndexerSm100(enum.IntEnum):
    """Kernel-local named barriers."""

    TmemPtr = enum.auto()
    Final = enum.auto()


class Q8KV8PrefillIndexerSm100:
    """Compute raw FP32 historical-page maxima with native FP8 UMMA.

    Q rows are ``token * num_heads + head``: all index heads share the single
    K head, so heads simply extend the Q (M) dimension of every task.
    """

    supported_compute_capabilities = frozenset({(10, 0), (10, 3), (10, 7)})
    head_dim = 128
    q_tile = 256
    k_tile = 128
    q_stages = 1
    # SM100/SM103: one 12-warp CTA per SM. SM107 uses launch_config instead.
    k_stages = 12
    acc_stages = 4
    ctas_per_sm = 1
    num_task_buckets = 8
    # Pages per coalesced score store group, and the padded row pitch of the
    # per-warp staging tile (odd, so row-major writes avoid bank conflicts).
    score_group_pages = 8
    score_buf_pitch = 9
    cta_group_size = 2
    # Task descriptor words: global Q row, Q position, first logical page,
    # batch index, and (q_rows - 1) | (page_count - 1) << task_page_count_shift.
    task_descriptor_words = 5
    task_page_count_shift = 8
    # Operand formats with a dedicated SM107 MMA instruction (K=64 per
    # instruction). Any other format, and every other architecture, goes
    # through the generic F8F6F4 instruction at K=32.
    sm107_fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)

    @classmethod
    def uses_sm107_fp8_mma(
        cls,
        compute_capability: tuple[int, int],
        io_dtype=cutlass.Float8E4M3FN,
    ) -> bool:
        # MiniMax-M3.1 vendoring (fmha_sm100_q8kv8): the exact mode (MMA_K == 32, the default) always takes the generic
        # F8F6F4 instruction at K = 32, whose accumulation order is the served fmha proxy's (bitwise, CTX 840485).
        return MMA_K == 64 and compute_capability == (10, 7) and io_dtype in cls.sm107_fp8_dtypes

    @classmethod
    def uses_two_ctas_per_sm(cls, compute_capability: tuple[int, int]) -> bool:
        """SM107 runs two 8-warp CTAs per SM and stages its score writes."""
        return compute_capability == (10, 7)

    @classmethod
    def launch_config(cls, compute_capability: tuple[int, int]) -> tuple[int, int, int]:
        """(CTAs per SM, K stages, accumulator stages) for an architecture."""
        if cls.uses_two_ctas_per_sm(compute_capability):
            return 2, 8, 2
        return cls.ctas_per_sm, cls.k_stages, cls.acc_stages

    @classmethod
    def mma_inst_k(
        cls,
        compute_capability: tuple[int, int],
        io_dtype=cutlass.Float8E4M3FN,
    ) -> int:
        return 64 if cls.uses_sm107_fp8_mma(compute_capability, io_dtype) else 32

    def __init__(
        self,
        *,
        compute_capability: tuple[int, int],
        num_persistent_clusters: int,
    ) -> None:
        if compute_capability not in self.supported_compute_capabilities:
            raise ValueError(
                "compute_capability must be SM100, SM103 or SM107, "
                f"got SM{compute_capability[0]}{compute_capability[1]}"
            )
        if num_persistent_clusters <= 0:
            raise ValueError("num_persistent_clusters must be positive")
        self.io_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.compute_capability = compute_capability
        # Two Q stages on SM107 so the next task's Q tile can land while the
        # current task's pages are still being consumed; the MMA warp reads
        # its Q operand from the current consumer stage.
        self.q_stages = 2 if compute_capability == (10, 7) else 1
        # SM107: two CTAs per SM (8 warps, 8 K stages, 2 accumulator stages and
        # 256 TMEM columns each) and coalesced score writes. Other architectures
        # keep one 12-warp CTA per SM with two score warp groups.
        self.two_ctas_per_sm = self.uses_two_ctas_per_sm(compute_capability)
        self.ctas_per_sm, self.k_stages, self.acc_stages = self.launch_config(
            compute_capability
        )
        self.num_persistent_clusters = num_persistent_clusters
        self.use_tmem_load_reduce = compute_capability in ((10, 3), (10, 7))
        self.cluster_shape_mnk = (2, 1, 1)
        self.mma_tiler_mnk = (self.q_tile, self.k_tile, self.head_dim)
        self.use_sm107_fp8_mma = self.uses_sm107_fp8_mma(
            compute_capability, self.io_dtype
        )
        self.mma_inst_shape_mnk = (
            self.q_tile,
            self.k_tile,
            self.mma_inst_k(compute_capability, self.io_dtype),
        )

        if self.two_ctas_per_sm:
            self.score_warp_ids = tuple(range(4))
            self.q_load_warp_id = 4
            self.k_load_warp_id = 5
            self.mma_warp_id = 6
            self.threads_per_cta = cute.arch.WARP_SIZE * 8
        else:
            self.score_warp_ids = tuple(range(8))
            self.q_load_warp_id = 8
            self.k_load_warp_id = 9
            self.mma_warp_id = 10
            self.score_worker_groups = 2
            self.threads_per_cta = cute.arch.WARP_SIZE * 12
        self.score_threads = cute.arch.WARP_SIZE * len(self.score_warp_ids)
        self.score_consumer_warps = 4
        # Register budgets per warp role. The launch allocates 168 registers
        # per thread (64K / 384 threads). On SM107 every role keeps that: the
        # score warps fit in 168, and growing them to 192 starved the loader
        # and MMA warps. SM100 and SM103 keep the original split. If a split is
        # used, 8 * 32 * (score - 168) <= 4 * 32 * (168 - other) must hold or
        # the score warps spin forever in setmaxregister_increase.
        if compute_capability == (10, 7):
            self.num_regs_score = 168
            self.num_regs_other = 88
        else:
            self.num_regs_score = 192
            self.num_regs_other = 32
        if self.two_ctas_per_sm:
            self.tmem_alloc_cols = self.acc_stages * self.k_tile
        else:
            self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPageTable: cute.Tensor,
        mScores: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        num_heads: cutlass.Int32,
        stream: cuda.CUstream = None,
    ):
        """Build descriptors and launch the 2-CTA kernel."""
        if cutlass.const_expr(self.use_sm107_fp8_mma):
            op = tcgen05.SM107MmaFP8Op(
                self.io_dtype,
                self.io_dtype,
                self.acc_dtype,
                self.mma_inst_shape_mnk,
                tcgen05.CtaGroup.TWO,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        else:
            op = tcgen05.MmaF8F6F4Op(
                self.io_dtype,
                self.io_dtype,
                self.acc_dtype,
                self.mma_inst_shape_mnk,
                tcgen05.CtaGroup.TWO,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        tiled_mma = cute.make_tiled_mma(op)
        q_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler_mnk,
            self.io_dtype,
            self.q_stages,
        )
        k_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler_mnk,
            self.io_dtype,
            self.k_stages,
        )
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.select(mQ.layout, mode=[0, 2, 1]),
        )[None, None, 0]
        mK = cute.make_tensor(
            mK.iterator,
            cute.select(mK.layout, mode=[1, 2, 0]),
        )
        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(
            cta_layout_mnk,
            (tiled_mma.thr_id,),
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO
        )
        q_tma_atom, q_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(q_smem_layout, mode=[0, 1, 2]),
            self.mma_tiler_mnk,
            tiled_mma,
            cta_layout_vmnk.shape,
        )
        k_tma_atom, k_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(k_smem_layout, mode=[0, 1, 2]),
            self.mma_tiler_mnk,
            tiled_mma,
            cta_layout_vmnk.shape,
        )

        # Score staging tile and row limits (SM107); one word otherwise.
        score_buf_words = 1
        row_limit_words = 1
        if cutlass.const_expr(self.two_ctas_per_sm):
            score_buf_words = (
                cute.arch.WARP_SIZE * self.score_buf_pitch * len(self.score_warp_ids)
            )
            row_limit_words = cute.arch.WARP_SIZE * len(self.score_warp_ids)

        @cute.struct
        class SharedStorage:
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stages * 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.acc_stages * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buffer: cutlass.Int32
            score_buf: cute.struct.MemRange[cutlass.Float32, score_buf_words]
            score_row_limit: cute.struct.MemRange[cutlass.Int32, row_limit_words]

        self.shared_storage = SharedStorage
        launcher = self.kernel(
            tiled_mma,
            q_tma_atom,
            q_tma_tensor,
            k_tma_atom,
            k_tma_tensor,
            cta_layout_vmnk,
            q_smem_layout,
            k_smem_layout,
            mPageTable,
            mScores,
            mTaskDescriptors,
            mTaskCounts,
            num_heads,
        )
        grid = (self.num_persistent_clusters * self.cta_group_size, 1, 1)
        if cutlass.const_expr(self.two_ctas_per_sm):
            launcher.launch(
                grid=grid,
                block=(self.threads_per_cta, 1, 1),
                cluster=self.cluster_shape_mnk,
                stream=stream,
                # Dynamic SMEM: the storage struct plus the 128B-aligned Q and K
                # stages; the launch sizes the carveout for ctas_per_sm from it.
                smem=self.shared_storage.size_in_bytes()
                + cute.size_in_bytes(self.io_dtype, q_smem_layout)
                + cute.size_in_bytes(self.io_dtype, k_smem_layout)
                + 2 * 128,
                min_blocks_per_mp=self.ctas_per_sm,
            )
        else:
            launcher.launch(
                grid=grid,
                block=(self.threads_per_cta, 1, 1),
                cluster=self.cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.jit
    def _get_task_descriptor(
        self,
        bucket: cutlass.Int32,
        task_idx: cutlass.Int32,
        mTaskDescriptors: cute.Tensor,
    ):
        """Load one task descriptor uniformly within a warp."""
        batch_idx = cutlass.Int32(0)
        q_global_begin = cutlass.Int32(0)
        q_position_begin = cutlass.Int32(0)
        page_begin = cutlass.Int32(0)
        q_rows = cutlass.Int32(0)
        page_count = cutlass.Int32(0)
        if cute.arch.lane_idx() == cutlass.Int32(0):
            q_global_begin = mTaskDescriptors[bucket, task_idx, 0]
            q_position_begin = mTaskDescriptors[bucket, task_idx, 1]
            page_begin = mTaskDescriptors[bucket, task_idx, 2]
            batch_idx = mTaskDescriptors[bucket, task_idx, 3]
            packed = mTaskDescriptors[bucket, task_idx, 4]
            page_count_shift = cutlass.Int32(self.task_page_count_shift)
            q_rows = (
                packed & ((cutlass.Int32(1) << page_count_shift) - cutlass.Int32(1))
            ) + cutlass.Int32(1)
            page_count = (packed >> page_count_shift) + cutlass.Int32(1)
        batch_idx = cute.arch.shuffle_sync(batch_idx, 0)
        q_global_begin = cute.arch.shuffle_sync(q_global_begin, 0)
        q_position_begin = cute.arch.shuffle_sync(q_position_begin, 0)
        page_begin = cute.arch.shuffle_sync(page_begin, 0)
        q_rows = cute.arch.shuffle_sync(q_rows, 0)
        page_count = cute.arch.shuffle_sync(page_count, 0)
        return (
            batch_idx,
            q_global_begin,
            q_position_begin,
            page_begin,
            q_rows,
            page_count,
        )

    @cute.jit
    def _flush_scores(
        self,
        sScoreBuf: cute.Tensor,
        sRowLimit: cute.Tensor,
        score_warp: cutlass.Int32,
        q_row_base: cutlass.Int32,
        group_page: cutlass.Int32,
        group_len: cutlass.Int32,
        mScores: cute.Tensor,
    ):
        """Write a warp's staged (32 rows x group) maxima, consecutive pages per lane group."""
        cute.arch.sync_warp()
        tile_shape = (cute.arch.WARP_SIZE, self.score_group_pages)
        gScores = cute.local_tile(
            cute.domain_offset((q_row_base, group_page), mScores), tile_shape, (0, 0)
        )
        sScores = sScoreBuf[(None, None, score_warp)]
        cScores = cute.make_identity_tensor(tile_shape)
        # 4 rows x 8 pages per store: lanes 8r..8r+7 write consecutive pages of row r.
        tiled_copy = copy_utils.tiled_copy_2d(
            cutlass.Float32, self.score_group_pages, cute.arch.WARP_SIZE
        )
        thr_copy = tiled_copy.get_slice(cute.arch.lane_idx())
        tSsS = thr_copy.partition_S(sScores)
        tSgS = thr_copy.partition_D(gScores)
        tScS = thr_copy.partition_S(cScores)
        for m in cutlass.range_constexpr(cute.size(tSgS, mode=[1])):
            for n in cutlass.range_constexpr(cute.size(tSgS, mode=[2])):
                row, col = tScS[(0, m, n)]
                # Pages inside the group and below the row's staged causal limit
                # (0 for rows past the task); branched, not predicated, so the warp
                # skips stores with no live lane.
                if col < group_len:
                    if group_page + col < sRowLimit[(row, score_warp)]:
                        tSgS[(None, m, n)].store(tSsS[(None, m, n)].load())
        cute.arch.sync_warp()

    @cute.jit
    def _reduce_accumulator_row(self, tTR_rAcc: cute.Tensor) -> cutlass.Float32:
        """Reduce one 128-token accumulator row with FP32 CUDA Core max."""
        block_max_0 = tTR_rAcc[0]
        block_max_1 = tTR_rAcc[1]
        block_max_2 = tTR_rAcc[2]
        block_max_3 = tTR_rAcc[3]
        for token_group in cutlass.range_constexpr(1, self.k_tile // 4):
            token_idx = token_group * 4
            block_max_0 = cute.arch.fmax(block_max_0, tTR_rAcc[token_idx])
            block_max_1 = cute.arch.fmax(block_max_1, tTR_rAcc[token_idx + 1])
            block_max_2 = cute.arch.fmax(block_max_2, tTR_rAcc[token_idx + 2])
            block_max_3 = cute.arch.fmax(block_max_3, tTR_rAcc[token_idx + 3])
        block_max_01 = cute.arch.fmax(block_max_0, block_max_1)
        block_max_23 = cute.arch.fmax(block_max_2, block_max_3)
        return cute.arch.fmax(block_max_01, block_max_23)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        cta_layout_vmnk: cute.Layout,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        mPageTable: cute.Tensor,
        mScores: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        num_heads: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
        mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
        is_leader_cta = mma_tile_coord_v == 0
        cluster_idx = bidx // cutlass.Int32(self.cta_group_size)

        if warp_idx == self.q_load_warp_id:
            cpasync.prefetch_descriptor(q_tma_atom)
        if warp_idx == self.k_load_warp_id:
            cpasync.prefetch_descriptor(k_tma_atom)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        if cutlass.const_expr(self.two_ctas_per_sm):
            sScoreBuf = storage.score_buf.get_tensor(
                cute.make_layout(
                    (
                        cute.arch.WARP_SIZE,
                        self.score_group_pages,
                        len(self.score_warp_ids),
                    ),
                    stride=(
                        self.score_buf_pitch,
                        1,
                        cute.arch.WARP_SIZE * self.score_buf_pitch,
                    ),
                )
            )
            sRowLimit = storage.score_row_limit.get_tensor(
                cute.make_layout((cute.arch.WARP_SIZE, len(self.score_warp_ids)))
            )
        sQ = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=q_smem_layout.outer,
            byte_alignment=128,
            swizzle=q_smem_layout.inner,
        )
        sK = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=k_smem_layout.outer,
            byte_alignment=128,
            swizzle=k_smem_layout.inner,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierIndexerSm100.TmemPtr),
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        thread_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_tma_bytes = (
            cute.size_in_bytes(
                self.io_dtype,
                cute.select(q_smem_layout, mode=[0, 1, 2]),
            )
            * self.cta_group_size
        )
        k_tma_bytes = (
            cute.size_in_bytes(
                self.io_dtype,
                cute.select(k_smem_layout, mode=[0, 1, 2]),
            )
            * self.cta_group_size
        )
        q_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stages,
            producer_group=thread_group,
            consumer_group=thread_group,
            tx_count=q_tma_bytes,
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        k_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stages,
            producer_group=thread_group,
            consumer_group=thread_group,
            tx_count=k_tma_bytes,
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=thread_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.cta_group_size * self.score_consumer_warps,
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        gK = cute.local_tile(
            k_tma_tensor,
            cute.select(self.mma_tiler_mnk, mode=[1, 2]),
            (None, 0, None),
        )
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgK = thr_mma.partition_B(gK)
        tKsK, tKgK = cpasync.tma_partition(
            k_tma_atom,
            cta_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tCgK, 0, 3),
        )
        q_mcast_mask = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk,
            cta_coord_vmnk,
            mcast_mode=2,
        )
        k_mcast_mask = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk,
            cta_coord_vmnk,
            mcast_mode=1,
        )
        tCrQ = tiled_mma.make_fragment_A(sQ)
        tCrK = tiled_mma.make_fragment_B(sK)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.acc_stages)
        )

        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
        tmem.allocate(self.tmem_alloc_cols)
        if cutlass.const_expr(self.two_ctas_per_sm):
            # Release the TMEM allocation permit right away so a second CTA can launch on this SM.
            tmem.relinquish_alloc_permit()
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_staged = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        # SM107 reads each 128-column accumulator row in two 64-column halves.
        tmem_chunks = 1
        tmem_repetition = tcgen05.Repetition.x128
        tmem_reduce_repetition = tcgen05.copy.Repetition.x128
        if cutlass.const_expr(self.two_ctas_per_sm):
            tmem_chunks = 2
            tmem_repetition = tcgen05.Repetition.x64
            tmem_reduce_repetition = tcgen05.copy.Repetition.x64
        tmem_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tmem_repetition),
            self.acc_dtype,
        )
        if cutlass.const_expr(self.use_tmem_load_reduce):
            tmem_load_reduce_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tmem_reduce_repetition,
                    redOp=tcgen05.TmemLoadRedOp.MAX,
                ),
                self.acc_dtype,
            )
        else:
            tmem_load_reduce_atom = tmem_atom
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_atom,
            tCtAcc_staged[(None, None, None, 0)],
        )
        # Score thread i reads accumulator row i of this CTA's 128-row half; with
        # two score warp groups (SM100/SM103) each group covers all 128 rows.
        score_tid = tidx % cutlass.Int32(self.score_threads)
        tile_worker_idx = score_tid // cutlass.Int32(self.q_tile // 2)
        tmem_tid = score_tid - tile_worker_idx * cutlass.Int32(self.q_tile // 2)
        tmem_thr_copy = tmem_tiled_copy.get_slice(tmem_tid)
        tTR_tAcc_staged = tmem_thr_copy.partition_S(tCtAcc_staged)
        tCcC = thr_mma.partition_C(
            cute.make_identity_tensor(self.mma_tiler_mnk[:2])
        )
        tTR_cC = tmem_thr_copy.partition_D(tCcC)
        tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)
        tTR_rBlockMax = cute.make_rmem_tensor(
            cute.make_layout((1, tmem_chunks), stride=(0, 1)),
            self.acc_dtype,
        )

        if warp_idx <= self.score_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_score)
            row_m = tTR_cC[0][0]
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.acc_stages,
            )
            if cutlass.const_expr(self.two_ctas_per_sm):
                # 32x32b TMEM loads: lane i of a warp owns accumulator row base + i.
                row_base = row_m - cute.arch.lane_idx()
                score_warp = warp_idx
                # Buckets in decreasing cost; cluster c takes every N-th task of each,
                # and the start rotates by each bucket's count so leftovers spread out.
                worker_base = cutlass.Int32(0)
                for rev_bucket in cutlass.range(self.num_task_buckets, unroll=1):
                    bucket = cutlass.Int32(self.num_task_buckets - 1 - rev_bucket)
                    task_count = mTaskCounts[bucket]
                    task_idx = cluster_idx - worker_base
                    if task_idx < cutlass.Int32(0):
                        task_idx += cutlass.Int32(self.num_persistent_clusters)
                    while task_idx < task_count:
                        (
                            _,
                            q_global_begin,
                            q_position_begin,
                            page_begin,
                            q_rows,
                            page_count,
                        ) = self._get_task_descriptor(
                            bucket,
                            task_idx,
                            mTaskDescriptors,
                        )
                        q_global = q_global_begin + row_m
                        q_valid = row_m < q_rows
                        local_block = (
                            q_position_begin + row_m // num_heads
                        ) >> cutlass.Int32(7)
                        row_limit = cutlass.Int32(0)
                        if q_valid:
                            row_limit = local_block
                        cute.arch.sync_warp()
                        sRowLimit[(cute.arch.lane_idx(), score_warp)] = row_limit
                        for tile_idx in cutlass.range(page_count, unroll=1):
                            acc_pipe.consumer_wait(acc_consumer_state)
                            tTR_tAcc = tTR_tAcc_staged[
                                (None, None, None, None, acc_consumer_state.index)
                            ]
                            if cutlass.const_expr(self.use_tmem_load_reduce):
                                for chunk in cutlass.range_constexpr(2):
                                    cute.copy_atom_call(
                                        tmem_load_reduce_atom,
                                        tTR_tAcc[(None, chunk, 0, 0)],
                                        (
                                            tTR_rAcc[(None, chunk, 0, 0)],
                                            tTR_rBlockMax[(None, chunk)],
                                        ),
                                    )
                            else:
                                cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipe.consumer_release(acc_consumer_state)
                            if cutlass.const_expr(self.use_tmem_load_reduce):
                                block_max = cute.arch.fmax(
                                    tTR_rBlockMax[(0, 0)], tTR_rBlockMax[(0, 1)]
                                )
                            else:
                                block_max = self._reduce_accumulator_row(tTR_rAcc)
                            slot = tile_idx % cutlass.Int32(self.score_group_pages)
                            sScoreBuf[(cute.arch.lane_idx(), slot, score_warp)] = block_max
                            if (
                                slot == cutlass.Int32(self.score_group_pages - 1)
                                or tile_idx == page_count - cutlass.Int32(1)
                            ):
                                self._flush_scores(
                                    sScoreBuf,
                                    sRowLimit,
                                    score_warp,
                                    q_global_begin + row_base,
                                    page_begin + tile_idx - slot,
                                    slot + cutlass.Int32(1),
                                    mScores,
                                )
                            acc_consumer_state.advance()
                        task_idx += cutlass.Int32(self.num_persistent_clusters)
                    worker_base = (
                        worker_base + task_count
                    ) % cutlass.Int32(self.num_persistent_clusters)
            else:
                if tile_worker_idx == cutlass.Int32(1):
                    acc_consumer_state.advance()
                tile_base = cutlass.Int32(0)
                worker_base = cutlass.Int32(0)
                for rev_bucket in cutlass.range(self.num_task_buckets, unroll=1):
                    bucket = cutlass.Int32(self.num_task_buckets - 1 - rev_bucket)
                    task_count = mTaskCounts[bucket]
                    task_idx = cluster_idx - worker_base
                    if task_idx < cutlass.Int32(0):
                        task_idx += cutlass.Int32(self.num_persistent_clusters)
                    while task_idx < task_count:
                        (
                            _,
                            q_global_begin,
                            q_position_begin,
                            page_begin,
                            q_rows,
                            page_count,
                        ) = self._get_task_descriptor(
                            bucket,
                            task_idx,
                            mTaskDescriptors,
                        )
                        q_global = q_global_begin + row_m
                        q_valid = row_m < q_rows
                        local_block = (
                            q_position_begin + row_m // num_heads
                        ) >> cutlass.Int32(7)
                        local_start = tile_worker_idx ^ (
                            tile_base & cutlass.Int32(1)
                        )
                        worker_tiles = (
                            page_count
                            + cutlass.Int32(1)
                            - local_start
                        ) // cutlass.Int32(self.score_worker_groups)
                        for worker_tile_idx in cutlass.range(
                            worker_tiles,
                            unroll=1,
                        ):
                            tile_idx = (
                                local_start
                                + worker_tile_idx
                                * cutlass.Int32(self.score_worker_groups)
                            )
                            acc_pipe.consumer_wait(acc_consumer_state)
                            tTR_tAcc = tTR_tAcc_staged[
                                (None, None, None, None, acc_consumer_state.index)
                            ]
                            if cutlass.const_expr(self.use_tmem_load_reduce):
                                for chunk in cutlass.range_constexpr(1):
                                    cute.copy_atom_call(
                                        tmem_load_reduce_atom,
                                        tTR_tAcc[(None, chunk, 0, 0)],
                                        (
                                            tTR_rAcc[(None, chunk, 0, 0)],
                                            tTR_rBlockMax[(None, chunk)],
                                        ),
                                    )
                            else:
                                cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipe.consumer_release(acc_consumer_state)
                            logical_page = page_begin + tile_idx
                            if q_valid and logical_page < local_block:
                                if cutlass.const_expr(self.use_tmem_load_reduce):
                                    block_max = tTR_rBlockMax[(0, 0)]
                                else:
                                    block_max = self._reduce_accumulator_row(tTR_rAcc)
                                mScores[q_global, logical_page] = block_max
                            acc_consumer_state.advance()
                            acc_consumer_state.advance()
                        tile_base += page_count
                        task_idx += cutlass.Int32(self.num_persistent_clusters)
                    worker_base = (
                        worker_base + task_count
                    ) % cutlass.Int32(self.num_persistent_clusters)
        elif warp_idx == self.q_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.q_stages,
            )
            # Buckets in decreasing cost; cluster c takes every N-th task of each,
            # and the start rotates by each bucket's count so leftovers spread out.
            worker_base = cutlass.Int32(0)
            for rev_bucket in cutlass.range(self.num_task_buckets, unroll=1):
                bucket = cutlass.Int32(self.num_task_buckets - 1 - rev_bucket)
                task_count = mTaskCounts[bucket]
                task_idx = cluster_idx - worker_base
                if task_idx < cutlass.Int32(0):
                    task_idx += cutlass.Int32(self.num_persistent_clusters)
                while task_idx < task_count:
                    (
                        _,
                        q_global_begin,
                        _,
                        _,
                        _,
                        page_count,
                    ) = self._get_task_descriptor(
                        bucket,
                        task_idx,
                        mTaskDescriptors,
                    )
                    if page_count > cutlass.Int32(0):
                        mQ_task = cute.domain_offset(
                            (q_global_begin, 0),
                            q_tma_tensor,
                        )
                        gQ = cute.local_tile(
                            mQ_task,
                            cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
                            (None, None),
                        )
                        tCgQ = thr_mma.partition_A(gQ)
                        tQsQ, tQgQ = cpasync.tma_partition(
                            q_tma_atom,
                            cta_coord_vmnk[2],
                            cute.make_layout(
                                cute.size(cta_layout_vmnk, mode=[2])
                            ),
                            cute.group_modes(sQ, 0, 3),
                            cute.group_modes(tCgQ, 0, 3),
                        )
                        q_pipe.producer_acquire(q_producer_state)
                        cute.copy(
                            q_tma_atom,
                            tQgQ[(None, 0, 0)],
                            tQsQ[(None, q_producer_state.index)],
                            tma_bar_ptr=q_pipe.producer_get_barrier(
                                q_producer_state
                            ),
                            mcast_mask=q_mcast_mask,
                        )
                        q_producer_state.advance()
                    task_idx += cutlass.Int32(self.num_persistent_clusters)
                worker_base = (
                    worker_base + task_count
                ) % cutlass.Int32(self.num_persistent_clusters)
            q_pipe.producer_tail(q_producer_state)
        elif warp_idx == self.k_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.k_stages,
            )
            # Buckets in decreasing cost; cluster c takes every N-th task of each,
            # and the start rotates by each bucket's count so leftovers spread out.
            worker_base = cutlass.Int32(0)
            for rev_bucket in cutlass.range(self.num_task_buckets, unroll=1):
                bucket = cutlass.Int32(self.num_task_buckets - 1 - rev_bucket)
                task_count = mTaskCounts[bucket]
                task_idx = cluster_idx - worker_base
                if task_idx < cutlass.Int32(0):
                    task_idx += cutlass.Int32(self.num_persistent_clusters)
                while task_idx < task_count:
                    (
                        batch_idx,
                        _,
                        _,
                        page_begin,
                        _,
                        page_count,
                    ) = self._get_task_descriptor(
                        bucket,
                        task_idx,
                        mTaskDescriptors,
                    )
                    if cutlass.const_expr(self.compute_capability == (10, 7)):
                        # The physical page index is a dependent global load on
                        # the TMA issue path; fetch the next page's index while
                        # this page's TMA is issued so the two latencies overlap.
                        # The last iteration re-reads the final entry (in bounds).
                        last_page = page_begin + page_count - cutlass.Int32(1)
                        physical_page = mPageTable[batch_idx, page_begin]
                        for tile_idx in cutlass.range(page_count, unroll=1):
                            logical_page = page_begin + tile_idx
                            next_page = cutlass.min(
                                logical_page + cutlass.Int32(1), last_page
                            )
                            next_physical_page = mPageTable[batch_idx, next_page]
                            k_pipe.producer_acquire(k_producer_state)
                            cute.copy(
                                k_tma_atom,
                                tKgK[(None, 0, physical_page)],
                                tKsK[(None, k_producer_state.index)],
                                tma_bar_ptr=k_pipe.producer_get_barrier(
                                    k_producer_state
                                ),
                                mcast_mask=k_mcast_mask,
                            )
                            k_producer_state.advance()
                            physical_page = next_physical_page
                    else:
                        physical_page = cutlass.Int32(0)
                        for tile_idx in cutlass.range(page_count, unroll=1):
                            logical_page = page_begin + tile_idx
                            physical_page = mPageTable[batch_idx, logical_page]
                            k_pipe.producer_acquire(k_producer_state)
                            cute.copy(
                                k_tma_atom,
                                tKgK[(None, 0, physical_page)],
                                tKsK[(None, k_producer_state.index)],
                                tma_bar_ptr=k_pipe.producer_get_barrier(
                                    k_producer_state
                                ),
                                mcast_mask=k_mcast_mask,
                            )
                            k_producer_state.advance()
                    task_idx += cutlass.Int32(self.num_persistent_clusters)
                worker_base = (
                    worker_base + task_count
                ) % cutlass.Int32(self.num_persistent_clusters)
            k_pipe.producer_tail(k_producer_state)
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.q_stages,
            )
            k_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.k_stages,
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.acc_stages,
            )
            num_k_blocks = cute.size(tCrQ, mode=[2])
            if is_leader_cta:
                # Buckets in decreasing cost; cluster c takes every N-th task of each,
                # and the start rotates by each bucket's count so leftovers spread out.
                worker_base = cutlass.Int32(0)
                for rev_bucket in cutlass.range(self.num_task_buckets, unroll=1):
                    bucket = cutlass.Int32(self.num_task_buckets - 1 - rev_bucket)
                    task_count = mTaskCounts[bucket]
                    task_idx = cluster_idx - worker_base
                    if task_idx < cutlass.Int32(0):
                        task_idx += cutlass.Int32(self.num_persistent_clusters)
                    while task_idx < task_count:
                        (
                            _,
                            _,
                            _,
                            _,
                            _,
                            page_count,
                        ) = self._get_task_descriptor(
                            bucket,
                            task_idx,
                            mTaskDescriptors,
                        )
                        q_pipe.consumer_wait(q_consumer_state)
                        for _ in cutlass.range(page_count, unroll=1):
                            k_pipe.consumer_wait(k_consumer_state)
                            acc_pipe.producer_acquire(acc_producer_state)
                            tCtAcc = tCtAcc_staged[
                                (
                                    None,
                                    None,
                                    None,
                                    acc_producer_state.index,
                                )
                            ]
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for k_block_idx in cutlass.range_constexpr(
                                num_k_blocks
                            ):
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrQ[
                                        (
                                            None,
                                            None,
                                            k_block_idx,
                                            q_consumer_state.index,
                                        )
                                    ],
                                    tCrK[
                                        (
                                            None,
                                            None,
                                            k_block_idx,
                                            k_consumer_state.index,
                                        )
                                    ],
                                    tCtAcc,
                                )
                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    True,
                                )
                            k_pipe.consumer_release(k_consumer_state)
                            k_consumer_state.advance()
                            acc_pipe.producer_commit(acc_producer_state)
                            acc_producer_state.advance()
                        q_pipe.consumer_release(q_consumer_state)
                        q_consumer_state.advance()
                        task_idx += cutlass.Int32(self.num_persistent_clusters)
                    worker_base = (
                        worker_base + task_count
                    ) % cutlass.Int32(self.num_persistent_clusters)
                acc_pipe.producer_tail(acc_producer_state)
        else:
            # The last warp (11 of 12; 7 of 8 on SM107) completes its warpgroup.
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        if cutlass.const_expr(not self.two_ctas_per_sm):
            tmem.relinquish_alloc_permit()
        pipeline.sync(barrier_id=int(NamedBarrierIndexerSm100.Final))
        if cutlass.const_expr(self.two_ctas_per_sm):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        tmem.free(tmem_ptr)


__all__ = ["Q8KV8PrefillIndexerSm100"]
