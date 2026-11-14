# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CuTe DSL Q8KV8 decode indexer for SM100/SM103.

Ported from MiniMax-AI/MSA nv_dev at 9ae7751cf6b28f1c551a284de90dddf9971d621e,
inference/msa_v1/indexer/decode/indexer_gemm.py. Preserves the vLLM MSA
eight-token, token-major interface and padded KV-page strides. Worker ranges
are derived from dev's existing page-prefix workspace, without a new planner API.
"""

from __future__ import annotations

import enum

import cutlass
import cutlass.cute as cute  # noqa: PLR0402
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Float32, Int32, Int64

# isort: split
import cuda.bindings.driver as cuda


class _NamedBarrier(enum.IntEnum):
    TmemPtr = enum.auto()
    Score = enum.auto()
    Final = enum.auto()


class Q8KV8DecodeIndexerSm100:
    """Compute per-head page scores with balanced persistent workers."""

    supported_num_heads = (1, 2, 4)
    query_length = 8
    page_size = 128
    head_dim = 128
    k_chunk = head_dim
    chunks_per_page = head_dim // k_chunk
    m_tile = page_size
    k_stages = 6
    q_stages = chunks_per_page
    acc_stages = 4
    threads_per_warp = 32
    score_warp_begin = 3
    score_warps = 4
    score_threads = score_warps * threads_per_warp
    threads_per_cta = (score_warp_begin + score_warps) * threads_per_warp
    tmem_copy_threads = 128
    k_cache_evict_first = 0x12F0000000000000

    def __init__(self, *, sm_count: int, num_heads: int) -> None:
        if sm_count <= 0:
            raise ValueError("sm_count must be positive")
        if num_heads not in self.supported_num_heads:
            raise ValueError(f"num_heads must be one of {self.supported_num_heads}")
        self.num_index_heads = num_heads
        self.input_dtype = cutlass.Float8E4M3FN
        self.score_scale = 1.0
        self.query_columns = self.query_length * num_heads
        self.n_tile = self.query_columns
        self.tmem_columns = max(
            32, 1 << (self.query_columns * self.acc_stages - 1).bit_length()
        )
        # One persistent worker per SM, partitioning dev's page-prefix schedule.
        self.grid_ctas = sm_count

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mPageTable: cute.Tensor,
        mSeqLens: cute.Tensor,
        mOut: cute.Tensor,
        mSchedulerStorage: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if cutlass.const_expr(
            mQ.element_type != self.input_dtype
            or mKCache.element_type != self.input_dtype
        ):
            raise TypeError("q and k_cache must be Float8E4M3FN")
        if cutlass.const_expr(
            mPageTable.element_type is not Int32 or mSeqLens.element_type is not Int32
        ):
            raise TypeError("page_table and seq_lens must be Int32")
        if cutlass.const_expr(mOut.element_type is not Float32):
            raise TypeError("out must be Float32")

        # Preserve the caller's physical-page stride (pages may be padded).
        mK_tdp = cute.make_tensor(
            mKCache.iterator,
            cute.select(mKCache.layout, mode=[1, 2, 0]),
        )
        mQ_nkb = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (self.query_columns, self.head_dim, mSeqLens.shape[0]),
                stride=(self.head_dim, 1, self.query_columns * self.head_dim),
            ),
        )
        mQ_kcr = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (
                    self.k_chunk,
                    self.query_columns,
                    mSeqLens.shape[0],
                ),
                stride=(1, self.head_dim, self.query_columns * self.head_dim),
            ),
        )
        mPageTable_lb = cute.make_tensor(
            mPageTable.iterator,
            cute.make_layout(
                (mPageTable.shape[1], mSeqLens.shape[0]),
                stride=(1, mPageTable.shape[1]),
            ),
        )
        # Upstream writes [head, token, page]; address our [batch, Q*H, page]
        # allocation through a strided view, without allocating or copying.
        mOut_pqb = cute.make_tensor(
            mOut.iterator,
            cute.make_layout(
                (self.num_index_heads, mSeqLens.shape[0] * self.query_length, mOut.shape[2]),
                stride=(mOut.shape[2], self.num_index_heads * mOut.shape[2], 1),
            ),
        )
        mScheduler = cute.make_tensor(
            cute.recast_ptr(mSchedulerStorage.iterator, dtype=Int32),
            cute.make_layout(mSeqLens.shape[0] + 1),
        )

        qk_tiler = (self.m_tile, self.n_tile, self.k_chunk)
        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            mKCache.element_type,
            mQ.element_type,
            utils.LayoutEnum.from_tensor(mK_tdp).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mQ_nkb).mma_major_mode(),
            Float32,
            cta_group,
            qk_tiler[:2],
            tcgen05.OperandSource.SMEM,
        )
        sK_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            qk_tiler,
            mKCache.element_type,
            self.k_stages,
        )
        sQ_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            qk_tiler,
            mQ.element_type,
            self.q_stages,
        )
        # MMA stores consecutive 128-byte K sectors before advancing to the
        # next sector. BF16 therefore has two sectors per 128-element row.
        sector_elements = 128 * 8 // self.input_dtype.width
        k_sectors = self.k_chunk // sector_elements
        sK_tma_layout = cute.make_composed_layout(
            sK_layout.inner,
            0,
            cute.make_layout(
                (self.page_size, (sector_elements, k_sectors), self.k_stages),
                stride=(
                    sector_elements,
                    (1, self.page_size * sector_elements),
                    self.page_size * self.k_chunk,
                ),
            ),
        )
        sQ_tma_layout = cute.make_composed_layout(
            sQ_layout.inner,
            0,
            cute.make_layout(
                (
                    (sector_elements, k_sectors),
                    self.query_columns,
                    self.chunks_per_page,
                ),
                stride=(
                    (1, self.query_columns * sector_elements),
                    sector_elements,
                    self.query_columns * self.k_chunk,
                ),
            ),
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mK_tdp,
            sK_tma_layout,
            (self.page_size, self.k_chunk),
        )
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mQ_kcr,
            sQ_tma_layout,
            (
                self.k_chunk,
                self.query_columns,
                self.chunks_per_page,
            ),
        )

        @cute.struct
        class SharedStorage:
            k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stages * 2]
            q_mbar_ptr: cute.struct.MemRange[Int64, 2]
            acc_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stages * 2]
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_Q,
            tma_tensor_Q,
            mPageTable_lb,
            mSeqLens,
            mOut_pqb,
            mScheduler,
            Int32(self.query_columns // self.num_index_heads),
            sK_layout,
            sK_tma_layout,
            sQ_layout,
            sQ_tma_layout,
        ).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        mK_tdp: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mQ_kcr: cute.Tensor,
        mPageTable_lb: cute.Tensor,
        mSeqLens: cute.Tensor,
        mOut_pqb: cute.Tensor,
        mScheduler: cute.Tensor,
        query_length: Int32,
        sK_layout: cute.ComposedLayout,
        sK_tma_layout: cute.ComposedLayout,
        sQ_layout: cute.ComposedLayout,
        sQ_tma_layout: cute.ComposedLayout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        worker_idx, _, _ = cute.arch.block_idx()

        # dev supplies only the cumulative historical-page counts. Reconstruct
        # nv_dev's quotient/remainder worker split instead of requiring its
        # separately materialized per-worker plan.
        batch_size = Int32(mSeqLens.shape[0])
        total_pages = Int32(0)
        if lane_idx == Int32(0):
            total_pages = mScheduler[batch_size]
        total_pages = cute.arch.shuffle_sync(total_pages, 0)
        quotient = total_pages // Int32(self.grid_ctas)
        remainder = total_pages % Int32(self.grid_ctas)
        extra = remainder
        if worker_idx < remainder:
            extra = worker_idx
        global_page_begin = worker_idx * quotient + extra
        num_pages = quotient
        if worker_idx < remainder:
            num_pages += Int32(1)

        # Find the request containing this worker's first page, skipping empty
        # requests. Idle workers may resolve to batch_size and execute no loads.
        batch_idx = Int32(0)
        if lane_idx == Int32(0):
            upper = batch_size
            while batch_idx < upper:
                middle = (batch_idx + upper) // Int32(2)
                if mScheduler[middle + 1] <= global_page_begin:
                    batch_idx = middle + Int32(1)
                else:
                    upper = middle
        batch_idx = cute.arch.shuffle_sync(batch_idx, 0)
        logical_page_begin = global_page_begin - mScheduler[batch_idx]

        if warp_idx == Int32(1):
            cpasync.prefetch_descriptor(tma_atom_K)
        elif warp_idx == Int32(2):
            cpasync.prefetch_descriptor(tma_atom_Q)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK = smem.allocate_tensor(
            element_type=self.input_dtype,
            layout=sK_layout.outer,
            swizzle=sK_layout.inner,
            byte_alignment=128,
        )
        sK_tma = cute.make_tensor(sK.iterator, sK_tma_layout.outer)
        sQ = smem.allocate_tensor(
            element_type=self.input_dtype,
            layout=sQ_layout.outer,
            swizzle=sQ_layout.inner,
            byte_alignment=128,
        )
        sQ_tma = cute.make_tensor(sQ.iterator, sQ_tma_layout.outer)
        sPartialMax = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (2, self.score_warps, self.query_columns),
                stride=(self.score_warps * self.query_columns, self.query_columns, 1),
            ),
            byte_alignment=16,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.TmemPtr),
            num_threads=self.threads_per_cta,
        )
        score_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.Score),
            num_threads=self.score_threads,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        # Every accumulator stage owns query_columns TMEM columns.
        tmem.allocate(self.tmem_columns)

        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.page_size * self.k_chunk * self.input_dtype.width // 8,
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.query_columns * self.head_dim * self.input_dtype.width // 8,
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.tmem_copy_threads,
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        pipeline.pipeline_init_arrive(is_relaxed=True)

        gK = cute.flat_divide(mK_tdp, (self.page_size, self.k_chunk))
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK_tma, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        gQ = cute.flat_divide(
            mQ_kcr,
            (
                self.k_chunk,
                self.query_columns,
                self.chunks_per_page,
            ),
        )
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_tma, 0, 3),
            cute.group_modes(gQ, 0, 3),
        )

        tCrK = tiled_mma.make_fragment_A(sK)
        tCrQ = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C((self.m_tile, self.n_tile))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        pipeline.pipeline_init_wait()
        thr_mma = tiled_mma.get_slice(0)
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tCtAcc_staged = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x8),
            Float32,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_load_atom,
            tCtAcc_staged[(None, None, None, 0)],
        )
        copy_tidx = tidx % Int32(self.tmem_copy_threads)
        thr_tmem_copy = tmem_tiled_copy.get_slice(copy_tidx)
        tTR_tAcc_staged = thr_tmem_copy.partition_S(tCtAcc_staged)
        cScores = cute.make_identity_tensor((self.m_tile, self.n_tile))
        tCcScores = thr_mma.partition_C(cScores)
        tTR_cScores = thr_tmem_copy.partition_D(tCcScores)

        if warp_idx == Int32(1):
            current_batch = batch_idx
            logical_page = logical_page_begin
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                for _ in cutlass.range(segment_pages, unroll=1):
                    physical_page = Int32(0)
                    if lane_idx == Int32(0):
                        physical_page = mPageTable_lb[
                            logical_page,
                            current_batch,
                        ]
                    physical_page = cute.arch.shuffle_sync(physical_page, 0)
                    for chunk in cutlass.range_constexpr(self.chunks_per_page):
                        k_empty = k_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_K,
                            tKgK[(None, 0, Int32(chunk), physical_page)],
                            tKsK[(None, k_empty.index)],
                            tma_bar_ptr=k_empty.barrier,
                            cache_policy=Int64(self.k_cache_evict_first),
                        )
                    global_page += Int32(1)
                    logical_page += Int32(1)
                pages_remaining -= segment_pages
                while (
                    current_batch < Int32(mSeqLens.shape[0] - 1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
                    logical_page = global_page - mScheduler[current_batch]
            k_producer.tail()
        elif warp_idx == Int32(2):
            current_batch = batch_idx
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                q_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_Q,
                    tQgQ[(None, Int32(0), Int32(0), current_batch)],
                    tQsQ,
                    tma_bar_ptr=q_empty.barrier,
                )
                global_page += segment_pages
                pages_remaining -= segment_pages
                while (
                    current_batch < Int32(mSeqLens.shape[0] - 1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
            q_producer.tail()
        elif warp_idx == Int32(0):
            num_k_blocks = cute.size(tCrK, mode=[2])
            current_batch = batch_idx
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                q_full = q_consumer.wait_and_advance()
                for _ in cutlass.range(segment_pages, unroll=1):
                    acc_empty = acc_producer.acquire_and_advance()
                    tCtAcc = tCtAcc_staged[(None, None, None, acc_empty.index)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for chunk in cutlass.range_constexpr(self.chunks_per_page):
                        k_full = k_consumer.wait_and_advance()
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrK[
                                    (
                                        None,
                                        None,
                                        k_block_idx,
                                        k_full.index,
                                    )
                                ],
                                tCrQ[
                                    (
                                        None,
                                        None,
                                        k_block_idx,
                                        Int32(chunk),
                                    )
                                ],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        k_full.release()
                    acc_empty.commit()
                q_full.release()
                global_page += segment_pages
                pages_remaining -= segment_pages
                while (
                    current_batch < Int32(mSeqLens.shape[0] - 1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
            acc_producer.tail()

        rScores = cute.make_rmem_tensor(tTR_cScores.shape, Float32)
        if warp_idx >= Int32(self.score_warp_begin):
            score_warp_idx = warp_idx - Int32(self.score_warp_begin)
            rScoresFlat = cute.make_tensor(
                rScores.iterator,
                cute.make_layout(self.query_columns),
            )
            current_batch = batch_idx
            logical_page = logical_page_begin
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                for _ in cutlass.range(segment_pages, unroll=1):
                    partial_stage = global_page % Int32(2)
                    acc_full = acc_consumer.wait_and_advance()
                    cute.copy(
                        tmem_tiled_copy,
                        tTR_tAcc_staged[(None, None, None, None, acc_full.index)],
                        rScores,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    for query_idx in cutlass.range_constexpr(self.query_columns):
                        rScoresFlat[query_idx] = cute.arch.warp_redux_sync(
                            rScoresFlat[query_idx],
                            "fmax",
                        )
                    if lane_idx == Int32(0):
                        cute.autovec_copy(
                            rScoresFlat,
                            sPartialMax[partial_stage, score_warp_idx, None],
                        )
                    acc_full.release()
                    score_barrier.arrive_and_wait()

                    if score_warp_idx == Int32(0):
                        for column_group in cutlass.range_constexpr(
                            (self.query_columns + 31) // 32
                        ):
                            query_idx = lane_idx + column_group * 32
                            if query_idx < query_length * self.num_index_heads:
                                row_max = -Float32.inf
                                for partial_idx in cutlass.range_constexpr(
                                    self.score_warps
                                ):
                                    row_max = cute.arch.fmax(
                                        row_max,
                                        sPartialMax[
                                            partial_stage, partial_idx, query_idx
                                        ],
                                    )
                                local_block = (
                                    mSeqLens[current_batch]
                                    - query_length
                                    + query_idx // self.num_index_heads
                                ) // self.page_size
                                if logical_page < local_block:
                                    mOut_pqb[
                                        query_idx % self.num_index_heads,
                                        current_batch * query_length
                                        + query_idx // self.num_index_heads,
                                        logical_page,
                                    ] = row_max * self.score_scale
                    global_page += Int32(1)
                    logical_page += Int32(1)
                pages_remaining -= segment_pages
                while (
                    current_batch < Int32(mSeqLens.shape[0] - 1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
                    logical_page = global_page - mScheduler[current_batch]

        tmem.relinquish_alloc_permit()
        pipeline.sync(barrier_id=int(_NamedBarrier.Final))
        tmem.free(tmem_ptr)


__all__ = ["Q8KV8DecodeIndexerSm100"]
