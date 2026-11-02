# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from the CUTLASS CuTe DSL example dense_gemm_persistent.py (Blackwell), under its
# BSD-3-Clause license below: the static persistent schedule, cut to the configuration used here.
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Batched dense GEMM for the spatial Winograd products on Blackwell (sm100).

``PersistentDenseGemmKernel`` computes ``C[M, N, L] = A[M, K, L] . B[N, K, L]^T`` for ``L``
independent problems (one per Winograd transform point): TMA loads into shared memory,
``tcgen05.mma`` into tensor memory, and warp-specialized load / MMA / epilogue roles on a
static persistent tile schedule.
"""

from typing import NamedTuple, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


def _identity(x):
    return x


def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: type[cutlass.Numeric],
    b_dtype: type[cutlass.Numeric],
    c_dtype: type[cutlass.Numeric],
    smem_capacity: int,
    c_smem_layout: cute.ComposedLayout,
) -> Tuple[int, int, int]:
    """Stage counts ``(accumulator, A/B, C)``: two accumulator stages in tensor memory, as many
    A/B stages as fit in shared memory next to two C stages, the leftover going to more C
    stages."""
    num_acc_stage = 2
    num_c_stage = 2

    a_smem_layout_stage_one = utils.sm100.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
    b_smem_layout_staged_one = utils.sm100.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
    ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
        b_dtype, b_smem_layout_staged_one
    )
    mbar_helpers_bytes = 1024
    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    num_ab_stage = (smem_capacity - (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage
    num_c_stage += (
        smem_capacity - ab_bytes_per_stage * num_ab_stage - (mbar_helpers_bytes + c_bytes)
    ) // c_bytes_per_stage
    return num_acc_stage, num_ab_stage, num_c_stage


class _Operands(NamedTuple):
    """The kernel's per-CTA views of A, B and C, shared by the warp roles."""

    a_mcast_mask: object
    b_mcast_mask: object
    tAsA: cute.Tensor  # TMA destination of A: ((atom_v, rest_v), STAGE)
    tAgA: cute.Tensor  # TMA source of A: ((atom_v, rest_v), RestM, RestK, RestL)
    tBsB: cute.Tensor
    tBgB: cute.Tensor
    tCrA: cute.Tensor  # MMA fragment of A: (MMA, MMA_M, MMA_K, STAGE)
    tCrB: cute.Tensor
    tCgC: cute.Tensor  # MMA partition of C: (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
    tCtAcc_fake: cute.Tensor  # accumulator layout: (MMA, MMA_M, MMA_N, STAGE)
    k_tile_cnt: object


class PersistentDenseGemmKernel:
    """Batched GEMM with fp16 ``A`` / ``B``, fp32 accumulation in tensor memory and fp16 or
    bf16 ``C``. Each MMA tile is ``mma_tiler_mn = (256, N)`` (N a multiple of 32 up to 256)
    on a CTA pair (2-CTA ``tcgen05.mma``), in clusters of ``cluster_shape_mn`` CTAs (M a
    multiple of 2). The Tensor Memory Accelerator (TMA) loads A / B into shared memory and
    stores C from it. One warp issues the loads, one the MMAs, four run the epilogue, on a
    static persistent tile schedule."""

    def __init__(self, mma_tiler_mn: Tuple[int, int], cluster_shape_mn: Tuple[int, int]):
        assert mma_tiler_mn[0] == 256, "2-CTA MMA tiles are 256 rows"
        assert mma_tiler_mn[1] % 32 == 0 and 32 <= mma_tiler_mn[1] <= 256
        assert cluster_shape_mn[0] % 2 == 0, "a CTA pair spans two cluster rows"

        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = True
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cluster_shape_mn = cluster_shape_mn
        # K is set in _setup_attributes from the MMA instruction shape
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilogue_warp_id)
        )
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2

    def _setup_attributes(self) -> cute.TiledMma:
        """Tile shapes, cluster layout, epilogue subtile, stage counts, shared memory layouts
        and tensor memory columns for the operand dtypes and layouts; returns the tiled MMA."""
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        c_smem_layout = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            utils.get_smem_capacity_in_bytes(),
            c_smem_layout,
        )

        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, arch="sm_100")
        return tiled_mma

    def _make_tma_atoms(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, tiled_mma):
        """TMA load atoms of A and B (multicast across the cluster) and the TMA store atom of
        C, with their TMA views of the tensors; sets the bytes one A/B stage loads."""
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * cute.size(tiled_mma.thr_id.shape)

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )
        return tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_c, tma_tensor_c

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Compute ``C[M, N, L] = A[M, K, L] . B[N, K, L]^T`` for the ``L`` batched problems,
        launched on ``stream`` and returning before it completes. ``A`` and ``B`` are fp16,
        ``C`` fp16 or bf16; each tensor's contiguous dimension is 16-byte aligned.
        ``max_active_clusters`` bounds the persistent grid.

        :raises TypeError: If the dtypes are not the ones above.
        """
        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.c_dtype: type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(
            self.a_dtype is not cutlass.Float16 or self.b_dtype is not cutlass.Float16
        ):
            raise TypeError(f"A and B must be fp16, got {self.a_dtype} and {self.b_dtype}")
        if cutlass.const_expr(self.c_dtype not in (cutlass.Float16, cutlass.BFloat16)):
            raise TypeError(f"C must be fp16 or bf16, got {self.c_dtype}")

        tiled_mma = self._setup_attributes()
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_c, tma_tensor_c = (
            self._make_tma_atoms(a, b, c, tiled_mma)
        )
        tile_sched_params, grid = self._compute_grid(c, max_active_clusters)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage())
        ab_producer, ab_consumer, acc_pipeline, tmem = self._make_pipelines(
            storage, cluster_layout_vmnk
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        ops = self._partition_operands(
            smem,
            tiled_mma,
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            mC_mnl,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            mma_tile_coord_v,
            block_in_cluster_coord_vmnk,
        )

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        # The roles share one program order: each returns the scheduler, the work tile and the
        # objects it advanced, and its branch hands them on to the next role.
        if warp_idx == self.tma_warp_id:
            work_tile, ab_producer, tile_sched = self._tma_load_warp(
                tma_atom_a, tma_atom_b, ops, ab_producer, tiled_mma, tile_sched, work_tile
            )
        if warp_idx == self.mma_warp_id:
            work_tile, tmem, ab_consumer, tiled_mma, tile_sched = self._mma_warp(
                tiled_mma,
                ops,
                ab_consumer,
                acc_pipeline,
                tmem,
                is_leader_cta,
                tile_sched,
                work_tile,
            )

        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        if warp_idx < self.mma_warp_id:
            work_tile, tmem, tile_sched = self._epilogue_warps(
                tidx,
                warp_idx,
                tiled_mma,
                tma_atom_c,
                sC,
                epi_tile,
                ops,
                acc_pipeline,
                tmem,
                tile_sched,
                work_tile,
            )

    def _shared_storage(self):
        """Shared memory barriers and the tensor memory address handed to the other warps."""

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        return SharedStorage

    def _make_pipelines(self, storage, cluster_layout_vmnk: cute.Layout):
        """The A/B pipeline (TMA warp -> MMA warp) as producer / consumer participants, the
        accumulator pipeline (MMA warp -> epilogue warps) and the tensor memory allocator."""
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()

        # Both CTAs of the pair release the accumulator.
        num_acc_consumer_threads = len(self.epilogue_warp_id) * 2
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_acc_consumer_threads
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        return ab_producer, ab_consumer, acc_pipeline, tmem

    def _partition_operands(
        self,
        smem,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        mma_tile_coord_v,
        block_in_cluster_coord_vmnk,
    ) -> _Operands:
        """Shared memory stages of A and B, their TMA and MMA partitions, the MMA partition of
        C and the multicast masks of the A / B loads."""
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        return _Operands(
            a_full_mcast_mask,
            b_full_mcast_mask,
            tAsA,
            tAgA,
            tBsB,
            tBgB,
            tCrA,
            tCrB,
            tCgC,
            tCtAcc_fake,
            k_tile_cnt,
        )

    @staticmethod
    def _mma_tile_coord(tiled_mma: cute.TiledMma, work_tile):
        """The (M, N, L) MMA tile of a scheduler work tile: a CTA pair shares one M tile."""
        cur_tile_coord = work_tile.tile_idx
        return (
            cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cur_tile_coord[1],
            cur_tile_coord[2],
        )

    @cute.jit
    def _tma_load_warp(
        self,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        ops: _Operands,
        ab_producer,
        tiled_mma: cute.TiledMma,
        tile_sched,
        work_tile,
    ):
        """Load the A / B k-tiles of every scheduled tile into the A/B pipeline stages."""
        tAgA = ops.tAgA
        tAsA = ops.tAsA
        tBgB = ops.tBgB
        tBsB = ops.tBsB
        a_full_mcast_mask = ops.a_mcast_mask
        b_full_mcast_mask = ops.b_mcast_mask
        k_tile_cnt = ops.k_tile_cnt
        while work_tile.is_valid_tile:
            mma_tile_coord_mnl = self._mma_tile_coord(tiled_mma, work_tile)
            # ((atom_v, rest_v), RestK)
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            # ((atom_v, rest_v), RestK)
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

            # Try-acquire the first A/B stage; the loop waits on it only if it is still busy.
            ab_producer.reset()
            peek_ab_empty_status = ab_producer.try_acquire()

            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, handle.count)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, handle.count)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=b_full_mcast_mask,
                )

                # Try-acquire the next A/B stage
                peek_ab_empty_status = cutlass.Boolean(1)
                if handle.count + 1 < k_tile_cnt:
                    peek_ab_empty_status = ab_producer.try_acquire()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Wait until the MMA warp has released every A/B stage
        ab_producer.tail()
        return work_tile, ab_producer, tile_sched

    @cute.jit
    def _mma_warp(
        self,
        tiled_mma: cute.TiledMma,
        ops: _Operands,
        ab_consumer,
        acc_pipeline,
        tmem,
        is_leader_cta,
        tile_sched,
        work_tile,
    ):
        """Accumulate every scheduled tile into a tensor memory accumulator stage; the leader
        CTA of the pair issues the 2-CTA MMAs."""
        tCrA = ops.tCrA
        tCrB = ops.tCrB
        k_tile_cnt = ops.k_tile_cnt
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, ops.tCtAcc_fake.layout)

        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )

        while work_tile.is_valid_tile:
            # (MMA, MMA_M, MMA_N)
            tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

            # Try-wait the first A/B stage
            ab_consumer.reset()
            peek_ab_full_status = cutlass.Boolean(1)
            if is_leader_cta:
                peek_ab_full_status = ab_consumer.try_wait()

            if is_leader_cta:
                acc_pipeline.producer_acquire(acc_producer_state)

            for k_tile in range(k_tile_cnt):
                if is_leader_cta:
                    handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                    # tCtAcc += tCrA * tCrB
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    tile_crd = (None, None, None, handle.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

                    handle.release()

                    # Try-wait the next A/B stage
                    peek_ab_full_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_full_status = ab_consumer.try_wait()

            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Wait until the epilogue has drained every accumulator stage
        acc_pipeline.producer_tail(acc_producer_state)
        return work_tile, tmem, ab_consumer, tiled_mma, tile_sched

    @cute.jit
    def _epilogue_warps(
        self,
        tidx,
        warp_idx,
        tiled_mma: cute.TiledMma,
        tma_atom_c: cute.CopyAtom,
        sC: cute.Tensor,
        epi_tile: cute.Tile,
        ops: _Operands,
        acc_pipeline,
        tmem,
        tile_sched,
        work_tile,
    ):
        """Allocate the tensor memory, then convert every accumulator stage to C and TMA-store
        it through shared memory; free the tensor memory at the end."""
        tCgC = ops.tCgC
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, ops.tCtAcc_fake.layout)

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            32 * len(self.epilogue_warp_id),
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_c_stage, producer_group=c_producer_group
        )
        while work_tile.is_valid_tile:
            mma_tile_coord_mnl = self._mma_tile_coord(tiled_mma, work_tile)
            # Pre-advance to the next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

            num_tiles_executed = tile_sched.num_tiles_executed
            acc_consumer_state = utils.gemm.sm100.epilogue_tma_store(
                self,
                tidx,
                warp_idx,
                tma_atom_c,
                tCtAcc_base,
                sC,
                tCgC,
                epi_tile,
                num_tiles_executed,
                _identity,
                mma_tile_coord_mnl,
                acc_consumer_state,
                acc_pipeline,
                c_pipeline,
            )

        # Wait for the C stores to complete
        c_pipeline.producer_tail()

        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
        return work_tile, tmem, tile_sched

    def _compute_grid(
        self, c: cute.Tensor, max_active_clusters: cutlass.Constexpr
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Persistent tile scheduler parameters over the CTA tiles of ``C`` and the grid of at
        most ``max_active_clusters`` clusters."""
        c_shape = cute.slice_(self.cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*self.cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid
