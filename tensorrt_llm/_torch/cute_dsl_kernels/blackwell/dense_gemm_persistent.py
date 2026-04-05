# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This file is copied and modified from cutlass example https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py

from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from .utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
)


def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
    use_tma_store: bool,
    c_smem_layout: Union[cute.Layout, None],
) -> Tuple[int, int, int]:
    """Computes the number of stages for A/B/C operands based on heuristics."""
    num_acc_stage = 2
    num_c_stage = 2 if use_tma_store else 0

    a_smem_layout_stage_one = utils.sm100.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
    b_smem_layout_staged_one = utils.sm100.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)

    ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
        b_dtype, b_smem_layout_staged_one
    )
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    num_ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage

    if use_tma_store:
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
    return num_acc_stage, num_ab_stage, num_c_stage


class PersistentDenseGemmKernel:
    """Persistent batched dense GEMM (C = A x B) for Blackwell SM100 using CuTe DSL.

    Supports BF16/FP16 inputs with FP32 accumulator and BF16/FP16 output.

    Notes:
        - A and B tensor must have the same data type.
        - Supported A/B data types: Float16, BFloat16, TFloat32, Float8E4M3FN,
          Float8E5M2, Int8, Uint8
        - Supported accumulator: Float32, Float16, Int32
        - MMA tiler M: 64/128 (1CTA) or 128/256 (2CTA)
        - MMA tiler N: 32-256, step 32
        - Cluster M must be multiple of 2 if 2CTA
        - Cluster M*N <= 16, positive powers of 2
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool = True,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along = raster_along
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.arch = "sm_100"

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilogue_warp_id)
        )
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs."""
        tiled_mma = self._create_tiled_mma()

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

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        c_smem_layout = None
        if cutlass.const_expr(self.use_tma_store):
            c_smem_layout = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, 1
            )

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
            c_smem_layout,
        )

        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        self.c_smem_layout_staged = None
        if self.use_tma_store:
            self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
            )

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation."""
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        # Setup TMA load for B
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
            )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
            max_active_clusters,
        )

        # Launch the kernel
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """GPU device kernel performing the Persistent batched GEMM computation."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch tma desc
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup cta/thread coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        _tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            _tmem_dealloc_barrier = pipeline.NamedBarrier(  # noqa: F841
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Setup smem tensor A/B
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        # Compute multicast mask for A/B buffer full
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Partition global tensor for TiledMMA_A/B/C
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Partition global/shared tensor for TMA load A/B
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

        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Construct the scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        # PDL: Wait for previous grid to finish
        griddepcontrol_wait()

        # Specialized TMA load warp
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

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

                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # Reset ACCUMULATE for each new output tile
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # Inner loop over kblocks within each K tile.
                        # Set ACCUMULATE=True after first gemm call to
                        # avoid clearing the accumulator on each sub-MMA.
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_crd = (None, None, kblock_idx, handle.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_crd], tCrB[kblock_crd], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        handle.release()

                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        # Specialized epilogue warps
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # -- Epilogue partition setup (TMA store path) --
            assert cutlass.const_expr(self.use_tma_store)
            assert tma_atom_c is not None and sC is not None

            # TMEM -> RMEM copy setup
            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile,
                use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

            gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
            tTR_rAcc = cute.make_fragment(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
            )

            # RMEM -> SMEM copy setup
            copy_atom_r2s = utils.sm100.get_smem_store_op(
                self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            # SMEM -> GMEM TMA store setup
            sC_for_tma = cute.group_modes(sC, 0, 2)
            gC_for_tma = cute.group_modes(gC_mnl_epi, 0, 2)
            bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                tma_atom_c, 0, cute.make_layout(1), sC_for_tma, gC_for_tma
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )

            # -- Epilogue tile scheduling loop --
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                num_tiles_executed = tile_sched.num_tiles_executed

                # Slice to per mma tile
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # Store accumulator to global memory in sub-tiles
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = (num_tiles_executed - 1) * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    # Load accumulator from TMEM to RMEM
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Convert to output type and apply epilogue op
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    tRS_rC.store(acc_vec)

                    # Store to SMEM
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])

                    # Fence and barrier
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    epilog_threads = 32 * len(self.epilogue_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    # TMA store from SMEM to GMEM
                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                # Release accumulator buffer
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            # Wait for C store complete and deallocate TMEM
            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # PDL: Launch dependent kernels
        griddepcontrol_launch_dependents()

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        swizzle_size: int,
        raster_along: Literal["m", "n"],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid size using persistent tile scheduler."""
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, swizzle_size, raster_along == "m"
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """Compute the number of tensor memory allocation columns."""
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols

    @staticmethod
    def check_supported_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Check if the dtypes are valid."""
        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.TFloat32,
            cutlass.Uint8,
            cutlass.Int8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }
        if a_dtype not in valid_ab_dtypes or b_dtype not in valid_ab_dtypes:
            return False
        if a_dtype != b_dtype:
            return False
        if acc_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.Int32}:
            return False

        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.TFloat32,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {cutlass.Float16, cutlass.Float8E4M3FN, cutlass.Float8E5M2},
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        if a_dtype not in acc_ab_compatibility.get(acc_dtype, set()):
            return False

        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
            cutlass.Float16: {cutlass.BFloat16, cutlass.Float16},
            cutlass.Int32: {
                cutlass.BFloat16,
                cutlass.Float16,
                cutlass.Float32,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        if c_dtype not in acc_c_compatibility.get(acc_dtype, set()):
            return False

        return True

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the mma tiler and cluster shape are valid."""
        if not (
            (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
            or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
        ):
            return False
        if mma_tiler_mn[1] not in range(32, 257, 32):
            return False
        if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
            return False
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            return False
        return True

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        batch_size: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if the tensor alignment is valid (contiguous dim 16-byte aligned)."""

        def check_contiguous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contiguous_16B_alignment(ab_dtype, a_major == "m", (m, k, batch_size))
            or not check_contiguous_16B_alignment(ab_dtype, b_major == "n", (n, k, batch_size))
            or not check_contiguous_16B_alignment(c_dtype, c_major == "m", (m, n, batch_size))
        ):
            return False
        return True

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        batch_size: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if the gemm can be implemented."""
        if not PersistentDenseGemmKernel.check_supported_dtypes(
            ab_dtype, ab_dtype, acc_dtype, c_dtype
        ):
            return False
        if not PersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            use_2cta_instrs, mma_tiler_mn, cluster_shape_mn
        ):
            return False
        if not PersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, batch_size, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False
        # Check epilogue store alignment for non-TMA store
        # (TMA store handles OOB; we always use TMA store)
        return True

    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Executes the wrapped GEMM kernel with dynamically shaped tensors.

        Args:
            m: The M dimension of the GEMM problem.
            n: The N dimension of the GEMM problem.
            k: The K dimension of the GEMM problem.
            batch_size: The batch dimension.
            a_ptr: Pointer to the A tensor (M x K x batch_size, row-major K).
            b_ptr: Pointer to the B tensor (N x K x batch_size, row-major K).
            c_tensor: Output tensor as cute.Tensor for TVM FFI stream detection.
            max_active_clusters: Maximum number of active clusters.
            stream: CUDA stream for the operation.
        """
        # m, k, batch_size with inner most dimension as k
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, batch_size), order=(1, 0, 2)),
        )
        # n, k, batch_size with inner most dimension as k
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (n, k, batch_size),
                order=(1, 0, 2),
            ),
        )

        self(
            a_tensor,
            b_tensor,
            c_tensor,
            max_active_clusters,
            stream,
        )

    @cute.jit
    def wrapper_strided(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        a_stride_m: cutlass.Int32,
        a_stride_batch: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Executes the GEMM kernel with explicit A tensor strides.

        Like ``wrapper`` but allows non-contiguous A tensors by accepting
        the M and batch strides directly.  The K stride is assumed to be 1
        (row-major in K).  B is always contiguous.

        Args:
            m: The M dimension of the GEMM problem.
            n: The N dimension of the GEMM problem.
            k: The K dimension of the GEMM problem.
            batch_size: The batch dimension.
            a_ptr: Pointer to the A tensor data.
            b_ptr: Pointer to the B tensor data.
            c_tensor: Output tensor as cute.Tensor.
            a_stride_m: Stride of A along the M dimension (in elements).
            a_stride_batch: Stride of A along the batch dimension (in elements).
            max_active_clusters: Maximum number of active clusters.
            stream: CUDA stream for the operation.
        """
        # A with explicit strides: (M, K, batch_size), K stride = 1
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_layout(
                (m, k, batch_size),
                stride=(a_stride_m, 1, a_stride_batch),
            ),
        )
        # B is always contiguous: (N, K, batch_size) with K innermost
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (n, k, batch_size),
                order=(1, 0, 2),
            ),
        )

        self(
            a_tensor,
            b_tensor,
            c_tensor,
            max_active_clusters,
            stream,
        )
