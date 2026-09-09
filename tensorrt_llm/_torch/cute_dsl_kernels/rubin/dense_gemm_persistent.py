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

# Adapted from the NVIDIA CUTLASS CuTe DSL Rubin dense GEMM example:
# python/CuTeDSL/examples/rubin/dense_gemm_persistent.py

from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils

from ...cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
    from cutlass import testing
else:
    import cutlass.cute.testing as testing
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import CollectorOp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

# Customized epilogue with a_scale/b_scale support for FP8 per-tensor GEMM
from tensorrt_llm._torch.cute_dsl_kernels.utils.gemm import sm100 as gemm_sm100

from ..blackwell.dense_gemm_persistent import (
    PersistentDenseGemmKernel as SM100PersistentDenseGemmKernel,
)
from ..blackwell.dense_gemm_persistent import _compute_stages

"""
A high-performance persistent batched dense GEMM example for the NVIDIA Rubin SM107 architecture
using CuTe DSL, extending the Blackwell implementation.

This kernel supports per-tensor FP8 quantization with input_scale and weight_scale parameters,
computing: C = (input_scale * weight_scale) * (A @ B)

Comparison: SM107 (Rubin) vs. SM100 (Blackwell)
- Shared memory (SMEM): 328 KiB on SM107; 228 KiB on SM100
- Tensor memory (TMEM): 576 columns for SM107; 512 columns for SM100
- MMA K dimension: SM107 supports both K=32 and K=64 (SM100 only supports K=32)
- CollectorOp: Enhanced support in SM107 for advanced TMEM accumulator handling

.. code-block:: bash

    python examples/rubin/dense_gemm_persistent.py                      \
        --a_dtype Float8E4M3FN --b_dtype Float8E5M2                     \
        --c_dtype Float16 --acc_dtype Float32                           \
        --input_scale 0.5 --weight_scale 2.0                            \
        --mma_tiler 512,256,128 --mma_inst_shape 256,256,64             \
        --cluster_shape_mn 2,1                                          \
        --mnkl 8192,8192,8192,1                                         \
        --use_tma_store --use_2cta_instrs

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/rubin/dense_gemm_persistent.py                  \
        --a_dtype Float8E4M3FN --b_dtype Float8E4M3FN                   \
        --c_dtype Float16 --acc_dtype Float32                           \
        --input_scale 1.0 --weight_scale 1.0                            \
        --mma_tiler 256,256,128 --mma_inst_shape 256,256,64             \
        --cluster_shape_mn 2,1                                          \
        --mnkl 8192,8192,8192,1                                         \
        --use_tma_store --use_2cta_instrs                               \
        --warmup_iterations 1 --iterations 10 --skip_ref_check


Additional constraints:
- Only FP8 inputs are supported (Float8E4M3FN or Float8E5M2) for now
- Only Breuse-Bkeep pattern is supported for now(if mma_tiler[0] is the double of mma_inst_shape[0] then enable
  Breuse-Bkeep pattern)
- For K=64: M in the MMA tiler must be 128 (1 CTA) or 256 (2 CTAs)
- For K=32: same constraints as Blackwell
- input_scale and weight_scale are float32 scalars applied in epilogue as:
  output = (input_scale * weight_scale) * accumulator
"""


class SM107PersistentDenseGemmKernel(SM100PersistentDenseGemmKernel):
    """Persistent dense GEMM kernel for Rubin with per-tensor FP8 quantization support.

    Extends `SM100PersistentDenseGemmKernel` with SM107-specific behavior and limits.

    SM107 adds support for the Bkeep-Breuse pattern optimization which reuses
    the B matrix across two separate GEMM operations.

    This kernel supports per-tensor FP8 quantization via input_scale and weight_scale:
    - Formula: C = (input_scale * weight_scale) * (A @ B)
    - Scales are applied in the epilogue after MMA completes
    - Both scales are float32 scalars for numerical accuracy

    :param mma_tiler: MMA tile shape (M, N, K). K may be 32 or 64 on SM107
    :type mma_tiler: Tuple[int, int, int]

    See `SM100PersistentDenseGemmKernel` for all other parameters.

    notes:
    - Data types: FP8 only (Float8E4M3FN, Float8E5M2)
    - K=64 constraint: M must be 128 (1 CTA) or 256 (2 CTAs)
    - Resources: larger SMEM (328 KiB) and TMEM (576 columns)
    - Bkeep-Breuse pattern: Optimizes B matrix reuse
    - Per-tensor scaling: Applied in epilogue via lambda operation

    **Example:**

    .. code-block:: python

        gemm = SM107PersistentDenseGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler=(256, 128, 128),
            mma_inst_shape=(256, 128, 64),
            cluster_shape_mn=(2, 1),
            use_tma_store=True,
            swizzle_size=1,
            raster_along="m",
        )
        # Call with per-tensor scales
        gemm(a, b, c, input_scale=0.5, weight_scale=2.0,
             max_active_clusters, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler: Tuple[int, int, int],
        mma_inst_shape: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
    ):
        """Initialize the Rubin persistent dense GEMM kernel.

        :param mma_tiler: MMA tiler (M, N, K).
        :type mma_tiler: Tuple[int, int, int]
        :param mma_inst_shape: MMA instruction shape (M, N, K).
        :type mma_inst_shape: Tuple[int, int, int]

        Other parameters are identical to the base class.
        """
        super().__init__(
            acc_dtype,
            use_2cta_instrs,
            mma_inst_shape[0:2],
            cluster_shape_mn,
            use_tma_store,
            swizzle_size,
            raster_along,
        )
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.mma_tiler = mma_tiler
        self.mma_inst_shape = mma_inst_shape
        # Bkeep-Breuse pattern is controlled by mma_inst_shape and mma_tiler
        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])

        else:
            return (1, 1, 1)

    def _create_tiled_mma(self):
        return utils.sm107.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape,
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

    def _create_tiled_mma_bkeep(self):
        """Create TiledMma for keep operation (with fill collector for B).

        This is used in the Bkeep-Breuse pattern for the first GEMM operation.
        The 'fill' collector operation indicates that B data should be kept
        for reuse in subsequent operations.
        """
        return utils.sm107.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape,
            permutation_mnk=self._get_mma_permutation_mnk(),
            b_collector_op=CollectorOp.FILL,
        )

    def _create_tiled_mma_breuse(self):
        """Create TiledMma for reuse operation (with lastuse collector for B).

        This is used in the Bkeep-Breuse pattern for the second GEMM operation.
        The 'lastuse' collector operation indicates that this is the last use
        of the B data that was kept from the previous operation.
        """
        return utils.sm107.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape,
            permutation_mnk=self._get_mma_permutation_mnk(),
            b_collector_op=CollectorOp.LASTUSE,
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Configure tiled mma
        tiled_mma = self._create_tiled_mma()

        # Compute mma/cluster/tile shapes
        self.mma_inst_tile_k = self.mma_tiler[2] // self.mma_inst_shape[2]

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        c_smem_layout = None
        if cutlass.const_expr(self.use_tma_store):
            c_smem_layout = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, 1
            )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        _, self.num_ab_stage, self.num_c_stage = _compute_stages(
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
        if self.cta_tile_shape_mnk[1] == 256 and self.enable_breuse:
            self.num_acc_stage = 1
        else:
            self.num_acc_stage = 2

        # Compute A/B/C shared memory layout
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

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = SM100PersistentDenseGemmKernel._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
        )

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
        tiled_mma_bkeep: Optional[cute.TiledMma] = None,
        tiled_mma_breuse: Optional[cute.TiledMma] = None,
        a_scale: Optional[cute.Tensor] = None,
        b_scale: Optional[cute.Tensor] = None,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
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

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
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

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # TMA load A/B
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

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)

            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        tile_crd = (None, None, None, handle.index)

                        # Get current stage tensors (3D)
                        # tCrA has shape (MMA, MMA_M, MMA_K, STAGE) → (MMA, MMA_M, MMA_K)
                        tCrA_stage = tCrA[tile_crd]
                        # tCrB has shape (MMA, MMA_N, MMA_K, STAGE) → (MMA, MMA_N, MMA_K)
                        tCrB_stage = tCrB[tile_crd]

                        # Check if we should use Bkeep-Breuse pattern
                        if cutlass.const_expr(self.enable_breuse):
                            # Slice accumulator once (shared across k_phase)
                            tCtAcc_keep = tCtAcc[(None, 0, 0)]
                            tCtAcc_reuse = tCtAcc[(None, 1, 0)]

                            for k_phase in range(self.mma_inst_tile_k):
                                # Bkeep-Breuse pattern

                                # B slice - select N=0 from (MMA, MMA_N, MMA_K) → (MMA, MMA_K)
                                tCrB_slice = tCrB_stage[(None, 0, k_phase)]

                                # Keep operation - first A slice
                                # Select M=0 from (MMA, MMA_M, MMA_K) → (MMA, MMA_K)
                                tCrA_keep = tCrA_stage[(None, 0, k_phase)]

                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_phase != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_keep,
                                    tCrA_keep,
                                    tCrB_slice,
                                    tCtAcc_keep,
                                )

                                # Reuse operation - second A slice
                                # Select M=1 from (MMA, MMA_M, MMA_K) → (MMA, MMA_K)
                                tCrA_reuse = tCrA_stage[(None, 1, k_phase)]

                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_phase != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_reuse,
                                    tCrA_reuse,
                                    tCrB_slice,
                                    tCtAcc_reuse,
                                )
                        else:
                            # Regular kernel pattern
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA_stage,
                                tCrB_stage,
                                tCtAcc,
                            )

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop for epilogue
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )

            if cutlass.const_expr(self.use_tma_store):
                # (EPI_TILE_M, EPI_TILE_N, STAGE)
                # sC = storage.sC.get_tensor(
                #     c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
                # )

                assert tma_atom_c is not None and sC is not None

                gemm_sm100.epilogue_tma_store(
                    self,
                    tidx,
                    warp_idx,
                    acc_pipeline,
                    tiled_mma,
                    tma_atom_c,
                    tCtAcc,
                    sC,
                    tCgC,
                    epi_tile,
                    tile_sched,
                    epilogue_op,
                    a_scale=a_scale,
                    b_scale=b_scale,
                )
            else:
                gemm_sm100.epilogue(
                    self,
                    tidx,
                    acc_pipeline,
                    tiled_mma,
                    tCtAcc,
                    tCgC,
                    epi_tile,
                    tile_sched,
                    epilogue_op,
                    tmem_dealloc_barrier,
                    a_scale=a_scale,
                    b_scale=b_scale,
                )

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    def check_supported_dtypes(self, a_dtype, b_dtype, c_dtype):
        """Validate data types for Rubin.

        Inputs must be FP8 (Float8E4M3FN or Float8E5M2). The accumulator must
        be Float32 or Float16.

        :raises testing.CantImplementError: If the dtypes are not supported
        """
        if a_dtype not in {cutlass.Float8E4M3FN, cutlass.Float8E5M2} or b_dtype not in {
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            raise testing.CantImplementError(
                f"This example only supports FP8 input types, got {a_dtype} and {b_dtype}"
            )
        if self.acc_dtype not in {cutlass.Float32, cutlass.Float16}:
            raise testing.CantImplementError(
                f"This example only supports Float32 or Float16 accumulator, got {self.acc_dtype}"
            )
        if not SM100PersistentDenseGemmKernel.check_supported_dtypes(
            a_dtype,
            b_dtype,
            self.acc_dtype,
            c_dtype,
            allow_mixed_ab=True,
        ):
            raise testing.CantImplementError(
                f"Unsupported dtype combination: {a_dtype}, {b_dtype}, {self.acc_dtype}, {c_dtype}"
            )

    def check_mma_tiler_and_cluster_shape(self):
        """Validate the MMA tiler and cluster shape for Rubin.

        :raises testing.CantImplementError: If the mma tiler is invalid
        """
        # Rubin constraint for K=64
        if self.mma_inst_shape[2] == 64:
            if not self.use_2cta_instrs and self.mma_inst_shape[0] != 128:
                raise testing.CantImplementError(
                    f"For K=64 with use_2cta_instrs=False, mma_inst_shape M must be 128, got {self.mma_inst_shape[0]}"
                )
            elif self.use_2cta_instrs and self.mma_inst_shape[0] != 256:
                raise testing.CantImplementError(
                    f"For K=64 with use_2cta_instrs=True, mma_inst_shape M must be 256, got {self.mma_inst_shape[0]}"
                )
        if (
            self.mma_tiler[0] // self.mma_inst_shape[0] != 2
            and self.mma_tiler[0] // self.mma_inst_shape[0] != 1
        ) or self.mma_tiler[1] != self.mma_inst_shape[1]:
            raise testing.CantImplementError(
                f"Invalid mma tiler: {self.mma_tiler} with mma_inst_shape: {self.mma_inst_shape}"
            )
        # Call parent to check common constraints
        super().check_mma_tiler_and_cluster_shape()

    def can_implement(
        self,
        mnkl: Tuple[int, int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Determine whether this Rubin kernel supports the configuration."""
        try:
            self.check_supported_dtypes(a_dtype, b_dtype, c_dtype)
            self.check_mma_tiler_and_cluster_shape()
            m, n, k, batch_size = mnkl
            self.check_tensor_alignment(
                m,
                n,
                k,
                batch_size,
                a_dtype,
                b_dtype,
                c_dtype,
                a_major,
                b_major,
                c_major,
            )
            self.check_epilog_store_option(m, n)
        except testing.CantImplementError:
            return False
        return True

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        a_scale: Optional[cute.Tensor] = None,
        b_scale: Optional[cute.Tensor] = None,
    ):
        """Override parent __call__ to pass Bkeep-Breuse tiled_mma objects to kernel."""
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        tiled_mma = self._create_tiled_mma()
        # Create Bkeep-Breuse tiled_mma variants if enabled
        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = self._create_tiled_mma_bkeep()
            tiled_mma_breuse = self._create_tiled_mma_breuse()

        # Setup attributes that dependent on gemm inputs
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

        # Launch the kernel synchronously with Bkeep-Breuse parameters
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
            tiled_mma_bkeep,  # Pass Bkeep tiled_mma
            tiled_mma_breuse,  # Pass Breuse tiled_mma
            a_scale,
            b_scale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_scale_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Executes the wrapped GEMM kernel with dynamically shaped tensors.

        Args:
            m (int): The M dimension of the GEMM problem.
            n (int): The N dimension of the GEMM problem.
            k (int): The K dimension of the GEMM problem.
            a_ptr (cute.Pointer): Pointer to the A tensor.
            b_ptr (cute.Pointer): Pointer to the B tensor.
            a_scale_ptr (cute.Pointer): Pointer to the A scale tensor.
            b_scale_ptr (cute.Pointer): Pointer to the B scale tensor.
            c_tensor (cute.Tensor): C tensor, used for tvm ffi stream detection.
            max_active_clusters (cutlass.Constexpr): Maximum number of active
                clusters.
            stream (cuda.CUstream): CUDA stream for the operation.
        """

        # m, k, batch_size with inner most dimension as k
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)),
        )
        # n, k, batch_size with inner most dimension as k
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (n, k, 1),
                order=(1, 0, 2),
            ),
        )
        a_scale_tensor = cute.make_tensor(a_scale_ptr, layout=cute.make_layout((1,)))
        b_scale_tensor = cute.make_tensor(b_scale_ptr, layout=cute.make_layout((1,)))
        self(
            a_tensor,
            b_tensor,
            c_tensor,
            max_active_clusters,
            stream,
            a_scale=a_scale_tensor,
            b_scale=b_scale_tensor,
        )
