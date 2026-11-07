# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import re
from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

try:
    from .custom_pipeline import PipelineCpAsyncUmma
    from .utils import (
        TRTLLM_ENABLE_PDL,
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
        silu_f32,
    )
except ImportError:
    from custom_pipeline import PipelineCpAsyncUmma
    from utils import (
        TRTLLM_ENABLE_PDL,
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
        silu_f32,
    )


"""
High-performance persistent contiguous grouped dense GEMM with gather and SwiGLU fusion
for BF16/FP16 inputs (C = up * silu(gate), where up and gate come from interleaved weight
matrix B) on the NVIDIA Rubin (SM107) architecture using CUTE DSL.

This kernel performs FC1 layer computation with SwiGLU activation fusion:
1. GEMM: acc = alpha * A[token_ids] * B
2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)

No block scaling or scale factors are used. Uses MmaF16BF16Op (K=16 per instruction).

- Matrix A is MxKx1, row-major("K"), ValidM is composed of valid m in different groups
- Matrix B is NxKxL, column-major("K"), L is grouped dimension (number of experts)
  - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]
- Matrix C is Mx(N/2)x1, row-major("N"), N is halved due to SwiGLU fusion
- Token ID mapping tensor enables gather operation for A

This GEMM kernel supports the following features:
    - Utilizes LDGSTS (Load Global to Shared with Swizzle) for A with gather operation
    - Utilizes Tensor Memory Access (TMA) for B matrix
    - Utilizes tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. SCHEDULER warp (warp 10): Dispatches tile information to all consumer warps via tile_info_pipeline.
2. LDGSTS A warps (warps 4-7):
    - Load A matrix from global memory (GMEM) to shared memory (SMEM) using LDGSTS instructions with gather.
    - Uses token_id_mapping to perform permutation/gather during load.
3. TMA B warp (warp 9):
    - Load B matrix from GMEM to SMEM using TMA operations with multicast.
4. MMA warp (warp 8):
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
5. EPILOGUE warps (warps 0-3):
    - Load two accumulator subtiles (up and gate) from tensor memory (TMEM) to registers (RMEM).
    - Apply alpha scaling: up_scaled = alpha * up, gate_scaled = alpha * gate
    - Compute SwiGLU activation: output = up_scaled * silu(gate_scaled)
    - Type convert output to c_dtype.
    - Store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA.

Constraints:
* Supported input data types: BFloat16, Float16
* A/B tensor must have the same data type
* Mma tiler M must be 128 or 256 (use_2cta_instrs)
* Mma tiler N must be 128/256
* Mma tiler K must be 64, MMA instruction K must be 16
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256 (use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned
"""


class Sm107ContiguousGatherGroupedGemmSwigluFusionKernel:
    """Rubin (SM107) contiguous grouped matrix multiplication with gather operation and SwiGLU fusion
    for FC1 layer computation (C = up * silu(gate), where up/gate come from interleaved GEMM result).

    The computation flow:
    1. GEMM: acc = alpha * A[token_ids] * B
    2. SwiGLU: C = up * silu(gate), extracted from interleaved acc with granularity=64

    Note: Output C has N/2 columns since pairs of (up, gate) are combined by SwiGLU.
    No block scaling or scale factors are used. Inputs are BF16/FP16.

    Key Features:
    - Uses LDGSTS instructions for loading A matrix with gather/permutation capability
    - Uses TMA (Tensor Memory Access) for loading B matrix with multicast
    - Token ID mapping enables efficient gather operation during A load
    - SwiGLU activation fusion in epilogue (up * silu(gate) with interleaved weights)
    - Warp specialization: Scheduler (warp 10), A Sync Transform (warp 11, only used when
      use_2cta_instrs is True), LDGSTS A (warps 4-7), TMA B (warp 9), MMA (warp 8),
      Epilogue (warps 0-3)

    :param mma_inst_shape: Shape of MMA instruction (M, N, K)
    :param mma_tiler: Shape of MMA tiler (M, N, K)
    :param cluster_shape_mn: Cluster dimensions (M, N)
    :param vectorized_f32: Whether to use vectorized f32x2 operations
    :param topk: Number of experts selected per token
    :param raster_along_m: If True, raster tiles along M dimension first
    """

    def __init__(
        self,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        topk: cutlass.Int64,
        raster_along_m: bool = False,
    ):
        self.topk = topk
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m

        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.ldgsts_a_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        self.sync_transform_warp_id = 11
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                *self.ldgsts_a_warp_id,
                self.tma_b_warp_id,
                *self.epilog_warp_id,
                self.sched_warp_id,
                self.sync_transform_warp_id,
            )
        )
        self.warps_wo_sched = (
            len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    self.sync_transform_warp_id,
                    *self.ldgsts_a_warp_id,
                )
            )
            if self.use_2cta_instrs
            else len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    *self.ldgsts_a_warp_id,
                )
            )
        )
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

        self.num_smem_capacity = self.smem_capacity
        # num_tmem_alloc_cols already set in __init__

        self.vectorized_f32 = vectorized_f32

        # For epilogue compatibility
        self.epilogue_warp_id = self.epilog_warp_id

        # B-reuse pattern control (disabled for BF16/FP16)
        self.enable_breuse = False

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

        # Configure tiled mma (Rubin SM107, BF16/FP16)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            (self.mma_tiler[0], self.mma_tiler[1]),
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_c = (
            self.mma_tiler[0],
            self.mma_tiler[1] // 2,
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Number of LDGSTS.128 loads per thread for A matrix (each loads 16 M-rows)
        self.a_num_loads = self.cta_tile_shape_mnk[0] // 16

        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epilogue tile for SwiGLU: epi_tile_m must not exceed cta_tile_m_c.
        # Use (64, 32) when cta_tile_m_c == 64 (mma_tiler_m == 64, 1CTA),
        # otherwise (128, 32) for M >= 128.
        epi_tile_m = min(128, self.cta_tile_shape_mnk_c[0])
        self.epi_tile = (epi_tile_m, 32)
        self.epi_tile_n = cute.size(self.epi_tile[1])
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // cute.size(self.epi_tile[0]),
            self.cta_tile_shape_mnk_c[1] // cute.size(self.epi_tile[1]),
        )

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        )

        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Compute TMEM column counts (no scale factors for BF16/FP16)
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the contiguous grouped GEMM with gather operation and SwiGLU fusion.

        This method performs FC1 layer computation for BF16/FP16:
        1. GEMM: acc = alpha * A[token_ids] * B
        2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=32)

        Data loading:
        - A is loaded using LDGSTS instructions with token-based gather
        - B is loaded using TMA instructions with multicast
        - B weights are interleaved: [up_0:32, gate_32:64, up_64:96, gate_96:128, ...]

        :param a: Input tensor A (MxKx1), will be gathered using token_id_mapping
        :type a: cute.Tensor
        :param b: Input tensor B (NxKxL), L is the number of experts/groups
        :type b: cute.Tensor
        :param c: Output tensor C (Mx(N/2)x1), N is halved due to SwiGLU fusion
        :type c: cute.Tensor
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID
        :type tile_idx_to_expert_idx: cute.Tensor
        :param tile_idx_to_mn_limit: Mapping from tile index to M-N dimension limit
        :type tile_idx_to_mn_limit: cute.Tensor
        :param token_id_mapping_tensor: Token ID mapping for gather operation
        :type token_id_mapping_tensor: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles to process
        :type num_non_exiting_tiles: cute.Tensor
        :param alpha: Alpha tensor for each group
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            (self.mma_tiler[0], self.mma_tiler[1]),
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for B
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

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = b_copy_size * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage1cta:
            # (bidx, bidy, bidz, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorage2cta:
            # (bidx, bidy, bidz, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            a_sync_transform_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = (
            SharedStorage2cta if cutlass.const_expr(self.use_2cta_instrs) else SharedStorage1cta
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            alpha,
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
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
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
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline Init: Initialize A pipeline for LDGSTS operations
        # Producer: 4 warps (warps 4-7) with 128 threads total for LDGSTS operations
        # Consumer: MMA warp for consuming A data
        a_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 4,
        )

        a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=a_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize A SYNC Transform pipeline when use_2cta_instrs is True
        # Producer: 1 warp (warp 11) for LDGSTS SYNC transformation operations
        # Consumer: MMA warp for consuming A data
        if cutlass.const_expr(self.use_2cta_instrs):
            a_sync_transform_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * cute.size(cluster_layout_vmnk, mode=[0]),
            )
            a_sync_transform_pipeline = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.a_sync_transform_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=a_sync_transform_pipeline_producer_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        # Pipeline Init: Initialize B pipeline for TMA operations
        # Using PipelineTmaUmma for B since it uses TMA load with multicast support
        # Producer: TMA B warp (warp 9) - 1 warp issuing TMA operations
        # Consumer: MMA warp for consuming B data
        b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_b
        b_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,  # Total bytes loaded by TMA (B only)
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init:Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init (Rubin uses pipeline_init_arrive)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (bidx, bidy, bidz, valid, mn_limit)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        gToken_ml = cute.local_tile(
            token_id_mapping_tensor, cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)), (None,)
        )

        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load B
        #
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
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

        griddepcontrol_wait()

        #
        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                    if mma_tile_coord_m < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                    if mma_tile_coord_m < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized LDGSTS A warps (warps 4-7)
        # These warps use LDGSTS instructions to load A from global to shared memory
        # with gather/permutation capability enabled by token_id_mapping
        #
        if warp_idx <= self.ldgsts_a_warp_id[-1] and warp_idx >= self.ldgsts_a_warp_id[0]:
            #
            # Setup LDGSTS copy atoms for A (BF16/FP16)
            # A: LDGSTS.128 per thread with swizzle_128B for A matrix (8 bf16 elements per 128-bit load)
            #
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mA_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout((1, 8), stride=(8, 1))
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )

            tidx_in_warpgroup = tidx % 128

            sA_tiled = cute.make_tensor(
                sA.iterator,
                layout=cute.make_layout(
                    (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2], self.num_ab_stage),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                    ),
                ),
            )
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tAsA_tiled = a_thr_copy.partition_D(sA_tiled)

            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((self.a_num_loads,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((self.a_num_loads,)),
                cutlass.Boolean,
            )
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Load token IDs for gather operation
                # For A: each thread loads a_num_loads token offsets
                gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                for i in range(self.a_num_loads):
                    token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                    a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
                    a_predicate_tensor[i] = (
                        cutlass.Boolean(1)
                        if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                        < tile_info[4]
                        else cutlass.Boolean(0)
                    )
                    a_token_offset_tensor[i] = (
                        a_token_offset_tensor[i] // self.topk
                        if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                        < tile_info[4]
                        else 0
                    )

                tAgA = gA_mkl[(None, None, 0, None, 0)]
                A_gmem_thread_offset = cute.assume((tidx_in_warpgroup % 8) * 8, divby=8)

                # Peek (try_wait) A buffer empty
                a_producer_state.reset_count()
                peek_a_empty_status = cutlass.Boolean(1)
                if a_producer_state.count < k_tile_cnt:
                    peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                #
                # Load A with LDGSTS and gather/permutation
                # Each K-tile iteration loads one K-tile of A from GMEM to SMEM
                # using LDGSTS instructions with token-based gather addressing
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for A buffer empty
                    a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)

                    tAgA_ktile = tAgA[(None, None, a_producer_state.count)]
                    tAsA_ktile = tAsA_tiled[(None, None, None, a_producer_state.index)]

                    for i in range(self.a_num_loads):
                        #
                        # Load A matrix: a_num_loads x LDGSTS.128 per thread with swizzle_128B
                        # Each LDGSTS.128 loads 8 bf16 elements (128 bits) from GMEM to SMEM
                        # Global memory address is computed using token offset for gather operation
                        # Predicate mask guards against invalid token IDs (padding tokens marked as -1)
                        #
                        A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                            a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride, divby=8
                        )
                        A_gmem_slice_offset = cute.assume(A_gmem_slice_offset, divby=8)
                        tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                        tAgA_slice = cute.make_tensor(tAgA_slice_ptr, layout=cute.make_layout((8,)))

                        tAsA_slice = cute.make_tensor(
                            tAsA_ktile[(None, i, None)].iterator, layout=cute.make_layout((8,))
                        )
                        a_predicate_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        a_predicate_slice[0] = a_predicate_tensor[i]

                        cute.copy_atom_call(
                            a_atom_copy, tAgA_slice, tAsA_slice, pred=a_predicate_slice
                        )

                    a_pipeline.producer_commit(a_producer_state)

                    # Peek (try_wait) A buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    a_producer_state.advance()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait A pipeline buffer empty
            #
            a_pipeline.producer_tail(a_producer_state)

        #
        # Specialized A Sync Transform Warp (warp 11) when use_2cta_instrs is True
        # This warp serves as sync transformation for A
        #
        if warp_idx == self.sync_transform_warp_id:
            if cutlass.const_expr(self.use_2cta_instrs):
                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                # First tile
                work_tile = tile_sched.initial_work_tile_info()

                a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # Get the first tile info
                valid_tile_info = cute.make_rmem_tensor((1,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                is_valid_tile = valid_tile_info[0] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    # Peek (try_wait) A buffer full for k_tile = 0
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)
                    # Peek (try_wait) a sync transform buffer empty
                    a_sync_transform_producer_state.reset_count()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Conditionally wait for A buffer full
                        a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)

                        a_sync_transform_pipeline.producer_commit(a_sync_transform_producer_state)
                        a_sync_transform_producer_state.advance()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                    #
                    # Advance to next tile
                    #
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                    is_valid_tile = valid_tile_info[0] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                #
                # Wait A sync transform buffer empty
                #
                a_sync_transform_pipeline.producer_tail(a_sync_transform_producer_state)

        #
        # Specialized TMA B load warp (warp 9)
        # This warp uses TMA instructions to load B from global to shared memory
        # with multicast support to reduce L2 memory traffic
        #
        if warp_idx == self.tma_b_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) B buffer empty for k_tile = prefetch_k_tile_cnt
                b_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = b_pipeline.producer_try_acquire(b_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for B buffer empty
                    b_pipeline.producer_acquire(b_producer_state, peek_ab_empty_status)

                    tBgB_k = tBgB_slice[(None, b_producer_state.count)]
                    tBsB_pipe = tBsB[(None, b_producer_state.index)]

                    tma_bar = b_pipeline.producer_get_barrier(b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) B buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    b_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = b_pipeline.producer_try_acquire(b_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait B buffer empty
            #
            b_pipeline.producer_tail(b_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            if cutlass.const_expr(self.use_2cta_instrs):
                a_sync_transform_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )

            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_consumer_state.reset_count()
                    peek_a_sync_transform_full_status = cutlass.Boolean(1)
                    if a_sync_transform_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_a_sync_transform_full_status = (
                            a_sync_transform_pipeline.consumer_try_wait(
                                a_sync_transform_consumer_state
                            )
                        )
                    a_consumer_state.reset_count()
                else:
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                b_consumer_state.reset_count()
                peek_b_full_status = cutlass.Boolean(1)
                if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)

                # Get accumulator stage index
                acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                for k_tile in cutlass.range(k_tile_cnt):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_pipeline.consumer_wait(
                                a_sync_transform_consumer_state, peek_a_sync_transform_full_status
                            )
                        else:
                            a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)
                        b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)

                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                b_consumer_state.index,
                            )

                            tiled_mma.set(
                                tcgen05.Field.ACCUMULATE,
                                k_tile != 0 or kblock_idx != 0,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                        # Async arrive AB buffer empty
                        a_pipeline.consumer_release(a_consumer_state)
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_pipeline.consumer_release(
                                a_sync_transform_consumer_state
                            )
                        b_pipeline.consumer_release(b_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    if cutlass.const_expr(self.use_2cta_instrs):
                        a_sync_transform_consumer_state.advance()
                        peek_a_sync_transform_full_status = cutlass.Boolean(1)
                        if a_sync_transform_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_a_sync_transform_full_status = (
                                    a_sync_transform_pipeline.consumer_try_wait(
                                        a_sync_transform_consumer_state
                                    )
                                )
                        a_consumer_state.advance()
                    else:
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                    b_consumer_state.advance()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue (Rubin: transform both accumulator and C layout)
            # transform_partitioned_tensor_layout merges (MMA_ATOM, MMA_M) into flat M.
            # This is unconditional to match the reference kernel pattern.
            #
            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgC_for_epi = transform_partitioned_tensor_layout(tCgC)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_transformed, tCgC_for_epi, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
                self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC_for_epi, epi_tile, sC
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Get alpha for current group
                #

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]

                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]

                # Get accumulator stage index
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                #
                # Process accumulator subtiles with SwiGLU fusion and store to global memory
                # Each iteration processes a pair of subtiles (up, gate) and computes
                # up * silu(gate)
                #
                # The accumulator has full N columns with interleaved [up, gate] at
                # granularity=32. Output C has N/2 columns. With epi_tile_n, we iterate
                # over M and N output subtiles separately to correctly map up/gate pairs.
                #
                # tTR_tAcc shape: (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE) before group
                # After selecting acc_stage, shape is (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                # bSG_gC shape: ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
                #   -> after slicing: ((ATOM_V, REST_V), EPI_M, EPI_N)
                #
                interleave_granularity = 32
                gate_offset = interleave_granularity // self.epi_tile_n
                epi_m_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                acc_n_subtile_cnt = cute.size(tTR_tAcc.shape, mode=[4])
                out_n_subtile_cnt = acc_n_subtile_cnt // 2  # N/2 output subtiles per M subtile

                for epi_m_idx in cutlass.range(epi_m_cnt):
                    for out_n_idx in cutlass.range(out_n_subtile_cnt):
                        # Map output N subtile to accumulator N subtile:
                        # For each interleave block of 2*gate_offset N-subtiles in acc,
                        # first gate_offset subtiles are up, next gate_offset are gate
                        block_idx = out_n_idx // gate_offset
                        within_block = out_n_idx % gate_offset
                        up_n_subtile = block_idx * 2 * gate_offset + within_block
                        gate_n_subtile = block_idx * 2 * gate_offset + gate_offset + within_block
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, epi_m_idx, up_n_subtile)]
                        tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, epi_m_idx, gate_n_subtile)]

                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                        acc_vec_up = tTR_rAcc_up.load()
                        acc_vec_gate = tTR_rAcc_gate.load()

                        #
                        # SwiGLU activation: output = up * silu(gate)
                        # where silu(x) = x * sigmoid(x)
                        # up and gate are extracted from interleaved accumulator subtiles
                        #
                        tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                        if cutlass.const_expr(self.vectorized_f32):
                            # SwiGLU Packed Version: uses f32x2 packed operations for better performance
                            # Computes: output = (alpha * up) * silu(alpha * gate)
                            # where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                            LOG2_E = cutlass.Float32(1.4426950408889634)
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_up), 2):
                                acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                                    (acc_vec_up[i], acc_vec_up[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                )
                                acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                )
                                tCompute_log2e = cute.arch.mul_packed_f32x2(
                                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                    (-LOG2_E, -LOG2_E),
                                )
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.add_packed_f32x2(
                                    (
                                        cute.math.exp2(tCompute_log2e[0], fastmath=True),
                                        cute.math.exp2(tCompute_log2e[1], fastmath=True),
                                    ),
                                    (1.0, 1.0),
                                )
                                tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                                tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (tCompute[i], tCompute[i + 1]),
                                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                )
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (tCompute[i], tCompute[i + 1]),
                                    (acc_vec_up_alpha[0], acc_vec_up_alpha[1]),
                                )
                        else:
                            # SwiGLU Unpacked Version: scalar operations
                            # Computes: output = (alpha * up) * silu(alpha * gate)
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                                acc_vec_up_alpha = acc_vec_up[i] * cutlass.Float32(alpha_val)
                                acc_vec_gate_alpha = acc_vec_gate[i] * cutlass.Float32(alpha_val)
                                tCompute[i] = acc_vec_up_alpha * silu_f32(
                                    acc_vec_gate_alpha, fastmath=True
                                )

                        #
                        # Convert to C type
                        #
                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                        tRS_rC.store(acc_vec)

                        #
                        # Store C to shared memory
                        #
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage

                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        #
                        # TMA store C to global memory
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, epi_m_idx, out_n_idx)],
                            )
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory
        (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_up: The partitioned accumulator tensor for acc up
            - tTR_rAcc_gate: The partitioned accumulator tensor for acc gate
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load (Rubin uses transformed layout)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # tAcc is already transformed: (M, N, STAGE) layout
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc,
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # gC_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(gC_mnl, epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register
        array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # gC_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        num_smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout of operand C.
        :type c_layout: utils.LayoutEnum
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages (always 2 for BF16/FP16, no B-reuse)
        num_acc_stage = 2

        # Default C stages
        num_c_stage = 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        a_bytes = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
        b_bytes = cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        ab_bytes_per_stage = a_bytes + b_bytes
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def is_valid_dtypes(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid for BF16/FP16 GEMM.

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        if ab_dtype not in {cutlass.Float16, cutlass.BFloat16}:
            return False
        if c_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
            return False
        return True

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        return True

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the mma tiler and cluster shape are valid for BF16/FP16."""
        # Check valid mma_inst_shape
        if mma_inst_shape[0] not in [64, 128, 256]:
            return False
        # SwiGLU Fusion requires even epi_tile counts
        if mma_inst_shape[1] not in [128, 256]:
            return False

        # Check valid mma_tiler
        if mma_tiler[0] not in [64, 128, 256]:
            return False
        if mma_tiler[1] not in [128, 256]:
            return False

        # No B-reuse for BF16/FP16: mma_tiler[0] == mma_inst_shape[0] always
        if mma_tiler[0] != mma_inst_shape[0]:
            return False
        if mma_tiler[1] != mma_inst_shape[1]:
            return False

        # K-dimension constraints for BF16/FP16: K=16 per MMA instruction, mma_tiler K=64
        if mma_tiler[2] != 64 or mma_inst_shape[2] != 16:
            return False

        # Check 2CTA cluster shape constraint
        if cluster_shape_mn[0] % (2 if mma_inst_shape[0] == 256 else 1) != 0:
            return False

        # Check cluster shape validity
        def _is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            return False

        # We only support cluster shape n = 1 for now
        if cluster_shape_mn[1] != 1:
            return False
        return True

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented for BF16/FP16.

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_inst_shape: MMA instruction shape (M, N, K)
        :type mma_inst_shape: Tuple[int, int, int]
        :param mma_tiler: MMA tiler shape (M, N, K)
        :type mma_tiler: Tuple[int, int, int]
        :param cluster_shape_mn: Cluster shape (M, N)
        :type cluster_shape_mn: Tuple[int, int]
        :param m: M dimension
        :param n: N dimension
        :param k: K dimension
        :param l: L dimension (number of groups)
        :param a_major: Major axis of A
        :param b_major: Major axis of B
        :param c_major: Major axis of C

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        # Check data types
        if not cls.is_valid_dtypes(a_dtype, c_dtype):
            return False

        # Check layouts
        if not cls.is_valid_layouts(a_dtype, c_dtype, a_major, b_major, c_major):
            return False

        # Check MMA tiler and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(
            a_dtype, b_dtype, mma_inst_shape, mma_tiler, cluster_shape_mn
        ):
            return False

        # Check tensor alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        # Check A/B layout
        if not (a_major == "k" and b_major == "k"):
            return False
        return True

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_stride_m: cutlass.Int64 = cutlass.Int64(0),
    ):
        interm_size = n // 2
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        actual_c_stride_m = interm_size if c_stride_m == 0 else c_stride_m
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout(
                (m, interm_size, 1), stride=(actual_c_stride_m, 1, m * actual_c_stride_m)
            ),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        token_id_mapping = cute.make_tensor(token_id_mapping_ptr, layout=cute.make_layout((m,)))
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )

        return self(
            a,
            b,
            c,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
        )


# ============================================================================
# Run utilities
# ============================================================================


def create_mask(group_m_list, mma_tiler_m, permuted_m=None):
    """Create mask and group mapping for contiguous grouped GEMM with gather and SwiGLU.

    :param group_m_list: List of M values for each group (will be aligned to mma_tiler_m)
    :param mma_tiler_m: MMA tile size in M dimension, also used for alignment
    :param permuted_m: Optional padded M dimension for cuda_graph support
    :return: Tuple of (valid_m, aligned_group_m_list, tile_idx_to_expert_idx,
             tile_idx_to_mn_limit, num_non_exiting_tiles)
    """
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []
    tile_idx_to_mn_limit = []

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        aligned_group_m_list.append(aligned_group_m)

        num_tiles_in_group = aligned_group_m // mma_tiler_m
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)
        for tile_idx_in_group in range(num_tiles_in_group):
            tile_idx_to_mn_limit.append(
                valid_m + min(tile_idx_in_group * mma_tiler_m + mma_tiler_m, group_m)
            )
        valid_m += aligned_group_m

    num_non_exiting_tiles = len(tile_idx_to_expert_idx)

    if permuted_m is not None:
        if permuted_m < valid_m:
            raise ValueError(f"permuted_m ({permuted_m}) must be >= valid_m ({valid_m}).")
        if permuted_m > valid_m:
            num_padding_tiles = (permuted_m - valid_m) // mma_tiler_m
            tile_idx_to_expert_idx.extend([int(-2e9)] * num_padding_tiles)
            tile_idx_to_mn_limit.extend([int(-2e9)] * num_padding_tiles)

    tile_idx_to_expert_idx = torch.tensor(tile_idx_to_expert_idx, device="cuda", dtype=torch.int32)
    num_non_exiting_tiles_tensor = torch.tensor(
        [num_non_exiting_tiles], device="cuda", dtype=torch.int32
    )
    tile_idx_to_mn_limit_tensor = torch.tensor(
        tile_idx_to_mn_limit, device="cuda", dtype=torch.int32
    )

    return (
        valid_m,
        aligned_group_m_list,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles_tensor,
        tile_idx_to_mn_limit_tensor,
    )


def create_token_id_mapping_tensor(group_m_list, mma_tiler_m, max_token_id, permuted_m=None):
    """Create token_id_mapping tensor for gather operation with random distribution."""
    valid_m = 0
    for group_m in group_m_list:
        valid_m += ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m

    tensor_m = permuted_m if permuted_m is not None else valid_m

    base_data = torch.full((tensor_m,), -1, dtype=torch.int32)

    accumulated_m = 0
    for group_m in group_m_list:
        start_idx = accumulated_m
        rounded_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        random_token_ids = torch.randint(0, max_token_id, (group_m,), dtype=torch.int32)
        base_data[start_idx : start_idx + group_m] = random_token_ids
        accumulated_m += rounded_group_m

    token_id_mapping_ref = base_data.clone()
    token_id_mapping_tensor, token_id_mapping_torch = cutlass_torch.cute_tensor_like(
        token_id_mapping_ref, cutlass.Int32, is_dynamic_layout=True, assumed_align=4
    )
    return token_id_mapping_ref, token_id_mapping_tensor, token_id_mapping_torch


def create_tensors(
    num_groups,
    group_m_list,
    n,
    k,
    a_major,
    b_major,
    cd_major,
    a_dtype,
    b_dtype,
    c_dtype,
    mma_tiler_m,
    permuted_m=None,
):
    """Create tensors for contiguous grouped GEMM with gather operation and SwiGLU fusion.

    Output C has N/2 columns since SwiGLU combines pairs of (up, gate) from interleaved B weights.
    """
    torch.manual_seed(1111)

    alpha_torch_cpu = torch.randn((num_groups,), dtype=torch.float32)

    (
        valid_m,
        aligned_group_m_list,
        _tile_idx_to_expert_idx,
        _num_non_exiting_tiles,
        _tile_idx_to_mn_limit,
    ) = create_mask(group_m_list, mma_tiler_m, permuted_m)

    max_m = max(group_m_list)

    tensor_m = permuted_m if permuted_m is not None else valid_m

    a_torch_cpu = cutlass_torch.matrix(1, max_m, k, a_major == "m", cutlass.Float32)
    b_torch_cpu = cutlass_torch.matrix(num_groups, n, k, b_major == "n", cutlass.Float32)
    c_torch_cpu = cutlass_torch.matrix(1, tensor_m, n // 2, cd_major == "m", cutlass.Float32)

    a_tensor, a_torch_gpu = cutlass_torch.cute_tensor_like(
        a_torch_cpu, a_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch_gpu = cutlass_torch.cute_tensor_like(
        b_torch_cpu, b_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    a_tensor.mark_compact_shape_dynamic(
        mode=1 if a_major == "k" else 0,
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=16,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1 if b_major == "k" else 0,
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=16,
    )
    c_tensor.mark_compact_shape_dynamic(
        mode=1 if cd_major == "n" else 0,
        stride_order=(2, 0, 1) if cd_major == "n" else (2, 1, 0),
        divisibility=16,
    )

    token_id_mapping_cpu, token_id_mapping, token_id_mapping_torch = create_token_id_mapping_tensor(
        group_m_list, mma_tiler_m, max_token_id=max_m, permuted_m=permuted_m
    )

    tile_idx_to_expert_idx = from_dlpack(_tile_idx_to_expert_idx).mark_layout_dynamic()
    tile_idx_to_mn_limit = from_dlpack(_tile_idx_to_mn_limit).mark_layout_dynamic()
    num_non_exiting_tiles = from_dlpack(_num_non_exiting_tiles).mark_layout_dynamic()

    alpha = from_dlpack(alpha_torch_cpu.cuda()).mark_layout_dynamic()

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        aligned_group_m_list,
        valid_m,
        token_id_mapping_cpu,
    )


def run(
    nkl: Tuple[int, int, int],
    group_m_list: Tuple[int, ...],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_inst_shape: Tuple[int, int, int],
    mma_tiler: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    permuted_m: int = None,
    raster_along_m: bool = False,
    use_cupti: bool = False,
    **kwargs,
):
    """Run contiguous grouped GEMM with gather and SwiGLU fusion on Rubin (BF16/FP16)."""
    mma_tiler_m = mma_tiler[0]

    print(
        "Running Rubin Persistent Contiguous Grouped GEMM with Gather and SwiGLU Fusion (BF16/FP16):"
    )
    print(f"nkl: {nkl}")
    print(f"group_m_list: {group_m_list}")
    print(f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}")
    if permuted_m is not None:
        print(f"Padded M (CUDA graph support): {permuted_m}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"MMA Inst Shape: {mma_inst_shape}, MMA Tiler: {mma_tiler}")
    print(f"Cluster Shape: {cluster_shape_mn}")
    print(f"Raster along M: {raster_along_m}")
    print(f"Use CUPTI: {'True' if use_cupti else 'False'}")

    n, k, num_groups = nkl

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    if not Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.can_implement(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        mma_inst_shape=mma_inst_shape,
        mma_tiler=mma_tiler,
        cluster_shape_mn=cluster_shape_mn,
        m=mma_tiler_m,
        n=n,
        k=k,
        l=num_groups,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
    ):
        raise TypeError(
            f"Unsupported testcase a_dtype={a_dtype}, b_dtype={b_dtype}, "
            f"c_dtype={c_dtype}, mma_inst_shape={mma_inst_shape}, "
            f"mma_tiler={mma_tiler}, cluster_shape_mn={cluster_shape_mn}"
        )

    (
        a_tensor,
        b_tensor,
        c_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        aligned_group_m_list,
        valid_m,
        token_id_mapping_cpu,
    ) = create_tensors(
        num_groups,
        group_m_list,
        n,
        k,
        a_major,
        b_major,
        c_major,
        a_dtype,
        b_dtype,
        c_dtype,
        mma_tiler_m,
        permuted_m,
    )

    gemm = Sm107ContiguousGatherGroupedGemmSwigluFusionKernel(
        mma_inst_shape,
        mma_tiler,
        cluster_shape_mn,
        True,
        topk=1,
        raster_along_m=raster_along_m,
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        max_active_clusters,
        current_stream,
    )

    compiled_gemm(
        a_tensor,
        b_tensor,
        c_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        current_stream,
    )

    torch.cuda.synchronize()

    if not skip_ref_check:
        print("Verifying results...")
        interleave_granularity = 32
        n_out = n // 2

        # Step 1: Compute full GEMM (no scale factors for BF16/FP16)
        gemm_result = torch.empty((1, valid_m, n), dtype=torch.float32)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            res_a = a_torch_cpu[token_id_mapping_cpu[start:end], :, 0]
            res_b = b_torch_cpu[:, :, i]
            gemm_result[0, start:end, :] = (
                torch.einsum("mk,nk->mn", res_a, res_b) * alpha_torch_cpu[i]
            )
            start = end

        # Step 2: Apply SwiGLU on interleaved GEMM result
        assert n % (2 * interleave_granularity) == 0
        ref = torch.empty((1, valid_m, n_out), dtype=torch.float32)
        for n_block in range(0, n, 2 * interleave_granularity):
            up_result = gemm_result[0, :, n_block : n_block + interleave_granularity]
            gate_result = gemm_result[
                0, :, n_block + interleave_granularity : n_block + 2 * interleave_granularity
            ]
            silu_gate = gate_result * torch.sigmoid(gate_result)
            output_block = up_result * silu_gate
            out_start = n_block // 2
            out_end = out_start + interleave_granularity
            ref[0, :, out_start:out_end] = output_block

        ref = ref.permute((1, 2, 0))

        # Convert c back to f32 for comparison
        res = c_torch_cpu.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(res, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )

        res = res[:valid_m]
        mask = token_id_mapping_cpu[:valid_m] >= 0
        res = res.cpu()[mask]
        ref = ref[mask]

        print(f"valid_m: {valid_m}, ref.shape: {ref.shape}, res.shape: {res.shape}")

        torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-02)

    def generate_tensors():
        (
            a_tensor,
            b_tensor,
            c_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            *_,
        ) = create_tensors(
            num_groups,
            group_m_list,
            n,
            k,
            a_major,
            b_major,
            c_major,
            a_dtype,
            b_dtype,
            c_dtype,
            mma_tiler_m,
            permuted_m,
        )
        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            c_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        tensor_m = permuted_m if permuted_m is not None else valid_m
        one_workspace_bytes = (
            a_torch_gpu.numel() * a_torch_gpu.element_size()
            + b_torch_gpu.numel() * b_torch_gpu.element_size()
            + c_torch_gpu.numel() * c_torch_gpu.element_size()
            + (tensor_m // mma_tiler_m) * 4
            + (tensor_m // mma_tiler_m) * 4
            + tensor_m * 4
            + 1 * 4
            + alpha_torch_cpu.numel() * alpha_torch_cpu.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cute.testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cupti=use_cupti,
    )

    return exec_time


def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def read_benchmark_file(
    filepath: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, ...]]:
    """Read benchmark file and return nkl and group_m_list."""
    problems = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                dims = parts[1].split("x")
                if len(dims) == 3:
                    m, n, k = int(dims[0]), int(dims[1]), int(dims[2])
                    problems.append((m, n, k))

        if not problems:
            raise ValueError(f"No valid problems found in benchmark file: {filepath}")

        m_first, n, k = problems[0]
        num_groups = len(problems)
        m_values = tuple(m for m, _, _ in problems)

        print(f"Loaded {num_groups} problems from benchmark file")
        print(f"Using N={n}, K={k}, L={num_groups}")
        print(f"M values per group: {m_values}")

        return ((n, k, num_groups), m_values)

    except FileNotFoundError:
        raise argparse.ArgumentTypeError(f"Benchmark file not found: {filepath}")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Error reading benchmark file: {e}")


def parse_benchmark_arg(
    arg: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, ...]]:
    """Parse benchmark argument string."""
    match_list = re.match(r"\[([\d,\s]+)\]\s*x\s*(\d+)\s*x\s*(\d+)", arg)
    if match_list:
        m_str = match_list.group(1)
        n = int(match_list.group(2))
        k = int(match_list.group(3))
        try:
            m_values = tuple(int(x.strip()) for x in m_str.split(","))
            num_groups = len(m_values)
            return ((n, k, num_groups), m_values)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer list in benchmark argument: {arg}")

    parts = arg.split("x")
    if len(parts) == 4:
        try:
            m, n, k, num_groups = [int(x.strip()) for x in parts]
            m_values = tuple([m] * num_groups)
            return ((n, k, num_groups), m_values)
        except ValueError:
            pass

    raise argparse.ArgumentTypeError(f"Invalid benchmark argument format. Got: {arg}")


def main():
    """Main entry point for running the Rubin BF16/FP16 SwiGLU fusion kernel."""
    parser = argparse.ArgumentParser(
        description="Rubin Contiguous Gather Grouped GEMM with SwiGLU Fusion (BF16/FP16)."
    )

    parser.add_argument("--nkl", type=parse_comma_separated_ints, default=(256, 512, 1))
    parser.add_argument("--fixed_m", type=int, default=None)
    parser.add_argument("--custom_mask", type=parse_comma_separated_ints, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--permuted_m", type=int, default=None)
    parser.add_argument("--mma_inst_shape", type=parse_comma_separated_ints, default=(128, 128, 16))
    parser.add_argument("--mma_tiler", type=parse_comma_separated_ints, default=(128, 128, 64))
    parser.add_argument("--cluster_shape_mn", type=parse_comma_separated_ints, default=(1, 1))
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--tolerance", type=float, default=1e-01)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true", default=False)
    parser.add_argument("--raster_along_m", action="store_true", default=False)
    parser.add_argument(
        "--use_cupti", action="store_true", default=False, help="Use Cupti profiler"
    )

    args = parser.parse_args()

    if args.benchmark:
        if os.path.isfile(args.benchmark):
            nkl, group_m_list = read_benchmark_file(args.benchmark)
        else:
            nkl, group_m_list = parse_benchmark_arg(args.benchmark)
    else:
        if len(args.nkl) != 3:
            parser.error("--nkl must contain exactly 3 values")
        n, k, num_groups = args.nkl
        nkl = (n, k, num_groups)

        if args.custom_mask is not None:
            group_m_list = args.custom_mask
            if len(group_m_list) != num_groups:
                parser.error(f"--custom_mask must have exactly {num_groups} values")
        elif args.fixed_m is not None:
            group_m_list = tuple([args.fixed_m] * num_groups)
        else:
            group_m_list = tuple([128] * num_groups)

    if len(args.mma_inst_shape) != 3:
        parser.error("--mma_inst_shape must contain exactly 3 values")
    if len(args.mma_tiler) != 3:
        parser.error("--mma_tiler must contain exactly 3 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        nkl,
        group_m_list,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_inst_shape,
        args.mma_tiler,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.permuted_m,
        args.raster_along_m,
        args.use_cupti,
    )
    print(f"Execution time: {exec_time:.2f} us")
    print("PASS")


if __name__ == "__main__":
    main()
