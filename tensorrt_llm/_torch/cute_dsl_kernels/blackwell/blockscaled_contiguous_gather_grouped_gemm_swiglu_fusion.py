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

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Int32

from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    TRTLLM_ENABLE_PDL,
    fmin,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
    silu_f32,
)

"""
High-performance persistent blockscaled contiguous grouped dense GEMM with gather and SwiGLU fusion
(C = up * silu(gate), where up and gate come from interleaved weight matrix B)
example for the NVIDIA Blackwell architecture using CUTE DSL.

This kernel performs FC1 layer computation with SwiGLU activation fusion:
1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
3. Optional Quant: When c_dtype is Float4E2M1FN, generates scale factor C and quantizes output

- Matrix A is MxKx1, A can be row-major("K"), ValidM is composed of valid m in different groups
- Matrix B is NxKxL, B can be column-major("K"), L is grouped dimension (number of experts)
  - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]
- Matrix C is Mx(N/2)x1, C can be row-major("N"), N is halved due to SwiGLU fusion
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk,
  which has M×ceil_div(K, sf_vec_size)×1 elements
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk,
  which has N×ceil_div(K, sf_vec_size)×L elements
- Token ID mapping tensor enables gather operation for A and SFA

Matrix A/C Memory Layout Diagrams:

   ```
    Group 0    Group 1   Group 2
   -+---------+---------+---------+
    |         |         |         |
   K| ValidM0 | ValidM1 | ValidM2 |
    |         |         |         |
   -+---------+---------+---------+
    |<-        ValidM           ->|
   ```
   Note: the Group(L) dimension will be flatted into M dimension, and the rest Group(L) size is 1.
         each ValidM will be aligned to 256 or 128. The alignment is determined by the mma_tiler_mn parameter.
         For NVFP4, 2CTA, the alignment is 256. For NVFP4, 1CTA, the alignment is 128.

This GEMM kernel supports the following features:
    - Utilizes LDGSTS (Load Global to Shared with Swizzle) for A and SFA with gather operation
    - Utilizes Tensor Memory Access (TMA) for B and SFB matrices
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. SCHEDULER warp (warp 10): Dispatches tile information to all consumer warps via tile_info_pipeline.
2. LDGSTS A/SFA warps (warps 4-7):
    - Load A matrix from global memory (GMEM) to shared memory (SMEM) using LDGSTS instructions with gather.
    - Load SFA (scale factor A) from GMEM to SMEM using LDGSTS instructions.
    - Uses token_id_mapping to perform permutation/gather during load.
3. TMA B/SFB warp (warp 9):
    - Load B and SFB matrices from GMEM to SMEM using TMA operations with multicast.
4. MMA warp (warp 8):
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
5. EPILOGUE warps (warps 0-3):
    - Load two accumulator subtiles (up and gate) from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Apply alpha scaling: up_scaled = alpha * up, gate_scaled = alpha * gate
    - Compute SwiGLU activation: output = up_scaled * silu(gate_scaled), where silu(x) = x * sigmoid(x)
    - If c_dtype is Float4E2M1FN: generate scale factor C (SFC) and quantize output
    - Type convert output to c_dtype.
    - Store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.

CUDA Graph Support:
* For CUDA graph support, the tile_idx_to_expert_idx, token_id_mapping, A/C matrices,
  and scale factor A can be padded to a larger size
  (e.g., permuted_m = m*topK + num_local_experts*(256-1),
  example: 4096*8 + (256/32)*255 = 34808)
* Use create_tensors() with permuted_m parameter to automatically pad:
  - tile_idx_to_expert_idx: padded for invalid tiles (set to -2e9 for padding tiles)
  - token_id_mapping: padded to permuted_m size (invalid tokens set to -1)
  - A matrix: padded to permuted_m rows (padding rows contain dummy data)
  - C matrix: padded to permuted_m rows (output buffer for cuda_graph)
  - Scale factor A: padded to match A matrix dimensions
* Kernel handling of padding:
  - Scheduler warp checks if tile_idx >= num_non_exiting_tiles to exit
  - Only valid tiles (tile_idx < num_non_exiting_tiles) are written to tile_info pipeline
  - LDGSTS warps use token_id_mapping predicates to skip invalid tokens (token_id == -1)
  - When no more valid tiles exist, outer loop exits and calls producer_tail()
  - Consumer warps process only valid tiles from pipeline
  - No deadlock or synchronization issues
* Consumer warps check initial tile against num_non_exiting_tiles and set
  is_valid_tile=False if tile_idx >= num_non_exiting_tiles
* Only rows within (aligned_groupm[0]+aligned_groupm[1]+...) contain valid data
* Padding rows in C matrix will not be written by the kernel
"""


# TODO: Remove this hook helper function after nvidia-cutlass-dsl 4.4 is released.
def hooked_PersistentTileSchedulerParams_init(
    self,
    problem_shape_ntile_mnl: cute.Shape,
    cluster_shape_mnk: cute.Shape,
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    *,
    loc=None,
    ip=None,
):
    if cluster_shape_mnk[2] != 1:
        raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")
    if swizzle_size < 1:
        raise ValueError(f"expect swizzle_size >= 1, but get {swizzle_size}")

    self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
    # cluster_shape_mnk is kept for reconstruction
    self._cluster_shape_mnk = cluster_shape_mnk
    self.cluster_shape_mn = cluster_shape_mnk[:2]
    self.swizzle_size = swizzle_size
    self._raster_along_m = raster_along_m
    self._loc = loc

    # Apply swizzle if swizzle_size > 1
    if swizzle_size > 1:
        problem_shape_ncluster_mnl = cute.round_up(
            self.problem_layout_ncluster_mnl.shape,
            (1, swizzle_size, 1) if raster_along_m else (swizzle_size, 1, 1),
        )

        if raster_along_m:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                (
                    problem_shape_ncluster_mnl[0],
                    (swizzle_size, problem_shape_ncluster_mnl[1] // swizzle_size),
                    problem_shape_ncluster_mnl[2],
                ),
                stride=(
                    swizzle_size,
                    (1, swizzle_size * problem_shape_ncluster_mnl[0]),
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        else:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                (
                    (swizzle_size, problem_shape_ncluster_mnl[0] // swizzle_size),
                    problem_shape_ncluster_mnl[1],
                    problem_shape_ncluster_mnl[2],
                ),
                stride=(
                    (1, swizzle_size * problem_shape_ncluster_mnl[1]),
                    swizzle_size,
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )

    # Create FastDivmod divisors (only when swizzle_size == 1 for correctness)
    # FastDivmod assumes simple col-major/row-major layout, incompatible with swizzled layouts
    if swizzle_size == 1:
        problem_shape_ncluster_mnl = cute.ceil_div(
            self.problem_shape_ntile_mnl, cluster_shape_mnk[:2], loc=loc, ip=ip
        )
        if raster_along_m:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                problem_shape_ncluster_mnl,
                stride=(
                    1,
                    problem_shape_ncluster_mnl[0],
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        else:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                problem_shape_ncluster_mnl,
                stride=(
                    problem_shape_ncluster_mnl[1],
                    1,
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        problem_layout_size = cute.size(self.problem_layout_ncluster_mnl, loc=loc, ip=ip)
        cluster_count_m = self.problem_layout_ncluster_mnl.shape[0]
        cluster_count_n = self.problem_layout_ncluster_mnl.shape[1]

        # batch_fdd: Used to map linear_idx to work_unit_id (handles persistent scheduling)
        self.batch_fdd = cute.fast_divmod_create_divisor(problem_layout_size, loc=loc, ip=ip)

        # cluster_shape_m_fdd: Used to decode work_unit_id to cluster coordinates
        self.cluster_shape_m_fdd = cute.fast_divmod_create_divisor(cluster_count_m, loc=loc, ip=ip)

        # cluster_shape_n_fdd: Used for the second level decomposition
        self.cluster_shape_n_fdd = cute.fast_divmod_create_divisor(cluster_count_n, loc=loc, ip=ip)
    else:
        # FastDivmod not applicable with swizzling, set to None
        self.batch_fdd = None
        self.cluster_shape_m_fdd = None
        self.cluster_shape_n_fdd = None


def hooked_get_cluster_work_idx_with_fastdivmod(
    self, current_work_linear_idx: Int32, *, loc=None, ip=None
) -> Tuple[Int32, Int32, Int32]:
    work_iteration, work_unit_id = divmod(current_work_linear_idx, self.params.batch_fdd)

    if self.params._raster_along_m:
        # raster_along_m=True means column major (m is fastest)
        # First, get cluster_m using cluster_shape_m_fdd
        cluster_n_batch, cluster_m = divmod(work_unit_id, self.params.cluster_shape_m_fdd)

        # Then decode cluster_n_batch to get cluster_n and batch_l using FastDivmod
        batch_l, cluster_n = divmod(cluster_n_batch, self.params.cluster_shape_n_fdd)
    else:
        # raster_along_m=False means row major (n is fastest)
        # First, get cluster_n using cluster_shape_n_fdd
        cluster_m_batch, cluster_n = divmod(work_unit_id, self.params.cluster_shape_n_fdd)

        # Then decode cluster_m_batch to get cluster_m and batch_l using FastDivmod
        batch_l, cluster_m = divmod(cluster_m_batch, self.params.cluster_shape_m_fdd)

    return (cluster_m, cluster_n, batch_l)


cutlass.utils.PersistentTileSchedulerParams.__init__ = hooked_PersistentTileSchedulerParams_init
cutlass.utils.StaticPersistentTileScheduler._get_cluster_work_idx_with_fastdivmod = (
    hooked_get_cluster_work_idx_with_fastdivmod
)


class BlockScaledContiguousGatherGroupedGemmKernel:
    """This class implements contiguous grouped matrix multiplication with gather operation and SwiGLU fusion
    for FC1 layer computation (C = up * silu(gate), where up/gate come from interleaved GEMM result).

    The computation flow:
    1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
    2. SwiGLU: C = up * silu(gate), extracted from interleaved acc with granularity=64
    3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

    Note: Output C has N/2 columns since pairs of (up, gate) are combined by SwiGLU.

    Key Features:
    - Uses LDGSTS instructions for loading A and SFA matrices with gather/permutation capability
    - Uses TMA (Tensor Memory Access) for loading B and SFB matrices with multicast
    - Token ID mapping enables efficient gather operation during A/SFA load
    - SwiGLU activation fusion in epilogue (up * silu(gate) with interleaved weights)
    - Optional quantization fusion for Float4E2M1FN output with scale factor generation
    - Warp specialization: Scheduler (warp 10), A Sync Transform (warp 11, only used when
      use_2cta_instrs is True), LDGSTS A/SFA (warps 4-7), TMA B/SFB (warp 9), MMA (warp 8),
      Epilogue (warps 0-3)

    :param sf_vec_size: Scalefactor vector size (16 for NVF4, 32 for MXF4/MXF8).
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N).
        Note: use_2cta_instrs is automatically inferred from mma_tiler_mn[0]
        (True when M=256, False when M=128).
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param vectorized_f32: Whether to use vectorized f32x2 operations for better performance.
    :type vectorized_f32: bool

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
        # Note: Float4E2M1FN output includes SFC generation and quantization support for internal testing.
        - Float4E2M1FN (with scale factor generation)

    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> # Note: use_2cta_instrs is auto-inferred from mma_tiler_mn[0]
        >>> # (True when M=256, False when M=128)
        >>> gemm = BlockScaledContiguousGatherGroupedGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),  # use_2cta_instrs=True since M=256
        ...     cluster_shape_mn=(2, 1),
        ...     vectorized_f32=True,
        ... )
        >>> gemm(
        ...     a=a_tensor,
        ...     b=b_tensor,
        ...     c=c_tensor,
        ...     sfa=sfa_tensor,
        ...     sfb=sfb_tensor,
        ...     sfc_tensor=None,
        ...     norm_const_tensor=None,
        ...     tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        ...     tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        ...     token_id_mapping_tensor=token_id_mapping_tensor,
        ...     num_non_exiting_tiles=num_non_exiting_tiles,
        ...     alpha=alpha,
        ...     max_active_clusters=max_active_clusters,
        ...     stream=stream,
        ... )
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        topk: cutlass.Int64,
        raster_along_m: bool = False,
    ):
        """Initializes the configuration for a Blackwell blockscaled dense GEMM kernel with
        gather operation and SwiGLU fusion.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Automatically inferred from mma_tiler_mn[0]
              (True when M=256, False when M=128).

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3.  Scale Factor Configuration:
            - sf_vec_size: Vector size for block-scaled quantization.

        4.  Performance Optimization:
            - vectorized_f32: Enable vectorized f32x2 operations.

        5.  MoE Configuration:
            - topk: Number of experts selected per token (used for token ID mapping).

        :param sf_vec_size: Vector size for scale factors (16 for NVF4, 32 for MXF4/MXF8).
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
            use_2cta_instrs is automatically set based on M (True if M=256, False if M=128).
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param vectorized_f32: Enable vectorized f32x2 operations for better performance.
        :type vectorized_f32: bool
        :param topk: Number of experts selected per token (used for token ID mapping).
        :type topk: cutlass.Int64
        """

        self.sf_vec_size = sf_vec_size
        self.topk = topk
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.raster_along_m = raster_along_m

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

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

        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.vectorized_f32 = vectorized_f32

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

        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Configure tiled mma
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.mma_tiler_sfa = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k // 16,
        )

        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.mma_tiler_c = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1] // 2,
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk_sfa = (
            self.mma_tiler_sfa[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfa[1],
            self.mma_tiler_sfa[2],
        )

        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

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

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = (128, 64)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_c[1] // self.epi_tile[1],
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
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
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
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = self.num_acc_stage == 1

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        self.epi_tile_n_required = 2 * cute.size(self.epi_tile[1])
        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n_required

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        sfc_tensor: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
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

        This method performs FC1 layer computation:
        1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
        2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
        3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

        Data loading:
        - A and SFA are loaded using LDGSTS instructions with token-based gather
        - B and SFB are loaded using TMA instructions with multicast
        - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]

        Execution steps:
        1. Setup static attributes before smem/grid computation
        2. Setup TMA load/store atoms for B, SFB, and C (no TMA for A/SFA)
        3. Compute grid size with regard to hardware constraints
        4. Define shared storage for kernel
        5. Launch the kernel synchronously with warp specialization:
           - Scheduler warp: Dispatches tile information
           - LDGSTS warps: Load A and SFA with gather
           - A Sync Transform warps: Transform the sync signal of A and SFA from global to
             shared memory when use_2cta_instrs is True
           - TMA warp: Load B and SFB with multicast
           - MMA warp: Perform matrix multiply-accumulate
           - Epilogue warps: Apply SwiGLU activation, optional quantization, and store results

        :param a: Input tensor A (MxKx1), will be gathered using token_id_mapping
        :type a: cute.Tensor
        :param b: Input tensor B (NxKxL), L is the number of experts/groups, weights are interleaved for SwiGLU
        :type b: cute.Tensor
        :param c: Output tensor C (Mx(N/2)x1), N is halved due to SwiGLU fusion
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A, will be gathered using token_id_mapping
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param sfc_tensor: Scale factor tensor C for quantized output (None if not quantizing)
        :type sfc_tensor: Optional[cute.Tensor]
        :param norm_const_tensor: Normalization constant for scale factor generation
            (None if not quantizing)
        :type norm_const_tensor: Optional[cute.Tensor]
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID,
            shape (permuted_m/cta_tile_m,) where cta_tile_m is the CTA tile M size
        :type tile_idx_to_expert_idx: cute.Tensor
        :param tile_idx_to_mn_limit: Mapping from tile index to M-N dimension limit
            for boundary checking, shape (permuted_m/cta_tile_m,)
        :type tile_idx_to_mn_limit: cute.Tensor
        :param token_id_mapping_tensor: Token ID mapping for gather operation, shape (permuted_m,)
        :type token_id_mapping_tensor: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles to process (valid_m/cta_tile_m), shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param alpha: Alpha tensor for each group
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfb tensor by filling B tensor to scale factor atom layout
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        # Setup sfc tensor by filling C tensor to scale factor atom layout
        self.generate_sfc = sfc_tensor is not None and norm_const_tensor is not None
        if cutlass.const_expr(self.generate_sfc):
            sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(c.shape, self.sf_vec_size)
            sfc_tensor = cute.make_tensor(sfc_tensor.iterator, sfc_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
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

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # This modifies the layout to handle overlapping 256x(# of scale factors for a single column of B (nNSF))
        # logical blocks for SFB when cta_tile_shape_n=192.
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout)

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size

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
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
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
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = (
            SharedStorage2cta if cutlass.const_expr(self.use_2cta_instrs) else SharedStorage1cta
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            a,
            tma_atom_b,
            tma_tensor_b,
            sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            alpha,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
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

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to
        partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFC_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
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
            cpasync.prefetch_descriptor(tma_atom_sfb)
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

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline Init: Initialize A pipeline for LDGSTS operations
        # Producer: 4 warps (warps 4-7) with 128 threads total for LDGSTS operations
        # Consumer: MMA warp for consuming A/SFA data
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
        # Consumer: MMA warp for consuming A/SFA data
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
        # Using PipelineTmaUmma for B/SFB since they use TMA load with multicast support
        # Producer: TMA B/SFB warp (warp 9) - 1 warp issuing TMA operations
        # Consumer: MMA warp for consuming B/SFB data
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
            tx_count=self.num_tma_load_bytes,  # Total bytes loaded by TMA (B + SFB)
            cta_layout_vmnk=cluster_layout_vmnk,
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
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bidx, bidy, bidz, valid, mn_limit)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
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

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.cta_tile_shape_mnk_sfa, (None, 0, None)), (None, None, None)
        )

        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
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
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
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

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

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
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

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
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
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
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
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
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized LDGSTS A/SFA warps (warps 4-7)
        # These warps use LDGSTS instructions to load A and SFA from global to shared memory
        # with gather/permutation capability enabled by token_id_mapping
        #
        if warp_idx <= self.ldgsts_a_warp_id[-1] and warp_idx >= self.ldgsts_a_warp_id[0]:
            #
            # Setup LDGSTS copy atoms for A and SFA
            # A: 8x LDGSTS.128 per thread with swizzle_128B for A matrix (32 elements per thread)
            # SFA: 4x LDGSTS.32 per thread with 512-element block swizzling for scale factor A (4 elements per thread)
            #
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mA_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout((1, 32), stride=(32, 1))
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )

            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mSFA_mkl.element_type,
                num_bits_per_copy=32,
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
                cute.make_layout((8,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)),
                cutlass.Boolean,
            )
            sfa_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)),
                cutlass.Int32,
            )
            sfa_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)),
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
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Load token IDs for gather operation
                # For A matrix: each thread loads 8 token offsets (for 8 LDGSTS.128 operations)
                # For SFA matrix: each thread loads 1 token offset (for 4 LDGSTS.32 operations)
                gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                for i in range(8):
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

                token_ml_tile_offset = (
                    8 * (tidx_in_warpgroup // 32)
                    + 32 * ((tidx_in_warpgroup % 32) // 8)
                    + (tidx_in_warpgroup % 8)
                )
                sfa_token_offset_tensor[0] = gToken_ml_tile[token_ml_tile_offset] // self.topk
                sfa_predicate_tensor[0] = (
                    cutlass.Boolean(1)
                    if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                    < tile_info[4]
                    else cutlass.Boolean(0)
                )
                relative_sfa_token_offset = sfa_token_offset_tensor[0]

                tAgA = gA_mkl[(None, None, 0, None, 0)]
                A_gmem_thread_offset = cute.assume((tidx_in_warpgroup % 8) * 32, divby=32)
                tAgSFA = gSFA_mkl[(relative_sfa_token_offset, None, 0, None, 0)]

                tAsSFA = sSFA[
                    (
                        (
                            (
                                (
                                    8 * (tidx_in_warpgroup // 32) + (tidx_in_warpgroup % 8),
                                    (tidx_in_warpgroup % 32) // 8,
                                ),
                                None,
                            ),
                            None,
                        ),
                        None,
                        None,
                        None,
                    )
                ]

                # Peek (try_wait) SCALE buffer empty
                a_producer_state.reset_count()
                peek_a_empty_status = cutlass.Boolean(1)
                if a_producer_state.count < k_tile_cnt:
                    peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                #
                # Load A and SFA with LDGSTS and gather/permutation
                # Each K-tile iteration loads one K-tile of A and SFA from GMEM to SMEM
                # using LDGSTS instructions with token-based gather addressing
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)

                    tAgA_ktile = tAgA[(None, None, a_producer_state.count)]
                    tAsA_ktile = tAsA_tiled[(None, None, None, a_producer_state.index)]

                    tAgSFA_ktile = tAgSFA[(None, a_producer_state.count)]
                    tAsSFA_ktile = tAsSFA[
                        (
                            None,
                            None,
                            None,
                            None,
                            a_producer_state.index,
                        )
                    ]

                    for i in range(8):
                        #
                        # Load A matrix: 8x LDGSTS.128 per thread with swizzle_128B
                        # Each LDGSTS.128 loads 32 elements (128 bits) from GMEM to SMEM
                        # Global memory address is computed using token offset for gather operation
                        # Predicate mask guards against invalid token IDs (padding tokens marked as -1)
                        #
                        A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                            a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride, divby=32
                        )
                        A_gmem_slice_offset = cute.assume(A_gmem_slice_offset, divby=32)
                        tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                        tAgA_slice = cute.make_tensor(
                            tAgA_slice_ptr, layout=cute.make_layout((32,))
                        )

                        tAsA_slice = cute.make_tensor(
                            tAsA_ktile[(None, i, None)].iterator, layout=cute.make_layout((32,))
                        )
                        a_predicate_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        a_predicate_slice[0] = a_predicate_tensor[i]

                        cute.copy_atom_call(
                            a_atom_copy, tAgA_slice, tAsA_slice, pred=a_predicate_slice
                        )

                    for i in range(4):
                        #
                        # Load SFA: 4x LDGSTS.32 per thread with 512-element block swizzling
                        # Each LDGSTS.32 loads 4 scale factor elements (32 bits) from GMEM to SMEM
                        # Uses same token offset as A matrix for consistent gather operation
                        #
                        swizzled_iterator = (tidx_in_warpgroup % 32) // 8 ^ i
                        tAgSFA_slice_ptr = tAgSFA_ktile.iterator + 4 * swizzled_iterator
                        tAgSFA_slice = cute.make_tensor(
                            tAgSFA_slice_ptr, layout=cute.make_layout((4,))
                        )

                        tAsSFA_slice_ptr = tAsSFA_ktile.iterator + 512 * swizzled_iterator
                        tAsSFA_slice = cute.make_tensor(tAsSFA_slice_ptr, cute.make_layout((4,)))

                        cute.copy_atom_call(
                            sfa_atom_copy, tAgSFA_slice, tAsSFA_slice, pred=sfa_predicate_tensor
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
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait A pipeline buffer empty
            #
            a_pipeline.producer_tail(a_producer_state)

        #
        # Specialized A/SFA Sync Transform Warp (warp 11) when use_2cta_instrs is True
        # This warp serve as sync transformation for A and SFA
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
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
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
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                #
                # Wait A sync transform buffer empty
                #
                a_sync_transform_pipeline.producer_tail(a_sync_transform_producer_state)

        #
        # Specialized TMA B/SFB load warp (warp 9)
        # This warp uses TMA instructions to load B and SFB from global to shared memory
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
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
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

                # Apply SFB slicing hack when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
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
                    tBgSFB_k = tBgSFB_slice[(None, b_producer_state.count)]
                    tBsB_pipe = tBsB[(None, b_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, b_producer_state.index)]

                    tma_bar = b_pipeline.producer_get_barrier(b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # TMA load SFB
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
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
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
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

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

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
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
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

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Apply TMEM pointer offset hack when cta_tile_shape_n=192 or
                # cta_tile_shape_n=64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 case by two words
                    # (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

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

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            b_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                b_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

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
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
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
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            if cutlass.const_expr(self.generate_sfc):
                norm_const = norm_const_tensor[0]
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gSFC_mnl = cute.local_tile(mSFC_mnl, epi_tile, (None, None, None))

                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgSFC_mnl = thr_copy_t2r.partition_D(gSFC_mnl)
                tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrSFC = cute.make_rmem_tensor(
                    tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype
                )
                tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)

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
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
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
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                if cutlass.const_expr(self.generate_sfc):
                    # (T2R, T2R_M, T2R_N, RestM, RestN)
                    tCgSFC_mn = tCgSFC_mnl[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Process accumulator subtiles with SwiGLU fusion and store to global memory
                # Each iteration processes a pair of subtiles (up, gate) and computes
                # up * silu(gate)
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                for subtile_idx in cutlass.range(0, subtile_cnt, 2):
                    real_subtile_idx = subtile_idx // 2
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n_required
                                - 1
                                - subtile_idx // 2
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]
                    tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, real_subtile_idx * 2 + 1)]

                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx // 2 == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

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
                                (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]), (-LOG2_E, -LOG2_E)
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

                    if cutlass.const_expr(self.generate_sfc):
                        #
                        # Quantization path for Float4E2M1FN output:
                        # 1. Compute per-vector absolute max from SwiGLU result
                        # 2. Generate scale factor C (SFC) based on max values
                        # 3. Store SFC to global memory
                        # 4. Quantize output by scaling with reciprocal of SFC
                        #
                        # Assume subtile partitioned always happens on n dimension
                        sfc_subtile_idx_mn = (
                            tile_info[0] * self.epi_tile_cnt[0],
                            tile_info[1] * self.epi_tile_cnt[1] + real_subtile_idx,
                        )
                        tCgSFC = tCgSFC_mn[
                            (
                                None,
                                None,
                                None,
                                *sfc_subtile_idx_mn,
                            )
                        ]

                        #
                        # Get absolute max across a vector and Compute SFC
                        #
                        tTR_rAcc_frg = cute.logical_divide(
                            tCompute, cute.make_layout(self.sf_vec_size)
                        )
                        acc_frg = tTR_rAcc_frg.load()
                        acc_frg = epilogue_op(acc_frg)

                        # Apply element-wise absolute value using math.absf (supports vectors)
                        abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                        abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)

                        if cutlass.const_expr(self.vectorized_f32):
                            for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                    cute.ReductionOp.MAX,
                                    cutlass.Float32(0.0),
                                    0,  # Use 0.0 as init for abs values
                                )
                            for vi in cutlass.range_constexpr(0, abs_acc_frg.shape[1], 2):
                                tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                        (
                                            self.get_dtype_rcp_limits(self.c_dtype),
                                            self.get_dtype_rcp_limits(self.c_dtype),
                                        ),
                                    )
                                )
                                tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                        (norm_const, norm_const),
                                    )
                                )
                        else:
                            for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                tCrSFC_pvscale[vi] = (
                                    abs_acc_frg[None, vi].reduce(
                                        cute.ReductionOp.MAX,
                                        cutlass.Float32(0.0),
                                        0,  # Use 0.0 as init for abs values
                                    )
                                    * self.get_dtype_rcp_limits(self.c_dtype)
                                    * norm_const
                                )

                        # TODO: need to add f32x2 -> f8x2 conversion
                        tCrSFC.store(tCrSFC_pvscale.load().to(self.sf_dtype))

                        #
                        # Store SFC to global memory
                        #
                        # TODO: Need to think about predicate on it
                        # if cute.elem_less():
                        cute.autovec_copy(tCrSFC, tCgSFC)

                        #
                        # Compute quantized output values and convert to C type
                        #
                        # TODO: need to add f8x2 -> f32x2 conversion
                        tCrSFC_qpvscale_up = tCrSFC.load().to(cutlass.Float32)
                        fp32_max = cutlass.Float32(3.40282346638528859812e38)
                        if cutlass.const_expr(self.vectorized_f32):
                            for vi in cutlass.range_constexpr(0, cute.size(tCrSFC), 2):
                                acc_scale = cute.arch.mul_packed_f32x2(
                                    (
                                        cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi]),
                                        cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi + 1]),
                                    ),
                                    (norm_const, norm_const),
                                )
                                acc_scale_min0 = fmin(acc_scale[0], fp32_max, nan=True)
                                acc_scale_min1 = fmin(acc_scale[1], fp32_max, nan=True)

                                vec0 = tTR_rAcc_frg[None, vi]
                                vec1 = tTR_rAcc_frg[None, vi + 1]
                                for ei in cutlass.range_constexpr(self.sf_vec_size):
                                    vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                                        (vec0[ei], vec1[ei]),
                                        (acc_scale_min0, acc_scale_min1),
                                    )
                        else:
                            for vi in cutlass.range_constexpr(cute.size(tCrSFC)):
                                # TODO:Need to add E8M0 rcp approximation
                                acc_scale = norm_const * cute.arch.rcp_approx(
                                    tCrSFC_qpvscale_up[vi]
                                )
                                acc_scale = fmin(acc_scale, fp32_max, nan=True)

                                vec = tTR_rAcc_frg[None, vi]
                                for ei in cutlass.range_constexpr(self.sf_vec_size):
                                    vec[ei] = vec[ei] * acc_scale

                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                    else:
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
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.epilog_sync_barrier.arrive_and_wait()
                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
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
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
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
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

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
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
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
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
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
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

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

        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
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
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Calculates the reciprocal of the maximum absolute value for a given data type.

        :param dtype: Data type
        :type dtype: Type[cutlass.Numeric]

        :return: An float representing the reciprocal of the maximum absolute value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

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
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in (128, 256):
            is_valid = False
        # Skip invalid mma tile n
        # SwiGlu Fusion requires even epi_tile counts,
        # based on epi_tile_n = 64, only mma_tiler_n = 128 and 256 are supported
        if mma_tiler_mn[1] not in (128, 256):
            is_valid = False

        # Skip illegal cluster shape
        if (mma_tiler_mn[0] // cluster_shape_mn[0]) != 128:
            is_valid = False

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False

        # We only support cluster shape n = 1 for now
        # TODO: Support cluster shape n > 1
        if cluster_shape_mn[1] != 1:
            is_valid = False
        return is_valid

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
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
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
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False

        # Skip unsupported layouts
        if not cls.is_valid_layouts(ab_dtype, c_dtype, a_major, b_major, c_major):
            can_implement = False

        # Skip invalid mma tile shape and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(mma_tiler_mn, cluster_shape_mn):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip unsupported A/B layout
        if not (a_major == "k" and b_major == "k"):
            can_implement = False
        return can_implement

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        scale_k = k // scaling_vector_size
        interm_size = n // 2
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        a_sf = cute.make_tensor(
            a_sf_ptr, layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2))
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2))
        )
        c_sf = cute.make_tensor(
            c_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, interm_size // (scaling_vector_size * 4), l),
                order=(2, 1, 4, 0, 3, 5),
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
        global_sf = cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,)))

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            c_sf,
            global_sf,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
        )


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


@cute.jit
def cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
    sf_swizzled_tensor: cute.Tensor,
    sf_unswizzled_tensor: cute.Tensor,
):
    """Convert scale factor tensor from mma specification M(32x4xrest_m)xK(4xrest_k)xL layout to MKL layout"""
    # sf_swizzled_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 0, 3)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_unswizzled_tensor)):
        mkl_coord = sf_unswizzled_tensor.layout.get_hier_coord(i)
        sf_unswizzled_tensor[mkl_coord] = sf_swizzled_tensor[mkl_coord]
