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
"""SM107 dense block-scaled GEMM: the persistent kernels and their MXFP8 runners (Rubin).

Based on the kernels of ``tensorrt_llm/_torch/kernels/rubin/dense_blockscaled_gemm_persistent.py``
and on the SM107 block-scaled Linear runners of
``tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`` (unused code removed: the never-read SFB multicast
flag), renamed: ``RubinBlockScaledGemmRunner`` (the datatype-agnostic runner) and
``RubinMxfp8GemmRunner`` (MXFP8 with fine-grained low-M tuning, the one the family calls). As in
the engine, the epilogue completes its TMEM loads before its barriers publish them, TMEM is freed
only after every warp role has drained, and the runners refuse a split-K tactic the kernel cannot
run before they allocate or launch anything. This copy differs from the engine runners in three
ways:

* the runners write BF16 only, into an output they allocate;
* split-K runs into FP32 partial planes that :func:`reduce_split_k_fp32` sums in order and
  rounds once to BF16, instead of reduce-adding BF16-rounded partials in no fixed order;
* ``RubinMxfp8GemmRunner`` profiles only the tactics :func:`_is_kimi_k3_mxfp8_tactic_profiled`
  keeps.

The base kernel launches with programmatic dependent launch (PDL): its TMA warp waits for the
previous grid (``griddepcontrol.wait``) before its first load, and its MMA warp lets the next
kernel launch once every MMA of its CTA is issued (``griddepcontrol.launch_dependents``). The
results do not change. The mixed-cluster kernel runs the same two instructions but launches
without PDL, so its wait returns at once.

No torch op is registered: an OpWrapper builds a runner and calls it. The runners compile in
process (no on-disk cache) and keep kernel caches apart from the engine runners'. The engine
file's standalone CLI and emulated-dtype harness are left out.
"""

# ruff: noqa: E501,E731,E741

import dataclasses
import itertools
from typing import ClassVar, List, Literal, NamedTuple, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch.autotuner import (
    ConstraintSpec,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
    _clc_raster_n_has_launchable_grid,
    _get_cute_dsl_swap_ab_candidates,
    _get_kimi_k3_mxfp8_tuning_buckets,
    _kimi_k3_mxfp8_tuning_bucket,
    _launchable_raster_order,
    get_max_activate_clusters,
)
from tensorrt_llm._torch.cute_dsl_utils import reserved_max_active_clusters
from tensorrt_llm._torch.kernels.blackwell.dense_blockscaled_gemm_persistent import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)
from tensorrt_llm._torch.kernels.blackwell.utils import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    make_ptr,
)
from tensorrt_llm._torch.utils import mxfp8_scale_infer_shape
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up


@cute.jit
def _retire_tmem_loads(value: cute.TensorSSA) -> cute.TensorSSA:
    """Complete TMEM reads before the epilogue's barriers publish their data."""
    cute.arch.fence_view_async_tmem_load()
    return value


@cute.kernel
def _reduce_split_k_fp32_kernel(
    partials: cute.Tensor, output: cute.Tensor, split_k: cutlass.Constexpr
) -> None:
    idx = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
    numel = cute.size(output)
    if idx < numel:
        value = cutlass.Float32(0)
        for split in cutlass.range_constexpr(split_k):
            value = value + partials[idx + split * numel]
        output[idx] = value.to(cutlass.BFloat16)


@cute.jit
def reduce_split_k_fp32(
    partials: cute.Tensor,
    output: cute.Tensor,
    split_k: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Reduce contiguous FP32 split planes in order, then round once to BF16."""
    _reduce_split_k_fp32_kernel(partials, output, split_k).launch(
        grid=(cute.ceil_div(cute.size(output), 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


class S2TCopyBundle(NamedTuple):
    """Bundle of tiled copy and partitioned tensors for smem-to-tmem copies."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor  # Partitioned source (smem)
    tSF_compact: cute.Tensor  # Partitioned destination (tmem)


class Sm107BlockScaledPersistentDenseGemmKernel(Sm100BlockScaledPersistentDenseGemmKernel):
    """Persistent dense block scaled GEMM kernel for Rubin
    This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Rubin GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_inst_shape: Shape of the Matrix Multiply-Accumulate (MMA) instruction (M,N,K)
    :type mma_inst_shape: Tuple[int, int, int]
    :param mma_tiler: Shape of the Matrix Multiply-Accumulate (MMA) instruction (M,N,K)
    :type mma_tiler: Tuple[int, int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param swizzle_size: Swizzling size in the unit of cluster for improving L2 cache hit rate, defaults to 1
    :type swizzle_size: int
    :param raster_order: Rasterization order of clusters ('m' or 'n'), defaults to 'm'
    :type raster_order: Literal["m", "n"]
    :param split_k: Number of CTAs that partition K per output tile, defaults to 1. Above 1,
        each writes its own FP32 plane of C (batch extent L * split_k), and the caller reduces
        the planes into its final output.
    :type split_k: int

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF8xF4: A: Float8E5M2/Float8E4M3FN, B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4xF8: A: Float4E2M1FN, B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN/FloatNV8E5M3FNU + sf_vec_size: 16/32

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16

    :note: Constraints:
        - Mma tiler M must be 128, 256 or 512, MMA instruction shape M can be 128 or 256
        - Mma tiler N and MMA instruction shape N must be 64/128/192/256
        - B-reuse feature is enabled if (MMA tiler M // MMA instruction shape M) == 2
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors
        - FP4 operands require K-major layout (a_major="k" / b_major="k")
        - The contiguous dimension of A/B/C must be 16-byte aligned (a multiple of 16 Float8 or
          32 Float4 elements)
        - Split-K requires a linear epilogue because each K partition applies the
          epilogue before its partial plane is reduced

    Example:
        >>> gemm = Sm107BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_inst_shape=(128, 128, 128),
        ...     mma_tiler=(256, 128, 256),
        ...     cluster_shape_mn=(2, 1),
        ...     swizzle_size=1,
        ...     raster_order="m",
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        prefetch_dist: Union[int, None] = None,
        swizzle_size: int = 1,
        raster_order: Literal["m", "n"] = "m",
        scheduler_type: Type = None,
        split_k: int = 1,
    ):
        super().__init__(sf_vec_size, (mma_tiler[0], mma_tiler[1]), cluster_shape_mn)

        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)
        self.swizzle_size = swizzle_size
        self.raster_order = raster_order

        # utils.gemm.sm100.epilogue_tma_store reads these members from the kernel.
        self.epilogue_warp_id = self.epilog_warp_id
        self.epilog_sync_bar_id = self.epilog_sync_barrier.barrier_id

        # Bkeep-Breuse pattern is controlled by mma_inst_shape and mma_tiler
        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False

        # Prefetch configuration: None=auto (num_ab_stage), 0=disable, >0=explicit distance
        self.prefetch_dist_param = prefetch_dist

        if scheduler_type is None:
            scheduler_type = utils.StaticPersistentTileScheduler
        self.scheduler_type = scheduler_type
        self.use_clc_dynamic_scheduler = issubclass(
            self.scheduler_type, utils.ClcDynamicPersistentTileScheduler
        )
        self.sched_warp_id = 6
        self.num_clc_stage = 1 if self.use_clc_dynamic_scheduler else 0
        self.num_clc_response_bytes = 16
        if self.use_clc_dynamic_scheduler:
            self.threads_per_cta = self.threads_per_warp * (len(self.epilog_warp_id) + 3)

        # Each split computes a disjoint K range into its own FP32 output plane.
        if split_k < 1:
            raise ValueError(f"split_k must be >= 1, got {split_k}")
        self.split_k = split_k

    def _compute_grid(
        self,
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters,
        swizzle_size: int,
        raster_along_m: bool,
    ):
        """Compute static or CLC-dynamic persistent scheduler parameters."""
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        if self.use_clc_dynamic_scheduler:
            tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
                num_ctas_mnl, cluster_shape_mnl, swizzle_size, raster_along_m
            )
            grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        else:
            tile_sched_params = utils.PersistentTileSchedulerParams(
                num_ctas_mnl, cluster_shape_mnl, swizzle_size, raster_along_m
            )
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params, max_active_clusters
            )
        return tile_sched_params, grid

    # Override parent's wrapper to use Rubin-specific __call__ interface
    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,  # noqa: E741
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream: cuda.CUstream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        layouts: cutlass.Constexpr = None,
    ):
        """Executes the wrapped GEMM kernel with dynamically shaped tensors.

        This wrapper adapts the Blackwell-style interface to the Rubin kernel's
        __call__ method, which expects pointers, layouts, and problem dimensions.

        Args:
            m (cutlass.Int64): The M dimension of the GEMM problem.
            n (cutlass.Int64): The N dimension of the GEMM problem.
            k (cutlass.Int64): The K dimension of the GEMM problem.
            sf_m (cutlass.Int64): The M dimension of the scale factor tensor.
            sf_n (cutlass.Int64): The N dimension of the scale factor tensor.
            sf_k (cutlass.Int64): The K dimension of the scale factor tensor.
            l (cutlass.Constexpr): The batch dimension (L) of the GEMM problem.
            a_ptr (cute.Pointer): Pointer to the A tensor.
            b_ptr (cute.Pointer): Pointer to the B tensor.
            a_sf_ptr (cute.Pointer): Pointer to the scale factor tensor for A.
            b_sf_ptr (cute.Pointer): Pointer to the scale factor tensor for B.
            c_ptr (cute.Pointer): Pointer to the C tensor.
            alpha_tensor (cute.Tensor): Device tensor containing alpha scaling factor.
            max_active_clusters (cutlass.Constexpr): Maximum number of active clusters.
            current_stream (cuda.CUstream): CUDA stream for the operation.
            swap_ab (cutlass.Constexpr, optional): Whether to swap A and B. Defaults to False.
            epilogue_op (cutlass.Constexpr, optional): Elementwise lambda for epilogue.
                Must be linear when split_k is greater than one.
            layouts (cutlass.Constexpr, optional): A/B operand major modes and C layout.
                Defaults to the layouts selected by swap_ab.
        """
        if cutlass.const_expr(layouts is None):
            if cutlass.const_expr(swap_ab):
                c_layout = utils.LayoutEnum.COL_MAJOR
            else:
                c_layout = utils.LayoutEnum.ROW_MAJOR
            layouts = (OperandMajorMode.K, OperandMajorMode.K, c_layout)
        problem_mnkl = (cutlass.Int32(m), cutlass.Int32(n), cutlass.Int32(k), l)

        # Alpha scaling is applied inside the kernel epilogue (fused).
        # alpha_tensor is passed as an explicit kernel parameter to avoid
        # MLIR region isolation errors (same pattern as Blackwell kernel).
        self(
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            alpha_tensor,
            layouts,
            problem_mnkl,
            max_active_clusters,
            current_stream,
            epilogue_op,
        )

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])

        else:
            return (1, 1, 1)

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
        smem_capacity: int,
        occupancy: int,
        with_breuse: bool,
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
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        # Note that here we have assumed the kernel have access to all TMEM capacity
        # associated with sm_107 architecture.
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
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
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
        # Compute mma instruction shapes
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
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
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = sm107_utils.compute_epilogue_tile_shape(
            tiled_mma.op,
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.enable_breuse,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
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

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        # Each column entry in TMEM is 32-bit wide, and so we recast the TMEM layout
        # from its original data type to a 32-bit wide data type. Moreover, TMEM
        # addresses are expressed as (row << 16) | col, which in CUTE are expressed
        # as an affine transformation row * (1<<16) + col, which can be seen as a CUTE
        # layout of (row, col):(1<<16, 1). As a result, by masking out the upper 16 bits
        # (keeping only the lower 16 bits), we extract the cosize corresponding
        # to only the columns.
        self.num_sfa_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage * (2 if self.enable_breuse else 1)
        )

        # Set prefetch distance for both initial and rolling prefetch (unified control)
        # None = use num_ab_stage (default), 0 = disable prefetch, >0 = explicit distance
        if self.prefetch_dist_param is None:
            self.prefetch_dist = self.num_ab_stage
        else:
            self.prefetch_dist = self.prefetch_dist_param

        # Check if prefetch is enabled (prefetch_dist > 0)
        self.prefetch_enabled = self.prefetch_dist > 0

    def _is_interleaved_utccp(self) -> bool:
        # Enable interleaving UTCCP for Bkeep-Breuse case for 4xFP4 kernel
        return self.a_dtype.width == 4 and self.b_dtype.width == 4 and self.enable_breuse

    def _mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> S2TCopyBundle:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A named tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: S2TCopyBundle
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

        # This is a workaround, specifically needed for vector size 16 which also
        # works for other cases such as vector size 32. For 4x32dp128bit UTCCPs,
        # the lack of broadcasting mode in the source tensor makes the partitioned
        # layouts insufficient. As a workaround for non-swizzled shared memory layout,
        # it seems that adding the broadcasting mode in the pre-partitioned
        # tensor will lead to a better partitioned layout suitable for the destination
        # TMEM layout.
        def appendMNBroadcastMode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1]))
            layout = cute.append(cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator, appendMNBroadcastMode(tCsSF_compact.layout)
        )

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    def _mainloop_s2t_copies(
        self,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        s2t_stage_coord = (
            None,
            None,
            None,
            None,
            stage_idx,
        )

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfa_s2t_bundle.tSF_compact,
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )

    def _mainloop_s2t_interleaved_copies(
        self,
        k_block: int,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        # Two MMA atom along M-dimension -- fine grained control over
        # SFA and SFB
        #                       ┌─────┐
        #                       │ B0  │
        #                       ├─────┤
        #                       │ B1  │
        #                       └─────┘
        #     ┌─────┬─────┐     ┌─────┐
        #     │ A0  │ A2  │     │MMA0 │
        #     ├─────┼─────┤     ├─────┤
        #     │ A1  │ A3  │     │MMA1 │
        #     └─────┴─────┘     └─────┘
        # k_block 0 UTCCP SFA: A0 -> SFB: B0 -> SFA: A1 -> MMA0 & MMA1
        # k_block 1 UTCCP SFA: A2 -> SFB: B1 -> SFA: A3 -> MMA0 & MMA1

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        s_sfa_crd_keep = (None, 0, None, k_block, stage_idx)
        s_sfa_crd_reuse = (None, 1, None, k_block, stage_idx)
        s_sfb_crd = (None, None, None, k_block, stage_idx)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        t_sfa_crd_keep = (None, 0, None, k_block)
        t_sfa_crd_reuse = (None, 1, None, k_block)
        t_sfb_crd = (None, None, None, k_block)

        # SFA (A0/A2)
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_keep],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_keep],
        )

        # SFB (B0/B1)
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s_sfb_crd],
            sfb_s2t_bundle.tSF_compact[t_sfb_crd],
        )

        # SFA (A1/A3)
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_reuse],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_reuse],
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha: cute.Tensor,
        layouts: cutlass.Constexpr[Tuple[OperandMajorMode, OperandMajorMode, utils.LayoutEnum]],
        problem_mnkl: Tuple[int, int, int, int],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_ptr.value_type
        self.b_dtype: Type[cutlass.Numeric] = b_ptr.value_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_ptr.value_type
        self.c_dtype: Type[cutlass.Numeric] = c_ptr.value_type
        if cutlass.const_expr(self.split_k > 1 and self.c_dtype != cutlass.Float32):
            raise ValueError("Split-K partials require FP32 output storage")

        m, n, k, l = problem_mnkl
        c_l = l * self.split_k
        self.a_major_mode, self.b_major_mode, self.c_layout = layouts

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        a_layout = cute.make_ordered_layout((m, cute.assume(k, 32), l), order=(0, 1, 2))
        if cutlass.const_expr(self.a_major_mode == OperandMajorMode.K):
            a_layout = cute.make_ordered_layout((cute.assume(m, 32), k, l), order=(1, 0, 2))
        b_layout = cute.make_ordered_layout((n, cute.assume(k, 32), l), order=(0, 1, 2))
        if cutlass.const_expr(self.b_major_mode == OperandMajorMode.K):
            b_layout = cute.make_ordered_layout((cute.assume(n, 32), k, l), order=(1, 0, 2))
        # C is dense, with 64-bit strides.
        if cutlass.const_expr(self.c_layout == utils.LayoutEnum.ROW_MAJOR):
            c_ld = cutlass.Int64(n)
            c_out_layout = cute.make_layout(
                (m, cute.assume(n, 32), c_l), stride=(c_ld, 1, m * c_ld)
            )
        else:
            c_ld = cutlass.Int64(m)
            c_out_layout = cute.make_layout(
                (cute.assume(m, 32), n, c_l), stride=(1, c_ld, n * c_ld)
            )
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(c_ptr, c_out_layout)

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )

        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            # The 'FILL' collector operation indicates that B data should be kept
            # for reuse in subsequent operations.
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            # The 'LASTUSE' collector operation indicates that this is the last use
            # of the B data that was kept from the previous operation.
            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )

            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # This modifies the layout to handle overlapping 256x(# of scale factors for a single column of B (nNSF)) logical blocks for SFB when cta_tile_shape_n=192
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

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.swizzle_size,
            self.raster_order == "m",
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, self.num_clc_response_bytes // 4]
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
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            alpha,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,  # the TMA warp waits for the previous grid before it loads
        )
        return

    # GPU device kernel
    @cute.jit
    def kernel_impl(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params,
        epilogue_op: cutlass.Constexpr,
        cluster_shape_mn: Tuple[int, int],
        num_tma_producer: int,
        is_a_mcast: bool,
        is_b_mcast: bool,
        alpha: cute.Tensor,
    ):
        """
        GPU device kernel implementation performing the Persistent batched GEMM computation.
        """
        alpha_value = alpha[0].to(self.c_dtype)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
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

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
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

        # CLC dynamic scheduling publishes each cancelled cluster's tile
        # coordinate to the TMA, MMA, and epilogue warp roles in every CTA.
        clc_pipeline = None
        clc_response_ptr = None
        clc_consumer_state = None
        if cutlass.const_expr(self.use_clc_dynamic_scheduler):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            # Use the per-branch cluster shape. In the mixed-cluster kernel,
            # the fallback branch differs from self.cluster_shape_mn.
            cluster_size = cute.size(cluster_shape_mn)
            num_clc_consumer_threads = 32 * len(
                (
                    self.sched_warp_id,
                    *(
                        cluster_size
                        * (
                            self.tma_warp_id,
                            self.mma_warp_id,
                            *self.epilog_warp_id,
                        )
                    ),
                )
            )
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            )
            clc_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                num_stages=self.num_clc_stage,
                producer_group=clc_pipeline_producer_group,
                consumer_group=clc_pipeline_consumer_group,
                tx_count=self.num_clc_response_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_clc_stage
            )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(is_a_mcast or is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
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
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
        if cutlass.const_expr(self.split_k > 1):
            k_tile_cnt = k_tile_cnt // self.split_k

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
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
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMALDG_SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMALDG_SFB partition_S/D
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
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)
        # The allocator issues tcgen05.alloc only from its designated warp.
        # Initialize its bookkeeping before the warp roles diverge.
        tmem.allocate(self.num_tmem_alloc_cols)

        #
        # Construct the scheduler
        #
        if cutlass.const_expr(self.use_clc_dynamic_scheduler):
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
        else:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            # The loads below read what the previous grid wrote.
            griddepcontrol_wait()
            #
            # Persistent tile scheduling loop
            #
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                k_tile_start = 0
                if cutlass.const_expr(self.split_k > 1):
                    k_tile_start = cutlass.Int32(
                        (mma_tile_coord_mnl[2] % self.split_k) * k_tile_cnt
                    )
                    mma_tile_coord_mnl = (
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        mma_tile_coord_mnl[2] // self.split_k,
                    )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]

                # Apply SFB slicing hack when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                #
                # Prefetch: Initial batch of prefetches to prime the pipeline
                #
                if self.prefetch_enabled:
                    for pf_k_tile in cutlass.range(
                        0, min(self.prefetch_dist, k_tile_cnt), unroll=1
                    ):
                        prefetch_k = k_tile_start + pf_k_tile
                        cute.prefetch(tma_atom_a, tAgA_slice[(None, prefetch_k)])
                        cute.prefetch(tma_atom_b, tBgB_slice[(None, prefetch_k)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, prefetch_k)])
                        cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, prefetch_k)])

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    # TMA load A/B/SFA/SFB
                    load_k = k_tile_start + ab_producer_state.count
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, load_k)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, load_k)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, load_k)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, load_k)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Prefetch: Rolling prefetch for next tiles
                    if self.prefetch_enabled:
                        if k_tile < k_tile_cnt - self.prefetch_dist:
                            future_k_tile = load_k + self.prefetch_dist
                            cute.prefetch(
                                tma_atom_a,
                                tAgA_slice[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_b,
                                tBgB_slice[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, future_k_tile)],
                            )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Advance to next tile
                #
                if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            # The dynamic scheduler carries SSA state for the local response
            # buffer, so each specialized role creates its own view.
            if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                )
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            #
            # Partition for S2T copy of SFA/SFB
            #
            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
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
                if cutlass.const_expr(self.split_k > 1):
                    mma_tile_coord_mnl = (
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        mma_tile_coord_mnl[2] // self.split_k,
                    )

                # Get accumulator stage index
                acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # Apply TMEM pointer offset hack when cta_tile_shape_n=192 or 64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] in {64, 192}):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 or 64 case by two words (ignores first 64 columns of SFB)
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
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        if cutlass.const_expr(not self._is_interleaved_utccp()):
                            # Unless UTCCPs are to be interleaved, all SFA/SFB for all
                            # k_blocks are copied before MMA are executed
                            self._mainloop_s2t_copies(
                                ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle
                            )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (
                                    None,
                                    0,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                a_kblk_crd_reuse = (
                                    None,
                                    1,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                b_kblk_crd = (None, 0, k_block, ab_consumer_state.index)

                                sfa_kblk_crd_keep = (None, 0, k_block)
                                sfa_kblk_crd_reuse = (None, 1, k_block)
                                sfb_kblk_crd = (None, 0, k_block)

                                if cutlass.const_expr(self._is_interleaved_utccp()):
                                    self._mainloop_s2t_interleaved_copies(
                                        k_block,
                                        ab_consumer_state.index,
                                        sfa_s2t_bundle,
                                        sfb_s2t_bundle,
                                    )

                                # Keep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Reuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtSFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                kblk_crd = (
                                    None,
                                    None,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                sf_kblk_crd = (None, None, k_block)

                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    [tCrA[kblk_crd], tCtSFA[sf_kblk_crd]],
                                    [tCrB[kblk_crd], tCtSFB_mma[sf_kblk_crd]],
                                    tCtAcc,
                                )

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            # Every MMA of this CTA is issued: the next grid may launch.
            griddepcontrol_launch_dependents()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

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

            #
            # Persistent tile scheduling loop
            #
            work_tile = tile_sched.initial_work_tile_info()
            epi_tiles_executed = cutlass.Int32(0)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                epi_tiles_executed += 1

                store_op = lambda x: epilogue_op(alpha_value * _retire_tmem_loads(x))
                acc_consumer_state = utils.gemm.sm100.epilogue_tma_store(
                    self,
                    tidx,
                    warp_idx,
                    tma_atom_c,
                    tCtAcc,
                    sC,
                    tCgC,
                    epi_tile,
                    epi_tiles_executed,
                    store_op,
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    c_pipeline,
                )

            c_pipeline.producer_tail()

        #
        # Specialized scheduler warp (CLC dynamic only)
        #
        if cutlass.const_expr(self.use_clc_dynamic_scheduler):
            is_first_cta_in_cluster = cta_rank_in_cluster == 0
            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                clc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.ProducerConsumer,
                    self.num_clc_stage,
                )
                while work_tile.is_valid_tile:
                    clc_pipeline.producer_acquire(clc_producer_state)
                    mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                    tile_sched.advance_to_next_work(mbarrier_addr)
                    clc_producer_state.advance()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                clc_pipeline.producer_tail(clc_producer_state)

        # Join every warp role after its pipeline drains before the allocator
        # warp frees storage that the epilogue warps read asynchronously.
        cute.arch.sync_threads()
        if warp_idx < self.mma_warp_id:
            tmem.relinquish_alloc_permit()
            tmem.free(tmem.retrieve_ptr(self.acc_dtype))

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        self.kernel_impl(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tma_atom_sfa,
            mSFA_mkl,
            tma_atom_sfb,
            mSFB_nkl,
            tma_atom_c,
            mC_mnl,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            tCtSFA_layout,
            tCtSFB_layout,
            c_smem_layout_staged,
            epi_tile,
            tile_sched_params,
            epilogue_op,
            self.cluster_shape_mn,
            self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1,
            self.is_a_mcast,
            self.is_b_mcast,
            alpha,
        )

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ):
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :raises testing.CantImplementError: If data types and/or scale factors are invalid
        """

        # Check valid
        # Supported combinations of (a_dtype, b_dtype, sf_dtype, sf_vec_size)
        valid_combinations = {
            # 4xFP4
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 16),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 32),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.FloatNV8E5M3FNU, 16),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.FloatNV8E5M3FNU, 32),
            # 2xFP8
            (cutlass.Float8E5M2, cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E4M3FN, cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
            # 2xFP8xFP4
            (cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E5M2, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
            # 2xFP4xFP8
            (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float4E2M1FN, cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
        }

        # Check if the current combination is valid
        current_combination = (a_dtype, b_dtype, sf_dtype, sf_vec_size)
        if current_combination not in valid_combinations:
            raise testing.CantImplementError(
                f"Unsupported combination of data types and scale factor vector size: "
                f"a_dtype={a_dtype}, b_dtype={b_dtype}, sf_dtype={sf_dtype}, sf_vec_size={sf_vec_size}. "
                f"Please refer to the supported combinations in the function documentation."
            )

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
        }:
            raise testing.CantImplementError(f"Unsupported output data type: {c_dtype}")

    @staticmethod
    def is_valid_layouts(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: Literal["m", "k"],
        b_major: Literal["n", "k"],
        c_major: Literal["m", "n"],
    ):
        """
        Check if layouts and dtypes are valid combinations

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: Literal["m", "k"]
        :param b_major: The major dimension of the B tensor
        :type b_major: Literal["n", "k"]
        :param c_major: The major dimension of the C tensor
        :type c_major: Literal["m", "n"]

        :raises testing.CantImplementError if invalid input/output layouts
        """

        if a_dtype is cutlass.Float4E2M1FN and a_major != "k":
            raise testing.CantImplementError(
                f"FP4 operand A requires K-major layout, got a_major={a_major}"
            )
        if b_dtype is cutlass.Float4E2M1FN and b_major != "k":
            raise testing.CantImplementError(
                f"FP4 operand B requires K-major layout, got b_major={b_major}"
            )
        # FP4 output is N-major only.
        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            raise testing.CantImplementError(f"Unsupported output layout: {c_major}")

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        """
        Check if the mma tiler and cluster shape are valid

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param mma_inst_shape: The (M, N, K) shape of the MMA instruction
        :type mma_inst_shape: Tuple[int, int, int]
        :param mma_tiler: The (M, N, K) shape of the MMA tiler
        :type mma_tiler: Tuple[int, int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :raises testing.CantImplementError: If mma tiler or cluster shapes are invalid
        """

        # Skip invalid mma tile shape
        if mma_inst_shape[0] not in [128, 256]:
            raise testing.CantImplementError(f"Invalid mma_inst_shape_m: {mma_inst_shape[0]}")
        if mma_inst_shape[1] not in [64, 128, 192, 256]:
            raise testing.CantImplementError(f"Invalid mma_inst_shape_n: {mma_inst_shape[1]}")
        if mma_tiler[0] not in [128, 256, 512]:
            raise testing.CantImplementError(f"Invalid mma_tiler_m: {mma_tiler[0]}")
        if mma_tiler[1] not in [64, 128, 192, 256]:
            raise testing.CantImplementError(f"Invalid mma_tiler_n: {mma_tiler[1]}")

        # Checking for valid MMA tilers versus MMA instructions.
        b_reuse = mma_tiler[0] // mma_inst_shape[0] == 2
        if mma_tiler[0] != mma_inst_shape[0] and not b_reuse:
            raise testing.CantImplementError(
                f"Unsupported M-mode for the MMA tiler/instruction shape. "
                f"mma_tiler: {mma_tiler}, mma_inst_shape: {mma_inst_shape}"
            )
        if mma_tiler[1] != mma_inst_shape[1]:
            raise testing.CantImplementError(
                f"Unsupported N-mode for the MMA tiler/instruction shape. "
                f"mma_tiler: {mma_tiler}, mma_inst_shape: {mma_inst_shape}"
            )

        # MXF8F6F4 blockscaled kernels (FP8xFP8, FP8xFP4, FP4xFP8): mma_tiler_k=128, mma_inst_shape_k=64
        _is_mxf8f6f4 = (
            a_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2, cutlass.Float4E2M1FN}
            and b_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2, cutlass.Float4E2M1FN}
            and not (a_dtype is cutlass.Float4E2M1FN and b_dtype is cutlass.Float4E2M1FN)
        )
        if _is_mxf8f6f4:
            if mma_tiler[2] != 128 or mma_inst_shape[2] != 64:
                raise testing.CantImplementError(
                    f"Unsupported K-mode for the MMA tiler/instruction shape. "
                    f"mma_tiler: {mma_tiler}, mma_inst_shape: {mma_inst_shape}"
                )
        else:
            # 4xFP4 blockscaled kernels only support mma_tiler_k=256, mma_inst_shape_k=128
            if mma_tiler[2] != 256 or mma_inst_shape[2] != 128:
                raise testing.CantImplementError(
                    f"Unsupported K-mode for the MMA tiler/instruction shape. "
                    f"mma_tiler: {mma_tiler}, mma_inst_shape: {mma_inst_shape}"
                )

        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_inst_shape[0] == 256 else 1) != 0:
            raise testing.CantImplementError(
                f"Invalid cluster shape for a 2CTA MMA, cluster_shape_m: {cluster_shape_mn[0]}"
            )
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
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
            raise testing.CantImplementError(
                f"Unsupported cluster shape: ({cluster_shape_mn[0]}, {cluster_shape_mn[1]})"
            )

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: Literal["m", "k"],
        b_major: Literal["n", "k"],
        c_major: Literal["m", "n"],
    ):
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: Literal["m", "k"]
        :param b_major: The major axis of the B tensor
        :type b_major: Literal["n", "k"]
        :param c_major: The major axis of the C tensor
        :type c_major: Literal["m", "n"]

        :raises testing.CantImplementError: If misaligned tensors.
        """

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(b_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            raise testing.CantImplementError("Invalid tensor alignment")

    @staticmethod
    def can_implement(
        mnkl: Tuple[int, int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: Literal["m", "k"],
        b_major: Literal["n", "k"],
        c_major: Literal["m", "n"],
        sf_vec_size: int,
        mma_tiler: Tuple[int, int, int],
        mma_inst_shape: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        split_k: int = 1,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param mnkl: The problem size as a tuple (M, N, K, L).
        :type mnkl: Tuple[int, int, int, int]
        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: Literal["m", "k"]
        :param b_major: The major axis of the B tensor
        :type b_major: Literal["n", "k"]
        :param c_major: The major axis of the C tensor
        :type c_major: Literal["m", "n"]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler: The (M, N, K) shape of the MMA tiler
        :type mma_tiler: Tuple[int, int, int]
        :param mma_inst_shape: The (M, N, K) shape of the MMA instruction
        :type mma_inst_shape: Tuple[int, int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param split_k: Number of CTAs that partition the K dimension per output tile
        :type split_k: int
        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """

        try:
            # Skip unsupported types
            Sm107BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
                a_dtype, b_dtype, sf_dtype, sf_vec_size, c_dtype
            )
            # Skip unsupported layouts
            Sm107BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
                a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
            )
            # Skip invalid mma tile shape and cluster shape
            Sm107BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
                a_dtype, b_dtype, mma_inst_shape, mma_tiler, cluster_shape_mn
            )
            # Skip illegal problem shape for load/store alignment
            m, n, k, l = mnkl
            Sm107BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
                m, n, k, l, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
            )
            if split_k < 1:
                raise testing.CantImplementError(f"split_k must be >= 1, got {split_k}")
            if split_k > 1:
                num_k_tiles = k // mma_tiler[2]
                if l != 1 or k % mma_tiler[2] != 0 or num_k_tiles % split_k != 0:
                    raise testing.CantImplementError(
                        f"split_k={split_k} requires L==1 and K tiles "
                        f"({num_k_tiles}) divisible by split_k"
                    )
            # The grid uses a half-sized per-CTA M tile for 2-CTA MMA.
            # Requiring enough real M tiles avoids phantom cluster peers, which
            # can issue out-of-bounds TMA accesses for skinny-M problems.
            cta_tile_m = mma_tiler[0] // (2 if mma_inst_shape[0] == 256 else 1)
            ctas_m = (m + cta_tile_m - 1) // cta_tile_m
            if ctas_m < cluster_shape_mn[0]:
                return False
        except testing.CantImplementError:
            return False
        return True


# Mixed-cluster variant kept in this file with the base Rubin NVFP4 kernel.
class Sm107BlockScaledPersistentDenseGemmMixedClustersKernel(
    Sm107BlockScaledPersistentDenseGemmKernel
):
    """Persistent dense block scaled GEMM kernel for Rubin
    This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Rubin GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_inst_shape: Shape of the Matrix Multiply-Accumulate (MMA) instruction (M,N,K)
    :type mma_inst_shape: Tuple[int, int, int]
    :param mma_tiler: Shape of the Matrix Multiply-Accumulate (MMA) instruction (M,N,K)
    :type mma_tiler: Tuple[int, int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param swizzle_size: Swizzling size in units of clusters, defaults to 1
    :type swizzle_size: int
    :param raster_order: Rasterization order of clusters ('m' or 'n'), defaults to 'm'
    :type raster_order: Literal["m", "n"]
    :param scheduler_type: Tile scheduler class; StaticPersistentTileScheduler
        (default) or ClcDynamicPersistentTileScheduler
    :type scheduler_type: Type

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN/FloatNV8E5M3FNU + sf_vec_size: 16/32

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16

    :note: Constraints:
        - Mma tiler M must be 128, 256 or 512, MMA instruction shape M can be 128 or 256
        - Mma tiler N and MMA instruction shape N must be 64/128/192/256
        - B-reuse feature is enabled if (MMA tiler M // MMA instruction shape M) == 2
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> gemm = Sm107BlockScaledPersistentDenseGemmMixedClustersKernel(
        ...     sf_vec_size=16,
        ...     mma_inst_shape=(128, 128, 128),
        ...     mma_tiler=(256, 128, 256),
        ...     preferred_cluster_shape_mn=(4, 2),
        ...     fallback_cluster_shape_mn=(2, 1),
        ... )
        >>> gemm = cute.compile(
        ...     a_ptr,
        ...     b_ptr,
        ...     sfa_ptr,
        ...     sfb_ptr,
        ...     c_ptr,
        ...     layouts,
        ...     problem_mnkl,
        ...     preferred_max_cluster_size,
        ...     fallback_max_cluster_size,
        ...     stream,
        ...     epilogue_op,
        ... )
        >>> gemm(
        ...     a_tensor,
        ...     b_tensor,
        ...     sfa_tensor,
        ...     sfb_tensor,
        ...     c_tensor,
        ...     preferred_max_active_clusters,
        ...     fallback_max_active_clusters,
        ...     stream,
        ... )
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
        prefetch_dist: Union[int, None] = None,
        swizzle_size: int = 1,
        raster_order: Literal["m", "n"] = "m",
        scheduler_type: Type = None,
    ):
        super().__init__(
            sf_vec_size,
            mma_inst_shape,
            mma_tiler,
            preferred_cluster_shape_mn,
            prefetch_dist,
            swizzle_size=swizzle_size,
            raster_order=raster_order,
            scheduler_type=scheduler_type,
        )

        # Providing explicit cluster shapes for preferred and fallback
        self.preferred_cluster_shape_mn = preferred_cluster_shape_mn
        self.fallback_cluster_shape_mn = fallback_cluster_shape_mn

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
        # Compute mma instruction shapes
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.preferred_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.preferred_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.fallback_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.fallback_cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.fallback_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.fallback_cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_preferred_mcast_ctas_a = cute.size(self.preferred_cluster_layout_vmnk.shape[2])
        self.num_preferred_mcast_ctas_b = cute.size(self.preferred_cluster_layout_vmnk.shape[1])
        self.is_preferred_a_mcast = self.num_preferred_mcast_ctas_a > 1
        self.is_preferred_b_mcast = self.num_preferred_mcast_ctas_b > 1

        self.num_fallback_mcast_ctas_a = cute.size(self.fallback_cluster_layout_vmnk.shape[2])
        self.num_fallback_mcast_ctas_b = cute.size(self.fallback_cluster_layout_vmnk.shape[1])
        self.is_fallback_a_mcast = self.num_fallback_mcast_ctas_a > 1
        self.is_fallback_b_mcast = self.num_fallback_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = sm107_utils.compute_epilogue_tile_shape(
            tiled_mma.op,
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.enable_breuse,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
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

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        # Each column entry in TMEM is 32-bit wide, and so we recast the TMEM layout
        # from its original data type to a 32-bit wide data type. Moreover, TMEM
        # addresses are expressed as (row << 16) | col, which in CUTE are expressed
        # as an affine transformation row * (1<<16) + col, which can be seen as a CUTE
        # layout of (row, col):(1<<16, 1). As a result, by masking out the upper 16 bits
        # (keeping only the lower 16 bits), we extract the cosize corresponding
        # to only the columns.
        self.num_sfa_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage * (2 if self.enable_breuse else 1)
        )

        # Set prefetch distance for both initial and rolling prefetch (unified control)
        # None = use num_ab_stage (default), 0 = disable prefetch, >0 = explicit distance
        if self.prefetch_dist_param is None:
            self.prefetch_dist = self.num_ab_stage
        else:
            self.prefetch_dist = self.prefetch_dist_param

        # Check if prefetch is enabled (prefetch_dist > 0)
        self.prefetch_enabled = self.prefetch_dist > 0

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        layouts: cutlass.Constexpr[Tuple[OperandMajorMode, OperandMajorMode, utils.LayoutEnum]],
        problem_mnkl: Tuple[int, int, int, int],
        preferred_max_active_clusters: cutlass.Constexpr,
        fallback_max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        alpha: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param preferred_max_active_clusters: Maximum number of preferred active clusters
        :type preferred_max_active_clusters: cutlass.Constexpr
        :param fallback_max_active_clusters: Maximum number of fallback active clusters
        :type fallback_max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_ptr.value_type
        self.b_dtype: Type[cutlass.Numeric] = b_ptr.value_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_ptr.value_type
        self.c_dtype: Type[cutlass.Numeric] = c_ptr.value_type

        m, n, k, l = problem_mnkl  # noqa: E741
        self.a_major_mode, self.b_major_mode, self.c_layout = layouts

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        a_layout = cute.make_ordered_layout((m, cute.assume(k, 32), l), order=(0, 1, 2))
        if cutlass.const_expr(self.a_major_mode == OperandMajorMode.K):
            a_layout = cute.make_ordered_layout((cute.assume(m, 32), k, l), order=(1, 0, 2))
        b_layout = cute.make_ordered_layout((n, cute.assume(k, 32), l), order=(0, 1, 2))
        if cutlass.const_expr(self.b_major_mode == OperandMajorMode.K):
            b_layout = cute.make_ordered_layout((cute.assume(n, 32), k, l), order=(1, 0, 2))
        # C is dense.
        if cutlass.const_expr(self.c_layout == utils.LayoutEnum.ROW_MAJOR):
            c_layout = cute.make_layout((m, cute.assume(n, 32), l), stride=(n, 1, m * n))
        else:
            c_layout = cute.make_layout((cute.assume(m, 32), n, l), stride=(1, m, n * m))
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(c_ptr, c_layout)

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )

        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            # The 'FILL' collector operation indicates that B data should be kept
            # for reuse in subsequent operations.
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            # The 'LASTUSE' collector operation indicates that this is the last use
            # of the B data that was kept from the previous operation.
            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )

            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(
                self.preferred_cluster_shape_mn, tiled_mma.thr_id
            ),
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
        )
        tma_atom_a_fallback, tma_tensor_a_fallback = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            ),
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(
                self.preferred_cluster_shape_mn, tiled_mma.thr_id
            ),
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
        )

        tma_atom_b_fallback, tma_tensor_b_fallback = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            ),
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(
                self.preferred_cluster_shape_mn, tiled_mma.thr_id
            ),
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        tma_atom_sfa_fallback, tma_tensor_sfa_fallback = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            ),
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.preferred_cluster_shape_mn, tiled_mma.thr_id
            ),
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.preferred_cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        tma_atom_sfb_fallback, tma_tensor_sfb_fallback = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            ),
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.fallback_cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # This modifies the layout to handle overlapping 256x(# of scale
        # factors for a single column of B (nNSF)) logical blocks for SFB
        # when cta_tile_shape_n=192
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

            # A flexible launch may execute either branch, so the fallback SFB
            # tensor needs the same N=192 logical-block remapping.
            fallback_x = tma_tensor_sfb_fallback.stride[0][1]
            fallback_y = cute.ceil_div(tma_tensor_sfb_fallback.shape[0][1], 4)
            fallback_shape = (
                (
                    tma_tensor_sfb_fallback.shape[0][0],
                    ((2, 2), fallback_y),
                ),
                tma_tensor_sfb_fallback.shape[1],
                tma_tensor_sfb_fallback.shape[2],
            )
            fallback_stride = (
                (
                    tma_tensor_sfb_fallback.stride[0][0],
                    ((fallback_x, fallback_x), 3 * fallback_x),
                ),
                tma_tensor_sfb_fallback.stride[1],
                tma_tensor_sfb_fallback.stride[2],
            )
            fallback_layout = cute.make_layout(fallback_shape, stride=fallback_stride)
            tma_tensor_sfb_fallback = cute.make_tensor(
                tma_tensor_sfb_fallback.iterator, fallback_layout
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # Static scheduling uses one persistent tile partition. CLC dynamic
        # scheduling needs branch-local parameters so cancelled cluster IDs are
        # decoded using the cluster shape selected by the flexible launch.
        (
            self.tile_sched_params,
            self.fallback_tile_sched_params,
            grid,
        ) = self._compute_mixed_cluster_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.preferred_cluster_shape_mn,
            self.fallback_cluster_shape_mn,
            preferred_max_active_clusters,
            fallback_max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # CLC hardware writes a 128-bit opaque response into this buffer.
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, self.num_clc_response_bytes // 4]
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
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.mixed_cluster_kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            (tma_atom_a, tma_atom_a_fallback),
            (tma_tensor_a, tma_tensor_a_fallback),
            (tma_atom_b, tma_atom_b_fallback),
            (tma_tensor_b, tma_tensor_b_fallback),
            (tma_atom_sfa, tma_atom_sfa_fallback),
            (tma_tensor_sfa, tma_tensor_sfa_fallback),
            (tma_atom_sfb, tma_atom_sfb_fallback),
            (tma_tensor_sfb, tma_tensor_sfb_fallback),
            tma_atom_c,
            tma_tensor_c,
            (self.preferred_cluster_layout_vmnk, self.fallback_cluster_layout_vmnk),
            (
                self.preferred_cluster_layout_sfb_vmnk,
                self.fallback_cluster_layout_sfb_vmnk,
            ),
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            self.fallback_tile_sched_params,
            alpha,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.preferred_cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            smem_merge_branch_allocs=True,
        )
        return

    # GPU device kernel with preferred & fallback cluster sizes.
    @cute.kernel
    def mixed_cluster_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: Tuple[cute.CopyAtom, cute.CopyAtom],
        mA_mkl: Tuple[cute.Tensor, cute.Tensor],
        tma_atom_b: Tuple[cute.CopyAtom, cute.CopyAtom],
        mB_nkl: Tuple[cute.Tensor, cute.Tensor],
        tma_atom_sfa: Tuple[cute.CopyAtom, cute.CopyAtom],
        mSFA_mkl: Tuple[cute.Tensor, cute.Tensor],
        tma_atom_sfb: Tuple[cute.CopyAtom, cute.CopyAtom],
        mSFB_nkl: Tuple[cute.Tensor, cute.Tensor],
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: Tuple[cute.Layout, cute.Layout],
        cluster_layout_sfb_vmnk: Tuple[cute.Layout, cute.Layout],
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params,
        fallback_tile_sched_params,
        alpha: cute.Tensor,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.

        This kernel implements the flexible CGA feature, meaning that it tries to launch with
        the preferred_cluster_shape_mn as its priority, but if at runtime, there is not enough
        resources available to do so, it then uses the fallback_cluster_shape_mn
        """

        # Get cluster coordinates to determine if this is a preferred cluster
        cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
        is_preferred_cluster = (
            cbdim_x == self.preferred_cluster_shape_mn[0]
            and cbdim_y == self.preferred_cluster_shape_mn[1]
            and cbdim_z == 1
        )

        # mega-kernel approach has 2 mutually exclusive code branches, only one path runs per launch,
        # specify `smem_merge_branch_allocs=True` at launch to enables shared memory reuse between two paths
        if is_preferred_cluster:
            self.kernel_impl(
                tiled_mma,
                tiled_mma_bkeep,
                tiled_mma_breuse,
                tiled_mma_sfb,
                tma_atom_a[0],
                mA_mkl[0],
                tma_atom_b[0],
                mB_nkl[0],
                tma_atom_sfa[0],
                mSFA_mkl[0],
                tma_atom_sfb[0],
                mSFB_nkl[0],
                tma_atom_c,
                mC_mnl,
                cluster_layout_vmnk[0],
                cluster_layout_sfb_vmnk[0],
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                tCtSFA_layout,
                tCtSFB_layout,
                c_smem_layout_staged,
                epi_tile,
                tile_sched_params,
                epilogue_op,
                self.preferred_cluster_shape_mn,
                self.num_preferred_mcast_ctas_a + self.num_preferred_mcast_ctas_b - 1,
                self.is_preferred_a_mcast,
                self.is_preferred_b_mcast,
                alpha,
            )
        else:
            self.kernel_impl(
                tiled_mma,
                tiled_mma_bkeep,
                tiled_mma_breuse,
                tiled_mma_sfb,
                tma_atom_a[1],
                mA_mkl[1],
                tma_atom_b[1],
                mB_nkl[1],
                tma_atom_sfa[1],
                mSFA_mkl[1],
                tma_atom_sfb[1],
                mSFB_nkl[1],
                tma_atom_c,
                mC_mnl,
                cluster_layout_vmnk[1],
                cluster_layout_sfb_vmnk[1],
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                tCtSFA_layout,
                tCtSFB_layout,
                c_smem_layout_staged,
                epi_tile,
                fallback_tile_sched_params,
                epilogue_op,
                self.fallback_cluster_shape_mn,
                self.num_fallback_mcast_ctas_a + self.num_fallback_mcast_ctas_b - 1,
                self.is_fallback_a_mcast,
                self.is_fallback_b_mcast,
                alpha,
            )

    def _compute_mixed_cluster_grid(
        self,
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
        preferred_max_active_clusters: cutlass.Constexpr,
        fallback_max_active_clusters: cutlass.Constexpr,
    ):
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param preferred_cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type preferred_cluster_shape_mn: tuple[int, int]
        :param fallback_cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type fallback_cluster_shape_mn: tuple[int, int]
        :param preferred_max_active_clusters: Maximum number of preferred active clusters.
        :type preferred_max_active_clusters: cutlass.Constexpr
        :param fallback_max_active_clusters: Maximum number of fallback active clusters.
        :type fallback_max_active_clusters: cutlass.Constexpr
        :return: A tuple containing:
            - preferred_tile_sched_params: Parameters for the preferred persistent tile scheduler.
            - fallback_tile_sched_params: Parameters for the fallback persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape

        # With CLC scheduling the launch covers the full preferred-cluster tile
        # grid. Running clusters steal pending work. Each flexible-launch branch
        # must decode CLC responses using its own runtime cluster geometry.
        raster_along_m = self.raster_order == "m"
        if self.use_clc_dynamic_scheduler:
            preferred_tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
                num_ctas_mnl,
                (*preferred_cluster_shape_mn, 1),
                self.swizzle_size,
                raster_along_m,
            )
            fallback_tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
                num_ctas_mnl,
                (*fallback_cluster_shape_mn, 1),
                self.swizzle_size,
                raster_along_m,
            )
            grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(
                preferred_tile_sched_params
            )
            return (
                preferred_tile_sched_params,
                fallback_tile_sched_params,
                grid,
            )

        # Note that the grid calculation here is only valid for a static persistent
        # tile scheduler.

        # Tile scheduler and grid shape for the preferred cluster
        preferred_cluster_shape_mnl = (*preferred_cluster_shape_mn, 1)

        preferred_tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            preferred_cluster_shape_mnl,
            self.swizzle_size,
            raster_along_m,
        )
        preferred_grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            preferred_tile_sched_params, preferred_max_active_clusters
        )

        # Tile scheduler and grid shape for the fallback cluster
        fallback_cluster_shape_mnl = (*fallback_cluster_shape_mn, 1)

        fallback_tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            fallback_cluster_shape_mnl,
            self.swizzle_size,
            raster_along_m,
        )
        fallback_grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            fallback_tile_sched_params, fallback_max_active_clusters
        )

        # Align preferred grid to cluster shape
        preferred_grid = cute.round_up(preferred_grid, preferred_cluster_shape_mnl)

        # Compute max preferred clusters: total blocks <= fallback total,
        # and is multiple of cluster size
        preferred_cluster_size_mn = preferred_cluster_shape_mn[0] * preferred_cluster_shape_mn[1]
        max_ctas_for_fallback_cluster = fallback_grid[0] * fallback_grid[1] * fallback_grid[2]
        # Use floor division (not ceil_div) to compute max preferred cluster count.
        # The preferred cluster total CTA count must not exceed the fallback total,
        # otherwise when the division is not exact, the extra partial wave of
        # preferred clusters may force the hardware to schedule one additional wave,
        # causing significant performance regression.
        max_preferred_cluster_count = max_ctas_for_fallback_cluster // preferred_cluster_size_mn
        preferred_grid = (
            preferred_grid[0],
            preferred_grid[1],
            max_preferred_cluster_count,
        )

        # Static scheduling uses the preferred params in both branches because
        # tile assignment is CTA-linear over the shared launch grid.
        return (
            preferred_tile_sched_params,
            preferred_tile_sched_params,
            preferred_grid,
        )

    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,  # noqa: E741
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        preferred_max_active_clusters: cutlass.Constexpr,
        fallback_max_active_clusters: cutlass.Constexpr,
        current_stream: cuda.CUstream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Wrapped entrypoint for TVM-FFI/raw-pointer callers."""
        a_major_mode = OperandMajorMode.K
        b_major_mode = OperandMajorMode.K
        if cutlass.const_expr(swap_ab):
            c_layout = utils.LayoutEnum.COL_MAJOR
        else:
            c_layout = utils.LayoutEnum.ROW_MAJOR

        self(
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            (a_major_mode, b_major_mode, c_layout),
            (m, n, k, l),
            preferred_max_active_clusters,
            fallback_max_active_clusters,
            current_stream,
            alpha_tensor,
            epilogue_op,
        )

    @staticmethod
    def can_implement(
        mnkl: Tuple[int, int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: Literal["m", "k"],
        b_major: Literal["n", "k"],
        c_major: Literal["m", "n"],
        sf_vec_size: int,
        mma_tiler: Tuple[int, int, int],
        mma_inst_shape: Tuple[int, int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param mnkl: The problem size as a tuple (M, N, K, L).
        :type mnkl: Tuple[int, int, int, int]
        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: Literal["m", "k"]
        :param b_major: The major axis of the B tensor
        :type b_major: Literal["n", "k"]
        :param c_major: The major axis of the C tensor
        :type c_major: Literal["m", "n"]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler: The (M, N, K) shape of the MMA tiler
        :type mma_tiler: Tuple[int, int, int]
        :param mma_inst_shape: The (M, N, K) shape of the MMA instruction
        :type mma_inst_shape: Tuple[int, int, int]
        :param preferred_cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type preferred_cluster_shape_mn: Tuple[int, int]
        :param fallback_cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type fallback_cluster_shape_mn: Tuple[int, int]
        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """

        try:
            # Most can_implement rules are the same with the base kernel
            # (with preferred_cluster_shape_mn as its cluster shape)
            if not Sm107BlockScaledPersistentDenseGemmKernel.can_implement(
                mnkl,
                a_dtype,
                b_dtype,
                sf_dtype,
                c_dtype,
                a_major,
                b_major,
                c_major,
                sf_vec_size,
                mma_tiler,
                mma_inst_shape,
                preferred_cluster_shape_mn,
            ):
                return False

            if fallback_cluster_shape_mn[0] % (2 if mma_inst_shape[0] == 256 else 1) != 0:
                raise testing.CantImplementError(
                    f"Invalid fallback cluster shape for a 2CTA MMA,"
                    f" fallback_cluster_shape_m: {fallback_cluster_shape_mn[0]}"
                )

            # Check preferred is multiple of fallback
            if (
                preferred_cluster_shape_mn[0] % fallback_cluster_shape_mn[0] != 0
                or preferred_cluster_shape_mn[1] % fallback_cluster_shape_mn[1] != 0
            ):
                raise testing.CantImplementError(
                    f"Preferred cluster shape {preferred_cluster_shape_mn} must be "
                    f"integer multiple of fallback cluster shape {fallback_cluster_shape_mn}"
                )

            # Check problem size is at least as large as the preferred CGA tile size.
            # The mixed clusters kernel computes max_preferred_cluster_count as:
            #   max_ctas_for_fallback_cluster // preferred_cluster_size_mn
            # If the problem is smaller than one preferred CGA tile, this count
            # becomes zero, resulting in an invalid grid shape.
            m, n, k, l = mnkl  # noqa: E741
            # A 2-CTA MMA tiler spans the CTA pair, while the scheduler tile is
            # the portion owned by one CTA.
            cta_tile_m = mma_tiler[0] // (2 if mma_inst_shape[0] == 256 else 1)
            cga_tile_m = cta_tile_m * preferred_cluster_shape_mn[0]
            cga_tile_n = mma_tiler[1] * preferred_cluster_shape_mn[1]
            if m < cga_tile_m or n < cga_tile_n:
                raise testing.CantImplementError(
                    f"Problem size ({m}, {n}) is smaller than the CGA tile size "
                    f"({cga_tile_m}, {cga_tile_n})"
                )
        except testing.CantImplementError:
            return False
        return True


# ---------------------------------------------------------------------------
# Runners: the SM107 block-scaled Linear runners, autotuned per (M bucket, N, K).
# ---------------------------------------------------------------------------


def _reserved_scheduler(scheduler_mode: str, reserve_sms: int) -> str:
    """The persistent scheduler of a launch that keeps ``reserve_sms`` SMs out of its grid: the
    static one, whose grid the active-cluster count bounds (a CLC-dynamic grid covers every tile).
    Both run each output tile, and each split-K slice of it, on one CTA through the tactic's K
    loop, so the output is the same."""
    return "static" if reserve_sms > 0 else scheduler_mode


def _reserved_clusters(max_active_clusters: int, cluster_size: int, reserve_sms: int) -> int:
    """``max_active_clusters`` less the clusters that free ``reserve_sms`` SMs, never below one: the
    rule ``TRTLLM_CUTEDSL_SM_RESERVE`` applies to every grid, here for one launch."""
    return reserved_max_active_clusters(max_active_clusters, cluster_size, reserve_sms)


# Largest tuning M at which the K3 MXFP8 runner profiles split-K tactics.
_KIMI_K3_MXFP8_MAX_SPLIT_K_M = 256


def _is_kimi_k3_mxfp8_tactic_profiled(tactic: tuple, m: int) -> bool:
    """Whether the K3 MXFP8 runner profiles an SM107 tactic at tuning M ``m``.

    Skips 1-CTA b-reuse tiles (mma_tiler_m 256, mma_inst_m 128), prefetch
    except on the unswapped 2-CTA b-reuse tile (512, 256) or a (4, 2) cluster,
    and split-K above M 256. The prefetching (4, 2) tactics are the fastest of
    the full SM107 space on some K3 MXFP8 weights.
    """
    if tactic[0] == "base":
        mma_tiler_mnk, mma_inst_shape, cluster, swap_ab, use_prefetch = (
            tactic[1],
            tactic[2],
            tactic[3],
            tactic[4],
            tactic[5],
        )
        split_k = tactic[8] if len(tactic) > 8 else 1
    else:
        mma_tiler_mnk, mma_inst_shape, cluster, swap_ab, use_prefetch = (
            tactic[1],
            tactic[2],
            tactic[3],
            tactic[5],
            tactic[6],
        )
        split_k = 1
    mma_m = (mma_tiler_mnk[0], mma_inst_shape[0])
    if mma_m == (256, 128):
        return False
    if use_prefetch and (swap_ab or mma_m != (512, 256)) and tuple(cluster) != (4, 2):
        return False
    return split_k == 1 or m <= _KIMI_K3_MXFP8_MAX_SPLIT_K_M


class RubinBlockScaledGemmRunner(TunableRunner):
    """Datatype-agnostic block-scaled Linear runner for SM107.

    Concrete datatypes provide their element types, scale-factor
    geometry, K alignment, MMA shape, and tuning constraints.
    """

    kernel_class = Sm107BlockScaledPersistentDenseGemmKernel
    kernel_cache: ClassVar[dict]
    gemm_name: ClassVar[str]
    a_dtype: ClassVar[type]
    b_dtype: ClassVar[type]
    sf_dtype: ClassVar[type]
    sf_vec_size: ClassVar[int]
    packed_k_multiplier: ClassVar[int]
    k_alignment: ClassVar[int]
    ab_assumed_align: ClassVar[int]
    sf_assumed_align: ClassVar[int]
    mma_tiler_k: ClassVar[int]
    mma_inst_shape_k: ClassVar[int]
    allow_unswapped_small_m_fallback: ClassVar[bool]
    raster_order_candidates: ClassVar[Tuple[str, ...]] = ("m",)
    scheduler_mode_candidates: ClassVar[Tuple[str, ...]] = ("static", "clc_dynamic")
    mixed_scheduler_mode_candidates: ClassVar[Tuple[str, ...]] = ("static",)
    # Narrowed by the ``max_split_k`` argument, see _split_k_candidates().
    split_k_candidates: ClassVar[Tuple[int, ...]] = (1, 2, 4, 8)
    max_split_k_workspace_bytes: ClassVar[int] = 64 * 1024 * 1024
    split_k_reduction_cache: ClassVar[dict] = {}
    tuning_config: ClassVar[TuningConfig]

    def __init__(self, max_split_k: Optional[int] = None):
        """``max_split_k`` caps the split-K factor of the tactics (None: no limit). The output is
        BF16."""
        super().__init__()

        if max_split_k is not None and max_split_k < 1:
            raise ValueError(f"max_split_k must be a positive integer, got {max_split_k!r}")
        self.max_split_k = max_split_k

    def unique_id(self):
        # The split-K cap narrows the tactic space, so it is part of the autotuner cache key: an
        # entry tuned with split-K must not reach a capped run (or the reverse).
        return (self._split_k_candidates(),)

    def __hash__(self):
        return hash(self.unique_id())

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.unique_id() == other.unique_id()

    def _split_k_candidates(self) -> Tuple[int, ...]:
        """``split_k_candidates`` up to ``max_split_k``."""
        if self.max_split_k is None:
            return self.split_k_candidates
        return tuple(s for s in self.split_k_candidates if s <= self.max_split_k)

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        # SM107 dense block-scaled GEMM only.
        if (sm_version := get_sm_version()) != 107:
            logger.debug(
                f"CuteDSL: SM version {sm_version} is not supported. "
                f"CuteDSL {self.gemm_name} SM107 runner. "
                f"Skipping all tactics."
            )
            return []

        assert inputs[0].dim() == 2
        assert inputs[1].dim() == 2

        m = inputs[0].shape[0]
        n = inputs[1].shape[0]
        k = inputs[0].shape[1]
        real_k = k * self.packed_k_multiplier
        batch_size = 1
        sf_vec_size = self.sf_vec_size

        # A and B are always K-major.
        a_major = "k"
        b_major = "k"

        # Early exit: Check K dimension alignment
        if real_k % self.k_alignment != 0:
            logger.debug(
                f"CuteDSL: K={real_k} does not meet 16-byte alignment requirement "
                f"(K%{self.k_alignment}={real_k % self.k_alignment}, "
                f"expected 0). Skipping all tactics."
            )
            return []

        # Both swap orientations use the original N as C's physical
        # contiguous dimension and require 16-byte BF16 alignment.
        output_aligned = n % 8 == 0
        if not output_aligned:
            logger.debug(
                f"CuteDSL: Output N={n} does not meet the 16-byte "
                f"alignment requirement (N%8={n % 8}). Skipping all tactics."
            )
            return []

        base_swap_ab_candidates = _get_cute_dsl_swap_ab_candidates(m, output_aligned)
        mixed_swap_ab_candidates = _get_cute_dsl_swap_ab_candidates(
            m, output_aligned, include_alternative=True
        )
        if not base_swap_ab_candidates:
            logger.debug(f"CuteDSL: No valid C layout for M={m}, N={n}. Skipping all tactics.")
            return []

        logger.debug(
            f"CuteDSL: M={m}, N={n}(aligned={output_aligned}), K={real_k}(aligned=True). "
            f"Using base swap_ab={base_swap_ab_candidates}, "
            f"mixed-cluster swap_ab={mixed_swap_ab_candidates}"
        )

        # SM107 dense block-scaled tile sizes.
        # b-reuse is enabled when mma_tiler_m = 2 * mma_inst_shape_m
        mma_tiler_k = self.mma_tiler_k
        mma_inst_shape_k = self.mma_inst_shape_k

        # Valid (mma_tiler_m, mma_inst_shape_m) combinations:
        # Without b-reuse: mma_tiler_m == mma_inst_shape_m
        #   - (128, 128): no b-reuse
        #   - (256, 256): no b-reuse, 2CTA mode
        # With b-reuse: mma_tiler_m == 2 * mma_inst_shape_m
        #   - (256, 128): b-reuse
        #   - (512, 256): b-reuse, 2CTA mode
        mma_m_candidates = [
            (128, 128),  # no b-reuse, 1CTA
            (256, 256),  # no b-reuse, 2CTA
            (256, 128),  # b-reuse, 1CTA
            (512, 256),  # b-reuse, 2CTA
        ]
        mma_n_candidates = [64, 128, 192, 256]

        # Cluster shapes - must be divisible by 2 for 2CTA mode (mma_inst_m=256)
        cluster_shape_mn_candidates = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (4, 2),
        ]
        raster_order_candidates = self.raster_order_candidates
        use_prefetch_candidates = [True, False]
        split_k_candidates = self._split_k_candidates()

        valid_tactics = []
        for (
            mma_tiler_m,
            mma_inst_m,
        ), mma_n, cluster_shape_mn, raster_order, swap_ab, use_prefetch in itertools.product(
            mma_m_candidates,
            mma_n_candidates,
            cluster_shape_mn_candidates,
            raster_order_candidates,
            base_swap_ab_candidates,
            use_prefetch_candidates,
        ):
            # Check 2CTA cluster constraint: cluster_m must be divisible by 2
            if mma_inst_m == 256 and cluster_shape_mn[0] % 2 != 0:
                continue

            mma_tiler_mnk = (mma_tiler_m, mma_n, mma_tiler_k)
            mma_inst_shape = (mma_inst_m, mma_n, mma_inst_shape_k)

            if swap_ab:
                c_major = "m"
                kernel_m = n
                kernel_n = m
            else:
                c_major = "n"
                kernel_m = m
                kernel_n = n

            for split_k in split_k_candidates:
                if split_k > 1 and (
                    real_k % mma_tiler_k != 0 or (real_k // mma_tiler_k) % split_k != 0
                ):
                    continue
                fp32_partials = self._fp32_split_k_partials(split_k)
                if fp32_partials and m * n * split_k * 4 > self.max_split_k_workspace_bytes:
                    continue
                if self.__class__.kernel_class.can_implement(
                    (kernel_m, kernel_n, real_k, batch_size),
                    self.a_dtype,
                    self.b_dtype,
                    self.sf_dtype,
                    cutlass.Float32 if fp32_partials else cutlass.BFloat16,
                    a_major,
                    b_major,
                    c_major,
                    sf_vec_size,
                    mma_tiler_mnk,
                    mma_inst_shape,
                    cluster_shape_mn,
                    split_k,
                ):
                    for scheduler_mode in self.scheduler_mode_candidates:
                        # CUDA caps grid Z at 65535; CLC raster-N
                        # tactics beyond that cannot launch.
                        if (
                            scheduler_mode == "clc_dynamic"
                            and raster_order == "n"
                            and not _clc_raster_n_has_launchable_grid(
                                kernel_m,
                                kernel_n,
                                batch_size * split_k,
                                mma_tiler_mnk[:2],
                                mma_inst_m == 256,
                                cluster_shape_mn,
                            )
                        ):
                            continue
                        valid_tactics.append(
                            (
                                "base",
                                mma_tiler_mnk,
                                mma_inst_shape,
                                cluster_shape_mn,
                                swap_ab,
                                use_prefetch,
                                scheduler_mode,
                                raster_order,
                                split_k,
                            )
                        )

        preferred_cluster_shape_mn = (4, 2)
        fallback_cluster_shape_mn = (2, 1)
        for (
            mma_tiler_m,
            mma_inst_m,
        ), mma_n, raster_order, swap_ab, use_prefetch in itertools.product(
            mma_m_candidates,
            mma_n_candidates,
            raster_order_candidates,
            mixed_swap_ab_candidates,
            use_prefetch_candidates,
        ):
            mma_tiler_mnk = (mma_tiler_m, mma_n, mma_tiler_k)
            mma_inst_shape = (mma_inst_m, mma_n, mma_inst_shape_k)

            if swap_ab:
                c_major = "m"
                kernel_m = n
                kernel_n = m
            else:
                c_major = "n"
                kernel_m = m
                kernel_n = n

            if Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.can_implement(
                (kernel_m, kernel_n, real_k, batch_size),
                self.a_dtype,
                self.b_dtype,
                self.sf_dtype,
                cutlass.BFloat16,
                a_major,
                b_major,
                c_major,
                sf_vec_size,
                mma_tiler_mnk,
                mma_inst_shape,
                preferred_cluster_shape_mn,
                fallback_cluster_shape_mn,
            ):
                if self.mixed_scheduler_mode_candidates == ("static",):
                    # Preserve the existing raster-only tactic schema
                    # for runners that do not profile mixed-cluster
                    # schedulers (currently NVFP4).
                    valid_tactics.append(
                        (
                            "mixed_clusters",
                            mma_tiler_mnk,
                            mma_inst_shape,
                            preferred_cluster_shape_mn,
                            fallback_cluster_shape_mn,
                            swap_ab,
                            use_prefetch,
                            raster_order,
                        )
                    )
                else:
                    for scheduler_mode in self.mixed_scheduler_mode_candidates:
                        # CUTLASS DSL 4.5 loses flexible-CGA x/y
                        # offsets in its z-linearized CLC path. Keep
                        # dynamic mixed clusters on the direct-grid
                        # raster-M path; static mixed clusters can
                        # still profile both raster orders.
                        if scheduler_mode == "clc_dynamic" and raster_order != "m":
                            continue
                        valid_tactics.append(
                            (
                                "mixed_clusters",
                                mma_tiler_mnk,
                                mma_inst_shape,
                                preferred_cluster_shape_mn,
                                fallback_cluster_shape_mn,
                                swap_ab,
                                use_prefetch,
                                scheduler_mode,
                                raster_order,
                            )
                        )

        valid_tactics = list(dict.fromkeys(valid_tactics))

        logger.debug(
            f"CuteDSL SM107: Found {len(valid_tactics)} valid tactics for M={m}, N={n}, K={real_k}"
        )
        return valid_tactics

    def make_cute_dsl_global_pointer(self, tensor: torch.Tensor, dtype, assumed_align: int):
        return make_ptr(
            dtype,
            tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=assumed_align,
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic,
        **kwargs,
    ) -> torch.Tensor:
        """
        Performs a dense block-scaled GEMM using CuTe DSL on SM107.

        Args:
            inputs (List[torch.Tensor]):
                inputs[0]: Input tensor of shape (m, k).
                inputs[1]: Weight tensor of shape (n, k).
                inputs[2]: Input scale tensor in R128c4 layout.
                inputs[3]: Weight scale tensor in R128c4 layout.
                inputs[4]: Alpha scaling factor. dtype: float32.
            tactic: Tiling and cluster strategy:
                base: ("base", mma_tiler_mnk, mma_inst_shape,
                cluster_shape, swap_ab, use_prefetch, scheduler_mode,
                raster_order, split_k)
                mixed: ("mixed_clusters", mma_tiler_mnk,
                mma_inst_shape, preferred_cluster_shape,
                fallback_cluster_shape, swap_ab, use_prefetch,
                scheduler_mode, raster_order)

        Returns:
            torch.Tensor: Output tensor of shape (m, n), dtype: bf16.
        """
        sf_vec_size = self.sf_vec_size
        split_k = 1

        if isinstance(tactic, tuple) and len(tactic) > 0 and isinstance(tactic[0], str):
            kernel_variant = tactic[0]
            if kernel_variant == "mixed_clusters":
                if len(tactic) not in {7, 8, 9}:
                    raise ValueError(
                        "CuteDSL SM107 mixed-cluster tactics must use "
                        "Boolean use_prefetch with optional scheduler "
                        "mode and raster order"
                    )
                (
                    _,
                    mma_tiler_mnk,
                    mma_inst_shape,
                    preferred_cluster_shape_mn,
                    cluster_shape_mn,
                    swap_ab,
                    use_prefetch,
                    *tuning_options,
                ) = tactic
                scheduler_mode = "static"
                if len(tuning_options) == 0:
                    raster_order = "m"
                elif len(tuning_options) == 1:
                    # Accept both prior length-8 schemas: raster-only
                    # and scheduler-only.
                    if tuning_options[0] in self.mixed_scheduler_mode_candidates:
                        scheduler_mode = tuning_options[0]
                        raster_order = "m"
                    else:
                        raster_order = tuning_options[0]
                else:
                    scheduler_mode, raster_order = tuning_options
            else:
                if len(tactic) not in {6, 7, 8, 9}:
                    raise ValueError(
                        "CuteDSL SM107 base tactics must use Boolean "
                        "use_prefetch with an optional scheduler mode, "
                        "raster order and split_k"
                    )
                (
                    _,
                    mma_tiler_mnk,
                    mma_inst_shape,
                    cluster_shape_mn,
                    swap_ab,
                    use_prefetch,
                    *tuning_options,
                ) = tactic
                preferred_cluster_shape_mn = None
                scheduler_mode = "static"
                raster_order = "m"
                if len(tuning_options) >= 1:
                    scheduler_mode = tuning_options[0]
                if len(tuning_options) >= 2:
                    raster_order = tuning_options[1]
                if len(tuning_options) >= 3:
                    split_k = tuning_options[2]
        elif isinstance(tactic, tuple) and len(tactic) == 5:
            kernel_variant = "base"
            (mma_tiler_mnk, mma_inst_shape, cluster_shape_mn, swap_ab, use_prefetch) = tactic
            raster_order = "m"
            scheduler_mode = "static"
            preferred_cluster_shape_mn = None
        else:
            # fallback to default tactic
            kernel_variant = "base"
            mma_tiler_mnk = (128, 128, self.mma_tiler_k)
            mma_inst_shape = (128, 128, self.mma_inst_shape_k)
            cluster_shape_mn = (1, 1)
            swap_ab = False
            use_prefetch = False
            raster_order = "m"
            scheduler_mode = "static"
            preferred_cluster_shape_mn = None

        if not isinstance(use_prefetch, bool):
            raise ValueError("CuteDSL SM107 use_prefetch must be Boolean")
        if not isinstance(split_k, int) or isinstance(split_k, bool) or split_k < 1:
            raise ValueError(f"CuteDSL SM107 split_k must be a positive integer, got {split_k!r}")
        if raster_order not in {"m", "n"}:
            raise ValueError(f"CuteDSL SM107 raster_order must be 'm' or 'n', got {raster_order!r}")
        if (
            kernel_variant == "mixed_clusters"
            and scheduler_mode not in self.mixed_scheduler_mode_candidates
        ):
            raise ValueError(
                f"Unsupported CuteDSL SM107 mixed-cluster scheduler mode {scheduler_mode!r}"
            )
        if (
            kernel_variant == "mixed_clusters"
            and scheduler_mode == "clc_dynamic"
            and raster_order != "m"
        ):
            raise ValueError(
                "CuteDSL SM107 mixed-cluster CLC scheduling only supports raster_order='m'"
            )
        if scheduler_mode not in {"static", "clc_dynamic"}:
            raise ValueError(f"Unsupported CuteDSL SM107 scheduler mode {scheduler_mode!r}")
        # ``reserve_sms``: SMs this launch keeps out of its grid, for a kernel that runs beside it
        # and needs them from its start (the static scheduler over fewer clusters).
        reserve_sms = int(kwargs.get("reserve_sms") or 0)
        scheduler_mode = _reserved_scheduler(scheduler_mode, reserve_sms)

        a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor = inputs

        m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]

        if split_k > 1:
            # Before anything is allocated or launched.
            self._check_split_k(split_k, k * self.packed_k_multiplier, mma_tiler_mnk[2])

        # A cached profile can cover larger M than its tuning sample.
        # Keep the workspace bound valid for the actual invocation.
        if (
            self._fp32_split_k_partials(split_k)
            and m * n * split_k * 4 > self.max_split_k_workspace_bytes
        ):
            split_k = 1

        c_tensor = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")
        if swap_ab:
            c_tensor = c_tensor.permute(1, 0)
        fp32_partials = self._fp32_split_k_partials(split_k)
        if fp32_partials:
            partials = torch.empty((split_k, m, n), dtype=torch.float32, device=a_tensor.device)
            kernel_c_tensor = partials
            kernel_c_dtype = cutlass.Float32
        else:
            kernel_c_tensor = c_tensor
            kernel_c_dtype = cutlass.BFloat16

        real_k = k * self.packed_k_multiplier
        sf_m = pad_up(m, 128)
        sf_k = pad_up(real_k // sf_vec_size, 4)
        sf_n = pad_up(n, 128)

        # Scale factors already use CuTe DSL's physical R128c4 layout.
        # Only flatten their storage here; no memory copy is required.
        expected_a_sf_size = sf_m * sf_k
        expected_b_sf_size = sf_n * sf_k

        if a_sf_tensor.numel() != expected_a_sf_size:
            raise ValueError(
                f"CuteDSL: act scale factor size mismatch. "
                f"Expected {expected_a_sf_size} (sf_m={sf_m} * sf_k={sf_k}), "
                f"got {a_sf_tensor.numel()} for shape M={m}, K={real_k}"
            )
        if b_sf_tensor.numel() != expected_b_sf_size:
            raise ValueError(
                f"CuteDSL: weight scale factor size mismatch. "
                f"Expected {expected_b_sf_size} (sf_n={sf_n} * sf_k={sf_k}), "
                f"got {b_sf_tensor.numel()} for shape N={n}, K={real_k}"
            )
        if alpha_tensor.numel() != 1:
            raise ValueError(
                f"CuteDSL: alpha size mismatch. Expected 1, got {alpha_tensor.numel()}"
            )

        # Reshape to CuteDSL's expected format (just a view, no copy)
        a_sf_tensor = a_sf_tensor.reshape(sf_m * sf_k)
        b_sf_tensor = b_sf_tensor.reshape(sf_n * sf_k)

        if swap_ab:
            kernel_m = n
            kernel_n = m
            kernel_sf_m = sf_n
            kernel_sf_n = sf_m

            kernel_a_tensor = b_tensor
            kernel_a_sf_tensor = b_sf_tensor
            kernel_b_tensor = a_tensor
            kernel_b_sf_tensor = a_sf_tensor
        else:
            kernel_m = m
            kernel_n = n
            kernel_sf_m = sf_m
            kernel_sf_n = sf_n

            kernel_a_tensor = a_tensor
            kernel_a_sf_tensor = a_sf_tensor
            kernel_b_tensor = b_tensor
            kernel_b_sf_tensor = b_sf_tensor

        if kernel_variant == "base":
            launch_raster_order = _launchable_raster_order(
                scheduler_mode,
                raster_order,
                kernel_m,
                kernel_n,
                split_k,
                mma_tiler_mnk[:2],
                mma_inst_shape[0] == 256,
                cluster_shape_mn,
            )
            if launch_raster_order != raster_order:
                logger.debug(
                    f"CuteDSL SM107: CLC raster-N exceeds the grid-Z "
                    f"limit for M={kernel_m}, N={kernel_n}, "
                    f"split_k={split_k}; using raster-M."
                )
                raster_order = launch_raster_order

        # The wrapper entry points declare m/n/k and all scale-tile
        # counts (sf_m/sf_n/sf_k) as dynamic cutlass.Int64
        # runtime arguments, so the compiled artifact is independent
        # of ALL problem shapes: cache tactic-only, matching the
        # Blackwell NVFP4 and SM107 BF16 runners. A shape in the key
        # would cost a ~1.3 s cute.compile per new ceil(M/128) bucket
        # at serving time (an iteration stall) and multiply cold
        # autotuning compiles across weight families sharing a
        # tactic. Per-shape tactic validity is still enforced by
        # can_implement at tactic-selection time.
        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        max_active_clusters = _reserved_clusters(
            get_max_activate_clusters(cluster_size), cluster_size, reserve_sms
        )
        max_active_preferred_clusters = None
        if kernel_variant == "mixed_clusters":
            preferred_size = preferred_cluster_shape_mn[0] * preferred_cluster_shape_mn[1]
            max_active_preferred_clusters = _reserved_clusters(
                get_max_activate_clusters(preferred_size), preferred_size, reserve_sms
            )
        cache_key = (
            kernel_variant,
            sf_vec_size,
            mma_tiler_mnk,
            mma_inst_shape,
            preferred_cluster_shape_mn,
            cluster_shape_mn,
            swap_ab,
            use_prefetch,
            raster_order,
            scheduler_mode,
            split_k,
            fp32_partials,
            max_active_preferred_clusters,
            max_active_clusters,
        )

        if cache_key not in self.__class__.kernel_cache:
            a_ptr = self.make_cute_dsl_global_pointer(a_tensor, self.a_dtype, self.ab_assumed_align)
            b_ptr = self.make_cute_dsl_global_pointer(b_tensor, self.b_dtype, self.ab_assumed_align)
            a_sf_ptr = self.make_cute_dsl_global_pointer(
                a_sf_tensor, self.sf_dtype, self.sf_assumed_align
            )
            b_sf_ptr = self.make_cute_dsl_global_pointer(
                b_sf_tensor, self.sf_dtype, self.sf_assumed_align
            )
            c_ptr = self.make_cute_dsl_global_pointer(kernel_c_tensor, kernel_c_dtype, 16)
            alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)
            # make faked stream
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            if swap_ab:
                kernel_a_ptr = b_ptr
                kernel_a_sf_ptr = b_sf_ptr
                kernel_b_ptr = a_ptr
                kernel_b_sf_ptr = a_sf_ptr
            else:
                kernel_a_ptr = a_ptr
                kernel_a_sf_ptr = a_sf_ptr
                kernel_b_ptr = b_ptr
                kernel_b_sf_ptr = b_sf_ptr

            scheduler_types = {
                "static": cutlass.utils.StaticPersistentTileScheduler,
                "clc_dynamic": cutlass.utils.ClcDynamicPersistentTileScheduler,
            }
            scheduler_type = scheduler_types[scheduler_mode]

            # The SM107 kernels take mma_inst_shape and mma_tiler as 3-tuples.
            if kernel_variant == "mixed_clusters":
                gemm = Sm107BlockScaledPersistentDenseGemmMixedClustersKernel(
                    sf_vec_size,
                    mma_inst_shape,
                    mma_tiler_mnk,
                    preferred_cluster_shape_mn,
                    cluster_shape_mn,
                    None if use_prefetch else 0,
                    raster_order=raster_order,
                    scheduler_type=scheduler_type,
                )
            else:
                gemm = self.__class__.kernel_class(
                    sf_vec_size,
                    mma_inst_shape,
                    mma_tiler_mnk,
                    cluster_shape_mn,
                    None if use_prefetch else 0,
                    raster_order=raster_order,
                    scheduler_type=scheduler_type,
                    split_k=split_k,
                )
            # Note: when tvm_ffi fake stream is used, at least one parameter should be tensor type,
            # so we make alpha as the cute.Tensor type in the jit func.
            compile_args = [
                kernel_m,
                kernel_n,
                real_k,
                kernel_sf_m // 128,
                kernel_sf_n // 128,
                sf_k // 4,
                1,  # batch
                kernel_a_ptr,
                kernel_b_ptr,
                kernel_a_sf_ptr,
                kernel_b_sf_ptr,
                c_ptr,
                alpha_cute_tensor,
            ]
            if kernel_variant == "mixed_clusters":
                compile_args.extend([max_active_preferred_clusters, max_active_clusters])
            else:
                compile_args.append(max_active_clusters)
            compile_args.extend([stream, swap_ab])

            compiled_gemm = cute.compile(
                gemm.wrapper, *compile_args, options="--opt-level 2 --enable-tvm-ffi"
            )

            self.__class__.kernel_cache[cache_key] = compiled_gemm
        else:
            compiled_gemm = self.__class__.kernel_cache[cache_key]

        # Launch gemm kernel via wrapper interface
        compiled_gemm(
            kernel_m,
            kernel_n,
            real_k,
            kernel_sf_m // 128,
            kernel_sf_n // 128,
            sf_k // 4,
            kernel_a_tensor.data_ptr(),
            kernel_b_tensor.data_ptr(),
            kernel_a_sf_tensor.data_ptr(),
            kernel_b_sf_tensor.data_ptr(),
            kernel_c_tensor.data_ptr(),
            alpha_tensor,
        )

        if swap_ab:
            c_tensor = c_tensor.permute(1, 0)
        if fp32_partials:
            self._reduce_split_k_partials(partials, c_tensor, split_k)
        return c_tensor

    def _check_split_k(self, split_k: int, real_k: int, k_tile: int) -> None:
        """Refuse a split-K tactic (split_k > 1) the kernel cannot run.

        get_valid_tactics offers split-K only where can_implement
        passes, but forward also runs explicit tactics, and neither it
        nor the kernel checked more than split_k >= 1: such a launch
        can fault or hang the GPU. Refused on the host, before anything
        is allocated or launched:

        - K that is not a whole number of the tactic's K tiles, or a
          tile count that split_k does not divide (can_implement's
          rule): a split would get part of a K tile, or none.
        """
        if k_tile < 1 or real_k % k_tile != 0 or (real_k // k_tile) % split_k != 0:
            raise ValueError(
                f"CuteDSL SM107 {self.gemm_name}: split_k={split_k} "
                f"needs K={real_k} in whole K tiles of {k_tile}, a "
                "count split_k divides"
            )

    def _fp32_split_k_partials(self, split_k: int) -> bool:
        """Whether a launch writes FP32 partial planes: every split-K launch does. The planes are
        summed in FP32 and the sum is rounded once to BF16."""
        return split_k > 1

    def _reduce_split_k_partials(
        self, partials: torch.Tensor, output: torch.Tensor, split_k: int
    ) -> None:
        partials_flat = partials.view(-1)
        output_flat = output.view(-1)
        if split_k not in self.split_k_reduction_cache:
            self.split_k_reduction_cache[split_k] = cute.compile(
                reduce_split_k_fp32,
                cute.runtime.from_dlpack(partials_flat).mark_layout_dynamic(),
                cute.runtime.from_dlpack(output_flat).mark_layout_dynamic(),
                split_k,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--opt-level 2 --enable-tvm-ffi",
            )
        self.split_k_reduction_cache[split_k](partials_flat, output_flat)


class RubinMxfp8GemmRunner(RubinBlockScaledGemmRunner):
    """SM107 MXFP8 runner backed by the persistent dense kernel, with fine-grained tuning for
    K3's low-M range.

    ``tuning_config`` tunes on every rank by itself (``INDEPENDENT``): under attention DP each rank
    multiplies its own rows, and ranks may run different tactics. ``replicated_tuning_config``
    merges the ranks' measurements into one choice (``MERGE``), for a tuning group whose ranks serve
    the same requests (TP, Helix CP), whose replicated projections must produce the same bits.
    """

    kernel_cache = dict()
    gemm_name = "MXFP8"
    a_dtype = cutlass.Float8E4M3FN
    b_dtype = cutlass.Float8E4M3FN
    sf_dtype = cutlass.Float8E8M0FNU
    sf_vec_size = 32
    packed_k_multiplier = 1
    k_alignment = 16
    ab_assumed_align = 16
    sf_assumed_align = 16
    mma_tiler_k = 128
    mma_inst_shape_k = 64
    allow_unswapped_small_m_fallback = True
    raster_order_candidates = ("m", "n")
    mixed_scheduler_mode_candidates = ("static", "clc_dynamic")
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                0, 0, _get_kimi_k3_mxfp8_tuning_buckets, _kimi_k3_mxfp8_tuning_bucket
            ),
        ),
        constraint_specs=(ConstraintSpec(2, 0, mxfp8_scale_infer_shape),),
        use_cold_l2_cache=True,
        distributed_tuning_strategy=DistributedTuningStrategy.INDEPENDENT,
        use_cuda_graph=True,
    )
    replicated_tuning_config = dataclasses.replace(
        tuning_config, distributed_tuning_strategy=DistributedTuningStrategy.MERGE
    )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[tuple]:
        tactics = super().get_valid_tactics(inputs, profile, **kwargs)
        m = inputs[0].shape[0]
        return [tactic for tactic in tactics if _is_kimi_k3_mxfp8_tactic_profiled(tactic, m)]
