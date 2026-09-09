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

# ruff: noqa: E501, E741

import argparse
from functools import lru_cache
from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import testing
from cutlass.cute.arch.smem import map_dsmem_ptr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.cpasync import CopyDsmemStoreOp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import create_cute_tensor_for_fp8, is_fp8_dtype

"""
A high-performance cluster launch control(CLC) dynamic persistent batched dense GEMM example
for the NVIDIA Blackwell SM100 architecture using CUTE DSL.

The CLC dynamic persistent scheduling technique performs dynamic loading balancing.
It has the ability to adapt available SMs rather than a statically selected number. To support this,
a new instruction is introduced to query for a new tile to compute. This new instruction is similar
to programmatic multicast in context of clusters in that the same starting tile ID for a given cluster
is broadcasted to all threadblocks in the cluster.
See `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel>`.

- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support CLC dynamic persistent tile scheduling to have near perfect load balancing
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp: Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.
    - Optionally accept an elementwise lambda function epilogue_op to apply to the output tensor:
      e.g., relu can set epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is same as dense_gemm.py.

.. code-block:: bash

    python examples/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py                          \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
      --mnkl 8192,8192,8192,1                                                   \
      --use_tma_store --use_2cta_instrs

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py                     \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                 \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,8192,1                                                  \
      --use_tma_store --use_2cta_instrs                                        \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints are same as dense_gemm.py:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below PersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type
* Mma tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* Mma tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


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
    """Computes the number of stages for A/B/C operands based on heuristics.

    :param tiled_mma: The tiled MMA object defining the core computation.
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
    :type mma_tiler_mnk: tuple[int, int, int]
    :param a_dtype: Data type of operand A.
    :type a_dtype: type[cutlass.Numeric]
    :param b_dtype: Data type of operand B.
    :type b_dtype: type[cutlass.Numeric]
    :param c_dtype: Data type of operand C (output).
    :type c_dtype: type[cutlass.Numeric]
    :param smem_capacity: Total available shared memory capacity in bytes.
    :type smem_capacity: int
    :param occupancy: Target number of CTAs per SM (occupancy).
    :type occupancy: int
    :param use_tma_store: Whether TMA store is enabled.
    :type use_tma_store: bool
    :param c_smem_layout: Layout of C operand in shared memory, or None if not using TMA store.
    :type c_smem_layout: Union[cute.Layout, None]

    :return: A tuple containing the computed number of stages for:
             (ACC stages, A/B operand stages, C stages)
    :rtype: tuple[int, int, int]
    """
    # Default ACC stages
    num_acc_stage = 2

    # Default C stages
    num_c_stage = 2 if use_tma_store else 0

    # Calculate smem layout and size for one stage of A, B, and C with 1-stage
    a_smem_layout_stage_one = utils.sm100.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
    b_smem_layout_staged_one = utils.sm100.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)

    ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
        b_dtype, b_smem_layout_staged_one
    )
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    # Calculate A/B stages:
    # Start with total smem per CTA (capacity / occupancy)
    # Subtract reserved bytes and initial C stages bytes
    # Divide remaining by bytes needed per A/B stage
    num_ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage

    # Refine epilogue stages:
    # Calculate remaining smem after allocating for A/B stages and reserved bytes
    # Add remaining unused smem to epilogue
    if use_tma_store:
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
    return num_acc_stage, num_ab_stage, num_c_stage


class PersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
    :type use_tma_store: bool

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp32, fp16, and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    **Example:**
        gemm = PersistentDenseGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(2, 2),
            use_tma_store=True
        )
        gemm(a, b, c, stream, epilogue_op)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
        split_k: int = 1,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3. Output C tensor store mode:
            - use_tma_store: Boolean indicating whether to use Tensor Memory Access (TMA) for storing results.

        4. Cluster Split-K:
            - split_k: Number of CTAs along cluster Z that split the K dimension.
              Each CTA computes a K-partition and they reduce via DSMEM scatter.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Use Tensor Memory Access (TMA) or normal store for output C tensor.
        :type use_tma_store: bool
        :param split_k: cluster split-K factor. CTAs along cluster Z split K and reduce via DSMEM.
        :type split_k: int
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along = raster_along
        self.split_k = split_k
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.arch = "sm_100"

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        # Set specialized warp ids
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                *self.epilogue_warp_id,
            )
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

        # DSMEM mailbox sizing: each of 128 epilogue threads owns
        # (cta_m * cta_n / 128) accumulator elems, sharded split_k ways.
        # use_2cta_instrs halves the per-CTA M tile.
        cta_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
        self._mailbox_elems_per_thread = (cta_m * mma_tiler_mn[1]) // 128
        shard_ept = self._mailbox_elems_per_thread // max(split_k, 1)
        self._mailbox_total_elems = max(split_k - 1, 0) * 128 * shard_ept

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
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

        # Compute cluster layout (M,N only -- used for TMA multicast)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Full cluster layout including split_k Z-dimension (used for CLC pipeline signaling)
        self.clc_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, self.split_k)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
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

        # Reserve SMEM for DSMEM mailbox when using split-K
        dsmem_mailbox_bytes = 0
        if self.split_k > 1:
            dsmem_mailbox_bytes = self._mailbox_total_elems * 4 + 16  # data + 2 barriers

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity - dsmem_mailbox_bytes,
            self.occupancy,
            self.use_tma_store,
            c_smem_layout,
        )

        # Setup clc stage by default
        self.num_clc_stage = 1
        assert self.num_clc_stage == 1, "Only single-stage CLC pipeline is supported"

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
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()

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
        # Response size is 4B * 4 elements
        self.num_clc_response_bytes = 16

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
            self.split_k,
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.cluster_layout_vmnk,
            self.clc_cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, self.split_k),
            stream=stream,
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
        clc_cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
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
        bidx, _bidy, _bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        # When split_k>1 the physical cluster extends along Z; fold out the Z stride
        # to get the MN-only rank for TMA partition. split_rank is this CTA's Z position.
        mn_cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        cta_rank_in_mn_cluster = cta_rank_in_cluster % mn_cluster_size
        (
            block_in_cluster_x,
            block_in_cluster_y,
            split_rank,
        ) = cute.arch.block_in_cluster_idx()
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_mn_cluster)
        full_block_in_cluster_coord_vmnk = clc_cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
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
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response_align_bytes = self.num_clc_response_bytes
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4],
                clc_response_align_bytes,
            ]
            dsmem_mailbox_full_barrier: cute.struct.MemRange[
                cutlass.Int64, 1 if self.split_k > 1 else 0
            ]
            dsmem_mailbox_empty_barrier: cute.struct.MemRange[
                cutlass.Int64, 1 if self.split_k > 1 else 0
            ]
            dsmem_mailbox: cute.struct.MemRange[cutlass.Float32, self._mailbox_total_elems]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Warp)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            # Pipeline signaling must use the full physical cluster layout so internal
            # block_idx_in_cluster -> get_flat_coord() mappings remain valid when split_k>1.
            # The actual TMA multicast masks still use the MN-only cluster layout below.
            cta_layout_vmnk=clc_cluster_layout_vmnk,
            enable_multicast_signaling=True,
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
            cta_layout_vmnk=clc_cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize clc_pipeline (barrier) and states
        # Use full cluster size (including split_k) for CLC pipeline signaling
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        full_cluster_size = cute.size(self.cluster_shape_mn) * self.split_k
        num_clc_consumer_threads = 32 * (
            1 + full_cluster_size * (1 + len(self.epilogue_warp_id) + 1)
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
            cta_layout_vmnk=clc_cluster_layout_vmnk,
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
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # DSMEM mailbox barriers (split-K only): FULL is armed per-tile by
        # arrive_and_expect_tx; EMPTY collects one ack from each peer after it reads.
        # Guard mbarrier_init with a single-warp check -- otherwise every specialized
        # warp's elect_one would race on the same barrier (7 concurrent writers).
        # The reference tgv_gemm split-K epilogue uses the same single-warp pattern.
        dsmem_mailbox_full_mbar = storage.dsmem_mailbox_full_barrier.data_ptr()
        dsmem_mailbox_empty_mbar = storage.dsmem_mailbox_empty_barrier.data_ptr()
        dsmem_mailbox_ptr = storage.dsmem_mailbox.data_ptr()
        if cutlass.const_expr(self.split_k > 1):
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(dsmem_mailbox_full_mbar, 1)
                    cute.arch.mbarrier_init(dsmem_mailbox_empty_mbar, self.split_k - 1)

        # Cluster arrive after barrier init
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
        else:
            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Initial clc response pointer
        clc_response_ptr = storage.clc_response.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )

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
                clc_cluster_layout_vmnk, full_block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                clc_cluster_layout_vmnk, full_block_in_cluster_coord_vmnk, mcast_mode=1
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

        # Cluster split-K: each CTA owns k_tiles_per_split tiles starting at k_start,
        # clamped so the last Z CTA gets only what's left. Reduces to the full range
        # when split_k=1.
        k_tiles_per_split = (k_tile_cnt + self.split_k - 1) // self.split_k
        k_start = split_rank * k_tiles_per_split
        k_start = min(k_start, k_tile_cnt)
        k_end = k_start + k_tiles_per_split
        k_end = min(k_end, k_tile_cnt)
        k_tile_count = k_end - k_start

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
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()
        else:
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Construct the scheduler
        #
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler. Grid Z is L*split_k so divide
                # out the split_k factor to recover the batch index (no-op when split_k=1).
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2] // self.split_k,
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
                # Tma load loop (K-partitioned for split-K)
                #
                for _k_tile in cutlass.range(0, k_tile_count, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # Offset into global K tiles for this CTA's partition
                    k_tile_global = handle.count + k_start

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_tile_global)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_tile_global)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for next k_tile
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_count:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Sched warp
        #
        if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
            #
            # Persistent tile scheduling loop
            #
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
            )

            while work_tile.is_valid_tile:
                #
                # Advance to next tile
                #
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

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
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2] // self.split_k,
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
                # Mma mainloop (K-partitioned for split-K)
                #
                for k_tile in range(k_tile_count):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_count:
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
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
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
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            if cutlass.const_expr(self.use_tma_store):
                assert tma_atom_c is not None and sC is not None
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilogue_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group
                )

            # DSMEM split-K: pre-compute epilogue partitioning outside persistent loop
            if cutlass.const_expr(self.split_k > 1):
                tCgC_xf = utils.gemm.sm100.transform_partitioned_tensor_layout(tCgC)
                tCtAcc_xf = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc_base)

                tiled_copy_t2r, tTR_tAcc_base, _ = (
                    utils.gemm.sm100.epilogue_tmem_copy_and_partition(
                        self, tidx, tCtAcc_xf, tCgC_xf, epi_tile, self.use_2cta_instrs
                    )
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_gC_partitioned = thr_copy_t2r.partition_D(cute.flat_divide(tCgC_xf, epi_tile))

                epilog_sync_barrier = pipeline.NamedBarrier(
                    barrier_id=self.epilog_sync_bar_id,
                    num_threads=32 * len(self.epilogue_warp_id),
                )
                dsmem_full_phase = cutlass.Int32(0)
                dsmem_empty_phase = cutlass.Int32(0)
                num_tiles_done = cutlass.Int32(0)

                # (epi_tid, elem_in_shard, sender_slot) -> flat mailbox offset.
                # epi_tid is innermost (stride 1) for coalesced DSMEM stores.
                shard_ept_static = self._mailbox_elems_per_thread // self.split_k
                mailbox_layout = cute.make_layout(
                    (128, shard_ept_static, max(self.split_k - 1, 1)),
                    stride=(1, 128, 128 * shard_ept_static),
                )
                dsmem_store_atom = cute.make_copy_atom(CopyDsmemStoreOp(), cutlass.Float32)
                scatter_val = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2] // self.split_k,
                )
                num_tiles_executed = tile_sched.num_tiles_executed
                if cutlass.const_expr(self.split_k > 1):
                    # ---- Cluster split-K epilogue with DSMEM scatter-reduce ----
                    # Non-TMA epilogue sets epi_tile == cta_tile so subtile_cnt == 1.
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]
                    tTR_gC = tTR_gC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                    acc_pipeline.consumer_wait(acc_consumer_state)

                    for subtile_idx in range(subtile_cnt):
                        # Block until every peer has read our previous sender slot.
                        if num_tiles_done > 0 or subtile_idx > 0:
                            cute.arch.mbarrier_wait(dsmem_mailbox_empty_mbar, dsmem_empty_phase)
                            dsmem_empty_phase = dsmem_empty_phase ^ 1

                        tTR_gC_subtile = tTR_gC[(None, None, None, subtile_idx)]
                        tTR_rAcc = cute.make_rmem_tensor(tTR_gC_subtile.shape, self.acc_dtype)
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, subtile_idx)],
                            tTR_rAcc,
                        )

                        # Release acc pipeline once the final subtile is fetched from TMEM.
                        if subtile_idx == subtile_cnt - 1:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                        # Derive shard size from the actual rmem tensor (avoids baking in
                        # a per-subtile constant that may differ if epi_tile < cta_tile).
                        shard_ept = cute.size(tTR_rAcc) // self.split_k
                        # Expected bytes from (split_k - 1) peers: each peer's 128
                        # epilogue threads (4 warps * 32) send shard_ept elements
                        # of 4 B (Float32) into our mailbox.
                        mailbox_tx_total = (self.split_k - 1) * 128 * shard_ept * 4
                        my_shard_start = split_rank * shard_ept

                        # Arm mailbox FULL (expecting TX from split_k-1 peers) then scatter.
                        if tidx == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                dsmem_mailbox_full_mbar, mailbox_tx_total
                            )
                        epilog_sync_barrier.arrive_and_wait()

                        # Each peer writes to a slot in our mailbox indexed by the sender
                        # rank with self-rank compressed out (slot = split_rank if
                        # split_rank < peer else split_rank - 1).
                        for peer in range(self.split_k):
                            if peer != split_rank:
                                sender_idx = split_rank - (
                                    1 if split_rank > cutlass.Int32(peer) else 0
                                )
                                peer_rank = (
                                    block_in_cluster_x
                                    + cutlass.Int32(self.cluster_shape_mn[0]) * block_in_cluster_y
                                    + cutlass.Int32(peer * mn_cluster_size)
                                )
                                remote_mbar = map_dsmem_ptr(dsmem_mailbox_full_mbar, peer_rank)
                                for i in range(shard_ept):
                                    scatter_val[0] = tTR_rAcc[peer * shard_ept + i]
                                    remote_ptr = map_dsmem_ptr(
                                        dsmem_mailbox_ptr + mailbox_layout((tidx, i, sender_idx)),
                                        peer_rank,
                                    )
                                    dst = cute.make_tensor(remote_ptr, cute.make_layout(1))
                                    cute.copy(
                                        dsmem_store_atom,
                                        scatter_val,
                                        dst,
                                        mbar_ptr=remote_mbar,
                                    )

                        # Do not let local FULL waits move ahead of this CTA's
                        # remote DSMEM stores to peer mailboxes.
                        cute.arch.fence_acq_rel_cta()
                        cute.arch.mbarrier_wait(dsmem_mailbox_full_mbar, dsmem_full_phase)
                        dsmem_full_phase = dsmem_full_phase ^ 1

                        # Reduce received shards into our own partial.
                        for s in range(self.split_k - 1):
                            for i in range(shard_ept):
                                tTR_rAcc[my_shard_start + i] = (
                                    tTR_rAcc[my_shard_start + i]
                                    + (dsmem_mailbox_ptr + mailbox_layout((tidx, i, s))).load()
                                )

                        # Intra-CTA sync: tidx=0 must not signal peer empty barriers
                        # until every thread in our CTA has finished reading the mailbox,
                        # otherwise the peer's next-tile scatter may overwrite slots we
                        # are still reading.
                        epilog_sync_barrier.arrive_and_wait()

                        # Signal each peer that we've read their shard.
                        if tidx == 0:
                            for peer in range(self.split_k):
                                if peer != split_rank:
                                    peer_rank = (
                                        block_in_cluster_x
                                        + cutlass.Int32(self.cluster_shape_mn[0])
                                        * block_in_cluster_y
                                        + cutlass.Int32(peer * mn_cluster_size)
                                    )
                                    cute.arch.mbarrier_arrive(
                                        dsmem_mailbox_empty_mbar,
                                        peer_cta_rank_in_cluster=peer_rank,
                                    )

                        # Predicated GMEM store: only write our shard to C.
                        # (Two-pass predicate setup avoids Python chained comparisons
                        # and `or` on runtime Int32, which don't short-circuit correctly.)
                        tTR_rC = cute.make_rmem_tensor(tTR_gC_subtile.shape, self.c_dtype)
                        tTR_rC.store(epilogue_op(tTR_rAcc.load().to(self.c_dtype)))
                        pred = cute.make_rmem_tensor(tTR_gC_subtile.shape, cutlass.Boolean)
                        for i in range(cute.size(tTR_rC)):
                            pred[i] = cutlass.Boolean(True)
                        for i in range(cute.size(tTR_rC)):
                            if i < my_shard_start or i >= my_shard_start + shard_ept:
                                pred[i] = cutlass.Boolean(False)
                        cute.basic_copy_if(pred, tTR_rC, tTR_gC_subtile)

                    num_tiles_done = num_tiles_done + 1

                elif cutlass.const_expr(self.use_tma_store):
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
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                else:
                    acc_consumer_state = utils.gemm.sm100.epilogue(
                        self,
                        tidx,
                        tCtAcc_base,
                        tCgC,
                        epi_tile,
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                    )
                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            if cutlass.const_expr(self.use_tma_store):
                # Wait for C store complete
                c_pipeline.producer_tail()
            elif cutlass.const_expr(self.split_k <= 1):
                # Synchronize before TMEM dealloc (done by the caller)
                tmem_dealloc_barrier.arrive_and_wait()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        if cutlass.const_expr(self.split_k > 1):
            # Keep DSMEM mailbox barriers alive until all peer CTAs finish the
            # final scatter/read/ack sequence for this physical cluster.
            # Otherwise races
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        swizzle_size: int,
        raster_along: Literal["m", "n"],
        split_k: int = 1,
    ) -> Tuple[utils.ClcDynamicPersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param split_k: cluster split-K factor (CTAs along cluster Z).
        :type split_k: int

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.ClcDynamicPersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        # Grid Z is expanded by split_k so the physical launch uses a 3D cluster, but the
        # existing CLC scheduler still reasons about output-tile ownership only in M/N.
        # Keep the scheduler's logical cluster shape at K=1 and let the kernel reinterpret
        # cluster Z as the split-k rank.
        num_ctas_mnl_expanded = (
            num_ctas_mnl[0],
            num_ctas_mnl[1],
            num_ctas_mnl[2] * split_k,
        )
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl_expanded, cluster_shape_mnl, swizzle_size, raster_along == "m"
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)

        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
        arch: str,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: The stage of the accumulator tensor.
        :type num_acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, arch=arch)

        return num_tmem_alloc_cols

    def check_supported_dtypes(
        self,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ):
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :raises testing.CantImplementError: If the dtypes are invalid
        """
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
            raise testing.CantImplementError(f"Unsupported AB dtype: {a_dtype} and {b_dtype}")

        if self.acc_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.Int32}:
            raise testing.CantImplementError(f"Unsupported accumulator dtype: {self.acc_dtype}")

        # Define compatibility mapping between accumulator type and AB type
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.TFloat32,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },  # Float32 accumulator supports floating point AB types only
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # Check compatibility between accumulator type and AB type
        if (
            a_dtype not in acc_ab_compatibility[self.acc_dtype]
            or b_dtype not in acc_ab_compatibility[self.acc_dtype]
        ):
            raise testing.CantImplementError(
                f"Unsupported AB dtype: {a_dtype} and {b_dtype} for accumulator dtype: {self.acc_dtype}"
            )

        # Define compatibility mapping between accumulator type and C type
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
            cutlass.Float16: {
                cutlass.BFloat16,
                cutlass.Float16,
            },
            cutlass.Int32: {
                cutlass.BFloat16,
                cutlass.Float16,
                cutlass.Float32,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # Check compatibility between accumulator type and C type
        if c_dtype not in acc_c_compatibility[self.acc_dtype]:
            raise testing.CantImplementError(
                f"Unsupported C dtype: {c_dtype} for accumulator dtype: {self.acc_dtype}"
            )

    def check_mma_tiler_and_cluster_shape(self):
        """Check if the mma tiler and cluster shape are valid.

        :raises testing.CantImplementError: If the mma tiler and cluster shape are invalid
        """
        # Skip invalid mma tile shape
        if not (
            (not self.use_2cta_instrs and self.mma_tiler_mn[0] in [64, 128])
            or (self.use_2cta_instrs and self.mma_tiler_mn[0] in [128, 256])
        ):
            raise testing.CantImplementError(
                f"Invalid mma tiler & use_2cta_instrs: {self.mma_tiler_mn}, {self.use_2cta_instrs}"
            )
        if self.mma_tiler_mn[1] not in range(32, 257, 32):
            raise testing.CantImplementError(f"Invalid mma tiler N: {self.mma_tiler_mn[1]}")
        # Skip illegal cluster shape
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            raise testing.CantImplementError(f"Invalid cluster shape M: {self.cluster_shape_mn[0]}")

        # Skip invalid cluster shape (total cluster including split_k must be <= 16)
        def is_power_of_2(value: int) -> bool:
            return value > 0 and (value & (value - 1)) == 0

        total_cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1] * self.split_k
        if (
            total_cluster_size > 16
            or self.cluster_shape_mn[0] <= 0
            or self.cluster_shape_mn[1] <= 0
            or not is_power_of_2(self.cluster_shape_mn[0])
            or not is_power_of_2(self.cluster_shape_mn[1])
        ):
            raise testing.CantImplementError(
                f"Invalid cluster shape: {self.cluster_shape_mn} with split_k={self.split_k}"
            )
        # split_k > 1 requires non-TMA store epilogue
        if self.split_k > 1 and self.use_tma_store:
            raise testing.CantImplementError("split_k > 1 requires use_tma_store=False")
        # The current split-k integration layers cluster Z in software on top of the
        # existing 2D CLC scheduler. Swizzled or raster_along='n' launches would require
        # a deeper scheduler refactor to encode cluster Z in the scheduler's working id space.
        if self.split_k > 1 and (self.swizzle_size != 1 or self.raster_along != "m"):
            raise testing.CantImplementError(
                "split_k > 1 requires swizzle_size=1 and raster_along='m'"
            )
        # Accumulator elements per thread must be divisible by split_k
        if self.split_k > 1 and self._mailbox_elems_per_thread % self.split_k != 0:
            raise testing.CantImplementError(
                f"elems_per_thread={self._mailbox_elems_per_thread} must be divisible by split_k={self.split_k}"
            )

    def check_tensor_alignment(
        self,
        m: int,
        n: int,
        k: int,
        l: int,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
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
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :raises testing.CantImplementError: If the tensor alignment is invalid
        """

        # TODO: move to utils
        def check_contiguous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contiguous_16B_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contiguous_16B_alignment(b_dtype, b_major == "n", (n, k, l))
            or not check_contiguous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            raise testing.CantImplementError(
                f"Invalid tensor alignment: {m}, {n}, {k}, {l}, {a_dtype}, {b_dtype}, {c_dtype}, {a_major}, {b_major}, {c_major}"
            )

    def check_epilog_store_option(self, m: int, n: int):
        """
        Check if the epilogue store option is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int

        :raises testing.CantImplementError: If the epilogue store option is invalid
        """

        # Non TMA store version does not have predication, can not support OOB tiles
        cta_tile_shape_mn = (
            self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1),
            self.mma_tiler_mn[1],
        )
        if not self.use_tma_store:
            if not (m % cta_tile_shape_mn[0] == 0 and n % cta_tile_shape_mn[1] == 0):
                raise testing.CantImplementError(
                    f"Problem shape {m}, {n} must be divisible by cta tile shape {cta_tile_shape_mn} for non TMA store"
                )
            # Make sure the swizzle size divides the cta/cga count since non TMA epilouge don't support OOB tiles.
            # CTA swizzling improves the L2 cache utilization and reduces the number of cache misses.
            m_per_swizzle = (m // cta_tile_shape_mn[0]) // self.cluster_shape_mn[0]
            n_per_swizzle = (n // cta_tile_shape_mn[1]) // self.cluster_shape_mn[1]
            if (m_per_swizzle % self.swizzle_size != 0) or (n_per_swizzle % self.swizzle_size != 0):
                raise testing.CantImplementError(
                    f"Problem shape {m}, {n} must be divisible by swizzle size {self.swizzle_size} for non TMA store"
                )

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
        """
        Determine if the given tensor configuration can be implemented by this kernel.

        :param mnkl: Problem size as a tuple (M, N, K, L).
        :type mnkl: Tuple[int, int, int, int]
        :param a_dtype: Data type for input tensors A and B.
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: Data type for input tensors B and B.
        :type b_dtype: Type[cutlass.Numeric]
        :param c_dtype: Data type for output tensor C.
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: Major dimension of the A tensor layout ("m" or "k").
        :type a_major: str
        :param b_major: Major dimension of the B tensor layout ("n" or "k").
        :type b_major: str
        :param c_major: Major dimension of the C tensor layout ("m" or "n").
        :type c_major: str
        :return: True if the kernel supports the given configuration, False otherwise.
        :rtype: bool
        """

        try:
            # Skip unsupported types
            self.check_supported_dtypes(a_dtype, b_dtype, c_dtype)

            # Skip invalid mma tile shape and cluster shape
            self.check_mma_tiler_and_cluster_shape()

            m, n, k, l = mnkl
            self.check_tensor_alignment(
                m, n, k, l, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
            )
            self.check_epilog_store_option(m, n)
        except testing.CantImplementError:
            return False
        return True


@cute.jit
def bmm(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (l, m, k)
    b: cute.Tensor,  # (l, k, n)
    c: cute.Tensor,  # (l, m, n)
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    """
    Wrapper API for persistent GEMM kernel to follow the convention of PyTorch's batch matrix-multiply (bmm).

    Internally, the tensors are permuted to match CuTe's convention:
      - a: (m, k, l)
      - b: (n, k, l)
      - c: (m, n, l)

    :param gemm_op: Kernel operation, expects (a, b, c, stream, epilogue_op)
    :type gemm_op: cutlass.Constexpr
    :param a: Input tensor of shape (l, m, k)
    :type a: cute.Tensor
    :param b: Input tensor of shape (l, k, n)
    :type b: cute.Tensor
    :param c: Output tensor of shape (l, m, n)
    :type c: cute.Tensor
    :param epilogue_op: Optional elementwise lambda function to apply per output element, defaults to identity
    :type epilogue_op: cutlass.Constexpr, optional
    """
    # (l,m,k) -> (m,k,l)
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    # (l,k,n) -> (n,k,l)
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    # (l,m,n) -> (m,n,l)
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))

    gemm_op(a, b, c, stream, epilogue_op)


@lru_cache(maxsize=1)
def prepare_tensors(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    init_random: bool = True,
    normal_mean: float = 0.0,
    normal_std: float = 1.0,
):
    """Prepare tensors for GEMM.

    Returns:
        Tuple of (a_f32, b_f32, c_f32, a_storage, b_storage, c_storage):
        - *_f32: Float32 tensors with the logical data (for reference and fp8 conversion)
        - *_storage: Storage tensors for DLPack (uint8 for fp8, otherwise the target dtype)
    """
    import torch
    from cutlass.torch import dtype as torch_dtype

    m, n, k, l = mnkl

    if a_major == "k":
        a_f32 = torch.empty((l, m, k), dtype=torch.float32, device="cuda")
    elif a_major == "m":
        a_f32 = torch.empty((l, k, m), dtype=torch.float32, device="cuda").permute(0, 2, 1)

    if b_major == "n":
        b_f32 = torch.empty((l, k, n), dtype=torch.float32, device="cuda")
    elif b_major == "k":
        b_f32 = torch.empty((l, n, k), dtype=torch.float32, device="cuda").permute(0, 2, 1)

    if c_major == "n":
        c_f32 = torch.empty((l, m, n), dtype=torch.float32, device="cuda")
    elif c_major == "m":
        c_f32 = torch.empty((l, n, m), dtype=torch.float32, device="cuda").permute(0, 2, 1)

    # Initialize tensors with either uniform random or normal distribution
    for tensor in [a_f32, b_f32, c_f32]:
        if init_random:
            tensor.random_(-2, 3)
        else:
            tensor.normal_(mean=normal_mean, std=normal_std)

    # For float8 types, use uint8 as storage type to avoid dlpack limitation
    # (dlpack doesn't support float8 types)
    # For other types, convert to the target dtype
    a_storage_dtype = torch.uint8 if is_fp8_dtype(a_dtype) else torch_dtype(a_dtype)
    b_storage_dtype = torch.uint8 if is_fp8_dtype(b_dtype) else torch_dtype(b_dtype)
    c_storage_dtype = torch.uint8 if is_fp8_dtype(c_dtype) else torch_dtype(c_dtype)

    a_storage = a_f32.to(dtype=a_storage_dtype)
    b_storage = b_f32.to(dtype=b_storage_dtype)
    c_storage = c_f32.to(dtype=c_storage_dtype)

    return (a_f32, b_f32, c_f32, a_storage, b_storage, c_storage)


@lru_cache(maxsize=1)
def compile_bmm(
    mnkl: Tuple[int, int, int, int],
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler: Union[Tuple[int, int], Tuple[int, int, int]] = (256, 256),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    use_tma_store: bool = True,
    swizzle_size: int = 1,
    raster_along: Literal["m", "n"] = "m",
    epilogue_op: cutlass.Constexpr = lambda x: x,
    split_k: int = 1,
):
    from cutlass.cute.runtime import make_fake_stream

    # Build GEMM object
    gemm = PersistentDenseGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler,
        cluster_shape_mn,
        use_tma_store,
        swizzle_size,
        raster_along,
        split_k,
    )

    # Check if configuration can be implemented
    can_implement = gemm.can_implement(
        mnkl, a.element_type, b.element_type, c.element_type, a_major, b_major, c_major
    )
    if not can_implement:
        raise testing.CantImplementError(
            f"The current config which is invalid/unsupported: use_2cta_instrs = {use_2cta_instrs}, "
            f"mma_tiler = {mma_tiler}, cluster_shape_mn = {cluster_shape_mn}, "
            f"use_tma_store = {use_tma_store}, "
            f"swizzle_size = {swizzle_size}, "
            f"raster_along = {raster_along}, "
            f"split_k = {split_k}, "
            f"mnkl = {mnkl}"
        )

    stream = make_fake_stream()
    return cute.compile(bmm, gemm, a, b, c, stream, epilogue_op)


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    swizzle_size: int = 1,
    raster_along: Literal["m", "n"] = "m",
    use_2cta_instrs: bool = True,
    use_tma_store: bool = True,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    benchmark: bool = False,
    init_normal: bool = False,
    normal_mean: float = 0.0,
    normal_std: float = 1.0,
    split_k: int = 1,
    **kwargs,
):
    """
    Execute a persistent batched dense GEMM operation on Blackwell architecture with performance benchmarking.

    Prepares input tensors, configures and launches the persistent GEMM kernel,
    optionally performs reference validation, and benchmarks execution.

    :param mnkl: Problem size as a tuple (M, N, K, L).
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: Data type for input tensors A and B.
    :type ab_dtype: Type[cutlass.Numeric]
    :param c_dtype: Data type for output tensor C.
    :type c_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Accumulator data type for the matrix multiplication.
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major: Memory layout of tensor A.
    :type a_major: str
    :param b_major: Memory layout of tensor B.
    :type b_major: str
    :param c_major: Memory layout of tensor C.
    :type c_major: str
    :param mma_tiler_mn: MMA tiling size (M, N), defaults to (256, 256).
    :type mma_tiler_mn: Tuple[int, int], optional
    :param cluster_shape_mn: Cluster shape (M, N), defaults to (2, 1).
    :type cluster_shape_mn: Tuple[int, int], optional
    :param use_2cta_instrs: Whether to use 2CTA MMA instructions, defaults to True.
    :type use_2cta_instrs: bool, optional
    :param use_tma_store: Whether to use TMA store, defaults to True.
    :type use_tma_store: bool, optional
    :param tolerance: Tolerance for reference validation, defaults to 1e-01.
    :type tolerance: float, optional
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0.
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1.
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False.
    :type skip_ref_check: bool, optional
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False.
    :type use_cold_l2: bool, optional
    :param benchmark: Whether to only benchmark the kernel, defaults to False.
    :type benchmark: bool, optional
    :param init_normal: Whether to initialize tensors using normal distribution
        instead of uniform random, defaults to False.
    :type init_normal: bool, optional
    :param normal_mean: Mean for normal distribution initialization, defaults to 0.0.
    :type normal_mean: float, optional
    :param normal_std: Standard deviation for normal distribution initialization,
        defaults to 1.0.
    :type normal_std: float, optional
    :raises RuntimeError: If CUDA GPU is not available.
    :raises ValueError: If the configuration is invalid or unsupported by the kernel.
    :return: Execution time of the GEMM kernel.
    :rtype: float
    """
    import torch
    from cutlass.torch import dtype as torch_dtype

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # Run and verify BMM with torch
    a_f32, b_f32, c_f32, a_storage, b_storage, c_storage = prepare_tensors(
        mnkl,
        ab_dtype,
        ab_dtype,
        c_dtype,
        a_major,
        b_major,
        c_major,
        init_random=not init_normal,
        normal_mean=normal_mean,
        normal_std=normal_std,
    )

    leading_dim_a = 2 if a_major == "k" else 1
    leading_dim_b = 1 if b_major == "k" else 2
    leading_dim_c = 2 if c_major == "n" else 1

    # Create CuTe tensors, passing float32 source for fp8 conversion
    a_ = create_cute_tensor_for_fp8(a_storage, ab_dtype, leading_dim_a, source_f32_tensor=a_f32)
    b_ = create_cute_tensor_for_fp8(b_storage, ab_dtype, leading_dim_b, source_f32_tensor=b_f32)
    c_ = create_cute_tensor_for_fp8(c_storage, c_dtype, leading_dim_c, source_f32_tensor=c_f32)

    print("Compile Blackwell Persistent Dense GEMM with:")
    print(f"ab_dtype: {ab_dtype}, c_dtype: {c_dtype}, acc_dtype: {acc_dtype}")
    print(f"a_major: {a_major}, b_major: {b_major}, c_major: {c_major}")
    print(f"mma_tiler_mn: {mma_tiler_mn}, cluster_shape_mn: {cluster_shape_mn}")
    print(f"use_2cta_instrs: {use_2cta_instrs}, use_tma_store: {use_tma_store}")
    print(f"swizzle_size: {swizzle_size}, raster_along: {raster_along}")
    print(f"split_k: {split_k}")

    compiled_fn = compile_bmm(
        mnkl,
        a_,
        b_,
        c_,
        acc_dtype,
        a_major,
        b_major,
        c_major,
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs,
        use_tma_store,
        epilogue_op=lambda x: x,
        swizzle_size=swizzle_size,
        raster_along=raster_along,
        split_k=split_k,
    )

    print("Running Blackwell Persistent Dense GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    if not skip_ref_check:
        # Use small random number for deterministic result for reference check
        compiled_fn(a_, b_, c_, current_stream)

        # Manually quantize to be comparable
        # Use float32 source data for reference calculation
        ref = torch.bmm(a_f32, b_f32).to(dtype=torch_dtype(c_dtype)).to(dtype=torch.float32)
        torch.testing.assert_close(
            c_storage.view(torch_dtype(c_dtype)).to(dtype=torch.float32),
            ref,
            atol=tolerance,
            rtol=1e-03,
        )

    if not benchmark:
        return 0

    def generate_tensors():
        # Use init_normal from outer scope, but force random init for Int8/Uint8 types
        use_normal_init = init_normal and (ab_dtype not in [cutlass.Int8, cutlass.Uint8])
        a_f32, b_f32, c_f32, a_st, b_st, c_st = prepare_tensors(
            mnkl,
            ab_dtype,
            ab_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
            init_random=not use_normal_init,
            normal_mean=normal_mean,
            normal_std=normal_std,
        )
        a_ = create_cute_tensor_for_fp8(a_st, ab_dtype, leading_dim_a, source_f32_tensor=a_f32)
        b_ = create_cute_tensor_for_fp8(b_st, ab_dtype, leading_dim_b, source_f32_tensor=b_f32)
        c_ = create_cute_tensor_for_fp8(c_st, c_dtype, leading_dim_c, source_f32_tensor=c_f32)
        return testing.JitArguments(a_, b_, c_, current_stream)

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_storage.numel() * a_storage.element_size()
            + b_storage.numel() * b_storage.element_size()
            + c_storage.numel() * c_storage.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    # Return execution time in microseconds
    exec_time = testing.benchmark(
        compiled_fn,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    print(f"[DSL INFO] Execution time: {exec_time} microseconds per iteration")
    return exec_time


def benchmark_nsight(
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    use_tma_store: bool,
) -> None:
    import torch

    try:
        import nsight

        print("[DSL INFO] nsight is installed, running nsight benchmark")
    except ImportError:
        print("nsight-python is not installed, skipping Nsight benchmark")
        return

    from cutlass.torch import dtype as torch_dtype

    def compute_tflops(time_ns, m, n, k):
        return 2.0 * m * n * k / time_ns / 1000.0

    @nsight.analyze.plot(
        "dense_gemm_persistent.png",
        title="Blackwell Dense Persistent GEMM example TFLOPS",
        ylabel="Performance (TFLOPS)",  # Custom y-axis label
        annotate_points=True,  # Show values on the plot
    )
    @nsight.analyze.kernel(
        configs=[(2**i, 2**i, 2**i) for i in range(10, 15)],
        runs=20,
        derive_metric=compute_tflops,
    )
    def _details(m, n, k):
        mnkl = (m, n, k, 1)

        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        a_f32, b_f32, c_f32, a_storage, b_storage, c_storage = prepare_tensors(
            mnkl, ab_dtype, ab_dtype, c_dtype, a_major, b_major, c_major
        )

        leading_dim_a = 2 if a_major == "k" else 1
        leading_dim_b = 1 if b_major == "k" else 2
        leading_dim_c = 2 if c_major == "n" else 1

        a_ = create_cute_tensor_for_fp8(a_storage, ab_dtype, leading_dim_a, source_f32_tensor=a_f32)
        b_ = create_cute_tensor_for_fp8(b_storage, ab_dtype, leading_dim_b, source_f32_tensor=b_f32)
        c_ = create_cute_tensor_for_fp8(c_storage, c_dtype, leading_dim_c, source_f32_tensor=c_f32)

        compiled_fn = compile_bmm(
            mnkl,
            a_,
            b_,
            c_,
            acc_dtype,
            a_major,
            b_major,
            c_major,
            mma_tiler_mn,
            cluster_shape_mn,
            use_2cta_instrs,
            use_tma_store,
        )

        with nsight.annotate("CuTe DSL BMM"):
            compiled_fn(a_, b_, c_, current_stream)

        with nsight.annotate("PyTorch BMM"):
            with torch.cuda.current_stream():
                # Use float32 data for PyTorch BMM (fp8 is converted internally)
                torch.bmm(a_f32, b_f32).to(dtype=torch_dtype(c_dtype))

    _details()


def _parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from error


def prepare_parser():
    parser = argparse.ArgumentParser(description="Example of Dense Persistent GEMM on Blackwell.")

    parser.add_argument(
        "--mnkl",
        type=_parse_comma_separated_ints,
        default=(256, 256, 512, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.TFloat32)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)

    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="default",
        choices=[
            "nsight",
            "default",
            "none",
        ],
        help="Benchmark the kernel with nsight or default (cutlass.testing.benchmark) or none",
    )
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )
    testing.add_tensor_init_args(parser, supports_int_dtypes=True)

    return parser


if __name__ == "__main__":
    parser = prepare_parser()

    # Kernel Configurations
    parser.add_argument("--use_tma_store", action="store_true", help="Use tma store or not")
    parser.add_argument(
        "--cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )

    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )

    parser.add_argument(
        "--mma_tiler_mn",
        type=_parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )

    parser.add_argument(
        "--swizzle_size",
        type=int,
        default=1,
        help="Swizzling size in the unit of cluster for improving L2 cache hit rate",
    )
    parser.add_argument(
        "--raster_order",
        type=str,
        choices=["m", "n"],
        default="m",
        help="Rasterization order of clusters",
    )
    parser.add_argument(
        "--cluster_split_k",
        type=int,
        default=1,
        help="cluster split-k factor (CTAs along cluster Z that split K and reduce via DSMEM)",
    )

    args = parser.parse_args()

    testing.validate_tensor_init_args(args, parser)

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    print("[DSL INFO] Compiling Blackwell Persistent Dense GEMM with:")
    print(
        f"[DSL INFO] A dtype: {args.ab_dtype}, B dtype: {args.c_dtype}, C dtype: {args.acc_dtype}, Acc dtype: {args.acc_dtype}"
    )
    print(f"[DSL INFO] Matrix majors - A: {args.a_major}, B: {args.b_major}, C: {args.c_major}")
    print(f"[DSL INFO] Mma Tiler (M, N): {args.mma_tiler_mn}")
    print(f"[DSL INFO] Cluster Shape (M, N): {args.cluster_shape_mn}")
    print(f"[DSL INFO] 2CTA MMA instructions: {'True' if args.use_2cta_instrs else 'False'}")
    print(f"[DSL INFO] Use TMA Store: {'True' if args.use_tma_store else 'False'}")
    print(f"[DSL INFO] Cluster Split-K: {args.cluster_split_k}")

    if args.benchmark == "nsight":
        benchmark_nsight(
            args.ab_dtype,
            args.c_dtype,
            args.acc_dtype,
            args.a_major,
            args.b_major,
            args.c_major,
            args.mma_tiler_mn,
            args.cluster_shape_mn,
            args.use_2cta_instrs,
            args.use_tma_store,
        )
        exit(0)

    run(
        args.mnkl,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.swizzle_size,
        args.raster_order,
        args.use_2cta_instrs,
        args.use_tma_store,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.benchmark == "default",
        args.init_normal,
        args.normal_mean,
        args.normal_std,
        args.cluster_split_k,
    )
    print("PASS")
