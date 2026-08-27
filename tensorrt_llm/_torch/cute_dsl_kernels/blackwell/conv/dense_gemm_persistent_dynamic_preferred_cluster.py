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

# ruff: noqa: E501

import argparse
from functools import lru_cache
from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import testing
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import create_cute_tensor_for_fp8

from .dense_gemm_persistent_dynamic import PersistentDenseGemmKernel, prepare_tensors

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

    python examples/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic_preferred_cluster.py                          \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
      --mma_tiler_mn 256,128                                                   \
      --preferred_cluster_shape_mn 4,2  --fallback_cluster_shape_mn 2,1        \
      --mnkl 8192,8192,8192,1                                                   \
      --use_tma_store --use_2cta_instrs

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic_preferred_cluster.py                     \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                 \
      --mma_tiler_mn 256,128                                                   \
      --preferred_cluster_shape_mn 4,2  --fallback_cluster_shape_mn 2,1        \
      --mnkl 8192,8192,8192,1                                                  \
      --use_tma_store --use_2cta_instrs                                        \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints are same as dense_gemm.py:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below PersistentDenseGemmKernelDynamicPreferredCluster class documentation
* A/B tensor must have the same data type
* Mma tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* Mma tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* Preferred cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Preferred cluster shape M must be multiple of 2 if use_2cta_instrs=True
* Preferred cluster shape M/N must be multiple of cluster shape M/N
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


class PersistentDenseGemmKernelDynamicPreferredCluster(PersistentDenseGemmKernel):
    """This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param fallback_cluster_shape_mn: Fallback cluster dimensions (M,N) for parallel processing
    :type fallback_cluster_shape_mn: Tuple[int, int]
    :param use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
    :type use_tma_store: bool
    :param preferred_cluster_shape_mn: Preferred cluster dimensions (M,N) for optimal performance. Defaults to (4, 4).
    :type preferred_cluster_shape_mn: Tuple[int, int]

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
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Preferred cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Preferred cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Preferred cluster shape M/N must be multiple of cluster shape M/N

    Example:
        >>> gemm = PersistentDenseGemmKernelDynamicPreferredCluster(
        ...     acc_dtype=cutlass.Float32,
        ...     use_2cta_instrs=True,
        ...     mma_tiler_mn=(128, 128),
        ...     fallback_cluster_shape_mn=(2, 1),
        ...     preferred_cluster_shape_mn=(4, 2),
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel with CLC dynamic scheduling and preferred cluster support.

        This extends the base PersistentDenseGemmKernel with support for:
        - CLC dynamic persistent scheduling for better load balancing
        - Preferred and fallback cluster shapes for mixed cluster shape optimization

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param preferred_cluster_shape_mn: Preferred cluster shape for optimal performance.
        :type preferred_cluster_shape_mn: Tuple[int, int]
        :param fallback_cluster_shape_mn: Tuple (ClusterM, ClusterN) fallback shape of the cluster.
        :type fallback_cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Use Tensor Memory Access (TMA) or normal store for output C tensor.
        :type use_tma_store: bool
        """
        # Initialize base class with CLC dynamic scheduler and fallback cluster shape
        super().__init__(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=fallback_cluster_shape_mn,
            use_tma_store=use_tma_store,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
        )

        # Add preferred cluster specific attributes
        self.preferred_cluster_shape_mn = preferred_cluster_shape_mn
        self.fallback_cluster_shape_mn = fallback_cluster_shape_mn

        # Override barrier IDs for dynamic preferred cluster version
        self.cta_sync_bar_id = 0
        # tmem_alloc_sync_bar_id, tmem_dealloc_sync_bar_id already set by parent

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
        """Set up configurations for dynamic preferred cluster GEMM.

        Extends the base _setup_attributes to add:
        - Preferred cluster layout computation
        - Multicast CTA counts for both preferred and fallback clusters
        """
        # Set up the base CLC scheduler attributes.
        super()._setup_attributes()

        # Compute preferred cluster layout (in addition to fallback already set by parent)
        tiled_mma = self._create_tiled_mma()
        self.preferred_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        # Calculate multicast CTA counts for preferred cluster
        self.num_preferred_mcast_ctas_a = cute.size(self.preferred_cluster_layout_vmnk.shape[2])
        self.num_preferred_mcast_ctas_b = cute.size(self.preferred_cluster_layout_vmnk.shape[1])
        self.is_preferred_a_mcast = self.num_preferred_mcast_ctas_a > 1
        self.is_preferred_b_mcast = self.num_preferred_mcast_ctas_b > 1

        self.fallback_cluster_layout_vmnk = self.cluster_layout_vmnk
        self.num_fallback_mcast_ctas_a = self.num_mcast_ctas_a
        self.num_fallback_mcast_ctas_b = self.num_mcast_ctas_b
        self.is_fallback_a_mcast = self.is_a_mcast
        self.is_fallback_b_mcast = self.is_b_mcast

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

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        # Setup preferred TMA load for A
        a_op_preferred = utils.sm100.cluster_shape_to_tma_atom_A(
            self.preferred_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_a_preferred, tma_tensor_a_preferred = cute.nvgpu.make_tiled_tma_atom_A(
            a_op_preferred,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        # Setup fallback TMA load for A
        a_op_fallback = utils.sm100.cluster_shape_to_tma_atom_A(
            self.fallback_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_a_fallback, tma_tensor_a_fallback = cute.nvgpu.make_tiled_tma_atom_A(
            a_op_fallback,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        # Setup b_smem_layout
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))

        # Setup preferred TMA load for B
        b_op_preferred = utils.sm100.cluster_shape_to_tma_atom_B(
            self.preferred_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_b_preferred, tma_tensor_b_preferred = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_preferred,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        # Setup fallback TMA load for B
        b_op_fallback = utils.sm100.cluster_shape_to_tma_atom_B(
            self.fallback_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_b_fallback, tma_tensor_b_fallback = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_fallback,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
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

        # Compute grid size and scheduler params for both cluster shapes
        # regular cluster shape
        self.fallback_tile_sched_params, _ = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.fallback_cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
        )

        # preferred cluster shape
        self.preferred_tile_sched_params, preferred_grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.preferred_cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.preferred_cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.preferred_tile_sched_params,
            self.fallback_tile_sched_params,
            epilogue_op,
        ).launch(
            grid=preferred_grid,  # pass grid computed with preferred cluster shape
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.preferred_cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a_preferred: cute.CopyAtom,
        mA_mkl_preferred: cute.Tensor,
        tma_atom_a_fallback: cute.CopyAtom,
        mA_mkl_fallback: cute.Tensor,
        tma_atom_b_preferred: cute.CopyAtom,
        mB_nkl_preferred: cute.Tensor,
        tma_atom_b_fallback: cute.CopyAtom,
        mB_nkl_fallback: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        preferred_cluster_layout_vmnk: cute.Layout,
        fallback_cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        preferred_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        fallback_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel entry point for kernel with preferred cluster shape.
        """
        # Get cluster coordinates to determine if this is a preferred cluster
        cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
        is_preferred_cluster = (
            cbdim_x == self.preferred_cluster_shape_mn[0]
            and cbdim_y == self.preferred_cluster_shape_mn[1]
            and cbdim_z == 1
        )

        # as for now, only support preferred cluster kernel via the megakernel approach
        # mega-kernel approach has 2 mutually exclusive code branches, only one path runs per launch,
        # specify `smem_merge_branch_allocs=True` at launch to enables shared memory reuse between two paths
        if is_preferred_cluster:
            self.cluster_specific_kernel(
                tiled_mma,
                tma_atom_a_preferred,
                mA_mkl_preferred,
                tma_atom_b_preferred,
                mB_nkl_preferred,
                tma_atom_c,
                mC_mnl,
                preferred_cluster_layout_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                c_smem_layout_staged,
                epi_tile,
                preferred_tile_sched_params,
                epilogue_op,
                self.is_preferred_a_mcast,
                self.is_preferred_b_mcast,
                self.preferred_cluster_shape_mn,
            )
        else:
            self.cluster_specific_kernel(
                tiled_mma,
                tma_atom_a_fallback,
                mA_mkl_fallback,
                tma_atom_b_fallback,
                mB_nkl_fallback,
                tma_atom_c,
                mC_mnl,
                fallback_cluster_layout_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                c_smem_layout_staged,
                epi_tile,
                fallback_tile_sched_params,
                epilogue_op,
                self.is_fallback_a_mcast,
                self.is_fallback_b_mcast,
                self.fallback_cluster_shape_mn,
            )

    @cute.jit()
    def cluster_specific_kernel(
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
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        effective_is_a_mcast: bool,
        effective_is_b_mcast: bool,
        cluster_shape: Tuple[int, int],
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
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response_align_bytes = self.num_clc_response_bytes
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4],
                clc_response_align_bytes,
            ]

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
            cta_layout_vmnk=cluster_layout_vmnk,
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
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize clc_pipeline (barrier) and states
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_shape)
        num_clc_consumer_threads = 32 * (1 + cluster_size * (1 + len(self.epilogue_warp_id) + 1))
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

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_shape, is_relaxed=True)

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
        if cutlass.const_expr(effective_is_a_mcast or effective_is_b_mcast or use_2cta_instrs):
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
        pipeline_init_wait(cluster_shape_mn=cluster_shape)

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
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
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
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

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

            #
            # Persistent tile scheduling loop for epilogue
            #
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
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                num_tiles_executed = tile_sched.num_tiles_executed
                if cutlass.const_expr(self.use_tma_store):
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
            else:
                # Synchronize before TMEM dealloc (done by the caller)
                tmem_dealloc_barrier.arrive_and_wait()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


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
    Wrapper API for persistent GEMM preferred cluster kernel to follow the convention of PyTorch's batch matrix-multiply (bmm).

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
    preferred_cluster_shape_mn: Tuple[int, int] = (2, 1),
    fallback_cluster_shape_mn: Tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    use_tma_store: bool = True,
    swizzle_size: int = 1,
    raster_along: Literal["m", "n"] = "m",
    use_tvm_ffi: bool = False,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    from cutlass.cute.runtime import make_fake_stream

    # Build GEMM object
    gemm = PersistentDenseGemmKernelDynamicPreferredCluster(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler,
        preferred_cluster_shape_mn,
        fallback_cluster_shape_mn,
        use_tma_store,
        swizzle_size,
        raster_along,
    )

    # Check if configuration can be implemented
    can_implement = gemm.can_implement(
        mnkl, a.element_type, b.element_type, c.element_type, a_major, b_major, c_major
    )
    if not can_implement:
        raise testing.CantImplementError(
            f"The current config which is invalid/unsupported: use_2cta_instrs = {use_2cta_instrs}, "
            f"mma_tiler = {mma_tiler}, "
            f"preferred_cluster_shape_mn = {preferred_cluster_shape_mn}, fallback_cluster_shape_mn = {fallback_cluster_shape_mn}, "
            f"use_tma_store = {use_tma_store}, "
            f"swizzle_size = {swizzle_size}, "
            f"raster_along = {raster_along}"
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
    preferred_cluster_shape_mn: Tuple[int, int] = (2, 1),
    fallback_cluster_shape_mn: Tuple[int, int] = (2, 1),
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

    # Prepare tensors
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
        preferred_cluster_shape_mn,
        fallback_cluster_shape_mn,
        use_2cta_instrs,
        use_tma_store,
        epilogue_op=lambda x: x,
        swizzle_size=swizzle_size,
        raster_along=raster_along,
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
        from cutlass.torch import dtype as torch_dtype

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
    parser.add_argument(
        "--mma_tiler_mn",
        type=_parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--preferred_cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(4, 4),
        help="Preferred cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--fallback_cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(2, 1),
        help="Fallback cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.TFloat32)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--use_tma_store", action="store_true", help="Use tma store or not")
    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
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
        "--benchmark",
        type=str,
        default="default",
        choices=[
            "default",
            "none",
        ],
        help="Benchmark the kernel with nsight or default (cutlass.testing.benchmark) or none",
    )
    testing.add_tensor_init_args(parser, supports_int_dtypes=True)
    return parser


if __name__ == "__main__":
    parser = prepare_parser()

    args = parser.parse_args()

    testing.validate_tensor_init_args(args, parser)

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.preferred_cluster_shape_mn) != 2:
        parser.error("--preferred_cluster_shape_mn must contain exactly 2 values")

    if len(args.fallback_cluster_shape_mn) != 2:
        parser.error("--fallback_cluster_shape_mn must contain exactly 2 values")

    print("[DSL INFO] Compiling Blackwell Persistent Dense GEMM with:")
    print(
        f"[DSL INFO] A dtype: {args.ab_dtype}, B dtype: {args.c_dtype}, C dtype: {args.acc_dtype}, Acc dtype: {args.acc_dtype}"
    )
    print(f"[DSL INFO] Matrix majors - A: {args.a_major}, B: {args.b_major}, C: {args.c_major}")
    print(f"[DSL INFO] Mma Tiler (M, N): {args.mma_tiler_mn}")
    print(f"[DSL INFO] Preferred Cluster Shape (M, N): {args.preferred_cluster_shape_mn}")
    print(f"[DSL INFO] Fallback Cluster Shape (M, N): {args.fallback_cluster_shape_mn}")
    print(f"[DSL INFO] 2CTA MMA instructions: {'True' if args.use_2cta_instrs else 'False'}")
    print(f"[DSL INFO] Use TMA Store: {'True' if args.use_tma_store else 'False'}")

    run(
        args.mnkl,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.preferred_cluster_shape_mn,
        args.fallback_cluster_shape_mn,
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
    )
    print("PASS")
