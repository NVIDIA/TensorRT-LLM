# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from ..utils import (
    atomic_add_func,
    vectorized_atomic_add_bf16x8,
    vectorized_atomic_add_fp16x8,
    vectorized_atomic_add_fp32x2,
)

"""
This example provides an experimental implementation of the SM100 batched dense blockscaled
GEMM kernel, please note that the APIs and implementation details related to this kernel
may change in future releases.

A high-performance persistent batched dense blockscaled GEMM example for the NVIDIA Blackwell
SM100 architecture using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
  for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
  for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk,
  which has M×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk,
  which has N×ceil_div(K, sf_vec_size)×L elements respectively

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp:
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global
      memory (GMEM) with TMA operations, or directly store C matrix from registers (RMEM)
      to global memory (GMEM) without TMA operations.
    - Optionally accept an elementwise lambda function epilogue_op to apply to the output tensor:
      e.g., relu can set epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is shown below:

.. code-block:: bash

    python examples/blackwell/dense_blockscaled_gemm_persistent.py            \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_blockscaled_gemm_persistent.py        \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1                                                  \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 128 or 256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""


class Sm100BlockScaledPersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

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
        # Note: We don't have SFD generation support in this example for now,
        # so Float4E2M1FN output is only for internal testing and will not be released.
        - Float4E2M1FN

    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        # TODO: Add 64 and 192 support
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        expert_count: int,
        weight_per_expert: int,
        use_prefetch: bool = False,
        prefetch_dist: int = 3,
        split_k: int = 1,
        swap_ab: bool = False,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param expert_count: The number of experts
        :type expert_count: int
        :param weight_per_expert: The number of weights per expert
        :type weight_per_expert: int
        :param use_prefetch: Enable prefetch operations (default: False).
        :type use_prefetch: bool
        :param prefetch_dist: Prefetch distance for TMA operations (default: 3).
        :type prefetch_dist: int
        """

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.alpha_scale_load_warp_id = 6
        self.dummy_warp_id = 7
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.epilog_warp_id,
                self.alpha_scale_load_warp_id,
                self.dummy_warp_id,
            )
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
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
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.expert_count = expert_count
        self.weight_per_expert = weight_per_expert

        self.use_prefetch = use_prefetch
        self.prefetch_dist = prefetch_dist
        self.split_k = split_k
        self.swap_ab = swap_ab

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
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        # TODO: round up to 128, it is prepared for supporting N=64 or 192.
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

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
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
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
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        # For swapAB: use ROW_MAJOR for epi_tile computation to keep consistent
        # thread-to-(M,N) mapping. The actual C layout (COL_MAJOR) only affects
        # the TMA store path, not the register computation.
        self.epi_compute_layout = utils.LayoutEnum.ROW_MAJOR if self.swap_ab else self.c_layout
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.epi_compute_layout,
            self.c_dtype,
        )
        self.epi_tile_m_size = cute.size(self.epi_tile[0])
        self.epi_tile_n_size = cute.size(self.epi_tile[1])

        # Atomic add parameters for split-K epilogue
        if self.split_k > 1:
            epi_tile_m = cute.size(self.epi_tile[0])
            epi_tile_n = cute.size(self.epi_tile[1])
            num_epilogue_threads = 32 * len(self.epilog_warp_id)
            self.ttr_racc_size = (epi_tile_m * epi_tile_n) // num_epilogue_threads
            if self.swap_ab:
                # SwapAB: C is M-major, N elements are strided → scalar atomic add
                self.epi_layout_atomic = cute.make_layout(shape=(self.ttr_racc_size,), stride=(1,))
                self.epi_loop_size_atomic = self.ttr_racc_size
                self.element_offset_atomic = 1
            elif self.c_dtype in (cutlass.Float16, cutlass.BFloat16):
                self.epi_layout_atomic = cute.make_layout(
                    shape=(self.ttr_racc_size // 8, 4, 2), stride=(8, 2, 1)
                )
                self.epi_loop_size_atomic = self.ttr_racc_size // 8
                self.element_offset_atomic = 8
            elif self.c_dtype == cutlass.Float32:
                self.epi_layout_atomic = cute.make_layout(
                    shape=(self.ttr_racc_size // 2, 2), stride=(2, 1)
                )
                self.epi_loop_size_atomic = self.ttr_racc_size // 2
                self.element_offset_atomic = 2
            else:
                self.epi_layout_atomic = cute.make_layout(shape=(self.ttr_racc_size,), stride=(1,))
                self.epi_loop_size_atomic = self.ttr_racc_size
                self.element_offset_atomic = 1

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage, self.num_alpha_scale_stage = (
            self._compute_stages(
                tiled_mma,
                self.mma_tiler,
                self.a_dtype,
                self.b_dtype,
                self.epi_tile,
                self.c_dtype,
                self.epi_compute_layout,
                self.sf_dtype,
                self.sf_vec_size,
                self.smem_capacity,
                self.occupancy,
            )
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

        # SwapAB: alpha is per-N, smem holds cta_tile_N values per stage
        # Non-swap: alpha is per-M, smem holds cta_tile_M values per stage
        self.alpha_scale_dim = (
            self.cta_tile_shape_mnk[1] if self.swap_ab else self.cta_tile_shape_mnk[0]
        )
        self.alpha_scale_smem_layout_staged = cute.make_layout(
            (
                self.alpha_scale_dim,
                self.num_alpha_scale_stage,
            ),
            stride=(
                1,
                self.alpha_scale_dim,
            ),
        )

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        alpha_scale_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
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
        :param alpha_scale_tensor: Alpha scale tensor
        :type alpha_scale_tensor: cute.Tensor
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
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

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
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator,
                cute.make_layout(new_shape, stride=new_stride),
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

        # Compute grid size (inflated by split_k for split-K decomposition)
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.split_k,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
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
            # Alpha scale pipeline mbarriers (producer=warp6, consumer=epilogue warps)
            alpha_scale_load_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_alpha_scale_stage * 2
            ]
            # Alpha scale shared memory
            sAlphaScale: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(self.alpha_scale_smem_layout_staged)
                ],
                16,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            alpha_scale_tensor,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.alpha_scale_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            c_tensor,
            self.epi_layout_atomic if self.split_k > 1 else cute.make_layout(1),
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        malpha_scale_mnl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        alpha_scale_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        mC_raw: cute.Tensor,
        epi_layout_atomic: cute.Layout,
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
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
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
        )

        # Initialize alpha_scale_pipeline (barrier) and states
        alpha_scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            32 * 1,  # alpha_scale_load_warp_id threads
            32 * 1,
        )
        alpha_scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            32 * len(self.epilog_warp_id),  # epilogue warps
            32 * len(self.epilog_warp_id),
        )
        alpha_scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.alpha_scale_load_mbar_ptr.data_ptr(),
            num_stages=self.num_alpha_scale_stage,
            producer_group=alpha_scale_pipeline_producer_group,
            consumer_group=alpha_scale_pipeline_consumer_group,
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
        # Alpha scale shared memory tensor
        sAlphaScale = storage.sAlphaScale.get_tensor(alpha_scale_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
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

        # Tile alpha by alpha_scale_dim (cta_tile_N for swap_ab, cta_tile_M for non-swap)
        alpha_tiler = (self.alpha_scale_dim, 1, 1)
        galpha_scale_mnl = cute.local_tile(
            malpha_scale_mnl,
            cute.slice_(alpha_tiler, (None, 0, 0)),
            (None, None, None),
        )

        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_total = cute.size(gA_mkl, mode=[3])
        # For split-K: each CTA processes k_tile_total // split_k K-tiles
        k_tiles_per_split = k_tile_total // self.split_k
        k_tile_cnt_local = cutlass.Int32(k_tiles_per_split)

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

        # TMA load SFA partition_S/D
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
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

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

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx

                # Split-K: decompose L coord into batch_idx and split_k_idx
                # Input tensors use batch_idx for L; K-tiles offset by k_start
                # For split_k=1: batch_idx = coord[2], k_start = 0
                batch_idx = cur_tile_coord[2] // self.split_k
                k_start = (cur_tile_coord[2] % self.split_k) * k_tile_cnt_local

                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    batch_idx,
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
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                prefetch_dist = self.prefetch_dist
                # Sending a batch of inflight Prefetches before starting TMALDG loop
                # Prefetch logic: use_prefetch for both A&B, or explicit A-only/B-only
                if self.use_prefetch:
                    # Prefetch both A and B (default behavior)
                    for k_tile in cutlass.range(0, min(prefetch_dist, k_tile_cnt_local), unroll=1):
                        # Prefetch both A and B (default behavior)
                        cute.prefetch(
                            tma_atom_a,
                            tAgA_slice[(None, k_tile + k_start)],
                        )
                        cute.prefetch(
                            tma_atom_b,
                            tBgB_slice[(None, k_tile + k_start)],
                        )
                        cute.prefetch(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, k_tile + k_start)],
                        )
                        cute.prefetch(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, k_tile + k_start)],
                        )

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt_local:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt_local, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    # TMA load A/B/SFA/SFB (offset by k_start for split-K)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count + k_start)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count + k_start)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count + k_start)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count + k_start)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Prefetch logic in the loop: use_prefetch for both A&B, or explicit A-only/B-only
                    if k_tile < k_tile_cnt_local - prefetch_dist:
                        if self.use_prefetch:
                            # Prefetch both A and B (default behavior)
                            cute.prefetch(
                                tma_atom_a,
                                tAgA_slice[
                                    (None, ab_producer_state.count + k_start + prefetch_dist)
                                ],
                            )
                            cute.prefetch(
                                tma_atom_b,
                                tBgB_slice[
                                    (None, ab_producer_state.count + k_start + prefetch_dist)
                                ],
                            )
                            cute.prefetch(
                                tma_atom_sfa,
                                tAgSFA_slice[
                                    (None, ab_producer_state.count + k_start + prefetch_dist)
                                ],
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[
                                    (None, ab_producer_state.count + k_start + prefetch_dist)
                                ],
                            )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt_local:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized Alpha Scale Load warp
        #
        if warp_idx == self.alpha_scale_load_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            alpha_scale_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_alpha_scale_stage
            )

            # Setup copy atom for alpha scale loading
            atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                malpha_scale_mnl.element_type,
                num_bits_per_copy=malpha_scale_mnl.element_type.width,
            )
            tiled_copy_alpha_scale = cute.make_tiled_copy_tv(
                atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
            )
            thr_copy_alpha_scale = tiled_copy_alpha_scale.get_slice(cute.arch.lane_idx())

            tiles_per_expert_w6 = self.weight_per_expert // self.mma_tiler[2]
            experts_per_split_w6 = k_tile_cnt_local // tiles_per_expert_w6

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx

                # Split-K: decompose L coord into batch_idx and expert_offset
                batch_idx_w6 = cur_tile_coord[2] // self.split_k
                expert_offset_w6 = (cur_tile_coord[2] % self.split_k) * experts_per_split_w6

                # Reset producer state for this tile
                alpha_scale_producer_state.reset_count()
                peek_alpha_scale_empty_status = cutlass.Boolean(1)
                if alpha_scale_producer_state.count < k_tile_cnt_local:
                    peek_alpha_scale_empty_status = alpha_scale_pipeline.producer_try_acquire(
                        alpha_scale_producer_state
                    )

                # Non-swap: alpha indexed by M tile coord; SwapAB: alpha indexed by N tile coord
                if cutlass.const_expr(self.swap_ab):
                    alpha_tile_idx = cur_tile_coord[1]
                else:
                    alpha_tile_idx = cur_tile_coord[0]
                galpha_scale_mnl_current_tile = galpha_scale_mnl[
                    None, alpha_tile_idx, None, batch_idx_w6
                ]

                tAgAlphaScale = thr_copy_alpha_scale.partition_S(galpha_scale_mnl_current_tile)
                tAsAlphaScale = thr_copy_alpha_scale.partition_D(sAlphaScale)

                # Load alpha scale for each k tile (one per expert in this split)
                for k_tile in cutlass.range(0, k_tile_cnt_local, 1, unroll=1):
                    # Calculate expert index for this k_tile (with split-K offset)
                    expert_idx = expert_offset_w6 + k_tile // tiles_per_expert_w6

                    # Slice alpha scale for current tile and expert
                    tAgAlphaScale_slice = tAgAlphaScale[(None, None, expert_idx)]
                    tAsAlphaScale_slice = tAsAlphaScale[
                        (None, None, alpha_scale_producer_state.index)
                    ]

                    # Wait for alpha scale buffer empty
                    alpha_scale_pipeline.producer_acquire(
                        alpha_scale_producer_state, peek_alpha_scale_empty_status
                    )

                    num_iters = cute.size(tAgAlphaScale_slice, mode=[1])

                    # Load alpha scale from global to shared memory
                    for iter_idx in cutlass.range(num_iters, unroll_full=True):
                        iter_coord = (None, iter_idx)
                        pred = cutlass.Boolean(
                            32 * iter_idx + cute.arch.lane_idx() < malpha_scale_mnl.shape[0]
                        )
                        if pred:
                            cute.copy(
                                tiled_copy_alpha_scale,
                                tAgAlphaScale_slice[iter_coord],
                                tAsAlphaScale_slice[iter_coord],
                            )

                    # Commit and advance
                    alpha_scale_pipeline.producer_commit(alpha_scale_producer_state)
                    alpha_scale_producer_state.advance()

                    # Peek next
                    peek_alpha_scale_empty_status = cutlass.Boolean(1)
                    if alpha_scale_producer_state.count < k_tile_cnt_local:
                        peek_alpha_scale_empty_status = alpha_scale_pipeline.producer_try_acquire(
                            alpha_scale_producer_state
                        )

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for alpha scale buffer empty
            alpha_scale_pipeline.producer_tail(alpha_scale_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
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
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
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
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
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
            #
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

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # Shift TMEM start address for odd tiles (ignores first 64 cols of SFB)
                    offset = (
                        cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt_local and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Mma mainloop with expert grouping
                # Accumulate tiles_per_expert k-tiles per acc buffer to halve
                # acc pipeline traffic (512 → 256 round-trips).
                # For split-K: only process k_tiles_per_split K-tiles (a subset of experts).
                #
                tiles_per_expert_mma = self.weight_per_expert // self.mma_tiler[2]
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                for k_tile in range(k_tiles_per_split):
                    is_first_of_expert = k_tile % tiles_per_expert_mma == 0
                    is_last_of_expert = k_tile % tiles_per_expert_mma == tiles_per_expert_mma - 1

                    if is_first_of_expert:
                        tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                    if is_leader_cta:
                        if is_first_of_expert:
                            acc_pipeline.producer_acquire(acc_producer_state)

                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
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

                        if is_first_of_expert:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

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

                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt_local:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                    if is_last_of_expert:
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
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
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
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rAcc_final,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            # SwapAB: use tiled_copy_r2s (shares T2R thread-value layout) to partition
            # alpha smem for loading. partition_S on smem gives per-thread alpha source.
            if cutlass.const_expr(self.swap_ab):
                thr_copy_r2s_alpha = tiled_copy_r2s.get_slice(epi_tidx)

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

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            # Alpha scale consumer state (for non-swap path using warp 6 pipeline)
            alpha_scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_alpha_scale_stage
            )

            # Create copy atom for loading alpha scale from smem
            alpha_scale_copy_atom_s2r = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float32,
                num_bits_per_copy=32,
            )
            tiled_copy_alpha_scale_s2r = cute.make_tiled_copy_tv(
                alpha_scale_copy_atom_s2r,
                cute.make_layout((32 * len(self.epilog_warp_id),)),
                cute.make_layout((1,)),
            )
            thr_copy_alpha_scale_s2r = tiled_copy_alpha_scale_s2r.get_slice(tidx)

            m_total = malpha_scale_mnl.shape[0]

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx

                # Split-K: decompose L coord into batch_idx
                # For split_k=1: batch_idx = coord[2]
                batch_idx = cur_tile_coord[2] // self.split_k

                # Use batch_idx for output L coordinate (correct for both split_k=1 and >1)
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    batch_idx,
                )

                #
                # Slice to per mma tile index (for TMA store path)
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # initialize the final accumulator
                tTR_rAcc_final.fill(0.0)

                m_start = cur_tile_coord[0] * self.cta_tile_shape_mnk[0]
                m_in_bounds = m_start < m_total
                thread_in_bounds = epi_tidx < (m_total - m_start) if m_in_bounds else False

                if cutlass.const_expr(not self.swap_ab):
                    #
                    # Standard path: alpha loaded by warp 6 via alpha_scale_pipeline.
                    # One scalar alpha per M position per expert.
                    #
                    alpha_scale_consumer_state.reset_count()
                    peek_alpha_scale_full_status = cutlass.Boolean(1)
                    if alpha_scale_consumer_state.count < k_tile_cnt_local:
                        peek_alpha_scale_full_status = alpha_scale_pipeline.consumer_try_wait(
                            alpha_scale_consumer_state
                        )

                    acc_consumer_state.reset_count()
                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt_local:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                    for k_tile in cutlass.range(k_tile_cnt_local):
                        tTR_tAcc = tTR_tAcc_base[
                            (None, None, None, None, None, acc_consumer_state.index)
                        ]
                        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                        # Wait for alpha scale buffer full
                        alpha_scale_pipeline.consumer_wait(
                            alpha_scale_consumer_state, peek_alpha_scale_full_status
                        )

                        # Load alpha scale from shared memory for current expert
                        alpha_scale_smem_slice = sAlphaScale[
                            (None, alpha_scale_consumer_state.index)
                        ]
                        tAsAlphaScale_slice = thr_copy_alpha_scale_s2r.partition_S(
                            alpha_scale_smem_slice
                        )
                        current_alpha_scale_reg = cute.make_rmem_tensor(
                            tAsAlphaScale_slice.shape, cutlass.Float32
                        )
                        cute.copy(
                            alpha_scale_copy_atom_s2r,
                            tAsAlphaScale_slice,
                            current_alpha_scale_reg,
                        )
                        current_alpha_scale = current_alpha_scale_reg[0]

                        # Wait for accumulator buffer full
                        acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                        for subtile_idx in cutlass.range(subtile_cnt):
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                            tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                            acc_vec = tTR_rAcc.load()
                            final_vec = tTR_rAcc_subtile.load()
                            final_vec = acc_vec * current_alpha_scale + final_vec
                            tTR_rAcc_subtile.store(final_vec.to(self.acc_dtype))

                        # Release alpha scale buffer
                        alpha_scale_pipeline.consumer_release(alpha_scale_consumer_state)
                        alpha_scale_consumer_state.advance()

                        peek_alpha_scale_full_status = cutlass.Boolean(1)
                        if alpha_scale_consumer_state.count < k_tile_cnt_local:
                            peek_alpha_scale_full_status = alpha_scale_pipeline.consumer_try_wait(
                                alpha_scale_consumer_state
                            )

                        # Async arrive accumulator buffer empty
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                        peek_acc_full_status = cutlass.Boolean(1)
                        if acc_consumer_state.count < k_tile_cnt_local:
                            peek_acc_full_status = acc_pipeline.consumer_try_wait(
                                acc_consumer_state
                            )

                else:
                    #
                    # SwapAB path: warp 6 loads cta_N alphas per expert via pipeline.
                    # Epilogue waits on pipeline, then copies epi_tile_n alpha per subtile
                    # to stage-0 region of sAlphaScale for broadcast read.
                    #
                    alpha_scale_consumer_state.reset_count()
                    peek_alpha_scale_full_status = cutlass.Boolean(1)
                    if alpha_scale_consumer_state.count < k_tile_cnt_local:
                        peek_alpha_scale_full_status = alpha_scale_pipeline.consumer_try_wait(
                            alpha_scale_consumer_state
                        )

                    acc_consumer_state.reset_count()
                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt_local:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                    for k_tile in cutlass.range(k_tile_cnt_local):
                        tTR_tAcc = tTR_tAcc_base[
                            (None, None, None, None, None, acc_consumer_state.index)
                        ]
                        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                        # Wait for warp 6 to load cta_N alphas for this expert
                        alpha_scale_pipeline.consumer_wait(
                            alpha_scale_consumer_state, peek_alpha_scale_full_status
                        )

                        # Get contiguous alpha slice for this stage: (cta_N,) stride (1,)
                        alpha_smem_slice = sAlphaScale[(None, alpha_scale_consumer_state.index)]

                        # Wait for acc
                        acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                        # Create 2D broadcast view of alpha smem:
                        # (epi_tile_m, cta_N) stride (0, 1) — broadcast M, stride-1 N
                        sAlpha_bcast = cute.make_tensor(
                            alpha_smem_slice.iterator,
                            cute.make_layout(
                                shape=(self.epi_tile_m_size, self.cta_tile_shape_mnk[1]),
                                stride=(0, 1),
                            ),
                        )
                        # Partition smem using r2s copy (shares T2R thread-value layout)
                        tRS_sAlpha = thr_copy_r2s_alpha.partition_D(sAlpha_bcast)

                        for subtile_idx in cutlass.range(subtile_cnt):
                            # Load alpha from partitioned smem (32 N values)
                            tRS_sAlpha_sub = tRS_sAlpha[(None, None, subtile_idx)]
                            rAlpha_n = cute.make_rmem_tensor(tRS_sAlpha_sub.shape, cutlass.Float32)
                            rAlpha_n.store(tRS_sAlpha_sub.load())

                            # Broadcast alpha across M: (32N,1M) → (32N,32M)
                            rAlpha_bcast = cute.make_tensor(
                                rAlpha_n.iterator,
                                cute.make_layout(
                                    shape=tTR_rAcc.shape,
                                    stride=(
                                        (1, 0),  # (N_stride=1, M_stride=0)
                                        0,
                                        0,
                                    ),
                                ),
                            )

                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                            tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                            acc_vec = tTR_rAcc.load()
                            final_vec = tTR_rAcc_subtile.load()
                            alpha_vec = rAlpha_bcast.load()
                            final_vec = acc_vec * alpha_vec + final_vec
                            tTR_rAcc_subtile.store(final_vec.to(self.acc_dtype))

                        # Release alpha scale buffer
                        alpha_scale_pipeline.consumer_release(alpha_scale_consumer_state)
                        alpha_scale_consumer_state.advance()

                        peek_alpha_scale_full_status = cutlass.Boolean(1)
                        if alpha_scale_consumer_state.count < k_tile_cnt_local:
                            peek_alpha_scale_full_status = alpha_scale_pipeline.consumer_try_wait(
                                alpha_scale_consumer_state
                            )

                        # Async arrive accumulator buffer empty
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                        peek_acc_full_status = cutlass.Boolean(1)
                        if acc_consumer_state.count < k_tile_cnt_local:
                            peek_acc_full_status = acc_pipeline.consumer_try_wait(
                                acc_consumer_state
                            )

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_rAcc_final.shape, mode=[3])

                if cutlass.const_expr(self.split_k > 1 and self.swap_ab):
                    #
                    # SwapAB split-K: stage through non-swizzled smem for vectorized
                    # atomic add. C is M-major → M elements contiguous in global mem.
                    # Each thread writes its register elements to smem by (M, N) coord,
                    # then threads read M-contiguous chunks for fp16x8 atomic add.
                    # Reuse sA storage (mainloop done, sA no longer needed).
                    #
                    epi_m = self.epi_tile_m_size
                    epi_n = self.epi_tile_n_size
                    m_groups = epi_m // 8
                    groups_per_thread = (m_groups * epi_n) // (32 * len(self.epilog_warp_id))
                    m_base_tile = cur_tile_coord[0] * self.cta_tile_shape_mnk[0]

                    # Non-swizzled M-major smem staging (reuse sC memory,
                    # which is unused during split-K atomic store).
                    # Remove sC's swizzle by creating a raw pointer view.
                    sStaging = cute.make_tensor(
                        cute.recast_ptr(
                            sC.iterator, swizzle_=cute.make_swizzle(0, 0, 0), dtype=self.c_dtype
                        ),
                        cute.make_layout(
                            shape=(epi_m, epi_n),
                            stride=(1, epi_m),
                        ),
                    )

                    for subtile_idx in cutlass.range(subtile_cnt):
                        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                        acc_vec = tTR_rAcc_subtile.load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))

                        # Each thread writes its elements to smem at (epi_tidx, n)
                        # epi_tidx = M position, register elements span N positions
                        rC_flat = cute.make_tensor(
                            tTR_rC.iterator,
                            cute.make_layout((cute.size(tTR_rC),)),
                        )
                        tTR_rC.store(acc_vec)
                        for ei in cutlass.range(cute.size(tTR_rC), unroll_full=True):
                            sStaging[(epi_tidx, ei)] = rC_flat[ei]

                        self.epilog_sync_barrier.arrive_and_wait()

                        # Read M-contiguous groups from staging and vectorized atomic
                        n_base_subtile = (
                            cur_tile_coord[1] * self.cta_tile_shape_mnk[1] + subtile_idx * epi_n
                        )

                        for gi in cutlass.range(groups_per_thread, unroll_full=True):
                            group_idx = epi_tidx * groups_per_thread + gi
                            n_local = group_idx // m_groups
                            m_local_start = (group_idx % m_groups) * 8

                            m_global = m_base_tile + m_local_start
                            n_global = n_base_subtile + n_local
                            if m_global + 7 < m_total:
                                rVec = cute.make_rmem_tensor(
                                    cute.make_layout((4, 2), stride=(2, 1)),
                                    self.c_dtype,
                                )
                                rVec_flat = cute.make_tensor(rVec.iterator, cute.make_layout((8,)))
                                for j in cutlass.range(8, unroll_full=True):
                                    rVec_flat[j] = sStaging[(m_local_start + j, n_local)]

                                scatter_out = cute.domain_offset(
                                    (m_global, n_global, batch_idx), mC_raw
                                )
                                if cutlass.const_expr(self.c_dtype == cutlass.Float16):
                                    vectorized_atomic_add_fp16x8(rVec, scatter_out)
                                elif cutlass.const_expr(self.c_dtype == cutlass.BFloat16):
                                    vectorized_atomic_add_bf16x8(rVec, scatter_out)
                                else:
                                    for j in cutlass.range(8, unroll_full=True):
                                        scatter_j = cute.domain_offset((j, 0, 0), scatter_out)
                                        atomic_add_func(rVec_flat[j], scatter_j)

                        self.epilog_sync_barrier.arrive_and_wait()

                elif cutlass.const_expr(self.split_k > 1):
                    #
                    # Non-swap split-K atomic add path: each CTA atomically adds its
                    # partial result directly to the output C tensor.
                    # N-major C → vectorize along N.
                    #
                    rOut_epi = cute.make_tensor(tTR_rC.iterator, epi_layout_atomic)
                    m_coord = cur_tile_coord[0] * self.cta_tile_shape_mnk[0] + epi_tidx
                    scatter_base = cute.domain_offset((m_coord, 0, batch_idx), mC_raw)

                    if thread_in_bounds:
                        for subtile_idx in cutlass.range(subtile_cnt):
                            tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                            acc_vec = tTR_rAcc_subtile.load()
                            acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                            tTR_rC.store(acc_vec)

                            base_coord_n = cur_tile_coord[1] * self.cta_tile_shape_mnk[
                                1
                            ] + subtile_idx * cute.size(tTR_rC)

                            for index in cutlass.range(self.epi_loop_size_atomic, unroll_full=True):
                                coord_n = base_coord_n + index * self.element_offset_atomic
                                scatter_out = cute.domain_offset((0, coord_n, 0), scatter_base)
                                if cutlass.const_expr(self.c_dtype == cutlass.Float16):
                                    vectorized_atomic_add_fp16x8(
                                        rOut_epi[index, None, None], scatter_out
                                    )
                                elif cutlass.const_expr(self.c_dtype == cutlass.BFloat16):
                                    vectorized_atomic_add_bf16x8(
                                        rOut_epi[index, None, None], scatter_out
                                    )
                                elif cutlass.const_expr(self.c_dtype == cutlass.Float32):
                                    vectorized_atomic_add_fp32x2(rOut_epi[index, None], scatter_out)
                                else:
                                    atomic_add_func(rOut_epi[index], scatter_out)
                else:
                    #
                    # Standard TMA store path (split_k=1)
                    #
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                    num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                    for subtile_idx in cutlass.range(subtile_cnt):
                        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]

                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc_subtile).load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                        tRS_rC.store(acc_vec)

                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # Wait for C store complete (only needed for TMA store path)
            #
            if cutlass.const_expr(self.split_k <= 1):
                c_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it
        to partition smem memory (source) and tensor memory (destination).

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

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
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

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.epi_compute_layout,
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

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_rAcc_final_ = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rAcc_final = cute.group_modes(tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_))

        return (
            tiled_copy_t2r,
            tTR_tAcc,
            tTR_rAcc,
            tTR_rAcc_final,
        )

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition
        register array (source) and shared memory (destination).

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
            self.epi_compute_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
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
        partition shared memory (source) and global memory (destination) for TMA store version.

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

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
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
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int, int]:
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
                 (ACC stages, A/B operand stages, C stages, alpha scale stages)
        :rtype: tuple[int, int, int, int]
        """
        # ACC stages: match base gemm heuristic
        num_acc_stage = 2
        if mma_tiler_mnk[1] == 256:
            num_acc_stage = 1
        elif mma_tiler_mnk[1] <= 128:
            num_acc_stage = 3

        # Default C stages
        num_c_stage = 2

        # Default Alpha scale stages
        num_alpha_scale_stage = 10

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

        # Alpha scale shared memory (element-contiguous within stage)
        # Use max(M, N) to cover both swap_ab (cta_tile_N) and non-swap (cta_tile_M)
        alpha_dim = max(mma_tiler_mnk[0] // tiled_mma.thr_id.shape, mma_tiler_mnk[1])
        alpha_bytes = cute.size_in_bytes(
            cutlass.Float32,
            cute.make_layout(
                (alpha_dim, num_alpha_scale_stage),
                stride=(1, alpha_dim),
            ),
        )
        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes + alpha_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes + alpha_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage, num_alpha_scale_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        split_k: int = 1,
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
        :param split_k: Split-K factor to inflate L dimension.
        :type split_k: int

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        if split_k > 1:
            num_ctas_mnl = (num_ctas_mnl[0], num_ctas_mnl[1], num_ctas_mnl[2] * split_k)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
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
        # TODO: Currently we don't support m major output for Float4E2M1FN
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

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [64, 128, 192, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False

        # Skip invalid cluster shape
        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

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
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
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

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_scale_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        expert_count: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Wrapper function to create cute.Tensor objects from raw pointers and call the kernel.

        :param a_ptr: Pointer to input tensor A (M, K, L) - K-major
        :type a_ptr: cute.Pointer
        :param b_ptr: Pointer to weight tensor B (N, K, L) - K-major
        :type b_ptr: cute.Pointer
        :param a_sf_ptr: Pointer to scale factor tensor for A
        :type a_sf_ptr: cute.Pointer
        :param b_sf_ptr: Pointer to scale factor tensor for B
        :type b_sf_ptr: cute.Pointer
        :param alpha_scale_ptr: Pointer to alpha scale tensor (M, expert_count, L)
        :type alpha_scale_ptr: cute.Pointer
        :param c_ptr: Pointer to output tensor C (M, N, L) - N-major
        :type c_ptr: cute.Pointer
        :param m: M dimension (number of tokens)
        :type m: cutlass.Int64
        :param n: N dimension (output hidden size)
        :type n: cutlass.Int64
        :param k: K dimension (weight_per_expert * expert_count)
        :type k: cutlass.Int64
        :param l: L dimension (batch, typically 1)
        :type l: cutlass.Int64
        :param expert_count: Number of experts
        :type expert_count: cutlass.Constexpr
        :param scaling_vector_size: Scale factor vector size
        :type scaling_vector_size: cutlass.Constexpr
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream
        :type stream: cuda.CUstream
        """
        # Cast Int64 → Int32 so all derived tensor shapes and cute.size() calls stay in 32-bit,
        # which is required by cutlass.range(). The wrapper accepts Int64 for API compatibility
        # but practical tensor dimensions always fit in Int32.
        m = cutlass.Int32(m)
        n = cutlass.Int32(n)
        k = cutlass.Int32(k)
        l = cutlass.Int32(l)  # noqa: E741
        scale_k = k // scaling_vector_size

        # Create A tensor (M, K, L) - K-major
        a = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )

        # Create B tensor (N, K, L) - K-major
        b = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)),
        )

        # Create C tensor (M, N, L)
        if cutlass.const_expr(self.swap_ab):
            # SwapAB: M-major (col-major) to transpose output back
            c = cute.make_tensor(
                c_ptr,
                layout=cute.make_ordered_layout((m, n, l), order=(0, 1, 2)),
            )
        else:
            # Default: N-major (row-major)
            c = cute.make_tensor(
                c_ptr,
                layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
            )

        # Create A scale factor tensor (swizzled layout)
        # Shape: (32, 4, m // 128, 4, scale_k // 4, l)
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )

        # Create B scale factor tensor (swizzled layout)
        # Shape: (32, 4, n // 128, 4, scale_k // 4, l)
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )

        # Create alpha scale tensor - token-major for coalesced global memory access
        # Standard: (M, expert_count, L) — alpha per M (token)
        # SwapAB: (N, expert_count, L) — alpha per N (token)
        alpha_token_dim = n if cutlass.const_expr(self.swap_ab) else m
        alpha_scale = cute.make_tensor(
            alpha_scale_ptr,
            layout=cute.make_ordered_layout((alpha_token_dim, expert_count, l), order=(0, 1, 2)),
        )

        # Call the kernel
        self.__call__(
            a,
            b,
            a_sf,
            b_sf,
            alpha_scale,
            c,
            max_active_clusters,
            stream,
        )

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
        expert_count: int,
        weight_per_expert: int,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str
        :param expert_count: The number of experts
        :type expert_count: int
        :param weight_per_expert: The number of weights per expert
        :type weight_per_expert: int

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False

        # Add specific check for weight_per_expert, expert_count and k size.
        # skip unsupported weight_per_expert
        mma_tile_shape_k = 256
        if weight_per_expert % mma_tile_shape_k != 0:
            can_implement = False

        # mma_tile_shape_k = 256
        if not (k % expert_count == 0 and k // expert_count == weight_per_expert):
            can_implement = False

        return can_implement


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
