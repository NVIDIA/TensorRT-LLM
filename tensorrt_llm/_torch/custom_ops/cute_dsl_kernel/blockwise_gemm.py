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

import argparse
from typing import Tuple, Type
import math
import random

import torch
from cuda import cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

"""
High-performance blockwise dense GEMM (C = (SFA * A) * (SFB * B)) example for the NVIDIA Hopper architecture
using CUTE DSL.
    - Matrix A is MxKxL, L is the group dimension, A can be row-major("K")
    - Matrix B is NxKxL, L is the group dimension, B can be column-major("K")
    - Matrix C is MxNxL, L is the group dimension, C is always row-major("N")
    - For each iteration, the kernel will compute C = A * B and then apply the scale factor C *= SFA * SFB

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Perform scale and add operations to the result.
4. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/hopper/blockwise_gemm.py                               \
      --mnkl 2048,2048,2048,4 --tile_shape_mnk 128,128,128                 \
      --cluster_shape_mn 1,1 --a_dtype Float8E4M3FN --b_dtype Float8E4M3FN \
      --sfa_dtype Float32 --sfb_dtype Float32                              \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

The above example command compute batched gemm with M=2048, N=2048, K=2048,
group_count=4. The Hopper WGMMA tile shape is 128x128x128 and the cluster shape
is (1,1). The input, mma accumulator and output data type are set as fp8, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/blockwise_gemm.py                           \
      --mnkl 2048,2048,2048,4 --tile_shape_mnk 128,128,128                 \
      --cluster_shape_mn 1,1 --a_dtype Float8E4M3FN --b_dtype Float8E4M3FN \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n  

Constraints:
* Supported input data types: fp8 (e4m3fn)
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* Tile shape M must be 64 or 128
* Tile shape N must be 128
* Tile shape K must be 128
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
# /////////////////////////////////////////////////////////////////////////////
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL GEMM on Hopper.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=[2048, 2048, 2048, 4],
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        choices=[(128, 128, 128), (64, 128, 128)],
        default=(128, 128, 128),
        help="Tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2), (1, 4)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.Float8E4M3FN,
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.Float8E4M3FN,
    )
    parser.add_argument(
        "--sfa_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument(
        "--sfb_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mnk) != 3:
        parser.error("--tile_shape_mnk must contain exactly 3 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    return args


# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
# /////////////////////////////////////////////////////////////////////////////


class HopperBlockwiseGemmKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mnk: Shape of the tile (M,N,K)
    :type tile_shape_mnk: Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float8E4M3FN

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - Tile M must be 64 or 128
        - Tile N must be 128
        - Tile K must be 128
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperBlockwiseGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mnk=(128, 128, 128),
        ...     cluster_shape_mnk=(1, 1),
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mnk: tuple[int, int, int],
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mnk: Shape of the tile (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """
        self.acc_dtype = acc_dtype

        # {$nv-internal-release begin}
        # TODO: get from args
        # {$nv-internal-release end}
        self.scale_granularity_m = 1
        self.scale_granularity_n = 128
        self.scale_granularity_k = 128
        self.scale_m_per_tile = tile_shape_mnk[0] // self.scale_granularity_m
        self.scale_n_per_tile = tile_shape_mnk[1] // self.scale_granularity_n
        self.scale_k_per_tile = tile_shape_mnk[2] // self.scale_granularity_k

        self.cluster_shape_mnk = cluster_shape_mnk
        self.mma_inst_shape_mn = None
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.atom_layout_mnk = (2, 1, 1) if tile_shape_mnk[0] > 64 else (1, 1, 1)
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = sm90_utils.SMEM_CAPACITY["sm90"]

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """
        Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C/SFA/SFB stage counts in shared memory
        - Computing A/B/C/SFA/SFB shared memory layout
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64 or 128")
        if self.tile_shape_mnk[1] not in [128]:
            raise ValueError("CTA tile shape N must be 128")
        if self.tile_shape_mnk[2] not in [128]:
            raise ValueError("CTA tile shape K must be 128")

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sfa_dtype,
            self.sfb_dtype,
            self.scale_m_per_tile * self.scale_k_per_tile,
            self.scale_n_per_tile * self.scale_k_per_tile,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )
        self.sfa_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.ab_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_n, self.scale_n_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.ab_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """
        Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param sfa: Input tensor SFA
        :type sfa: cute.Tensor
        :param sfb: Input tensor SFB
        :type sfb: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sfa_dtype = sfa.element_type
        self.sfb_dtype = sfb.element_type
        self.a_layout = utils.Layout.from_tensor(a)
        self.b_layout = utils.Layout.from_tensor(b)
        self.c_layout = utils.Layout.from_tensor(c)

        if cutlass.const_expr(self.a_dtype.width != 8):
            raise TypeError(f"a_dtype should be float8")

        self._setup_attributes()

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[1],
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mnk)

        # Change the scale tensor layout to be consistent with the smem layout
        sfa = cute.make_tensor(
            sfa.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_m, sfa.shape[0]),
                    (self.scale_granularity_k, sfa.shape[1]),
                    sfa.shape[2],
                ),
                stride=(
                    (0, sfa.layout.stride[0]),
                    (0, sfa.layout.stride[1]),
                    sfa.layout.stride[2],
                ),
            ),
        )
        sfb = cute.make_tensor(
            sfb.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_n, sfb.shape[0]),
                    (self.scale_granularity_k, sfb.shape[1]),
                    sfb.shape[2],
                ),
                stride=(
                    (0, sfb.layout.stride[0]),
                    (0, sfb.layout.stride[1]),
                    sfb.layout.stride[2],
                ),
            ),
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sfa_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sfb_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            sfa,
            sfb,
            tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param mSFA_mkl: Input tensor SFA
        :type mSFA_mkl: cute.Tensor
        :param mSFB_nkl: Input tensor SFB
        :type mSFB_nkl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, bidz = cute.arch.block_idx()
        bdimx, bdimy, bdimz = cute.arch.grid_dim()

        # {$nv-internal-release begin}
        # FIXME: remove this part as it will cause IMA
        # # CTA Swizzle to promote L2 data reuse.
        # if cutlass.const_expr(self.is_masked):
        #     tile_layout = cute.tiled_divide(
        #         cute.make_layout(fake_c.shape),
        #         cute.slice_(self.tile_shape_mnk, (None, None, 0)),
        #     )
        # else:
        #     tile_layout = cute.tiled_divide(
        #         cute.make_layout(mC_mnl.shape),
        #         cute.slice_(self.tile_shape_mnk, (None, None, 0)),
        #     )
        # num_pid_m = cute.size(tile_layout, mode=[1])
        # num_pid_n = cute.size(tile_layout, mode=[2])
        # GROUP_SIZE_M = cutlass.min(num_pid_m, 8)
        # cta_layout = cute.make_layout(
        #     ((GROUP_SIZE_M, num_pid_n), (num_pid_m + GROUP_SIZE_M - 1) // GROUP_SIZE_M)
        # )
        # pid = bidx + bidy * bdimx + bidz * bdimx * bdimy
        # cta_coord = cta_layout.get_hier_coord(pid)
        # pid_m = cta_coord[0][0] + cta_coord[1] * GROUP_SIZE_M
        # pid_n = cta_coord[0][1]
        # FIXME: WAR to use no swizzle cga
        # {$nv-internal-release end}
        pid_m = bidx
        pid_n = bidy
        num_pid_n = bdimy

        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )

        tile_coord_mnkl_a = (
            pid_m,
            pid_n,
            None,
            bidz,
        )
        tile_coord_mnkl_b = (pid_m, pid_n, None, bidz)
        tile_coord_mnkl_c = (
            pid_m,
            pid_n,
            None,
            bidz,
        )

        tidx, _, _ = cute.arch.thread_idx()
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = utils.CooperativeGroup(utils.Agent.Thread)
        # Set the consumer arrive count to the number of mcast size
        consumer_arrive_cnt = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        mainloop_pipeline_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread, consumer_arrive_cnt
        )

        mainloop_pipeline = utils.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_mnk,
        )

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.core.make_tensor(sC_ptr, epi_smem_layout_staged.outer)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, loopK)
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl_a, proj=(1, None, 1)
        )
        # (bN, bK, loopK)
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl_b, proj=(None, 1, 1)
        )
        # (bM, bN)
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl_c, proj=(1, 1, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, self.tile_shape_mnk, tile_coord_mnkl_a, proj=(1, None, 1)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, self.tile_shape_mnk, tile_coord_mnkl_b, proj=(None, 1, 1)
        )
        # coordinate
        cSFA_mkl = cute.make_identity_tensor(cute.shape(mSFA_mkl))
        cSFB_nkl = cute.make_identity_tensor(cute.shape(mSFB_nkl))
        cSFA = cute.local_tile(
            cSFA_mkl, self.tile_shape_mnk, tile_coord_mnkl_a, proj=(1, None, 1)
        )
        cSFB = cute.local_tile(
            cSFB_nkl, self.tile_shape_mnk, tile_coord_mnkl_b, proj=(None, 1, 1)
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC_mnl)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for ScaleFactor
        # //////////////////////////////////////////////////////////////////////////////
        # scale viewed as C tensor
        sSFA_view_as_C_layout = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                self.tile_shape_mnk[1],
                self.ab_stage,
            ),
            stride=((0, 1), 0, self.scale_m_per_tile),
        )
        sSFB_view_as_C_layout = cute.make_layout(
            (
                self.tile_shape_mnk[0],
                (self.scale_granularity_n, self.scale_n_per_tile),
                self.ab_stage,
            ),
            stride=(0, (0, 1), self.scale_n_per_tile),
        )
        sSFA_view_as_C = cute.make_tensor(sSFA.iterator, sSFA_view_as_C_layout)
        sSFB_view_as_C = cute.make_tensor(sSFB.iterator, sSFB_view_as_C_layout)
        tCsSFA = tiled_mma.get_slice(tidx).partition_C(sSFA_view_as_C)
        tCsSFB = tiled_mma.get_slice(tidx).partition_C(sSFB_view_as_C)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sa_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sa_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sb_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sb_for_tma_partition,
            gB_for_tma_partition,
        )

        # load scaleA/scaleB
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mSFA_mkl.element_type,
            num_bits_per_copy=mSFA_mkl.element_type.width,
        )
        tiled_copy_sfa = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        tiled_copy_sfb = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        thr_copy_sfa = tiled_copy_sfa.get_slice(tidx)
        thr_copy_sfb = tiled_copy_sfb.get_slice(tidx)
        tAgSFA_mkl = thr_copy_sfa.partition_S(gSFA_mkl)
        tAcSFA = thr_copy_sfa.partition_S(cSFA)
        tAsSFA = thr_copy_sfa.partition_D(sSFA)
        tBgSFB_nkl = thr_copy_sfb.partition_S(gSFB_nkl)
        tBcSFB = thr_copy_sfb.partition_S(cSFB)
        tBsSFB = thr_copy_sfb.partition_D(sSFB)

        tApSFA = cute.make_fragment(
            cute.make_layout(
                cute.filter_zeros(cute.slice_(tAsSFA, (None, None, None, 0))).shape
            ),
            cutlass.Boolean,
        )
        tBpSFB = cute.make_fragment(
            cute.make_layout(
                cute.filter_zeros(cute.slice_(tBsSFB, (None, None, None, 0))).shape
            ),
            cutlass.Boolean,
        )

        tAcSFA_compact = cute.filter_zeros(cute.slice_(tAcSFA, (None, None, None, 0)))
        tBcSFB_compact = cute.filter_zeros(cute.slice_(tBcSFB, (None, None, None, 0)))
        # {$nv-internal-release begin}
        # TODO: Skip more unnecessary load
        # {$nv-internal-release end}
        for i in range(cute.size(tApSFA, mode=[1])):
            tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                tAcSFA_compact[(i)][0], mSFA_mkl.shape[0]
            )
        for i in range(cute.size(tBpSFB, mode=[1])):
            tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                tBcSFB_compact[(i)][0], mSFB_nkl.shape[0]
            )

        # //////////////////////////////////////////////////////////////////////////////
        #  Make frangments
        # //////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrSFA = cute.make_fragment(
            cute.make_layout(tCsSFA[(None, None, None, 0)].shape), self.sfa_dtype
        )
        tCrSFB = cute.make_fragment(
            cute.make_layout(tCsSFA[(None, None, None, 0)].shape), self.sfb_dtype
        )

        acc_shape = tCgC.shape
        accumulators = cute.make_fragment(acc_shape, self.acc_dtype)
        final_accumulators = cute.make_fragment(acc_shape, self.acc_dtype)
        # initialize final_accumulators to 0
        final_accumulators.fill(0)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cluster wait
        # ///////////////////////////////////////////////////////////////////////////////
        # cluster wait for barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()
        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch
        # /////////////////////////////////////////////////////////////////////////////
        k_tile_cnt = cute.size(gA_mkl, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage - 1, k_tile_cnt), 0)

        mainloop_producer_state = utils.make_pipeline_state(
            utils.PipelineUserType.Producer, self.ab_stage
        )
        #  Peek (try_wait) AB buffer empty
        peek_ab_empty_status = cutlass.Boolean(1)
        if mainloop_producer_state.count < k_tile_cnt:
            peek_ab_empty_status = mainloop_pipeline.producer_try_acquire(
                mainloop_producer_state
            )

        if warp_idx == 0:
            # /////////////////////////////////////////////////////////////////////////////
            # Prefetch TMA load
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range_dynamic(prefetch_k_tile_cnt, unroll=1):

                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(
                    mainloop_producer_state, peek_ab_empty_status
                )
                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                tAsSFA_pipe = cute.filter_zeros(
                    tAsSFA[(None, None, None, mainloop_producer_state.index)]
                )
                tBsSFB_pipe = cute.filter_zeros(
                    tBsSFB[(None, None, None, mainloop_producer_state.index)]
                )
                tAgSFA_k = cute.filter_zeros(
                    tAgSFA_mkl[(None, None, None, mainloop_producer_state.count)]
                )
                tBgSFB_k = cute.filter_zeros(
                    tBgSFB_nkl[(None, None, None, mainloop_producer_state.count)]
                )

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(tiled_copy_sfa, tAgSFA_k, tAsSFA_pipe, pred=tApSFA)
                cute.copy(tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe, pred=tBpSFB)

                cute.arch.cp_async_commit_group()

                tAcSFA_compact_ = cute.filter_zeros(
                    cute.slice_(
                        tAcSFA, (None, None, None, mainloop_producer_state.count)
                    )
                )
                tBcSFB_compact_ = cute.filter_zeros(
                    cute.slice_(
                        tBcSFB, (None, None, None, mainloop_producer_state.count)
                    )
                )

                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()
                peek_ab_empty_status = cutlass.Boolean(1)
                if mainloop_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = mainloop_pipeline.producer_try_acquire(
                        mainloop_producer_state
                    )

                # {$nv-internal-release begin}
                # TODO: Skip more unnecessary load
                # {$nv-internal-release end}
                for i in range(cute.size(tApSFA, mode=[1])):
                    tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                        tAcSFA_compact_[(i)][0], mSFA_mkl.shape[0]
                    )
                for i in range(cute.size(tBpSFB, mode=[1])):
                    tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                        tBcSFB_compact_[(i)][0], mSFB_nkl.shape[0]
                    )

            for prefetch_idx in range_dynamic(
                self.ab_stage - prefetch_k_tile_cnt, unroll=1
            ):
                cute.arch.cp_async_commit_group()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////
        # {$nv-internal-release begin}
        # TODO: only k_pipe_mmas = 1 is supported for now, need to fix unroll issue
        # {$nv-internal-release end}

        num_k_blocks = cute.size(tCrA, mode=[2])

        #  Wait for AB buffer full
        mainloop_consumer_read_state = utils.make_pipeline_state(
            utils.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = utils.make_pipeline_state(
            utils.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in range_dynamic(k_tile_cnt, unroll=1):

            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for TMA copies to complete
            # /////////////////////////////////////////////////////////////////////////////
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )
            cute.arch.cp_async_wait_group(self.ab_stage - 1)
            cute.arch.barrier()

            scale_atom_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.sfa_dtype,
                num_bits_per_copy=self.sfa_dtype.width,
            )
            cute.copy(
                scale_atom_copy,
                tCsSFA[(None, None, None, mainloop_consumer_read_state.index)],
                tCrSFA,
            )
            cute.copy(
                scale_atom_copy,
                tCsSFB[(None, None, None, mainloop_consumer_read_state.index)],
                tCrSFB,
            )

            # /////////////////////////////////////////////////////////////////////////////
            #  WGMMA
            # /////////////////////////////////////////////////////////////////////////////
            cute.nvgpu.warpgroup.fence()
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            # Wait on the wgmma barrier for previous wgmmas to complete
            cute.nvgpu.warpgroup.wait_group(0)

            acc_vec = accumulators.load()
            final_vec = final_accumulators.load()
            scale_a = tCrSFA.load()
            scale_b = tCrSFB.load()
            scale = scale_a * scale_b
            final_vec = acc_vec * scale + final_vec
            final_accumulators.store(final_vec.to(self.acc_dtype))

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

            # /////////////////////////////////////////////////////////////////////////////
            #  TMA load
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:

                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(
                    mainloop_producer_state, peek_ab_empty_status
                )

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                tAsSFA_pipe = cute.filter_zeros(
                    tAsSFA[(None, None, None, mainloop_producer_state.index)]
                )
                tBsSFB_pipe = cute.filter_zeros(
                    tBsSFB[(None, None, None, mainloop_producer_state.index)]
                )
                tAgSFA_k = cute.filter_zeros(
                    tAgSFA_mkl[(None, None, None, mainloop_producer_state.count)]
                )
                tBgSFB_k = cute.filter_zeros(
                    tBgSFB_nkl[(None, None, None, mainloop_producer_state.count)]
                )

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

                cute.copy(tiled_copy_sfa, tAgSFA_k, tAsSFA_pipe, pred=tApSFA)
                cute.copy(tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe, pred=tBpSFB)
                cute.arch.cp_async_commit_group()

                tAcSFA_compact_ = cute.filter_zeros(
                    cute.slice_(
                        tAcSFA, (None, None, None, mainloop_producer_state.count)
                    )
                )
                tBcSFB_compact_ = cute.filter_zeros(
                    cute.slice_(
                        tBcSFB, (None, None, None, mainloop_producer_state.count)
                    )
                )

                peek_ab_empty_status = cutlass.Boolean(1)
                if mainloop_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = mainloop_pipeline.producer_try_acquire(
                        mainloop_producer_state
                    )

                # {$nv-internal-release begin}
                # TODO: Skip more unnecessary load
                # {$nv-internal-release end}
                for i in range(cute.size(tApSFA, mode=[1])):
                    tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                        tAcSFA_compact_[(i)][0], mSFA_mkl.shape[0]
                    )
                for i in range(cute.size(tBpSFB, mode=[1])):
                    tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                        tBcSFB_compact_[(i)][0], mSFB_nkl.shape[0]
                    )

        # /////////////////////////////////////////////////////////////////////////////
        #  EPILOG
        # /////////////////////////////////////////////////////////////////////////////
        cute.nvgpu.warpgroup.wait_group(0)

        # Wait for all threads in the cluster to finish, avoid early release of smem
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(final_accumulators)

        # Allocate D registers.
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tcgc_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tcgc_for_tma_partition,
        )

        epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
        epi_tile_shape = tcgc_for_tma_partition.shape[1]

        for epi_idx in range_dynamic(epi_tile_num, unroll=epi_tile_num):
            # Copy from accumulators to D registers
            for epi_v in range(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            # Type conversion
            tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            # Copy from D registers to shared memory
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            # barrier for sync
            cute.arch.barrier()

            # Get the global memory coordinate for the current epi tile.
            # {$nv-internal-release begin}
            # TODO: fail to move this to outside of the loop
            # {$nv-internal-release end}
            epi_tile_layout = cute.make_layout(
                epi_tile_shape, stride=(epi_tile_shape[1], 1)
            )
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(self.epi_stage - 1, read=True)

            cute.arch.barrier()

        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        sfa_dtype: type[cutlass.Numeric],
        sfb_dtype: type[cutlass.Numeric],
        sfa_count: int,
        sfb_count: int,
        smem_capacity: int,
        occupancy: int,
    ):
        """
        Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param sfa_dtype: Data type of operand SFA.
        :type sfa_dtype: type[cutlass.Numeric]
        :param sfb_dtype: Data type of operand SFB.
        :type sfb_dtype: type[cutlass.Numeric]
        :param sfa_count: Number of SFA elements per tile.
        :type sfa_count: int
        :param sfb_count: Number of SFB elements per tile.
        :type sfb_count: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        epi_stage = 4
        # epi_smem will reuse smem ab.
        epi_bytes = 0

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
            + sfa_count * sfa_dtype.width // 8
            + sfb_count * sfb_dtype.width // 8
        )
        # 1024B alignment
        ab_bytes_per_stage = math.ceil(ab_bytes_per_stage / 1024) * 1024
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """
        Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_dtype: type[cutlass.Numeric],
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout for matrix A
        :type a_layout: Layout
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout for matrix B
        :type b_layout: Layout
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: leading dimension of the output matrix C
        :type c_layout: Layout
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_major_mode_size = (
            tile_shape_mnk[2] if a_layout.is_k_major_a() else tile_shape_mnk[0]
        )
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(1, 0, 2) if a_layout.is_k_major_a() else (0, 1, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = (
            tile_shape_mnk[2] if b_layout.is_n_major_b() else tile_shape_mnk[1]
        )
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(1, 0, 2) if b_layout.is_n_major_b() else (0, 1, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mnk: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """
        Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mnk: Shape of each cluster in M, N, K dimensions.
        :type cluster_shape_mnk: tuple[int, int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnk)
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnk))
        return grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """
        Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """

        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_c.shape), epi_tile
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tma_tile_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """
        Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tma_tile_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # tested a_dtype
        if a_dtype not in {
            cutlass.Float8E4M3FN,
        }:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {
            cutlass.Float8E4M3FN,
        }:
            is_valid = False
        # tested acc_dtype
        if acc_dtype != cutlass.Float32:
            is_valid = False
        # tested c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False

        return is_valid


def run_blockwise_gemm(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sfa_dtype: Type[cutlass.Numeric],
    sfb_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    tile_shape_mnk: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.
    """

    print(f"Running Hopper Blockwise GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Tile Shape: {tile_shape_mnk}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")

    # Unpack parameters
    m, n, k, l = mnkl
    cluster_shape_mnk = (*cluster_shape_mn, 1)

    # Skip unsupported types
    if not HopperBlockwiseGemmKernel.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, c_dtype
    ):
        raise TypeError(
            f"Skipping due to unsupported types: {a_dtype}, {b_dtype}, {acc_dtype}, {c_dtype}"
        )

    # Prepare pytorch tensors: A, B (random from 0 to 2) and C (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(2025)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(
        l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
    ):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else : (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = (
            cutlass_torch.dtype(dtype)
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.uint8
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            cutlass_torch.dtype(dtype),
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.GAUSSIAN,
            init_config=cutlass.torch.GaussianInitConfig(
                mean=0,
                std=0.5,
            ),
        )

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype torch tensor (gpu)
        # WAR as torch dlpack does not support fp8 type
        torch_tensor = torch_tensor_cpu.view(torch_dtype).cuda()

        # Create dtype cute tensor (gpu)
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=(0 if is_mode0_major else 1)
            )
        cute_tensor = cutlass.torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a, mA, a_torch = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
    b, mB, b_torch = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
    c, mC, c_torch = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
    sfa, mSFA, sfa_torch = create_and_permute_tensor(
        l, m, math.ceil(k / 128), True, sfa_dtype
    )
    sfb, mSFB, sfb_torch = create_and_permute_tensor(
        l, math.ceil(n / 128), math.ceil(k / 128), False, sfb_dtype
    )

    gemm = HopperBlockwiseGemmKernel(
        acc_dtype,
        tile_shape_mnk,
        cluster_shape_mnk,
    )

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    # compile gemm kernel
    compiled_gemm = cute.compile(gemm, mA, mB, mC, mSFA, mSFB, stream)
    # execution
    compiled_gemm(mA, mB, mC, mSFA, mSFB, stream)

    torch.cuda.synchronize()

    # update
    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(sfa, a)
    updated_b = pad_and_multiply(sfb, b)

    ref = torch.einsum("mkl,nkl->mnl", updated_a, updated_b).to(
        cutlass_torch.dtype(c_dtype)
    )
    res = c_torch.view(cutlass_torch.dtype(c_dtype))

    torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-03)


if __name__ == "__main__":
    args = parse_arguments()
    run_blockwise_gemm(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.sfa_dtype,
        args.sfb_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.tile_shape_mnk,
        args.cluster_shape_mn,
        args.tolerance,
    )
    print("PASS")
