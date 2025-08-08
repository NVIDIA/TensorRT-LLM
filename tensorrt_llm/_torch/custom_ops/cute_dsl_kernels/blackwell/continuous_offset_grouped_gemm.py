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
import math
import random
from typing import Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda import cuda
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack


class BlockwiseContiguousGroupedGemmKernel:
    """This class implements batched matrix multiplication (C = (SFA * A) * (SFB * B)) with support for fp8 (e4m3fn, e5m2)
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

    :note: Supported A/B data types:
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128
        - MMA tiler N must be 128
        - Cluster shape M must be 1
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    Example:
        >>> gemm = BlockwiseContiguousGroupedGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     use_2cta_instrs=True,
        ...     mma_tiler_mn=(128, 128),
        ...     cluster_shape_mn=(2, 2)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
    ):
        """Initializes the configuration for a Blackwell blockwise dense GEMM kernel.

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
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store

        self.cta_group = (tcgen05.CtaGroup.TWO
                          if use_2cta_instrs else tcgen05.CtaGroup.ONE)

        self.occupancy = 1
        # Set specialized warp ids
        self.acc_update_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.epilog_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.scale_warp_id = 10
        self.sched_warp_id = 11
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len((
            *self.acc_update_warp_id,
            *self.epilog_warp_id,
            self.mma_warp_id,
            self.tma_warp_id,
            self.scale_warp_id,
            self.sched_warp_id,
        ))
        self.threads_wo_sched = self.threads_per_warp * len((
            *self.acc_update_warp_id,
            *self.epilog_warp_id,
            self.mma_warp_id,
            self.tma_warp_id,
            self.scale_warp_id,
        ))
        # TODO: find a better config
        self.num_regs_uniform_warps = 32
        self.num_regs_tiled_warps = 64
        self.num_regs_epilogue_warps = 208
        self.num_regs_acc_update_warps = 216

        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.sched_sync_bar_id = 3
        self.pdl_sync_bar_id = 4
        self.num_smem_capacity = sm100_utils.SMEM_CAPACITY["sm100"]
        # TMEM offset for final accumulator
        self.tmem_final_offset = 384

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
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        # limin: mma_inst_shape_k = 32
        # print(f"limin: mma_inst_shape_k = {mma_inst_shape_k}")
        mma_inst_tile_k = 4
        # mnk: 128, 128, 128
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        # limin: mma_a_tiler = (1, 128, 128)
        self.mma_a_tiler = (1, self.mma_tiler[1], self.mma_tiler[2])
        # cta_tile_shape_mnk = 128, 128, 128
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        # print(f"limin: mma_a_tiler = {self.mma_a_tiler}")
        # print(f"limin: cta_tile_shape_mnk = {self.cta_tile_shape_mnk}")

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape, ),
        )

        # {$nv-internal-release begin}
        # TODO: get from args
        # {$nv-internal-release end}
        self.scale_granularity_m = 1
        self.scale_granularity_n = 128
        self.scale_granularity_k = 128
        # limin: 128, 1, 1
        self.scale_m_per_tile = self.cta_tile_shape_mnk[
            0] // self.scale_granularity_m
        self.scale_n_per_tile = self.cta_tile_shape_mnk[
            1] // self.scale_granularity_n
        self.scale_k_per_tile = self.cta_tile_shape_mnk[
            2] // self.scale_granularity_k
        # print(f"limin: scale_m_per_tile = {self.scale_m_per_tile}, scale_n_per_tile = {self.scale_n_per_tile}, scale_k_per_tile = {self.scale_k_per_tile}")

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_scale_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sfa_dtype,
            self.sfb_dtype,
            self.scale_m_per_tile * self.scale_k_per_tile,
            self.scale_n_per_tile * self.scale_k_per_tile,
            self.num_smem_capacity,
            self.occupancy,
            self.use_tma_store,
        )
        # print(f"limin: num_acc_stage = {self.num_acc_stage}, num_ab_stage = {self.num_ab_stage}, num_c_stage = {self.num_c_stage}, num_scale_stage = {self.num_scale_stage}, num_tile_stage = {self.num_tile_stage}")

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
        self.c_smem_layout_staged = (sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        ) if cutlass.const_expr(self.use_tma_store) else None)
        # limin-TODO: 为什么要这样设置？
        # （1， 128), (128, 1), (num_stages): (0, 1), (0, 1), (128)
        self.sfa_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )
        # (128, 1), (128, 1), (num_stages): (0, 1), (0, 1), (1)
        self.sfb_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_n, self.scale_n_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        offset_mapping: cute.Tensor,
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

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param gidx_mapping: Group index mapping tensor
        :type gidx_mapping: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
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
        self.sfa_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.sfb_dtype: Type[cutlass.Numeric] = sfb.element_type
        self.a_major_mode = utils.Layout.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.Layout.from_tensor(b).mma_major_mode()
        self.c_layout = utils.Layout.from_tensor(c)

        # set group count
        self.group_count = b.shape[2]

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(
                f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = self._get_tma_atom_kind(atom_thr_size, self.is_a_mcast)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged,
                                    (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tma_tile_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32
                           if a.element_type is cutlass.Float32 else None),
        )

        # Setup TMA load for B
        b_op = self._get_tma_atom_kind(atom_thr_size, self.is_b_mcast)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged,
                                    (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tma_tile_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32
                           if b.element_type is cutlass.Float32 else None),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            c_cta_v_layout = cute.composition(
                cute.make_identity_layout(c.shape), self.epi_tile)
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged,
                                          (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tma_tile_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                c_cta_v_layout,
            )

        tensor_sfa = cute.make_tensor(
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
        tensor_sfb = cute.make_tensor(
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

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn,
            max_active_clusters)

        self.buffer_align_bytes = 1024

        c_smem_size = (cute.cosize(self.c_smem_layout_staged.outer)
                       if cutlass.const_expr(self.use_tma_store) else 0)

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                              self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                 self.num_scale_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                               self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                     self.num_tile_stage * 2]
            epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    c_smem_size,
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sfa_dtype,
                                     cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sfb_dtype,
                                     cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # print(f"limin: grid = {grid}, block = {self.threads_per_cta}, cluster = {self.cluster_shape_mn}")
        # cute.printf(f"limin: grid = {grid}")
        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if cutlass.const_expr(self.use_tma_store) else c,
            c,
            tensor_sfa,
            tensor_sfb,
            offset_mapping,
            self.group_count,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
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
        tensor_c: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        offset_mapping: cute.Tensor,
        group_count: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

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
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = utils.CooperativeGroup(utils.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread, num_tma_producer)
        ab_pipeline = utils.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        scale_pipeline_producer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            self.threads_per_warp * 1,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        scale_pipeline = utils.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = utils.CooperativeGroup(utils.Agent.Thread)
        num_acc_consumer_threads = len(
            self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = utils.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        epi_pipeline_producer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            self.threads_per_warp * len(self.acc_update_warp_id),
            self.threads_per_warp * len(self.acc_update_warp_id),
        )
        epi_pipeline_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        epi_pipeline = utils.PipelineAsync.create(
            barrier_storage=storage.epi_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
        )

        tile_info_pipeline_producer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            self.threads_per_warp * 1,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread, self.threads_wo_sched, self.threads_wo_sched)
        tile_info_pipeline = utils.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init_arrive_cnt(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = (storage.sC.get_tensor(c_smem_layout_staged.outer,
                                    swizzle=c_smem_layout_staged.inner)
              if cutlass.const_expr(self.use_tma_store) else None)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer,
                                   swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer,
                                   swizzle=b_smem_layout_staged.inner)
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = storage.sInfo.get_tensor(info_layout)
        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if self.is_a_mcast or self.is_b_mcast or use_2cta_instrs:
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(mA_mkl,
                                 cute.slice_(self.mma_a_tiler, (None, 0, None)),
                                 (None, None, None))
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(mB_nkl,
                                 cute.slice_(self.mma_tiler, (0, None, None)),
                                 (None, None, None))
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(mC_mnl,
                                 cute.slice_(self.mma_a_tiler, (None, None, 0)),
                                 (None, None, None))
        gC_mnl_simt = cute.local_tile(
            tensor_c, cute.slice_(self.mma_a_tiler, (None, None, 0)),
            (None, None, None))
        # (bM, bK, loopM, loopK, loopL)
        gSFA_mkl = cute.local_tile(mSFA_mkl,
                                   cute.slice_(self.mma_tiler, (None, 0, None)),
                                   (None, None, None))
        # (bN, bK, loopN, loopK, loopL)
        gSFB_nkl = cute.local_tile(mSFB_nkl,
                                   cute.slice_(self.mma_tiler, (0, None, None)),
                                   (None, None, None))
        # coordinate
        cSFA_mkl = cute.make_identity_tensor(cute.shape(mSFA_mkl))
        cSFB_nkl = cute.make_identity_tensor(cute.shape(mSFB_nkl))
        # (bM, bK, loopM, loopK, loopL)
        cSFA = cute.local_tile(cSFA_mkl,
                               cute.slice_(self.mma_tiler, (None, 0, None)),
                               (None, None, None))
        # (bN, bK, loopN, loopK, loopL)
        cSFB = cute.local_tile(cSFB_nkl,
                               cute.slice_(self.mma_tiler, (0, None, None)),
                               (None, None, None))
        k_block_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgC_tiled = thr_mma.partition_C(gC_mnl_simt)

        # scale viewed as C tensor
        sSFA_view_as_C_layout = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                self.cta_tile_shape_mnk[1],
                self.num_scale_stage,
            ),
            stride=((0, 1), 0, self.scale_m_per_tile),
        )
        sSFB_view_as_C_layout = cute.make_layout(
            (
                self.cta_tile_shape_mnk[0],
                (self.scale_granularity_n, self.scale_n_per_tile),
                self.num_scale_stage,
            ),
            stride=(0, (0, 1), self.scale_n_per_tile),
        )
        sSFA_view_as_C = cute.make_tensor(sSFA.iterator, sSFA_view_as_C_layout)
        sSFB_view_as_C = cute.make_tensor(sSFB.iterator, sSFB_view_as_C_layout)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
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
        # Partition global/shared tensor for TMA load A/B
        #
        # load scaleA/scaleB
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mSFA_mkl.element_type,
            num_bits_per_copy=mSFA_mkl.element_type.width,
        )
        tiled_copy_sfa = cute.make_tiled_copy_tv(atom_copy,
                                                 cute.make_layout((32, )),
                                                 cute.make_layout((1, )))
        tiled_copy_sfb = cute.make_tiled_copy_tv(atom_copy,
                                                 cute.make_layout((32, )),
                                                 cute.make_layout((1, )))
        thr_copy_sfa = tiled_copy_sfa.get_slice(lane_idx)
        thr_copy_sfb = tiled_copy_sfb.get_slice(lane_idx)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAgSFA_mkl = thr_copy_sfa.partition_S(gSFA_mkl)
        tAsSFA = thr_copy_sfa.partition_D(sSFA)
        tAcSFA = thr_copy_sfa.partition_S(cSFA)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopN, loopK, loopL)
        tBgSFB_nkl = thr_copy_sfb.partition_S(gSFB_nkl)
        tBsSFB = thr_copy_sfb.partition_D(sSFB)
        tBcSFB = thr_copy_sfb.partition_S(cSFB)

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
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(barrier_id=self.cta_sync_bar_id,
                              number_of_threads=self.threads_per_cta)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.sched_warp_id:
            # # smem early release
            # cute.nvgpu.setsmemsize_sync(
            #         self.buffer_align_bytes
            #         + cute.size_in_bytes(self.c_dtype, c_smem_layout_staged)
            # )
            # cute.arch.barrier_arrive(
            #     barrier_id=self.pdl_sync_bar_id,
            #     number_of_threads=self.threads_per_cta,
            # )

            cute.arch.warpgroup_reg_dealloc(self.num_regs_tiled_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            tile_sched.initial_work_tile_info()

            tile_info_producer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Producer, self.num_tile_stage)

            # TODO: skip the searched group
            last_tile_count = cutlass.Int32(0)
            cur_m_boundary = cutlass.Int32(0)
            cur_tile_count = cutlass.Int32(0)
            cur_group_idx = cutlass.Int32(0)
            cur_m_offset = cutlass.Int32(0)
            cur_m_start = cutlass.Int32(0)

            not_last_tile = cutlass.Boolean(1)
            while not_last_tile:

                tile_info_pipeline.producer_acquire(tile_info_producer_state)

                # store the tile info
                linear_idx = tile_sched._current_work_linear_idx
                ntile_shape = tile_sched.params.problem_layout_ncluster_mnl.shape
                int_max = 2147483647
                ntile_layout = cute.make_layout((int_max, ntile_shape[1]),
                                                stride=(ntile_shape[1], 1))
                mma_tile_coord_mn = ntile_layout.get_hier_coord(linear_idx)

                # TODO: only one thread in the warp will execute the group search.
                # it will cause hang sometimes
                (
                    last_tile_count,
                    cur_m_boundary,
                    cur_tile_count,
                    cur_group_idx,
                    cur_m_offset,
                    cur_m_start,
                ) = self.group_search(
                    group_count,
                    mma_tile_coord_mn[0],
                    last_tile_count,
                    cur_m_boundary,
                    cur_tile_count,
                    cur_group_idx,
                    cur_m_offset,
                    offset_mapping,
                )

                not_last_tile = cur_group_idx <= group_count
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = cur_m_start
                    sInfo[(
                        1,
                        tile_info_producer_state.index)] = mma_tile_coord_mn[1]
                    sInfo[(2,
                           tile_info_producer_state.index)] = cur_group_idx - 1
                    sInfo[(3,
                           tile_info_producer_state.index)] = (cur_m_boundary -
                                                               cur_m_start)

                # fence view async shared
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                cute.arch.barrier(
                    barrier_id=self.sched_sync_bar_id,
                    number_of_threads=self.threads_per_warp,
                )
                # pipeline commit
                tile_info_pipeline.producer_commit(tile_info_producer_state)

                # advance to next tile
                tile_info_producer_state.advance()

                tile_sched.advance_to_next_work()
                tile_sched.get_current_work()

            tile_info_pipeline.producer_tail(tile_info_producer_state)

        if warp_idx == self.tma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)

            ab_producer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Producer, self.num_ab_stage)

            tile_info_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_fragment(
                cute.make_layout((4, )).shape, cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in range(4):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[2] < group_count
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            cute.arch.griddepcontrol_wait()
            # cute.arch.griddepcontrol_launch_dependents()

            while is_valid_tile:
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tAgA_slice = tAgA[(None, tile_info[0], None, 0)]
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[(None, tile_info[1], None, tile_info[2])]

                # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state)
                #
                # Tma load loop
                #
                for k_block in cutlass.range_dynamic(0,
                                                     k_block_cnt,
                                                     1,
                                                     unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(
                        ab_producer_state)

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state,
                                                 peek_ab_empty_status)

                    # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt + k_block + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in range(4):
                    tile_info[idx] = sInfo[(idx,
                                            tile_info_consumer_state.index)]
                is_valid_tile = tile_info[2] < group_count
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # cute.arch.griddepcontrol_launch_dependents()
            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

            # # smem early release
            # cute.nvgpu.setsmemsize_sync(
            #         self.buffer_align_bytes
            #         + cute.size_in_bytes(self.c_dtype, c_smem_layout_staged)
            # )
            # cute.arch.barrier_arrive(
            #     barrier_id=self.pdl_sync_bar_id,
            #     number_of_threads=self.threads_per_cta,
            # )

        if warp_idx == self.scale_warp_id:
            # with cute.arch.elect_one():
            #     cute.printf(f"mSFA_mkl = {mSFA_mkl.shape}, {mSFA_mkl.layout}, {mSFA_mkl.iterator}")
            #     cute.printf(f"mSFB_nkl = {mSFB_nkl.shape}, {mSFB_nkl.layout}, {mSFB_nkl.iterator}")
            #     cute.print_tensor(mSFA_mkl, verbose=True)
            #     cute.print_tensor(mSFB_nkl, verbose=True)

            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)

            scale_producer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Producer, self.num_scale_stage)

            tile_info_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_fragment(
                cute.make_layout((4, )).shape, cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in range(4):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[2] < group_count
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            cute.arch.griddepcontrol_wait()

            while is_valid_tile:

                #
                # Prepare the mask for scaleA/scaleB
                #
                tApSFA = cute.make_fragment(
                    cute.make_layout(
                        cute.filter_zeros(
                            cute.slice_(tAsSFA, (None, None, None, 0))).shape),
                    cutlass.Boolean,
                )
                tBpSFB = cute.make_fragment(
                    cute.make_layout(
                        cute.filter_zeros(
                            cute.slice_(tBsSFB, (None, None, None, 0))).shape),
                    cutlass.Boolean,
                )

                # Peek (try_wait) SCALE buffer empty
                scale_producer_state.reset_count()
                peek_scale_empty_status = cutlass.Boolean(1)
                if scale_producer_state.count < k_block_cnt:
                    peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                        scale_producer_state)

                #
                # load loop
                #
                for k_block in cutlass.range_dynamic(0,
                                                     k_block_cnt,
                                                     1,
                                                     unroll=1):
                    #
                    # Slice to per mma tile index
                    #
                    # if k_block == 2:
                    # with cute.arch.elect_one():
                    #     cute.printf(f"tAsSFA = {tAsSFA.shape}, {tAsSFA.layout}, {tAsSFA.iterator}")
                    #     cute.printf(f"tBsSFB = {tBsSFB.shape}, {tBsSFB.layout}, {tBsSFB.iterator}")
                    #     cute.print_tensor(tAsSFA, verbose=True)
                    #     cute.print_tensor(tBsSFB, verbose=True)

                    tAsSFA_pipe = cute.filter_zeros(
                        tAsSFA[(None, None, None, scale_producer_state.index)])
                    tBsSFB_pipe = cute.filter_zeros(
                        tBsSFB[(None, None, None, scale_producer_state.index)])
                    # TODO: hack the layout
                    tAgSFA_k = cute.filter_zeros(tAgSFA_mkl[(
                        None,
                        None,
                        None,
                        0,
                        scale_producer_state.count,
                        tile_info[0],
                    )])
                    tBgSFB_k = cute.filter_zeros(tBgSFB_nkl[(
                        None,
                        None,
                        None,
                        tile_info[1],
                        scale_producer_state.count,
                        tile_info[2],
                    )])
                    # with cute.arch.elect_one():
                    #     cute.printf(f"tAgSFA_k = {tAgSFA_k.shape}, {tAgSFA_k.layout}")
                    #     cute.printf(f"tBgSFB_k = {tBgSFB_k.shape}, {tBgSFB_k.layout}")

                    tAcSFA_compact = cute.filter_zeros(
                        cute.slice_(
                            tAcSFA,
                            (
                                None,
                                None,
                                None,
                                0,
                                scale_producer_state.count,
                                tile_info[0],
                            ),
                        ))
                    tBcSFB_compact = cute.filter_zeros(
                        cute.slice_(
                            tBcSFB,
                            (
                                None,
                                None,
                                None,
                                tile_info[1],
                                scale_producer_state.count,
                                tile_info[2],
                            ),
                        ))
                    # {$nv-internal-release begin}
                    # TODO: Skip more unnecessary load
                    # {$nv-internal-release end}
                    for i in range(cute.size(tApSFA, mode=[1])):
                        coord = tAcSFA_compact[(i)]
                        tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                            # tAcSFA_compact[(i)][0], mSFA_mkl.shape[0]
                            coord[0][1] + coord[2],
                            mSFA_mkl.shape[0][1])
                    for i in range(cute.size(tBpSFB, mode=[1])):
                        tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                            tBcSFB_compact[(i)][0], mSFB_nkl.shape[0])

                    # Conditionally wait for Scale buffer empty
                    scale_pipeline.producer_acquire(scale_producer_state,
                                                    peek_scale_empty_status)

                    # load scaleA/scaleB
                    """
                    with cute.arch.elect_one():
                        print(f"limiN: tiled_copy_sfa = {tiled_copy_sfa}")
                        print(f"limiN: tAgSFA_k = {tAgSFA_k}, ptr = {tAgSFA_k.iterator}")
                        print(f"limiN: tAsSFA_pipe = {tAsSFA_pipe}, ptr = {tAsSFA_pipe.iterator}")
                        print(f"limiN: tApSFA = {tApSFA}, ptr = {tApSFA.iterator}")
                        # cute.printf(f"limin: tAgSFA_k.iterator = {tAgSFA_k.iterator}")
                        # cute.printf(f"limin: tAsSFA_pipe.iterator = {tAsSFA_pipe.iterator}")
                        # cute.printf(f"limin: tApSFA.iterator = {tApSFA.iterator}")
                        cute.print_tensor(tApSFA, verbose=True)
                        cute.print_tensor(tAsSFA_pipe, verbose=True)
                        cute.print_tensor(tAgSFA_k, verbose=True)
                    """
                    cute.copy(tiled_copy_sfa,
                              tAgSFA_k,
                              tAsSFA_pipe,
                              pred=tApSFA)
                    cute.copy(tiled_copy_sfb,
                              tBgSFB_k,
                              tBsSFB_pipe,
                              pred=tBpSFB)

                    scale_pipeline.producer_commit(scale_producer_state)

                    # Peek (try_wait) Scale buffer empty
                    scale_producer_state.advance()
                    peek_scale_empty_status = cutlass.Boolean(1)
                    if scale_producer_state.count < k_block_cnt:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                            scale_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in range(4):
                    tile_info[idx] = sInfo[(idx,
                                            tile_info_consumer_state.index)]
                is_valid_tile = tile_info[2] < group_count
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait Scale buffer empty
            #
            scale_pipeline.producer_tail(scale_producer_state)

            # # smem early release
            # cute.nvgpu.setsmemsize_sync(
            #         self.buffer_align_bytes
            #         + cute.size_in_bytes(self.c_dtype, c_smem_layout_staged)
            # )
            # cute.arch.barrier_arrive(
            #     barrier_id=self.pdl_sync_bar_id,
            #     number_of_threads=self.threads_per_cta,
            # )

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem_ptr_read_threads = 32 * len(
                (self.mma_warp_id, *self.epilog_warp_id,
                 *self.acc_update_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            ab_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_fragment(
                cute.make_layout((4, )).shape, cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in range(4):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[2] < group_count
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:

                # Peek (try_wait) AB buffer full for k_block = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state)

                # Peek (try_wait) Acc buffer empty for k_block = 0
                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                        acc_producer_state)

                #
                # Mma mainloop
                #
                for k_block in cutlass.range_dynamic(0,
                                                     k_block_cnt,
                                                     1,
                                                     unroll=1):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)
                    tCtAcc = tCtAcc_base[(None, None, None,
                                          acc_producer_state.index)]

                    #
                    # Wait for accumulator buffer empty
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state,
                                                      peek_acc_empty_status)

                    #
                    # Reset the ACCUMULATE field for each tile
                    #
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state,
                                                  peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in range(num_kphases):
                            kphase_coord = (
                                None,
                                None,
                                kphase_idx,
                                ab_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kphase_coord],
                                tCrB[kphase_coord],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kphase
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_block = k_block + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state)

                    #
                    # Async arrive accumulator buffer full(each kblock)
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)

                    # Peek (try_wait) Acc buffer empty for k_block = k_block + 1
                    acc_producer_state.advance()
                    if acc_producer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                                acc_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in range(4):
                    tile_info[idx] = sInfo[(idx,
                                            tile_info_consumer_state.index)]
                is_valid_tile = tile_info[2] < group_count
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            cute.arch.griddepcontrol_launch_dependents()

            # # Early smem release, non blocking
            # cute.nvgpu.setsmemsize_sync(
            #     self.buffer_align_bytes
            #     + cute.size_in_bytes(self.c_dtype, c_smem_layout_staged)
            # )
            # cute.arch.barrier_arrive(
            #     barrier_id=self.pdl_sync_bar_id,
            #     number_of_threads=self.threads_per_cta,
            # )

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized acc update warps
        #
        if warp_idx <= self.acc_update_warp_id[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_acc_update_warps)
            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem_ptr_read_threads = 32 * len(
                (self.mma_warp_id, *self.epilog_warp_id,
                 *self.acc_update_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_final = cute.make_tensor(
                tCtAcc_base.iterator + self.tmem_final_offset,
                tCtAcc_base.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tiled_copy_r2t,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rAcc_final,
                tTR_sSFA,
                tTR_sSFB,
                tRT_rAcc,
                tRT_tAcc_base,
            ) = self.acc_update_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                tCtAcc_final,
                tCgC,
                sSFA_view_as_C,
                sSFB_view_as_C,
                epi_tile,
                use_2cta_instrs,
            )

            acc_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.num_acc_stage)

            scale_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.num_scale_stage)

            epi_producer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Producer, 1)

            tile_info_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_fragment(
                cute.make_layout((4, )).shape, cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in range(4):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[2] < group_count
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:

                # initialize the final accumulator
                tTR_rAcc_final.fill(0.0)

                tTR_rSFA = cute.make_fragment(
                    cute.slice_(tTR_sSFA, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )
                tTR_rSFB = cute.make_fragment(
                    cute.slice_(tTR_sSFB, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )

                scale_consumer_state.reset_count()
                peek_scale_full_status = cutlass.Boolean(1)
                if scale_consumer_state.count < k_block_cnt:
                    peek_scale_full_status = scale_pipeline.consumer_try_wait(
                        scale_consumer_state)

                acc_consumer_state.reset_count()
                peek_acc_full_status = cutlass.Boolean(1)
                if acc_consumer_state.count < k_block_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(
                        acc_consumer_state)

                for k_block in cutlass.range_dynamic(0,
                                                     k_block_cnt,
                                                     1,
                                                     unroll=1):
                    # Set tensor memory buffer for current tile
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None,
                                              acc_consumer_state.index)]

                    #
                    # Wait for accumulator buffer full
                    #
                    scale_pipeline.consumer_wait(scale_consumer_state,
                                                 peek_scale_full_status)

                    tTR_sSFA_slice = cute.slice_(
                        tTR_sSFA,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )
                    tTR_sSFB_slice = cute.slice_(
                        tTR_sSFB,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )

                    scale_atom_copy = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.acc_dtype,
                        num_bits_per_copy=self.acc_dtype.width,
                    )

                    cute.copy(scale_atom_copy, tTR_sSFA_slice, tTR_rSFA)
                    cute.copy(scale_atom_copy, tTR_sSFB_slice, tTR_rSFB)

                    #
                    # Wait for accumulator buffer full
                    #

                    acc_pipeline.consumer_wait(acc_consumer_state,
                                               peek_acc_full_status)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3,
                                                cute.rank(tTR_tAcc))

                    #
                    # Update accumulator by scale factor in sub-tiles
                    #
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    for subtile_idx in cutlass.range_dynamic(subtile_cnt):
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        #
                        # Update accumulator by scale factor
                        #
                        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None,
                                                           subtile_idx)]
                        tTR_rSFA_subtile = tTR_rSFA[(None, None, None,
                                                     subtile_idx)]
                        tTR_rSFB_subtile = tTR_rSFB[(None, None, None,
                                                     subtile_idx)]

                        acc_vec = tTR_rAcc.load()
                        final_vec = tTR_rAcc_subtile.load()
                        scale_a = tTR_rSFA_subtile.load()
                        scale_b = tTR_rSFB_subtile.load()
                        scale = scale_a * scale_b
                        final_vec = acc_vec * scale + final_vec
                        tTR_rAcc_subtile.store(final_vec.to(self.acc_dtype))

                    #
                    # Async arrive accumulator buffer empty
                    #
                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                    peek_scale_full_status = cutlass.Boolean(1)
                    if scale_consumer_state.count < k_block_cnt:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(
                            scale_consumer_state)
                    #
                    # Async arrive accumulator buffer empty
                    #
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_block_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(
                            acc_consumer_state)

                tRT_tAcc = tRT_tAcc_base[(None, None, None, None, None, 0)]
                tRT_tAcc = cute.group_modes(tRT_tAcc, 3, cute.rank(tRT_tAcc))
                epi_pipeline.producer_acquire(epi_producer_state)
                cute.copy(tiled_copy_r2t, tTR_rAcc_final, tRT_tAcc)
                epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in range(4):
                    tile_info[idx] = sInfo[(idx,
                                            tile_info_consumer_state.index)]
                is_valid_tile = tile_info[2] < group_count
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # # wait for all other warp have finish setsmemsize, i.e., ensure a/b smem has been consumed in tma warp.
            # cute.arch.barrier(
            #     barrier_id=self.pdl_sync_bar_id,
            #     number_of_threads=self.threads_per_cta,
            # )
            # cute.nvgpu.setsmemsize_sync(
            #     self.buffer_align_bytes
            #     + cute.size_in_bytes(self.c_dtype, c_smem_layout_staged)
            # )
            # cute.nvgpu.setsmemsize_flush()

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[
                -1] and warp_idx >= self.epilog_warp_id[0]:

            # # Early smem release, non blocking
            # cute.nvgpu.setsmemsize_sync(
            #     self.buffer_align_bytes
            #     + cute.size_in_bytes(self.c_dtype, c_smem_layout_staged)
            # )
            # cute.arch.barrier_arrive(
            #     barrier_id=self.pdl_sync_bar_id,
            #     number_of_threads=self.threads_per_cta,
            # )

            cute.arch.warpgroup_reg_alloc(self.num_regs_epilogue_warps)
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem_ptr_read_threads = 32 * len(
                (self.mma_warp_id, *self.epilog_warp_id,
                 *self.acc_update_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_final = cute.make_tensor(
                tCtAcc_base_.iterator + self.tmem_final_offset,
                tCtAcc_base_.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_final,
                                                    tCgC, epi_tile,
                                                    use_2cta_instrs))

            tTR_rC = None
            tiled_copy_r2s = None
            simt_atom = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_gC_partitioned = None
            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC)
            tma_atom_c, bSG_sC, bSG_gC_partitioned, simt_atom, tTR_gC_partitioned = (
                self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c,
                                                    tiled_copy_t2r, tCgC,
                                                    tCgC_tiled, epi_tile, sC))

            thr_mapping = cute.make_identity_tensor(
                (self.mma_tiler[0], self.mma_tiler[1]))
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            m_thr_offset = thr_copy_t2r.partition_D(thr_mapping)[0][0]

            epi_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, 1)

            c_pipeline = None
            if cutlass.const_expr(self.use_tma_store):
                # Threads/warps participating in tma store pipeline
                c_producer_group = utils.CooperativeGroup(
                    utils.Agent.Thread,
                    32 * len(self.epilog_warp_id),
                    32 * len(self.epilog_warp_id),
                )
                c_pipeline = utils.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

            tile_info_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_fragment(
                cute.make_layout((4, )).shape, cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in range(4):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[2] < group_count

            cutlass.Int32(0)

            while is_valid_tile:

                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                tTR_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[(
                    None,
                    None,
                    None,
                    tile_info[0],
                    tile_info[1],
                    0,
                )]
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                tTR_gC = tTR_gC_partitioned[(
                    None,
                    None,
                    None,
                    None,
                    None,
                    tile_info[0] + m_thr_offset,
                    tile_info[1],
                    0,
                )]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None,
                                          epi_consumer_state.index)]

                #
                # Wait for accumulator buffer full
                #
                epi_pipeline.consumer_wait(epi_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

                #
                # Store accumulator to global memory in sub-tiles
                #

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for subtile_idx in cutlass.range_dynamic(subtile_cnt):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # if tile_info[3] >= self.mma_tiler[0]:
                    #     #
                    #     # Convert to C type
                    #     #
                    #     acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    #     acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    #     tRS_rC.store(acc_vec)

                    #     #
                    #     # Store C to shared memory
                    #     #
                    #     num_prev_subtiles = num_prev_subtiles + 1
                    #     c_buffer = num_prev_subtiles % self.num_c_stage
                    #     cute.copy(
                    #         tiled_copy_r2s,
                    #         tRS_rC,
                    #         tRS_sC[(None, None, None, c_buffer)],
                    #     )
                    #     # Fence and barrier to make sure shared memory store is visible to TMA store
                    #     cute.arch.fence_proxy(
                    #         cute.arch.ProxyKind.async_shared,
                    #         space=cute.arch.SharedSpace.shared_cta,
                    #     )
                    #     epilog_threads = 32 * len(self.epilog_warp_id)
                    #     cute.arch.barrier(
                    #         barrier_id=self.epilog_sync_bar_id,
                    #         number_of_threads=epilog_threads,
                    #     )

                    #     #
                    #     # TMA store C to global memory
                    #     #
                    #     if warp_idx == self.epilog_warp_id[0]:
                    #         cute.copy(
                    #             tma_atom_c,
                    #             bSG_sC[(None, c_buffer)],
                    #             bSG_gC[(None, subtile_idx)],
                    #         )
                    #         # Fence and barrier to make sure shared memory store is visible to TMA store
                    #         c_pipeline.producer_commit()
                    #         c_pipeline.producer_acquire()
                    #     cute.arch.barrier(
                    #         barrier_id=self.epilog_sync_bar_id,
                    #         number_of_threads=epilog_threads,
                    #     )
                    # else:
                    #     #
                    #     # Convert to C type
                    #     #
                    #     acc_vec = tTR_rAcc.load()
                    #     acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    #     tTR_rC.store(acc_vec)

                    #     #
                    #     # Store C to global memory
                    #     #
                    #     if m_thr_offset < tile_info[3]:
                    #         cute.copy(
                    #             simt_atom,
                    #             tTR_rC,
                    #             tTR_gC[(None, None, None, subtile_idx)],
                    #         )

                    #
                    # Convert to C type
                    #
                    acc_vec = tTR_rAcc.load()
                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    tTR_rC.store(acc_vec)

                    #
                    # Store C to global memory
                    #
                    if m_thr_offset < tile_info[3]:
                        cute.copy(
                            simt_atom,
                            tTR_rC,
                            tTR_gC[(None, None, None, subtile_idx)],
                        )

                #
                # Async arrive accumulator buffer empty
                #
                epi_pipeline.consumer_release(epi_consumer_state)
                epi_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in range(4):
                    tile_info[idx] = sInfo[(idx,
                                            tile_info_consumer_state.index)]
                is_valid_tile = tile_info[2] < group_count

            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(
                    is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id,
                              number_of_threads=epilog_threads)
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr,
                                              cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(tmem_ptr,
                                       self.num_tmem_alloc_cols,
                                       is_two_cta=use_2cta_instrs)
            #
            # Wait for C store complete
            #
            if cutlass.const_expr(self.use_tma_store):
                c_pipeline.producer_tail()

            # cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def group_search(
        self,
        group_count: cutlass.Int32,
        linear_idx: cutlass.Int32,
        last_tile_count: cutlass.Int32,
        cur_m_boundary: cutlass.Int32,
        cur_tile_count: cutlass.Int32,
        cur_group_idx: cutlass.Int32,
        cur_m_offset: cutlass.Int32,
        offset_mapping: cute.Tensor,
    ):
        not_found = linear_idx >= cur_tile_count
        next_m_boundary = cutlass.Int32(0)
        if not_found:
            cur_group_idx = cur_group_idx + 1
        # print(f"limin: not_found = {not_found}, cur_group_idx = {cur_group_idx}, group_count = {group_count}")
        # print(f"limin: offset_mapping = {offset_mapping}")
        # cute.printf(f"limin: not_found = {not_found}")
        # cute.printf(f"limin: cur_group_idx = {cur_group_idx}")
        # cute.printf(f"limin: group_count = {group_count}")
        # cute.printf(f"limin: i am error")
        while not_found and cur_group_idx <= group_count:
            next_m_boundary = offset_mapping[cur_group_idx]
            num_m_blocks = cute.ceil_div((next_m_boundary - cur_m_boundary),
                                         self.cta_tile_shape_mnk[0])
            next_tile_count = num_m_blocks + cur_tile_count
            not_found = linear_idx >= next_tile_count

            last_tile_count = cur_tile_count
            cur_m_offset = cur_m_boundary
            cur_m_boundary = next_m_boundary
            cur_tile_count = next_tile_count
            if not_found:
                cur_group_idx = cur_group_idx + 1

        cur_m_start = cur_m_offset + self.cta_tile_shape_mnk[0] * (
            linear_idx - last_tile_count)

        return (
            last_tile_count,
            cur_m_boundary,
            cur_tile_count,
            cur_group_idx,
            cur_m_offset,
            cur_m_start,
        )

    def acc_update_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tAcc_final: cute.Tensor,
        gC_mnl: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

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
            - tiled_copy_r2t: The tiled copy operation for register to tmem copy(r2t)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
            - tTR_rAcc_final: The accumulated tensor in register used to hold all t2r results
            - tTR_sSFA: The partitioned tensor SFA by tiled_copy_t2r
            - tTR_sSFB: The partitioned tensor SFB by tiled_copy_t2r
            - tRT_rAcc_final: The accumulated tensor in register used to hold all r2t results
            - tRT_tAcc_final: The partitioned accumulator tensor by tiled_copy_r2t
        :rtype: Tuple[cute.TiledCopy, cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        tmem_load_atom = None
        tmem_store_atom = None
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        elif cutlass.const_expr(self.mma_tiler[0] == 128):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
        else:
            # default: 16dp
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(1)),
                self.acc_dtype,
            )
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        elif cutlass.const_expr(self.mma_tiler[0] == 128):
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
        else:
            # default: 16dp
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(1)),
                self.acc_dtype,
            )

        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tAcc_final_epi = cute.flat_divide(
            tAcc_final[((None, None), 0, 0, None)], epi_tile)

        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom,
                                                tAcc_epi[(None, None, 0, 0, 0)])
        tiled_copy_r2t = tcgen05.make_tmem_copy(
            tmem_store_atom, tAcc_final_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)

        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        sSFA_epi = cute.flat_divide(sSFA, epi_tile)
        sSFB_epi = cute.flat_divide(sSFB, epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_sSFA = thr_copy_t2r.partition_D(sSFA_epi)
        tTR_sSFB = thr_copy_t2r.partition_D(sSFB_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_fragment(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_rAcc_final_ = cute.make_fragment(
            tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape,
            self.acc_dtype)
        tTR_rAcc_final = cute.group_modes(tTR_rAcc_final_, 3,
                                          cute.rank(tTR_rAcc_final_))

        tRT_gC = thr_copy_r2t.partition_S(gC_mnl_epi)
        tRT_tAcc_final = thr_copy_r2t.partition_D(tAcc_final_epi)
        tRT_rAcc_final_ = cute.make_fragment(
            tRT_gC[(None, None, None, None, None, 0, 0, 0)].shape,
            self.acc_dtype)
        tRT_rAcc_final = cute.group_modes(tRT_rAcc_final_, 3,
                                          cute.rank(tRT_rAcc_final_))

        return (
            tiled_copy_t2r,
            tiled_copy_r2t,
            tTR_tAcc,
            tTR_rAcc,
            tTR_rAcc_final,
            tTR_sSFA,
            tTR_sSFB,
            tRT_rAcc_final,
            tRT_tAcc_final,
        )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

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
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r,
                                                tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_fragment(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)

        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

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
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout,
                                                      self.c_dtype,
                                                      self.acc_dtype,
                                                      tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy(
            copy_atom_r2s,
            layout_tv=tiled_copy_t2r.layout_dst_tv_tiled,
            tiler_mn=tiled_copy_t2r.tiler_mn,
        )
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tma_atom_c: cute.CopyAtom,
        tiled_copy_t2r: cute.TiledCopy,
        gC_mnl_tma: cute.Tensor,
        gC_mnl_simt: cute.Tensor,
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

        :return: A tuple containing either:
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
            - For non-TMA store: (simt_atom, tTR_rC, tTR_gC) where:
                - simt_atom: The SIMT copy atom
                - tTR_rC: The register tensor C
                - tTR_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi_tma = cute.flat_divide(
            gC_mnl_tma[((None, None), 0, 0, None, None, None)], epi_tile)
        gC_epi_simt = cute.flat_divide(
            gC_mnl_simt[((None, None), 0, 0, None, None, None)], epi_tile)
        # TMA store
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi_tma, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        # STG store
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_gC = thr_copy_t2r.partition_D(gC_epi_simt)
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                        self.c_dtype)

        return tma_atom_c, bSG_sC, bSG_gC, simt_atom, tTR_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.Layout,
        sfa_dtype: Type[cutlass.Numeric],
        sfb_dtype: Type[cutlass.Numeric],
        sfa_count: int,
        sfb_count: int,
        num_smem_capacity: int,
        occupancy: int,
        use_tma_store: bool,
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
        :type c_layout: utils.Layout
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int
        :param use_tma_store: Whether TMA store is enabled.
        :type use_tma_store: bool

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 3 if mma_tiler_mnk[0] == 128 else 6

        # Default C stages
        num_c_stage = 2 if use_tma_store else 0

        # Default ScaleA/B stages
        num_scale_stage = 10

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
        c_smem_layout_staged_one = (sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        ) if use_tma_store else None)

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
                b_dtype, b_smem_layout_staged_one)
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = (cute.size_in_bytes(
            c_dtype, c_smem_layout_staged_one) if use_tma_store else 0)
        c_bytes = c_bytes_per_stage * num_c_stage
        sfa_bytes = sfa_count * (sfa_dtype.width // 8) * num_scale_stage
        sfb_bytes = sfb_count * (sfb_dtype.width // 8) * num_scale_stage
        scale_bytes = math.ceil((sfa_bytes + sfb_bytes) / 1024) * 1024

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy -
            (mbar_helpers_bytes + c_bytes + scale_bytes)) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if use_tma_store:
            num_c_stage += (num_smem_capacity - occupancy * ab_bytes_per_stage *
                            num_ab_stage - occupancy *
                            (mbar_helpers_bytes + c_bytes + scale_bytes)) // (
                                occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_scale_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
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
            num_ctas_mnl, cluster_shape_mnl)
        grid = (cluster_shape_mn[0], cluster_shape_mn[1], max_active_clusters)

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[cpasync.CopyBulkTensorTileG2SMulticastOp,
               cpasync.CopyBulkTensorTileG2SOp]:
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
            return cpasync.CopyBulkTensorTileG2SMulticastOp(
                tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(
                tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def is_valid_dtypes(
        ab_dtype: Type[cutlass.Numeric],
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
        if ab_dtype not in {
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.TFloat32,
                cutlass.Uint8,
                cutlass.Int8,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
        }:
            is_valid = False
        if (acc_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.Int32}
                or acc_dtype == cutlass.Float16 and ab_dtype not in {
                    cutlass.Float16, cutlass.Float8E4M3FN, cutlass.Float8E5M2
                } or acc_dtype == cutlass.Int32
                and ab_dtype not in {cutlass.Uint8, cutlass.Int8}):
            is_valid = False
        if (acc_dtype == cutlass.Float32 and c_dtype not in {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
        } or acc_dtype == cutlass.Float16 and c_dtype not in {
                cutlass.BFloat16,
                cutlass.Float16,
        } or acc_dtype == cutlass.Int32 and c_dtype not in {
                cutlass.BFloat16,
                cutlass.Float16,
                cutlass.Float32,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
        }):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        use_2cta_instrs: bool,
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
        if not ((not use_2cta_instrs and mma_tiler_mn[0] in [64, 128]) or
                (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])):
            is_valid = False
        # Skip invalid mma tile n
        if mma_tiler_mn[1] not in (128, ):
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (cluster_shape_mn[0] * cluster_shape_mn[1] > 16
                or cluster_shape_mn[0] <= 0 or cluster_shape_mn[1] <= 0
                or not is_power_of_2(cluster_shape_mn[0])
                or not is_power_of_2(cluster_shape_mn[1])):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
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

        if (not check_contigous_16B_alignment(ab_dtype, a_major == "m",
                                              (m, k, l))
                or not check_contigous_16B_alignment(ab_dtype, b_major == "n",
                                                     (n, k, l))
                or not check_contigous_16B_alignment(c_dtype, c_major == "m",
                                                     (m, n, l))):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_epilog_store_option(
        use_2cta_instrs: bool,
        use_tma_store: bool,
        m: int,
        n: int,
        mma_tiler_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the epilogue store option is valid

        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param use_tma_store: Whether to use TMA store
        :type use_tma_store: bool
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]

        :return: True if the epilogue store option is valid, False otherwise
        :rtype: bool
        """

        is_valid = True
        # None TMA store version does not have predication, can not support OOB tiles
        cta_tile_shape_mn = (
            mma_tiler_mn[0] // (2 if use_2cta_instrs else 1),
            mma_tiler_mn[1],
        )
        if not use_tma_store:
            if not (m % cta_tile_shape_mn[0] == 0
                    and n % cta_tile_shape_mn[1] == 0):
                is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Whether to use TMA store
        :type use_tma_store: bool
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

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported cluster shape
        if cluster_shape_mn[0] != 1:
            can_implement = False
        # Skip use_2cta_instrs
        if use_2cta_instrs:
            can_implement = False
        # Skip unsupported types
        if not BlockwiseContiguousGroupedGemmKernel.is_valid_dtypes(
                ab_dtype, acc_dtype, c_dtype):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not BlockwiseContiguousGroupedGemmKernel.is_valid_mma_tiler_and_cluster_shape(
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not BlockwiseContiguousGroupedGemmKernel.is_valid_tensor_alignment(
                m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major):
            can_implement = False
        # Skip invalid epilogue store option
        if not BlockwiseContiguousGroupedGemmKernel.is_valid_epilog_store_option(
                use_2cta_instrs, use_tma_store, m, n, mma_tiler_mn):
            can_implement = False
        return can_implement


def run_contiguous_grouped_blockwise_gemm(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    scale_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    use_tma_store: bool,
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.
    """
    print(
        f"Running Blackwell Persistent Dense Contiguous Grouped GEMM test with:"
    )
    print(f"mnkl: {mnkl}")
    print(
        f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}, Scale dtype: {scale_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(
        f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}"
    )
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Use TMA Store: {'True' if use_tma_store else 'False'}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")

    # Unpack parameters
    m, n, k, l = mnkl

    # Skip unsupported testcase
    if not BlockwiseContiguousGroupedGemmKernel.can_implement(
            ab_dtype,
            acc_dtype,
            c_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            use_tma_store,
            m,
            n,
            k,
            l,
            a_major,
            b_major,
            c_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {acc_dtype}, {c_dtype}, {use_2cta_instrs}, {mma_tiler_mn}, {cluster_shape_mn}, {use_tma_store}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(
        l,
        mode0,
        mode1,
        is_mode0_major,
        dtype,
        is_dynamic_layout=True,
    ):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else: (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = (cutlass_torch.dtype(dtype) if dtype not in {
            cutlass.Float8E5M2, cutlass.Float8E4M3FN
        } else torch.uint8)

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            cutlass_torch.dtype(dtype),
            permute_order=permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=0 if is_unsigned else -2,
                max_val=4 if is_unsigned else 2),
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
                leading_dim=(0 if is_mode0_major else 1))
        cute_tensor = cutlass_torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    def create_mask(num_groups: int, expect_m: int, m_aligned: int):
        valid_m = 0
        group_m_list = []
        # initialize
        offset_mapping = [0]
        for i in range(num_groups):
            group_m = random.randint(int(expect_m * 0.7), int(expect_m * 1.3))
            valid_m += group_m
            # handle the case that valid_m == 0
            if (i == num_groups - 1) and (valid_m == 0):
                group_m = m_aligned
                valid_m += group_m
            group_m_list.append(group_m)
            offset_mapping.append(valid_m)

        offset_mapping = torch.tensor(offset_mapping,
                                      device="cuda",
                                      dtype=torch.int32)

        return valid_m, group_m_list, offset_mapping

    valid_m, group_m_list, _offset_mapping = create_mask(l, m, mma_tiler_mn[0])
    a_ref, a_tensor, a_torch = create_and_permute_tensor(1,
                                                         valid_m,
                                                         k,
                                                         a_major == "m",
                                                         ab_dtype,
                                                         is_dynamic_layout=True)
    b_ref, b_tensor, b_torch = create_and_permute_tensor(
        l,
        n,
        k,
        b_major == "n",
        ab_dtype,
        is_dynamic_layout=True,
    )
    c_ref, c_tensor, c_torch = create_and_permute_tensor(
        1,
        valid_m,
        n,
        c_major == "m",
        c_dtype,
        is_dynamic_layout=True,
    )
    sfa, sfa_tensor, sfa_torch = create_and_permute_tensor(
        1,
        valid_m,
        math.ceil(k / 128),
        True,
        scale_dtype,
    )
    sfb, sfb_tensor, sfb_torch = create_and_permute_tensor(
        l,
        math.ceil(n / 128),
        math.ceil(k / 128),
        False,
        scale_dtype,
    )
    offset_mapping = from_dlpack(_offset_mapping).mark_layout_dynamic()

    # Configure gemm kernel
    gemm = BlockwiseContiguousGroupedGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1])

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    # Compile gemm kernel
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        offset_mapping,
        max_active_clusters,
        current_stream,
    )

    # Launch GPU kernel
    # Warm up
    for i in range(warmup_iterations):
        compiled_gemm(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            offset_mapping,
            current_stream,
        )
    # Execution
    for i in range(iterations):
        compiled_gemm(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            offset_mapping,
            current_stream,
        )

    torch.cuda.synchronize()

    # Compute reference result
    if not skip_ref_check:
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

        updated_a = pad_and_multiply(sfa, a_ref)
        updated_b = pad_and_multiply(sfb, b_ref)

        ref = torch.empty((1, valid_m, n), dtype=torch.float32)
        start = 0
        for i, group_m in enumerate(group_m_list):
            end = start + group_m
            ref[0, start:end, :] = torch.einsum("mk,nk->mn",
                                                updated_a[start:end, :, 0],
                                                updated_b[:, :, i])
            start = end

        ref = ref.permute((1, 2, 0)).to(cutlass_torch.dtype(c_dtype))
        res = c_torch.view(cutlass_torch.dtype(c_dtype))

        torch.testing.assert_close(res.cpu(),
                                   ref.cpu(),
                                   atol=tolerance,
                                   rtol=1e-03)


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers.")

    parser = argparse.ArgumentParser(
        description="Example of Dense Persistent GEMM on Blackwell.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(256, 256, 512, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype",
                        type=cutlass.dtype,
                        default=cutlass.Float8E5M2)
    parser.add_argument("--c_dtype",
                        type=cutlass.dtype,
                        default=cutlass.BFloat16)
    parser.add_argument("--acc_dtype",
                        type=cutlass.dtype,
                        default=cutlass.Float32)
    parser.add_argument("--scale_dtype",
                        type=cutlass.dtype,
                        default=cutlass.Float32)
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--use_tma_store",
                        action="store_true",
                        help="Use tma store or not")
    parser.add_argument("--tolerance",
                        type=float,
                        default=1e-01,
                        help="Tolerance for validation")
    parser.add_argument("--warmup_iterations",
                        type=int,
                        default=0,
                        help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--skip_ref_check",
                        action="store_true",
                        help="Skip reference checking")

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    run_contiguous_grouped_blockwise_gemm(
        args.mnkl,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.scale_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_2cta_instrs,
        args.use_tma_store,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
    )
    print("PASS")
