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

"""
Rubin (SM107) Contiguous Grouped GEMM Kernel with Finalize Fusion (BF16/FP16)

This module implements a contiguous grouped GEMM kernel for Rubin architecture
with fused MoE finalize (scatter-add) operation for BF16/FP16 data types.

Key features:
- Rubin-specific MMA features (standard non-blockscaled MMA)
- Tile scheduling logic for contiguous grouped GEMM
- Fused finalize operation with atomic add for MoE scatter-add

The finalize fusion performs:
1. GEMM: C_permuted = A * B
2. Scatter-add: c[token_idx] += token_scale * C_permuted[permuted_row]

Example usage:
    python rubin_contiguous_grouped_gemm_finalize_fusion.py \\
        --ab_dtype BFloat16 --c_dtype BFloat16 \\
        --mma_inst_shape 128,128,16 --mma_tiler 128,128,64 \\
        --cluster_shape_mn 1,1 --seq_len 4096 \\
        --benchmark 128x7168x2048x8 --iterations 1
"""

import argparse
import os
import re
from typing import List, Tuple, Type

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
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

# ============================================================================
# Inline utility functions
# ============================================================================


@dsl_user_op
def blk_reduce_bf16(dst_gemm, src_smem, size, loc=None, ip=None):
    """Block reduce for BF16 using cp.reduce.async.bulk."""
    from cutlass._mlir.dialects import llvm

    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp32(dst_gemm, src_smem, size, loc=None, ip=None):
    """Block reduce for FP32 using cp.reduce.async.bulk."""
    from cutlass._mlir.dialects import llvm

    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp16(dst_gemm, src_smem, size, loc=None, ip=None):
    """Block reduce for FP16 using cp.reduce.async.bulk."""
    from cutlass._mlir.dialects import llvm

    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


class Sm107ContiguousGroupedGemmFinalizeFusionKernel:
    """Rubin (SM107) Contiguous Grouped GEMM Kernel with Finalize Fusion (BF16/FP16).

    This kernel implements batched matrix multiplication (c = scatter_add(A x B * token_scale))
    with contiguous grouped GEMM support and fused MoE finalize for Rubin GPUs.

    Key features:
    - Persistent tile scheduling with dedicated scheduler warp
    - Warp specialization (scheduler, TMA, MMA, epilogue warps)
    - Fused finalize with atomic add scatter

    :param mma_inst_shape: Shape of MMA instruction (M, N, K)
    :param mma_tiler: Shape of MMA tiler (M, N, K)
    :param cluster_shape_mn: Cluster dimensions (M, N)
    :param raster_along_m: If True, raster tiles along M dimension first
    :param topK: Number of experts selected per token
    """

    def __init__(
        self,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        raster_along_m: bool = False,
        topK: int = 1,
    ):
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        self.topK = topK

        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.occupancy = 1

        # Warp IDs for warp specialization
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
            )
        )

        # Set barriers for synchronization
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

        # For epilogue compatibility
        self.epilogue_warp_id = self.epilog_warp_id
        self.epilog_sync_bar_id = self.epilog_sync_barrier.barrier_id

        # B-reuse pattern disabled for BF16/FP16
        self.enable_breuse = False

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        cta_tile: Tuple[int, int, int],
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int, int]:
        """Compute the number of stages for A/B/C/tile_info operands."""
        # ACC stages (always 2 for BF16/FP16, no B-reuse)
        num_acc_stage = 2

        # Default C stages and tile info stages
        num_c_stage = 1
        num_tile_stage = 2

        # Calculate smem layout and size for one stage
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )

        # Shared memory for epilogue block reduce
        swizzled_pad = 16 // (c_dtype.width // 8)
        c_smem_layout_staged_one = cute.make_layout(
            (cta_tile[0], cta_tile[1]), stride=(cta_tile[1] + swizzled_pad, 1)
        )

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    def _setup_attributes(self):
        """Set up configurations dependent on GEMM inputs."""
        # Configure tiled mma (Rubin SM107, BF16/FP16 - non-blockscaled)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            (self.mma_tiler[0], self.mma_tiler[1]),
        )

        # Compute mma/cluster/tile shapes
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

        # Compute number of multicast CTAs
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile using CUTLASS heuristic to ensure
        # compatibility with the T2R copy atom selected by get_tmem_load_op.
        from cutlass.utils.blackwell_helpers import compute_epilogue_tile_shape

        self.epi_tile = compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup stage counts
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
            self.c_dtype,
            self.cta_tile_shape_mnk,
            self.smem_capacity,
            self.occupancy,
        )

        # Compute A/B/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        # C smem layout for block reduce
        swizzled_pad = 16 // (self.c_dtype.width // 8)
        self.c_smem_layout_staged = cute.make_layout(
            (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], self.num_c_stage),
            stride=(
                self.cta_tile_shape_mnk[1] + swizzled_pad,
                1,
                self.cta_tile_shape_mnk[0] * (self.cta_tile_shape_mnk[1] + 8),
            ),
        )

        # Compute TMEM column counts (no scale factors for BF16/FP16)
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage

        # Epilogue vectorization config
        if cutlass.const_expr(self.c_dtype == cutlass.BFloat16):
            self.element_offset = 8
            self.epi_loop_size = self.epi_tile_n // 8
        elif cutlass.const_expr(self.c_dtype == cutlass.Float32):
            self.element_offset = 2
            self.epi_loop_size = self.epi_tile_n // 2
        else:
            self.element_offset = 1
            self.epi_loop_size = self.epi_tile_n

        # copy_size is in bytes for cp.reduce.async.bulk instruction
        self.copy_size = self.cta_tile_shape_mnk[1] * (self.c_dtype.width // 8)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the contiguous grouped GEMM with finalize fusion (BF16/FP16).

        :param a: Input tensor A (permuted_m, k, 1)
        :param b: Input tensor B (n, k, l)
        :param c: Output tensor (seq_len, n, 1)
        :param tile_idx_to_group_idx: Mapping from tile index to group ID
        :param num_non_exiting_tiles: Number of valid tiles
        :param tile_idx_to_mn_limit: M limit for each tile
        :param max_active_clusters: Maximum number of active clusters
        :param stream: CUDA stream
        :param permuted_idx_to_expanded_idx: Mapping from permuted row to expanded index
        :param token_final_scales: Final scales for each token (seq_len, topK)
        :param epilogue_op: Optional epilogue operation
        """
        # Setup static attributes
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.final_scale_dtype = cutlass.Float32
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.ROW_MAJOR  # Always N-major for GEMM output

        # Setup attributes
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

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
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
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Compute grid size based on GEMM shape
        gemm_m = a.shape[0]
        gemm_n = b.shape[0]
        gemm_l = a.shape[2]
        gemm_shape = (gemm_m, gemm_n, gemm_l)

        self.tile_sched_params, grid = self._compute_grid(
            gemm_shape,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            # (bidx, bidy, expert_idx, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            token_final_scales,
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
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """GPU device kernel for contiguous grouped GEMM with finalize fusion (BF16/FP16)."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Allocate shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize pipelines
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

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

        # Initialize tile info pipeline
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

        # Initialize tensor memory allocator
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Setup smem tensors
        sC = storage.sC.get_tensor(c_smem_layout_staged)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        # Compute multicast masks
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
            c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        # Partition global tensors for TiledMMA
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Partition for TMA load
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

        # Partition for MMA
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized Scheduler warp
        #
        if warp_idx == self.sched_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_valid_tiles = num_non_exiting_tiles[0]

            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)

                    expert_idx = tile_idx_to_group_idx[mma_tile_coord_m]
                    tile_idx = mma_tile_coord_m

                    if tile_idx < num_valid_tiles:
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]

                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        cute.arch.fence_proxy("async.shared", space="cta")

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

                    expert_idx = tile_idx_to_group_idx[mma_tile_coord_m]
                    tile_idx = mma_tile_coord_m

                    if tile_idx < num_valid_tiles:
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        cute.arch.fence_proxy("async.shared", space="cta")

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            # Signal end of work
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info (4 elements: bidx, bidy, expert_idx, valid)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                # Get next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # MMA mainloop
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (
                                None,
                                None,
                                k_block,
                                ab_consumer_state.index,
                            )

                            tiled_mma.set(
                                tcgen05.Field.ACCUMULATE,
                                k_tile != 0 or k_block != 0,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblk_crd],
                                tCrB[kblk_crd],
                                tCtAcc,
                            )

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # Get next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps with finalize fusion
        #
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Layout transformation for tCtAcc_base and tCgC
            tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgC_transformed = transform_partitioned_tensor_layout(tCgC)

            # Partition for epilogue
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = epilogue_tmem_copy_and_partition(
                self, epi_tidx, tCtAcc, tCgC_transformed, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

            # Setup smem copy for block reduce
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(atom, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            token_idx = cutlass.Int32(0)
            token_scale = self.final_scale_dtype(0.0)

            # Get first tile info (5 elements)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Compute base row index for this tile
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]

                # Get accumulator stage index
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                # Group tRS_sC modes
                tRS_sC_grouped = cute.group_modes(tRS_sC, 1, 3)

                # Get m-iteration count and n-subtile count from tTR_tAcc shape
                m_iter_cnt = cute.size(tTR_tAcc.shape[3], mode=[0])
                n_subtile_cnt = cute.size(tTR_tAcc.shape[3], mode=[1])

                cta_tile_m = self.cta_tile_shape_mnk[0]
                cta_tile_n = self.cta_tile_shape_mnk[1]

                if cutlass.const_expr(cta_tile_m >= 128):
                    # Fast path: 1:1 thread-to-row mapping (cta_tile_m >= 128).
                    # Each of 128 epilogue threads owns exactly one row.
                    for m_iter_idx in cutlass.range(m_iter_cnt):
                        permuted_row = tile_m_start + m_iter_idx * 128 + epi_tidx
                        expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                        is_valid_row = permuted_row < tile_info[4]

                        token_idx = cutlass.Int32(0)
                        scale_val = cutlass.Float32(1.0)
                        if is_valid_row:
                            token_idx = expanded_idx // self.topK
                            topk_idx = expanded_idx % self.topK
                            scale_val = token_final_scales[(token_idx, topk_idx)]

                        for n_iter_idx in cutlass.range(n_subtile_cnt):
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, (m_iter_idx, n_iter_idx))]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            acc_vec_final = (scale_val * acc_vec).to(self.c_dtype)
                            tRS_rC.store(acc_vec_final)
                            if is_valid_row:
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rC[None, 0, 0],
                                    tRS_sC_grouped[(None, (m_iter_idx, n_iter_idx), 0)],
                                )

                            cute.arch.fence_proxy("async.shared", space="cta")

                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    # Block reduce
                    for m_iter_idx in cutlass.range(m_iter_cnt):
                        permuted_row = tile_m_start + m_iter_idx * 128 + epi_tidx
                        is_valid_row = permuted_row < tile_info[4]

                        if is_valid_row:
                            expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                            token_idx = expanded_idx // self.topK
                            coord_n = mma_tile_coord_mnl[1] * cta_tile_n
                            scatter_out_offset = cute.domain_offset((token_idx, coord_n, 0), c)
                            sC_row = m_iter_idx * 128 + epi_tidx

                            if cutlass.const_expr(self.c_dtype == cutlass.BFloat16):
                                blk_reduce_bf16(
                                    scatter_out_offset,
                                    sC[sC_row, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                            elif cutlass.const_expr(self.c_dtype == cutlass.Float32):
                                blk_reduce_fp32(
                                    scatter_out_offset,
                                    sC[sC_row, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                            elif cutlass.const_expr(self.c_dtype == cutlass.Float16):
                                blk_reduce_fp16(
                                    scatter_out_offset,
                                    sC[sC_row, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                else:
                    # Small tile path: cta_tile_m < 128 (e.g., 64).
                    # 128 epilogue threads share rows in the T2R copy.
                    # Two-phase approach:
                    # Phase 1: ALL threads write accumulator data to sC via R2S copy.
                    # Phase 2: First cta_tile_m threads apply per-row
                    #          token_scale in-place on sC, then block reduce.
                    for m_iter_idx in cutlass.range(m_iter_cnt):
                        for n_iter_idx in cutlass.range(n_subtile_cnt):
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, (m_iter_idx, n_iter_idx))]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            acc_vec_final = acc_vec.to(self.c_dtype)
                            tRS_rC.store(acc_vec_final)
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rC[None, 0, 0],
                                tRS_sC_grouped[(None, (m_iter_idx, n_iter_idx), 0)],
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")

                    self.epilog_sync_barrier.arrive_and_wait()

                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    for m_iter_idx in cutlass.range(m_iter_cnt):
                        if epi_tidx < cta_tile_m:
                            permuted_row = tile_m_start + m_iter_idx * cta_tile_m + epi_tidx
                            is_valid_row = permuted_row < tile_info[4]

                            if is_valid_row:
                                expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                                token_idx = expanded_idx // self.topK
                                topk_idx = expanded_idx % self.topK
                                token_scale = token_final_scales[(token_idx, topk_idx)]
                                coord_n = mma_tile_coord_mnl[1] * cta_tile_n
                                scatter_out_offset = cute.domain_offset((token_idx, coord_n, 0), c)
                                sC_row = m_iter_idx * cta_tile_m + epi_tidx

                                # Apply per-row token_scale to sC in-place
                                for col in cutlass.range(cta_tile_n, unroll_full=True):
                                    val = sC[(sC_row, col, 0)]
                                    sC[(sC_row, col, 0)] = (
                                        val.to(cutlass.Float32) * token_scale
                                    ).to(self.c_dtype)

                                cute.arch.fence_proxy("async.shared", space="cta")

                                if cutlass.const_expr(self.c_dtype == cutlass.BFloat16):
                                    blk_reduce_bf16(
                                        scatter_out_offset,
                                        sC[sC_row, None, 0],
                                        cutlass.Int32(self.copy_size),
                                    )
                                elif cutlass.const_expr(self.c_dtype == cutlass.Float32):
                                    blk_reduce_fp32(
                                        scatter_out_offset,
                                        sC[sC_row, None, 0],
                                        cutlass.Int32(self.copy_size),
                                    )
                                elif cutlass.const_expr(self.c_dtype == cutlass.Float16):
                                    blk_reduce_fp16(
                                        scatter_out_offset,
                                        sC[sC_row, None, 0],
                                        cutlass.Int32(self.copy_size),
                                    )
                self.epilog_sync_barrier.arrive_and_wait()

                # Get next tile (5 elements)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # Dealloc tensor memory
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

    @staticmethod
    def _compute_grid(
        gemm_shape: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid size based on GEMM shape."""
        (m, n, l) = gemm_shape  # noqa: E741

        num_ctas_m = cute.ceil_div(m, cta_tile_shape_mnk[0])
        num_ctas_n = cute.ceil_div(n, cta_tile_shape_mnk[1])
        num_ctas_l = l

        num_ctas_mnl = (num_ctas_m, num_ctas_n, num_ctas_l)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Check if the dtypes are valid for BF16/FP16 GEMM."""
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
        """Check if layouts and dtypes are valid combinations."""
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
        if mma_inst_shape[1] not in [64, 128, 192, 256]:
            return False

        # Check valid mma_tiler
        if mma_tiler[0] not in [64, 128, 256]:
            return False
        if mma_tiler[1] not in [64, 128, 192, 256]:
            return False

        # No B-reuse for BF16/FP16: mma_tiler[0] == mma_inst_shape[0] always
        if mma_tiler[0] != mma_inst_shape[0]:
            return False
        if mma_tiler[1] != mma_inst_shape[1]:
            return False

        # K-dimension constraints for BF16/FP16: K=16 per MMA instruction, mma_tiler K=64
        if mma_tiler[2] != 64 or mma_inst_shape[2] != 16:
            return False

        # 2CTA BF16 MMA requires N >= 128 (N=64 produces incorrect results
        # with sm100_utils.make_trivial_tiled_mma + CtaGroup.TWO; the
        # blockscaled MMA works at N=64 because it uses a different instruction)
        if mma_inst_shape[0] == 256 and mma_inst_shape[1] < 128:
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

        return True

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
        """Check if the tensor alignment is valid (16B alignment)."""

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
            return False

        return True

    @classmethod
    def can_implement(
        cls,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if the GEMM can be implemented for BF16/FP16."""
        if not cls.is_valid_dtypes(a_dtype, c_dtype):
            return False

        if not cls.is_valid_layouts(a_dtype, c_dtype, a_major, b_major, c_major):
            return False

        if not cls.is_valid_mma_tiler_and_cluster_shape(
            a_dtype, b_dtype, mma_inst_shape, mma_tiler, cluster_shape_mn
        ):
            return False

        if not cls.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        return True

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        num_tokens: cutlass.Int64,
        top_k: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_stride_row: cutlass.Int64 = cutlass.Int64(0),
    ):
        num_tiles = m // tile_size
        a = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        actual_c_stride_row = n if c_stride_row == 0 else c_stride_row
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout(
                (num_tokens, n, 1),
                stride=(actual_c_stride_row, 1, num_tokens * actual_c_stride_row),
            ),
        )
        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        permuted_idx_to_expanded_idx = cute.make_tensor(
            permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        token_final_scales = cute.make_tensor(
            token_final_scales_ptr,
            layout=cute.make_ordered_layout((num_tokens, top_k), order=(1, 0)),
        )

        return self(
            a,
            b,
            c,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            max_active_clusters=max_active_clusters,
            stream=stream,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            epilogue_op=epilogue_op,
        )


# ============================================================================
# Run utilities
# ============================================================================


def create_mask(group_m_list, cta_tile_mn, permuted_m=None):
    """Create mask and group mapping for contiguous grouped GEMM.

    :param group_m_list: List of M values for each group
    :param cta_tile_mn: CTA tile size tuple (M, N)
    :param permuted_m: Optional padded M dimension for cuda_graph support.
    """
    m_aligned = cta_tile_mn[0]
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []
    tile_idx_to_mn_limit = []

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + m_aligned - 1) // m_aligned) * m_aligned
        aligned_group_m_list.append(aligned_group_m)

        num_tiles_in_group = aligned_group_m // cta_tile_mn[0]
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)

        tile_idx_to_mn_limit.extend([group_m + valid_m] * num_tiles_in_group)
        valid_m += aligned_group_m

    num_non_exiting_tiles = len(tile_idx_to_expert_idx)

    if permuted_m is not None:
        if permuted_m < valid_m:
            raise ValueError(f"permuted_m ({permuted_m}) must be >= valid_m ({valid_m}).")
        if permuted_m > valid_m:
            num_padding_tiles = (permuted_m - valid_m) // cta_tile_mn[0]
            tile_idx_to_expert_idx.extend([0] * num_padding_tiles)

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


def create_fused_finalize_tensors(seq_len, topK, permuted_m, group_m_list, mma_tiler_mn):
    """Create tensors for fused finalize operation."""
    m_aligned = mma_tiler_mn[0]
    permuted_idx_to_expanded_idx_tensor = torch.empty(
        (permuted_m,), dtype=torch.int32, device="cuda"
    ).fill_(-1)
    token_final_scales = torch.rand(seq_len, topK).to(dtype=torch.float32).cuda()
    token_final_scales = token_final_scales / token_final_scales.sum(dim=1, keepdim=True)

    start_idx = 0
    for group_idx in range(len(group_m_list)):
        m_per_group = group_m_list[group_idx]

        if m_per_group > 0:
            expert_set_idx = group_idx // topK
            k_in_set = group_idx % topK
            start_token = expert_set_idx * m_per_group

            token_indices = torch.arange(
                start_token, start_token + m_per_group, dtype=torch.int32, device="cuda"
            )
            token_indices = token_indices % seq_len
            expanded_idx = token_indices * topK + k_in_set

            permuted_idx_to_expanded_idx_tensor[start_idx : (start_idx + m_per_group)] = (
                expanded_idx
            )
        m_aligned_per_group = ((m_per_group + m_aligned - 1) // m_aligned) * m_aligned
        start_idx += m_aligned_per_group

    return (
        permuted_idx_to_expanded_idx_tensor,
        token_final_scales,
        from_dlpack(permuted_idx_to_expanded_idx_tensor).mark_layout_dynamic(),
        from_dlpack(token_final_scales).mark_layout_dynamic(),
    )


def create_tensors(
    l,  # noqa: E741
    group_m_list,
    n,
    k,
    a_major,
    b_major,
    cd_major,
    ab_dtype,
    c_dtype,
    mma_tiler_mn,
    permuted_m=None,
    seq_len=None,
):
    """Create tensors for contiguous grouped GEMM with finalize fusion (BF16/FP16)."""
    torch.manual_seed(1111)

    alpha_torch_cpu = torch.ones((l,), dtype=torch.float32) * 0.1

    (
        valid_m,
        aligned_group_m_list,
        _tile_idx_to_expert_idx,
        _num_non_exiting_tiles,
        _tile_idx_to_mn_limit,
    ) = create_mask(group_m_list, mma_tiler_mn, permuted_m)

    tensor_m = permuted_m if permuted_m is not None else valid_m

    a_torch_cpu = cutlass_torch.matrix(1, tensor_m, k, a_major == "m", cutlass.Float32)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    c_torch_cpu = cutlass_torch.matrix(1, seq_len, n, cd_major == "m", cutlass.Float32)

    c_torch_cpu.fill_(0)

    a_tensor, a_torch_gpu = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch_gpu = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
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

    tile_idx_to_expert_idx = from_dlpack(_tile_idx_to_expert_idx).mark_layout_dynamic()
    num_non_exiting_tiles = from_dlpack(_num_non_exiting_tiles).mark_layout_dynamic()
    tile_idx_to_mn_limit = from_dlpack(_tile_idx_to_mn_limit).mark_layout_dynamic()

    alpha = from_dlpack(alpha_torch_cpu.cuda()).mark_layout_dynamic()

    c_torch_gpu.fill_(0)

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        tile_idx_to_mn_limit,
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
    )


def verify_reference_result(
    a_torch_cpu: torch.Tensor,
    b_torch_cpu: torch.Tensor,
    permuted_idx_to_expanded_idx_torch: torch.Tensor,
    token_final_scales_torch: torch.Tensor,
    group_m_list: List[int],
    aligned_group_m_list: List[int],
    c_dtype: torch.dtype,
    valid_m: int,
    n: int,
    topK: int,
    seq_len: int,
) -> torch.Tensor:
    """Compute reference result for validation (BF16/FP16 - no scale factors)."""
    gemm_output = torch.empty((1, valid_m, n), dtype=torch.float32)
    valid_mask = torch.zeros((valid_m,), dtype=torch.bool, device="cuda")

    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        res_a = a_torch_cpu[start:end, :, 0]
        res_b = b_torch_cpu[:, :, i]
        gemm_output[0, start:end, :] = torch.einsum("mk,nk->mn", res_a, res_b)
        valid_mask[start : start + group_m_list[i]] = 1
        start = end

    gemm_output = gemm_output.permute((1, 2, 0)).cuda()

    final_output = torch.zeros((seq_len, n), dtype=c_dtype).cuda()

    gemm_output = gemm_output[:valid_m, :, 0].clone()
    expanded_idx_all = permuted_idx_to_expanded_idx_torch[:valid_m]

    expanded_idx_valid = expanded_idx_all[valid_mask]
    gemm_output_valid = gemm_output[valid_mask]

    token_idx = expanded_idx_valid // topK
    topk_idx = expanded_idx_valid % topK
    scales = token_final_scales_torch[token_idx, topk_idx]
    scaled_output = gemm_output_valid * scales.unsqueeze(1)
    scaled_output = scaled_output.to(c_dtype)

    for i in range(len(token_idx)):
        final_output[token_idx[i]] += scaled_output[i]

    return final_output


def run(
    nkl: Tuple[int, int, int],
    group_m_list: Tuple[int, ...],
    ab_dtype: Type[cutlass.Numeric],
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
    topK: int = 8,
    seq_len: int = 4096,
    raster_along_m: bool = False,
    use_cupti: bool = False,
    **kwargs,
):
    """Run the Rubin contiguous grouped GEMM kernel with finalize fusion (BF16/FP16)."""
    mma_tiler_mn = (mma_tiler[0], mma_tiler[1])
    m_aligned = mma_tiler[0]

    print(
        "Running Rubin Persistent Dense Contiguous Grouped GEMM Finalize Fusion (BF16/FP16) test with:"
    )
    print(f"nkl: {nkl}")
    print(f"group_m_list: {group_m_list}")
    print(f"AB dtype: {ab_dtype}, C dtype: {c_dtype}")
    print(f"Group M alignment: {m_aligned}")
    if permuted_m is not None:
        print(f"Padded M (CUDA graph support): {permuted_m}")
    print(f"Fused finalize enabled with topK={topK}")
    print(f"Sequence length: {seq_len}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, Out: {c_major}")
    print(f"Mma Inst Shape (M, N, K): {mma_inst_shape}")
    print(f"Mma Tiler (M, N, K): {mma_tiler}")
    print(f"Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Raster along M: {raster_along_m}")
    print(f"Use CUPTI: {'True' if use_cupti else 'False'}")
    n, k, l = nkl  # noqa: E741

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    if not Sm107ContiguousGroupedGemmFinalizeFusionKernel.can_implement(
        ab_dtype,
        ab_dtype,
        c_dtype,
        mma_inst_shape,
        mma_tiler,
        cluster_shape_mn,
        m_aligned,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        raise TypeError(
            f"Unsupported testcase ab_dtype={ab_dtype}, c_dtype={c_dtype}, "
            f"mma_inst_shape={mma_inst_shape}, mma_tiler={mma_tiler}, "
            f"cluster_shape_mn={cluster_shape_mn}, n={n}, k={k}, l={l}, "
            f"a_major={a_major}, b_major={b_major}, c_major={c_major}, m_aligned={m_aligned}"
        )

    (
        a_tensor,
        b_tensor,
        c_tensor,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        tile_idx_to_mn_limit,
        _alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        _alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        aligned_group_m_list,
        valid_m,
    ) = create_tensors(
        l,
        group_m_list,
        n,
        k,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        c_dtype,
        mma_tiler_mn,
        permuted_m,
        seq_len,
    )

    tensor_m = permuted_m if permuted_m is not None else valid_m

    (
        permuted_idx_to_expanded_idx_torch,
        token_final_scales_torch,
        permuted_idx_to_expanded_idx,
        token_final_scales,
    ) = create_fused_finalize_tensors(
        seq_len,
        topK,
        tensor_m,
        group_m_list,
        mma_tiler_mn,
    )

    gemm = Sm107ContiguousGroupedGemmFinalizeFusionKernel(
        mma_inst_shape,
        mma_tiler,
        cluster_shape_mn,
        raster_along_m,
        topK,
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    print(f"max_active_clusters: {max_active_clusters}")

    current_stream = cutlass_torch.default_stream()

    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        tile_idx_to_mn_limit,
        max_active_clusters,
        current_stream,
        permuted_idx_to_expanded_idx,
        token_final_scales,
    )

    if not skip_ref_check:
        compiled_gemm(
            a_tensor,
            b_tensor,
            c_tensor,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            current_stream,
            permuted_idx_to_expanded_idx,
            token_final_scales,
        )

        torch.cuda.synchronize()
        print("Verifying results...")
        ref_result = verify_reference_result(
            a_torch_cpu,
            b_torch_cpu,
            permuted_idx_to_expanded_idx_torch,
            token_final_scales_torch,
            group_m_list,
            aligned_group_m_list,
            c_torch_gpu.dtype,
            valid_m,
            n,
            topK,
            seq_len,
        )

        actual_result = c_torch_gpu[:, :, 0]
        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(actual_result, ref_result, atol=tolerance, rtol=1e-02)

    def generate_tensors():
        (
            a_tensor,
            b_tensor,
            c_tensor,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            _alpha,
            *_,
        ) = create_tensors(
            l,
            group_m_list,
            n,
            k,
            a_major,
            b_major,
            c_major,
            ab_dtype,
            c_dtype,
            mma_tiler_mn,
            permuted_m,
            seq_len,
        )

        (
            _,
            _,
            permuted_idx_to_expanded_idx,
            token_final_scales,
        ) = create_fused_finalize_tensors(
            seq_len,
            topK,
            tensor_m,
            group_m_list,
            mma_tiler_mn,
        )

        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            c_tensor,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            current_stream,
            permuted_idx_to_expanded_idx,
            token_final_scales,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch_gpu.numel() * a_torch_gpu.element_size()
            + b_torch_gpu.numel() * b_torch_gpu.element_size()
            + c_torch_gpu.numel() * c_torch_gpu.element_size()
            + (tensor_m // mma_tiler_mn[0]) * 4
            + 1 * 4
            + 0  # alpha removed
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
    """Parse comma-separated integers from string."""
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
        l = len(problems)  # noqa: E741

        m_values = tuple(m for m, _, _ in problems)

        print(f"Loaded {l} problems from benchmark file")
        print(f"Using N={n}, K={k}, L={l}")
        print(f"M values per group: {m_values}")

        return ((n, k, l), m_values)

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
            l = len(m_values)  # noqa: E741
            print(f"Parsed benchmark arg: N={n}, K={k}, L={l}")
            print(f"M values per group: {m_values}")
            return ((n, k, l), m_values)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer list in benchmark argument: {arg}")

    parts = arg.split("x")
    if len(parts) == 4:
        try:
            m, n, k, l = [int(x.strip()) for x in parts]  # noqa: E741
            m_values = tuple([m] * l)
            print(f"Parsed benchmark arg: M={m}, N={n}, K={k}, L={l}")
            return ((n, k, l), m_values)
        except ValueError:
            pass

    raise argparse.ArgumentTypeError(f"Invalid benchmark argument format. Got: {arg}")


def main():
    """Main entry point for running the kernel."""
    parser = argparse.ArgumentParser(
        description="Rubin (SM107) BF16/FP16 Contiguous Grouped GEMM Finalize Fusion kernel."
    )

    parser.add_argument(
        "--nkl",
        type=parse_comma_separated_ints,
        default=(256, 512, 1),
        help="nkl dimensions: N, K, L (comma-separated)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark file or 'MxNxKxL' or '[m0,m1,...]xNxK'.",
    )

    parser.add_argument(
        "--permuted_m",
        type=int,
        default=None,
        help="Optional padded M dimension for CUDA graph support.",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Sequence length for MoE, used by fused finalize.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Top-K experts per token (used for fused finalize)",
    )

    parser.add_argument(
        "--mma_inst_shape",
        type=parse_comma_separated_ints,
        default=(128, 128, 16),
        help="MMA instruction shape M, N, K (comma-separated).",
    )

    parser.add_argument(
        "--mma_tiler",
        type=parse_comma_separated_ints,
        default=(128, 128, 64),
        help="MMA tile shape M, N, K (comma-separated).",
    )

    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )

    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
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
        parser.error("No benchmark file or benchmark argument provided")

    if len(args.mma_inst_shape) != 3:
        parser.error("--mma_inst_shape must contain exactly 3 values")

    if len(args.mma_tiler) != 3:
        parser.error("--mma_tiler must contain exactly 3 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        nkl,
        group_m_list,
        args.ab_dtype,
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
        args.topk,
        args.seq_len,
        args.raster_along_m,
        args.use_cupti,
    )
    print("exec_time: ", exec_time)
    print("PASS")


if __name__ == "__main__":
    main()
