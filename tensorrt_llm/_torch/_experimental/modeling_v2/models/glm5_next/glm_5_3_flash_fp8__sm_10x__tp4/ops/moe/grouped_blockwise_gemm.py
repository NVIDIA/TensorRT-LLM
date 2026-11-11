# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The kernel body is adapted from tensorrt_llm/_torch/kernels/blackwell/blockwise_gemm/
# blockwise_gemm.py, itself derived from the CUTLASS CuTe DSL blockwise GEMM example:
#
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
"""Grouped (contiguous MoE layout) FP8 GEMM with FP32 block scales on SM100, in CuTe DSL.

This is TensorRT-LLM's persistent, warp-specialized dense blockwise GEMM
(``Sm100BlockwiseGemmKernel``) with the batch mode turned into the expert index. The FP32
scales are applied by promotion, exactly as in the dense kernel, so arbitrary (not just
power-of-two) 1 x 128 activation and 128 x 128 weight scales are kept: the result is as
precise as the dense kernel's.

Layouts (``rows`` is the worst-case row capacity, a multiple of 128):

* ``a`` ``[rows, K]`` FP8 E4M3, row-major. Rows are grouped by expert in contiguous segments,
  each starting at a multiple of 128.
* ``sfa`` ``[K / 128, rows]`` FP32: K-block-major, the layout ``fp8_quantize_1x128`` returns.
* ``b`` ``[E, N, K]`` FP8 E4M3, K-major per expert; ``sfb`` ``[E, N / 128, K / 128]`` FP32.
* ``tile_expert`` ``[rows / 128]`` Int32 on the GPU: the expert of each 128-row tile, -1 if
  unused.
* ``out`` ``[rows, N]`` BF16.

The tile scheduler walks every (M tile, N tile) of the worst case, so the grid never depends
on device data. The scheduler warp, which already broadcasts each tile's coordinates to the
other warps through shared memory, reads ``tile_expert`` for the tile, skips tiles whose
expert is -1 (no loads, MMA or store) and publishes the expert in the batch slot. The TMA and
scale warps then index B and SFB by that expert, and A, SFA and C by batch 0.

A cluster must work on one expert, so the MMA tile is at most 128 rows and the cluster grows
only along N (a 2-CTA MMA uses 64-row CTA tiles inside one 128-row group).

``grouped_blockwise_fp8_gemm_gathered`` reads A as ``x[row_index]`` from the un-permuted
activations instead of a pre-gathered copy. Each lane of the TMA warp issues one
``tile::gather4`` (four source rows of 128 bytes) per K tile into the SW128 A stage, through a
2D descriptor whose box is one row; with an A-multicast cluster the CTAs split the row groups.
The scale warp copies each row's scales with 4-byte ``cp.async`` from the same indices. The
bytes in shared memory match the TMA tile load's, so the output is bitwise the same.
"""

from typing import Dict, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch.kernels.blackwell.blockwise_gemm.blockwise_gemm import (
    Sm100BlockwiseGemmKernel,
)


@dsl_user_op
def _tma_gather4_load(tma_atom, smem_dst, mbar, col, rows, mcast_mask, *, loc=None, ip=None):
    """Issue one 1-CTA ``cp.async.bulk.tensor.2d...tile::gather4``: rows ``rows[0..3]``,
    columns ``[col, col + box)``, into four consecutive smem box rows at ``smem_dst``.

    A row index outside the tensor (e.g. -1) makes the TMA unit fill that row with zeros.
    With ``mcast_mask`` the rows land at the same offset in every CTA of the mask.
    """
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(), cute.AddressSpace.generic, 64
    )
    desc = _cute_nvgpu_ir.get_tma_desc_addr(desc_ptr_ty, exec_atom, loc=loc, ip=ip)
    operands = [
        cutlass.Int32(smem_dst.toint()).ir_value(loc=loc, ip=ip),
        desc.toint().ir_value(loc=loc, ip=ip),
        cutlass.Int32(mbar.toint()).ir_value(loc=loc, ip=ip),
        cutlass.Int32(col).ir_value(loc=loc, ip=ip),
    ] + [cutlass.Int32(rows[j]).ir_value(loc=loc, ip=ip) for j in range(4)]
    if mcast_mask is None:
        asm = (
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
            ".mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, $6, $7}], [$2];"
        )
        constraints = "r,l,r,r,r,r,r,r"
    else:
        asm = (
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
            ".mbarrier::complete_tx::bytes.multicast::cluster"
            " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8;"
        )
        operands.append(cutlass.Int16(mcast_mask).ir_value(loc=loc, ip=ip))
        constraints = "r,l,r,r,r,r,r,r,h"
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class Sm100GroupedBlockwiseGemmKernel(Sm100BlockwiseGemmKernel):
    """``Sm100BlockwiseGemmKernel`` over a contiguous grouped layout, B indexed per M tile.

    :param use_2cta_instrs: Use the 2-CTA MMA (then the MMA tile M must be 128).
    :param mma_tiler_mn: MMA tile; M in {64, 128}, N 128.
    :param cluster_shape_mn: (1, cn), or (2, cn) with the 2-CTA MMA; cn a power of two.
    :param raster_along_m: Walk tiles M first. N first (the default) keeps the concurrently
        running CTAs on the same A rows, which lets A come from L2.
    """

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        tile_expert: cute.Tensor,
        row_index: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Set up the TMA atoms, the grid and the shared storage, then launch the kernel.

        ``a`` is (rows, K, 1), ``b`` is (N, K, E), ``c`` is (rows, N, 1), ``sfa`` is
        (rows, K / 128, 1), ``sfb`` is (N / 128, K / 128, E) and ``tile_expert`` is
        (rows / 128,) Int32. With ``gather_a``, ``a`` and ``sfa`` are the un-permuted
        (T, K, 1) and (T, K / 128, 1) tensors and ``row_index`` (rows,) Int32 gives each
        output row's source row; otherwise ``row_index`` is unused.
        """
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sfa_dtype = sfa.element_type
        self.sfb_dtype = sfb.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

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
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        if cutlass.const_expr(self.gather_a):
            # A 2D descriptor whose box is one 128-byte row with the 128-byte swizzle of the
            # K-major A tile: a tile::gather4 through it writes four rows at the offsets the
            # MMA reads (row m of a stage at m * 128 bytes).
            a_row_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout((1, self.mma_tiler[2]), stride=(self.mma_tiler[2], 1)),
            )
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                a[(None, None, 0)],
                a_row_layout,
                (1, self.mma_tiler[2]),
            )
        else:
            a_op = self._get_tma_atom_kind(atom_thr_size, self.is_a_mcast)
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
        b_op = self._get_tma_atom_kind(atom_thr_size, self.is_b_mcast)
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
        c_cta_v_layout = cute.composition(cute.make_identity_layout(c.shape), self.epi_tile)
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
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
        self.tile_sched_params, grid = self._compute_grid_grouped(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024

        c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_scale_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
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
                cute.struct.MemRange[self.sfa_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sfb_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tensor_sfa,
            tensor_sfb,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            tile_expert,
            row_index,
            sfa,
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
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        mTileExpert: cute.Tensor,
        mRowIndex: cute.Tensor,
        mSFX: cute.Tensor,
        epilogue_op: cutlass.Constexpr,
    ):
        """Persistent grouped GEMM. The L coordinate a warp reads from the tile info is the
        tile's expert: it indexes B and SFB, while A, C and SFA always use L = 0."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
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

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Initialize mainloop ab_pipeline (barrier) and states
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

        # Initialize mainloop scale_pipeline (barrier) and states
        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
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

        # Initialize epilogue pipeline (barrier) and states
        epi_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.acc_update_warp_id),
        )
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        epi_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.epi_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize tile info pipeline (barrier) and states
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
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

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
        # (bidx, bidy, bidz, valid)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = storage.sInfo.get_tensor(info_layout)

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
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (bM, bK, loopM, loopK, loopL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # coordinate
        cSFA_mkl = cute.make_identity_tensor(cute.shape(mSFA_mkl))
        cSFB_nkl = cute.make_identity_tensor(cute.shape(mSFB_nkl))
        # (bM, bK, loopM, loopK, loopL)
        cSFA = cute.local_tile(
            cSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        cSFB = cute.local_tile(
            cSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gB_nkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)

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
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA = None
        tAgA = None
        if cutlass.const_expr(not self.gather_a):
            # (bM, bK, loopM, loopK, loopL)
            gA_mkl = cute.local_tile(
                mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
            tCgA = thr_mma.partition_A(gA_mkl)
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
        tiled_copy_sfa = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        tiled_copy_sfb = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
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
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized Schedule warp
        #
        if warp_idx == self.sched_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_sched_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()
            cur_expert = self._tile_expert(work_tile, mTileExpert)
            while work_tile.is_valid_tile & (cur_expert < 0):
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                cur_expert = self._tile_expert(work_tile, mTileExpert)

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            while work_tile.is_valid_tile:
                # query next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                cur_expert = self._tile_expert(work_tile, mTileExpert)
                while work_tile.is_valid_tile & (cur_expert < 0):
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                    cur_expert = self._tile_expert(work_tile, mTileExpert)

                # acquire tile info pipeline
                tile_info_pipeline.producer_acquire(tile_info_producer_state)

                # store the tile info
                cur_tile_coord = work_tile.tile_idx
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                    sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                    sInfo[(2, tile_info_producer_state.index)] = cur_expert
                    sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                        work_tile.is_valid_tile
                    )

                # fence view async shared
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.sched_sync_barrier.arrive_and_wait()
                # commit tile info pipeline
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()

            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()
            cur_expert = self._tile_expert(work_tile, mTileExpert)
            while work_tile.is_valid_tile & (cur_expert < 0):
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                cur_expert = self._tile_expert(work_tile, mTileExpert)

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_expert
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                tAgA_slice = None
                a_rows = None
                if cutlass.const_expr(self.gather_a):
                    # Lane l gathers rows 4 l .. 4 l + 3 of the tile.
                    a_rows = cute.make_rmem_tensor((4,), cutlass.Int32)
                    a_row_base = tile_info[0] * self.cta_tile_shape_mnk[0] + lane_idx * 4
                    for j in cutlass.range_constexpr(4):
                        a_rows[j] = mRowIndex[a_row_base + j]
                else:
                    # ((atom_v, rest_v), loopK)
                    tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    a_stage = ab_producer_state.index
                    tAgA_k = None
                    tAsA_pipe = None
                    if cutlass.const_expr(not self.gather_a):
                        tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                        tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                    # TMA load A/B
                    if cutlass.const_expr(self.gather_a):
                        a_dst = sA.iterator + (
                            a_stage * self.cta_tile_shape_mnk[0] * self.mma_tiler[2]
                            + lane_idx * 4 * self.mma_tiler[2]
                        )
                        if cutlass.const_expr(self.is_a_mcast):
                            # The cluster's CTAs split the row groups and multicast them.
                            if lane_idx % self.num_mcast_ctas_a == block_in_cluster_coord_vmnk[2]:
                                _tma_gather4_load(
                                    tma_atom_a,
                                    a_dst,
                                    tma_bar,
                                    k_tile * self.mma_tiler[2],
                                    a_rows,
                                    a_full_mcast_mask,
                                )
                        else:
                            _tma_gather4_load(
                                tma_atom_a,
                                a_dst,
                                tma_bar,
                                k_tile * self.mma_tiler[2],
                                a_rows,
                                None,
                            )
                    else:
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
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized Scale load warp
        #
        if warp_idx == self.scale_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()
            cur_expert = self._tile_expert(work_tile, mTileExpert)
            while work_tile.is_valid_tile & (cur_expert < 0):
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                cur_expert = self._tile_expert(work_tile, mTileExpert)

            scale_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_scale_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_expert
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                #
                # Prepare the mask for scaleA/scaleB
                #
                tApSFA = cute.make_rmem_tensor(
                    cute.make_layout(
                        cute.filter_zeros(cute.slice_(tAsSFA, (None, None, None, 0))).shape
                    ),
                    cutlass.Boolean,
                )
                tBpSFB = cute.make_rmem_tensor(
                    cute.make_layout(
                        cute.filter_zeros(cute.slice_(tBsSFB, (None, None, None, 0))).shape
                    ),
                    cutlass.Boolean,
                )
                sfa_rows = None
                if cutlass.const_expr(self.gather_a):
                    # Lane l loads the scales of rows 4 l .. 4 l + 3, from their source rows.
                    # A padding row's index may be -1; any valid row serves for it.
                    sfa_rows = cute.make_rmem_tensor((4,), cutlass.Int32)
                    sfa_row_base = tile_info[0] * self.cta_tile_shape_mnk[0] + lane_idx * 4
                    for j in cutlass.range_constexpr(4):
                        src_row = mRowIndex[sfa_row_base + j]
                        if src_row < 0:
                            src_row = cutlass.Int32(0)
                        sfa_rows[j] = src_row

                # Peek (try_wait) SCALE buffer empty
                scale_producer_state.reset_count()
                peek_scale_empty_status = cutlass.Boolean(1)
                if scale_producer_state.count < k_tile_cnt:
                    peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                        scale_producer_state
                    )

                #
                # load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    #
                    # Slice to per mma tile index
                    #
                    tAsSFA_pipe = cute.filter_zeros(
                        tAsSFA[(None, None, None, scale_producer_state.index)]
                    )
                    tBsSFB_pipe = cute.filter_zeros(
                        tBsSFB[(None, None, None, scale_producer_state.index)]
                    )
                    tAgSFA_k = cute.filter_zeros(
                        tAgSFA_mkl[
                            (
                                None,
                                None,
                                None,
                                tile_info[0],
                                scale_producer_state.count,
                                0,
                            )
                        ]
                    )
                    tBgSFB_k = cute.filter_zeros(
                        tBgSFB_nkl[
                            (
                                None,
                                None,
                                None,
                                tile_info[1],
                                scale_producer_state.count,
                                tile_info[2],
                            )
                        ]
                    )

                    tAcSFA_compact = cute.filter_zeros(
                        cute.slice_(
                            tAcSFA,
                            (
                                None,
                                None,
                                None,
                                tile_info[0],
                                scale_producer_state.count,
                                0,
                            ),
                        )
                    )
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
                        )
                    )
                    for i in cutlass.range_constexpr(cute.size(tApSFA, mode=[1])):
                        tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                            tAcSFA_compact[(i)][0], mSFA_mkl.shape[0]
                        )
                    for i in cutlass.range_constexpr(cute.size(tBpSFB, mode=[1])):
                        tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                            tBcSFB_compact[(i)][0], mSFB_nkl.shape[0]
                        )

                    # Conditionally wait for Scale buffer empty
                    scale_pipeline.producer_acquire(scale_producer_state, peek_scale_empty_status)

                    # load scaleA/scaleB
                    if cutlass.const_expr(self.gather_a):
                        for j in cutlass.range_constexpr(4):
                            sfa_src = cute.make_tensor(
                                mSFX.iterator
                                + cute.crd2idx(
                                    (sfa_rows[j], scale_producer_state.count, 0), mSFX.layout
                                ),
                                cute.make_layout((1,)),
                            )
                            sfa_dst = cute.make_tensor(
                                sSFA.iterator
                                + (
                                    scale_producer_state.index * self.cta_tile_shape_mnk[0]
                                    + lane_idx * 4
                                    + j
                                ),
                                cute.make_layout((1,)),
                            )
                            cute.copy(atom_copy, sfa_src, sfa_dst)
                    else:
                        cute.copy(tiled_copy_sfa, tAgSFA_k, tAsSFA_pipe, pred=tApSFA)
                    cute.copy(tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe, pred=tBpSFB)

                    scale_pipeline.producer_commit(scale_producer_state)

                    # Peek (try_wait) Scale buffer empty
                    scale_producer_state.advance()
                    peek_scale_empty_status = cutlass.Boolean(1)
                    if scale_producer_state.count < k_tile_cnt:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                            scale_producer_state
                        )

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
            # Wait Scale buffer empty
            #
            scale_pipeline.producer_tail(scale_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
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
            cur_expert = self._tile_expert(work_tile, mTileExpert)
            while work_tile.is_valid_tile & (cur_expert < 0):
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                cur_expert = self._tile_expert(work_tile, mTileExpert)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_expert
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = 0
                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                #
                # Mma mainloop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                    #
                    # Wait for accumulator buffer empty
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)

                    #
                    # Reset the ACCUMULATE field for each tile
                    #
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
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
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                    #
                    # Async arrive accumulator buffer full(each kblock)
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)

                    # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                    acc_producer_state.advance()
                    if acc_producer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                                acc_producer_state
                            )

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
        # Specialized acc update warps
        #
        if warp_idx <= self.acc_update_warp_id[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_acc_update_warps)
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
            tCtAcc_final = cute.make_tensor(
                tCtAcc_base.iterator + self.tmem_final_offset, tCtAcc_base.layout
            )

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
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            cur_expert = self._tile_expert(work_tile, mTileExpert)
            while work_tile.is_valid_tile & (cur_expert < 0):
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                cur_expert = self._tile_expert(work_tile, mTileExpert)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_scale_stage
            )

            epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_expert
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                # initialize the final accumulator
                tTR_rAcc_final.fill(0.0)

                tTR_rSFA = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFA, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )
                tTR_rSFB = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFB, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )

                scale_consumer_state.reset_count()
                peek_scale_full_status = cutlass.Boolean(1)
                if scale_consumer_state.count < k_tile_cnt:
                    peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)

                acc_consumer_state.reset_count()
                peek_acc_full_status = cutlass.Boolean(1)
                if acc_consumer_state.count < k_tile_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Set tensor memory buffer for current tile
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]

                    #
                    # Wait for scale buffer full
                    #
                    scale_pipeline.consumer_wait(scale_consumer_state, peek_scale_full_status)

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

                    acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    #
                    # Update accumulator by scale factor in subtiles
                    #
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    for subtile_idx in cutlass.range(subtile_cnt):
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        #
                        # Update accumulator by scale factor
                        #
                        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                        tTR_rSFA_subtile = tTR_rSFA[(None, None, None, subtile_idx)]
                        tTR_rSFB_subtile = tTR_rSFB[(None, None, None, subtile_idx)]

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
                    if scale_consumer_state.count < k_tile_cnt:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(
                            scale_consumer_state
                        )
                    #
                    # Async arrive accumulator buffer empty
                    #
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                tRT_tAcc = tRT_tAcc_base[(None, None, None, None, None, 0)]
                tRT_tAcc = cute.group_modes(tRT_tAcc, 3, cute.rank(tRT_tAcc))

                #
                # Wait for epilogue buffer empty
                #
                epi_pipeline.producer_acquire(epi_producer_state)

                # copy the accumulator to tensor memory buffer
                cute.copy(tiled_copy_r2t, tTR_rAcc_final, tRT_tAcc)
                cute.arch.fence_view_async_tmem_store()

                #
                # Async arrive epilogue buffer full
                #
                epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

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
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1] and warp_idx >= self.epilog_warp_id[0]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_epilogue_warps)
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
            tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_final = cute.make_tensor(
                tCtAcc_base_.iterator + self.tmem_final_offset, tCtAcc_base_.layout
            )

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_final, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            # simt_atom = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            cur_expert = self._tile_expert(work_tile, mTileExpert)
            while work_tile.is_valid_tile & (cur_expert < 0):
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                cur_expert = self._tile_expert(work_tile, mTileExpert)

            epi_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

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

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_expert
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            num_prev_subtiles = cutlass.Int32(0)

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
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

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, epi_consumer_state.index)]

                #
                # Wait for accumulator buffer full
                #
                epi_pipeline.consumer_wait(epi_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for subtile_idx in cutlass.range(subtile_cnt):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Convert to C type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
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
                            bSG_gC[(None, subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                epi_pipeline.consumer_release(epi_consumer_state)
                epi_consumer_state.advance()

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

    def __init__(
        self,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        raster_along_m: bool = False,
        gather_a: bool = False,
    ):
        super().__init__(cutlass.Float32, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)
        self.raster_along_m = raster_along_m
        self.gather_a = gather_a

    @cute.jit
    def _tile_expert(self, work_tile, mTileExpert: cute.Tensor) -> cutlass.Int32:
        """The expert of the 128-row group that holds the work tile's CTA tile, or -1."""
        expert = cutlass.Int32(-1)
        if work_tile.is_valid_tile:
            expert = mTileExpert[
                (work_tile.tile_idx[0] * self.cta_tile_shape_mnk[0]) // GROUP_ROW_ALIGNMENT
            ]
        return expert

    def _compute_grid_grouped(
        self,
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Persistent grid over every (M tile, N tile) of ``c``.

        With ``raster_along_m=False`` consecutive work indices walk N first, so the CTAs that
        run at the same time share their A tiles in L2 and A is read from HBM about once.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            (*cluster_shape_mn, 1),
            swizzle_size=1,
            raster_along_m=self.raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def can_implement_grouped(
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        n: int,
        k: int,
        gather_a: bool = False,
    ) -> bool:
        """Whether a tile configuration can run the grouped GEMM.

        Every CTA of a cluster must work on one expert, so a cluster spans one 128-row group:
        the MMA tile is at most 128 rows and the cluster only grows along N. The gathered A
        load (one tile::gather4 of four rows per lane of the TMA warp) needs a 1-CTA MMA
        with 128-row tiles.
        """
        if gather_a and (use_2cta_instrs or mma_tiler_mn[0] != GROUP_ROW_ALIGNMENT):
            return False
        if not Sm100BlockwiseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            use_2cta_instrs, mma_tiler_mn, cluster_shape_mn
        ):
            return False
        if mma_tiler_mn[0] > GROUP_ROW_ALIGNMENT:
            return False
        if cluster_shape_mn[0] != (2 if use_2cta_instrs else 1):
            return False
        n_tile = mma_tiler_mn[1] * cluster_shape_mn[1]
        return n % n_tile == 0 and k % 128 == 0

    @cute.jit
    def wrapper_grouped(
        self,
        rows: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        num_experts: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        tile_expert_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Build the tensors from raw pointers and launch.

        ``c_tensor`` is (rows, N, 1) and stays a tensor so TVM FFI can find the stream.
        """
        sf_k = (k + 127) // 128
        sf_n = (n + 127) // 128
        a_tensor = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((rows, k, 1), order=(1, 0, 2))
        )
        b_tensor = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, num_experts), order=(1, 0, 2))
        )
        # [K / 128, rows] in memory: rows are contiguous.
        sfa_tensor = cute.make_tensor(
            a_sf_ptr, layout=cute.make_ordered_layout((rows, sf_k, 1), order=(0, 1, 2))
        )
        # [E, N / 128, K / 128] in memory: K blocks are contiguous.
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout((sf_n, sf_k, num_experts), order=(1, 0, 2)),
        )
        tile_expert = cute.make_tensor(
            tile_expert_ptr, layout=cute.make_layout((rows // GROUP_ROW_ALIGNMENT,))
        )
        self(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            tile_expert,
            tile_expert,  # row_index: unused without gather_a
            max_active_clusters,
            stream,
        )

    @cute.jit
    def wrapper_gathered(
        self,
        rows: cutlass.Int32,
        num_tokens: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        num_experts: cutlass.Int32,
        sfx_stride: cutlass.Int32,
        x_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        x_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        tile_expert_ptr: cute.Pointer,
        row_index_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """``wrapper_grouped`` with A and its scales read through ``row_index``."""
        sf_k = (k + 127) // 128
        sf_n = (n + 127) // 128
        x_tensor = cute.make_tensor(
            x_ptr, layout=cute.make_ordered_layout((num_tokens, k, 1), order=(1, 0, 2))
        )
        # [K / 128, T] in memory with row stride ``sfx_stride``: tokens are contiguous.
        sfx_tensor = cute.make_tensor(
            x_sf_ptr,
            layout=cute.make_layout((num_tokens, sf_k, 1), stride=(1, sfx_stride, 0)),
        )
        b_tensor = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, num_experts), order=(1, 0, 2))
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout((sf_n, sf_k, num_experts), order=(1, 0, 2)),
        )
        tile_expert = cute.make_tensor(
            tile_expert_ptr, layout=cute.make_layout((rows // GROUP_ROW_ALIGNMENT,))
        )
        row_index = cute.make_tensor(row_index_ptr, layout=cute.make_layout((rows,)))
        self(
            x_tensor,
            b_tensor,
            c_tensor,
            sfx_tensor,
            sfb_tensor,
            tile_expert,
            row_index,
            max_active_clusters,
            stream,
        )


GROUP_ROW_ALIGNMENT = 128

# (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, raster_along_m)
GroupedGemmTactic = Tuple[bool, Tuple[int, int], Tuple[int, int], bool]

# Best measured per (N, K) on B200 at 13312 tokens x top-8 over 288 experts.
DEFAULT_TACTICS: Dict[Tuple[int, int], GroupedGemmTactic] = {
    (1024, 4096): (False, (128, 128), (1, 2), False),
    (4096, 512): (False, (128, 128), (1, 2), False),
}
FALLBACK_TACTIC: GroupedGemmTactic = (False, (128, 128), (1, 1), False)
# The gathered A load keeps up with the MMA only when the A multicast of a wide cluster splits
# the row groups: (1, 8) covers FC1's eight N tiles.
DEFAULT_GATHERED_TACTICS: Dict[Tuple[int, int], GroupedGemmTactic] = {
    (1024, 4096): (False, (128, 128), (1, 8), False),
}

_compiled_kernels: Dict[Tuple, object] = {}


def _c_view(out: torch.Tensor) -> torch.Tensor:
    """``out`` [rows, N] as the kernel's (rows, N, 1) C tensor."""
    return out.view(1, *out.shape).permute(1, 2, 0)


def _compile(n: int, k: int, tactic: GroupedGemmTactic, out: torch.Tensor, gather_a: bool = False):
    use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, raster_along_m = tactic
    key = (n, k, tactic, gather_a)
    compiled = _compiled_kernels.get(key)
    if compiled is not None:
        return compiled
    if not Sm100GroupedBlockwiseGemmKernel.can_implement_grouped(
        use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, n, k, gather_a
    ):
        raise ValueError(f"grouped blockwise FP8 GEMM cannot run N={n}, K={k} with {tactic}")
    gemm = Sm100GroupedBlockwiseGemmKernel(
        use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, raster_along_m, gather_a
    )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    def ptr(dtype, align=16):
        return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)

    c_fake = cute.runtime.from_dlpack(_c_view(out)).mark_layout_dynamic(leading_dim=1)
    if gather_a:
        compiled = cute.compile(
            gemm.wrapper_gathered,
            out.shape[0],
            1,
            n,
            k,
            1,
            1,
            ptr(cutlass.Float8E4M3FN),
            ptr(cutlass.Float8E4M3FN),
            ptr(cutlass.Float32),
            ptr(cutlass.Float32),
            ptr(cutlass.Int32, 4),
            ptr(cutlass.Int32, 16),
            c_fake,
            max_active_clusters=max_active_clusters,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _compiled_kernels[key] = compiled
        return compiled
    compiled = cute.compile(
        gemm.wrapper_grouped,
        out.shape[0],
        n,
        k,
        1,
        ptr(cutlass.Float8E4M3FN),
        ptr(cutlass.Float8E4M3FN),
        ptr(cutlass.Float32),
        ptr(cutlass.Float32),
        ptr(cutlass.Int32, 4),
        c_fake,
        max_active_clusters=max_active_clusters,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _compiled_kernels[key] = compiled
    return compiled


def grouped_blockwise_fp8_gemm(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    tile_expert: torch.Tensor,
    out: torch.Tensor,
    tactic: Union[GroupedGemmTactic, None] = None,
) -> torch.Tensor:
    """Write ``out[tile] = (sfa * a[tile]) @ (sfb[e] * b[e]).T`` for every used 128-row tile.

    Args:
        a: ``[rows, K]`` FP8 E4M3, row-major. ``rows`` is a multiple of 128 and the rows are
            grouped by expert in segments that start at multiples of 128.
        sfa: ``[K / 128, rows]`` FP32, the 1 x 128 activation scales, K-block-major as
            ``fp8_quantize_1x128`` returns them.
        b: ``[E, N, K]`` FP8 E4M3, K-major per expert.
        sfb: ``[E, N / 128, K / 128]`` FP32, the 128 x 128 weight block scales.
        tile_expert: ``[rows / 128]`` Int32 on the GPU, the expert of each 128-row tile, or
            -1 for an unused tile. Unused tiles get no MMA and no store, so their ``out`` rows
            keep their old contents; rows of a used tile past its expert's segment hold
            whatever the padding input produces.
        out: ``[rows, N]`` BF16, written in place and returned.
        tactic: ``(use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, raster_along_m)``;
            defaults to the tuned choice for (N, K).

    N must be a multiple of 128 times the cluster's N size and K a multiple of 128. The call
    reads no device data on the host, so it is safe inside CUDA graph capture.
    """
    rows, k = a.shape
    num_experts, n, k_b = b.shape
    if k_b != k or rows % GROUP_ROW_ALIGNMENT != 0:
        raise ValueError(f"bad shapes: a {tuple(a.shape)}, b {tuple(b.shape)}")
    sfb_shape = (num_experts, n // 128, k // 128)
    if tuple(sfa.shape) != (k // 128, rows) or tuple(sfb.shape) != sfb_shape:
        raise ValueError(f"bad scale shapes: sfa {tuple(sfa.shape)}, sfb {tuple(sfb.shape)}")
    if tile_expert.numel() != rows // GROUP_ROW_ALIGNMENT or tile_expert.dtype != torch.int32:
        raise ValueError("tile_expert must be Int32 with rows / 128 entries")
    if tuple(out.shape) != (rows, n) or out.dtype != torch.bfloat16:
        raise ValueError(f"bad output: {tuple(out.shape)} {out.dtype}")
    if tactic is None:
        tactic = DEFAULT_TACTICS.get((n, k), FALLBACK_TACTIC)
    compiled = _compile(n, k, tactic, out)
    compiled(
        rows,
        n,
        k,
        num_experts,
        a.data_ptr(),
        b.data_ptr(),
        sfa.data_ptr(),
        sfb.data_ptr(),
        tile_expert.data_ptr(),
        _c_view(out),
    )
    return out


def grouped_blockwise_fp8_gemm_gathered(
    x: torch.Tensor,
    sfx: torch.Tensor,
    row_index: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    tile_expert: torch.Tensor,
    out: torch.Tensor,
    tactic: Union[GroupedGemmTactic, None] = None,
) -> torch.Tensor:
    """``grouped_blockwise_fp8_gemm`` with A read as ``x[row_index]``, without the copy.

    Row ``r`` of the grouped operand is ``x[row_index[r]]`` with scales
    ``sfx[:, row_index[r]]``. The TMA warp loads each A tile with ``tile::gather4`` (four
    source rows per instruction) and the scale warp copies the matching scales, so the
    result is bitwise the one of gathering ``x`` and ``sfx`` first.

    Args:
        x: ``[T, K]`` FP8 E4M3, the un-permuted activations, row-major.
        sfx: ``[K / 128, T]`` FP32, their 1 x 128 scales, K-block-major (tokens contiguous;
            the row stride may exceed ``T``).
        row_index: ``[rows]`` Int32, the source row in ``x`` of every grouped row. Padding
            rows may hold -1 (or any valid row); their output is don't-care.
        b, sfb, tile_expert, out: as in ``grouped_blockwise_fp8_gemm``.
        tactic: as in ``grouped_blockwise_fp8_gemm``; must be a 1-CTA 128-row tile.
    """
    num_tokens, k = x.shape
    num_experts, n, k_b = b.shape
    rows = out.shape[0]
    if k_b != k or rows % GROUP_ROW_ALIGNMENT != 0:
        raise ValueError(f"bad shapes: x {tuple(x.shape)}, b {tuple(b.shape)}, out {rows}")
    if tuple(sfx.shape) != (k // 128, num_tokens) or sfx.stride(1) != 1:
        raise ValueError(f"bad sfx: {tuple(sfx.shape)} strides {sfx.stride()}")
    if tuple(sfb.shape) != (num_experts, n // 128, k // 128):
        raise ValueError(f"bad sfb: {tuple(sfb.shape)}")
    if row_index.numel() != rows or row_index.dtype != torch.int32:
        raise ValueError("row_index must be Int32 with one entry per output row")
    if tile_expert.numel() != rows // GROUP_ROW_ALIGNMENT or tile_expert.dtype != torch.int32:
        raise ValueError("tile_expert must be Int32 with rows / 128 entries")
    if tuple(out.shape) != (rows, n) or out.dtype != torch.bfloat16:
        raise ValueError(f"bad output: {tuple(out.shape)} {out.dtype}")
    if tactic is None:
        tactic = DEFAULT_GATHERED_TACTICS.get((n, k), FALLBACK_TACTIC)
    compiled = _compile(n, k, tactic, out, gather_a=True)
    compiled(
        rows,
        num_tokens,
        n,
        k,
        num_experts,
        sfx.stride(0),
        x.data_ptr(),
        b.data_ptr(),
        sfx.data_ptr(),
        sfb.data_ptr(),
        tile_expert.data_ptr(),
        row_index.data_ptr(),
        _c_view(out),
    )
    return out
