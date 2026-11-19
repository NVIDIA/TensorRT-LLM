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

from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync

LINEAR = 0
SWIGLU = 1
QKV = 2
PACKED_SWIGLU = 3
FP4_SHIFT_BITS = 2


def _pack_pair(high, low):
    dst_type = ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type)
    return cutlass.Int8(
        nvvm.convert_f32x2_to_f4x2(
            cutlass.Float32(high).ir_value(), cutlass.Float32(low).ir_value(), dst_type
        )
    )


def make_sm120_blockscaled_mma_op(a_dtype, b_dtype, acc_dtype, sf_dtype, sf_vec_size):
    if (
        a_dtype == cutlass.Float8E4M3FN
        and b_dtype == cutlass.Float8E4M3FN
        and acc_dtype == cutlass.Float32
        and sf_dtype == cutlass.Float8E8M0FNU
        and sf_vec_size == 32
    ):
        return cute.nvgpu.warp.MmaMXF8Op(a_dtype, acc_dtype, sf_dtype), True
    if (
        a_dtype != cutlass.Float4E2M1FN
        or b_dtype != cutlass.Float4E2M1FN
        or acc_dtype != cutlass.Float32
        or sf_dtype != cutlass.Float8E4M3FN
        or sf_vec_size != 16
    ):
        raise ValueError("Flux.3 SM120 kernel requires NVFP4 E2M1 or FP8 E4M3 operands")
    return cute.nvgpu.warp.MmaMXF4NVF4Op(a_dtype, acc_dtype, sf_dtype), False


def make_ldmatrix_atom(operand_dtype, transpose, num_matrices=4, mixed_mode=False):
    if mixed_mode:
        raise ValueError("Mixed operand dtypes are not supported")
    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices),
        operand_dtype,
    )


class Flux3Sm120Nvfp4Kernel:
    def __init__(
        self,
        acc_dtype,
        sf_vec_size,
        tile_shape_mnk,
        epi_tile,
        epilogue=LINEAR,
        columns=128,
        architecture="sm_120",
    ):
        self.columns = columns
        self.epilogue = epilogue
        self.scratch_dtype = cutlass.BFloat16 if epilogue == QKV else cutlass.Float32
        self.scratch_shape = (128, 128 if epilogue == QKV else 64)
        self.acc_dtype = acc_dtype
        self.sf_vec_size = sf_vec_size
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.epi_tile = tuple(epi_tile)
        self.tiled_mma = None

        self.occupancy = 1
        self.num_mma_warps = 8
        self.tma_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes(architecture)

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_op, use_mxf8f6f4 = make_sm120_blockscaled_mma_op(
            self.a_dtype,
            self.b_dtype,
            self.acc_dtype,
            self.sf_dtype,
            self.sf_vec_size,
        )
        self.mixed_mode = self.a_dtype != self.b_dtype
        # a_fp4_in_mixed / b_fp4_in_mixed: this side carries an FP4 operand in the
        # mixed FP4 x FP8 mode, so SMEM/TMA see Int8 storage and the mma.sync
        # consumer needs the LDSM b4x16_p64 unpack + register `<< FP4_SHIFT_BITS`.
        a_fp4_in_mixed = self.mixed_mode and self.a_dtype.width < 8
        b_fp4_in_mixed = self.mixed_mode and self.b_dtype.width < 8
        self.smem_alloc_a_dtype = cutlass.Int8 if a_fp4_in_mixed else self.a_dtype
        self.smem_alloc_b_dtype = cutlass.Int8 if b_fp4_in_mixed else self.b_dtype
        # `internal_type` for `_make_tma_atoms_and_tensors` — None when the dtype
        # already matches (TMA sees the native dtype), Int8 when we recast for FP4.
        self.tma_internal_a_dtype = cutlass.Int8 if a_fp4_in_mixed else None
        self.tma_internal_b_dtype = cutlass.Int8 if b_fp4_in_mixed else None
        atom_shape = (4, 2, 1)
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, use_mxf8f6f4
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute the smem size of SFA/SFB
        sfa_smem_layout_per_stage = blockscaled_utils.sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        sfb_smem_layout_per_stage = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.smem_alloc_a_dtype,
            self.smem_alloc_b_dtype,
            self.sf_dtype,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        self.ab_stage = 2 if self.a_dtype == cutlass.Float8E4M3FN else 3
        self.epi_stage = 1

        assert self.epi_stage > 0, (
            "epi_stage <= 0, no enough shared memory. This case will be skipped."
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.smem_alloc_a_dtype,
            self.a_layout,
            self.smem_alloc_b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        output: cute.Tensor,
        output_scale: cute.Tensor,
        inverse_output_scale: cute.Pointer,
        global_scale: cute.Pointer,
        activation_scale: cute.Pointer,
        weight_scale: cute.Pointer,
        q_norm: cute.Pointer,
        k_norm: cute.Pointer,
        cos_sin: cute.Pointer,
        positions: cute.Pointer,
        qkv_scale: cute.Pointer,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
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
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
            internal_type=self.tma_internal_a_dtype,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
            internal_type=self.tma_internal_b_dtype,
        )

        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
            internal_type=cutlass.Int16,
        )

        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
            internal_type=cutlass.Int16,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sScratch: cute.struct.Align[
                cute.struct.MemRange[self.scratch_dtype, cute.size(self.scratch_shape)],
                16,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c,
            output,
            output_scale,
            inverse_output_scale,
            global_scale,
            activation_scale,
            weight_scale,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            qkv_scale,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
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
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        output: cute.Tensor,
        output_scale: cute.Tensor,
        inverse_output_scale: cute.Pointer,
        global_scale: cute.Pointer,
        activation_scale: cute.Pointer,
        weight_scale: cute.Pointer,
        q_norm: cute.Pointer,
        k_norm: cute.Pointer,
        cos_sin: cute.Pointer,
        positions: cute.Pointer,
        qkv_scale: cute.Pointer,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
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

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        # bidx, bidy, bidz = cute.arch.block_idx()
        # bdimx, bdimy, bdimz = cute.arch.grid_dim()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                cpasync.prefetch_descriptor(tma_atom_sfa)
                cpasync.prefetch_descriptor(tma_atom_sfb)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )
        if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
            tma_copy_bytes += cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            tma_copy_bytes += cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sScratch = storage.sScratch.get_tensor(
            cute.make_layout(self.scratch_shape, stride=(self.scratch_shape[1], 1))
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (tM, tK, loopM, loopK, loopL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (tN, tK, loopN, loopK, loopL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #  Make frangments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
        tCrSFB = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)
        # Keep residual K modes nested to match the C++ SM120 block-scaled mainloop.
        tCrSFA = cute.group_modes(tCrSFA, 2, cute.rank(tCrSFA))
        tCrSFB = cute.group_modes(tCrSFB, 2, cute.rank(tCrSFB))
        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
            tCrSFA.fill(cutlass.Float8E8M0FNU(1.0))
            tCrSFB.fill(cutlass.Float8E8M0FNU(1.0))

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # cluster wait for barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Create the tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Create the pipeline states for producer and consumer
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # ///////////////////////////////////////////////////////////////////////////////
            # Copy Atom A/B retiling for TMA load A/B
            # ///////////////////////////////////////////////////////////////////////////////
            atom_copy_ldmatrix_A = make_ldmatrix_atom(
                self.a_dtype,
                transpose=self.a_layout.is_m_major_a(),
                num_matrices=4,
                mixed_mode=self.mixed_mode,
            )
            atom_copy_ldmatrix_B = make_ldmatrix_atom(
                self.b_dtype,
                transpose=self.b_layout.is_n_major_b(),
                num_matrices=4,
                mixed_mode=self.mixed_mode,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            atom_copy_ldmatrix_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.sf_dtype,
            )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                sm120_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                sm120_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view = thr_copy_ldmatrix_SFA.partition_S(sSFA)
            tCrSFA_copy_view = thr_copy_ldmatrix_SFA.retile(tCrSFA)
            tCsSFB_copy_view = thr_copy_ldmatrix_SFB.partition_S(sSFB)
            tCrSFB_copy_view = thr_copy_ldmatrix_SFB.retile(tCrSFB)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                # Clear the accumulator
                accumulators.fill(0.0)

                # /////////////////////////////////////////////////////////////////////////////
                #  Pipelined MAINLOOP
                # /////////////////////////////////////////////////////////////////////////////

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                #  Wait for TMA copies to complete
                mainloop_pipeline.consumer_wait(mainloop_consumer_state, peek_ab_full_status)
                # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = tCsSFA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFB_p = tCsSFB_copy_view[None, None, None, mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)

                if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                    cute.copy(
                        smem_tiled_copy_SFA,
                        tCsSFA_p_filtered[None, None, 0],
                        tCrSFA_copy_view_filtered[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_SFB,
                        tCsSFB_p_filtered[None, None, 0],
                        tCrSFB_copy_view_filtered[None, None, 0],
                    )

                for k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    # unroll the loop
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                            tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                            tCsSFA_p = tCsSFA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFB_p = tCsSFB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        # Mixed FP4 x FP8 register-side bit shift before mma.sync
                        # to move the FP4 nibble (in low half of each byte after
                        # b4x16_p64 ldmatrix) into the middle as mxf8f6f4 expects.
                        if cutlass.const_expr(self.mixed_mode and self.a_dtype.width < 8):
                            a_view = cute.recast_tensor(tCrA[None, None, k_block_idx], cutlass.Int8)
                            for _i in cutlass.range_constexpr(cute.size(a_view)):
                                a_view[_i] = cutlass.Int8(a_view[_i] << FP4_SHIFT_BITS)
                        if cutlass.const_expr(self.mixed_mode and self.b_dtype.width < 8):
                            b_view = cute.recast_tensor(tCrB[None, None, k_block_idx], cutlass.Int8)
                            for _i in cutlass.range_constexpr(cute.size(b_view)):
                                b_view[_i] = cutlass.Int8(b_view[_i] << FP4_SHIFT_BITS)
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            [
                                tCrA[None, None, k_block_idx],
                                tCrSFA[None, None, k_block_idx],
                            ],
                            [
                                tCrB[None, None, k_block_idx],
                                tCrSFB[None, None, k_block_idx],
                            ],
                            accumulators,
                        )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )

                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)
                        if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                            cute.copy(
                                smem_tiled_copy_SFA,
                                tCsSFA_p_filtered[None, None, k_block_next],
                                tCrSFA_copy_view_filtered[None, None, k_block_next],
                            )
                            cute.copy(
                                smem_tiled_copy_SFB,
                                tCsSFB_p_filtered[None, None, k_block_next],
                                tCrSFB_copy_view_filtered[None, None, k_block_next],
                            )

                        # end of for loop
                # end of for loop
                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1

                    if k_block_idx == num_k_blocks - 1:
                        cute.arch.fence_proxy("async.shared", space="cta")
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)
                        if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                            cute.copy(
                                smem_tiled_copy_SFA,
                                tCsSFA_p_filtered[None, None, k_block_next],
                                tCrSFA_copy_view_filtered[None, None, k_block_next],
                            )
                            cute.copy(
                                smem_tiled_copy_SFB,
                                tCsSFB_p_filtered[None, None, k_block_next],
                                tCrSFB_copy_view_filtered[None, None, k_block_next],
                            )
                    # Mixed FP4 x FP8 register-side bit shift before mma.sync (hoisted tail).
                    if cutlass.const_expr(self.mixed_mode and self.a_dtype.width < 8):
                        a_view_h = cute.recast_tensor(tCrA[None, None, k_block_idx], cutlass.Int8)
                        for _i in cutlass.range_constexpr(cute.size(a_view_h)):
                            a_view_h[_i] = cutlass.Int8(a_view_h[_i] << FP4_SHIFT_BITS)
                    if cutlass.const_expr(self.mixed_mode and self.b_dtype.width < 8):
                        b_view_h = cute.recast_tensor(tCrB[None, None, k_block_idx], cutlass.Int8)
                        for _i in cutlass.range_constexpr(cute.size(b_view_h)):
                            b_view_h[_i] = cutlass.Int8(b_view_h[_i] << FP4_SHIFT_BITS)
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        [
                            tCrA[None, None, k_block_idx],
                            tCrSFA[None, None, k_block_idx],
                        ],
                        [
                            tCrB[None, None, k_block_idx],
                            tCrSFB[None, None, k_block_idx],
                        ],
                        accumulators,
                    )

                # /////////////////////////////////////////////////////////////////////////////
                #  FUSED EPILOG
                # /////////////////////////////////////////////////////////////////////////////
                thread_coords = thr_mma.partition_C(cute.make_identity_tensor((128, 128)))
                tile_m = tile_coord_mnl[0]
                tile_n = tile_coord_mnl[1]
                if cutlass.const_expr(self.epilogue == LINEAR):
                    factor = global_scale.load()
                    for value_idx in cutlass.range_constexpr(cute.size(accumulators)):
                        coord = thread_coords[value_idx]
                        row = tile_m * 128 + coord[0]
                        col = tile_n * 128 + coord[1]
                        value = accumulators[value_idx] * factor
                        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                            value *= (activation_scale + row).load() * (weight_scale + col).load()
                        output[(row, col, 0)] = cutlass.BFloat16(value)
                elif cutlass.const_expr(self.epilogue == SWIGLU):
                    factor = global_scale.load()
                    for pair_idx in cutlass.range_constexpr(cute.size(accumulators) // 2):
                        coord = thread_coords[2 * pair_idx]
                        row = tile_m * 128 + coord[0]
                        col = tile_n * 64 + coord[1] // 2
                        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                            row_scale = (activation_scale + row).load()
                            weight_col = tile_n * 128 + coord[1]
                            gate = (
                                accumulators[2 * pair_idx]
                                * row_scale
                                * (weight_scale + weight_col).load()
                            )
                            act = (
                                accumulators[2 * pair_idx + 1]
                                * row_scale
                                * (weight_scale + weight_col + 1).load()
                            )
                        else:
                            gate = accumulators[2 * pair_idx] * factor
                            act = accumulators[2 * pair_idx + 1] * factor
                        output[(row, col, 0)] = cutlass.BFloat16(
                            gate * act / (1.0 + cute.math.exp(-act, fastmath=True))
                        )
                elif cutlass.const_expr(self.epilogue == QKV):
                    heads = self.columns // 384
                    segment = tile_n // heads
                    factor = (qkv_scale + segment).load()
                    if segment < 2:
                        for value_idx in cutlass.range_constexpr(cute.size(accumulators)):
                            coord = thread_coords[value_idx]
                            row = tile_m * 128 + coord[0]
                            col = tile_n * 128 + coord[1]
                            value = accumulators[value_idx] * factor
                            if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                                value *= (activation_scale + row).load() * (
                                    weight_scale + col
                                ).load()
                            sScratch[(coord[0], coord[1])] = cutlass.BFloat16(value)
                        self.epilog_sync_barrier.arrive_and_wait()
                        lane = tidx % 32
                        norm = q_norm if segment == 0 else k_norm
                        for row_iteration in cutlass.range_constexpr(16):
                            row = row_iteration * 8 + tidx // 32
                            global_row = tile_m * 128 + row
                            sum_squares = cutlass.Float32(0.0)
                            for group in cutlass.range_constexpr(4):
                                value = sScratch[(row, group * 32 + lane)].to(cutlass.Float32)
                                sum_squares += value * value
                            for delta in (16, 8, 4, 2, 1):
                                sum_squares += cute.arch.shuffle_sync_op(
                                    sum_squares, delta, kind=nvvm.ShflKind.bfly
                                )
                            inv_rms = cute.rsqrt(
                                sum_squares / cutlass.Float32(128) + cutlass.Float32(1.0e-6)
                            )
                            position = (positions + global_row).load()
                            for pair_group in cutlass.range_constexpr(2):
                                pair = pair_group * 32 + lane
                                even_col = 2 * pair
                                even = (
                                    sScratch[(row, even_col)].to(cutlass.Float32)
                                    * inv_rms
                                    * (norm + even_col).load().to(cutlass.Float32)
                                )
                                odd = (
                                    sScratch[(row, even_col + 1)].to(cutlass.Float32)
                                    * inv_rms
                                    * (norm + even_col + 1).load().to(cutlass.Float32)
                                )
                                cosine = (cos_sin + position * 128 + pair).load()
                                sine = (cos_sin + position * 128 + 64 + pair).load()
                                output[(global_row, tile_n * 128 + even_col, 0)] = cutlass.BFloat16(
                                    even * cosine - odd * sine
                                )
                                output[(global_row, tile_n * 128 + even_col + 1, 0)] = (
                                    cutlass.BFloat16(odd * cosine + even * sine)
                                )
                        self.epilog_sync_barrier.arrive_and_wait()
                    else:
                        for value_idx in cutlass.range_constexpr(cute.size(accumulators)):
                            coord = thread_coords[value_idx]
                            row = tile_m * 128 + coord[0]
                            col = tile_n * 128 + coord[1]
                            value = accumulators[value_idx] * factor
                            if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                                value *= (activation_scale + row).load() * (
                                    weight_scale + col
                                ).load()
                            output[(row, col, 0)] = cutlass.BFloat16(value)
                elif cutlass.const_expr(self.epilogue == PACKED_SWIGLU):
                    factor = global_scale.load()
                    for pair_idx in cutlass.range_constexpr(cute.size(accumulators) // 2):
                        gate = accumulators[2 * pair_idx] * factor
                        act = accumulators[2 * pair_idx + 1] * factor
                        coord = thread_coords[2 * pair_idx]
                        sScratch[(coord[0], coord[1] // 2)] = (
                            gate * act / (1.0 + cute.math.exp(-act, fastmath=True))
                        )
                    self.epilog_sync_barrier.arrive_and_wait()
                    for block_iteration in cutlass.range_constexpr(2):
                        block_index = tidx + block_iteration * 256
                        row = block_index // 4
                        scale_column = block_index % 4
                        first_column = scale_column * 16
                        maximum = cutlass.Float32(0.0)
                        for element_idx in cutlass.range_constexpr(16):
                            maximum = cute.math.max(
                                maximum, cute.math.abs(sScratch[(row, first_column + element_idx)])
                            )
                        inverse = inverse_output_scale.load()
                        encoded_scale = cutlass.Float8E4M3FN(
                            maximum * inverse / cutlass.Float32(6.0)
                        )
                        multiplier = inverse / cute.math.max(
                            encoded_scale.to(cutlass.Float32), cutlass.Float32(1.0e-12)
                        )
                        global_row = tile_m * 128 + row
                        global_col = tile_n * 64 + first_column
                        for pair_idx in cutlass.range_constexpr(8):
                            low = sScratch[(row, first_column + 2 * pair_idx)] * multiplier
                            high = sScratch[(row, first_column + 2 * pair_idx + 1)] * multiplier
                            output[(global_row, global_col // 2 + pair_idx, 0)] = _pack_pair(
                                high, low
                            )
                        scale_col = tile_n * 4 + scale_column
                        output_columns = self.columns // 2
                        scale_index = (
                            scale_col % 4
                            + scale_col // 4 * 512
                            + global_row % 32 * 16
                            + global_row % 128 // 32 * 4
                            + global_row // 128 * 128 * (output_columns // 16)
                        )
                        output_scale[scale_index] = encoded_scale.bitcast(cutlass.Int8)
                    self.epilog_sync_barrier.arrive_and_wait()

                # Advance to the next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                # End of for k_tile loop
            # End of while loop
        # End of MMA warp group
        # Start of DMA warp group
        elif warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                tAgSFA_mkl = tAgSFA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgSFB_nkl = tBgSFB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    # /////////////////////////////////////////////////////////////////////////////
                    #  Wait for A/B buffers to be empty before loading into them
                    #  Also sets the transaction barrier for the A/B buffers
                    # /////////////////////////////////////////////////////////////////////////////
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    # /////////////////////////////////////////////////////////////////////////////
                    #  Slice to global/shared memref to current k_tile
                    # /////////////////////////////////////////////////////////////////////////////
                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    tAgSFA_k = tAgSFA_mkl[(None, mainloop_producer_state.count)]
                    tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                    tBgSFB_k = tBgSFB_nkl[(None, mainloop_producer_state.count)]
                    tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

                    # /////////////////////////////////////////////////////////////////////////////
                    #  TMA load A/B
                    # /////////////////////////////////////////////////////////////////////////////
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                    )
                    if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_k,
                            tAsSFA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_k,
                            tBsSFB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                    # Mainloop pipeline's producer commit is a NOP
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # end of while loop

            # Wait A/B buffer empty
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        sf_dtype: type[cutlass.Numeric],
        sfa_smem_layout: cute.Layout,
        sfb_smem_layout: cute.Layout,
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (tile_shape_mnk[0] // epi_tile[0])
        epi_stage = min(epi_stage_max, 4)
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        # A/B shared memory size
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        )
        # SFA/SFB shared memory size
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
        # Mbar helpers size
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy - mbar_helpers_bytes - epi_bytes
        ) // (ab_bytes_per_stage + sf_bytes_per_stage)
        return ab_stage, epi_stage

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
        sf_vec_size: int,
        tiled_mma: cute.TiledMma,
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

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

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
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
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
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )

        sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
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

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

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
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
        internal_type: Optional[Type[cutlass.Numeric]] = None,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

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
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type,
        )

        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        batch_count: int,
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
        :param batch_count: The number of columns in the C tensor
        :type batch_count: int
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
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, batch_count))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, batch_count))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, batch_count))
        ):
            is_valid = False
        return is_valid
