# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501,E731,E741
"""Kimi K3's SM107 dense MXFP8 GEMM with an FP8 E4M3 output (the context kv_b expansion).

:class:`K3Sm107MXFP8Fp8OutGemmKernel` / :class:`K3Sm107MXFP8Fp8OutMixedClustersGemmKernel` are the
shared Rubin persistent dense block-scaled GEMM kernels
(``_torch/kernels/rubin/dense_blockscaled_gemm_persistent``) with an E4M3 C: the epilogue scales and
rounds the accumulator through BF16, then converts to E4M3 with a unit scale
(:func:`.mxfp8_fp8_out_epilogue.epilogue_tma_store_fp8`), so the result equals the BF16 GEMM followed
by a unit-scale E4M3 quantization. Only MXFP8 E4M3 operands without split-K take it. The non-mixed
kernel also takes an epilogue subtile N (``epi_tile_n``) that must divide the CTA tile N.

Overridden from the shared kernel (the rest is inherited):

* ``kernel_impl``: the shared device code with the FP8 output's alpha type (BF16) and epilogue store;
* ``_setup_attributes``: the shared setup with the ``epi_tile_n`` override of the epilogue subtile;
* ``can_implement``: the shared checks with an E4M3 C (MXFP8 operands, split-K 1, 16-byte C rows).

``tests/.../ops/gemm/test_k3_mxfp8_gemm_fp8_out_source.py`` checks that the two copied methods are
the shared ones apart from those changes.
"""

from typing import Literal, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch.kernels.rubin.dense_blockscaled_gemm_persistent import (
    Sm107BlockScaledPersistentDenseGemmKernel,
    Sm107BlockScaledPersistentDenseGemmMixedClustersKernel,
)

from .mxfp8_fp8_out_epilogue import epilogue_tma_store_fp8


def _fp8_out_operands_ok(a_dtype, b_dtype, sf_dtype, sf_vec_size, split_k) -> bool:
    """Whether an E4M3 C is implemented for these operands: MXFP8 E4M3 with UE8M0 K32 scales, no
    split-K (split-K reduces in C)."""
    return (
        a_dtype == cutlass.Float8E4M3FN
        and b_dtype == cutlass.Float8E4M3FN
        and sf_dtype == cutlass.Float8E8M0FNU
        and sf_vec_size == 32
        and split_k == 1
    )


def _fp8_out_c_aligned(mnkl, a_dtype, b_dtype, a_major, b_major, c_major) -> bool:
    """The shared alignment rule (16-byte contiguous rows) with an E4M3 C."""
    m, n, k, l = mnkl
    try:
        Sm107BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, b_dtype, cutlass.Float8E4M3FN, a_major, b_major, c_major
        )
    except testing.CantImplementError:
        return False
    return True


class K3Sm107MXFP8Fp8OutGemmKernel(Sm107BlockScaledPersistentDenseGemmKernel):
    """The shared SM107 persistent dense block-scaled GEMM with an E4M3 C (see the module docstring).

    ``epi_tile_n``: None keeps the default epilogue subtile (``compute_epilogue_tile_shape``);
    otherwise the epilogue subtile N, which must divide the CTA tile N (CTA tile M >= 128).
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
        epi_tile_n: Union[int, None] = None,
    ):
        super().__init__(
            sf_vec_size,
            mma_inst_shape,
            mma_tiler,
            cluster_shape_mn,
            prefetch_dist,
            swizzle_size=swizzle_size,
            raster_order=raster_order,
            scheduler_type=scheduler_type,
            split_k=split_k,
        )
        # With a K-short mainloop, wider epilogue subtiles cut the per-tile TMEM-load /
        # SMEM-store / TMA-store iterations.
        self.epi_tile_n_override = epi_tile_n

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
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm107_utils.compute_epilogue_tile_shape(
            tiled_mma.op,
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        if self.epi_tile_n_override is not None:
            cta_tile_n = self.cta_tile_shape_mnk[1]
            if self.cta_tile_shape_mnk[0] < 128 or cta_tile_n % self.epi_tile_n_override != 0:
                raise ValueError(
                    f"Epilogue subtile N={self.epi_tile_n_override} needs a CTA tile with M >= 128 "
                    f"and N divisible by it, got {self.cta_tile_shape_mnk[:2]}."
                )
            self.epi_tile = (
                cute.make_layout(cute.size(self.epi_tile[0])),
                cute.make_layout(self.epi_tile_n_override),
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

        print(f"[DSL INFO] Prefetch distance: {self.prefetch_dist}")

        # Check if prefetch is enabled (prefetch_dist > 0)
        self.prefetch_enabled = self.prefetch_dist > 0

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
        # FP8 output scales and rounds through BF16 (fp8_output_epilogue).
        alpha_dtype = self.c_dtype
        if cutlass.const_expr(self.c_dtype == cutlass.Float8E4M3FN):
            alpha_dtype = cutlass.BFloat16
        alpha_value = alpha[0].to(alpha_dtype)

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
                if cutlass.const_expr(self.split_k > 1):
                    mma_tile_coord_mnl = (
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        mma_tile_coord_mnl[2] // self.split_k,
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

                epilogue_store = utils.gemm.sm100.epilogue_tma_store
                if cutlass.const_expr(self.c_dtype == cutlass.Float8E4M3FN):
                    epilogue_store = epilogue_tma_store_fp8
                acc_consumer_state = epilogue_store(
                    self,
                    tidx,
                    warp_idx,
                    tma_atom_c,
                    tCtAcc,
                    sC,
                    tCgC,
                    epi_tile,
                    epi_tiles_executed,
                    lambda x: epilogue_op(alpha_value * x),
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    c_pipeline,
                )

            c_pipeline.producer_tail()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)

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
        """The shared ``can_implement``; an E4M3 C also needs MXFP8 operands, split-K 1 and 16-byte C
        rows (the shared checks are independent of the C type otherwise)."""
        if c_dtype == cutlass.Float8E4M3FN:
            if not _fp8_out_operands_ok(a_dtype, b_dtype, sf_dtype, sf_vec_size, split_k):
                return False
            if not _fp8_out_c_aligned(mnkl, a_dtype, b_dtype, a_major, b_major, c_major):
                return False
            c_dtype = cutlass.BFloat16
        return Sm107BlockScaledPersistentDenseGemmKernel.can_implement(
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
            cluster_shape_mn,
            split_k,
        )


class K3Sm107MXFP8Fp8OutMixedClustersGemmKernel(
    Sm107BlockScaledPersistentDenseGemmMixedClustersKernel
):
    """The shared mixed-cluster SM107 dense block-scaled GEMM with the E4M3 C device code of
    :class:`K3Sm107MXFP8Fp8OutGemmKernel` (no epilogue subtile override)."""

    kernel_impl = K3Sm107MXFP8Fp8OutGemmKernel.__dict__["kernel_impl"]

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
        """The shared mixed-cluster ``can_implement`` with the E4M3 C rules of
        :meth:`K3Sm107MXFP8Fp8OutGemmKernel.can_implement`."""
        if c_dtype == cutlass.Float8E4M3FN:
            if not _fp8_out_operands_ok(a_dtype, b_dtype, sf_dtype, sf_vec_size, 1):
                return False
            if not _fp8_out_c_aligned(mnkl, a_dtype, b_dtype, a_major, b_major, c_major):
                return False
            c_dtype = cutlass.BFloat16
        return Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.can_implement(
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
            fallback_cluster_shape_mn,
        )
