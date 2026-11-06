# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 context-step MoE shared expert: gate|up MXF8 GEMM with the SwiGLU in its epilogue.

The plain shared MXF8 path (``self_attn_prefill_projections_mxf8`` op ``shared_up_mxf8``)
writes the gate|up product as BF16 ``[T, 1024]`` and a second launch
(``moe_sigmoid_topk_swiglu_rows.swiglu``) turns it into the BF16 ``[T, 512]`` SwiGLU the
down projection reads: 13.8 us per MoE layer at T ~ 7.3k, two thirds of it exposed on a
served context step.  This module runs the same block-scaled GEMM and computes the SwiGLU
in the epilogue instead, writing ``[T, 512]`` directly.

Bytes: every output is exactly that pair's.  The accumulator is converted to BF16
as the served epilogue converts it (round to nearest), then ``s = silu(g)`` and
``bf16(s * u)`` in FP32: ``s`` is the rows SwiGLU's ``g / (1 + exp(-g))`` (same precise
exp), with the IEEE divide's result produced branch-free (``silu_nr``: rcp.approx + one
FMA residual correction + three exact special cases), proven bit-identical for all 65536
BF16 ``g``.  The GEMM mainloop is the CuTe DSL example's, unchanged (same K order;
the 256x128 tile serves up to ``SMALL_MAX_ROWS``, the bytes do not depend on the tile);
only the weight's rows are permuted (``interleave_gate_up``): MXF8 block scales run along
K, so a row permutation permutes whole quantized rows and leaves every product and partial
sum as it was.

Speed (1 B300, CUDA graph, producer-written operand): T = 7310 37.1 us vs 46.9
for the GEMM + rows SwiGLU (GEMM alone 33.0); T = 2048 12.2 vs 21.8.

Layout contract: the epilogue subtile is 128 x 32 (``EPI_N``; BF16 output, no source C)
and the 32 dp x 32 bit TMEM load gives each epilogue thread one row's 32 consecutive
accumulator columns (the C coordinate tuple is (n, m), ``CRD_MN``).  The weight rows are
interleaved in blocks of ``HALF`` = 16 -- gate
rows 16b..16b+15, then up rows 16b..16b+15, b = 0..31 -- so a thread's subtile holds the
16 gate and the 16 matching up values of output columns 16b..16b+15 (b = subtile column /
32).  It stores them with two 16-byte stores to ``out[row, 16b:16b+16]`` (rows >= T are
skipped); there is no C staging in SMEM and no TMA store.  Everything before the epilogue
is the CuTe DSL example's ``Sm100BlockScaledPersistentDenseGemmKernel.kernel`` verbatim (from
``dense_blockscaled_gemm_persistent.py``).

Pattern: glm-ctx-shared-up-swiglu-mxf8
Device: SM100+ (Blackwell)
Shape: runtime T; A e4m3 [T, 6144] + blocked e8m0 SFA, B e4m3 [1024, 6144] (interleaved),
    out BF16 [T, 512]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from ..common import self_attn_prefill_act_quant_mxf8 as _quant
from ..common.cuda_stream import raw_stream as _raw_stream
from . import self_attn_prefill_projections_mxf8 as _proj

_bs = _proj._bs  # the CuTe DSL dense block-scaled persistent GEMM example module
cpasync = _bs.cpasync
tcgen05 = _bs.tcgen05
pipeline = _bs.pipeline
pipeline_init_arrive = _bs.pipeline_init_arrive
pipeline_init_wait = _bs.pipeline_init_wait
blockscaled_utils = _bs.blockscaled_utils
StaticPersistentTileScheduler = _bs.StaticPersistentTileScheduler
PersistentTileSchedulerParams = _bs.PersistentTileSchedulerParams

OP = "shared_up_mxf8"  # the plain op whose tiles, ABI and weight this module shares
GATE_UP = 1_024  # this rank's gate|up columns (2 x 512)
OUT = GATE_UP // 2  # SwiGLU width
HIDDEN = 6_144
EPI_N = 32  # epilogue subtile width (BF16 out, no source C: 4096 elements / 128 rows)
HALF = EPI_N // 2  # gate (and up) columns per subtile = outputs per thread per subtile


_INF = float("inf")


@cute.jit
def silu_rn(g: cutlass.Float32) -> cutlass.Float32:
    """The rows SwiGLU's ``silu(g)``: ``g / (1 + exp(-g))``, precise exp and IEEE divide."""
    return g / (cutlass.Float32(1.0) + cute.math.exp(-g))


@cute.jit
def silu_nr(g: cutlass.Float32) -> cutlass.Float32:
    """``silu_rn`` without the IEEE divide's slow-path branch: the same ``d = 1 + exp(-g)``,
    then ``q = g * rcp.approx(d)`` and one FMA residual correction, which rounds the
    quotient correctly for every finite ``d``; three operands are taken exactly instead:
    ``d = +inf`` (``exp`` overflow, g < -88.7) is ``g * 0`` (= ``g / inf``), ``g = +inf``
    is ``g`` (= ``inf / 1``), ``g = +-0`` is ``g`` (= ``+-0 / 2``; the residual FMA
    would turn -0 into +0).  ``s`` is a function of the 16 BF16 bits of ``g`` alone, so
    ``check_division`` proves it equal to ``silu_rn`` for all 65536 inputs (measured: the
    correction alone differed only at g = -0).  The precise divide's per-element branch
    keeps the 16 quotients of a subtile from interleaving: 46.9 -> 56.0 us with it,
    37.1 us without, vs 33.0 us for the GEMM alone (T = 7310)."""
    d = cutlass.Float32(1.0) + cute.math.exp(-g)
    r = cute.math.rcp(d, approx=True)
    q0 = g * r
    rem = cute.math.fma(-d, q0, g)
    q = cute.math.fma(rem, r, q0)
    q = cutlass.Float32(cutlass.select_(d == cutlass.Float32(_INF), g * cutlass.Float32(0.0), q))
    q = cutlass.Float32(cutlass.select_(g == cutlass.Float32(_INF), g, q))
    q = cutlass.Float32(cutlass.select_(g == cutlass.Float32(0.0), g, q))
    return q


class GlmMxf8SwigluGemm(_proj.GlmMxf8Gemm):
    """``GlmMxf8Gemm`` whose epilogue writes ``SwiGLU(C)`` as BF16 ``[M, N / 2]``.

    ``__call__(a, b, sfa, sfb, out, m, n, k, max_active_clusters, stream)``: ``out`` is the
    ``[M, N / 2]`` result; the base class still builds its C tensor (``[M, N]``) over
    ``out``'s pointer for the grid and the unused TMA-store descriptor, but nothing is ever
    stored through it.  The GEMM ``kernel`` the base ``__call__`` launches is routed to
    ``kernel_swiglu``, which takes ``out`` as one more argument.
    """

    def __init__(
        self,
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        crd_mn=(1, 0),
        debug_coords=False,
        divmode="nr",
    ):
        """``crd_mn``: which entries of a C coordinate (the TMA coordinate tensor's tuple) are
        the row and the column.  ``debug_coords`` (diagnostic build): instead of the SwiGLU
        store, every epilogue thread of a CTA's first tile writes the coordinates of its
        subtile-0 fragment elements 0, 1, HALF and EPI_N - 1 as int32 to ``out`` (slot =
        linear CTA id * 128 + thread)."""
        super().__init__(sf_vec_size, mma_tiler_mn, cluster_shape_mn)
        self.crd_m, self.crd_n = crd_mn
        self.debug_coords = debug_coords
        # "nr": silu_nr (served); "rn": silu_rn (the rows SwiGLU's expression; same bits, slower)
        self.divmode = divmode

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._swiglu_out = out
        _proj.GlmMxf8Gemm.__call__(self, a, b, sfa, sfb, out, m, n, k, max_active_clusters, stream)

    def kernel(self, *args):
        """The base ``__call__``'s launch, with the SwiGLU output appended."""
        return self.kernel_swiglu(*args, self._swiglu_out)

    @cute.kernel
    def kernel_swiglu(
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
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        mOut: cute.Tensor,
    ):
        """
        The persistent block-scaled GEMM with the SwiGLU epilogue (see the module docstring).
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
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Warp)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            self.threads_per_warp * len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        )
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

        # Tensor memory dealloc barrier init
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)  # noqa: F841
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
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

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

        #  TMA load scaled factor A partition_S/D
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

        # TMA load scaled factor B partition_S/D
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
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

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
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
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
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
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
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
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
            tile_sched = StaticPersistentTileScheduler.create(
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

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
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

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] in {64, 192}):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
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
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        #  Copy SFA/SFB from smem to tmem
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

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, ab_consumer_state.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            [tCrA[tile_crd], tCtSFA],
                            [tCrB[tile_crd], tCtSFB_mma],
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
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            #
            # SwiGLU epilogue (this module; replaces the C conversion, SMEM staging and
            # TMA store).  The T2R fragment of one subtile is one row's EPI_N consecutive
            # accumulator columns: HALF gate then the HALF matching up columns.
            #
            # (EPI_N == the subtile width is checked by ``compile_gemm`` after the compile.)
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            # (EPI_M, EPI_N, #EPI_M, #EPI_N, RestM, RestN, RestL): C coordinates
            gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            # (T2R, T2R_M, T2R_N, #EPI_M, #EPI_N, RestM, RestN, RestL)
            tTR_gC_base = thr_copy_t2r.partition_D(gC_mnl_epi)
            out_rows = cute.size(mOut, mode=[0])
            out_cols = cute.size(mOut, mode=[1])

            #
            # Persistent tile scheduling loop
            #
            tile_sched = StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                # (T2R, T2R_M, T2R_N, #EPI_M, #EPI_N)
                tTR_gC = tTR_gC_base[(None, None, None, None, None, *mma_tile_coord_mnl)]

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = True if acc_stage_index == 0 else False
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    #
                    # The served epilogue's C values (FP32 -> BF16, round to nearest), then
                    # the rows SwiGLU's arithmetic on them
                    #
                    acc_bf = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.BFloat16)
                    acc_bf.store(tTR_rAcc.load().to(cutlass.BFloat16))
                    res = cute.make_rmem_tensor((HALF,), cutlass.BFloat16)
                    for i in cutlass.range_constexpr(HALF):
                        g = acc_bf[i].to(cutlass.Float32)
                        u = acc_bf[i + HALF].to(cutlass.Float32)
                        if cutlass.const_expr(self.divmode == "nr"):
                            s = silu_nr(g)
                        else:
                            s = silu_rn(g)
                        res[i] = (s * u).to(cutlass.BFloat16)

                    #
                    # Store this row's HALF outputs (columns crd_n / 2 ...) straight to
                    # global memory: two 16-byte stores; rows past M are padding
                    #
                    frag_c = tTR_gC[(None, None, None, real_subtile_idx)]
                    crd = frag_c[0]
                    if cutlass.const_expr(self.debug_coords):
                        if tile_sched.num_tiles_executed == 0:
                            if subtile_idx == 0:
                                bx, by, bz = cute.arch.block_idx()
                                gx, gy, _ = cute.arch.grid_dim()
                                slot = ((bz * gy + by) * gx + bx) * 128 + epi_tidx
                                dbg = cute.make_tensor(
                                    cute.recast_ptr(mOut.iterator, dtype=cutlass.Int32),
                                    cute.make_layout((1 << 16, 8), stride=(8, 1)),
                                )
                                c1 = frag_c[1]
                                c16 = frag_c[HALF]
                                c31 = frag_c[EPI_N - 1]
                                dbg[slot, 0] = crd[0]
                                dbg[slot, 1] = crd[1]
                                dbg[slot, 2] = c1[0]
                                dbg[slot, 3] = c1[1]
                                dbg[slot, 4] = c16[0]
                                dbg[slot, 5] = c16[1]
                                dbg[slot, 6] = c31[0]
                                dbg[slot, 7] = c31[1]
                    else:
                        row = crd[self.crd_m]
                        if row < out_rows:
                            off = cutlass.Int64(row) * out_cols + cutlass.Int64(
                                crd[self.crd_n] // 2
                            )
                            dst = cute.make_tensor(
                                mOut.iterator + cute.assume(off, HALF),
                                cute.make_layout((HALF,)),
                            )
                            cute.autovec_copy(res, dst)

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

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


Compiled = Callable[..., object]
_COMPILED: dict[tuple, Compiled] = {}
_ALIGN = 16
# (row, column) entries of a C coordinate: the coordinate tensor is the TMA store's, whose tuple
# runs in TMA dimension order -- the contiguous N first -- so a row-major C coordinate is (n, m).
# (Measured: a probe dumped (col, row) with the column +1 per fragment element; (0, 1) faulted.)
CRD_MN = (1, 0)


def compile_gemm(
    tile: str = _proj.DEFAULT_TILE,
    device_index: int = 0,
    crd_mn: tuple[int, int] = CRD_MN,
    debug_coords: bool = False,
    divmode: str = "nr",
) -> Compiled:
    """``fn(a[M,6144] u8, b[1024,6144] u8, sfa u8, sfb u8, out[M,512] bf16, M, 1024, 6144,
    stream)``; ``b`` is the gate|up weight quantized after ``interleave_gate_up``."""
    key = (tile, device_index, crd_mn, debug_coords, divmode)
    if key not in _COMPILED:
        tiler, cluster = _proj.TILES[tile]
        gemm = GlmMxf8SwigluGemm(
            _proj._SF_VEC,
            tiler,
            cluster,
            crd_mn=crd_mn,
            debug_coords=debug_coords,
            divmode=divmode,
        )
        gemm.kernel_swiglu.set_name_prefix(
            "glm52_ctx_proj_mxf8_shared_up_swiglu", remove_cutlass_symbol=True
        )
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
        _COMPILED[key] = cute.compile(
            gemm,
            make_fake_tensor(
                cutlass.Uint8,
                (cute.sym_int32(), HIDDEN),
                stride=(HIDDEN, 1),
                assumed_align=_ALIGN,
            ),
            make_fake_tensor(
                cutlass.Uint8,
                (GATE_UP, HIDDEN),
                stride=(HIDDEN, 1),
                assumed_align=_ALIGN,
            ),
            make_fake_compact_tensor(
                cutlass.Uint8, (cute.sym_int32(),), assumed_align=_proj._SF_ALIGN
            ),
            make_fake_compact_tensor(
                cutlass.Uint8, (cute.sym_int32(),), assumed_align=_proj._SF_ALIGN
            ),
            make_fake_tensor(
                cutlass.BFloat16,
                (cute.sym_int32(), OUT),
                stride=(OUT, 1),
                assumed_align=_ALIGN,
            ),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            mac,
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
        if gemm.epi_tile_n != EPI_N:
            raise ValueError(
                f"SwiGLU epilogue needs a {EPI_N}-column epilogue subtile, the GEMM chose {gemm.epi_tile_n}"
            )
    return _COMPILED[key]


def interleave_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """BF16 gate|up weight ``[1024, K]`` (gate rows first, the served ``shared_up_mxf8``
    order) -> rows in blocks of ``HALF``: gate 16b..16b+15, up 16b..16b+15, b = 0..31.
    Quantize the result with ``self_attn_prefill_projections_mxf8.quantize_weight``."""
    n, k = weight.shape
    if n != GATE_UP:
        raise ValueError(f"interleave_gate_up: [{GATE_UP}, K] expected, got {tuple(weight.shape)}")
    return weight.reshape(2, OUT // HALF, HALF, k).permute(1, 0, 2, 3).reshape(n, k).contiguous()


# Largest row count on SMALL_TILE (256x128): the fused epilogue makes the 2-accumulator-stage
# tile faster than 256x256 up to ~4k rows (measured: T=2048 12.2 vs 19.4 us, 4096 19.9 vs 20.6,
# 7310 43.1 vs 37.1).  The bytes do not depend on the tile.
SMALL_MAX_ROWS = 4_096


class Mxf8SwigluProjection:
    """The fused launcher: ``__call__(x, w, out, pre=None)`` writes ``SwiGLU(x @ W^T)`` as
    BF16 ``out [T, 512]``, the bytes of ``Mxf8Projection("shared_up_mxf8")`` into
    ``[T, 1024]`` followed by ``moe_sigmoid_topk_swiglu_rows.swiglu``.  ``w`` is the weight
    quantized after ``interleave_gate_up``.  Tiles, ``pre`` (the producer-written MXF8
    operand) and the counters are ``Mxf8Projection``'s; the 256x128 tile serves up to
    ``SMALL_MAX_ROWS`` rows (the served op switches at 1024)."""

    def __init__(
        self,
        device_index: int = 0,
        crd_mn: tuple[int, int] = CRD_MN,
        small_max_rows: int = SMALL_MAX_ROWS,
        divmode: str = "nr",
    ):
        """``small_max_rows``: the largest row count on ``SMALL_TILE``; ``divmode``: the
        epilogue's ``silu`` (``"nr"`` served, ``"rn"`` the rows SwiGLU's expression)."""
        self.abi = _proj.ABIS[OP]
        self.device_index = device_index
        self.small_max_rows = small_max_rows
        self.tiles = (_proj.SMALL_TILE, _proj.DEFAULT_TILE)
        self._quant = _quant.compile_quant(device_index)
        self._gemms = {t: compile_gemm(t, device_index, crd_mn, False, divmode) for t in self.tiles}
        self.choice_hits = {t: 0 for t in self.tiles}
        self.pre_hits = 0

    def tile_for(self, rows: int) -> str:
        return _proj.SMALL_TILE if rows <= self.small_max_rows else _proj.DEFAULT_TILE

    def warm_rows(self) -> tuple[int, ...]:
        """One row count per compiled tile (each tile is its own kernel to load)."""
        return (min(128, self.small_max_rows), self.small_max_rows + 1)

    def __call__(
        self,
        x: torch.Tensor,
        w: "_proj.QuantizedWeight",
        out: torch.Tensor,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        rows = int(x.shape[0])
        if (
            x.shape[1] != HIDDEN
            or tuple(out.shape) != (rows, OUT)
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
            or out.data_ptr() % _ALIGN
        ):
            raise ValueError(
                f"shared_up_swiglu mxf8: x [T,{HIDDEN}] -> out bf16 [T,{OUT}] contiguous "
                f"16-byte aligned required, got {tuple(x.shape)} -> {tuple(out.shape)} {out.dtype}"
            )
        if pre is not None:
            xq, sfa = pre
            if (
                xq.dtype != torch.uint8
                or tuple(xq.shape) != (rows, HIDDEN)
                or not xq.is_contiguous()
                or xq.data_ptr() % _ALIGN
                or sfa.dtype != torch.uint8
                or not sfa.is_contiguous()
                or sfa.numel() < _quant.sf_bytes(rows, HIDDEN)
                or sfa.data_ptr() % _proj._SF_ALIGN
            ):
                raise ValueError(
                    f"shared_up_swiglu mxf8: pre-quantized operand must be uint8 [{rows},{HIDDEN}] "
                    f"contiguous + >= sf_bytes scale bytes, got {tuple(xq.shape)} {xq.dtype} / "
                    f"{sfa.numel()} {sfa.dtype}"
                )
            self.pre_hits += 1
        else:
            xq = torch.empty(rows, HIDDEN, dtype=torch.uint8, device=x.device)
            sfa = torch.empty(_quant.sf_bytes(rows, HIDDEN), dtype=torch.uint8, device=x.device)
            _quant.quantize(self._quant, x, xq, sfa)
        tile = self.tile_for(rows)
        self.choice_hits[tile] += 1
        self._gemms[tile](
            xq,
            w.q,
            sfa,
            w.sf,
            out,
            rows,
            GATE_UP,
            HIDDEN,
            _raw_stream(x.device),
        )
        return out
