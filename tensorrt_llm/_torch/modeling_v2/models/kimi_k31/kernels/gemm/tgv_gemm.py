# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# The kernel and MMA-warp bodies below are FlashInfer's TGV CuTe DSL GEMM (Apache-2.0,
# ``flashinfer/gemm/kernels/tgv_gemm_cute_ext.py``), with the 2-CTA TMEM dealloc moved into a pair teardown;
# everything else is reused through subclassing.
"""FlashInfer's TGV low-latency GEMM with a pair-synchronized 2-CTA teardown.

In the 2-CTA configurations the stock kernel lets each CTA of the cluster pair run ``tcgen05.dealloc.cta_group::2``
and exit as soon as its own epilogue is done; the dealloc then can address the peer CTA after the peer has exited
(an SM CGA exception, "cluster target block not present"). Here every thread of both CTAs meets on a cluster barrier
before the dealloc and again after it, so neither CTA deallocates while its peer is still running its epilogue, nor
exits while its peer deallocates. The GEMM itself (tiles, K order, accumulation, epilogue) is FlashInfer's, so the
output is the stock kernel's bit for bit; 1-CTA configurations are unchanged.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from flashinfer.gemm.kernels import tgv_gemm_cute_ext as _tgv

WorkTileInfo = _tgv.WorkTileInfo


class TgvGemmKernel(_tgv.TgvGemmCuteExtKernel):
    """FlashInfer's TGV GEMM with the 2-CTA TMEM dealloc and exit synchronized across the CTA pair."""

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0) — unused when has_bias=False
    ):
        """
        Device-side dispatcher: build MMA descriptor, allocate SMEM/TMEM/
        mbarriers, init barriers, dispatch into dma_a/dma_b/mma/epilog warps.
        """
        DMA_Stage = self.num_ab_stage

        # ---- Tiled MMA ----
        # make_trivial_tiled_mma picks the largest tcgen05.mma atom that fits
        # mma_tiler_mn. 1-CTA bf16 (64,8): Mma_M=(16,4)=64, Mma_N=8, Mma_K=16.
        # 2-CTA bf16 K-major (128,N): Mma_M=128 split across cluster, Mma_K=16.
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type  # bf16
        c_dtype = mC.element_type  # bf16
        d_layout = utils.LayoutEnum.from_tensor(mC)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)  # 1 (1-CTA) or 2 (2-CTA)

        # NumMma_K = CTA_K/Mma_K — inner-K trip count for the MMA warp.
        # bf16 default: Mma_K=16, CTA_K=128 → mma_inst_tile_k = 8.
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)  # CTA tile of A
        # 2-CTA splits B across cluster M (each CTA loads Mma_N/2 cols).
        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)  # CTA tile of C

        # ---- WorkTileInfo (static 1-tile-per-CTA scheduler) ----
        # blockIdx maps directly to (M_tile, N_tile, batch). K_idx_start/end
        # carry over from C++ for future split-K; here every CTA reduces the
        # full K range (= ceil(K/CTA_K) tiles, e.g. 12 for K=1536).
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # In 2-CTA mode: identify which CTA within the (2,1,1) cluster owns
        # the leader role (issues MMAs / multicasts commits). In 1-CTA mode
        # every CTA is its own leader.
        leader_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.use_2cta):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader = cta_rank_in_cluster == 0
        else:
            cta_rank_in_cluster = leader_rank
            is_leader = cutlass.Boolean(True)

        # ---- SMEM A/B (DMA_Stage-staged, swizzled ComposedLayout) ----
        # alignment=1024 covers TMA's natural alignment requirements. 2-CTA
        # halves the M extent of sA and the N extent of sB internally (the SS
        # atom splits A across cluster M and B across cluster M).
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — Sw<3,4,3>
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — Sw<3,4,3>

        # sA per-CTA M extent = cta_m for both modes; sB per-CTA N extent =
        # cta_n (1-CTA) or cta_n/2 (2-CTA, SS atom splits B across cluster M).
        sA = cute_ext.allocate(  # ((Mma_M_per_cta, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )
        sB = cute_ext.allocate(  # ((Mma_N_per_cta, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
            ab_dtype,
            cute.AddressSpace.smem,
            b_smem_layout_staged,
            alignment=1024,
        )

        # ---- TMEM accumulator layout (manual alloc; see MMA warp) ----
        # cute_ext.make_tmem_layout_acc accounts for cta_group when handed
        # the cluster MMA tile, so the resulting layout is already per-CTA.
        # acc_layout per-CTA: ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        # = (((16,4), cta_n), 1, 1, 1) for both 1-CTA and 2-CTA modes.
        # We still allocate TMEM manually (via cute.arch.alloc_tmem below) for better performance.
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma,
            self.mma_tiler_mn,
            acc_stage=1,
        )

        # ---- Raw mbarriers (Int64 SMEM arrays) + tmem_base_ptr Int32 slot ----
        # Two flavors of barriers exist on Hopper/Blackwell:
        #   1. Named barriers (bar.arv/bar.sync) — Ampere-era, 16 hardware
        #      barriers per SM, only handle intra-CTA thread sync.
        #   2. Mbarriers (mbarrier.* PTX) — 64-bit in SMEM per barrier,
        #      software-programmable. Used here for (a) thread sync within a
        #      CTA, (b) TMA transaction-count tracking (TMA→SM signaling),
        #      and (c) 1-arrival "wake the consumer warp" patterns.
        # Allocated as 1D tensors; .iterator gives a Pointer[Int64] supporting
        # `bar + stage` arithmetic. Arrival counts are in the module docstring.
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_tma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        # alloc_tmem writes the TMEM base address to this slot; both MMA and
        # EPILOG read it back via retrieve_tmem_ptr.
        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        bar_full = bar_full_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_empty = bar_empty_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_tma_epilog = bar_tma_epilog_arr.iterator  # Pointer[Int64]
        bar_mma_epilog = bar_mma_epilog_arr.iterator  # Pointer[Int64]
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator  # Pointer[Int64]
        tmem_base_ptr = tmem_base_arr.iterator  # Pointer[Int32]

        # ---- Barrier init (1 thread, PTX mbarrier.init is single-thread) ----
        # bar_full arrival = 2 * num_mma_ctas:
        #   1-CTA: 2 (own DMA_A + DMA_B)
        #   2-CTA: 4 (own + peer DMA_A/DMA_B, routed via SM100_TMA_LOAD_2SM
        #          onto the leader's bar). In 2-CTA the peer's copy is unused.
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(bar_tma_epilog, 32)  # whole DMA_B warp
                cute.arch.mbarrier_init(bar_mma_epilog, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + 128)  # MMA + 4 EPILOG warps

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.use_2cta):
            # Cluster-wide sync: ensure peer's barriers are visible before any
            # cross-CTA arrives. Splitting arrive/wait lets the per-CTA tile
            # setup below overlap with the cluster barrier. (Implicit
            # intra-CTA sync — cluster.arrive requires all threads to reach.)
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        work_tile_info = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k),
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        # ---- CTA-to-Value maps for TMA (constexpr) ----
        # Encodes "which cluster CTA is responsible for which slice", i.e.
        # the CTA-coord → logical-multicast-id layout. For our 1×1 cluster
        # (no multicast) it's an identity layout; cute_ext.tma_load still
        # needs this explicit layout to construct the TMA descriptor.
        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

        # ---- Per-CTA tile views ----
        # local_tile(t, tiler, coord) = zipped_divide(t, tiler)[coord]; modes
        # passed `None` stay free. Per-CTA M/N extents:
        #   gA M extent = cta_m   (both modes; SS atom splits cluster M)
        #   gB N extent = cta_n   (1-CTA) or cta_n//2 (2-CTA)
        #   gD M×N      = cta_m × cta_n (cluster N is identical for both CTAs)
        gA_tile = cute.local_tile(  # (cta_m, cta_k, Tiles_K)
            mA,
            a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )
        # gB n-index: 1-CTA just bidy; 2-CTA each cluster N_idx covers
        # num_mma_ctas tiles of b_tiler, each CTA loads one half.
        if cutlass.const_expr(self.use_2cta):
            gB_n_idx = work_tile_info.N_idx * num_mma_ctas + cta_rank_in_cluster
        else:
            gB_n_idx = work_tile_info.N_idx
        gB_tile = cute.local_tile(  # (cta_n//num_mma_ctas, cta_k, Tiles_K)
            mB,
            b_tiler_nk,
            (gB_n_idx, None, work_tile_info.L_idx),
        )
        gD_tile = cute.local_tile(  # (cta_m, cta_n)
            mC,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )
        # gBias_tile: same (cta_m, cta_n) shape as gD_tile but stride (1, 0)
        # — the same M element is repeated across N. local_tile preserves the
        # (1, 0, 0) stride from mBias, so this works automatically.
        gBias_tile = cute.local_tile(  # (cta_m, cta_n) stride (1, 0)
            mBias,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )

        if cutlass.const_expr(self.use_2cta):
            # Cluster sync deferred to here — tile-view setup above runs in
            # parallel with the cluster barrier latency.
            cute.arch.cluster_wait()

        # ---- Warp dispatch (warp 3 idle, kept for 256-thread parity) ----
        if warp_idx == 0:
            self.dma_a_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gA_tile,
                sA,
                a_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                is_leader,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx >= 4:
            # Epilog tid is 128..255 in the CTA; offset to 0..127 for partition.
            epi_tid = tidx - 128
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD_tile,
                gBias_tile,
                epi_tid,
                c_dtype,
                d_layout,
            )

        # ---- Pair teardown (2-CTA): both CTAs of the pair meet before the TMEM dealloc and again before exit ----
        self.pair_teardown(warp_idx, tmem_base_ptr)

    @cute.experimental.jit
    def mma_warp(
        self,
        is_leader: cutlass.Boolean,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot for TMEM addr
        acc_layout: cutlass.Constexpr,  # TMEM accumulator layout
        mma_inst_tile_k: cutlass.Constexpr,  # NumMma_K — inner loop count
        k_tile_count: cutlass.Int32,  # Tiles_K — outer loop count
    ):
        DMA_Stage = self.num_ab_stage

        # ---- TMEM allocation (manual) ----
        # We allocate TMEM explicitly here rather than via the convenience
        # helper ``cute_ext.allocate(tmem)`` so that we can *hide the TMEM
        # allocation latency*: the alloc must happen here in MMA's prolog
        # (we need the TMEM ptr before the first tcgen05.mma), but it
        # serializes against the next CTA's own alloc on the back-to-back
        # PDL launch path. By manually emitting
        # ``relinquish_tmem_alloc_permit`` right after our alloc completes,
        # the next CTA can start ITS allocation in parallel with this
        # CTA's MMA work rather than stalling at its own prolog waiting
        # for us. cute_ext.allocate(tmem) does not expose this early-
        # relinquish control — worth ~3% on the PDL hot path.
        #
        # TMEM on SM100 has 128 lanes × 512 columns × 4B = 256KB total. We
        # allocate half (256 of 512 cols) so the next CTA can prefetch its
        # alloc on the other half. For CTA_M=64 the accumulator only uses
        # 64 lanes (16 lanes × 4 subpartitions), well within half-of-TMEM.
        # 2-CTA: alloc/relinquish/dealloc are cluster-coherent (is_two_cta=True).
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # phase 0: 32 of 160
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)

        # Bind acc_layout to the just-allocated TMEM ptr. Drop the trailing
        # AccStage=1 mode so the MMA atom sees (MMA, Mma_M, Mma_N).
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        acc_view = tAcc[None, None, None, 0]  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N)

        if is_leader:
            # MMA atom: each cute_ext.dot call = 1 tcgen05.mma. ACCUMULATE:
            #   False → C  = A*B  (first inst of first k-tile only)
            #   True  → C += A*B
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            # 2-CTA: mask=0b11 multicasts the commit to both CTAs' bar_empty.
            commit_mask = 0b11 if self.use_2cta else None

            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % DMA_Stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)

                # Inner-K loop: NumMma_K straight-line tcgen05.mma instructions.
                # cute_ext.dot expects rank-3 operands in the MMA fragment
                # profile (MMA_atom, REST_M, REST_K). Slicing
                # `sA[None,None,k_block,stage]` leaves rank 2
                # ((Mma_M, Mma_K), NumMma_M), so pad explicitly via
                # cute.append_ones.
                for k_block in range(mma_inst_tile_k):
                    if k_block == 0:
                        # First inst of the very first k_tile clears TMEM.
                        mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    else:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                    a_frag = cute.append_ones(
                        sA[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    b_frag = cute.append_ones(
                        sB[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    cute_ext.dot(mma_atom, a_frag, b_frag, acc_view)

                # CRITICAL: tcgen05.commit MUST be inside elect_one(). Unlike
                # the C++ ``cutlass::arch::umma_arrive`` helper (which has an
                # internal elect_one_sync), the DSL's tcgen05.commit has no
                # guard — without this, all 32 lanes commit (32× redundant
                # arrivals on bar_empty). This was a major perf bug encountered
                # during development.
                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, commit_mask, self.cta_group)

                if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                    full_phase = full_phase ^ 1

            # Wake the epilog (1-thread arrive on the 1-arrival barrier).
            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)

        # ---- TMEM dealloc: wait for own EPILOG's tcgen05.ld to retire, free.
        # bar_tmem_alloc phase 1 fires after EPILOG's tcgen05.ld is observable
        # (post fence_view_async_tmem_load). 2-CTA dealloc is per-CTA (no
        # cross-CTA handshake) because each CTA owns its own physical TMEM
        # half; the cluster-shared accumulator is just a logical view.
        # Phase 0 (the allocation, with the epilogue's first arrivals) completes before this warp's phase-1 arrival.
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # phase 1: 32 of 160
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        # 2-CTA: the dealloc runs in pair_teardown, once both CTAs of the pair are there.
        if cutlass.const_expr(not self.use_2cta):
            cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=self.use_2cta)

    @cute.experimental.jit
    def pair_teardown(self, warp_idx, tmem_base_ptr):
        """2-CTA: every thread of both CTAs arrives once its role is done (the MMA warp after its own epilogue
        released TMEM), the MMA warp deallocates the pair's TMEM, and every thread meets again before exit."""
        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if warp_idx == 2:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
                cute.arch.dealloc_tmem(tmem_ptr, 256, is_two_cta=True)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()


TgvConfig = Tuple[int, int, int, bool]

#: Compiled kernels by (device, config, PDL, operand layout).
_COMPILED: Dict[tuple, object] = {}


def validate_config(cta_m: int, cta_n: int, ab_stages: int, use_2cta: bool) -> None:
    """Raise ValueError for a config the TGV kernel class cannot build (``cta_m`` 64 / 128; ``cta_n`` a multiple of
    8, of 16 with 2-CTA, up to 256)."""
    step = 16 if use_2cta else 8
    if (
        cta_m not in (64, 128)
        or cta_n % step
        or not step <= cta_n <= 256
        or not 1 <= ab_stages <= 16
    ):
        raise ValueError(f"no TGV config {(cta_m, cta_n, ab_stages, use_2cta)}")


def run_tgv(
    a: torch.Tensor, w_t: torch.Tensor, out: torch.Tensor, pdl: bool, cfg: TgvConfig
) -> torch.Tensor:
    """``out = a @ w_t`` (``a [M, K]`` row-major, ``w_t [K, N]`` a row-major weight's transpose, BF16) with the TGV
    config ``cfg = (cta_m, cta_n, ab_stages, use_2cta)``: FlashInfer's TGV GEMM, pair-synchronized at 2-CTA. The
    kernel compiles on the first call per (config, PDL, layout), which must not be under CUDA-graph capture."""
    cta_m, cta_n, ab_stages, use_2cta = cfg
    a_, b_, c_, _, layout = _tgv._to_cute_swap(a, w_t, out, None)
    key = (a.device.index, tuple(cfg), bool(pdl), layout)
    compiled = _COMPILED.get(key)
    if compiled is None:
        validate_config(*cfg)
        gemm = TgvGemmKernel(
            acc_dtype=cutlass.Float32,
            cta_m=cta_m,
            cta_n=cta_n,
            cta_k=_tgv._TGV_CUTE_EXT_CTA_K,
            num_ab_stage=ab_stages,
            use_2cta=use_2cta,
            use_pdl=bool(pdl),
            has_bias=False,
        )
        ra, rb, rc, _ = _tgv._make_compile_repr_tensors(torch.bfloat16, False, *layout)
        compiled = cute_ext.compile(_tgv._bmm_no_bias, gemm, ra, rb, rc, _tgv.make_fake_stream())
        _COMPILED[key] = compiled
    compiled(a_, b_, c_, cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream))
    return out


__all__ = ["TgvGemmKernel", "run_tgv", "validate_config"]
