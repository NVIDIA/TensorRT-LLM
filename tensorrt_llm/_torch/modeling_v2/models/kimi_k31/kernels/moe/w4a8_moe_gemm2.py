# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed experts: W4A8 grouped GEMMs on CTA pairs (``tcgen05`` ``cta_group::2``).

The operands, routing layout and epilogues are those of :mod:`.w4a8_moe_gemm` (read its
docstring first); this variant runs one GEMM (FC1 or FC2) per launch on clusters of two CTAs:

* one ``tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale`` per 32-deep K step computes a
  256-row weight m-pair x ``n_tile`` token rows: each CTA streams the 128 weight rows of its own
  m-tile (``2 * pair + rank``, TMA with the U4 -> U8 unpack) and gathers ``n_tile / 2`` of the
  tile's token rows (cp.async); the scale factors of all ``n_tile`` token rows go into each CTA's
  SFB atom (the MMA reads them from both CTAs);
* each CTA's tensor memory holds the FP32 accumulator of its 128 weight rows x ``n_tile`` tokens,
  so each CTA runs the unchanged FC1 / FC2 epilogue for its m-tile;
* the MMA N of a unit follows its tile's valid rows (rounded up to :data:`_MMA_N_STEP`), so a partly
  filled tile spends no tensor-core work (power, under the GPU's power cap) on padding rows.

A stage costs each CTA 16 KB of unpacked weights + ``n_tile / 2`` x 128 bytes of tokens, so a
128-row token tile streams its expert's weights once with the shared-memory footprint of a 64-row
1-CTA tile. Per output element the K order and arithmetic are the 1-CTA kernel's: every output row
is bitwise equal to :func:`run_w4a8_moe`'s on the same routing (any pair tile; the permuted rows
differ with the tile).

Synchronization (the leader CTA, rank 0, issues every MMA):

* ``full[s]`` (leader): the leader's weight TMA ``expect_tx`` (both CTAs' weight TMAs complete
  their bytes on it) + one arrival per CTA from its forwarding warp;
* ``bfull[s]`` (each CTA): one ``cp.async.mbarrier.arrive.noinc`` per token-gather lane; the
  forwarding warp waits for it, fences the generic-proxy writes for the tensor core and arrives on
  the leader's ``full[s]`` (cp.async cannot signal a peer barrier);
* ``empty[s]`` / ``acc_full[a]`` (each CTA): the leader's ``tcgen05.commit`` multicast to the pair;
* ``acc_empty[a]`` (leader): one arrival per epilogue warp of both CTAs.

Warp roles (32 x (7 + ceil(n_tile / 64)) threads): warps 0-3 epilogue, then MMA (leader) + TMEM
allocation, weight TMA, forwarding, token gather (32 rows each, the last one partial at n_tile 224).
"""

from __future__ import annotations

import functools
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr

from .w4a8_moe_gemm import (
    _EPI_BAR,
    _EPI_CHUNK,
    HIDDEN,
    KBLK,
    MMA_M,
    SF_STAGE_BYTES,
    SF_VEC,
    W4A8MoeKernel,
    _bar_sync,
    _cp_async_4,
    _cp_async_16,
    _e4m3x4,
    _fp4_ptr,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _i32_arg,
    _lds_f32,
    _mxfp8_scale,
    _rcp_approx,
    _sf_ptr,
    _shfl_xor_f32,
    _smem_capacity,
    _stg_u8,
    _stg_v4,
    _sts_f32,
    num_sms,
)

#: Weight rows per pair MMA (UMMA M of ``cta_group::2``).
PAIR_M = 2 * MMA_M
#: Token rows per tile (UMMA N of the pair; each CTA gathers half).
SUPPORTED_PAIR_N_TILES = (64, 128, 192, 224, 256)

#: Epilogue warps (one per TMEM lane quarter).
_EPI_WARPS = 4
#: Floats per token row of the FC1 staging buffer: accumulator row r sits at r + 8 (r // 32), and the
#: odd pitch puts a warp's two token rows on opposite bank parities, so the SiTU job reads are
#: conflict-free (the 1-CTA layout, pitch 132 without the shift, reads 4-way conflicted).
_EPI_ROW2 = 153
#: Granularity of a unit's MMA N: its tile's valid rows rounded up to it (compiled variants of N
#: ``_MMA_N_STEP, 2 _MMA_N_STEP, .., n_tile``).
_MMA_N_STEP = 32
#: Columns of one 32-row block of scale factors in tensor memory (4 K blocks of a stage).
_SF_COLS_PER_32 = 4


class W4A8MoeKernel2Cta(W4A8MoeKernel):
    """FC1 (SiTU + MXFP8 requant epilogue) or FC2 (BF16 epilogue) of the K3.1 routed experts on
    CTA pairs (see the module docstring).

    Schedule: a launch is ``tiles * pairs`` units (token tile ``c``, weight m-pair ``mp``), unit
    ``u = c * pairs + mp``; cluster ``q`` of ``Q`` takes units ``q, q + Q, ...`` (static, both
    CTAs of a cluster walk the same units).

    The MMA N of a unit is its tile's valid rows rounded up to :data:`_MMA_N_STEP`, and CTA ``r``
    gathers tile rows ``[r N / 2, (r + 1) N / 2)``: no tensor-core work on padding rows, which
    matters under the power cap (higher SM clock); outputs are unchanged (bitwise). The FC1
    epilogue (:meth:`_epi_fc1_w`) runs one thread per (token, 16 outputs).

    Args:
        n_tile: token rows per tile (pair UMMA N), one of :data:`SUPPORTED_PAIR_N_TILES` (256 holds
            one accumulator stage: two do not fit next to the scale factors in 512 TMEM columns).
        inter / hidden / fc1 / gate_cap / linear_cap: as :class:`.W4A8MoeKernel` (``fc1`` False =
            FC2).
        p2t_div: ``p2t`` holds tokens (1) or expanded indices ``token * p2t_div + slot``.
        max_acc_stages: accumulator stages in TMEM (bounded by the 512 columns).
        early_weights: FC2 only: the weight producer streams before the wait on the previous (FC1)
            grid; the token gather and the epilogue wait.
    """

    def __init__(
        self,
        n_tile: int,
        inter: int,
        hidden: int = HIDDEN,
        fc1: bool = True,
        gate_cap: float = 4.0,
        linear_cap: float = 25.0,
        max_stages: int = 16,
        p2t_div: int = 1,
        max_acc_stages: int = 2,
        early_weights: bool = False,
    ):
        assert n_tile in SUPPORTED_PAIR_N_TILES, n_tile
        assert (
            (2 * inter) % PAIR_M == 0
            and hidden % PAIR_M == 0
            and inter % KBLK == 0
            and hidden % KBLK == 0
        )
        self.n_tile = n_tile
        self.n_cta = n_tile // 2
        self.inter = inter
        self.hidden = hidden
        self.fc1 = fc1
        self.gate_cap = float(gate_cap)
        self.linear_cap = float(linear_cap)
        self.early_weights = bool(early_weights) and not fc1
        self.p2t_div = int(p2t_div)
        # warp roles: epilogue, MMA, weight TMA, forwarding, token gather
        self.epi_warps = _EPI_WARPS
        self.mma_warp = self.epi_warps
        self.tma_warp = self.epi_warps + 1
        self.xfer_warp = self.epi_warps + 2
        self.gather_warp = self.epi_warps + 3
        # MMA N variants (UMMA N: multiples of 16; each CTA holds N / 2 rows in 8-row swizzle atoms)
        self.mma_ns = tuple(range(_MMA_N_STEP, n_tile + 1, _MMA_N_STEP))
        self.mt = (2 * inter if fc1 else hidden) // PAIR_M
        self.kt = (hidden if fc1 else inter) // KBLK
        self.mma_tiler = (PAIR_M, n_tile, KBLK)
        self.b_stage_bytes = self.n_cta * KBLK
        # the SFB atoms cover the tile's rows rounded up to 128
        self.sfb_stage_bytes = (n_tile + 127) // 128 * SF_STAGE_BYTES
        stage = MMA_M * KBLK + self.b_stage_bytes + SF_STAGE_BYTES + self.sfb_stage_bytes
        self.gather_warps = (self.n_cta + 31) // 32
        self.threads = 32 * (self.gather_warp + self.gather_warps)
        # token columns per epilogue chunk
        self.epi_n = _EPI_CHUNK
        self.epi_floats = self.epi_n * _EPI_ROW2 if fc1 else self.epi_n * MMA_M // 2
        # barriers + the 1024-byte alignment of the operand rings
        reserve = 2048 + self.epi_floats * 4
        self.num_stages = max(2, min(max_stages, (_smem_capacity() - reserve) // stage))
        self.sfa_cols = (MMA_M // 32) * _SF_COLS_PER_32
        sf_cols = self.sfa_cols + (self.sfb_stage_bytes // SF_STAGE_BYTES) * 4 * _SF_COLS_PER_32
        # accumulator stages: as many as fit next to the scale factors (up to max_acc_stages), so
        # the MMA can run ahead of a slow epilogue
        self.acc_stages = max(1, min(int(max_acc_stages), (512 - sf_cols) // n_tile))
        self.acc_cols = self.acc_stages * n_tile
        cols = self.acc_cols + sf_cols
        assert cols <= 512, cols
        self.tmem_cols = 32
        while self.tmem_cols < cols:
            self.tmem_cols *= 2

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        w_sf_ptr: cute.Pointer,
        num_experts: Int32,
        x_addr: Int64,
        x_sf_addr: Int64,
        h_addr: Int64,
        h_sf_addr: Int64,
        y_addr: Int64,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        p2t: cute.Tensor,
        num_tiles: cute.Tensor,
        grid_ctas: Int32,
        stream: cuda.CUstream,
    ):
        S = self.num_stages
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            SF_VEC,
            tcgen05.CtaGroup.TWO,
            self.mma_tiler[:2],
        )
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, cutlass.Int8, S)
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, S
        )
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC, S)
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, SF_VEC, S)
        cl = cute.tiled_divide(cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,))
        m = 2 * self.inter if self.fc1 else self.hidden
        k = self.hidden if self.fc1 else self.inter
        a = cute.make_tensor(w_ptr, cute.make_ordered_layout((m, k, num_experts), order=(1, 0, 2)))
        sfa = cute.make_tensor(w_sf_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((2, 1), tiled_mma.thr_id)
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        tma_a, tA = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_one, self.mma_tiler, tiled_mma, cl.shape, internal_type=cutlass.Int8
        )
        tma_sfa, tSFA = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, sfa, sfa_one, self.mma_tiler, tiled_mma, cl.shape, internal_type=cutlass.Int16
        )
        # bytes one CTA's weight TMAs complete on the leader's full barrier per stage
        self.tx_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, a_one) + cute.size_in_bytes(
            cutlass.Float8E8M0FNU, sfa_one
        )

        @cute.struct
        class SharedStorage:
            full: cute.struct.MemRange[cutlass.Int64, S]
            empty: cute.struct.MemRange[cutlass.Int64, S]
            bfull: cute.struct.MemRange[cutlass.Int64, S]
            acc_full: cute.struct.MemRange[cutlass.Int64, self.acc_stages]
            acc_empty: cute.struct.MemRange[cutlass.Int64, self.acc_stages]
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            tmem_dealloc: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(b_layout.outer)], 1024
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfa_layout)], 128
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfb_layout)], 128
            ]
            sEpi: cute.struct.Align[cute.struct.MemRange[Float32, self.epi_floats], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_a,
            tA,
            tma_sfa,
            tSFA,
            x_addr,
            x_sf_addr,
            h_addr,
            h_sf_addr,
            y_addr,
            tile_expert,
            tile_limit,
            p2t,
            num_tiles,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            cl,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(2, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    # ------------------------------------------------------------------ device
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        x_addr: Int64,
        x_sf_addr: Int64,
        h_addr: Int64,
        h_sf_addr: Int64,
        y_addr: Int64,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        p2t: cute.Tensor,
        num_tiles: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        cl: cute.Layout,
    ):
        S = self.num_stages
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid_, _, _ = cute.arch.block_idx()
        nctas_, _, _ = cute.arch.grid_dim()
        rank = Int32(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()))
        pr = Int32(bid_) // 2
        npairs = Int32(nctas_) // 2

        st = utils.SmemAllocator().allocate(self.shared_storage)
        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(S):
                    # the leader's expect-tx arrival + one forwarding arrival per CTA
                    cute.arch.mbarrier_init(st.full.data_ptr() + i, 3)
                    cute.arch.mbarrier_init(st.empty.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.bfull.data_ptr() + i, 32 * self.gather_warps)
                for i in cutlass.range_constexpr(self.acc_stages):
                    cute.arch.mbarrier_init(st.acc_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.acc_empty.data_ptr() + i, 2 * self.epi_warps)
                cute.arch.mbarrier_init(st.tmem_ready, 1)
                cute.arch.mbarrier_init(st.tmem_free, self.epi_warps)
                cute.arch.mbarrier_init(st.tmem_dealloc, 32)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if warp == self.mma_warp:
            cute.arch.alloc_tmem(self.tmem_cols, st.tmem_holding_buf, is_two_cta=True)
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(st.tmem_ready)
        if cutlass.const_expr(not self.early_weights):
            _griddepcontrol_wait()
        nt = Int32(num_tiles[0])
        if warp == self.tma_warp:
            self._produce_weights2(st, tiled_mma, tma_a, mA, tma_sfa, mSFA, a_layout, sfa_layout, cl, tile_expert,
                                   nt, pr, npairs, rank)  # fmt: skip
        elif warp >= self.gather_warp:
            if cutlass.const_expr(self.early_weights):
                _griddepcontrol_wait()
            self._produce_tokens2(st, x_addr, x_sf_addr, h_addr, h_sf_addr, tile_limit, p2t, nt, pr, npairs, rank,
                                  warp - self.gather_warp)  # fmt: skip
        elif warp == self.xfer_warp:
            self._forward(st, nt, pr, npairs)
        elif warp == self.mma_warp:
            self._mma2(st, tiled_mma, a_layout, b_layout, sfa_layout, sfb_layout, tile_limit, nt, pr, npairs,
                       rank)  # fmt: skip
        elif warp < self.epi_warps:
            if cutlass.const_expr(self.early_weights):
                _griddepcontrol_wait()
            self._epilogue2(
                st, tiled_mma, tile_limit, h_addr, h_sf_addr, y_addr, nt, pr, npairs, rank, tidx
            )
        # no CTA of the pair leaves while its peer may still signal its barriers
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        _griddepcontrol_launch_dependents()

    @cute.jit
    def _tile_n(self, rows: Int32):
        """MMA N of a tile with ``rows`` valid rows: rounded up to :data:`_MMA_N_STEP`."""
        return cutlass.min(
            Int32(self.n_tile), (rows + (_MMA_N_STEP - 1)) // _MMA_N_STEP * _MMA_N_STEP
        )

    # ------------------------------------------------------------------ weight producer
    @cute.jit
    def _a_partition2(
        self, st, tiled_mma, tma_a, mA, tma_sfa, mSFA, a_layout, sfa_layout, cl, rank: Int32
    ):
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sSFA = st.sSFA.get_tensor(sfa_layout)
        thr_mma = tiled_mma.get_slice(rank)
        tiler = cute.slice_(self.mma_tiler, (None, 0, None))
        tCgA = thr_mma.partition_A(cute.local_tile(mA, tiler, (None, None, None)))
        tCgSFA = thr_mma.partition_A(cute.local_tile(mSFA, tiler, (None, None, None)))
        a_cta = cute.make_layout(cute.slice_(cl, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_a, 0, a_cta, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_sfa, 0, a_cta, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3)
        )
        return tAsA, tAgA, cute.filter_zeros(tAsSFA), cute.filter_zeros(tAgSFA)

    @cute.jit
    def _produce_weights2(self, st, tiled_mma, tma_a, mA, tma_sfa, mSFA, a_layout, sfa_layout, cl, tile_expert,
                          nt: Int32, pr: Int32, npairs: Int32, rank: Int32):  # fmt: skip
        """Each CTA's weight TMA (``cta_group::2`` form: the bytes complete on the leader's
        ``full[s]``) of its m-tile ``2 * mp + rank``; the leader arms ``full[s]`` for both."""
        S = self.num_stages
        mt = self.mt
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        tAsA, tAgA, tAsSFA, tAgSFA = self._a_partition2(
            st, tiled_mma, tma_a, mA, tma_sfa, mSFA, a_layout, sfa_layout, cl, rank
        )
        g = Int32(0)
        u = pr
        end = nt * mt
        while u < end:
            c, mp = self._unit(u)
            e = Int32(tile_expert[c])
            for kk in cutlass.range(0, self.kt, 1, unroll=1):
                s = g % S
                if g >= S:
                    cute.arch.mbarrier_wait(empty + s, ((g // S) & 1) ^ 1)
                if rank == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full + s, 2 * self.tx_bytes)
                cute.copy(tma_a, tAgA[(None, mp, kk, e)], tAsA[(None, s)], tma_bar_ptr=full + s)
                cute.copy(
                    tma_sfa, tAgSFA[(None, mp, kk, e)], tAsSFA[(None, s)], tma_bar_ptr=full + s
                )
                g += 1
            u += npairs

    # ------------------------------------------------------------------ token gather
    @cute.jit
    def _produce_tokens2(self, st, x_addr: Int64, x_sf_addr: Int64, h_addr: Int64, h_sf_addr: Int64, tile_limit,
                         p2t, nt: Int32, pr: Int32, npairs: Int32, rank: Int32, gw: Int32):  # fmt: skip
        """cp.async of tile rows ``rank * N / 2 + [32 gw, 32 gw + 32)`` (``N``: the unit's MMA N;
        local rows below ``N / 2`` only) into this CTA's B stage (the 1-CTA kernel's 128B-swizzled
        K-major rows) and of the scales of tile rows ``32 gw + lane + {0, n_tile / 2}`` into this
        CTA's SFB atom (all ``n_tile`` rows per CTA, at their tile positions);
        each lane signals ``bfull[s]`` once its copies landed."""
        S = self.num_stages
        n_tile = self.n_tile
        n_cta = self.n_cta
        mt = self.mt
        cpl = 32 * (KBLK // 16) // 32  # 16-byte chunks per lane and stage
        bfull, empty = st.bfull.data_ptr(), st.empty.data_ptr()
        lane = cute.arch.lane_idx()
        sb_base = st.sB.data_ptr().toint()
        sfb_base = st.sSFB.data_ptr().toint()
        src = cute.make_rmem_tensor((cpl,), Int64)
        ok = cute.make_rmem_tensor((cpl,), cutlass.Boolean)
        sf_src = cute.make_rmem_tensor((2,), Int64)
        sf_ok = cute.make_rmem_tensor((2,), cutlass.Boolean)
        base = x_addr if self.fc1 else h_addr
        sf_base = x_sf_addr if self.fc1 else h_sf_addr
        row_bytes = self.hidden if self.fc1 else self.inter
        g = Int32(0)
        u = pr
        end = nt * mt
        while u < end:
            c, _mp = self._unit(u)
            row0 = c * n_tile
            lim = Int32(tile_limit[c])
            # this CTA's rows of the tile: [rank * half, (rank + 1) * half) with half = MMA N / 2
            half = self._tile_n(lim - row0) // 2
            r_first = rank * half + gw * 32
            for i in cutlass.range_constexpr(cpl):
                idx = lane + 32 * i
                p = row0 + r_first + idx // 8
                ok[i] = p < lim
                ok[i] = ok[i] & (gw * 32 + idx // 8 < half)
                if cutlass.const_expr(n_cta % 32 != 0):
                    # the last gather warp is partial: its rows past the CTA's half are not its own
                    ok[i] = ok[i] & (gw * 32 + idx // 8 < n_cta)
                    p = cutlass.min(p, row0 + n_tile - 1)
                tok = p
                if cutlass.const_expr(self.fc1):
                    tok = Int32(p2t[p]) // self.p2t_div
                src[i] = base + Int64(tok) * row_bytes + Int64((idx % 8) * 16)
            for q in cutlass.range_constexpr(2):
                p = row0 + gw * 32 + lane + q * n_cta
                sf_ok[q] = p < lim
                if cutlass.const_expr(n_cta % 32 != 0):
                    sf_ok[q] = sf_ok[q] & (gw * 32 + lane < n_cta)
                    p = cutlass.min(p, row0 + n_tile - 1)
                tok = p
                if cutlass.const_expr(self.fc1):
                    tok = Int32(p2t[p]) // self.p2t_div
                sf_src[q] = sf_base + Int64(tok) * (row_bytes // SF_VEC)
            for kk in cutlass.range(0, self.kt, 1, unroll=1):
                s = g % S
                if g >= S:
                    cute.arch.mbarrier_wait(empty + s, ((g // S) & 1) ^ 1)
                b_stage = sb_base + s * self.b_stage_bytes
                for i in cutlass.range_constexpr(cpl):
                    if ok[i]:
                        idx = lane + 32 * i
                        n = gw * 32 + idx // 8
                        ch = idx % 8
                        dst = b_stage + (n // 8) * 1024 + (n % 8) * 128 + ((ch ^ (n % 8)) * 16)
                        _cp_async_16(dst, src[i] + Int64(kk * KBLK))
                for q in cutlass.range_constexpr(2):
                    if sf_ok[q]:
                        n = gw * 32 + lane + q * n_cta
                        dst = (sfb_base + s * self.sfb_stage_bytes + (n // 128) * SF_STAGE_BYTES + (n % 32) * 16
                               + ((n % 128) // 32) * 4)  # fmt: skip
                        _cp_async_4(dst, sf_src[q] + Int64(kk * (KBLK // SF_VEC)))
                cute.arch.cp_async_mbarrier_arrive_noinc(bfull + s)
                g += 1
            u += npairs

    @cute.jit
    def _forward(self, st, nt: Int32, pr: Int32, npairs: Int32):
        """Per stage: wait for this CTA's token copies, order them for the tensor core's (async
        proxy) reads, and arrive on the leader's ``full[s]``."""
        S = self.num_stages
        full, bfull = st.full.data_ptr(), st.bfull.data_ptr()
        g = Int32(0)
        u = pr
        end = nt * self.mt
        while u < end:
            for _kk in cutlass.range(0, self.kt, 1, unroll=1):
                s = g % S
                cute.arch.mbarrier_wait(bfull + s, (g // S) & 1)
                cute.arch.fence_proxy("async.shared", space="cta")
                # CTA-scope release (the CUTLASS 2-CTA gather pipelines' arrive): a cluster-scope
                # release is a MEMBAR.GPU per stage, which serializes the ring
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(full + s, Int32(0))
                g += 1
            u += npairs

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def _mma2(self, st, tiled_mma, a_layout, b_layout, sfa_layout, sfb_layout, tile_limit, nt: Int32, pr: Int32,
              npairs: Int32, rank: Int32):  # fmt: skip
        AS = self.acc_stages
        mt = self.mt
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        cute.arch.mbarrier_wait(st.tmem_ready, 0)
        acc_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=st.tmem_holding_buf
        )
        if rank == 0:
            sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
            sfb_one = cute.slice_(sfb_layout, (None, None, None, 0))
            # the scale factors of all n_tile rows; an MMA of N < n_tile reads the prefix of the SFB
            tCtSFA = cute.make_tensor(
                cute.recast_ptr(acc_ptr + self.acc_cols, dtype=cutlass.Float8E8M0FNU),
                blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC, sfa_one),
            )
            tCtSFB = cute.make_tensor(
                cute.recast_ptr(
                    acc_ptr + self.acc_cols + self.sfa_cols, dtype=cutlass.Float8E8M0FNU
                ),
                blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, SF_VEC, sfb_one),
            )
            s2t_sfa, tCsSFA, tCtSFA_s2t = _s2t_copy_and_partition2(
                st.sSFA.get_tensor(sfa_layout), tCtSFA
            )
            s2t_sfb, tCsSFB, tCtSFB_s2t = _s2t_copy_and_partition2(
                st.sSFB.get_tensor(sfb_layout), tCtSFB
            )
            sf = (tCtSFA, tCtSFB, s2t_sfa, tCsSFA, tCtSFA_s2t, s2t_sfb, tCsSFB, tCtSFB_s2t)
            g = Int32(0)
            j = Int32(0)
            u = pr
            end = nt * mt
            while u < end:
                c, _mp = self._unit(u)
                n_sel = self._tile_n(Int32(tile_limit[c]) - c * self.n_tile)
                a = j % AS
                if j >= AS:
                    cute.arch.mbarrier_wait(acc_empty + a, ((j // AS) & 1) ^ 1)
                for vi in cutlass.range_constexpr(len(self.mma_ns)):
                    if n_sel == self.mma_ns[vi]:
                        g = self._mma_unit(vi, st, a_layout, b_layout, acc_ptr, a, sf, g)
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full + a, 3, tcgen05.CtaGroup.TWO)
                j += 1
                u += npairs
        # TMEM (allocated by the pair) is freed once both CTAs' epilogues are done with it
        cute.arch.mbarrier_wait(st.tmem_free, 0)
        cute.arch.mbarrier_arrive(st.tmem_dealloc, rank ^ 1)
        cute.arch.mbarrier_wait(st.tmem_dealloc, 0)
        cute.arch.dealloc_tmem(acc_ptr, self.tmem_cols, is_two_cta=True)

    @cute.jit
    def _mma_unit(self, vi: cutlass.Constexpr, st, a_layout, b_layout, acc_ptr, a: Int32, sf,
                  g: Int32):  # fmt: skip
        """One unit's K loop with MMA N = ``mma_ns[vi]``: the accumulator, the B rows of each CTA
        (its first N / 2 smem rows) and the SFB are prefixes of the ``n_tile`` ones. Returns the
        stage counter."""
        S = self.num_stages
        nv = self.mma_ns[vi]
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        tCtSFA, tCtSFB, s2t_sfa, tCsSFA, tCtSFA_s2t, s2t_sfb, tCsSFB, tCtSFB_s2t = sf
        # a fresh TiledMma per unit: its ACCUMULATE / SF fields are set inside this K loop only
        tm = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            SF_VEC,
            tcgen05.CtaGroup.TWO,
            (PAIR_M, nv),
        )
        b_outer = b_layout.outer
        b_inner = b_layout.inner
        if cutlass.const_expr(nv != self.n_tile):
            b_v = sm100_utils.make_smem_layout_b(tm, (PAIR_M, nv, KBLK), cutlass.Float8E4M3FN, S)
            o = b_v.outer
            # the variant's rows of a stage, at the n_tile ring's stage stride
            b_outer = cute.make_layout(
                o.shape, stride=(o.stride[0], o.stride[1], o.stride[2], b_layout.outer.stride[3])
            )
            b_inner = b_v.inner
        tCrA = tm.make_fragment_A(st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner))
        tCrB = tm.make_fragment_B(st.sB.get_tensor(b_outer, swizzle=b_inner))
        tAcc = cute.make_tensor(acc_ptr + a * self.n_tile,
                                tm.make_fragment_C(tm.partition_shape_C((PAIR_M, nv))).layout)  # fmt: skip
        nkp = cute.size(tCrA, mode=[2])
        tm.set(tcgen05.Field.ACCUMULATE, False)
        for _kk in cutlass.range(0, self.kt, 1, unroll=1):
            s = g % S
            cute.arch.mbarrier_wait(full + s, (g // S) & 1)
            cute.copy(s2t_sfa, tCsSFA[(None, None, None, None, s)], tCtSFA_s2t)
            cute.copy(s2t_sfb, tCsSFB[(None, None, None, None, s)], tCtSFB_s2t)
            for kp in cutlass.range_constexpr(nkp):
                tm.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                tm.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                cute.gemm(
                    tm,
                    tAcc[(None, 0, 0)],
                    tCrA[(None, 0, kp, s)],
                    tCrB[(None, 0, kp, s)],
                    tAcc[(None, 0, 0)],
                )
                tm.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(empty + s, 3, tcgen05.CtaGroup.TWO)
            g += 1
        return g

    # ------------------------------------------------------------------ epilogue
    @cute.jit
    def _epilogue2(self, st, tiled_mma, tile_limit, h_addr: Int64, h_sf_addr: Int64, y_addr: Int64, nt: Int32,
                   pr: Int32, npairs: Int32, rank: Int32, tidx: Int32):  # fmt: skip
        """Per unit: this CTA's accumulator (its m-tile ``2 * mp + rank`` x the tile's tokens) in
        chunks of ``epi_n`` token columns through the 1-CTA FC1 / FC2 epilogue arithmetic. Each
        warp's last load of a unit releases the leader's ``acc_empty``."""
        n_tile = self.n_tile
        epi_n = self.epi_n
        n_chunks = n_tile // epi_n
        AS = self.acc_stages
        mt = self.mt
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        cute.arch.mbarrier_wait(st.tmem_ready, 0)
        acc_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=st.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tAcc_base = cute.make_tensor(
            acc_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, AS)).layout
        )
        copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(epi_n)), Float32)
        tAcc_epi = cute.flat_divide(tAcc_base[((None, None), 0, 0, None)], (MMA_M, epi_n))
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom, tAcc_epi[(None, None, 0, 0, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
        tTR_cAcc = thr_t2r.partition_D(cute.make_identity_tensor((MMA_M, epi_n)))
        rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, Float32)
        epi_base = st.sEpi.data_ptr().toint()
        rcp448 = _rcp_approx(Float32(448.0))
        warp = tidx // 32
        lane = tidx % 32
        hidden_col = 32 * warp + 4 * (lane % 8) + lane // 8

        j = Int32(0)
        u = pr
        end = nt * mt
        while u < end:
            c, mp = self._unit(u)
            mi = 2 * mp + rank
            row0 = c * n_tile
            lim = Int32(tile_limit[c])
            a = j % AS
            cute.arch.mbarrier_wait(acc_full + a, (j // AS) & 1)
            for ch in cutlass.range_constexpr(n_chunks):
                n0 = ch * epi_n
                # chunks past the tile's limit hold no stored rows (the last load is kept: it orders the
                # release of the accumulator)
                live = row0 + n0 < lim
                if live or ch == n_chunks - 1:
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, ch, a)], rAcc)
                if cutlass.const_expr(ch == n_chunks - 1):
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(acc_empty + a, Int32(0))
                if live:
                    if cutlass.const_expr(self.fc1):
                        self._epi_fc1_w(rAcc, tTR_cAcc, epi_base, rcp448, h_addr, h_sf_addr, row0 + n0, lim, mi,
                                        tidx)  # fmt: skip
                    else:
                        self._epi_fc2(
                            rAcc, tTR_cAcc, epi_base, hidden_col, y_addr, row0 + n0, lim, mi, tidx
                        )
            j += 1
            u += npairs
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(st.tmem_free)

    @cute.jit
    def _epi_fc1_w(self, rAcc, tTR_cAcc, epi_base, rcp448, h_addr: Int64, h_sf_addr: Int64, row0, lim, mi,
                   tidx):  # fmt: skip
        """FC1 epilogue of a 32-token chunk, same arithmetic as :meth:`.W4A8MoeKernel._epi_fc1`: thread ``t`` takes
        token ``t // 4`` and outputs ``16 w .. 16 w + 15`` (``w = t % 4``: the (linear, gate) pairs of
        32-row block ``w``) of the m-tile's 64; the MXFP8 block of 32 outputs is shared with thread
        ``t ^ 1`` (one shuffle), and the 16 E4M3 bytes go out in one 16-byte store. The staging reads
        are conflict-free (banks ``25 n + 8 w`` over a warp's 8 tokens x 4 blocks)."""
        for i in cutlass.range_constexpr(cute.size(rAcc)):
            r, n = tTR_cAcc[i]
            _sts_f32(epi_base + (n * _EPI_ROW2 + r + 8 * (r // 32)) * 4, rAcc[i])
        _bar_sync(_EPI_BAR, 128)
        n = tidx // 4
        w = tidx % 4
        v = cute.make_rmem_tensor((16,), Float32)
        amax = Float32(0.0)
        row = epi_base + (n * _EPI_ROW2 + 40 * w) * 4
        for pp in cutlass.range_constexpr(16):
            r_lin = (pp >> 1) + ((pp & 1) << 4)
            lin = _lds_f32(row + r_lin * 4)
            gate = _lds_f32(row + (r_lin + 8) * 4)
            val = self._situ(gate, lin)
            v[pp] = val
            amax = cute.arch.fmax(amax, cute.math.absf(val))
        amax = cute.arch.fmax(amax, _shfl_xor_f32(amax, 1))
        sf, sc = _mxfp8_scale(amax, rcp448)
        q0 = _e4m3x4(v[0] * sc, v[1] * sc, v[2] * sc, v[3] * sc)
        q1 = _e4m3x4(v[4] * sc, v[5] * sc, v[6] * sc, v[7] * sc)
        q2 = _e4m3x4(v[8] * sc, v[9] * sc, v[10] * sc, v[11] * sc)
        q3 = _e4m3x4(v[12] * sc, v[13] * sc, v[14] * sc, v[15] * sc)
        p = row0 + n
        if p < lim:
            _stg_v4(h_addr + Int64(p) * self.inter + mi * 64 + 16 * w, q0, q1, q2, q3)
            if w % 2 == 0:
                _stg_u8(h_sf_addr + Int64(p) * (self.inter // SF_VEC) + mi * 2 + w // 2, sf)
        _bar_sync(_EPI_BAR, 128)


def _s2t_copy_and_partition2(sSF: cute.Tensor, tSF: cute.Tensor):
    """tcgen05.cp (SMEM -> TMEM, ``cta_group::2``: each CTA's scale factors into its own TMEM)."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO), cutlass.Float8E8M0FNU
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


# ---------------------------------------------------------------------- launch
@functools.cache
def max_pair_ctas() -> int:
    """CTAs of the largest co-resident grid of 2-CTA clusters (one CTA per SM)."""
    pairs = num_sms() // 2
    try:
        hw = utils.HardwareInfo(torch.cuda.current_device())
        pairs = min(pairs, int(hw.get_max_active_clusters(2)))
    except (RuntimeError, ValueError, AttributeError):
        pass
    return 2 * max(1, pairs)


@functools.cache
def _compiled2(n_tile: int, inter: int, hidden: int, fc1: bool, gate_cap: float, linear_cap: float, max_stages: int,
               p2t_div: int, max_acc_stages: int, early_weights: bool):  # fmt: skip
    kernel = W4A8MoeKernel2Cta(n_tile, inter, hidden, fc1, gate_cap, linear_cap, max_stages, p2t_div,
                               max_acc_stages, early_weights)  # fmt: skip
    dev = torch.device("cuda")
    i32 = torch.zeros(64, dtype=torch.int32, device=dev)
    dummy = make_ptr(cutlass.Float4E2M1FN, 1 << 20, cute.AddressSpace.gmem, assumed_align=32)
    dummy_sf = make_ptr(cutlass.Float8E8M0FNU, 1 << 20, cute.AddressSpace.gmem, assumed_align=32)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        dummy,
        dummy_sf,
        Int32(1),
        a,
        a,
        a,
        a,
        a,
        _i32_arg(i32),
        _i32_arg(i32),
        _i32_arg(i32),
        _i32_arg(i32[:1]),
        Int32(2),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_w4a8_moe_2cta(
    w1: Optional[torch.Tensor],
    w1_sf: Optional[torch.Tensor],
    w2: Optional[torch.Tensor],
    w2_sf: Optional[torch.Tensor],
    x: Optional[torch.Tensor],
    x_sf: Optional[torch.Tensor],
    h: torch.Tensor,
    h_sf: torch.Tensor,
    y: Optional[torch.Tensor],
    tile_expert: torch.Tensor,
    tile_limit: torch.Tensor,
    p2t: torch.Tensor,
    num_tiles: torch.Tensor,
    n_tile: int = 128,
    gate_cap: float = 4.0,
    linear_cap: float = 25.0,
    max_stages: int = 16,
    p2t_div: int = 1,
    max_acc_stages: int = 2,
    early_weights: bool = False,
    grid: Optional[int] = None,
) -> None:
    """Launch FC1 (``w1`` given) or FC2 (``w2`` given) on CTA pairs, on the current stream (inputs
    not validated). Tensors as :func:`.run_w4a8_moe`; the routing tiles hold ``n_tile`` (64,
    128, 192, 224 or 256) permuted rows, ``p2t`` tokens or (``p2t_div = top_k``) expanded indices
    ``token * top_k + slot``. No workspace. ``grid``: CTAs (even; default :func:`max_pair_ctas`)."""
    fc1 = w1 is not None
    assert fc1 != (w2 is not None), "one GEMM per launch"
    w, w_sf = (w1, w1_sf) if fc1 else (w2, w2_sf)
    num_experts = w.shape[0]
    if fc1:
        inter, hidden = w.shape[1] // 2, 2 * w.shape[2]
    else:
        hidden, inter = w.shape[1], 2 * w.shape[2]
    compiled = _compiled2(int(n_tile), int(inter), int(hidden), fc1, float(gate_cap), float(linear_cap),
                          int(max_stages), int(p2t_div), int(max_acc_stages),
                          bool(early_weights) and not fc1)  # fmt: skip
    g = int(grid) if grid else max_pair_ctas()
    assert g % 2 == 0
    compiled(
        _fp4_ptr(w),
        _sf_ptr(w_sf),
        Int32(num_experts),
        Int64(x.data_ptr() if x is not None else 0),
        Int64(x_sf.data_ptr() if x_sf is not None else 0),
        Int64(h.data_ptr()),
        Int64(h_sf.data_ptr()),
        Int64(y.data_ptr() if y is not None else 0),
        _i32_arg(tile_expert),
        _i32_arg(tile_limit),
        _i32_arg(p2t),
        _i32_arg(num_tiles),
        Int32(g),
        cuda.CUstream(torch.cuda.current_stream(w.device).cuda_stream),
    )


__all__ = [
    "PAIR_M",
    "SUPPORTED_PAIR_N_TILES",
    "W4A8MoeKernel2Cta",
    "max_pair_ctas",
    "run_w4a8_moe_2cta",
]
