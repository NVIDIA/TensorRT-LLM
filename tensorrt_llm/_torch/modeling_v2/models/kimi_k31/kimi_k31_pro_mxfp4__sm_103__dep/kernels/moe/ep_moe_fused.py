# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 expert-parallel routed experts with the combine fused into the FC2 epilogue (CuTe DSL, SM100 / SM103).

Three kernels around the CTA-pair W4A8 grouped GEMM (:mod:`....kernels.moe.w4a8_moe_gemm2`) and the family's push
dispatch (:mod:`..comm.moe_a2a`), for ``W`` ranks of one EP group sharing a symmetric workspace
(``kimi_k31_pro_mxfp4__sm_103__dep/kernels/comm/moe_a2a.MoeA2AWorkspace``):

* :class:`EpDispatchKernel`: the family push dispatch with the rank's token count capped by a device word (the
  attention-DP CUDA-graph padding rows, the tail of the step, go nowhere).
* :class:`EpFc2PushKernel`: FC2 of the pair kernel whose epilogue stores each expert output row (BF16, the bits the
  pair kernel writes to its ``y``) straight into the slot buffer of the token's own rank over NVLink: row
  ``t * top_k + k`` of rank ``s``'s slot region for permuted row ``p`` with ``p2t[p] = (s * T + t) * top_k + k`` (the
  runner sort's expanded indices). The pushes leave the epilogue as the units finish, so the combine traffic runs
  under the FC2 weight stream. Each CTA counts itself done once its stores are issued; the grid's last CTA fences at
  system scope and sets this rank's flag on every peer to the call count (every pushing thread fences its peer stores
  at system scope before its CTA counts itself).
* :class:`EpCombineReduceKernel`: on the token's rank, waits for every peer's flag, then sums each token's ``top_k``
  slot rows bit for bit like the separate path (the finalize on each expert rank, then the one-sided all-to-all
  combine): per destination rank, in the order the ranks first appear among the token's experts, the FP32 slot-order
  sum of ``w * y`` over the slots that rank owns, rounded to BF16; then the engine combine's FP32 tree over those
  rank partials, rounded to BF16 once.

Per call each kernel reads the call count from the workspace; flags carry it, so they never need resetting. A rank
writes a peer's slot rows only after that peer finished its previous reduce (its dispatch of this layer waits for
every rank's dispatch, which follows each rank's previous reduce on its stream).
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.moe_finalize import (
    _f32_bits as _f32_of_bits,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.moe_finalize import (
    _hi_f32,
    _lo_f32,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.w4a8_moe_gemm import (
    _EPI_BAR,
    HIDDEN,
    MMA_M,
    SF_VEC,
    _bar_sync,
    _bf16x2,
    _fp4_ptr,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _i32_arg,
    _lds_v4,
    _sf_ptr,
    _sts_b16,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.w4a8_moe_gemm2 import (
    W4A8MoeKernel2Cta,
    max_pair_ctas,
)

from ..comm.moe_a2a import (
    DCOUNT,
    DEPOCH,
    DFLAGS,
    MAX_SOURCES,
    MoeA2ADispatchKernel,
    _arrive,
    _rank_mask,
    _wait_flags,
    add2_bf16x2,
    add_f32,
    atom_add_acq_rel_gpu,
    fence_acq_rel_sys,
    ld_acquire_sys_u32,
    ld_cg_v4,
    ld_nc_u64,
    ld_relaxed_u32,
    ld_v4,
    pack_bf16x2,
    st_relaxed_u32,
    st_v4,
    tree8_bf16x2,
)

#: Flag words of the fused path in the workspace's flag area (uint32 words, each on its own 128-byte line, clear of
#: the all-to-all kernels' words): the call count, the FC2 CTA count, the FC2-done flags (rank ``r`` at
#: ``DONE_FLAGS + r * FLAG_STRIDE``: no two ranks write one line).
CALL_WORD, CTA_COUNT_WORD, DONE_FLAGS, FLAG_STRIDE = 288, 320, 40960, 32


@cute.jit
def _placed(place: Int64, e: Int32, num_experts: cutlass.Constexpr):
    """Physical expert of routed expert ``e`` (ids outside ``[0, num_experts)`` unchanged)."""
    p = e
    if e >= 0:
        if e < num_experts:
            p = ld_b32(place + Int64(e) * 4)
    return p


@cute.jit
def _placed_rank_mask(row: Int64, place: Int64, vecs: cutlass.Constexpr, num_experts: cutlass.Constexpr,
                      experts_per_rank: cutlass.Constexpr):  # fmt: skip
    """Bit ``d`` set when one of the expert ids at ``row`` maps to a physical slot on rank ``d``."""
    mask = Int32(0)
    for q in cutlass.range_constexpr(vecs):
        w = ld_v4(row + q * 16)
        for k in cutlass.range_constexpr(4):
            s = w[k]
            if s >= 0:
                if s < num_experts:
                    mask = mask | (Int32(1) << (ld_b32(place + Int64(s) * 4) // experts_per_rank))
    return mask


class EpDispatchKernel(MoeA2ADispatchKernel):
    """The family push dispatch (the geometry of :class:`..comm.moe_a2a.MoeA2ADispatchKernel`; the rows, flags and
    order of that module's docstring) with this rank's token count capped by a device word: tokens at or past
    ``min(ntok, real_rows[0])`` are padding (the attention-DP CUDA-graph padding requests, the tail rows of the step)
    and go nowhere, like the rows past ``ntok``. ``real_rows`` 0 = no cap. ``place`` (int32 ``[num_experts]`` device
    table, 0 = identity) maps each routed (logical) expert to the physical expert slot
    ``rank * experts_per_rank + slot`` holding its weights: the token goes to the ranks of its experts' slots and its
    received id row carries the physical ids."""

    @cute.jit
    def __call__(
        self,
        hid: Int64,
        sf: Int64,
        ids: Int64,
        wts: Int64,
        peers: Int64,
        base: Int64,
        offs: Int64,
        ntok: Int32,
        tmax: Int32,
        rank: Int32,
        real_rows: Int64,
        place: Int64,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            hid, sf, ids, wts, peers, base, offs, ntok, tmax, rank, real_rows, place
        ).launch(grid=(ctas, 1, 1), block=(self.threads, 1, 1), smem=128, stream=stream)

    @cute.kernel
    def kernel(
        self,
        hid: Int64,
        sf: Int64,
        ids: Int64,
        wts: Int64,
        peers: Int64,
        base: Int64,
        offs: Int64,
        ntok_max: Int32,
        tmax: Int32,
        rank: Int32,
        real_rows: Int64,
        place: Int64,
    ):
        W, G = self.W, self.G
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        warp = tid // 32
        lane = tid - warp * 32
        wpb = self.threads // 32
        sitem = cutlass.utils.SmemAllocator().allocate_tensor(
            Int32, cute.make_layout((1,)), byte_alignment=16
        )
        ntok = ntok_max
        if real_rows != 0:
            ntok = cutlass.min(ntok_max, ld_b32(real_rows))
        epoch = ld_relaxed_u32(base + Int64(DEPOCH * 4)) + 1
        off_h = ld_nc_u64(offs)
        off_s = ld_nc_u64(offs + 8)
        off_i = ld_nc_u64(offs + 16)
        off_f = ld_nc_u64(offs + 24)
        hw = cute.make_rmem_tensor((self.HJ * 4,), Int32)
        xw = cute.make_rmem_tensor((4,), Int32)
        for it in cutlass.range(bidx * wpb + warp, tmax * G, nblk * wpb, unroll=1):
            t = it // G
            g = it - t * G
            row = Int64(rank) * Int64(tmax) + Int64(t)
            mask = Int32(0)
            if t < ntok:
                if place != 0:
                    mask = _placed_rank_mask(
                        ids + Int64(t) * self.IB, place, self.IV, self.NE, self.EPR
                    )
                else:
                    mask = _rank_mask(ids + Int64(t) * self.IB, self.IV, self.NE, self.EPR)
            want = Int32(0)
            for k in cutlass.range_constexpr(W // G):
                want = want | ((mask >> ((rank + 1 + g + k * G) % W)) & 1)
            if want != 0:
                for j in cutlass.range_constexpr(self.HJ):
                    v = lane + 32 * j
                    if v < self.HV:
                        wh = ld_v4(hid + Int64(t) * self.HB + Int64(v) * 16)
                        for e in cutlass.range_constexpr(4):
                            hw[j * 4 + e] = wh[e]
                if lane < self.SV:
                    ws = ld_v4(sf + Int64(t) * self.SB + Int64(lane) * 16)
                    for e in cutlass.range_constexpr(4):
                        xw[e] = ws[e]
                elif lane < self.SV + self.IV:
                    wi = ld_v4(ids + Int64(t) * self.IB + Int64(lane - self.SV) * 16)
                    for e in cutlass.range_constexpr(4):
                        xw[e] = wi[e]
                        if place != 0:
                            xw[e] = _placed(place, wi[e], self.NE)
                elif lane < self.SV + self.IV + self.FV:
                    wf = ld_v4(wts + Int64(t) * self.FB + Int64(lane - self.SV - self.IV) * 16)
                    for e in cutlass.range_constexpr(4):
                        xw[e] = wf[e]
            for k in cutlass.range_constexpr(W // G):
                d = (rank + 1 + g + k * G) % W
                peer = ld_nc_u64(peers + Int64(d) * 8)
                if ((mask >> d) & 1) != 0:
                    for j in cutlass.range_constexpr(self.HJ):
                        v = lane + 32 * j
                        if v < self.HV:
                            st_v4(peer + off_h + row * self.HB + Int64(v) * 16, hw[j * 4], hw[j * 4 + 1],
                                  hw[j * 4 + 2], hw[j * 4 + 3])  # fmt: skip
                    if lane < self.SV:
                        st_v4(
                            peer + off_s + row * self.SB + Int64(lane) * 16,
                            xw[0],
                            xw[1],
                            xw[2],
                            xw[3],
                        )
                    elif lane < self.SV + self.IV:
                        st_v4(peer + off_i + row * self.IB + Int64(lane - self.SV) * 16, xw[0], xw[1], xw[2],
                              xw[3])  # fmt: skip
                    elif lane < self.SV + self.IV + self.FV:
                        st_v4(peer + off_f + row * self.FB + Int64(lane - self.SV - self.IV) * 16, xw[0], xw[1],
                              xw[2], xw[3])  # fmt: skip
                else:
                    if lane < self.IV:
                        m1 = Int32(-1)
                        st_v4(peer + off_i + row * self.IB + Int64(lane) * 16, m1, m1, m1, m1)
        ticket = _arrive(base, peers, rank, epoch, nblk, tid, sitem, DCOUNT, DFLAGS, W)
        if ticket == nblk - 1:
            _wait_flags(base, epoch, tid, DFLAGS, W)
            if tid == 0:
                st_relaxed_u32(base + Int64(DEPOCH * 4), epoch)


class EpFc2PushKernel(W4A8MoeKernel2Cta):
    """FC2 of :class:`.W4A8MoeKernel2Cta` with the push epilogue and the done flags (see the module docstring).

    Args (beyond the pair kernel's FC2 ones): ``world`` ranks, ``top_k`` slots per token.
    """

    def __init__(self, n_tile: int, inter: int, world: int, top_k: int = 16, hidden: int = HIDDEN,
                 max_stages: int = 16, max_acc_stages: int = 3):  # fmt: skip
        super().__init__(
            n_tile,
            inter,
            hidden,
            fc1=False,
            max_stages=max_stages,
            max_acc_stages=max_acc_stages,
            early_weights=True,
        )
        assert 2 <= world <= 32 and top_k & (top_k - 1) == 0
        self.world = world
        self.top_k = top_k
        self.k_log2 = top_k.bit_length() - 1

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        w_sf_ptr: cute.Pointer,
        num_experts: Int32,
        h_addr: Int64,
        h_sf_addr: Int64,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        p2t: cute.Tensor,
        num_tiles: cute.Tensor,
        peers: Int64,
        base: Int64,
        slot_off: Int64,
        tmax: Int32,
        rank: Int32,
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
        a = cute.make_tensor(
            w_ptr, cute.make_ordered_layout((self.hidden, self.inter, num_experts), order=(1, 0, 2))
        )
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
            ticket: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
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
            h_addr,
            h_sf_addr,
            tile_expert,
            tile_limit,
            p2t,
            num_tiles,
            peers,
            base,
            slot_off,
            tmax,
            rank,
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
        h_addr: Int64,
        h_sf_addr: Int64,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        p2t: cute.Tensor,
        num_tiles: cute.Tensor,
        peers: Int64,
        base: Int64,
        slot_off: Int64,
        tmax: Int32,
        my_rank: Int32,
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
        # the push epilogue's routing: expanded index -> (rank, token, slot) and the ranks' slot regions
        self._push = (p2t, peers, slot_off, tmax)

        st = utils.SmemAllocator().allocate(self.shared_storage)
        if warp == self.tma_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_sfa)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(S):
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
        nt = Int32(num_tiles[0])
        epoch = ld_relaxed_u32(base + Int64(CALL_WORD * 4)) + 1
        zero = Int64(0)
        if warp == self.tma_warp:
            self._produce_weights2(st, tiled_mma, tma_a, mA, tma_sfa, mSFA, a_layout, sfa_layout, cl, tile_expert,
                                   nt, pr, npairs, rank)  # fmt: skip
        elif warp >= self.gather_warp:
            _griddepcontrol_wait()
            self._produce_tokens2(st, zero, zero, h_addr, h_sf_addr, tile_limit, p2t, nt, pr, npairs, rank,
                                  warp - self.gather_warp)  # fmt: skip
        elif warp == self.xfer_warp:
            self._forward(st, nt, pr, npairs)
        elif warp == self.mma_warp:
            self._mma2(st, tiled_mma, a_layout, b_layout, sfa_layout, sfb_layout, tile_limit, nt, pr, npairs,
                       rank)  # fmt: skip
        elif warp < self.epi_warps:
            _griddepcontrol_wait()
            self._epilogue2(st, tiled_mma, tile_limit, h_addr, h_sf_addr, zero, nt, pr, npairs, rank,
                            tidx)  # fmt: skip
            # every pushing thread orders its peer stores at system scope before the CTA counts itself done
            fence_acq_rel_sys()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # this CTA's pushes are issued: count it; the grid's last CTA publishes to every rank
        tk = st.ticket.get_tensor(cute.make_layout((4,)))
        cute.arch.barrier()
        if tidx == 0:
            tk[0] = atom_add_acq_rel_gpu(base + Int64(CTA_COUNT_WORD * 4), Int32(1))
        cute.arch.barrier()
        if tk[0] == Int32(nctas_) - 1:
            if tidx < self.world:
                fence_acq_rel_sys()
                apeer = ld_nc_u64(peers + Int64(tidx) * 8)
                st_relaxed_u32(apeer + Int64((DONE_FLAGS + my_rank * FLAG_STRIDE) * 4), epoch)
            if tidx == 0:
                st_relaxed_u32(base + Int64(CTA_COUNT_WORD * 4), Int32(0))
                st_relaxed_u32(base + Int64(CALL_WORD * 4), epoch)
        _griddepcontrol_launch_dependents()

    @cute.jit
    def _epi_fc2(self, rAcc, tTR_cAcc, epi_base, hidden_col, y_addr: Int64, row0, lim, mi, tidx):
        """The pair kernel's FC2 epilogue with each 16-byte vector of row ``p`` stored into its token's slot row on
        the token's rank (``p2t[p]`` = expanded index)."""
        p2t, peers, slot_off, tmax = self._push
        epi_n = self.epi_n
        for i in cutlass.range_constexpr(cute.size(rAcc)):
            _r, n = tTR_cAcc[i]
            _sts_b16(epi_base + (n * MMA_M + hidden_col) * 2, _bf16x2(rAcc[i], Float32(0.0)))
        _bar_sync(_EPI_BAR, 128)
        for rd in cutlass.range_constexpr(epi_n * 16 // 128):
            q = tidx + 128 * rd
            n = q // 16
            c16 = q % 16
            p = row0 + n
            if p < lim:
                x = Int32(p2t[p])
                g = x >> self.k_log2
                k = x - (g << self.k_log2)
                s = g // tmax
                t = g - s * tmax
                dst = ld_nc_u64(peers + Int64(s) * 8) + slot_off
                v0, v1, v2, v3 = _lds_v4(epi_base + (n * MMA_M + 8 * c16) * 2)
                st_v4(
                    dst
                    + ((Int64(t) * self.top_k + Int64(k)) * self.hidden + mi * MMA_M + 8 * c16) * 2,
                    v0,
                    v1,
                    v2,
                    v3,
                )
        _bar_sync(_EPI_BAR, 128)


class EpCombineReduceKernel:
    """The token-side sum of the slot rows (see the module docstring).

    Warp items (token, 32-vector column slice), one 16-byte vector per lane. Per item the warp ranks the token's
    slots once: each slot's rank group (the order its rank first appears), its position in the group-major,
    slot-ordered walk, the walk's group starts / ends. Per vector a lane loads the 16 slot vectors in walk order and
    runs the walk: the finalize's FP32 ``acc + w * y`` from zero at each group start, the BF16 rank partial at each
    group end into its group's shared-memory slot; then the engine combine's tree over the partials (missing ones
    are zero).
    """

    def __init__(
        self, world: int, top_k: int, num_experts: int, hidden: int = HIDDEN, threads: int = 128
    ):
        if not 2 <= world <= 32 or num_experts % world or top_k != 16 or hidden % 256:
            raise ValueError("reduce serves top-16 over 2..32 ranks and 256-column multiples")
        self.W, self.K, self.NE, self.EPR = world, top_k, num_experts, num_experts // world
        self.H, self.OV = hidden, hidden // 8
        self.SL = self.OV // 32
        self.threads = threads
        self.wpb = threads // 32

    @cute.jit
    def __call__(
        self,
        ids: Int64,
        wts: Int64,
        out: Int64,
        base: Int64,
        slot_off: Int64,
        ntok: Int32,
        real_rows: Int64,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(ids, wts, out, base, slot_off, ntok, real_rows).launch(
            grid=(ctas, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.wpb * MAX_SOURCES * 32 * 16,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        ids: Int64,
        wts: Int64,
        out: Int64,
        base: Int64,
        slot_off: Int64,
        ntok: Int32,
        real_rows: Int64,
    ):
        W, K, SL = self.W, self.K, self.SL
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        warp = tid // 32
        lane = tid - warp * 32
        wpb = self.wpb
        smem = cutlass.utils.SmemAllocator().allocate_tensor(
            Int32, cute.make_layout((wpb * MAX_SOURCES * 32 * 4,)), byte_alignment=16
        )
        part = smem.iterator.toint() + warp * (MAX_SOURCES * 32 * 16) + lane * 16
        _griddepcontrol_wait()
        epoch = ld_relaxed_u32(base + Int64(CALL_WORD * 4))
        if tid < W:
            wflag = base + Int64((DONE_FLAGS + tid * FLAG_STRIDE) * 4)
            wgot = ld_acquire_sys_u32(wflag)
            while wgot != epoch:
                wgot = ld_acquire_sys_u32(wflag)
        cute.arch.barrier()
        slots = base + slot_off
        wp = cute.make_rmem_tensor((K,), Float32)
        yv = cute.make_rmem_tensor((K * 4,), Int32)
        acc = cute.make_rmem_tensor((8,), Float32)
        below = (Int32(1) << lane) - 1
        nreal = ntok
        if real_rows != 0:
            nreal = cutlass.min(ntok, ld_b32(real_rows))
        # padding tokens (at or past the real rows) were dispatched nowhere: their output is zero
        for it in cutlass.range(nreal * SL + bidx * wpb + warp, ntok * SL, nblk * wpb, unroll=1):
            t = it // SL
            z = Int32(0)
            st_v4(out + Int64(t) * (self.H * 2) + Int64((it - t * SL) * 32 + lane) * 16, z, z, z, z)
        for it in cutlass.range(bidx * wpb + warp, nreal * SL, nblk * wpb, unroll=1):
            t = it // SL
            sl = it - t * SL
            vb = sl * 32 + lane
            # lane k < 16 ranks slot k: its rank, the first slot of that rank (its group's start), the group
            # (first-appearance order), the slot's walk position (earlier groups' slots, then its group's earlier slots)
            kl = lane & (K - 1)
            rnk = ld_b32(ids + Int64(t) * (K * 4) + Int64(kl) * 4) // self.EPR
            wbits = ld_u16(wts + Int64(t) * (K * 2) + Int64(kl) * 2)
            rnk = cutlass.select_(lane < K, rnk, Int32(64) + lane)
            same = match_any(rnk)
            f = lowest_bit(same)
            is_first = f == lane
            firsts = ballot(is_first) & Int32((1 << K) - 1)
            g = popc(firsts & ((Int32(1) << f) - 1))
            n = popc(firsts)
            pos = popc(same & below)
            for j2 in cutlass.range_constexpr(K):
                fj = shfl_idx(f, j2)
                pos = pos + cutlass.select_(fj < f, Int32(1), Int32(0))
            mine = lane < K
            perm_lo = redux_or(
                cutlass.select_(mine & (pos < 8), Int32(lane) << (pos * 4), Int32(0))
            )
            perm_hi = redux_or(
                cutlass.select_(mine & (pos >= 8), Int32(lane) << ((pos - 8) * 4), Int32(0))
            )
            g_lo = redux_or(cutlass.select_(mine & (pos < 8), g << (pos * 4), Int32(0)))
            g_hi = redux_or(cutlass.select_(mine & (pos >= 8), g << ((pos - 8) * 4), Int32(0)))
            starts = redux_or(cutlass.select_(mine & is_first, Int32(1) << pos, Int32(0)))
            ends = (starts >> 1) | Int32(1 << (K - 1))
            wl = _f32_of_bits(wbits << 16)
            perm = (Int64(perm_hi) << 32) | (Int64(perm_lo) & Int64(0xFFFFFFFF))
            gpos = (Int64(g_hi) << 32) | (Int64(g_lo) & Int64(0xFFFFFFFF))
            for j in cutlass.range_constexpr(K):
                wp[j] = shfl_idx_f32(wl, Int32((perm >> (4 * j)) & 15))
            row = slots + Int64(t) * (K * self.H * 2) + Int64(vb) * 16
            for j in cutlass.range_constexpr(K):
                kj = (perm >> (4 * j)) & 15
                v = ld_cg_v4(row + kj * (self.H * 2))
                for e in cutlass.range_constexpr(4):
                    yv[j * 4 + e] = v[e]
            for j in cutlass.range_constexpr(K):
                st = ((starts >> j) & 1) != 0
                w = wp[j]
                for e in cutlass.range_constexpr(4):
                    a_lo = cutlass.select_(st, Float32(0.0), acc[2 * e])
                    a_hi = cutlass.select_(st, Float32(0.0), acc[2 * e + 1])
                    acc[2 * e] = a_lo + w * _lo_f32(yv[j * 4 + e])
                    acc[2 * e + 1] = a_hi + w * _hi_f32(yv[j * 4 + e])
                if ((ends >> j) & 1) != 0:
                    gj = Int32((gpos >> (4 * j)) & 15)
                    _sts_v4_b32(
                        part + gj * (32 * 16),
                        _bf16x2(acc[0], acc[1]),
                        _bf16x2(acc[2], acc[3]),
                        _bf16x2(acc[4], acc[5]),
                        _bf16x2(acc[6], acc[7]),
                    )
            pw = []
            for si in cutlass.range_constexpr(MAX_SOURCES):
                ok = Int32(si) < n
                pv = _lds_v4(part + si * (32 * 16))
                pw.append([cutlass.select_(ok, pv[e], Int32(0)) for e in range(4)])
            res = [Int32(0)] * 4
            for e in cutlass.range_constexpr(4):
                a_lo, a_hi = tree8_bf16x2(*[pw[si][e] for si in range(8)])
                r_lo = a_lo
                r_hi = a_hi
                if n > 8:
                    b_lo, b_hi = tree8_bf16x2(*[pw[si][e] for si in range(8, 16)])
                    r_lo = add_f32(a_lo, b_lo)
                    r_hi = add_f32(a_hi, b_hi)
                if n <= 2:
                    c_lo, c_hi = add2_bf16x2(pw[0][e], pw[1][e])
                    r_lo = c_lo
                    r_hi = c_hi
                res[e] = pack_bf16x2(r_lo, r_hi)
            st_v4(out + Int64(t) * (self.H * 2) + Int64(vb) * 16, res[0], res[1], res[2], res[3])
        _griddepcontrol_launch_dependents()


def _asm1(res, ops, text, cons, loc, ip):
    return llvm.inline_asm(res, ops, text, cons, has_side_effects=True, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm1(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_u16(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                       "{\n.reg .b16 h;\nld.global.b16 h, [$1];\ncvt.u32.u16 $0, h;\n}", "=r,l", loc, ip))  # fmt: skip


@dsl_user_op
def match_any(v, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip)], "match.any.sync.b32 $0, $1, 0xffffffff;",
                       "=r,r", loc, ip))  # fmt: skip


@dsl_user_op
def lowest_bit(v, *, loc=None, ip=None):
    """Bit index of the lowest set bit of a nonzero word."""
    return Int32(_asm1(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip)],
                       "{\n.reg .b32 t;\nneg.s32 t, $1;\nand.b32 t, t, $1;\nbfind.u32 $0, t;\n}", "=r,r", loc,
                       ip))  # fmt: skip


@dsl_user_op
def ballot(p, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(cutlass.select_(p, Int32(1), Int32(0))).ir_value(loc=loc, ip=ip)],
                       "{\n.reg .pred q;\nsetp.ne.b32 q, $1, 0;\nvote.sync.ballot.b32 $0, q, 0xffffffff;\n}",
                       "=r,r", loc, ip))  # fmt: skip


@dsl_user_op
def popc(v, *, loc=None, ip=None):
    return Int32(
        _asm1(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r", loc, ip)
    )


@dsl_user_op
def redux_or(v, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip)], "redux.sync.or.b32 $0, $1, 0xffffffff;",
                       "=r,r", loc, ip))  # fmt: skip


@dsl_user_op
def shfl_idx(v, src, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip), Int32(src).ir_value(loc=loc, ip=ip)],
                       "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;", "=r,r,r", loc, ip))  # fmt: skip


@dsl_user_op
def shfl_idx_f32(v, src, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(v).ir_value(loc=loc, ip=ip), Int32(src).ir_value(loc=loc, ip=ip)],
                         "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;", "=f,f,r", loc, ip))  # fmt: skip


@dsl_user_op
def _sts_v4_b32(addr, a, b, c, d, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------- launch
@functools.cache
def _compiled_fc2(n_tile: int, inter: int, world: int, top_k: int, hidden: int, max_stages: int,
                  max_acc_stages: int):  # fmt: skip
    kernel = EpFc2PushKernel(n_tile, inter, world, top_k, hidden, max_stages, max_acc_stages)  # fmt: skip
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
        _i32_arg(i32),
        _i32_arg(i32),
        _i32_arg(i32),
        _i32_arg(i32[:1]),
        a,
        a,
        a,
        Int32(1),
        Int32(0),
        Int32(2),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_ep_fc2_push(
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    h: torch.Tensor,
    h_sf: torch.Tensor,
    tile_expert: torch.Tensor,
    tile_limit: torch.Tensor,
    p2x: torch.Tensor,
    num_tiles: torch.Tensor,
    peers: torch.Tensor,
    base: int,
    slot_off: int,
    tmax: int,
    rank: int,
    world: int,
    n_tile: int = 128,
    top_k: int = 16,
    max_stages: int = 16,
    max_acc_stages: int = 3,
) -> None:
    """FC2 on CTA pairs pushing every expert output row into its token's slot row (``p2x``: the runner sort's
    permuted row -> expanded index ``(s * tmax + t) * top_k + k`` of the dispatched rows), then publishing this rank's
    done flag on every rank. ``peers``: int64 ``[world]`` base addresses of the ranks' workspace mappings, ``base``
    this rank's, ``slot_off`` the slot region's offset. On the current stream."""
    hidden, inter = w2.shape[1], 2 * w2.shape[2]
    compiled = _compiled_fc2(int(n_tile), int(inter), int(world), int(top_k), int(hidden), int(max_stages),
                             int(max_acc_stages))  # fmt: skip
    compiled(
        _fp4_ptr(w2),
        _sf_ptr(w2_sf),
        Int32(w2.shape[0]),
        Int64(h.data_ptr()),
        Int64(h_sf.data_ptr()),
        _i32_arg(tile_expert),
        _i32_arg(tile_limit),
        _i32_arg(p2x),
        _i32_arg(num_tiles),
        Int64(peers.data_ptr()),
        Int64(base),
        Int64(slot_off),
        Int32(tmax),
        Int32(rank),
        Int32(max_pair_ctas()),
        cuda.CUstream(torch.cuda.current_stream(w2.device).cuda_stream),
    )


@functools.cache
def _compiled_reduce(world: int, top_k: int, num_experts: int, hidden: int, threads: int):
    kernel = EpCombineReduceKernel(world, top_k, num_experts, hidden, threads)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        Int32(1),
        a,
        Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _compiled_dispatch(world: int, top_k: int, num_experts: int, hidden_bytes: int, sf_bytes: int, weight_bytes: int,
                       groups: int, threads: int):  # fmt: skip
    kernel = EpDispatchKernel(
        world, top_k, num_experts, hidden_bytes, sf_bytes, weight_bytes, groups, threads
    )
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(1),
        Int32(0),
        a,
        a,
        Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_ep_dispatch(
    hidden: torch.Tensor,
    hidden_sf: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    peers: torch.Tensor,
    base: int,
    offs: torch.Tensor,
    tmax: int,
    rank: int,
    world: int,
    num_experts: int,
    real_rows: torch.Tensor | None = None,
    place: torch.Tensor | None = None,
    groups: int = 4,
    threads: int = 512,
) -> None:
    """The family push dispatch of this rank's ``[ntok, ...]`` rows (activation, scales, int32 expert ids, routing
    weights) into every rank's receive arrays (``offs``: the workspace's region offsets), tokens at or past
    ``real_rows[0]`` (int32 device word, optional) left out like the rows past ``ntok``; ``place`` (int32
    ``[num_experts]``, optional): routed expert -> physical expert slot (see :class:`EpDispatchKernel`). On the
    current stream."""
    ntok, top_k = expert_ids.shape
    fn = _compiled_dispatch(int(world), int(top_k), int(num_experts), hidden.shape[1] * hidden.element_size(),
                            hidden_sf.shape[1] * hidden_sf.element_size(), weights.shape[1] * weights.element_size(),
                            int(groups), int(threads))  # fmt: skip
    fn(
        Int64(hidden.data_ptr()),
        Int64(hidden_sf.data_ptr()),
        Int64(expert_ids.data_ptr()),
        Int64(weights.data_ptr()),
        Int64(peers.data_ptr()),
        Int64(base),
        Int64(offs.data_ptr()),
        Int32(ntok),
        Int32(tmax),
        Int32(rank),
        Int64(real_rows.data_ptr() if real_rows is not None else 0),
        Int64(place.data_ptr() if place is not None else 0),
        Int32(torch.cuda.get_device_properties(hidden.device).multi_processor_count),
        cuda.CUstream(torch.cuda.current_stream(hidden.device).cuda_stream),
    )


def run_ep_combine_reduce(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    base: int,
    slot_off: int,
    world: int,
    num_experts: int,
    threads: int = 128,
    real_rows: torch.Tensor | None = None,
) -> None:
    """``out [ntok, hidden]`` BF16 = the sum of this rank's tokens' slot rows (see the module docstring), after every
    rank's FC2 flag; ``topk_ids`` int32 / ``topk_weights`` BF16 ``[ntok, 16]`` of this rank's tokens. ``real_rows``
    (int32 device word, optional): tokens at or past it are padding the dispatch skipped; their rows are zero."""
    ntok, top_k = topk_ids.shape
    if ntok == 0:
        return
    assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.bfloat16
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous() and out.is_contiguous()
    hidden = int(out.shape[1])
    fn = _compiled_reduce(int(world), int(top_k), int(num_experts), hidden, int(threads))
    n_items = ntok * (hidden // 256)
    wpb = threads // 32
    sms = torch.cuda.get_device_properties(out.device).multi_processor_count
    grid = min(4 * sms, -(-n_items // wpb))
    fn(
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(out.data_ptr()),
        Int64(base),
        Int64(slot_off),
        Int32(ntok),
        Int64(real_rows.data_ptr() if real_rows is not None else 0),
        Int32(grid),
        cuda.CUstream(torch.cuda.current_stream(out.device).cuda_stream),
    )


__all__ = [
    "FLAG_STRIDE",
    "CTA_COUNT_WORD",
    "CALL_WORD",
    "DONE_FLAGS",
    "EpDispatchKernel",
    "EpFc2PushKernel",
    "EpCombineReduceKernel",
    "run_ep_dispatch",
    "run_ep_fc2_push",
    "run_ep_combine_reduce",
]
