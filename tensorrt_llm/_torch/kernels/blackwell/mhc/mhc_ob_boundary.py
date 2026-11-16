# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention ``o_b``, its TP all-reduce and the mHC boundary in one launch, bit for bit the LowM
GEMM then ``MhcAllReduceBoundaryKernel``: ``o_b`` CTAs push tiles, token CTAs finish boundaries."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass.utils import SmemAllocator

from ..cute_ptx_utils import mma_sync
from ..lowm_fp8_mma_gemm import CHUNK, GROUP, N_TILE, _fp8x4_to_bf16x4_s2, _fp8x4_to_bf16x4_scaled
from ..utils import griddepcontrol_launch_dependents, griddepcontrol_wait
from .mhc_boundary import bf16x2_hi, bf16x2_lo, ldg_f32
from .mhc_split_boundary import (
    _NEG_ZERO,
    HC,
    VEC,
    MhcAllReduceBoundaryKernel,
    _arrive_barrier_1,
    _atom_release_add_if_u32,
    _bf16x2_neg_zero_to_zero,
    _fence_acq_rel_gpu,
    _ld_volatile_s32,
    _ld_volatile_s64,
    _ld_volatile_v4,
    _st_v4,
)


def _round16(nbytes: int) -> int:
    return (nbytes + 15) // 16 * 16


def ob_boundary_smem_bytes(
    hidden: int, k_warps: int, ctas_per_token: int = 2, n_tiles: int = 2, mbar_exchange: bool = True
) -> int:
    """Dynamic shared memory of one CTA: the allocations of both roles, which do not overlap."""
    warps = hidden // VEC // ctas_per_token // 32
    groups = (warps - 1) // k_warps
    cols_group = N_TILE * n_tiles
    ob_role = (
        _round16(groups * k_warps * 16 * cols_group * 4)
        + _round16(16 * cols_group * groups * 2)
        + 16
    )
    token_role = _round16(warps * 4) + (16 if mbar_exchange else 0)
    if ctas_per_token > 1:
        token_role += _round16(ctas_per_token * warps * 4)
    return ob_role + token_role + 128


def ob_boundary_tiles(
    hidden: int,
    k_extent: int,
    k_warps: int,
    smem_limit: int,
    ctas_per_token: int = 2,
    n_tiles: int = 2,
) -> bool:
    """Whether :class:`ObAllReduceBoundaryKernel` tiles an ``o_b`` of ``hidden`` x ``k_extent`` (the
    constructors' shape checks) within ``smem_limit`` bytes of shared memory per CTA."""
    threads = hidden // VEC
    if hidden % (32 * VEC) or threads > 1024 or threads % (32 * ctas_per_token):
        return False
    if k_warps < 1 or k_extent % (CHUNK * k_warps):
        return False
    groups = (threads // ctas_per_token // 32 - 1) // k_warps
    cols_cta = N_TILE * n_tiles * groups
    return (
        groups > 0
        and hidden % cols_cta == 0
        and (hidden // cols_cta) % ctas_per_token == 0
        and ob_boundary_smem_bytes(hidden, k_warps, ctas_per_token, n_tiles) <= smem_limit
    )


class ObAllReduceBoundaryKernel(MhcAllReduceBoundaryKernel):
    """``o_b`` + one-shot all-reduce + mHC boundary of ``num_rows`` (<= 16) tokens: A [M, K] BF16, B
    [H, K] E4M3, S [H, K/32] UE8M0 (linear); the rest as :class:`MhcAllReduceBoundaryKernel`.

    With ``mnnvl`` an ``o_b`` CTA also stores each pushed vector into this rank's own slot, so the
    token CTAs here need not wait for its multicast copy."""

    def __init__(
        self,
        *,
        hidden: int,
        k_extent: int,
        num_rows: int,
        k_warps: int,
        norm_eps: float,
        nranks: int,
        use_pdl: bool,
        ctas_per_token: int = 2,
        n_tiles: int = 2,
        mnnvl: bool = False,
        quant: str | None = None,
        mbar_exchange: bool = False,
        bf16_dequant: bool = False,
    ):
        super().__init__(
            hidden=hidden,
            norm_eps=norm_eps,
            nranks=nranks,
            use_pdl=use_pdl,
            ctas_per_token=ctas_per_token,
            mnnvl=mnnvl,
            quant=quant,
            mbar_exchange=mbar_exchange,
        )
        if not 1 <= num_rows <= 16:
            raise ValueError(f"num_rows={num_rows} must be in [1, 16]")
        if k_extent % (CHUNK * k_warps):
            raise ValueError(f"K={k_extent} must be a multiple of {CHUNK * k_warps}")
        if n_tiles not in (1, 2, 4):
            raise ValueError(f"n_tiles={n_tiles} must be 1, 2 or 4")
        warps = self.threads_cta // 32
        if warps - 1 < k_warps:
            raise ValueError(
                f"{warps} warps per CTA cannot hold k_warps={k_warps} + a control warp"
            )
        self.k_extent = k_extent
        self.num_rows = num_rows
        # K splits over k_warps as in the GEMM kernel (the bits depend on it, not on n_tiles); an
        # o_b CTA holds ``groups`` of k_warps warps and a control warp for the Lamport state.
        self.k_warps = k_warps
        self.chunks = k_extent // (CHUNK * k_warps)
        self.n_tiles = n_tiles
        self.bf16_dequant = bf16_dequant
        self.groups = (warps - 1) // k_warps
        self.cols_group = N_TILE * n_tiles
        self.cols_cta = self.cols_group * self.groups
        if hidden % self.cols_cta or (hidden // self.cols_cta) % ctas_per_token:
            raise ValueError(
                f"hidden={hidden} must split into whole clusters of {ctas_per_token} CTAs of "
                f"{self.cols_cta} columns"
            )
        self.ob_ctas = hidden // self.cols_cta
        self.grid = self.ob_ctas + num_rows * ctas_per_token
        self.smem_bytes = ob_boundary_smem_bytes(
            hidden, k_warps, ctas_per_token, n_tiles, self.mbar_exchange
        )

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gRes: cute.Tensor,
        gPost: cute.Tensor,
        gComb: cute.Tensor,
        gPre: cute.Tensor,
        gW: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        q_addr: Int64,
        sf_addr: Int64,
        scale_addr: Int64,
        stream,
    ) -> None:
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            Uint8,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        copy_f32 = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            Float32,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        store_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        self.ob_kernel(
            gA,
            gB,
            gS,
            gRes,
            gPost,
            gComb,
            gPre,
            gW,
            gResOut,
            gOut,
            gWs,
            rank,
            q_addr,
            sf_addr,
            scale_addr,
            copy_a,
            copy_b,
            copy_f32,
            store_bf16,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[self.threads_cta, 1, 1],
            cluster=[self.ctas, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def ob_kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gRes: cute.Tensor,
        gPost: cute.Tensor,
        gComb: cute.Tensor,
        gPre: cute.Tensor,
        gW: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        q_addr: Int64,
        sf_addr: Int64,
        scale_addr: Int64,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
        copy_f32: cute.CopyAtom,
        store_bf16: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        if bid < self.ob_ctas:
            self._ob_role(bid, tidx, gA, gB, gS, gWs, rank, copy_a, copy_b)
        else:
            self._token_role(
                bid - self.ob_ctas,
                tidx,
                rank,
                gRes,
                gPost,
                gComb,
                gPre,
                gW,
                gResOut,
                gOut,
                gWs,
                q_addr,
                sf_addr,
                scale_addr,
                copy_a,
                copy_f32,
                store_bf16,
            )

    @cute.jit
    def _ob_role(
        self,
        bid,
        tidx,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        """The GEMM tiles of o_b CTA ``bid``, pushed into every rank's Lamport slots."""
        warp = tidx // 32
        lane = tidx % 32
        r = lane // 4
        q = lane % 4
        num_rows: cutlass.Constexpr = self.num_rows
        k_warps: cutlass.Constexpr = self.k_warps
        chunks: cutlass.Constexpr = self.chunks
        n_tiles: cutlass.Constexpr = self.n_tiles
        groups: cutlass.Constexpr = self.groups
        cols_g: cutlass.Constexpr = self.cols_group
        cols: cutlass.Constexpr = self.cols_cta
        nranks: cutlass.Constexpr = self.nranks
        vecs: cutlass.Constexpr = self.hidden // VEC
        tot: cutlass.Constexpr = self.num_rows * (self.hidden // VEC)
        threads: cutlass.Constexpr = self.threads_cta
        control_warp: cutlass.Constexpr = self.threads_cta // 32 - 1
        group = warp // k_warps
        kw = warp % k_warps
        n0 = bid * cols + group * cols_g
        k_warp0 = kw * (chunks * CHUNK)

        # ---- weights + scales of the warp's tiles (independent of the producer kernel) ----
        gB_vec = cute.logical_divide(gB, (None, 16))
        b_regs = cute.make_rmem_tensor(
            cute.make_layout((n_tiles * chunks, 16), stride=(16, 1)), Uint8
        )
        s_bits = cute.make_rmem_tensor(cute.make_layout((n_tiles * chunks,)), Int32)
        s_regs = cute.recast_tensor(s_bits, Float32)
        if group < groups:
            for nt in cutlass.range_constexpr(n_tiles):
                b_row = n0 + N_TILE * nt + r
                for c in cutlass.range_constexpr(chunks):
                    k0 = k_warp0 + c * CHUNK + 16 * q
                    cute.copy(
                        copy_b, gB_vec[b_row, (None, k0 // 16)], b_regs[nt * chunks + c, None]
                    )
                    # UE8M0 -> the bit pattern of 2^(s - 127).
                    s_bits[nt * chunks + c] = gS[b_row, k0 // GROUP].to(Int32) << 23
        # A UE8M0 scale's FP32 bits shifted by 16 are its BF16 bits.
        s2 = cute.make_rmem_tensor(cute.make_layout((n_tiles * chunks,)), Uint32)
        if const_expr(self.bf16_dequant):
            for i in cutlass.range_constexpr(n_tiles * chunks):
                s2[i] = Uint32(s_bits[i]) | (Uint32(s_bits[i]) >> 16)
        # The workspace table holds addresses only, written by the host before any launch.
        push_base = Int64(0)
        own = Int64(0)
        peers = cute.make_rmem_tensor(cute.make_layout((nranks,)), Int64)
        if const_expr(self.mnnvl):
            own = Int64(gWs[0])
            push_base = Int64(gWs[1])
        else:
            for p in cutlass.range_constexpr(nranks):
                peers[p] = Int64(gWs[p])

        smem = SmemAllocator()
        red = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (groups, k_warps, 16, cols_g),
                stride=(k_warps * 16 * cols_g, 16 * cols_g, cols_g, 1),
            ),
            byte_alignment=16,
        )
        stage = smem.allocate_tensor(
            BFloat16, cute.make_layout((16, cols), stride=(cols, 1)), byte_alignment=16
        )
        offs = smem.allocate_tensor(Int64, cute.make_layout((2,)), byte_alignment=16)

        if const_expr(self.use_pdl):
            griddepcontrol_wait()
        # The CTA whose arrival is counted last advances the Lamport state: no CTA waits for others.
        state = Int32(0)
        hand_flags = Int64(0)
        hand_layout = Int64(0)
        counter = Int64(0)
        if warp == control_warp:
            if lane == 0:
                # This launch's Lamport buffer (the state is advanced only after every CTA arrived).
                if const_expr(self.mnnvl):
                    flags = Int64(gWs[2])
                    cur = _ld_volatile_s32(flags)
                    buf_bytes = Int64(_ld_volatile_s32(flags + Int64(8)))
                    offs[0] = Int64(cur) * buf_bytes
                    state = cur
                    hand_flags = flags
                    hand_layout = flags
                    counter = flags + Int64(32)
                else:
                    flag_addr = Int64(gWs[nranks])
                    layout_addr = Int64(gWs[nranks + 1])
                    flag = _ld_volatile_s32(flag_addr + Int64(8))
                    comm_size = _ld_volatile_s64(layout_addr + Int64(8))
                    offs[0] = Int64(flag % 3) * comm_size
                    state = flag
                    hand_flags = flag_addr
                    hand_layout = layout_addr
                    counter = flag_addr
        prior = _atom_release_add_if_u32(counter, Int32(1), Int32(tidx == control_warp * 32))

        # ---- C tiles: the GEMM kernel's main loop (same K split, MMA order and FP32 sums) ----
        gA_vec = cute.logical_divide(gA, (None, 8))
        acc = cute.make_rmem_tensor(cute.make_layout((n_tiles, 4), stride=(4, 1)), Float32)
        acc.fill(0.0)
        a_lo0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_lo1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_hi0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_hi1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_lo0_u32 = cute.recast_tensor(a_lo0, Uint32)
        a_lo1_u32 = cute.recast_tensor(a_lo1, Uint32)
        a_hi0_u32 = cute.recast_tensor(a_hi0, Uint32)
        a_hi1_u32 = cute.recast_tensor(a_hi1, Uint32)
        a_frag = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_frag_u32 = cute.recast_tensor(a_frag, Uint32)
        b_frag = cute.make_rmem_tensor(cute.make_layout((4,)), BFloat16)
        b_frag_u32 = cute.recast_tensor(b_frag, Uint32)
        tile = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        w_bf16 = cute.make_rmem_tensor(cute.make_layout((8,)), Uint32)
        if group < groups:
            for c in cutlass.range_constexpr(chunks):
                k0 = k_warp0 + c * CHUNK + 16 * q
                if const_expr(n_tiles == 1):
                    self._ob_convert(w_bf16, b_regs, s_regs, s2, c)
                a_lo0_u32.fill(0)
                a_lo1_u32.fill(0)
                a_hi0_u32.fill(0)
                a_hi1_u32.fill(0)
                if r < num_rows:
                    cute.copy(copy_a, gA_vec[r, (None, k0 // 8)], a_lo0)
                    cute.copy(copy_a, gA_vec[r, (None, k0 // 8 + 1)], a_lo1)
                if const_expr(8 < num_rows):
                    if r + 8 < num_rows:
                        cute.copy(copy_a, gA_vec[r + 8, (None, k0 // 8)], a_hi0)
                        cute.copy(copy_a, gA_vec[r + 8, (None, k0 // 8 + 1)], a_hi1)
                for nt in cutlass.range_constexpr(n_tiles):
                    if const_expr(n_tiles > 1):
                        self._ob_convert(w_bf16, b_regs, s_regs, s2, nt * chunks + c)
                    for e in cutlass.range_constexpr(4):
                        tile[e] = acc[nt, e]
                    for j in cutlass.range_constexpr(4):
                        if const_expr(j < 2):
                            a_frag_u32[0] = a_lo0_u32[2 * j]
                            a_frag_u32[1] = a_hi0_u32[2 * j]
                            a_frag_u32[2] = a_lo0_u32[2 * j + 1]
                            a_frag_u32[3] = a_hi0_u32[2 * j + 1]
                        else:
                            a_frag_u32[0] = a_lo1_u32[2 * j - 4]
                            a_frag_u32[1] = a_hi1_u32[2 * j - 4]
                            a_frag_u32[2] = a_lo1_u32[2 * j - 3]
                            a_frag_u32[3] = a_hi1_u32[2 * j - 3]
                        b_frag_u32[0] = w_bf16[2 * j]
                        b_frag_u32[1] = w_bf16[2 * j + 1]
                        tile.store(mma_sync(a_frag, b_frag, tile))
                    for e in cutlass.range_constexpr(4):
                        acc[nt, e] = tile[e]

        # ---- split-K sums in warp order, one BF16 rounding, staged per row ----
        if group < groups:
            for nt in cutlass.range_constexpr(n_tiles):
                c0 = N_TILE * nt + 2 * q
                red[group, kw, r, c0] = acc[nt, 0]
                red[group, kw, r, c0 + 1] = acc[nt, 1]
                red[group, kw, r + 8, c0] = acc[nt, 2]
                red[group, kw, r + 8, c0 + 1] = acc[nt, 3]
        cute.arch.sync_threads()
        for i in cutlass.range(tidx, groups * num_rows * cols_g, threads):
            g = i // (num_rows * cols_g)
            rem = i % (num_rows * cols_g)
            row = rem // cols_g
            col = rem % cols_g
            total = Float32(0.0)
            for w in cutlass.range_constexpr(k_warps):
                total = total + red[g, w, row, col]
            stage[row, g * cols_g + col] = total.to(BFloat16)
        cute.arch.sync_threads()
        data_off = offs[0]

        # ---- push: one 16-byte vector per (row, 8 columns), as the boundary kernel pushes it ----
        neg0 = Int32(_NEG_ZERO)
        stage_v = cute.logical_divide(stage, (None, N_TILE))
        vals = cute.make_rmem_tensor(cute.make_layout((N_TILE,)), BFloat16)
        words = cute.recast_tensor(vals, Int32)
        tiles_cta: cutlass.Constexpr = cols // N_TILE
        for v in cutlass.range(tidx, num_rows * tiles_cta, threads):
            row = v // tiles_cta
            nt = v % tiles_cta
            vcol = bid * tiles_cta + nt
            cute.autovec_copy(stage_v[row, (None, nt)], vals)
            w0 = words[0]
            w1 = words[1]
            w2 = words[2]
            w3 = words[3]
            slot = (Int64(rank) * Int64(tot) + Int64(row) * Int64(vecs) + Int64(vcol)) * Int64(16)
            if const_expr(self.mnnvl):
                w0 = _bf16x2_neg_zero_to_zero(w0)
                w1 = _bf16x2_neg_zero_to_zero(w1)
                w2 = _bf16x2_neg_zero_to_zero(w2)
                w3 = _bf16x2_neg_zero_to_zero(w3)
                _st_v4(push_base + data_off + slot, w0, w1, w2, w3)
                _st_v4(own + data_off + slot, w0, w1, w2, w3)
            else:
                if w0 == neg0:
                    w0 = Int32(0)
                if w1 == neg0:
                    w1 = Int32(0)
                if w2 == neg0:
                    w2 = Int32(0)
                if w3 == neg0:
                    w3 = Int32(0)
                for p in cutlass.range_constexpr(nranks):
                    _st_v4(peers[p] + data_off + slot, w0, w1, w2, w3)
        if const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
        if warp == control_warp:
            if lane == 0:
                self._handover_if_last(prior, state, hand_flags, hand_layout)

    @cute.jit
    def _handover_if_last(self, prior, state, flag_addr: Int64, layout_addr: Int64) -> None:
        """The CTA whose arrival was counted last advances the Lamport state."""
        if prior == Int32(self.grid - 1):
            _fence_acq_rel_gpu()
            self._advance_state(state, self.num_rows, flag_addr, layout_addr)

    @cute.jit
    def _ob_convert(
        self,
        w_bf16: cute.Tensor,
        b_regs: cute.Tensor,
        s_regs: cute.Tensor,
        s2: cute.Tensor,
        slot: cutlass.Constexpr,
    ) -> None:
        """The 16 E4M3 weights of register slot ``slot`` times their scale -> 8 BF16 pairs."""
        raw = cute.recast_tensor(b_regs[slot, None], Uint32)
        for w in cutlass.range_constexpr(4):
            if const_expr(self.bf16_dequant):
                pair = _fp8x4_to_bf16x4_s2(raw[w], s2[slot])
            else:
                pair = _fp8x4_to_bf16x4_scaled(raw[w], s_regs[slot])
            w_bf16[2 * w] = pair[0]
            w_bf16[2 * w + 1] = pair[1]

    @cute.jit
    def _token_role(
        self,
        b,
        tidx,
        rank: Int32,
        gRes: cute.Tensor,
        gPost: cute.Tensor,
        gComb: cute.Tensor,
        gPre: cute.Tensor,
        gW: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gWs: cute.Tensor,
        q_addr: Int64,
        sf_addr: Int64,
        scale_addr: Int64,
        copy_bf16: cute.CopyAtom,
        copy_f32: cute.CopyAtom,
        store_bf16: cute.CopyAtom,
    ) -> None:
        """The boundary of token ``b // ctas``: everything the boundary kernel does but the push."""
        ctas: cutlass.Constexpr = self.ctas
        nranks: cutlass.Constexpr = self.nranks
        vecs: cutlass.Constexpr = self.hidden // VEC
        tot: cutlass.Constexpr = self.num_rows * (self.hidden // VEC)
        tok = b // ctas
        vi = (b % ctas) * self.threads_cta + tidx
        neg0 = Int32(_NEG_ZERO)

        gRes_v = cute.logical_divide(gRes, (None, VEC))
        gResOut_v = cute.logical_divide(gResOut, (None, VEC))
        gOut_v = cute.logical_divide(gOut, (None, VEC))
        gW_v = cute.logical_divide(gW, (None, VEC))
        gPost_v = cute.logical_divide(gPost, (None, 4))
        gComb_v = cute.logical_divide(gComb, (None, 4))
        gPre_v = cute.logical_divide(gPre, (None, 4))

        # ---- the norm weight and the host-written workspace addresses: before the wait ----
        w_bf = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        cute.copy(copy_bf16, gW_v[0, (None, vi)], w_bf)
        gscale = Float32(0.0)
        if const_expr(self.quant == "nvfp4"):
            gscale = ldg_f32(scale_addr)
        own = Int64(0)
        flags = Int64(0)
        flag_addr = Int64(0)
        layout_addr = Int64(0)
        if const_expr(self.mnnvl):
            own = Int64(gWs[0])
            flags = Int64(gWs[2])
        else:
            flag_addr = Int64(gWs[nranks])
            layout_addr = Int64(gWs[nranks + 1])
            own = Int64(gWs[rank])
        mbar_addr = Int32(0)
        if const_expr(self.mbar_exchange):
            mbar_addr = self._mbar_setup(tidx)

        if const_expr(self.use_pdl):
            griddepcontrol_wait()
        # The wait covers the coefficient kernel too, a second prerequisite grid when it releases early.
        pre, post, comb = self._load_coefficients(tok, gPost_v, gComb_v, gPre_v, copy_f32)

        # ---- Lamport state of this launch (read before any CTA bumps the arrival counter) ----
        stage_sizes = cute.make_rmem_tensor(cute.make_layout((4,)), Int32)
        cur = Int32(0)
        dirty = Int32(0)
        buf_bytes = Int64(0)
        dirty_stages = Int32(0)
        flag = Int32(0)
        clear_size = Int64(0)
        clear_off = Int64(0)
        data_off = Int64(0)
        if const_expr(self.mnnvl):
            cur = _ld_volatile_s32(flags)
            dirty = _ld_volatile_s32(flags + Int64(4))
            buf_bytes = Int64(_ld_volatile_s32(flags + Int64(8)))
            dirty_stages = _ld_volatile_s32(flags + Int64(12))
            for s in cutlass.range_constexpr(4):
                stage_sizes[s] = _ld_volatile_s32(flags + Int64(16 + 4 * s))
            data_off = Int64(cur) * buf_bytes
            if tidx < 32:
                _arrive_barrier_1(Int32(self.threads_cta), wait=True)
            else:
                _arrive_barrier_1(Int32(self.threads_cta), wait=False)
        else:
            flag = _ld_volatile_s32(flag_addr + Int64(8))
            clear_size = _ld_volatile_s64(layout_addr)
            comm_size = _ld_volatile_s64(layout_addr + Int64(8))
            data_off = Int64(flag % 3) * comm_size
            clear_off = Int64((flag + 2) % 3) * comm_size
            if tidx < 32:
                _arrive_barrier_1(Int32(self.threads_cta), wait=True)
            else:
                _arrive_barrier_1(Int32(self.threads_cta), wait=False)
        # Thread 0 counts the CTA after warp 0's barrier wait.
        counter = flag_addr
        if const_expr(self.mnnvl):
            counter = flags + Int64(32)
        prior = _atom_release_add_if_u32(counter, Int32(1), Int32(tidx == 0))
        if const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
        # After the wait: launches since the residual's boundary may release before their own wait.
        # After the arrival too, whose release would otherwise wait for these loads.
        r_bf = cute.make_rmem_tensor(cute.make_layout((HC, VEC), stride=(VEC, 1)), BFloat16)
        for j in cutlass.range_constexpr(HC):
            cute.copy(copy_bf16, gRes_v[tok, (None, j * vecs + vi)], r_bf[j, None])

        idx = Int64(tok) * Int64(vecs) + Int64(vi)
        # ---- clear the slots the previous launch used ----
        if const_expr(self.mnnvl):
            for s in cutlass.range_constexpr(4):
                if Int32(s) < dirty_stages:
                    stage_bytes = buf_bytes // Int64(dirty_stages)
                    stage = own + (Int64(dirty) * Int64(dirty_stages) + Int64(s)) * stage_bytes
                    clear_vecs = Int64(stage_sizes[s]) // Int64(16)
                    c = idx
                    while c < clear_vecs:
                        _st_v4(stage + c * Int64(16), neg0, neg0, neg0, neg0)
                        c = c + Int64(tot)
        else:
            # Its size in bytes (every Lamport kernel on the workspace), as whole 16-byte stores.
            clear_vecs = (clear_size + Int64(15)) // Int64(16)
            c = idx
            while c < clear_vecs:
                _st_v4(own + clear_off + c * Int64(16), neg0, neg0, neg0, neg0)
                c = c + Int64(tot)

        if tidx == 0:
            if const_expr(self.mnnvl):
                self._handover_if_last(prior, cur, flags, flags)
            else:
                self._handover_if_last(prior, flag, flag_addr, layout_addr)

        # ---- wait for the token's slice of every rank (pushed by the o_b CTAs of every rank) ----
        got = cute.make_rmem_tensor(cute.make_layout((nranks, 4), stride=(4, 1)), Int32)
        pending = Int32(1)
        while pending != 0:
            pending = Int32(0)
            for p in cutlass.range_constexpr(nranks):
                a, bb, cc, d = _ld_volatile_v4(
                    own + data_off + (Int64(p) * Int64(tot) + idx) * Int64(16)
                )
                got[p, 0] = a
                got[p, 1] = bb
                got[p, 2] = cc
                got[p, 3] = d
                if a == neg0:
                    pending = Int32(1)
                if bb == neg0:
                    pending = Int32(1)
                if cc == neg0:
                    pending = Int32(1)
                if d == neg0:
                    pending = Int32(1)

        # ---- x = BF16(FP32 sum in rank order), as the library kernel's FP32 accumulation ----
        x_bf = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        for i in cutlass.range_constexpr(4):
            lo = bf16x2_lo(got[0, i])
            hi = bf16x2_hi(got[0, i])
            for p in cutlass.range_constexpr(1, nranks):
                lo = lo + bf16x2_lo(got[p, i])
                hi = hi + bf16x2_hi(got[p, i])
            x_bf[2 * i] = lo.to(BFloat16)
            x_bf[2 * i + 1] = hi.to(BFloat16)

        self._boundary(
            tok,
            tidx,
            vi,
            ctas,
            x_bf,
            r_bf,
            w_bf,
            pre,
            post,
            comb,
            gResOut_v,
            gOut_v,
            store_bf16,
            False,
            q_addr,
            sf_addr,
            gscale,
            mbar_addr,
        )
