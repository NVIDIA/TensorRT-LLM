# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Row-parallel projection + MNNVL all-reduce + residual + RMSNorm in one launch.

``GemmAllReduceNormKernel`` fuses this rank's partial of the attention ``o_proj`` / GDN ``out_proj`` (the
``low_m_mma_gemm`` warp-MMA GEMM) with the all-reduce, residual add, RMSNorm and optional FP8 output:

1. Weights, norm weight and workspace table load before ``griddepcontrol.wait``, activations after. Each CTA
   computes its columns exactly as the GEMM kernel does, so the all-reduce input is bitwise the unfused one.
2. The BF16 tile goes into the library's current Lamport buffer with the CuTe one-shot layout and flag
   protocol ([rank][row][H]), so launches may interleave with the library / CuTe all-reduces in one graph.
3. CTA ``c < M`` then reduces row ``c`` over the ranks, adds the residual and normalizes; other CTAs only
   do the GEMM. Dependents are released once the activations and the residual are consumed.

The norm's sum of squares may be ordered differently from the library kernel (last-bit BF16 difference in
the norm output); the residual output is identical.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, const_expr

from .low_m_mma_gemm import CHUNK, N_TILE, LowMMmaGemmKernel, _rcp_rn
from .mnnvl_allreduce_norm import (
    _NEG_ZERO,
    VEC,
    _add,
    _bf16r,
    _div_rn,
    _e4m3x2,
    _hi,
    _ld_s64,
    _ld_v4,
    _ld_volatile_v4,
    _lo,
    _mul,
    _pack_bf16,
    _rsqrt,
    _sanitize,
    _shfl_bfly,
    _sink4,
    _smem_addr,
    _st_s32,
    _st_v2,
    _st_v4,
)
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

try:
    from cutlass.memory import SmemAllocator as _SmemAllocator
except ImportError:  # older CuTe DSL
    from cutlass.utils import SmemAllocator as _SmemAllocator

MAX_ROWS = 32


class GemmAllReduceNormKernel(LowMMmaGemmKernel):
    """``num_rows`` x K (per rank) @ [H, K].T, all-reduced over ``nranks``, + residual, RMSNorm."""

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        k_warps: int,
        n_tiles: int,
        swap_ab: bool,
        fp8_act: str,
        hidden: int,
        norm_eps: float,
        nranks: int,
        use_pdl: bool,
        min_blocks: int = 2,
        norm_out: bool = True,
        quant: bool = False,
    ) -> None:
        if not 1 <= num_rows <= MAX_ROWS:
            raise ValueError(f"num_rows={num_rows} must be in [1, {MAX_ROWS}]")
        if nranks not in (2, 4, 8, 16):
            raise ValueError("unsupported group size")
        if (N_TILE * n_tiles) % VEC:
            raise ValueError("a CTA's columns must be whole 16-byte vectors")
        super().__init__(
            num_rows=num_rows,
            k_extent=k_extent,
            k_warps=k_warps,
            use_pdl=use_pdl,
            n_fp8=hidden,
            n_tiles=n_tiles,
            fp8_act=fp8_act,
            swap_ab=swap_ab,
            min_blocks=min_blocks,
        )
        self.hidden = hidden
        self.norm_eps = float(norm_eps)
        self.nranks = nranks
        self.norm_out = norm_out
        self.quant = quant
        self.ctas = self.fp8_ctas
        self.vpr = N_TILE * n_tiles // VEC  # 16-byte vectors of a row in one CTA
        self.threads = 32 * k_warps
        if num_rows * self.vpr > self.threads:
            raise ValueError("one thread per (row, vector) of the CTA tile")
        self.vpt = (hidden // VEC) // self.threads
        if (hidden // VEC) % self.threads or not 1 <= self.vpt <= 4 or num_rows > self.ctas:
            raise ValueError("hidden / 8 must be 1-4 vectors per thread and rows <= CTAs")

    @cute.jit
    def __call__(
        self,
        gA16: cute.Tensor,
        gA8: cute.Tensor,
        gB: cute.Tensor,
        gSw: cute.Tensor,
        gSin: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        stream,
    ) -> None:
        from .low_m_mma_gemm import _g2r_atom

        copy_a16 = _g2r_atom(BFloat16, streaming=False)
        copy_a8 = _g2r_atom(Uint8, streaming=False)
        copy_b = _g2r_atom(Uint8, streaming=True)
        cols = N_TILE * self.n_tiles
        smem = (
            self.k_warps * self.row_tiles * 16 * cols * 4
            + self.row_tiles * 16 * cols * 2
            + 64
            + 128
        )
        smem += self.k_warps * 4 + 32
        self.kernel(
            gA16,
            gA8,
            gB,
            gSw,
            gSin,
            gRes,
            gW,
            gScale,
            gResOut,
            gOut,
            gQuant,
            gWs,
            rank,
            copy_a16,
            copy_a8,
            copy_b,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            smem=smem,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        gA16: cute.Tensor,
        gA8: cute.Tensor,
        gB: cute.Tensor,
        gSw: cute.Tensor,
        gSin: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        copy_a16: cute.CopyAtom,
        copy_a8: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        r = lane // 4
        q = lane % 4

        num_rows: cutlass.Constexpr = self.num_rows
        row_tiles: cutlass.Constexpr = self.row_tiles
        k_warps: cutlass.Constexpr = self.k_warps
        chunks: cutlass.Constexpr = self.chunks_per_warp
        n_tiles: cutlass.Constexpr = self.n_tiles
        cols: cutlass.Constexpr = N_TILE * n_tiles
        vpr: cutlass.Constexpr = self.vpr
        nthreads: cutlass.Constexpr = self.threads
        ctas: cutlass.Constexpr = self.ctas
        nranks: cutlass.Constexpr = self.nranks
        hidden: cutlass.Constexpr = self.hidden
        vecs: cutlass.Constexpr = self.hidden // VEC
        fp8_mma: cutlass.Constexpr = self.fp8_act != "bf16"

        smem = _SmemAllocator()
        red = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (k_warps, row_tiles * 16, cols), stride=(row_tiles * 16 * cols, cols, 1)
            ),
            byte_alignment=16,
        )
        tile = smem.allocate_tensor(
            BFloat16, cute.make_layout((row_tiles * 16, cols), stride=(cols, 1)), byte_alignment=16
        )
        fl = smem.allocate_tensor(Int32, cute.make_layout((12,)), byte_alignment=16)
        vpt: cutlass.Constexpr = self.vpt
        nred = smem.allocate_tensor(Float32, cute.make_layout((k_warps,)), byte_alignment=16)
        n0 = cta * cols
        k_warp0 = warp * (chunks * CHUNK)
        item = tidx
        row = item // vpr
        v = item % vpr
        active = item < num_rows * vpr
        col0 = n0 + v * VEC  # first output column of the vector
        # row CTA: thread handles 16-byte vectors tidx + j * nthreads (j < vpt) of row `cta`
        ar_cta = cta < Int32(num_rows)
        tok_c = cta
        if cta >= Int32(num_rows):
            tok_c = Int32(0)  # in-bounds address for GEMM-only CTAs (never stored)
        ar_base = Int64(tok_c) * Int64(vecs) + Int64(tidx)  # vector index of j = 0 in [M, H / 8]

        # ---- before the dependency wait: weights, scales, the norm weight, the workspace table ----
        w8 = cute.make_rmem_tensor(cute.make_layout((n_tiles * chunks, 16), stride=(16, 1)), Uint8)
        w16 = cute.make_rmem_tensor(
            cute.make_layout((2 * n_tiles * chunks, 8), stride=(8, 1)), BFloat16
        )
        gB_vec = cute.logical_divide(gB, (None, 16))
        for nt in cutlass.range_constexpr(n_tiles):
            b_row = n0 + N_TILE * nt + r
            for c in cutlass.range_constexpr(chunks):
                k0 = k_warp0 + c * CHUNK + 16 * q
                cute.copy(copy_b, gB_vec[b_row, (None, k0 // 16)], w8[nt * chunks + c, None])
        alpha = gSw[0]
        factor = Float32(1.0)
        if const_expr(fp8_mma):
            s_in = gSin[0]
            alpha = s_in * alpha
            if const_expr(self.fp8_act == "quant"):
                factor = _rcp_rn(s_in)
        tbl = gWs.iterator.toint()
        own = _ld_s64(Int64(tbl))
        mcast = _ld_s64(Int64(tbl) + Int64(8))
        flags = _ld_s64(Int64(tbl) + Int64(16))
        nwr = cute.make_rmem_tensor((4 * vpt,), Int32)  # norm weight of the thread's row vectors
        for j in cutlass.range_constexpr(vpt):
            a0, a1, a2, a3 = _ld_v4(
                Int64(gW.iterator.toint()) + (Int64(tidx) + Int64(j * nthreads)) * Int64(16)
            )
            nwr[4 * j + 0] = a0
            nwr[4 * j + 1] = a1
            nwr[4 * j + 2] = a2
            nwr[4 * j + 3] = a3

        if const_expr(self.use_pdl):
            griddepcontrol_wait()

        # ---- after the wait: the residual (another kernel's output) and the Lamport state ----
        rres = cute.make_rmem_tensor((4 * vpt,), Int32)  # residual of the thread's row vectors
        for j in cutlass.range_constexpr(vpt):
            a0, a1, a2, a3 = _ld_v4(
                Int64(gRes.iterator.toint()) + (ar_base + Int64(j * nthreads)) * Int64(16)
            )
            rres[4 * j + 0] = a0
            rres[4 * j + 1] = a1
            rres[4 * j + 2] = a2
            rres[4 * j + 3] = a3
        # One thread per CTA reads the flags (all warps polling one line is an L2 hot spot); they are consumed
        # only after the main loop, so the loads stay in flight under it.
        lf0 = Int32(0)
        lf1 = Int32(0)
        lf2 = Int32(0)
        lf3 = Int32(0)
        lf4 = Int32(0)
        lf5 = Int32(0)
        lf6 = Int32(0)
        lf7 = Int32(0)
        if tidx == 0:
            lf0, lf1, lf2, lf3 = _ld_volatile_v4(flags)
            lf4, lf5, lf6, lf7 = _ld_volatile_v4(flags + Int64(16))

        # ---- GEMM: LowM's main loop and warp-order reduction (bitwise the unfused kernel) ----
        acc = cute.make_rmem_tensor(
            cute.make_layout((n_tiles * row_tiles, 4), stride=(4, 1)), Float32
        )
        acc.fill(0.0)
        swap: cutlass.Constexpr = self.swap_ab and n_tiles % 2 == 0
        if const_expr(fp8_mma):
            if const_expr(swap):
                self._mainloop_fp8_swap(
                    acc, gA16, gA8, w8, factor, copy_a16, copy_a8, k_warp0, r, q, n_tiles
                )
            else:
                self._mainloop_fp8(
                    acc, gA16, gA8, w8, factor, copy_a16, copy_a8, k_warp0, r, q, n_tiles
                )
        else:
            if const_expr(swap):
                self._mainloop_bf16_swap(acc, gA16, w8, w16, copy_a16, k_warp0, r, q, n_tiles, True)
            else:
                self._mainloop_bf16(acc, gA16, w8, w16, copy_a16, k_warp0, r, q, n_tiles, True)
        if const_expr(swap):
            for pair in cutlass.range_constexpr(n_tiles // 2):
                cc = 2 * N_TILE * pair + r
                red[warp, 2 * q, cc] = acc[pair, 0]
                red[warp, 2 * q + 1, cc] = acc[pair, 1]
                red[warp, 2 * q, cc + N_TILE] = acc[pair, 2]
                red[warp, 2 * q + 1, cc + N_TILE] = acc[pair, 3]
        else:
            for nt in cutlass.range_constexpr(n_tiles):
                for t in cutlass.range_constexpr(row_tiles):
                    cc = N_TILE * nt + 2 * q
                    red[warp, 16 * t + r, cc] = acc[nt * row_tiles + t, 0]
                    red[warp, 16 * t + r, cc + 1] = acc[nt * row_tiles + t, 1]
                    red[warp, 16 * t + r + 8, cc] = acc[nt * row_tiles + t, 2]
                    red[warp, 16 * t + r + 8, cc + 1] = acc[nt * row_tiles + t, 3]
        if tidx == 0:
            fl[0] = lf0
            fl[1] = lf1
            fl[2] = lf2
            fl[3] = lf3
            fl[4] = lf4
            fl[5] = lf5
            fl[6] = lf6
            fl[7] = lf7
        if const_expr(self.use_pdl):
            # residual loads must complete before dependents are released; the sink word fl[11] is never read
            sink = _smem_addr(fl.iterator) + Int32(44)
            for j in cutlass.range_constexpr(vpt):
                _sink4(sink, rres[4 * j + 0], rres[4 * j + 1], rres[4 * j + 2], rres[4 * j + 3])
        cute.arch.sync_threads()
        # PDL: this barrier follows the last access to allocator-recyclable memory; afterwards only the
        # persistent workspace / flags and the outputs are touched, so a dependent's pre-wait write into a
        # recycled block cannot race this kernel.
        if const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
        cur = fl[0]
        dirty = fl[1]
        buf_bytes = Int64(fl[2])
        dirty_stages = fl[3]
        sb0 = fl[4]
        sb1 = fl[5]
        sb2 = fl[6]
        sb3 = fl[7]
        for idx in cutlass.range(tidx, num_rows * cols, nthreads):
            rr = idx // cols
            cc = idx % cols
            value = Float32(0.0)
            for w in cutlass.range_constexpr(k_warps):
                value = value + red[w, rr, cc]
            tile[rr, cc] = (value * alpha).to(BFloat16)
        cute.arch.sync_threads()

        # ---- this rank's tile -> every rank's current Lamport buffer ([rank][row][H]) ----
        m = num_rows
        tot = Int64(m) * Int64(vecs)
        data_off = Int64(cur) * buf_bytes
        x0 = Int32(0)
        x1 = Int32(0)
        x2 = Int32(0)
        x3 = Int32(0)
        tile_u32 = cute.recast_tensor(tile, Int32)  # [rows, cols / 2] BF16 pairs
        if active:
            x0 = tile_u32[row, v * 4 + 0]
            x1 = tile_u32[row, v * 4 + 1]
            x2 = tile_u32[row, v * 4 + 2]
            x3 = tile_u32[row, v * 4 + 3]
            _st_v4(
                mcast
                + data_off
                + (Int64(rank) * tot + Int64(row) * Int64(vecs) + Int64(col0 // VEC)) * Int64(16),
                _sanitize(x0),
                _sanitize(x1),
                _sanitize(x2),
                _sanitize(x3),
            )

        # ---- clear this thread's share of the previous call's buffer (every dirty stage) ----
        neg0 = Int32(_NEG_ZERO)
        gtid = Int64(cta) * Int64(nthreads) + Int64(tidx)
        gthreads = Int64(ctas) * Int64(nthreads)
        for s_i in cutlass.range_constexpr(4):
            if Int32(s_i) < dirty_stages:
                stage_bytes = buf_bytes // Int64(dirty_stages)
                stage = own + (Int64(dirty) * Int64(dirty_stages) + Int64(s_i)) * stage_bytes
                sb = sb0 if s_i == 0 else (sb1 if s_i == 1 else (sb2 if s_i == 2 else sb3))
                clear_vecs = (Int64(sb) + Int64(15)) // Int64(16)
                c = gtid
                while c < clear_vecs:
                    _st_v4(stage + c * Int64(16), neg0, neg0, neg0, neg0)
                    c = c + gthreads

        # ---- row all-reduce: FP32 rank-order sum -> BF16, + residual; RMSNorm: BF16 squares summed per
        # thread (vector, then element order), per warp, per CTA ----
        if ar_cta:
            res = cute.make_rmem_tensor((VEC * vpt,), Float32)
            got = cute.make_rmem_tensor((vpt * nranks * 4,), Int32)
            pending = Int32(1)
            while pending != 0:
                pending = Int32(0)
                for j in cutlass.range_constexpr(vpt):
                    for rk in cutlass.range_constexpr(nranks):
                        a, b, cc2, d = _ld_volatile_v4(
                            own
                            + data_off
                            + (Int64(rk) * tot + ar_base + Int64(j * nthreads)) * Int64(16)
                        )
                        g0 = (j * nranks + rk) * 4
                        got[g0 + 0] = a
                        got[g0 + 1] = b
                        got[g0 + 2] = cc2
                        got[g0 + 3] = d
                        if a == neg0:
                            pending = Int32(1)
                        if b == neg0:
                            pending = Int32(1)
                        if cc2 == neg0:
                            pending = Int32(1)
                        if d == neg0:
                            pending = Int32(1)
            for j in cutlass.range_constexpr(vpt):
                for i in cutlass.range_constexpr(4):
                    lo = Float32(0.0)
                    hi = Float32(0.0)
                    for rk in cutlass.range_constexpr(nranks):
                        lo = _add(lo, _lo(got[(j * nranks + rk) * 4 + i]))
                        hi = _add(hi, _hi(got[(j * nranks + rk) * 4 + i]))
                    res[VEC * j + 2 * i] = _bf16r(_add(_bf16r(lo), _lo(rres[4 * j + i])))
                    res[VEC * j + 2 * i + 1] = _bf16r(_add(_bf16r(hi), _hi(rres[4 * j + i])))
            for j in cutlass.range_constexpr(vpt):
                _st_v4(
                    Int64(gResOut.iterator.toint()) + (ar_base + Int64(j * nthreads)) * Int64(16),
                    _pack_bf16(res[VEC * j + 0], res[VEC * j + 1]),
                    _pack_bf16(res[VEC * j + 2], res[VEC * j + 3]),
                    _pack_bf16(res[VEC * j + 4], res[VEC * j + 5]),
                    _pack_bf16(res[VEC * j + 6], res[VEC * j + 7]),
                )
            tsum = Float32(0.0)
            for e in cutlass.range_constexpr(VEC * vpt):
                tsum = _add(tsum, _bf16r(_mul(res[e], res[e])))
            for k in cutlass.range_constexpr(5):
                tsum = _add(tsum, _shfl_bfly(tsum, Int32(16 >> k)))
            if lane == 0:
                nred[warp] = tsum
            cute.arch.sync_threads()
            bsum = Float32(0.0)
            if lane < k_warps:
                bsum = nred[lane]
            for k in cutlass.range_constexpr(5):
                bsum = _add(bsum, _shfl_bfly(bsum, Int32(16 >> k)))
            rstd = _rsqrt(_add(_div_rn(bsum, Float32(float(hidden))), Float32(self.norm_eps)))
            inv = Float32(1.0)
            if const_expr(self.quant):
                inv = _div_rn(Float32(1.0), gScale[0])
            for j in cutlass.range_constexpr(vpt):
                y = cute.make_rmem_tensor((VEC,), Float32)
                for i in cutlass.range_constexpr(4):
                    y[2 * i] = _bf16r(_mul(_mul(res[VEC * j + 2 * i], rstd), _lo(nwr[4 * j + i])))
                    y[2 * i + 1] = _bf16r(
                        _mul(_mul(res[VEC * j + 2 * i + 1], rstd), _hi(nwr[4 * j + i]))
                    )
                vrow = ar_base + Int64(j * nthreads)
                if const_expr(self.norm_out):
                    _st_v4(
                        Int64(gOut.iterator.toint()) + vrow * Int64(16),
                        _pack_bf16(y[0], y[1]),
                        _pack_bf16(y[2], y[3]),
                        _pack_bf16(y[4], y[5]),
                        _pack_bf16(y[6], y[7]),
                    )
                if const_expr(self.quant):
                    q01 = _e4m3x2(_mul(y[0], inv), _mul(y[1], inv))
                    q23 = _e4m3x2(_mul(y[2], inv), _mul(y[3], inv))
                    q45 = _e4m3x2(_mul(y[4], inv), _mul(y[5], inv))
                    q67 = _e4m3x2(_mul(y[6], inv), _mul(y[7], inv))
                    _st_v2(
                        Int64(gQuant.iterator.toint()) + vrow * Int64(8),
                        q01 | (q23 << 16),
                        q45 | (q67 << 16),
                    )
        # ---- hand the triple buffer to the next call. No arrival counter needed: CTA 0 gets here only after its
        # whole row arrived, i.e. after every CTA of every rank read the flags and stored its tile ----
        if cta == 0:
            if tidx == 0:
                _st_s32(flags, (cur + Int32(1)) % Int32(3))
                _st_s32(flags + Int64(4), cur)
                _st_s32(flags + Int64(12), Int32(1))
                _st_s32(flags + Int64(16), Int32(m) * Int32(hidden * nranks * 2))
                _st_s32(flags + Int64(20), Int32(0))
                _st_s32(flags + Int64(24), Int32(0))
                _st_s32(flags + Int64(28), Int32(0))
                _st_s32(flags + Int64(32), Int32(0))
