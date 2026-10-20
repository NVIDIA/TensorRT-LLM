# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-parallel projection + MNNVL all-reduce + residual + RMSNorm in one launch (Qwen3.5 decode).

``Q35GemmArNormKernel`` fuses the attention ``o_proj`` / GDN ``out_proj`` (this rank's partial of
``x @ W.T``, the low-M warp-MMA GEMM of ``lowm_mma_gemm``) with the all-reduce of the partials, the
residual add and the following RMSNorm (and optionally the next projection's FP8 input), i.e. what the
pair ``LowM GEMM`` + ``Q35AllReduceNormKernel`` / the library's fused all-reduce computes, without the
kernel boundary between them:

1. Weights (and the norm weight / workspace table) are loaded before ``griddepcontrol.wait``; the
   activations after it. Each CTA computes its ``8 * n_tiles`` output columns for every row exactly as
   the GEMM kernel does (same MMAs, same warp-order reduction, same BF16 rounding): the all-reduce
   input is the unfused GEMM's output, bit for bit.
2. The CTA multicast-stores its BF16 tile, in 16-byte vectors, into the current Lamport buffer of the
   library's MNNVL workspace ([rank][row][H], the CuTe one-shot all-reduce's layout and flag protocol,
   so launches interleave with the library and CuTe all-reduce kernels in one graph).
3. ``row_ar`` (default): the CTAs switch roles. CTA ``c < M`` (``vpt`` 16-byte vectors of the row per
   thread) polls row ``c`` from every rank, sums in FP32 rank order, rounds to BF16, adds the residual
   (BF16) and writes the residual output; the RMSNorm sums BF16 squares per thread (vector, then element
   order), per warp (butterfly) and per CTA (butterfly over the warp sums): ``bf16(x * rstd * w)`` (+ FP8 =
   satfinite(norm * (1 / scale))). The other CTAs only take part in the GEMM. Dependents are released right
   after the dependency wait; the Lamport flags are read by one thread under the GEMM main loop.
4. ``row_ar=False`` (column slices): every CTA polls its own columns of every row from every rank, writes
   per-row partial sums of BF16 squares into an epoch-tagged local exchange (8-byte packets
   ``epoch << 32 | bits``, two slots by epoch parity), and every CTA sums all CTAs' partials of a row in the
   same fixed order. That all-CTA exchange is a cross-CTA wait (``ctas <= SMs * min_blocks``) and an L2
   hot spot; ``row_ar`` (above) avoids both.

Differences to the unfused pair: the RMSNorm's sum of squares may be accumulated in another order than the
library kernel's, so the norm output can differ in the last BF16 bit; the residual output is the same.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, const_expr

from .lowm_mma_gemm import CHUNK, N_TILE, LowMMmaGemmKernel, _rcp_rn
from .q35_allreduce_norm import (
    _NEG_ZERO,
    _PDL_TEST_ENTRY_TRIGGER,
    VEC,
    _add,
    _bf16r,
    _div_rn,
    _e4m3x2,
    _hi,
    _ld_acquire_s32,
    _ld_s64,
    _ld_v4,
    _ld_volatile_s32,
    _ld_volatile_v4,
    _lo,
    _mul,
    _pack_bf16,
    _red_release_add_s32,
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

from cutlass._mlir.dialects import llvm  # noqa: E402
from cutlass.cutlass_dsl import T, dsl_user_op  # noqa: E402


@dsl_user_op
def _ld_volatile_s64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.b64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_s64(addr: Int64, value: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.global.b64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _f32_bits(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bits_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _packet(epoch: Int32, bits: Int32, *, loc=None, ip=None) -> Int64:
    """(epoch << 32) | bits."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int32(epoch).ir_value(loc=loc, ip=ip), Int32(bits).ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, {$2, $1};",
            "=l,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pk_tag(p: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(p).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 lo;\n\tmov.b64 {lo, $0}, $1;\n\t}",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pk_val(p: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(p).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 hi;\n\tmov.b64 {$0, hi}, $1;\n\t}",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _smem_u32(ptr, *, loc=None, ip=None) -> Int32:
    return Int32(ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def _st_shared_b32(addr: Int32, v: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b32(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


MAX_ROWS = 32  # rows of one launch (the partial-sum exchange keeps one FP32 per row per thread)
PART_SLOT_ROWS = 64  # rows per slot of the partial-sum buffer


class Q35GemmArNormKernel(LowMMmaGemmKernel):
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
        max_ctas: int = 1024,
        stamps: bool = False,
        row_ar: bool = True,
        two_shot: bool = False,
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
        if self.ctas > max_ctas:
            raise ValueError(f"{self.ctas} CTAs exceed the partial-sum buffer ({max_ctas})")
        self.max_ctas = max_ctas
        self.stamps = (
            stamps  # debug: %globaltimer of 8 phase boundaries per CTA into gDbg [ctas, 8]
        )
        self.vpr = N_TILE * n_tiles // VEC  # 16-byte vectors of a row in one CTA
        self.threads = 32 * k_warps
        if num_rows * self.vpr > self.threads:
            raise ValueError("one thread per (row, vector) of the CTA tile")
        # row_ar: after the GEMM, CTA c < num_rows reduces row c over the ranks (vpt 16-byte vectors per thread)
        # and normalizes it; otherwise every CTA reduces its own columns and the norm needs an all-CTA
        # partial-sum exchange.
        self.row_ar = row_ar
        self.vpt = (hidden // VEC) // self.threads if row_ar else 1
        if row_ar and (
            (hidden // VEC) % self.threads or not 1 <= self.vpt <= 4 or num_rows > self.ctas
        ):
            raise ValueError("row_ar: hidden / 8 must be 1-4 vectors per thread, rows <= CTAs")
        # two_shot (row_ar only): the library's / CuTe two-shot data movement. Each GEMM tile row goes (unicast)
        # into the SCATTER stage [t / W][rank][H] of row t's owner (rank t % W), whose row CTA sums the ranks in
        # FP32 order, rounds to BF16 and multicasts into every rank's BROADCAST stage [t][H] (second half of the
        # buffer); every rank's row CTA then adds the residual and normalizes with rmsNormLamport's arithmetic
        # ((w * x) * rstd). ~1 / W of the one-shot's NVLink ingress.
        self.two_shot = two_shot
        if two_shot and not row_ar:
            raise ValueError("two_shot needs row_ar")

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
        gPart: cute.Tensor,
        gEpoch: cute.Tensor,
        gDbg: cute.Tensor,
        rank: Int32,
        stream,
    ) -> None:
        from .lowm_mma_gemm import _g2r_atom

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
        if self.row_ar:
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
                gPart,
                gEpoch,
                gDbg,
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
        else:
            smem += (
                self.num_rows * self.vpr * 4
                + (self.k_warps + 1) * MAX_ROWS * 4
                + self.num_rows * self.ctas * 4
            )
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
                gPart,
                gEpoch,
                gDbg,
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
        gPart: cute.Tensor,
        gEpoch: cute.Tensor,
        gDbg: cute.Tensor,
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
        row_ar: cutlass.Constexpr = self.row_ar
        two_shot: cutlass.Constexpr = self.two_shot
        if const_expr(row_ar):
            nred = smem.allocate_tensor(Float32, cute.make_layout((k_warps,)), byte_alignment=16)
        else:
            vpart = smem.allocate_tensor(
                Float32, cute.make_layout((num_rows * vpr,)), byte_alignment=16
            )
            wsum = smem.allocate_tensor(
                Float32,
                cute.make_layout((k_warps + 1, MAX_ROWS), stride=(MAX_ROWS, 1)),
                byte_alignment=16,
            )
            items: cutlass.Constexpr = (
                num_rows * ctas
            )  # (row, CTA) partial sums of the norm exchange
            vals = smem.allocate_tensor(Float32, cute.make_layout((items,)), byte_alignment=16)
        dbg = Int64(gDbg.iterator.toint()) + Int64(cta) * Int64(64)
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbg, _globaltimer())

        n0 = cta * cols
        k_warp0 = warp * (chunks * CHUNK)
        # (row, vector) of the CTA tile this thread moves through the all-reduce
        item = tidx
        row = item // vpr
        v = item % vpr
        active = item < num_rows * vpr
        col0 = n0 + v * VEC  # first output column of the vector
        row_c = row
        if item >= num_rows * vpr:
            row_c = Int32(0)  # in-bounds address for idle threads (never stored)
        row_off = (Int64(row_c) * Int64(vecs) + Int64(col0 // VEC)) * Int64(16)
        # row_ar: CTA `cta` < num_rows reduces row `cta` (the others only take part in the GEMM); thread =
        # 16-byte vectors tidx + j * nthreads (j < vpt) of the row
        ar_cta = cta < Int32(num_rows)
        tok_c = cta
        if cta >= Int32(num_rows):
            tok_c = Int32(0)  # in-bounds address for GEMM-only CTAs (never stored)
        ar_base = Int64(tok_c) * Int64(vecs) + Int64(tidx)  # vector index of j = 0 in [M, H / 8]

        # ---- ahead of the dependency wait: weights, scales, the norm weight, the workspace table ----
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
        peer = own  # two_shot: the unicast buffer of this thread's GEMM row's owner (rank row % W)
        if const_expr(two_shot):
            peer = _ld_s64(Int64(tbl) + Int64(24) + Int64(row_c % Int32(nranks)) * Int64(8))
        nw0, nw1, nw2, nw3 = _ld_v4(Int64(gW.iterator.toint()) + Int64(col0) * Int64(2))
        nwr = cute.make_rmem_tensor(
            (4 * vpt,), Int32
        )  # row_ar: norm weight of the thread's vectors
        if const_expr(row_ar):
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
            if const_expr(_PDL_TEST_ENTRY_TRIGGER):
                griddepcontrol_launch_dependents()
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbg + Int64(8), _globaltimer())

        # ---- after the wait: the residual (another kernel's output) and, by one thread, the Lamport state and the
        # epoch, whose latency the GEMM main loop hides (published by the barrier after it). Dependents are released
        # after that barrier, once the activations and the residual have been consumed (PDL rule) ----
        r0, r1, r2, r3 = _ld_v4(Int64(gRes.iterator.toint()) + row_off)
        rres = cute.make_rmem_tensor((4 * vpt,), Int32)  # row_ar: residual of the thread's vectors
        if const_expr(row_ar):
            for j in cutlass.range_constexpr(vpt):
                a0, a1, a2, a3 = _ld_v4(
                    Int64(gRes.iterator.toint()) + (ar_base + Int64(j * nthreads)) * Int64(16)
                )
                rres[4 * j + 0] = a0
                rres[4 * j + 1] = a1
                rres[4 * j + 2] = a2
                rres[4 * j + 3] = a3
        # one thread per CTA reads the Lamport state and the epoch (128 CTAs x every warp polling the same
        # lines was an L2 hot spot: ~2.6 us); the values are consumed (stored to smem) only after the main loop,
        # so the loads stay in flight under it
        lf0 = Int32(0)
        lf1 = Int32(0)
        lf2 = Int32(0)
        lf3 = Int32(0)
        lf4 = Int32(0)
        lf5 = Int32(0)
        lf6 = Int32(0)
        lf7 = Int32(0)
        lf8 = Int32(0)
        if tidx == 0:
            lf0, lf1, lf2, lf3 = _ld_volatile_v4(flags)
            lf4, lf5, lf6, lf7 = _ld_volatile_v4(flags + Int64(16))
            if const_expr(not row_ar):
                lf8 = _ld_volatile_s32(Int64(gEpoch.iterator.toint()))
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbg + Int64(16), _globaltimer())

        # ---- GEMM: LowM's main loop and warp-order reduction (the unfused kernel's values) ----
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
            fl[8] = lf8
        if const_expr(self.use_pdl):
            # the residual loads must have completed before the dependents are released (the activations were
            # consumed by the main loop); the sink word is fl[11], never read
            sink = _smem_addr(fl.iterator) + Int32(44)
            _sink4(sink, r0, r1, r2, r3)
            if const_expr(row_ar):
                for j in cutlass.range_constexpr(vpt):
                    _sink4(sink, rres[4 * j + 0], rres[4 * j + 1], rres[4 * j + 2], rres[4 * j + 3])
        cute.arch.sync_threads()
        # PDL invariant: dependents are released only after this CTA barrier, which follows
        # the last access to any allocator-recyclable memory (activations consumed by the main loop, residual sunk
        # into smem above); afterwards the kernel touches only the persistent Lamport workspace / flags and the
        # caller-held outputs, so a dependent's pre-wait write into a recycled block cannot race it.
        if const_expr(self.use_pdl and not _PDL_TEST_ENTRY_TRIGGER):
            griddepcontrol_launch_dependents()
        cur = fl[0]
        dirty = fl[1]
        buf_bytes = Int64(fl[2])
        dirty_stages = fl[3]
        sb0 = fl[4]
        sb1 = fl[5]
        sb2 = fl[6]
        sb3 = fl[7]
        epoch = fl[8]
        if const_expr(not row_ar):
            # column mode: CTA 0 hands the buffers on once every CTA has counted its flag read. Row mode needs
            # no counter: CTA 0 updates the flags after its whole row arrived, i.e. after every CTA of every
            # rank stored its tile, which each does only after reading the flags.
            if tidx == 0:
                _red_release_add_s32(flags + Int64(32), Int32(1))
        for idx in cutlass.range(tidx, num_rows * cols, nthreads):
            rr = idx // cols
            cc = idx % cols
            value = Float32(0.0)
            for w in cutlass.range_constexpr(k_warps):
                value = value + red[w, rr, cc]
            tile[rr, cc] = (value * alpha).to(BFloat16)
        cute.arch.sync_threads()
        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbg + Int64(24), _globaltimer())

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
            if const_expr(two_shot):
                lt_row = row // Int32(nranks)
                _st_v4(
                    peer
                    + data_off
                    + (
                        (Int64(lt_row) * Int64(nranks) + Int64(rank)) * Int64(vecs)
                        + Int64(col0 // VEC)
                    )
                    * Int64(16),
                    _sanitize(x0),
                    _sanitize(x1),
                    _sanitize(x2),
                    _sanitize(x3),
                )
            else:
                _st_v4(
                    mcast
                    + data_off
                    + (Int64(rank) * tot + Int64(row) * Int64(vecs) + Int64(col0 // VEC))
                    * Int64(16),
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

        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbg + Int64(32), _globaltimer())

        if const_expr(row_ar):
            # ---- row all-reduce: this CTA's row `cta` from every rank (vpt vectors per thread), FP32 sum in rank
            # order -> BF16, + residual; RMSNorm: BF16 squares summed per thread (vector, then element order),
            # per warp (butterfly), per CTA (butterfly over the warp sums) ----
            if ar_cta:
                res = cute.make_rmem_tensor((VEC * vpt,), Float32)
                if const_expr(two_shot):
                    half = buf_bytes // Int64(2)  # BROADCAST stage offset within the current buffer
                    lt = cta // Int32(nranks)
                    if cta % Int32(nranks) == rank:
                        # owner: every rank's vectors of this row from the SCATTER stage, FP32 rank-order sum,
                        # BF16, multicast into every rank's BROADCAST stage
                        gs = cute.make_rmem_tensor((vpt * nranks * 4,), Int32)
                        pend_o = Int32(1)
                        while pend_o != 0:
                            pend_o = Int32(0)
                            for j in cutlass.range_constexpr(vpt):
                                for rk in cutlass.range_constexpr(nranks):
                                    a, b, cc2, d = _ld_volatile_v4(
                                        own
                                        + data_off
                                        + (
                                            (Int64(lt) * Int64(nranks) + Int64(rk)) * Int64(vecs)
                                            + Int64(tidx)
                                            + Int64(j * nthreads)
                                        )
                                        * Int64(16)
                                    )
                                    g0 = (j * nranks + rk) * 4
                                    gs[g0 + 0] = a
                                    gs[g0 + 1] = b
                                    gs[g0 + 2] = cc2
                                    gs[g0 + 3] = d
                                    if a == neg0:
                                        pend_o = Int32(1)
                                    if b == neg0:
                                        pend_o = Int32(1)
                                    if cc2 == neg0:
                                        pend_o = Int32(1)
                                    if d == neg0:
                                        pend_o = Int32(1)
                        for j in cutlass.range_constexpr(vpt):
                            rsum = cute.make_rmem_tensor((4,), Int32)
                            for i in cutlass.range_constexpr(4):
                                slo = Float32(0.0)
                                shi = Float32(0.0)
                                for rk in cutlass.range_constexpr(nranks):
                                    slo = _add(slo, _lo(gs[(j * nranks + rk) * 4 + i]))
                                    shi = _add(shi, _hi(gs[(j * nranks + rk) * 4 + i]))
                                rsum[i] = _sanitize(_pack_bf16(slo, shi))
                            _st_v4(
                                mcast
                                + data_off
                                + half
                                + (ar_base + Int64(j * nthreads)) * Int64(16),
                                rsum[0],
                                rsum[1],
                                rsum[2],
                                rsum[3],
                            )
                    # every rank: the reduced row from the BROADCAST stage, + residual (BF16)
                    gb = cute.make_rmem_tensor((vpt * 4,), Int32)
                    pend_b = Int32(1)
                    while pend_b != 0:
                        pend_b = Int32(0)
                        for j in cutlass.range_constexpr(vpt):
                            a, b, cc2, d = _ld_volatile_v4(
                                own + data_off + half + (ar_base + Int64(j * nthreads)) * Int64(16)
                            )
                            gb[4 * j + 0] = a
                            gb[4 * j + 1] = b
                            gb[4 * j + 2] = cc2
                            gb[4 * j + 3] = d
                            if a == neg0:
                                pend_b = Int32(1)
                            if b == neg0:
                                pend_b = Int32(1)
                            if cc2 == neg0:
                                pend_b = Int32(1)
                            if d == neg0:
                                pend_b = Int32(1)
                    for j in cutlass.range_constexpr(vpt):
                        for i in cutlass.range_constexpr(4):
                            res[VEC * j + 2 * i] = _bf16r(
                                _add(_lo(gb[4 * j + i]), _lo(rres[4 * j + i]))
                            )
                            res[VEC * j + 2 * i + 1] = _bf16r(
                                _add(_hi(gb[4 * j + i]), _hi(rres[4 * j + i]))
                            )
                else:
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
                            res[VEC * j + 2 * i + 1] = _bf16r(
                                _add(_bf16r(hi), _hi(rres[4 * j + i]))
                            )
                for j in cutlass.range_constexpr(vpt):
                    _st_v4(
                        Int64(gResOut.iterator.toint())
                        + (ar_base + Int64(j * nthreads)) * Int64(16),
                        _pack_bf16(res[VEC * j + 0], res[VEC * j + 1]),
                        _pack_bf16(res[VEC * j + 2], res[VEC * j + 3]),
                        _pack_bf16(res[VEC * j + 4], res[VEC * j + 5]),
                        _pack_bf16(res[VEC * j + 6], res[VEC * j + 7]),
                    )
                if const_expr(self.stamps):
                    if tidx == 0:
                        _st_s64(dbg + Int64(40), _globaltimer())
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
                if const_expr(self.stamps):
                    if tidx == 0:
                        _st_s64(dbg + Int64(48), _globaltimer())
                inv = Float32(1.0)
                if const_expr(self.quant):
                    inv = _div_rn(Float32(1.0), gScale[0])
                for j in cutlass.range_constexpr(vpt):
                    y = cute.make_rmem_tensor((VEC,), Float32)
                    for i in cutlass.range_constexpr(4):
                        if const_expr(two_shot):  # rmsNormLamport: (w * x) * rstd
                            y[2 * i] = _bf16r(
                                _mul(_mul(_lo(nwr[4 * j + i]), res[VEC * j + 2 * i]), rstd)
                            )
                            y[2 * i + 1] = _bf16r(
                                _mul(_mul(_hi(nwr[4 * j + i]), res[VEC * j + 2 * i + 1]), rstd)
                            )
                        else:  # one-shot: (x * rstd) * w
                            y[2 * i] = _bf16r(
                                _mul(_mul(res[VEC * j + 2 * i], rstd), _lo(nwr[4 * j + i]))
                            )
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
        else:
            # ---- wait for this slice of every rank; FP32 sum in rank order -> BF16; + residual ----
            res = cute.make_rmem_tensor((VEC,), Float32)
            for e in cutlass.range_constexpr(VEC):
                res[e] = Float32(0.0)
            psum = Float32(0.0)
            if active:
                got = cute.make_rmem_tensor((nranks * 4,), Int32)
                pending = Int32(1)
                while pending != 0:
                    pending = Int32(0)
                    for rk in cutlass.range_constexpr(nranks):
                        a, b, cc2, d = _ld_volatile_v4(
                            own
                            + data_off
                            + (Int64(rk) * tot + Int64(row) * Int64(vecs) + Int64(col0 // VEC))
                            * Int64(16)
                        )
                        got[rk * 4 + 0] = a
                        got[rk * 4 + 1] = b
                        got[rk * 4 + 2] = cc2
                        got[rk * 4 + 3] = d
                        if a == neg0:
                            pending = Int32(1)
                        if b == neg0:
                            pending = Int32(1)
                        if cc2 == neg0:
                            pending = Int32(1)
                        if d == neg0:
                            pending = Int32(1)
                rw = (r0, r1, r2, r3)
                for i in cutlass.range_constexpr(4):
                    lo = Float32(0.0)
                    hi = Float32(0.0)
                    for rk in cutlass.range_constexpr(nranks):
                        lo = _add(lo, _lo(got[rk * 4 + i]))
                        hi = _add(hi, _hi(got[rk * 4 + i]))
                    res[2 * i] = _bf16r(_add(_bf16r(lo), _lo(rw[i])))
                    res[2 * i + 1] = _bf16r(_add(_bf16r(hi), _hi(rw[i])))
                _st_v4(
                    Int64(gResOut.iterator.toint()) + row_off,
                    _pack_bf16(res[0], res[1]),
                    _pack_bf16(res[2], res[3]),
                    _pack_bf16(res[4], res[5]),
                    _pack_bf16(res[6], res[7]),
                )
                for e in cutlass.range_constexpr(VEC):
                    psum = _add(psum, _bf16r(_mul(res[e], res[e])))
                vpart[item] = psum
            cute.arch.sync_threads()
            if const_expr(self.stamps):
                if tidx == 0:
                    _st_s64(dbg + Int64(40), _globaltimer())

            # ---- per-row partial over this CTA's columns -> epoch-tagged packet in the exchange buffer ----
            slot = epoch & Int32(1)
            part_base = Int64(gPart.iterator.toint()) + Int64(slot) * Int64(
                PART_SLOT_ROWS * self.max_ctas * 8
            )
            if tidx < num_rows:
                rs = Float32(0.0)
                for vv in cutlass.range_constexpr(vpr):
                    rs = _add(rs, vpart[tidx * vpr + vv])
                _st_s64(
                    part_base + (Int64(tidx) * Int64(self.max_ctas) + Int64(cta)) * Int64(8),
                    _packet(epoch, _f32_bits(rs)),
                )

            # ---- every CTA: all CTAs' partials of every row -> rstd. Every (row, CTA) packet has one owning
            # thread, all of a thread's packets are polled together (independent loads, not one round trip per
            # row); the per-row sums run in smem in a fixed order (lane-strided over CTAs, then a butterfly),
            # identical in every CTA, so every CTA gets the same rstd ----
            per_thread: cutlass.Constexpr = (items + nthreads - 1) // nthreads
            pk = cute.make_rmem_tensor((per_thread,), Int64)
            for j in cutlass.range_constexpr(per_thread):
                pk[j] = Int64(0)  # tag 0: never an epoch
            pend = Int32(1)
            while pend != 0:
                pend = Int32(0)
                for j in cutlass.range_constexpr(per_thread):
                    it = tidx + j * nthreads
                    if it < items:
                        if _pk_tag(pk[j]) != epoch:
                            pk[j] = _ld_volatile_s64(
                                part_base
                                + (Int64(it // ctas) * Int64(self.max_ctas) + Int64(it % ctas))
                                * Int64(8)
                            )
                            pend = Int32(1)
            for j in cutlass.range_constexpr(per_thread):
                it = tidx + j * nthreads
                if it < items:
                    vals[it] = _bits_f32(_pk_val(pk[j]))
            cute.arch.sync_threads()
            rows_per_warp: cutlass.Constexpr = (num_rows + k_warps - 1) // k_warps
            for jr in cutlass.range_constexpr(rows_per_warp):
                rr = warp + jr * k_warps
                if rr < num_rows:
                    s = Float32(0.0)
                    for c0 in cutlass.range_constexpr((ctas + 31) // 32):
                        cl = lane + c0 * 32
                        if cl < ctas:
                            s = _add(s, vals[rr * ctas + cl])
                    for k in cutlass.range_constexpr(5):
                        s = _add(s, _shfl_bfly(s, Int32(16 >> k)))
                    if lane == 0:
                        wsum[k_warps, rr] = _rsqrt(
                            _add(_div_rn(s, Float32(float(hidden))), Float32(self.norm_eps))
                        )
            cute.arch.sync_threads()
            if const_expr(self.stamps):
                if tidx == 0:
                    _st_s64(dbg + Int64(48), _globaltimer())

            # ---- normalize this thread's vector: (x * rstd) * w (+ FP8) ----
            if active:
                rstd = wsum[k_warps, row]
                ww = (nw0, nw1, nw2, nw3)
                y = cute.make_rmem_tensor((VEC,), Float32)
                for i in cutlass.range_constexpr(4):
                    y[2 * i] = _bf16r(_mul(_mul(res[2 * i], rstd), _lo(ww[i])))
                    y[2 * i + 1] = _bf16r(_mul(_mul(res[2 * i + 1], rstd), _hi(ww[i])))
                if const_expr(self.norm_out):
                    _st_v4(
                        Int64(gOut.iterator.toint()) + row_off,
                        _pack_bf16(y[0], y[1]),
                        _pack_bf16(y[2], y[3]),
                        _pack_bf16(y[4], y[5]),
                        _pack_bf16(y[6], y[7]),
                    )
                if const_expr(self.quant):
                    inv = _div_rn(Float32(1.0), gScale[0])
                    q01 = _e4m3x2(_mul(y[0], inv), _mul(y[1], inv))
                    q23 = _e4m3x2(_mul(y[2], inv), _mul(y[3], inv))
                    q45 = _e4m3x2(_mul(y[4], inv), _mul(y[5], inv))
                    q67 = _e4m3x2(_mul(y[6], inv), _mul(y[7], inv))
                    _st_v2(
                        Int64(gQuant.iterator.toint())
                        + (Int64(row) * Int64(vecs) + Int64(col0 // VEC)) * Int64(8),
                        q01 | (q23 << 16),
                        q45 | (q67 << 16),
                    )

        if const_expr(self.stamps):
            if tidx == 0:
                _st_s64(dbg + Int64(56), _globaltimer())

        # ---- hand the triple buffer and the epoch to the next call once every CTA has read them ----
        if cta == 0:
            if tidx == 0:
                if const_expr(not row_ar):
                    arrived = _ld_acquire_s32(flags + Int64(32))
                    while arrived != Int32(ctas):
                        arrived = _ld_acquire_s32(flags + Int64(32))
                _st_s32(flags, (cur + Int32(1)) % Int32(3))
                _st_s32(flags + Int64(4), cur)
                if const_expr(two_shot):
                    _st_s32(flags + Int64(12), Int32(2))
                    _st_s32(
                        flags + Int64(16),
                        ((Int32(m) + Int32(nranks - 1)) // Int32(nranks))
                        * Int32(hidden * nranks * 2),
                    )
                    _st_s32(flags + Int64(20), Int32(m) * Int32(hidden * 2))
                else:
                    _st_s32(flags + Int64(12), Int32(1))
                    _st_s32(flags + Int64(16), Int32(m) * Int32(hidden * nranks * 2))
                    _st_s32(flags + Int64(20), Int32(0))
                _st_s32(flags + Int64(24), Int32(0))
                _st_s32(flags + Int64(28), Int32(0))
                _st_s32(flags + Int64(32), Int32(0))
                if const_expr(not row_ar):
                    nxt = epoch + Int32(1)
                    if nxt == Int32(0):
                        nxt = Int32(2)
                    _st_s32(Int64(gEpoch.iterator.toint()), nxt)
