# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet MTP target-verify step on tcgen05: ``trtllm::gdn_verify`` (same arguments) with FP8 output.

Per (request, value head) pair: the cached-replay delta rule over the request's T draft rows against its pn history
rows (keys, U rows, cumulative gates) and its checkpoint state h; this step's rows appended to the history, or, when
pn + T > 16, the checkpoint commit (h' = bf16(h e^gStart + sum_j Uc_j k_j^T), Uc_j = u_j e^(gStart - G_j) split
into bf16 hi + lo, TMA-stored over h) with this step's rows starting the other history buffer; then the gated RMSNorm
and the E4M3 quantization of the out_proj input.

Work decomposition: persistent 128-thread CTAs (up to 4 per SM), each looping over pairs; thread v owns value column
v (TMEM lane v). For up to 128 requests the pairs are ordered so that the committing (folding) requests are spread
evenly over every CTA's pairs; above that, pair p = (request p / HV, head p % HV) and CTA b takes b, b + grid, ...
While a pair runs, the next pair's scalars are loaded and its checkpoint, history U and history-key rows are fetched
by TMA as soon as the current pair has released those buffers.

Shared memory per CTA: two 20 KB regions, one per 64-wide key-dim half (128-byte swizzle, K-major 128-byte rows):
  rows 0-15   [k_0..k_{T-1}, 0.., q_0..q_{T-1}, 0..] raw (thread-written; rows 8-15 later hold the Gram rows)
  rows 16-31  history keys (TMA, rows >= pn zeroed)
  rows 32-159 checkpoint state h (TMA; folded in place and TMA-stored on a commit)
plus the history U [v half][16 j][64 v] (TMA, MN-major), Uc hi / lo, the bf16 history coefficients, an FP32 scalar
block and the request-order table.
TMEM (128 columns): 0-15 state x [k|q], 16-31 Gram ([k|q|kh] x [k|q]), 32-47 history products, 64-127 fold chunk 0
(kd 0-63); fold chunk 1 (kd 64-127) reuses 0-63 once the output path has read 0-47.

PDL (``TRTLLM_ENABLE_PDL``, default on): barrier init, TMEM alloc, the order table and the first pair's scalar loads
and TMA run before griddepcontrol.wait; dependents are released when each CTA starts its last pair. With PDL off the
kernel is launched without the programmatic attribute, which makes both instructions no-ops.

Precision: every stage has the arithmetic and roundings of ``trtllm::gdn_verify``; results differ from it only
through the MMA accumulation order (tcgen05 instead of mma.sync) and the gated norm's 128-column sum order.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as MT
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from ...flashinfer_utils import get_env_enable_pdl

D = 128
HIST = 16
MIN_T, MAX_T = 2, 8  # draft rows per request: k and q rows each fit 8 of the 16 [k|q] rows
NTHR = 128
ROWB = 128  # bytes per swizzled row (64 bf16)
REG_ROWS = 160
REG_E = REG_ROWS * 64  # elements per region
REG_B = REG_ROWS * ROWB  # 20480 bytes
KH_ROW, H_ROW = 16, 32
UH_E = 2 * HIST * 64  # [v half][16 j][64]
TCOLS = 128
CTAS_PER_SM = 4  # resident CTAs per SM (registers capped to 65536 / (4 x 128) = 128 by minctasm)
QSCALE = 0.08838834764831845  # 1 / sqrt(128)
NMAX = 128  # request-order table entries (n > NMAX keeps the identity order)
NCH = NMAX // 32  # table chunks, unrolled so the build costs two load latencies
# small FP32 smem block, float offsets: beta, e^G / |k|, e^G scale / |q|, history gates, A, c, norm partials, 1 / norms
FS_BETA, FS_EGK, FS_EGQ, FS_GH = 0, 8, 16, 32
FS_A, FS_C, FS_NORM, FS_INV = 48, 112, 176, 208
FS_N = 224


# ----------------------------------------------------------------------------------------------- PTX helpers
def _asm(ret, args, ptx, cons, side, loc, ip):
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        ptx,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _f32op(ptx, nargs):
    cons = "=f" + ",f" * nargs

    @dsl_user_op
    def op(*args, loc=None, ip=None):
        return Float32(_asm(MT.f32(), [Float32(a) for a in args], ptx, cons, False, loc, ip))

    return op


_fma = _f32op("fma.rn.f32 $0, $1, $2, $3;", 3)
_fms = _f32op(
    "{\n\t.reg .f32 t;\n\tneg.f32 t, $1;\n\tfma.rn.f32 $0, t, $2, $3;\n\t}", 3
)  # c - a * b
_mul = _f32op("mul.rn.f32 $0, $1, $2;", 2)
_add = _f32op("add.rn.f32 $0, $1, $2;", 2)
_sub = _f32op("sub.rn.f32 $0, $1, $2;", 2)
# __expf / __log2f / __fdividef as libdevice lowers them without FTZ
_expf = _f32op(
    "{\n\t.reg .f32 t;\n\tmul.rn.f32 t, $1, 0f3FB8AA3B;\n\tex2.approx.f32 $0, t;\n\t}", 1
)
_lg2 = _f32op("lg2.approx.f32 $0, $1;", 1)
_fdiv = _f32op("div.approx.f32 $0, $1, $2;", 2)
_div_rn = _f32op("div.rn.f32 $0, $1, $2;", 2)
_rcp_ftz = _f32op("rcp.approx.ftz.f32 $0, $1;", 1)
_sqrt_ftz = _f32op("sqrt.approx.ftz.f32 $0, $1;", 1)
_rsqrt_ftz = _f32op("rsqrt.approx.ftz.f32 $0, $1;", 1)
_bf16r = _f32op(
    "{\n\t.reg .b16 b, z;\n\tmov.b16 z, 0;\n\tcvt.rn.bf16.f32 b, $1;\n\tmov.b32 $0, {z, b};\n\t}", 1
)


@dsl_user_op
def _ld_u16(addr, *, loc=None, ip=None) -> Int32:
    """The 16 raw bits at [addr], zero-extended (no conversion here, so the consumer can sit at its use)."""
    return Int32(
        _asm(
            MT.i32(),
            [Int64(addr)],
            "{\n\t.reg .b16 h;\n\tld.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,l,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16f(raw, *, loc=None, ip=None) -> Float32:
    """bf16 bits (low half of raw) -> the exact FP32 value."""
    return Float32(_asm(MT.f32(), [Int32(raw)], "shl.b32 $0, $1, 16;", "=f,r", False, loc, ip))


def _fold8x2(w, acc, eg):
    """h' = bf16(h e^gStart + acc) for 8 values (h = four bf16x2 words), FMA rn as packed f32x2 pairs, one rounding."""
    out = []
    for i in range(4):
        lo, hi = cute.arch.inline_ptx(
            "{\n\tshl.b32 {$w0}, {$r0}, 16;\n\tand.b32 {$w1}, {$r0}, 0xffff0000;\n\t}",
            write_only_types=[Float32, Float32],
            read_only_args=[Int32(w[i])],
        )
        r0, r1 = prims.fma_packed_f32x2(
            (lo, hi),
            (Float32(eg), Float32(eg)),
            (Float32(acc[2 * i]), Float32(acc[2 * i + 1])),
            rnd=prims.FPRoundingMode.RN,
        )
        out.append(
            cute.arch.inline_ptx(
                "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};",
                write_only_types=[Int32],
                read_only_args=[Float32(r0), Float32(r1)],
            )
        )
    return out


@dsl_user_op
def _ld_s32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(MT.i32(), [Int64(addr)], "ld.global.s32 $0, [$1];", "=r,l,~{memory}", True, loc, ip)
    )


@dsl_user_op
def _ld_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(MT.f32(), [Int64(addr)], "ld.global.f32 $0, [$1];", "=f,l,~{memory}", True, loc, ip)
    )


@dsl_user_op
def _ld_bf16(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            MT.f32(),
            [Int64(addr)],
            "{\n\t.reg .b16 h, z;\n\tmov.b16 z, 0;\n\tld.global.b16 h, [$1];\n\t"
            "mov.b32 $0, {z, h};\n\t}",
            "=f,l,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_f32(addr, v, *, loc=None, ip=None) -> None:
    _asm(None, [Int64(addr), Float32(v)], "st.global.f32 [$0], $1;", "l,f,~{memory}", True, loc, ip)


@dsl_user_op
def _st_bf16(addr, v, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr), Float32(v)],
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _st_e4m3(addr, v, *, loc=None, ip=None) -> None:
    """cvt.rn.satfinite.e4m3x2.f32 of (0, v); the low byte (v) to [addr]."""
    _asm(
        None,
        [Int64(addr), Float32(v)],
        "{\n\t.reg .b16 h;\n\tcvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, $1;\n\t"
        "st.global.b8 [$0], h;\n\t}",
        "l,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _st_zero8(addr, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr)],
        "{\n\t.reg .b16 z;\n\tmov.b16 z, 0;\n\tst.global.b8 [$0], z;\n\t}",
        "l,~{memory}",
        True,
        loc,
        ip,
    )


def _ldg128(addr):
    return cute.arch.inline_ptx(
        "ld.global.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32] * 4,
        read_only_args=[Int64(addr)],
    )


def _lds128(saddr):
    return cute.arch.inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32] * 4,
        read_only_args=[Int32(saddr)],
    )


def _sts128(saddr, w):
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[Int32(saddr), Int32(w[0]), Int32(w[1]), Int32(w[2]), Int32(w[3])],
    )


def _lds128f(saddr):
    return cute.arch.inline_ptx(
        "ld.shared.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32] * 4,
        read_only_args=[Int32(saddr)],
    )


def _sts128f(saddr, v):
    cute.arch.inline_ptx(
        "st.shared.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[Int32(saddr), Float32(v[0]), Float32(v[1]), Float32(v[2]), Float32(v[3])],
    )


def _lds_f32(saddr):
    return cute.arch.inline_ptx(
        "ld.shared.f32 {$w0}, [{$r0}];", write_only_types=[Float32], read_only_args=[Int32(saddr)]
    )


def _sts_f32(saddr, v):
    cute.arch.inline_ptx("st.shared.f32 [{$r0}], {$r1};", read_only_args=[Int32(saddr), Float32(v)])


def _sts_bf16(saddr, v):
    cute.arch.inline_ptx(
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, {$r1};\n\tst.shared.b16 [{$r0}], h;\n\t}",
        read_only_args=[Int32(saddr), Float32(v)],
    )


def _lds_bf16(saddr):
    return cute.arch.inline_ptx(
        "{\n\t.reg .b16 h, z;\n\tmov.b16 z, 0;\n\tld.shared.b16 h, [{$r0}];\n\t"
        "mov.b32 {$w0}, {z, h};\n\t}",
        write_only_types=[Float32],
        read_only_args=[Int32(saddr)],
    )


def _uc_split(w, cd):
    """Four bf16x2 words of history U (8 values) times cd (FP32), each split into bf16 hi + lo (RN):
    hi = bf16(u cd), lo = bf16(u cd - hi). Returns 4 hi words, 4 lo words."""
    body = "{\n\t.reg .f32 u<8>, h<8>;\n\t.reg .b32 t;\n\t"
    for i in range(4):
        body += (
            f"shl.b32 t, {{$r{i}}}, 16;\n\tmov.b32 u{2 * i}, t;\n\t"
            f"and.b32 t, {{$r{i}}}, 0xffff0000;\n\tmov.b32 u{2 * i + 1}, t;\n\t"
            f"mul.rn.f32 u{2 * i}, u{2 * i}, {{$r4}};\n\tmul.rn.f32 u{2 * i + 1}, u{2 * i + 1}, {{$r4}};\n\t"
            f"cvt.rn.bf16x2.f32 {{$w{i}}}, u{2 * i + 1}, u{2 * i};\n\t"
            f"shl.b32 t, {{$w{i}}}, 16;\n\tmov.b32 h{2 * i}, t;\n\t"
            f"and.b32 t, {{$w{i}}}, 0xffff0000;\n\tmov.b32 h{2 * i + 1}, t;\n\t"
            f"sub.rn.f32 h{2 * i}, u{2 * i}, h{2 * i};\n\tsub.rn.f32 h{2 * i + 1}, u{2 * i + 1}, h{2 * i + 1};\n\t"
            f"cvt.rn.bf16x2.f32 {{$w{4 + i}}}, h{2 * i + 1}, h{2 * i};\n\t"
        )
    body += "}"
    return cute.arch.inline_ptx(
        body,
        write_only_types=[Int32] * 8,
        read_only_args=[Int32(w[0]), Int32(w[1]), Int32(w[2]), Int32(w[3]), Float32(cd)],
    )


def _sf(c, a, b):
    """Float32 select (explicit, so no trace-time ternary on a dynamic condition)."""
    return Float32(cutlass.select_(c, Float32(a), Float32(b)))


def _si(c, a, b):
    return Int32(cutlass.select_(c, Int32(a), Int32(b)))


def _saddr(arr):
    return Int32(llvm.ptrtoint(Int32.mlir_type, arr.ir_value()))


def _shfl(v, src):
    return cute.arch.shuffle_sync(v, src)


def _bfly(v, m):
    return cute.arch.shuffle_sync_bfly(v, m)


@dsl_user_op
def _ballot(v, *, loc=None, ip=None) -> Int32:
    """vote.sync.ballot over the full warp of (v != 0)."""
    return Int32(
        _asm(
            MT.i32(),
            [Int32(v)],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tvote.sync.ballot.b32 $0, p, 0xffffffff;\n\t}",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _popc(v, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(MT.i32(), [Int32(v)], "popc.b32 $0, $1;", "=r,r", False, loc, ip))


@dsl_user_op
def _lanemask_lt(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(MT.i32(), [], "mov.u32 $0, %lanemask_lt;", "=r", False, loc, ip))


def _sts_s32(saddr, v):
    cute.arch.inline_ptx("st.shared.s32 [{$r0}], {$r1};", read_only_args=[Int32(saddr), Int32(v)])


def _lds_s32(saddr):
    return cute.arch.inline_ptx(
        "ld.shared.s32 {$w0}, [{$r0}];", write_only_types=[Int32], read_only_args=[Int32(saddr)]
    )


# ----------------------------------------------------------------------------------------------- kernel
@cute.kernel
def _gdn_verify_tcgen05_kernel(
    tm_h: cutlass.GridConstant[cuda_exp.TensorMap],
    tm_u: cutlass.GridConstant[cuda_exp.TensorMap],
    tm_k: cutlass.GridConstant[cuda_exp.TensorMap],
    gX: cute.Tensor,
    gZ: cute.Tensor,
    gA: cute.Tensor,
    gB: cute.Tensor,
    gAlog: cute.Tensor,
    gDtb: cute.Tensor,
    gNw: cute.Tensor,
    gScale: cute.Tensor,
    gSlots: cute.Tensor,
    gRepl: cute.Tensor,
    gOldU: cute.Tensor,
    gOldK: cute.Tensor,
    gOldG: cute.Tensor,
    gCbi: cute.Tensor,
    gPacc: cute.Tensor,
    gY: cute.Tensor,
    n_pairs: Int32,
    n_ctas: Int32,
    xs: Int32,
    zs: Int32,
    abst: Int32,
    uss: Int64,
    kss: Int64,
    gss: Int64,
    cbs: Int64,
    pas: Int64,
    eps: Float32,
    T_: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    HK: cutlass.Constexpr[int],
):
    GROUP: cutlass.Constexpr[int] = HV // HK
    BF: cutlass.Constexpr = cutlass.BFloat16
    SW: cutlass.Constexpr = prims.Tcgen05SmemSwizzle.SWIZZLE_128B

    reg = cutlass.Array(BF, 2 * REG_E, space=cutlass.AddressSpace.smem, alignment=1024)
    uhs = cutlass.Array(BF, UH_E, space=cutlass.AddressSpace.smem, alignment=1024)
    ucs = cutlass.Array(BF, 2 * UH_E, space=cutlass.AddressSpace.smem, alignment=1024)
    cfs = cutlass.Array(BF, 256, space=cutlass.AddressSpace.smem, alignment=128)
    fs = cutlass.Array(Float32, FS_N, space=cutlass.AddressSpace.smem, alignment=16)
    bars = cutlass.Array(Int64, 5, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_i32 = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    ordt = cutlass.Array(Int32, NMAX, space=cutlass.AddressSpace.smem, alignment=16)
    ordf = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)

    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tid % Int32(32)
    pX = Int64(gX.iterator.toint())
    pZ = Int64(gZ.iterator.toint())
    pY = Int64(gY.iterator.toint())
    sR = _saddr(reg)
    sU = _saddr(uhs)
    sUc = _saddr(ucs)
    sCf = _saddr(cfs)
    sF = _saddr(fs)
    s_order = _saddr(ordt)
    sOrdF = _saddr(ordf)
    pSl = Int64(gSlots.iterator.toint())
    # replay slots: the rows of the history (old_u / old_k / old_g, slot strides uss / kss / gss) and of its
    # bookkeeping (cache_buf_idx / prev_acc, strides cbs / pas); a padded request (state slot < 0) reads replay slot 0
    pRs = Int64(gRepl.iterator.toint())
    pPa = Int64(gPacc.iterator.toint())
    pCb = Int64(gCbi.iterator.toint())
    pOg = Int64(gOldG.iterator.toint())
    pAl = Int64(gAlog.iterator.toint())
    pDt = Int64(gDtb.iterator.toint())
    nw = _bf16f(_ld_u16(Int64(gNw.iterator.toint()) + Int64(tid) * Int64(2)))
    rscale = _div_rn(Float32(1.0), _ld_f32(Int64(gScale.iterator.toint())))

    # ---- once per CTA: warp 0 sets up the barriers and TMEM while warp 1 builds the request order ----
    nreq = n_pairs // Int32(HV)
    use_tab = nreq <= Int32(NMAX)
    if warp == Int32(0):
        if prims.elect_sync():
            for i in cutlass.range_constexpr(5):
                prims.mbarrier_init(bars.subview(i), 1)
        prims.fence_mbarrier_init()
        prims.tcgen05_alloc(tmem_i32, TCOLS, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    if warp == Int32(1):
        if use_tab:
            # folding requests (pn + T > HIST, slot >= 0) take positions 0, 1, ...; the others nreq - 1, ...
            s_l = []
            for ci in cutlass.range_constexpr(NCH):
                s_l.append(
                    _ld_s32(
                        pSl + Int64(cutlass.min(Int32(32 * ci) + lane, nreq - Int32(1))) * Int64(4)
                    )
                )
            pa_l = []
            for ci in cutlass.range_constexpr(NCH):
                r_l = _ld_s32(
                    pRs + Int64(cutlass.min(Int32(32 * ci) + lane, nreq - Int32(1))) * Int64(4)
                )
                r_l = _si(s_l[ci] >= Int32(0), r_l, 0)
                pa_l.append(_ld_s32(pPa + Int64(r_l) * pas * Int64(4)))
            lt_q = _lanemask_lt()
            fpos = Int32(0)
            opos = Int32(0)
            for ci in cutlass.range_constexpr(NCH):
                r_q = Int32(32 * ci) + lane
                ok_q = r_q < nreq
                fi_q = _si(
                    ok_q & (s_l[ci] >= Int32(0)) & ((pa_l[ci] + Int32(T_)) > Int32(HIST)), 1, 0
                )
                oi_q = _si(ok_q, 1, 0) - fi_q
                bf_q = _ballot(fi_q)
                bo_q = _ballot(oi_q)
                pos_q = _si(
                    fi_q > Int32(0),
                    fpos + _popc(bf_q & lt_q),
                    nreq - Int32(1) - opos - _popc(bo_q & lt_q),
                )
                if ok_q:
                    _sts_s32(s_order + pos_q * Int32(4), r_q)
                fpos = fpos + _popc(bf_q)
                opos = opos + _popc(bo_q)
            if lane == Int32(0):
                _sts_s32(sOrdF, fpos)
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    base = prims.TmemAddr(tmem_raw)
    npc = (
        n_pairs - bidx + n_ctas - Int32(1)
    ) // n_ctas  # pairs of this CTA: bidx, bidx + n_ctas, ...

    # ---- pair 0: its scalars and its checkpoint / history TMA, before the dependency wait ----
    # This CTA's pairs are b + G k' (the folding ones at k' < c_b); slot k takes a folding pair iff
    # rem_k + c_b >= K, rem_k = ((k + phi) c_b) mod K, phi = b mod K; icnt = folding pairs in slots before k.
    fcnt = _si(use_tab, _lds_s32(sOrdF), Int32(0))
    pf_b = fcnt * Int32(HV)
    c_b = _si(pf_b > bidx, (pf_b - bidx + n_ctas - Int32(1)) // n_ctas, Int32(0))
    k_b = cutlass.max(npc, Int32(1))
    phi_b = bidx - (bidx // k_b) * k_b
    rem = phi_b * c_b - ((phi_b * c_b) // k_b) * k_b
    icnt = Int32(0)
    isf0 = (rem + c_b) >= k_b
    pnx = bidx + _si(isf0, icnt, c_b - icnt) * n_ctas
    rem = rem + c_b - _si(isf0, k_b, Int32(0))
    icnt = icnt + _si(isf0, 1, 0)
    pq = pnx
    rq_n = _si(
        use_tab,
        _lds_s32(s_order + cutlass.min(pnx // Int32(HV), Int32(NMAX - 1)) * Int32(4)),
        pnx // Int32(HV),
    )
    slot_n = _ld_s32(pSl + Int64(rq_n) * Int64(4))
    rs_n = _ld_s32(pRs + Int64(rq_n) * Int64(4))
    # slot 1's position, request and slot (slot 0's when the CTA has one pair)
    isf1 = (rem + c_b) >= k_b
    pq_nn = _si(npc > Int32(1), bidx + _si(isf1, icnt, c_b + Int32(1) - icnt) * n_ctas, pnx)
    rem = rem + c_b - _si(isf1, k_b, Int32(0))
    icnt = icnt + _si(isf1, 1, 0)
    rq_nn = _si(
        use_tab,
        _lds_s32(s_order + cutlass.min(pq_nn // Int32(HV), Int32(NMAX - 1)) * Int32(4)),
        pq_nn // Int32(HV),
    )
    slot_nn = _ld_s32(pSl + Int64(rq_nn) * Int64(4))
    rs_nn = _ld_s32(pRs + Int64(rq_nn) * Int64(4))
    rs_n = _si(slot_n >= Int32(0), rs_n, 0)
    hvn = pnx - (pnx // Int32(HV)) * Int32(HV)
    pn_n = _ld_s32(pPa + Int64(rs_n) * pas * Int64(4))
    buf_n = _ld_s32(pCb + Int64(rs_n) * cbs * Int64(4))
    g0n = (
        pOg
        + (Int64(rs_n) * gss + Int64(hvn * Int32(HIST))) * Int64(4)
        + Int64(lane & Int32(15)) * Int64(4)
    )
    gh0_n = _ld_f32(g0n)
    gh1_n = _ld_f32(g0n + Int64(HV * HIST * 4))
    al_n = _ld_f32(pAl + Int64(hvn) * Int64(4))
    dt_n = _ld_f32(pDt + Int64(hvn) * Int64(4))
    has_nx = npc > Int32(0)
    if has_nx & (slot_n >= Int32(0)):
        hvx0 = pnx - (pnx // Int32(HV)) * Int32(HV)
        hkx0 = hvx0 // Int32(GROUP)
        nbx0 = (pn_n + Int32(7)) // Int32(8)
        if warp == Int32(0):
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(
                    bars.subview(0), Int32(2 * 16384) + nbx0 * Int32(4096)
                )
                for x in cutlass.range_constexpr(2):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        reg.subview(x * REG_E + H_ROW * 64),
                        tm_h.get_ptr(),
                        (Int32(64 * x), Int32(0), hvx0, slot_n),
                        bars.subview(0),
                    )
                for jb in cutlass.range_constexpr(2):
                    if Int32(jb) < nbx0:
                        for x in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                uhs.subview(x * HIST * 64 + jb * 8 * 64),
                                tm_u.get_ptr(),
                                (Int32(64 * x), hvx0, Int32(8 * jb), buf_n, rs_n),
                                bars.subview(0),
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                reg.subview(x * REG_E + (KH_ROW + 8 * jb) * 64),
                                tm_k.get_ptr(),
                                (Int32(64 * x), hkx0, Int32(8 * jb), buf_n, rs_n),
                                bars.subview(0),
                            )
    cute.arch.griddepcontrol_wait()

    j = Int32(0)  # pairs processed (phase of barriers 0-2)
    jf = Int32(0)  # folds done (phase of barrier 3)
    for kk in cutlass.range(npc):
        if kk == npc - Int32(1):
            # out_proj may launch once every CTA has started its last pair
            cute.arch.griddepcontrol_launch_dependents()
        # this pair's scalars (prefetched); then issue the next pair's scalar loads
        pid = pq  # this pair's position in the order; its request was located with it
        rq = rq_n
        slot = slot_n
        rslot = rs_n
        pn = pn_n
        buf = buf_n
        gh0 = gh0_n
        gh1 = gh1_n
        a_log = al_n
        dtb = dt_n
        has_nx = (kk + Int32(1)) < npc
        # the next pair was located one pair ago (this pair itself on the last one); locate slot kk + 2
        pnx = pq_nn
        rq_n = rq_nn
        slot_n = slot_nn
        rs_n = rs_nn
        isfn = (rem + c_b) >= k_b
        pnxx = _si(
            (kk + Int32(2)) < npc, bidx + _si(isfn, icnt, c_b + kk + Int32(2) - icnt) * n_ctas, pnx
        )
        rem = rem + c_b - _si(isfn, k_b, Int32(0))
        icnt = icnt + _si(isfn, 1, 0)
        pq_nn = pnxx
        rq_nn = _si(
            use_tab,
            _lds_s32(s_order + cutlass.min(pnxx // Int32(HV), Int32(NMAX - 1)) * Int32(4)),
            pnxx // Int32(HV),
        )
        slot_nn = _ld_s32(pSl + Int64(rq_nn) * Int64(4))
        rs_nn = _ld_s32(pRs + Int64(rq_nn) * Int64(4))
        pq = pnx
        rs_n = _si(slot_n >= Int32(0), rs_n, 0)
        hvn = pnx - (pnx // Int32(HV)) * Int32(HV)
        pn_n = _ld_s32(pPa + Int64(rs_n) * pas * Int64(4))
        # lane & (n_ctas >> 31) is 0 at run time but lane-dependent to the compiler, so buf_n stays a vector
        # value and its uniform copies are made at the TMA issues, not here
        buf_n = _ld_s32(pCb + (Int64(rs_n) * cbs + Int64(lane & (n_ctas >> Int32(31)))) * Int64(4))
        g0n = (
            pOg
            + (Int64(rs_n) * gss + Int64(hvn * Int32(HIST))) * Int64(4)
            + Int64(lane & Int32(15)) * Int64(4)
        )
        gh0_n = _ld_f32(g0n)
        gh1_n = _ld_f32(g0n + Int64(HV * HIST * 4))
        al_n = _ld_f32(pAl + Int64(hvn) * Int64(4))
        dt_n = _ld_f32(pDt + Int64(hvn) * Int64(4))
        n = rq
        hv = pid - (pid // Int32(HV)) * Int32(HV)
        hk = hv // Int32(GROUP)
        row0 = n * Int32(T_)
        ph = j & Int32(1)
        phf = jf & Int32(1)
        fpair = Int32(0)
        if slot < Int32(0):
            # padded request: defined zeros for out_proj; the next pair's TMA can go now (nothing is in flight)
            for t in cutlass.range_constexpr(T_):
                _st_zero8(pY + Int64(row0 + Int32(t)) * Int64(HV * D) + Int64(hv * Int32(D) + tid))
            if has_nx & (slot_n >= Int32(0)):
                hvx1 = pnx - (pnx // Int32(HV)) * Int32(HV)
                hkx1 = hvx1 // Int32(GROUP)
                nbx1 = (pn_n + Int32(7)) // Int32(8)
                if warp == Int32(0):
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            bars.subview(0), Int32(2 * 16384) + nbx1 * Int32(4096)
                        )
                        for x in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                reg.subview(x * REG_E + H_ROW * 64),
                                tm_h.get_ptr(),
                                (Int32(64 * x), Int32(0), hvx1, slot_n),
                                bars.subview(0),
                            )
                        for jb in cutlass.range_constexpr(2):
                            if Int32(jb) < nbx1:
                                for x in cutlass.range_constexpr(2):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        uhs.subview(x * HIST * 64 + jb * 8 * 64),
                                        tm_u.get_ptr(),
                                        (Int32(64 * x), hvx1, Int32(8 * jb), buf_n, rs_n),
                                        bars.subview(0),
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        reg.subview(x * REG_E + (KH_ROW + 8 * jb) * 64),
                                        tm_k.get_ptr(),
                                        (Int32(64 * x), hkx1, Int32(8 * jb), buf_n, rs_n),
                                        bars.subview(0),
                                    )
        else:
            is_write = (pn + Int32(T_)) > Int32(HIST)
            wbuf = _si(is_write, Int32(1) - buf, buf)
            woff = _si(is_write, 0, pn)
            ghs = _sf(buf != Int32(0), gh1, gh0)
            if tid < Int32(HIST):
                _sts_f32(sF + (Int32(FS_GH) + tid) * Int32(4), ghs)
            g_start = _sf(pn > Int32(0), _shfl(ghs, cutlass.max(pn - Int32(1), Int32(0))), 0.0)
            neg_a = Float32(0.0) - _expf(a_log)
            # ---- this step's rows: q / k into the swizzled [k|q] rows, v / z of this column, a / b of token lane ----
            r = tid >> Int32(3)
            c8 = tid & Int32(7)
            is_k = r < Int32(T_)
            is_q = (r >= Int32(8)) & (r < Int32(8 + T_))
            tok = _si(is_k, r, _si(is_q, r - Int32(8), 0))
            col = _si(is_k, Int32(HK * D) + hk * Int32(D), hk * Int32(D))
            for x in cutlass.range_constexpr(2):
                w = _ldg128(
                    pX
                    + (Int64(row0 + tok) * Int64(xs) + Int64(col + Int32(64 * x) + c8 * Int32(8)))
                    * Int64(2)
                )
                keep = is_k | is_q
                wz = [_si(keep, w[i], 0) for i in range(4)]
                _sts128(
                    sR + Int32(x * REG_B) + r * Int32(ROWB) + ((c8 ^ (r & Int32(7))) * Int32(16)),
                    wz,
                )
            vv = []
            zz = []
            for t in cutlass.range_constexpr(T_):
                vv.append(
                    _ld_bf16(
                        pX
                        + (
                            Int64(row0 + Int32(t)) * Int64(xs)
                            + Int64(Int32(2 * HK * D) + hv * Int32(D) + tid)
                        )
                        * Int64(2)
                    )
                )
                zz.append(
                    _ld_bf16(
                        pZ
                        + (Int64(row0 + Int32(t)) * Int64(zs) + Int64(hv * Int32(D) + tid))
                        * Int64(2)
                    )
                )
            lt = lane < Int32(T_)
            lc = _si(lt, lane, 0)
            a_t = _ld_bf16(
                Int64(gA.iterator.toint()) + (Int64(row0 + lc) * Int64(abst) + Int64(hv)) * Int64(2)
            )
            b_t = _ld_bf16(
                Int64(gB.iterator.toint()) + (Int64(row0 + lc) * Int64(abst) + Int64(hv)) * Int64(2)
            )
            # gating (every warp, lanes < T), cumulative log-decay by an 8-lane scan (the library's d = 1, 2, 4 order)
            xg = _add(a_t, dtb)
            spl = _mul(Float32(0.6931471805599453), _lg2(_add(Float32(1.0), _expf(xg))))
            spl = _sf(xg <= Float32(20.0), spl, xg)
            gl = _sf(lt, _mul(neg_a, spl), 0.0)
            beta = _sf(lt, _fdiv(Float32(1.0), _add(Float32(1.0), _expf(Float32(0.0) - b_t))), 0.0)
            for d in cutlass.range_constexpr(3):
                o = cute.arch.shuffle_sync_up(gl, 1 << d)
                gl = _sf(lane >= Int32(1 << d), _add(gl, o), gl)
            gv = _add(g_start, gl)
            tcf = tid >> Int32(4)
            jcf = tid & Int32(15)
            ec = _expf(_sub(_shfl(gv, tcf), _shfl(ghs, jcf)))
            ia = tid & Int32(63)
            ta = ia >> Int32(3)
            sa = ia & Int32(7)
            is_a = tid >= Int32(64)
            exa = _expf(_sub(_shfl(gv, ta), _shfl(gv, sa)))
            bta = _shfl(beta, ta)

            # ---- history rows beyond pn: zeros (0 x NaN from stale or uninitialized rows would poison the MMAs) ----
            while not prims.mbarrier_try_wait_parity(bars.subview(0), ph, time_limit=10_000_000):
                pass
            jr = tid >> Int32(3)
            if jr >= pn:
                zero4 = [Int32(0)] * 4
                for x in cutlass.range_constexpr(2):
                    _sts128(sU + Int32(x * HIST * ROWB) + jr * Int32(ROWB) + c8 * Int32(16), zero4)
                    _sts128(
                        sR + Int32(x * REG_B + KH_ROW * ROWB) + jr * Int32(ROWB) + c8 * Int32(16),
                        zero4,
                    )
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_sync(0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

            # ---- MMA 1: Gram ([k|q|kh|h...] x [k|q], TMEM 16-31) and state x [k|q] (TMEM 0-15) ----
            if warp == Int32(0):
                idesc16 = prims.Tcgen05InstrDesc.build(
                    c_dtype=Float32, a_dtype=BF, b_dtype=BF, n_dim=16, m_dim=128
                )
                if prims.elect_sync():
                    for x in cutlass.range_constexpr(2):
                        dq = prims.Tcgen05SmemDesc.build(
                            start_address=reg.subview(x * REG_E),
                            leading_byte_offset=16,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        dh = prims.Tcgen05SmemDesc.build(
                            start_address=reg.subview(x * REG_E + H_ROW * 64),
                            leading_byte_offset=16,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        for kq in cutlass.range_constexpr(4):
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                prims.TmemAddr.from_row_col(
                                    base.row_id, base.col_id + Int32(16)
                                ).as_ptr(Float32),
                                dq.advance_start_address(32 * kq),
                                dq.advance_start_address(32 * kq),
                                idesc16,
                                (x > 0) or (kq > 0),
                            )
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                prims.TmemAddr.from_row_col(base.row_id, base.col_id).as_ptr(
                                    Float32
                                ),
                                dh.advance_start_address(32 * kq),
                                dq.advance_start_address(32 * kq),
                                idesc16,
                                (x > 0) or (kq > 0),
                            )
                    prims.tcgen05_commit(bars.subview(1), group=prims.CTAGroup.CTA_1)

            # ---- fold operands (all threads): Uc = uh e^(gStart - G_j), split into bf16 hi + lo ----
            if is_write:
                fp_gh = _lds_f32(sF + (Int32(FS_GH) + jr) * Int32(4))
                fp_cd = _sf(jr < pn, _expf(_sub(g_start, fp_gh)), 0.0)
                for x in cutlass.range_constexpr(2):
                    fp_off = (
                        Int32(x * HIST * ROWB)
                        + jr * Int32(ROWB)
                        + ((c8 ^ (jr & Int32(7))) * Int32(16))
                    )
                    fp_hl = _uc_split(_lds128(sU + fp_off), fp_cd)
                    _sts128(sUc + fp_off, fp_hl[0:4])
                    _sts128(sUc + Int32(UH_E * 2) + fp_off, fp_hl[4:8])

            # ---- Gram rows and norms (warp 0 reads TMEM lanes 0-31) published to the q rows of both regions, free
            # once MMA 1 is done: Gram rows 0-15 -> region 0 rows 8-15, rows 16-31 -> region 1 rows 8-15 (64 B each)
            gram0 = sR + Int32(8 * ROWB)
            gram1 = sR + Int32(REG_B + 8 * ROWB)
            if warp == Int32(0):
                while not prims.mbarrier_try_wait_parity(
                    bars.subview(1), ph, time_limit=10_000_000
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                # MMA 1 was the last reader of the state rows and the key rows of a non-folding pair: load the next
                # pair's state and keys now (its U rows follow after MMA 2, under the same expect_tx)
                if ((pn + Int32(T_)) <= Int32(HIST)) & has_nx & (slot_n >= Int32(0)):
                    hvx5 = pnx - (pnx // Int32(HV)) * Int32(HV)
                    hkx5 = hvx5 // Int32(GROUP)
                    nbx5 = (pn_n + Int32(7)) // Int32(8)
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            bars.subview(0), Int32(2 * 16384) + nbx5 * Int32(4096)
                        )
                        for x in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                reg.subview(x * REG_E + H_ROW * 64),
                                tm_h.get_ptr(),
                                (Int32(64 * x), Int32(0), hvx5, slot_n),
                                bars.subview(0),
                            )
                        for jb in cutlass.range_constexpr(2):
                            if Int32(jb) < nbx5:
                                for x in cutlass.range_constexpr(2):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        reg.subview(x * REG_E + (KH_ROW + 8 * jb) * 64),
                                        tm_k.get_ptr(),
                                        (Int32(64 * x), hkx5, Int32(8 * jb), buf_n, rs_n),
                                        bars.subview(0),
                                    )
                gacc = cutlass.Array(Float32, 16, alignment=32)
                gacc.store(
                    prims.tcgen05_ld(
                        "32x32b",
                        prims.TmemAddr.from_row_col(base.row_id, base.col_id + Int32(16)).as_ptr(
                            Float32
                        ),
                        num=16,
                    ),
                    0,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                grow = _si(
                    lane < Int32(16),
                    gram0 + lane * Int32(64),
                    gram1 + (lane - Int32(16)) * Int32(64),
                )
                for q4 in cutlass.range_constexpr(4):
                    _sts128f(grow + Int32(16 * q4), [gacc[4 * q4 + e] for e in range(4)])
                dg = gacc[0]
                for i in cutlass.range_constexpr(1, 16):
                    dg = _sf(lane == Int32(i), gacc[i], dg)
                inv = _rcp_ftz(_sqrt_ftz(_add(dg, Float32(1e-6))))
                inv = _sf(
                    lane >= Int32(8), _mul(Float32(QSCALE), inv), inv
                )  # rows 8..: scale / |q_t|
                if lane < Int32(16):
                    _sts_f32(sF + (Int32(FS_INV) + lane) * Int32(4), inv)
                iql = _shfl(
                    inv, (lane + Int32(8)) & Int32(31)
                )  # iq of token = lane (held by lane 8 + t)
                if lane < Int32(8):
                    eg = _expf(gv)
                    _sts_f32(sF + (Int32(FS_BETA) + lane) * Int32(4), _sf(lt, beta, 0.0))
                    _sts_f32(sF + (Int32(FS_EGK) + lane) * Int32(4), _sf(lt, _mul(eg, inv), 0.0))
                    _sts_f32(sF + (Int32(FS_EGQ) + lane) * Int32(4), _sf(lt, _mul(eg, iql), 0.0))
                if lt:
                    gw = Int64(gOldG.iterator.toint()) + (
                        Int64(rslot) * gss
                        + (Int64(wbuf) * Int64(HV) + Int64(hv)) * Int64(HIST)
                        + Int64(woff + lane)
                    ) * Int64(4)
                    _st_f32(gw, _sf(is_write, gl, gv))
            prims.barrier_cta_sync(0)

            # ---- coefficients, every thread: one (token, history row) bf16 coefficient pair and one A or c entry ----
            # history coefficients (raw e^(G_t - G_j)) / |row t| (bf16); B operand, K-major, no swizzle: element
            # (n, j) at (n / 8) 256 + (j / 8) 128 + (n % 8) 16 + (j % 8) 2; rows n = T.. and 8 + T.. are zero
            hjc = (jcf < pn) & (tcf < Int32(T_))
            ck = _sf(
                hjc,
                _mul(
                    _mul(_lds_f32(gram1 + jcf * Int32(64) + tcf * Int32(4)), ec),
                    _lds_f32(sF + (Int32(FS_INV) + tcf) * Int32(4)),
                ),
                0.0,
            )
            cq = _sf(
                hjc,
                _mul(
                    _mul(_lds_f32(gram1 + jcf * Int32(64) + (tcf + Int32(8)) * Int32(4)), ec),
                    _lds_f32(sF + (Int32(FS_INV + 8) + tcf) * Int32(4)),
                ),
                0.0,
            )
            cbc = (
                sCf + (jcf >> Int32(3)) * Int32(128) + (jcf & Int32(7)) * Int32(2) + tcf * Int32(16)
            )
            _sts_bf16(cbc, ck)
            _sts_bf16(cbc + Int32(256), cq)
            # A[t][s] = beta_t ((kk ik_t) ik_s) e^(G_t - G_s), s < t (threads 64-127);
            # c[t][s] = ((qk e^(G_t - G_s)) ik_s) iq_t, s <= t (threads 0-63)
            iks = _lds_f32(sF + (Int32(FS_INV) + sa) * Int32(4))
            aval = _mul(
                _mul(
                    bta,
                    _mul(
                        _mul(
                            _lds_f32(gram0 + ta * Int32(64) + sa * Int32(4)),
                            _lds_f32(sF + (Int32(FS_INV) + ta) * Int32(4)),
                        ),
                        iks,
                    ),
                ),
                exa,
            )
            cval = _mul(
                _mul(_mul(_lds_f32(gram0 + (ta + Int32(8)) * Int32(64) + sa * Int32(4)), exa), iks),
                _lds_f32(sF + (Int32(FS_INV + 8) + ta) * Int32(4)),
            )
            acv = _sf(
                is_a,
                _sf((sa < ta) & (ta < Int32(T_)), aval, 0.0),
                _sf((sa <= ta) & (ta < Int32(T_)), cval, 0.0),
            )
            _sts_f32(sF + (_si(is_a, FS_A, FS_C) + ta * Int32(8) + sa) * Int32(4), acv)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.barrier_cta_sync(0)

            # ---- MMA 2: history products (TMEM 32-47); fold chunk 0 (kd 0-63, TMEM 64-127) ----
            if warp == Int32(0):
                idesc_h = prims.Tcgen05InstrDesc.build(
                    c_dtype=Float32, a_dtype=BF, b_dtype=BF, n_dim=16, m_dim=128, a_major=1
                )
                idesc_f = prims.Tcgen05InstrDesc.build(
                    c_dtype=Float32,
                    a_dtype=BF,
                    b_dtype=BF,
                    n_dim=64,
                    m_dim=128,
                    a_major=1,
                    b_major=1,
                )
                if prims.elect_sync():
                    du = prims.Tcgen05SmemDesc.build(
                        start_address=uhs,
                        leading_byte_offset=2048,
                        stride_byte_offset=1024,
                        layout=SW,
                    )
                    dc = prims.Tcgen05SmemDesc.build(
                        start_address=cfs,
                        leading_byte_offset=128,
                        stride_byte_offset=256,
                        layout=prims.Tcgen05SmemSwizzle.NONE,
                    )
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        prims.TmemAddr.from_row_col(base.row_id, base.col_id + Int32(32)).as_ptr(
                            Float32
                        ),
                        du,
                        dc,
                        idesc_h,
                        False,
                    )
                    if is_write:
                        dhi = prims.Tcgen05SmemDesc.build(
                            start_address=ucs,
                            leading_byte_offset=2048,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        dlo = prims.Tcgen05SmemDesc.build(
                            start_address=ucs.subview(UH_E),
                            leading_byte_offset=2048,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        dk0 = prims.Tcgen05SmemDesc.build(
                            start_address=reg.subview(KH_ROW * 64),
                            leading_byte_offset=REG_B,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        f0 = prims.TmemAddr.from_row_col(
                            base.row_id, base.col_id + Int32(64)
                        ).as_ptr(Float32)
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            f0,
                            dhi,
                            dk0,
                            idesc_f,
                            False,
                        )
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            f0,
                            dlo,
                            dk0,
                            idesc_f,
                            True,
                        )
                    prims.tcgen05_commit(bars.subview(2), group=prims.CTAGroup.CTA_1)

            # ---- output path: thread v = tid ----
            bt = _lds128f(sF + Int32(FS_BETA * 4)) + _lds128f(sF + Int32((FS_BETA + 4) * 4))
            egk = _lds128f(sF + Int32(FS_EGK * 4)) + _lds128f(sF + Int32((FS_EGK + 4) * 4))
            egq = _lds128f(sF + Int32(FS_EGQ * 4)) + _lds128f(sF + Int32((FS_EGQ + 4) * 4))
            while not prims.mbarrier_try_wait_parity(bars.subview(1), ph, time_limit=10_000_000):
                pass
            while not prims.mbarrier_try_wait_parity(bars.subview(2), ph, time_limit=10_000_000):
                pass
            if (pn + Int32(T_)) <= Int32(
                HIST
            ):  # not folding: history U free; the next pair's U rows now
                if has_nx & (slot_n >= Int32(0)):
                    hvx2 = pnx - (pnx // Int32(HV)) * Int32(HV)
                    nbx2 = (pn_n + Int32(7)) // Int32(8)
                    if warp == Int32(0):
                        if prims.elect_sync():
                            for jb in cutlass.range_constexpr(2):
                                if Int32(jb) < nbx2:
                                    for x in cutlass.range_constexpr(2):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            uhs.subview(x * HIST * 64 + jb * 8 * 64),
                                            tm_u.get_ptr(),
                                            (Int32(64 * x), hvx2, Int32(8 * jb), buf_n, rs_n),
                                            bars.subview(0),
                                        )
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            trow = base.row_id + warp * Int32(32)
            sacc = cutlass.Array(Float32, 16, alignment=32)
            hacc = cutlass.Array(Float32, 16, alignment=32)
            sacc.store(
                prims.tcgen05_ld(
                    "32x32b", prims.TmemAddr.from_row_col(trow, base.col_id).as_ptr(Float32), num=16
                ),
                0,
            )
            hacc.store(
                prims.tcgen05_ld(
                    "32x32b",
                    prims.TmemAddr.from_row_col(trow, base.col_id + Int32(32)).as_ptr(Float32),
                    num=16,
                ),
                0,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            if is_write:
                # TMEM 0-63 free: fold chunk 1 (kd 64-127)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.barrier_cta_sync(1, thread_count=NTHR)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if warp == Int32(0):
                    idesc_f1 = prims.Tcgen05InstrDesc.build(
                        c_dtype=Float32,
                        a_dtype=BF,
                        b_dtype=BF,
                        n_dim=64,
                        m_dim=128,
                        a_major=1,
                        b_major=1,
                    )
                    if prims.elect_sync():
                        dhi1 = prims.Tcgen05SmemDesc.build(
                            start_address=ucs,
                            leading_byte_offset=2048,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        dlo1 = prims.Tcgen05SmemDesc.build(
                            start_address=ucs.subview(UH_E),
                            leading_byte_offset=2048,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        dk1 = prims.Tcgen05SmemDesc.build(
                            start_address=reg.subview(REG_E + KH_ROW * 64),
                            leading_byte_offset=REG_B,
                            stride_byte_offset=1024,
                            layout=SW,
                        )
                        f1 = prims.TmemAddr.from_row_col(base.row_id, base.col_id).as_ptr(Float32)
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            f1,
                            dhi1,
                            dk1,
                            idesc_f1,
                            False,
                        )
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            f1,
                            dlo1,
                            dk1,
                            idesc_f1,
                            True,
                        )
                        prims.tcgen05_commit(bars.subview(3), group=prims.CTAGroup.CTA_1)

            # rhs = beta (v - e^G / |k| KS - KHI); forward solve (I + A) U = rhs; intra-chunk outputs; combine
            uu = []
            for i in cutlass.range_constexpr(T_):
                u = _mul(bt[i], _sub(_fms(egk[i], sacc[i], vv[i]), hacc[i]))
                if cutlass.const_expr(i > 0):
                    arow = _lds128f(sF + Int32((FS_A + 8 * i) * 4)) + _lds128f(
                        sF + Int32((FS_A + 8 * i + 4) * 4)
                    )
                    for s in cutlass.range_constexpr(i):
                        u = _fms(arow[s], uu[s], u)
                uu.append(u)
            xb = []
            for t in cutlass.range_constexpr(T_):
                qn = Float32(0.0)
                crow = _lds128f(sF + Int32((FS_C + 8 * t) * 4)) + _lds128f(
                    sF + Int32((FS_C + 8 * t + 4) * 4)
                )
                for s in cutlass.range_constexpr(t + 1):
                    qn = _fma(crow[s], uu[s], qn)
                ov = _add(_fma(egq[t], sacc[8 + t], hacc[8 + t]), qn)
                xb.append(_bf16r(ov))
            # history U rows
            ub = Int64(gOldU.iterator.toint()) + (
                Int64(rslot) * uss
                + ((Int64(wbuf) * Int64(HIST) + Int64(woff)) * Int64(HV) + Int64(hv)) * Int64(D)
                + Int64(tid)
            ) * Int64(2)
            for t in cutlass.range_constexpr(T_):
                _st_bf16(ub + Int64(t * HV * D * 2), uu[t])
            # gated RMSNorm: per-token sums over the 128 columns (transpose-reduce over the warp), then the 4 warp
            # partials
            p = [_mul(xb[t], xb[t]) for t in range(T_)] + [Float32(0.0)] * (8 - T_)
            up16 = (lane & Int32(16)) != Int32(0)
            q4 = [
                _add(_sf(up16, p[i + 4], p[i]), _bfly(_sf(up16, p[i], p[i + 4]), 16))
                for i in range(4)
            ]
            up8 = (lane & Int32(8)) != Int32(0)
            q2 = [
                _add(_sf(up8, q4[i + 2], q4[i]), _bfly(_sf(up8, q4[i], q4[i + 2]), 8))
                for i in range(2)
            ]
            up4 = (lane & Int32(4)) != Int32(0)
            q1 = _add(_sf(up4, q2[1], q2[0]), _bfly(_sf(up4, q2[0], q2[1]), 4))
            q1 = _add(q1, _bfly(q1, 2))
            q1 = _add(q1, _bfly(q1, 1))
            ntok = (
                ((lane >> Int32(4)) & Int32(1)) * Int32(4)
                + ((lane >> Int32(3)) & Int32(1)) * Int32(2)
                + ((lane >> Int32(2)) & Int32(1))
            )
            if (lane & Int32(3)) == Int32(0):
                _sts_f32(sF + (Int32(FS_NORM) + ntok * Int32(4) + warp) * Int32(4), q1)
            prims.barrier_cta_sync(0)
            for t in cutlass.range_constexpr(T_):
                nv = _lds128f(sF + Int32((FS_NORM + 4 * t) * 4))
                ss = _add(_add(_add(nv[0], nv[1]), nv[2]), nv[3])
                rstd = _rsqrt_ftz(_fma(ss, Float32(1.0 / D), eps))
                yv = _mul(_mul(xb[t], rstd), nw)
                sg = _fdiv(Float32(1.0), _add(Float32(1.0), _expf(Float32(0.0) - zz[t])))
                yv = _mul(yv, _mul(zz[t], sg))
                _st_e4m3(
                    pY + Int64(row0 + Int32(t)) * Int64(HV * D) + Int64(hv * Int32(D) + tid),
                    _mul(_bf16r(yv), rscale),
                )
            # history keys of this step (one value head per key head): bf16(k_raw / |k_t|)
            if (hv % Int32(GROUP)) == Int32(0):
                kb = Int64(gOldK.iterator.toint()) + (
                    Int64(rslot) * kss
                    + ((Int64(wbuf) * Int64(HIST) + Int64(woff)) * Int64(HK) + Int64(hk)) * Int64(D)
                    + Int64(tid)
                ) * Int64(2)
                kx = tid >> Int32(6)
                ke = tid & Int32(63)
                for t in cutlass.range_constexpr(T_):
                    kr = _lds_bf16(
                        sR
                        + kx * Int32(REG_B)
                        + Int32(t * ROWB)
                        + (((ke >> Int32(3)) ^ Int32(t & 7)) * Int32(16))
                        + (ke & Int32(7)) * Int32(2)
                    )
                    ikt = _lds_f32(sF + Int32((FS_INV + t) * 4))
                    _st_bf16(kb + Int64(t * HK * D * 2), _mul(kr, ikt))

            # ---- fold epilogue: h' = bf16(h e^gStart + acc), in place in shared memory, then TMA-stored ----
            if is_write:
                eg0 = _expf(g_start)
                for ch in cutlass.range_constexpr(2):
                    if cutlass.const_expr(ch == 1):
                        while not prims.mbarrier_try_wait_parity(
                            bars.subview(3), phf, time_limit=10_000_000
                        ):
                            pass
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    ccol = 64 if ch == 0 else 0
                    for half in cutlass.range_constexpr(2):
                        facc = cutlass.Array(Float32, 32, alignment=32)
                        facc.store(
                            prims.tcgen05_ld(
                                "32x32b",
                                prims.TmemAddr.from_row_col(
                                    trow, base.col_id + Int32(ccol + 32 * half)
                                ).as_ptr(Float32),
                                num=32,
                            ),
                            0,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        for c4 in cutlass.range_constexpr(4):
                            cc = 4 * half + c4
                            ha = (
                                sR
                                + Int32(ch * REG_B + H_ROW * ROWB)
                                + tid * Int32(ROWB)
                                + ((Int32(cc) ^ (tid & Int32(7))) * Int32(16))
                            )
                            nwv = _fold8x2(_lds128(ha), [facc[8 * c4 + e] for e in range(8)], eg0)
                            _sts128(ha, nwv)
                    # this half is folded by every thread: store it now, one bulk group per half. Bulk groups are per
                    # thread, so lane 0 of warp 0 commits and later waits (not elect_sync)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.barrier_cta_sync(1, thread_count=NTHR)
                    if warp == Int32(0):
                        if lane == Int32(0):
                            prims.cp_async_bulk_tensor_global_shared_cta(
                                tm_h.get_ptr(),
                                reg.subview(ch * REG_E + H_ROW * 64),
                                (Int32(64 * ch), Int32(0), hv, slot),
                            )
                            cute.arch.cp_async_bulk_commit_group()
                # the next pair's U and key rows are free (MMA 2 and both fold MMAs are done): load them now; each
                # state half once its store has been read out (lane 0: it owns the bulk groups)
                if warp == Int32(0):
                    if lane == Int32(0):
                        nx8 = has_nx & (slot_n >= Int32(0))
                        hvx8 = pnx - (pnx // Int32(HV)) * Int32(HV)
                        hkx8 = hvx8 // Int32(GROUP)
                        nbx8 = (pn_n + Int32(7)) // Int32(8)
                        if nx8:
                            prims.mbarrier_arrive_expect_tx(
                                bars.subview(0), Int32(2 * 16384) + nbx8 * Int32(4096)
                            )
                            for jb in cutlass.range_constexpr(2):
                                if Int32(jb) < nbx8:
                                    for x in cutlass.range_constexpr(2):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            uhs.subview(x * HIST * 64 + jb * 8 * 64),
                                            tm_u.get_ptr(),
                                            (Int32(64 * x), hvx8, Int32(8 * jb), buf_n, rs_n),
                                            bars.subview(0),
                                        )
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            reg.subview(x * REG_E + (KH_ROW + 8 * jb) * 64),
                                            tm_k.get_ptr(),
                                            (Int32(64 * x), hkx8, Int32(8 * jb), buf_n, rs_n),
                                            bars.subview(0),
                                        )
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                        if nx8:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                reg.subview(H_ROW * 64),
                                tm_h.get_ptr(),
                                (Int32(0), Int32(0), hvx8, slot_n),
                                bars.subview(0),
                            )
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        if nx8:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                reg.subview(REG_E + H_ROW * 64),
                                tm_h.get_ptr(),
                                (Int32(64), Int32(0), hvx8, slot_n),
                                bars.subview(0),
                            )

            j = j + Int32(1)
            jf = jf + _si(is_write, 1, 0)
            fpair = _si(is_write, 1, 0)
        # end of pair: every TMEM read and shared-memory use of this pair is done before the next one starts; a
        # folding pair already synced at its fold's barrier 1 (fpair is CTA-uniform)
        cute.arch.sync_warp()
        if fpair == Int32(0):
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_sync(0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

    if npc == Int32(0):
        cute.arch.griddepcontrol_launch_dependents()
    if warp == Int32(0):
        prims.tcgen05_dealloc(cutlass.inttoptr(tmem_raw, 6, Float32), TCOLS, group="cta_1")


@cute.jit
def _launch(
    x: cute.Tensor,
    z: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    a_log: cute.Tensor,
    dtb: cute.Tensor,
    nw: cute.Tensor,
    scale: cute.Tensor,
    state: cute.Tensor,
    slots: cute.Tensor,
    repl: cute.Tensor,
    oldu: cute.Tensor,
    oldk: cute.Tensor,
    oldg: cute.Tensor,
    cbi: cute.Tensor,
    pacc: cute.Tensor,
    y: cute.Tensor,
    stream,
    n_pairs: Int32,
    n_ctas: Int32,
    xs: Int32,
    zs: Int32,
    abst: Int32,
    s_slots: Int32,
    sss16: Int64,
    r_slots: Int32,
    uss: Int64,
    kss: Int64,
    gss: Int64,
    cbs: Int64,
    pas: Int64,
    eps: Float32,
    T_: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    HK: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[bool],
):
    BF = cutlass.BFloat16
    sw = cuda_exp.TensorMapSwizzle.s128b
    tm_h = cuda_exp.create_tensor_map_tiled(
        global_address=state.iterator.toint(),
        dtype=BF,
        global_dims=[D, D, HV, s_slots],
        global_strides=[D * 2 // 16, D * D * 2 // 16, sss16],
        box_dims=[64, D, 1, 1],
        swizzle=sw,
    )
    tm_u = cuda_exp.create_tensor_map_tiled(
        global_address=oldu.iterator.toint(),
        dtype=BF,
        global_dims=[D, HV, HIST, 2, r_slots],
        global_strides=[
            D * 2 // 16,
            HV * D * 2 // 16,
            HIST * HV * D * 2 // 16,
            uss * 2 // 16,
        ],
        box_dims=[64, 1, 8, 1, 1],
        swizzle=sw,
    )
    tm_k = cuda_exp.create_tensor_map_tiled(
        global_address=oldk.iterator.toint(),
        dtype=BF,
        global_dims=[D, HK, HIST, 2, r_slots],
        global_strides=[
            D * 2 // 16,
            HK * D * 2 // 16,
            HIST * HK * D * 2 // 16,
            kss * 2 // 16,
        ],
        box_dims=[64, 1, 8, 1, 1],
        swizzle=sw,
    )
    _gdn_verify_tcgen05_kernel(
        tm_h,
        tm_u,
        tm_k,
        x,
        z,
        a,
        b,
        a_log,
        dtb,
        nw,
        scale,
        slots,
        repl,
        oldu,
        oldk,
        oldg,
        cbi,
        pacc,
        y,
        n_pairs,
        n_ctas,
        xs,
        zs,
        abst,
        uss,
        kss,
        gss,
        cbs,
        pas,
        eps,
        T_,
        HV,
        HK,
    ).launch(
        grid=(n_ctas, 1, 1),
        block=(NTHR, 1, 1),
        stream=stream,
        use_pdl=PDL,
        min_blocks_per_mp=CTAS_PER_SM,
        max_number_threads=[NTHR, 1, 1],
    )


@functools.lru_cache(maxsize=None)
def _compiled(T_: int, HV: int, HK: int, use_pdl: bool):
    mk = cute.runtime.make_fake_compact_tensor
    bf, f32, i32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32
    return cute.compile(
        _launch,
        mk(bf, (1,), assumed_align=16),
        mk(bf, (1,), assumed_align=2),
        mk(bf, (1,), assumed_align=2),
        mk(bf, (1,), assumed_align=2),
        mk(f32, (1,), assumed_align=4),
        mk(f32, (1,), assumed_align=4),
        mk(bf, (1,), assumed_align=2),
        mk(f32, (1,), assumed_align=4),
        mk(bf, (1,), assumed_align=16),
        mk(i32, (1,), assumed_align=4),
        mk(i32, (1,), assumed_align=4),
        mk(bf, (1,), assumed_align=16),
        mk(bf, (1,), assumed_align=16),
        mk(f32, (1,), assumed_align=4),
        mk(i32, (1,), assumed_align=4),
        mk(i32, (1,), assumed_align=4),
        mk(cutlass.Uint8, (1,), assumed_align=1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int64(1),
        Int32(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Float32(1e-6),
        T_=T_,
        HV=HV,
        HK=HK,
        PDL=use_pdl,
        options="--enable-tvm-ffi",
    )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _one(t):
    return torch.as_strided(t, (1,), (1,))


def _packed_slot_rows(t: torch.Tensor) -> bool:
    """Whether the dimensions after t's leading (slot) dimension are packed."""
    expected = 1
    for d in range(t.dim() - 1, 0, -1):
        if t.size(d) != 1 and t.stride(d) != expected:
            return False
        expected *= t.size(d)
    return True


def gdn_verify_tcgen05_supported(T: int, HV: int, HK: int, fp8_out: bool) -> bool:
    """Whether the kernel compiles for T draft rows per request and HV / HK value / key heads per rank.

    Head dim 128 and history 16 are fixed (checked per call); the output must be FP8. The caller checks for an
    SM100-family GPU.
    """
    return bool(fp8_out) and MIN_T <= T <= MAX_T and HK > 0 and HV > 0 and HV % HK == 0


def compile_gdn_verify_tcgen05(
    T: int, HV: int, HK: int, device: torch.device | None = None
) -> None:
    """Compile (and cache) the kernel for (T, HV, HK) and the current ``TRTLLM_ENABLE_PDL`` ahead of any call, e.g.
    after loading (the GDN mixer's cache_derived_state, for an FP8 out_proj only), so that no compilation runs inside a
    warm-up or a CUDA-graph capture: a first call compiles lazily, also under capture."""
    assert gdn_verify_tcgen05_supported(T, HV, HK, True), (T, HV, HK)
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        _compiled(T, HV, HK, get_env_enable_pdl())
    _num_sms(index)


def gdn_verify_tcgen05(
    x: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    fp8_scale: torch.Tensor | None,
    eps: float,
    state: torch.Tensor,
    slots: torch.Tensor,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_g: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_acc: torch.Tensor,
    replay_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """``trtllm::gdn_verify`` with FP8 output (same arguments); returns y [n T, HV 128] float8_e4m3fn.

    state (the checkpoints), old_u, old_k and old_g are updated in place. The history (old_u, old_k, old_g) and its
    bookkeeping (cache_buf_idx, prev_acc) are rows of replay_indices, which default to the state slots; their slot
    strides may be those of a pool page (the other dimensions packed).
    """
    n = slots.numel()
    HV = old_u.shape[3]
    HK = old_k.shape[3]
    T_ = x.shape[0] // n
    assert n > 0 and T_ * n == x.shape[0]
    assert gdn_verify_tcgen05_supported(T_, HV, HK, fp8_scale is not None), (T_, HV, HK)
    # gdnVerifyOp.cpp's dtype checks (the kernel reads these operands as BF16) and its FP32 scalar scale.
    for name, t in (("qkv", x), ("z", z), ("a", a), ("b", b), ("old_u", old_u), ("old_k", old_k)):
        assert t.dtype == torch.bfloat16, f"{name} must be bfloat16, got {t.dtype}"
    assert state.dtype == torch.bfloat16, f"ssm_states must be bfloat16, got {state.dtype}"
    assert fp8_scale.dtype == torch.float32 and fp8_scale.numel() == 1, (
        "fp8_scale must be a float32 scalar tensor"
    )
    assert x.shape[1] >= (2 * HK + HV) * D and x.stride(1) == 1 and x.stride(0) % 8 == 0
    # The kernel is compiled for 16-byte aligned qkv / state bases (TMA and vector loads).
    assert x.data_ptr() % 16 == 0, "qkv must have a 16-byte aligned base"
    assert (
        old_u.dim() == 5 and old_u.shape[1] == 2 and old_u.shape[2] == HIST and old_u.shape[4] == D
    )
    assert (
        old_k.dim() == 5 and old_k.shape[1] == 2 and old_k.shape[2] == HIST and old_k.shape[4] == D
    )
    assert z.dim() == 3 and tuple(z.shape) == (x.shape[0], HV, D)
    assert z.stride(-1) == 1 and z.stride(-2) == D and a.stride(1) == 1 and b.stride(1) == 1
    assert (
        a.dim() == 2 and tuple(a.shape) == (x.shape[0], HV) and tuple(b.shape) == (x.shape[0], HV)
    )
    assert a.stride(0) == b.stride(0)
    assert state.shape[1] == HV and state.shape[-1] == D and state.shape[-2] == D
    assert (
        state.stride(3) == 1
        and state.stride(2) == D
        and state.stride(1) == D * D
        and state.stride(0) % 8 == 0
        and state.data_ptr() % 16 == 0
    ), "ssm_states must be [slots, HV, 128, 128] with packed, 16-byte aligned slots"
    if replay_indices is None:
        replay_indices = slots
    assert replay_indices.numel() == n
    # gdnVerifyOp.cpp's operand checks: int32 indices and bookkeeping, the history's replay slots, the gate / norm
    # parameters.
    assert slots.dtype == torch.int32 and slots.is_contiguous()
    assert replay_indices.dtype == torch.int32 and replay_indices.is_contiguous()
    assert cache_buf_idx.dtype == torch.int32 and cache_buf_idx.dim() == 1
    assert prev_acc.dtype == torch.int32 and prev_acc.dim() == 1
    assert old_g.dtype == torch.float32 and tuple(old_g.shape) == (old_u.shape[0], 2, HV, HIST)
    assert old_k.shape[0] == old_u.shape[0] == cache_buf_idx.shape[0] == prev_acc.shape[0]
    assert A_log.dtype == torch.float32 and A_log.is_contiguous() and A_log.numel() == HV
    assert dt_bias.dtype == torch.float32 and dt_bias.is_contiguous() and dt_bias.numel() == HV
    assert norm_w.dtype == torch.bfloat16 and norm_w.is_contiguous() and norm_w.numel() == D
    u_stride, k_stride, g_stride = old_u.stride(), old_k.stride(), old_g.stride()
    # packed slot rows (the per-slot strides a pool page gives are free); the TMA needs 16-byte slot strides and bases
    assert u_stride[1:] == (HIST * HV * D, HV * D, D, 1) or _packed_slot_rows(old_u), u_stride
    assert k_stride[1:] == (HIST * HK * D, HK * D, D, 1) or _packed_slot_rows(old_k), k_stride
    assert g_stride[1:] == (HV * HIST, HIST, 1) or _packed_slot_rows(old_g), g_stride
    assert u_stride[0] % 8 == 0 and k_stride[0] % 8 == 0
    assert old_u.data_ptr() % 16 == 0 and old_k.data_ptr() % 16 == 0
    assert (
        old_u.shape[2] == HIST
        and old_u.shape[4] == D
        and old_k.shape[2] == HIST
        and old_k.shape[4] == D
    )
    y = torch.empty((n * T_, HV * D), dtype=torch.float8_e4m3fn, device=x.device)
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    n_pairs = n * HV
    n_ctas = min(n_pairs, CTAS_PER_SM * _num_sms(dev))
    fn = _compiled(T_, HV, HK, get_env_enable_pdl())
    fn(
        _one(x),
        _one(z),
        _one(a),
        _one(b),
        _one(A_log),
        _one(dt_bias),
        _one(norm_w),
        _one(fp8_scale),
        _one(state),
        _one(slots),
        _one(replay_indices),
        _one(old_u),
        _one(old_k),
        _one(old_g),
        _one(cache_buf_idx),
        _one(prev_acc),
        _one(y.view(torch.uint8)),
        n_pairs,
        n_ctas,
        x.stride(0),
        z.stride(0),
        a.stride(0),
        state.shape[0],
        state.stride(0) * 2 // 16,
        old_u.shape[0],
        u_stride[0],
        k_stride[0],
        g_stride[0],
        cache_buf_idx.stride(0),
        prev_acc.stride(0),
        float(eps),
    )
    return y
