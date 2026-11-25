# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-layer accepted-prefix Conv4 commit + native 2FP4 checkpoint commit (CuTe DSL).

One launch. Work item = (layer, batch row, head):
  * Conv4 history slice for the head's 384 channels (always, for non-dummy rows).
  * history_len update (head 0 only).
  * If the row rolls over (P+A>=16): reconstruct the 128x128
    primary+residual 2FP4 state, decay it by exp(G_end), add
    U^T @ bf16(K * exp(G_end - G_i)) via bf16 mma.sync with fp32 accumulation,
    and RNE-repack primary then residual (E2M1 codes, E4M3 block-16 scales,
    per-head fp32 outer scale).
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .cute_dsl_kda_2fp4_commit import STATE_BYTES

THREADS = 256
# Descriptor row (int64 fields): checkpoint pointer and slot stride in bytes, Conv pointer
# and slot stride in BF16 elements, K/U/G/length pointers, old-length snapshot and
# candidate Conv pointers, then the K/U/G/length replay-slot strides in bytes.
DESC_FIELDS = 14
DESC_BYTES = DESC_FIELDS * 8

# ---------------------------------------------------------------------------
# inline PTX helpers
# ---------------------------------------------------------------------------


def _asm(res_types, args, text, cons, side, loc, ip):
    if len(res_types) == 0:
        llvm.inline_asm(
            None,
            args,
            text,
            cons,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return None
    if len(res_types) == 1:
        return llvm.inline_asm(
            res_types[0],
            args,
            text,
            cons,
            has_side_effects=side,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    st = llvm.StructType.get_literal(res_types)
    r = llvm.inline_asm(
        st,
        args,
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [llvm.extractvalue(res_types[i], r, [i], loc=loc, ip=ip) for i in range(len(res_types))]


def _i32(x, loc, ip):
    return Int32(x).ir_value(loc=loc, ip=ip)


def _i64(x, loc, ip):
    return Int64(x).ir_value(loc=loc, ip=ip)


def _f32(x, loc, ip):
    return Float32(x).ir_value(loc=loc, ip=ip)


@dsl_user_op
def ld_g_s32(addr, *, loc=None, ip=None):
    return Int32(
        _asm([T.i32()], [_i64(addr, loc, ip)], "ld.global.s32 $0, [$1];", "=r,l", True, loc, ip)
    )


@dsl_user_op
def ld_g_u8(addr, *, loc=None, ip=None):
    return Int32(
        _asm([T.i32()], [_i64(addr, loc, ip)], "ld.global.u8 $0, [$1];", "=r,l", True, loc, ip)
    )


@dsl_user_op
def ld_g_s64(addr, *, loc=None, ip=None):
    return Int64(
        _asm([T.i64()], [_i64(addr, loc, ip)], "ld.global.s64 $0, [$1];", "=l,l", True, loc, ip)
    )


@dsl_user_op
def st_g_s32(addr, v, *, loc=None, ip=None):
    _asm(
        [], [_i64(addr, loc, ip), _i32(v, loc, ip)], "st.global.s32 [$0], $1;", "l,r", True, loc, ip
    )


@dsl_user_op
def st_g_u16(addr, v, *, loc=None, ip=None):
    _asm(
        [], [_i64(addr, loc, ip), _i32(v, loc, ip)], "st.global.u16 [$0], $1;", "l,r", True, loc, ip
    )


@dsl_user_op
def st_g_f32(addr, v, *, loc=None, ip=None):
    _asm(
        [], [_i64(addr, loc, ip), _f32(v, loc, ip)], "st.global.f32 [$0], $1;", "l,f", True, loc, ip
    )


@dsl_user_op
def ld_g_v4_pred(addr, pred, *, loc=None, ip=None):
    r = _asm(
        [T.i32()] * 4,
        [_i64(addr, loc, ip), _i32(pred, loc, ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $5, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "@p ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r",
        True,
        loc,
        ip,
    )
    return [Int32(v) for v in r]


@dsl_user_op
def st_g_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        [],
        [
            _i64(addr, loc, ip),
            _i32(a, loc, ip),
            _i32(b, loc, ip),
            _i32(c, loc, ip),
            _i32(d, loc, ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_s_u32(saddr, v, *, loc=None, ip=None):
    _asm(
        [],
        [_i32(saddr, loc, ip), _i32(v, loc, ip)],
        "st.shared.u32 [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_s_f32(saddr, v, *, loc=None, ip=None):
    _asm(
        [],
        [_i32(saddr, loc, ip), _f32(v, loc, ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_s_f32(saddr, *, loc=None, ip=None):
    return Float32(
        _asm([T.f32()], [_i32(saddr, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", True, loc, ip)
    )


@dsl_user_op
def st_s_v4(saddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        [],
        [
            _i32(saddr, loc, ip),
            _i32(a, loc, ip),
            _i32(b, loc, ip),
            _i32(c, loc, ip),
            _i32(d, loc, ip),
        ],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_s_v4f(saddr, *, loc=None, ip=None):
    r = _asm(
        [T.f32()] * 4,
        [_i32(saddr, loc, ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        True,
        loc,
        ip,
    )
    return [Float32(v) for v in r]


@dsl_user_op
def ldsm_x4_t(saddr, *, loc=None, ip=None):
    r = _asm(
        [T.i32()] * 4,
        [_i32(saddr, loc, ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc,
        ip,
    )
    return [Int32(v) for v in r]


@dsl_user_op
def ldsm_x2_t(saddr, *, loc=None, ip=None):
    r = _asm(
        [T.i32()] * 2,
        [_i32(saddr, loc, ip)],
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        True,
        loc,
        ip,
    )
    return [Int32(v) for v in r]


@dsl_user_op
def mma_bf16_k8(a0, a1, b0, c, *, loc=None, ip=None):
    args = [_i32(a0, loc, ip), _i32(a1, loc, ip), _i32(b0, loc, ip)] + [_f32(x, loc, ip) for x in c]
    r = _asm(
        [T.f32()] * 4,
        args,
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};",
        "=f,=f,=f,=f,r,r,r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return [Float32(v) for v in r]


@dsl_user_op
def mma_bf16(a, b0, b1, c, *, loc=None, ip=None):
    args = (
        [_i32(x, loc, ip) for x in a]
        + [_i32(b0, loc, ip), _i32(b1, loc, ip)]
        + [_f32(x, loc, ip) for x in c]
    )
    r = _asm(
        [T.f32()] * 4,
        args,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return [Float32(v) for v in r]


@dsl_user_op
def fmax_f32(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            [T.f32()],
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def exp_f32(x, *, loc=None, ip=None):
    return Float32(
        _asm(
            [T.f32()],
            [_f32(x, loc, ip)],
            "{\n.reg .f32 t;\nmul.rn.f32 t, $1, 0f3FB8AA3B;\nex2.approx.ftz.f32 $0, t;\n}",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def outer_and_denom(amax, *, loc=None, ip=None):
    # outer = amax / 2688 ; denom = 6 * (outer == 0 ? 1 : outer)
    r = _asm(
        [T.f32(), T.f32(), T.f32()],
        [_f32(amax, loc, ip)],
        "{\n.reg .f32 o;\n.reg .pred p;\ndiv.rn.f32 $0, $3, 0f45280000;\n"
        "setp.eq.f32 p, $0, 0f00000000;\nselp.f32 o, 0f3F800000, $0, p;\nmul.rn.f32 $1, o, 0f40C00000;\n"
        "div.rn.f32 $2, 0f3F800000, $1;\n}",
        "=f,=f,=f,f",
        False,
        loc,
        ip,
    )
    return Float32(r[0]), Float32(r[1]), Float32(r[2])


@dsl_user_op
def adj_pair(kw, g0, g1, e0, e1, *, loc=None, ip=None):
    # bf16x2( K * exp(Gend - G) ), K packed bf16x2 (low = even column)
    return Int32(
        _asm(
            [T.i32()],
            [
                _i32(kw, loc, ip),
                _f32(g0, loc, ip),
                _f32(g1, loc, ip),
                _f32(e0, loc, ip),
                _f32(e1, loc, ip),
            ],
            "{\n.reg .f32 k0, k1, d0, d1;\n.reg .b32 t0, t1;\n.reg .b64 E, G, L, D, K;\n"
            "shl.b32 t0, $1, 16;\nand.b32 t1, $1, 0xffff0000;\nmov.b32 k0, t0;\nmov.b32 k1, t1;\n"
            "mov.b64 E, {$4, $5};\nmov.b64 G, {$2, $3};\nmov.b64 L, 0x3FB8AA3B3FB8AA3B;\n"
            "sub.rn.f32x2 D, E, G;\nmul.rn.f32x2 D, D, L;\nmov.b64 {d0, d1}, D;\n"
            "ex2.approx.ftz.f32 d0, d0;\nex2.approx.ftz.f32 d1, d1;\n"
            "mov.b64 D, {d0, d1};\nmov.b64 K, {k0, k1};\nmul.rn.f32x2 K, K, D;\nmov.b64 {k0, k1}, K;\n"
            "cvt.rn.bf16x2.f32 $0, k1, k0;\n}",
            "=r,r,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_s_v4(saddr, *, loc=None, ip=None):
    r = _asm(
        [T.i32()] * 4,
        [_i32(saddr, loc, ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc,
        ip,
    )
    return [Int32(v) for v in r]


@dsl_user_op
def ld_s_u32(saddr, *, loc=None, ip=None):
    return Int32(
        _asm([T.i32()], [_i32(saddr, loc, ip)], "ld.shared.u32 $0, [$1];", "=r,r", True, loc, ip)
    )


@dsl_user_op
def ld_s_v2f(saddr, *, loc=None, ip=None):
    r = _asm(
        [T.f32()] * 2,
        [_i32(saddr, loc, ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        True,
        loc,
        ip,
    )
    return [Float32(v) for v in r]


@dsl_user_op
def ld_s_u16(saddr, *, loc=None, ip=None):
    return Int32(
        _asm([T.i32()], [_i32(saddr, loc, ip)], "ld.shared.u16 $0, [$1];", "=r,r", True, loc, ip)
    )


@dsl_user_op
def ld_s_s64(saddr, *, loc=None, ip=None):
    return Int64(
        _asm([T.i64()], [_i32(saddr, loc, ip)], "ld.shared.s64 $0, [$1];", "=l,r", True, loc, ip)
    )


@dsl_user_op
def st_s_s64(saddr, v, *, loc=None, ip=None):
    _asm(
        [],
        [_i32(saddr, loc, ip), _i64(v, loc, ip)],
        "st.shared.s64 [$0], $1;",
        "r,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def mbar_init(saddr, cnt, *, loc=None, ip=None):
    _asm(
        [],
        [_i32(saddr, loc, ip), _i32(cnt, loc, ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def fence_mbar_init(*, loc=None, ip=None):
    _asm([], [], "fence.mbarrier_init.release.cluster;", "", True, loc, ip)


@dsl_user_op
def fence_proxy_async(*, loc=None, ip=None):
    _asm([], [], "fence.proxy.async.shared::cta;", "", True, loc, ip)


@dsl_user_op
def mbar_expect_tx(saddr, nbytes, *, loc=None, ip=None):
    _asm(
        [],
        [_i32(saddr, loc, ip), _i32(nbytes, loc, ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def mbar_expect_tx_only(saddr, nbytes, *, loc=None, ip=None):
    _asm(
        [],
        [_i32(saddr, loc, ip), _i32(nbytes, loc, ip)],
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def mbar_arrive(saddr, *, loc=None, ip=None):
    _asm(
        [],
        [_i32(saddr, loc, ip)],
        "mbarrier.arrive.shared::cta.b64 _, [$0];",
        "r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def mbar_try_wait(saddr, parity, *, loc=None, ip=None):
    return Int32(
        _asm(
            [T.i32()],
            [_i32(saddr, loc, ip), _i32(parity, loc, ip)],
            "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\nselp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def bulk_g2s(sdst, gsrc, nbytes, mbar, *, loc=None, ip=None):
    _asm(
        [],
        [_i32(sdst, loc, ip), _i64(gsrc, loc, ip), _i32(nbytes, loc, ip), _i32(mbar, loc, ip)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def sel_i32(p, a, b, *, loc=None, ip=None):
    # p != 0 ? a : b
    return Int32(
        _asm(
            [T.i32()],
            [_i32(p, loc, ip), _i32(a, loc, ip), _i32(b, loc, ip)],
            "{\n.reg .pred q;\nsetp.ne.s32 q, $1, 0;\nselp.b32 $0, $2, $3, q;\n}",
            "=r,r,r,r",
            False,
            loc,
            ip,
        )
    )


# ---------------------------------------------------------------------------
# packed-f32x2 hot-path helpers (sm_100+)
# ---------------------------------------------------------------------------


@dsl_user_op
def absmax8x(m, xs, *, loc=None, ip=None):
    text = (
        "{\n.reg .f32 a;\nmax.abs.f32 a, $2, $3, $4;\nmax.abs.f32 a, a, $5, $6;\n"
        "max.abs.f32 a, a, $7, $8;\nmax.abs.f32 $0, a, $9, $1;\n}"
    )
    args = [_f32(m, loc, ip)] + [_f32(x, loc, ip) for x in xs]
    return Float32(_asm([T.f32()], args, text, "=f,f" + ",f" * 8, False, loc, ip))


def _scale2(L, p, aa, ab, outer, rden):
    """Two blocks' E4M3 scales and reciprocal operand scales.  aa/ab: block maxima registers.
    Produces {p}hs (b16: lo = block a, hi = block b scale bytes), {p}RCA/{p}RCB (f32x2 1/e),
    {p}NEA/{p}NEB (f32x2 -e), predicates {p}pza/{p}pzb (e == 0)."""
    L.append(f"mov.b64 {p}A2, {{{aa}, {ab}}};")
    # correctly rounded a / denom (reciprocal + Markstein correction), both blocks at once
    L.append(f"mul.rn.f32x2 {p}S2, {p}A2, RD2;")
    L.append(f"fma.rn.f32x2 {p}T2, {p}S2, ND2, {p}A2;")
    L.append(f"fma.rn.f32x2 {p}S2, {p}T2, RD2, {p}S2;")
    L.append(f"mov.b64 {{{p}sfa, {p}sfb}}, {p}S2;")
    L.append(f"min.f32 {p}sfa, {p}sfa, 0f43E00000;")
    L.append(f"min.f32 {p}sfb, {p}sfb, 0f43E00000;")
    L.append(f"cvt.rn.satfinite.e4m3x2.f32 {p}hs, {p}sfb, {p}sfa;")
    L.append(f"cvt.rn.f16x2.e4m3x2 {p}hw, {p}hs;")
    L.append(f"mov.b32 {{{p}hl, {p}hh}}, {p}hw;")
    L.append(f"cvt.f32.f16 {p}sca, {p}hl;")
    L.append(f"cvt.f32.f16 {p}scb, {p}hh;")
    L.append(f"mov.b64 {p}SC2, {{{p}sca, {p}scb}};")
    L.append(f"mul.rn.f32x2 {p}E2, {p}SC2, OU2;")
    L.append(f"mov.b64 {{{p}ea, {p}eb}}, {p}E2;")
    # rcp.approx.f32 (subnormal-safe) as the compiler expands it, without the huge-input branch
    # (|e| <= amax / 6 < 2^126): s = |e| < 2^-126 ? 2^24 : 1; rc = s * rcp.ftz(e * s)
    L.append(f"abs.f32 {p}aa, {p}ea;")
    L.append(f"abs.f32 {p}ab, {p}eb;")
    L.append(f"setp.lt.f32 {p}pda, {p}aa, 0f00800000;")
    L.append(f"setp.lt.f32 {p}pdb, {p}ab, 0f00800000;")
    L.append(f"selp.f32 {p}ssa, 0f4B800000, 0f3F800000, {p}pda;")
    L.append(f"selp.f32 {p}ssb, 0f4B800000, 0f3F800000, {p}pdb;")
    L.append(f"mov.b64 {p}SS2, {{{p}ssa, {p}ssb}};")
    L.append(f"mul.rn.f32x2 {p}ES2, {p}E2, {p}SS2;")
    L.append(f"mov.b64 {{{p}esa, {p}esb}}, {p}ES2;")
    L.append(f"rcp.approx.ftz.f32 {p}ra, {p}esa;")
    L.append(f"rcp.approx.ftz.f32 {p}rb, {p}esb;")
    L.append(f"mov.b64 {p}R2, {{{p}ra, {p}rb}};")
    L.append(f"mul.rn.f32x2 {p}R2, {p}R2, {p}SS2;")
    L.append(f"mov.b64 {{{p}rca, {p}rcb}}, {p}R2;")
    L.append(f"setp.eq.f32 {p}pza, {p}ea, 0f00000000;")
    L.append(f"setp.eq.f32 {p}pzb, {p}eb, 0f00000000;")
    L.append(f"neg.f32 {p}nea, {p}ea;")
    L.append(f"neg.f32 {p}neb, {p}eb;")
    L.append(f"mov.b64 {p}RCA, {{{p}rca, {p}rca}};")
    L.append(f"mov.b64 {p}RCB, {{{p}rcb, {p}rcb}};")
    L.append(f"mov.b64 {p}NEA, {{{p}nea, {p}nea}};")
    L.append(f"mov.b64 {p}NEB, {{{p}neb, {p}neb}};")


def _codes2(L, p, XA, XB):
    """E2M1 codes of two blocks: XA/XB lists of 8 b64 pair registers; -> {p}wa0,{p}wa1,{p}wb0,{p}wb1."""
    for blk, X, RC in (("a", XA, f"{p}RCA"), ("b", XB, f"{p}RCB")):
        for i in range(8):
            L.append(f"mul.rn.f32x2 {p}Q, {X[i]}, {RC};")
            L.append(f"mov.b64 {{{p}q0, {p}q1}}, {p}Q;")
            L.append(f"cvt.rn.satfinite.e2m1x2.f32 {p}c{blk}{i}, {p}q1, {p}q0;")
        L.append(f"mov.b32 {p}w{blk}0, {{{p}c{blk}0, {p}c{blk}1, {p}c{blk}2, {p}c{blk}3}};")
        L.append(f"mov.b32 {p}w{blk}1, {{{p}c{blk}4, {p}c{blk}5, {p}c{blk}6, {p}c{blk}7}};")
        L.append(f"selp.b32 {p}w{blk}0, 0, {p}w{blk}0, {p}pz{blk};")
        L.append(f"selp.b32 {p}w{blk}1, 0, {p}w{blk}1, {p}pz{blk};")


def _quant32x_text():
    # $0..$8 outputs: P codes a (2 words), P codes b, R codes a, R codes b, scales (P hs | R hs << 16)
    # $9..$40: 32 inputs (block a = $9..$24, block b = $25..$40); $41 outer, $42 denom, $43 rden,
    # $44 / $45 block maxima of a / b
    xa = [f"${9 + i}" for i in range(16)]
    xb = [f"${25 + i}" for i in range(16)]
    L = [
        "{",
        ".reg .f32 nden;",
        ".reg .b64 RD2, ND2, OU2;",
        "neg.f32 nden, $42;",
        "mov.b64 RD2, {$43, $43};",
        "mov.b64 ND2, {nden, nden};",
        "mov.b64 OU2, {$41, $41};",
    ]
    for p in ("P", "R"):
        L.append(
            f".reg .f32 {p}sfa, {p}sfb, {p}sca, {p}scb, {p}ea, {p}eb, {p}aa, {p}ab, {p}ssa, {p}ssb, "
            f"{p}esa, {p}esb, {p}ra, {p}rb, {p}rca, {p}rcb, {p}nea, {p}neb, {p}q0, {p}q1;"
        )
        L.append(
            f".reg .b64 {p}A2, {p}S2, {p}T2, {p}SC2, {p}E2, {p}SS2, {p}ES2, {p}R2, {p}RCA, {p}RCB, "
            f"{p}NEA, {p}NEB, {p}Q;"
        )
        L.append(f".reg .b16 {p}hs, {p}hl, {p}hh;")
        L.append(f".reg .b32 {p}hw, {p}wa0, {p}wa1, {p}wb0, {p}wb1;")
        L.append(f".reg .b8 {p}ca<8>, {p}cb<8>;")
        L.append(f".reg .pred {p}pda, {p}pdb, {p}pza, {p}pzb;")
    L.append(".reg .b64 XA<8>, XB<8>, YA<8>, YB<8>, LV;")
    L.append(".reg .b32 dh;")
    L.append(".reg .b16 dl0, dl1;")
    L.append(".reg .f32 lv0, lv1, ya<16>, yb<16>, ma<7>, mb<7>, Ram, Rbm;")
    L.append(".reg .b32 s0, s1;")
    for i in range(8):
        L.append(f"mov.b64 XA{i}, {{{xa[2 * i]}, {xa[2 * i + 1]}}};")
        L.append(f"mov.b64 XB{i}, {{{xb[2 * i]}, {xb[2 * i + 1]}}};")
    # ---- primary pass (block maxima given)
    _scale2(L, "P", "$44", "$45", "$41", "$43")
    _codes2(L, "P", [f"XA{i}" for i in range(8)], [f"XB{i}" for i in range(8)])
    # ---- residual x - dequant(P): exact level from the code, one fma per element pair
    for blk, X, Y, NE, yv in (("a", "XA", "YA", "PNEA", "ya"), ("b", "XB", "YB", "PNEB", "yb")):
        for i in range(8):
            L.append(f"cvt.rn.f16x2.e2m1x2 dh, Pc{blk}{i};")
            L.append("mov.b32 {dl0, dl1}, dh;")
            L.append("cvt.f32.f16 lv0, dl0;")
            L.append("cvt.f32.f16 lv1, dl1;")
            L.append("mov.b64 LV, {lv0, lv1};")
            L.append(f"fma.rn.f32x2 {Y}{i}, LV, {NE}, {X}{i};")
            L.append(f"mov.b64 {{{yv}{2 * i}, {yv}{2 * i + 1}}}, {Y}{i};")
    # residual block maxima (balanced trees, order independent)
    for yv, m, out in (("ya", "ma", "Ram"), ("yb", "mb", "Rbm")):
        L.append(f"max.abs.f32 {m}0, {yv}0, {yv}1, {yv}2;")
        L.append(f"max.abs.f32 {m}1, {yv}3, {yv}4, {yv}5;")
        L.append(f"max.abs.f32 {m}2, {yv}6, {yv}7, {yv}8;")
        L.append(f"max.abs.f32 {m}3, {yv}9, {yv}10, {yv}11;")
        L.append(f"max.abs.f32 {m}4, {yv}12, {yv}13, {yv}14;")
        L.append(f"max.abs.f32 {m}5, {m}0, {m}1, {yv}15;")
        L.append(f"max.f32 {m}6, {m}2, {m}3, {m}4;")
        L.append(f"max.f32 {out}, {m}5, {m}6;")
    _scale2(L, "R", "Ram", "Rbm", "$41", "$43")
    _codes2(L, "R", [f"YA{i}" for i in range(8)], [f"YB{i}" for i in range(8)])
    L += [
        "mov.b32 $0, Pwa0;",
        "mov.b32 $1, Pwa1;",
        "mov.b32 $2, Pwb0;",
        "mov.b32 $3, Pwb1;",
        "mov.b32 $4, Rwa0;",
        "mov.b32 $5, Rwa1;",
        "mov.b32 $6, Rwb0;",
        "mov.b32 $7, Rwb1;",
        "cvt.u32.u16 s0, Phs;",
        "cvt.u32.u16 s1, Rhs;",
        "shl.b32 s1, s1, 16;",
        "or.b32 $8, s0, s1;",
        "}",
    ]
    return "\n".join(L)


_QUANT32X = _quant32x_text()


@dsl_user_op
def quant32x(xs, outer, denom, rden, bmax_a, bmax_b, *, loc=None, ip=None):
    args = [_f32(x, loc, ip) for x in xs] + [
        _f32(outer, loc, ip),
        _f32(denom, loc, ip),
        _f32(rden, loc, ip),
        _f32(bmax_a, loc, ip),
        _f32(bmax_b, loc, ip),
    ]
    r = _asm([T.i32()] * 9, args, _QUANT32X, ",".join(["=r"] * 9) + ",f" * 37, False, loc, ip)
    return [Int32(v) for v in r]


@dsl_user_op
def shfl_up_i32(v, d, *, loc=None, ip=None):
    return Int32(
        _asm(
            [T.i32()],
            [_i32(v, loc, ip), _i32(d, loc, ip)],
            "shfl.sync.up.b32 $0, $1, $2, 0x0, 0xffffffff;",
            "=r,r,r",
            True,
            loc,
            ip,
        )
    )


def _deccomb8h_text():
    L = [
        "{",
        ".reg .b8 p<4>, r<4>;",
        ".reg .b32 ph<4>, rh<4>;",
        ".reg .b16 pl<8>, rl<8>, sp, sr;",
        ".reg .f32 zf, t<8>, v<8>;",
        ".reg .b64 V<4>, D<4>, A<4>, O<4>;",
        "mov.f32 zf, 0f00000000;",
        "mov.b32 {p0, p1, p2, p3}, $8;",
        "mov.b32 {r0, r1, r2, r3}, $9;",
        "cvt.u16.u32 sp, $10;",
        "cvt.u16.u32 sr, $11;",
    ]
    for i in range(4):
        L.append(f"cvt.rn.f16x2.e2m1x2 ph{i}, p{i};")
        L.append(f"cvt.rn.f16x2.e2m1x2 rh{i}, r{i};")
        L.append(f"mov.b32 {{pl{2 * i}, pl{2 * i + 1}}}, ph{i};")
        L.append(f"mov.b32 {{rl{2 * i}, rl{2 * i + 1}}}, rh{i};")
    for i in range(8):
        # exact level*scale products (e2m1 x e4m3), summed once in fp32
        L.append(f"fma.rn.f32.f16 t{i}, pl{i}, sp, zf;")
        L.append(f"fma.rn.f32.f16 v{i}, rl{i}, sr, t{i};")
    for i in range(4):
        L.append(f"mov.b64 V{i}, {{v{2 * i}, v{2 * i + 1}}};")
        L.append(f"mov.b64 D{i}, {{${12 + 2 * i}, ${13 + 2 * i}}};")
        L.append(f"mov.b64 A{i}, {{${20 + 2 * i}, ${21 + 2 * i}}};")
        L.append(f"fma.rn.f32x2 O{i}, V{i}, D{i}, A{i};")
        L.append(f"mov.b64 {{${2 * i}, ${2 * i + 1}}}, O{i};")
    L.append("}")
    return "\n".join(L)


_DECCOMB8H = _deccomb8h_text()


@dsl_user_op
def deccomb8h(wp, wr, sph, srh, dec, acc, *, loc=None, ip=None):
    """8 elements: (lvl_p*sp + lvl_r*sr) * (outer*decay) + acc; sp/sr are f16 in the low half."""
    args = [_i32(wp, loc, ip), _i32(wr, loc, ip), _i32(sph, loc, ip), _i32(srh, loc, ip)]
    args += [_f32(x, loc, ip) for x in dec] + [_f32(x, loc, ip) for x in acc]
    r = _asm(
        [T.f32()] * 8,
        args,
        _DECCOMB8H,
        ",".join(["=f"] * 8) + ",r,r,r,r" + ",f" * 16,
        False,
        loc,
        ip,
    )
    return [Float32(v) for v in r]


@dsl_user_op
def e4m3x2_to_f16x2(w, *, loc=None, ip=None):
    return Int32(
        _asm(
            [T.i32()],
            [_i32(w, loc, ip)],
            "{\n.reg .b16 hs;\ncvt.u16.u32 hs, $1;\ncvt.rn.f16x2.e4m3x2 $0, hs;\n}",
            "=r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def st_g_v4_pred(addr, a, b, c, d, pred, *, loc=None, ip=None):
    _asm(
        [],
        [
            _i64(addr, loc, ip),
            _i32(a, loc, ip),
            _i32(b, loc, ip),
            _i32(c, loc, ip),
            _i32(d, loc, ip),
            _i32(pred, loc, ip),
        ],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $5, 0;\n@p st.global.v4.u32 [$0], {$1, $2, $3, $4};\n}",
        "l,r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def prmt_b32(a, b, sel, *, loc=None, ip=None):
    return Int32(
        _asm(
            [T.i32()],
            [_i32(a, loc, ip), _i32(b, loc, ip)],
            f"prmt.b32 $0, $1, $2, {sel};",
            "=r,r,r",
            False,
            loc,
            ip,
        )
    )


def _conv_case_words(case, w, crs):
    # 12 output words (24 bf16: channel c, position j at index 3c + j) of one octet's new Conv
    # window for accepted count case 1, 2 or >= 3 (3): position j takes sequence element A + j of
    # [old0, old1, old2, cand0, cand1, cand2...]; crs[j] holds candidate row A + j - 3.
    def src(c, j):
        k = case + j  # sequence index (case 3 stands for any A >= 3)
        if k < 3:
            idx = 3 * c + k
            return ("w", idx // 2, idx % 2)
        return ("c", j, c // 2, c % 2)

    def reg(sv):
        return w[sv[1]] if sv[0] == "w" else crs[sv[1]][sv[2]]

    words = []
    for m in range(12):
        lo = src(*divmod(2 * m, 3))
        hi = src(*divmod(2 * m + 1, 3))
        rl, rh = reg(lo), reg(hi)
        hl, hh = lo[-1], hi[-1]
        if rl is rh:
            sel = (2 * hl) | ((2 * hl + 1) << 4) | ((2 * hh) << 8) | ((2 * hh + 1) << 12)
            words.append(prmt_b32(rl, rl, sel))
        else:
            sel = (2 * hl) | ((2 * hl + 1) << 4) | ((4 + 2 * hh) << 8) | ((5 + 2 * hh) << 12)
            words.append(prmt_b32(rl, rh, sel))
    return words


@cute.jit
def _conv_words(A, w, crs):
    # new window words for runtime A >= 1 (byte permutes, three cases selected per word)
    w1 = _conv_case_words(1, w, crs)
    w2 = _conv_case_words(2, w, crs)
    w3 = _conv_case_words(3, w, crs)
    is1 = Int32(1) if A == 1 else Int32(0)
    is2 = Int32(1) if A == 2 else Int32(0)
    return [sel_i32(is1, w1[m], sel_i32(is2, w2[m], w3[m])) for m in range(12)]


# ---------------------------------------------------------------------------
# shared-memory map (bytes)
# ---------------------------------------------------------------------------
ST_STATE = 0  # 18448 B checkpoint head (bulk copy)
U_PITCH = 272  # padded U row pitch -> conflict-free ldmatrix.trans
U_REG = 4  # persistent fold: U rows in 4 bulk copies of 6 rows (regions 1552 B apart)
U_RM = 6
U_RP = 1552
ST_U = 18560  # 32 x 272 B  U rows   (per-row bulk copies, rows >= n stale/zero)
ST_K = 27264  # 24 x 256 B  K rows   (bulk copy)
ST_G = 33408  # 24 x 512 B  G rows   (bulk copy)
STAGE = 45696
SM_ADJ = 2 * STAGE  # 32 x 256 B  bf16 Adj, column permuted + swizzled
SM_DEC = SM_ADJ + 8192
SM_RED = SM_DEC + 512
SM_MBAR = SM_RED + 64
LIST_N = 64  # rollover items per list round
SM_LHB = SM_MBAR + 32
SM_LK = SM_LHB + 8 * LIST_N
SM_LU = SM_LK + 8 * LIST_N
SM_LG = SM_LU + 8 * LIST_N
SM_LN = SM_LG + 8 * LIST_N
SM_SCAN = SM_LN + 4 * LIST_N
PAIR_CACHE = 384  # (layer, row) pairs whose scan results are cached for the item list
SM_PAIR = SM_SCAN + 32  # 8 B per pair: (slot << 5) | n (n = 0 for dummy rows), replay slot
CV_ROWS = 192  # Conv octets per staged chunk (one per thread of warps 0-5)
ROWT_N = 64
SM_ROWT = SM_PAIR + 8 * PAIR_CACHE  # per batch row: (slot << 5) | (A << 1) | dummy
LAYT_N = 16
SM_LAYT = SM_ROWT + 4 * ROWT_N  # per layer: Conv base, Conv slot stride (bytes), candidate base
SM_CVD = SM_LAYT + 24 * LAYT_N  # 2 chunk slots x 32 B: candidate base, window base, A | active << 8
SM_CVS = SM_CVD + 64  # 3 x CV_ROWS x 16 B staged candidate rows
CV_BATCH = 9  # leftover chunks staged at once in the (then idle) fold ring
SM_TOTAL = SM_CVS + 3 * 16 * CV_ROWS


@cute.kernel
def _commit_kernel(
    p_desc: Int64,
    p_slots: Int64,
    p_rslots: Int64,
    p_acc: Int64,
    p_dummy: Int64,
    nL: Int32,
    nB: Int32,
    nH: Int32,
    cap: Int32,
    nT: Int32,
    nP: cutlass.Constexpr[int],
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sbuf = smem.allocate(SM_TOTAL, byte_alignment=1024)
    sb = sbuf.toint()

    warp = tid // 32
    lane = tid % 32

    C = nH * 384

    # Phase A (Conv window commit) is bandwidth bound and phase B (checkpoint rollover) is
    # latency bound per CTA.  Mid-size launches: the B-first half, which also receives the extra
    # rollover items, skips the Conv phase; the A-first half covers all octets.  Large launches:
    # every CTA stages one Conv chunk per fold (bulk copies issued at the fold's first barrier,
    # committed after its stores) and commits the chunks left over after its folds.
    octs_lb = nH * 48
    total_oct = nL * nB * octs_lb
    large = Int32(1) if nL * nB * nH > 20 * nP else Int32(0)
    cv_on = large
    if nB > ROWT_N:
        cv_on = Int32(0)
    if nL > LAYT_N:
        cv_on = Int32(0)
    if octs_lb % CV_ROWS != 0:
        cv_on = Int32(0)
    cpl = octs_lb // CV_ROWS
    cv_c0 = nP - 1 - bid
    cv_K = Int32(0)
    if cv_on != 0:
        n_chunks = nL * nB * cpl
        if cv_c0 < n_chunks:
            cv_K = (n_chunks - cv_c0 + nP - 1) // nP
        for r in cutlass.range(tid, nB, THREADS):
            a_r = ld_g_s32(p_acc + Int64(r) * 4)
            d_r = ld_g_u8(p_dummy + Int64(r))
            s_r = ld_g_s32(p_slots + Int64(r) * 4)
            st_s_u32(
                sb + SM_ROWT + r * 4, (s_r << 5) | (a_r << 1) | (Int32(1) if d_r != 0 else Int32(0))
            )
        for li in cutlass.range(tid, nL, THREADS):
            drow = p_desc + Int64(li) * DESC_BYTES
            st_s_s64(sb + SM_LAYT + li * 24, ld_g_s64(drow + 16))
            st_s_s64(sb + SM_LAYT + li * 24 + 8, ld_g_s64(drow + 24) * 2)
            st_s_s64(sb + SM_LAYT + li * 24 + 16, ld_g_s64(drow + 72))
        if tid == 0:
            mbar_init(sb + SM_MBAR + 16, 1)
            fence_mbar_init()
    conv_ctas = nP
    conv_rank = nP - 1 - bid
    b_first = Int32(1) if bid * 2 < nP else Int32(0)
    if large == 0:
        conv_ctas = nP // 2
        conv_rank = nP - 1 - bid if bid * 2 >= nP else total_oct
    if cv_on != 0:
        conv_rank = total_oct
        b_first = Int32(1)
    gstride = conv_ctas * THREADS
    cv_ph = Int32(0)
    for ph in cutlass.range(2, unroll=1):
        if ph == b_first:
            # ================= phase A: Conv4 accepted-prefix commit (8 channels / thread, 2 octets in flight)
            # Conv octets start at the highest CTA (rolling items start at CTA 0)
            for idx0 in cutlass.range(conv_rank * THREADS + tid, total_oct, 2 * gstride):
                metas = []
                for u in cutlass.range_constexpr(2):
                    idx = idx0 + u * gstride
                    valid = Int32(1) if idx < total_oct else Int32(0)
                    idxc = idx if idx < total_oct else idx0
                    o = idxc % octs_lb
                    lb = idxc // octs_lb
                    b = lb % nB
                    layer = lb // nB
                    dm = ld_g_u8(p_dummy + Int64(b))
                    A = ld_g_s32(p_acc + Int64(b) * 4)
                    slot = ld_g_s32(p_slots + Int64(b) * 4)
                    coff = ld_g_s64(p_desc + Int64(layer) * DESC_BYTES + 16)
                    cstr = ld_g_s64(p_desc + Int64(layer) * DESC_BYTES + 24) * 2
                    metas.append((valid, o, lb, dm, A, slot, coff, cstr))
                loaded_data = []
                for u in cutlass.range_constexpr(2):
                    valid, o, lb, dm, A, slot, coff, cstr = metas[u]
                    act = valid if dm == 0 else Int32(0)
                    act = act if A != 0 else Int32(0)
                    slotc = slot if act != 0 else Int32(0)
                    addr = coff + Int64(slotc) * cstr + Int64(o) * 48
                    need_old = act if A < 3 else Int32(0)
                    w = (
                        ld_g_v4_pred(addr, need_old)
                        + ld_g_v4_pred(addr + 16, need_old)
                        + ld_g_v4_pred(addr + 32, need_old)
                    )
                    cand = ld_g_s64(p_desc + Int64(lb // nB) * DESC_BYTES + 72)
                    crow_base = cand + (Int64(lb % nB) * Int64(nT)) * Int64(C) * 2 + Int64(o) * 16
                    crs = []
                    for j in cutlass.range_constexpr(3):
                        src = A + j
                        use_c = act if src >= 3 else Int32(0)
                        row = src - 3
                        row = row if row > 0 else Int32(0)
                        row = row if row < nT else nT - 1
                        crs.append(ld_g_v4_pred(crow_base + Int64(row) * Int64(C) * 2, use_c))
                    loaded_data.append((act, addr, A, w, crs))
                for u in cutlass.range_constexpr(2):
                    act, addr, A, w, crs = loaded_data[u]
                    words = _conv_words(A, w, crs)
                    st_g_v4_pred(addr, words[0], words[1], words[2], words[3], act)
                    st_g_v4_pred(addr + 16, words[4], words[5], words[6], words[7], act)
                    st_g_v4_pred(addr + 32, words[8], words[9], words[10], words[11], act)

        else:
            # ================= phase B: recurrent checkpoint commits (globally balanced)
            # Every CTA scans all (layer,row) pairs, block-scans the rolling-pair counts and
            # takes exactly every nP-th rolling (pair, head) item -> even split of heavy work.
            LB = nL * nB
            cs = (LB + THREADS - 1) // THREADS
            p0 = tid * cs
            p1 = p0 + cs
            p1 = p1 if p1 < LB else LB
            # The layer descriptors (nL x 14 int64) go to the idle Conv staging area for the first
            # list pass (one value per thread, stored after the scan so the load overlaps it).
            DW = DESC_BYTES // 8
            use_tbl = Int32(1) if nL * DW <= THREADS else Int32(0)
            tbl_v = Int64(0)
            if use_tbl != 0:
                if tid < nL * DW:
                    tbl_v = ld_g_s64(p_desc + Int64(tid) * 8)
            cnt_t = Int32(0)
            # two pairs per iteration: both pairs' independent loads (incl. the history-length
            # pointers) first, then both length loads
            for pa in cutlass.range(p0, p1, 2, unroll=1):
                pb = pa + 1
                vb = Int32(1) if pb < p1 else Int32(0)
                pbc = pb if pb < p1 else pa
                la = pa // nB
                ba = pa - la * nB
                lbb = pbc // nB
                bb = pbc - lbb * nB
                dma = ld_g_u8(p_dummy + Int64(ba))
                Aa = ld_g_s32(p_acc + Int64(ba) * 4)
                olda = ld_g_s64(p_desc + Int64(la) * DESC_BYTES + 64)
                sla = ld_g_s32(p_slots + Int64(ba) * 4)
                rpa = ld_g_s32(p_rslots + Int64(ba) * 4)
                dmb = ld_g_u8(p_dummy + Int64(bb))
                Ab = ld_g_s32(p_acc + Int64(bb) * 4)
                oldb = ld_g_s64(p_desc + Int64(lbb) * DESC_BYTES + 64)
                slb = ld_g_s32(p_slots + Int64(bb) * 4)
                rpb = ld_g_s32(p_rslots + Int64(bb) * 4)
                hla = ld_g_s64(p_desc + Int64(la) * DESC_BYTES + 56)
                hstride_a = ld_g_s64(p_desc + Int64(la) * DESC_BYTES + 104)
                hlb = ld_g_s64(p_desc + Int64(lbb) * DESC_BYTES + 56)
                hstride_b = ld_g_s64(p_desc + Int64(lbb) * DESC_BYTES + 104)
                na = ld_g_s32(olda + Int64(ba) * 4) + Aa
                nbv = ld_g_s32(oldb + Int64(bb) * 4) + Ab
                rla = Int32(1) if na >= 16 else Int32(0)
                rla = rla if dma == 0 else Int32(0)
                rlb = Int32(1) if nbv >= 16 else Int32(0)
                rlb = rlb if dmb == 0 else Int32(0)
                rlb = rlb if vb != 0 else Int32(0)
                cnt_t = cnt_t + rla + rlb
                if LB <= PAIR_CACHE:
                    st_s_u32(sb + SM_PAIR + pa * 8, (sla << 5) | (na if rla != 0 else Int32(0)))
                    st_s_u32(sb + SM_PAIR + pa * 8 + 4, rpa)
                    if vb != 0:
                        st_s_u32(
                            sb + SM_PAIR + pb * 8, (slb << 5) | (nbv if rlb != 0 else Int32(0))
                        )
                        st_s_u32(sb + SM_PAIR + pb * 8 + 4, rpb)
                # history length of the pair this CTA owns (p % nP == bid): new length, or 0
                if pa % nP == bid:
                    if dma == 0:
                        st_g_s32(hla + Int64(rpa) * hstride_a, Int32(0) if rla != 0 else na)
                if vb != 0:
                    if pb % nP == bid:
                        if dmb == 0:
                            st_g_s32(hlb + Int64(rpb) * hstride_b, Int32(0) if rlb != 0 else nbv)
            if use_tbl != 0:
                if tid < nL * DW:
                    st_s_s64(sb + SM_CVS + tid * 8, tbl_v)
            # block exclusive scan of cnt_t
            v = cnt_t
            for sh in cutlass.range_constexpr(5):
                y = shfl_up_i32(v, 1 << sh)
                v = v + (y if lane >= (1 << sh) else Int32(0))
            if lane == 31:
                st_s_u32(sb + SM_SCAN + warp * 4, v)
            cute.arch.barrier()
            wpre = Int32(0)
            tot = Int32(0)
            for w8 in cutlass.range_constexpr(8):
                ws = ld_s_u32(sb + SM_SCAN + w8 * 4)
                wpre = wpre + (ws if w8 < warp else Int32(0))
                tot = tot + ws
            excl = wpre + v - cnt_t
            r_tot = tot * nH
            my_total = Int32(0)
            if bid < r_tot:
                my_total = (r_tot - bid + nP - 1) // nP
            seq = Int32(0)
            g = lane // 4
            t = lane % 4
            mi = lane // 8
            rr = lane % 8
            # Conv-only/no-roll calls never touch the MMA staging ring.  Defer its
            # clear and barrier initialization until this CTA actually owns work.
            if my_total > 0:
                for q in cutlass.range_constexpr(5):
                    zi = tid + q * THREADS
                    if zi < 1088:
                        st_s_v4(
                            sb + (zi // 544) * STAGE + ST_U + (zi % 544) * 16,
                            Int32(0),
                            Int32(0),
                            Int32(0),
                            Int32(0),
                        )
                if tid < 128:
                    st_s_v4(
                        sb + SM_ADJ + 24 * 256 + tid * 16, Int32(0), Int32(0), Int32(0), Int32(0)
                    )
                if tid == 0:
                    mbar_init(sb + SM_MBAR, 1)
                    mbar_init(sb + SM_MBAR + 8, 1)
                    fence_mbar_init()
                fence_proxy_async()
                cute.arch.barrier()
            if tid == 224:
                if cv_K > 0:
                    _cv_describe(sb, sb + SM_CVD, cv_c0, cpl, nB, nT, C)
            for wbase in cutlass.range(0, my_total, LIST_N):
                cute.arch.barrier()
                rho = excl
                if cnt_t > 0:
                    list_layer = p0 // nB
                    list_batch = p0 - list_layer * nB
                    for p in cutlass.range(p0, p1):
                        b = list_batch
                        n = Int32(0)
                        slot = Int32(0)
                        replay = Int64(0)
                        rl = Int32(0)
                        if LB <= PAIR_CACHE:
                            pa = ld_s_u32(sb + SM_PAIR + p * 8)
                            n = pa & 31
                            slot = pa >> 5
                            replay = Int64(ld_s_u32(sb + SM_PAIR + p * 8 + 4))
                            rl = Int32(1) if n >= 16 else Int32(0)
                        else:
                            dm = ld_g_u8(p_dummy + Int64(b))
                            A = ld_g_s32(p_acc + Int64(b) * 4)
                            old = ld_g_s64(p_desc + Int64(p // nB) * DESC_BYTES + 64)
                            n = ld_g_s32(old + Int64(b) * 4) + A
                            rl = Int32(1) if n >= 16 else Int32(0)
                            rl = rl if dm == 0 else Int32(0)
                            slot = ld_g_s32(p_slots + Int64(b) * 4)
                            replay = Int64(ld_g_s32(p_rslots + Int64(b) * 4))
                        if rl != 0:
                            x = (rho * nH) % nP
                            h0 = (bid - x + nP) % nP
                            for h in cutlass.range(h0, nH, nP):
                                kk = (rho * nH + h - bid) // nP - wbase
                                if kk >= 0:
                                    if kk < LIST_N:
                                        layer = list_layer
                                        state = Int64(0)
                                        sstr = Int64(0)
                                        hk = Int64(0)
                                        hu = Int64(0)
                                        hg = Int64(0)
                                        use_t = use_tbl if wbase == 0 else Int32(0)
                                        if use_t != 0:
                                            trow = sb + SM_CVS + layer * DESC_BYTES
                                            state = ld_s_s64(trow)
                                            sstr = ld_s_s64(trow + 8)
                                            hk = ld_s_s64(trow + 32) + replay * ld_s_s64(trow + 80)
                                            hu = ld_s_s64(trow + 40) + replay * ld_s_s64(trow + 88)
                                            hg = ld_s_s64(trow + 48) + replay * ld_s_s64(trow + 96)
                                        else:
                                            row = p_desc + Int64(layer) * DESC_BYTES
                                            state = ld_g_s64(row)
                                            sstr = ld_g_s64(row + 8)
                                            hk = ld_g_s64(row + 32) + replay * ld_g_s64(row + 80)
                                            hu = ld_g_s64(row + 40) + replay * ld_g_s64(row + 88)
                                            hg = ld_g_s64(row + 48) + replay * ld_g_s64(row + 96)
                                        hb = state + Int64(slot) * sstr + Int64(h) * STATE_BYTES
                                        # Row (h, 0) of the request's replay slot.
                                        hist = Int64(h) * Int64(cap)
                                        st_s_s64(sb + SM_LHB + kk * 8, hb)
                                        st_s_s64(sb + SM_LK + kk * 8, hk + hist * 256)
                                        st_s_s64(sb + SM_LU + kk * 8, hu + hist * 256)
                                        st_s_s64(sb + SM_LG + kk * 8, hg + hist * 512)
                                        st_s_u32(sb + SM_LN + kk * 4, n)
                            rho = rho + 1
                        list_batch = list_batch + Int32(1)
                        if list_batch == nB:
                            list_batch = Int32(0)
                            list_layer = list_layer + Int32(1)
                cute.arch.barrier()
                cnt = my_total - wbase
                cnt = cnt if cnt < LIST_N else Int32(LIST_N)
                if tid == 0:
                    _issue_entry(sb, seq % 2, Int32(0))
                    if cnt > 1:
                        _issue_entry(sb, (seq + 1) % 2, Int32(1))
                for jx in cutlass.range(cnt):
                    q = seq + jx
                    stg = q % 2
                    par = (q // 2) % 2
                    n = ld_s_u32(sb + SM_LN + jx * 4)
                    hb = ld_s_s64(sb + SM_LHB + jx * 8)
                    stb = sb + stg * STAGE
                    done = Int32(0)
                    while done == 0:
                        done = mbar_try_wait(sb + SM_MBAR + stg * 8, par)
                    _commit_one_p(
                        sb,
                        stb,
                        hb,
                        n,
                        tid,
                        warp,
                        lane,
                        g,
                        t,
                        mi,
                        rr,
                        stg,
                        jx,
                        cnt,
                        wbase + jx,
                        cv_K,
                        cv_c0,
                        nP,
                        cpl,
                        nB,
                        nT,
                        C,
                    )
                    kc = wbase + jx
                    if kc < cv_K:
                        # describe the next chunk (idle warp 7), commit the chunk staged during this fold
                        if tid == 224:
                            if kc + 1 < cv_K:
                                _cv_describe(
                                    sb,
                                    sb + SM_CVD + ((kc + 1) % 2) * 32,
                                    cv_c0 + (kc + 1) * nP,
                                    cpl,
                                    nB,
                                    nT,
                                    C,
                                )
                        dsl = sb + SM_CVD + (kc % 2) * 32
                        flags = ld_s_u32(dsl + 16)
                        if (flags >> 8) != 0:
                            cdone = Int32(0)
                            while cdone == 0:
                                cdone = mbar_try_wait(sb + SM_MBAR + 16, cv_ph)
                            cv_ph = cv_ph ^ 1
                            if tid < CV_ROWS:
                                _cv_commit_octet(sb + SM_CVS, ld_s_s64(dsl + 8), tid, flags & 255)
                seq = seq + cnt
            if cv_on != 0:
                # chunks left over after the folds: staged CV_BATCH at a time in the idle fold ring
                k0 = my_total if my_total < cv_K else cv_K
                for kb in cutlass.range(k0, cv_K, CV_BATCH):
                    nb = cv_K - kb
                    nb = nb if nb < CV_BATCH else Int32(CV_BATCH)
                    cute.arch.barrier()
                    if tid == 0:
                        for q in cutlass.range(nb):
                            dq = sb + CV_BATCH * 48 * CV_ROWS + q * 32
                            _cv_describe(sb, dq, cv_c0 + (kb + q) * nP, cpl, nB, nT, C)
                            _cv_stage(sb + q * 48 * CV_ROWS, dq, sb + SM_MBAR + 16, nT, C, 0)
                        mbar_arrive(sb + SM_MBAR + 16)
                    cute.arch.barrier()
                    cdone = Int32(0)
                    while cdone == 0:
                        cdone = mbar_try_wait(sb + SM_MBAR + 16, cv_ph)
                    cv_ph = cv_ph ^ 1
                    for i in cutlass.range(tid, nb * CV_ROWS, THREADS):
                        q = i // CV_ROWS
                        dq = sb + CV_BATCH * 48 * CV_ROWS + q * 32
                        flags = ld_s_u32(dq + 16)
                        if (flags >> 8) != 0:
                            _cv_commit_octet(
                                sb + q * 48 * CV_ROWS,
                                ld_s_s64(dq + 8),
                                i - q * CV_ROWS,
                                flags & 255,
                            )


@cute.jit
def _cv_describe(sb, dsl, c, cpl, nB, nT, C):
    # Conv chunk c (CV_ROWS octets of one (layer, row)) -> descriptor slot: candidate base of the
    # chunk's octets in token row 0, window base, accepted count | active << 8
    lb = c // cpl
    o0 = (c - lb * cpl) * CV_ROWS
    b = lb % nB
    layer = lb // nB
    rowv = ld_s_u32(sb + SM_ROWT + b * 4)
    A = (rowv >> 1) & 15
    act = Int32(1) if (rowv & 1) == 0 else Int32(0)
    act = act if A != 0 else Int32(0)
    slot = rowv >> 5
    coff = ld_s_s64(sb + SM_LAYT + layer * 24)
    cstr = ld_s_s64(sb + SM_LAYT + layer * 24 + 8)
    cand = ld_s_s64(sb + SM_LAYT + layer * 24 + 16)
    st_s_s64(dsl, cand + (Int64(b) * Int64(nT)) * Int64(C) * 2 + Int64(o0) * 16)
    st_s_s64(dsl + 8, coff + Int64(slot) * cstr + Int64(o0) * 48)
    st_s_u32(dsl + 16, A | (act << 8))


@cute.jit
def _cv_stage(dst, dsl, mb, nT, C, arrive: cutlass.Constexpr[int]):
    # bulk copies of the three candidate rows feeding the accepted window of a described chunk
    flags = ld_s_u32(dsl + 16)
    if (flags >> 8) != 0:
        A = flags & 255
        if arrive == 1:
            mbar_expect_tx(mb, Int32(3 * 16 * CV_ROWS))
        else:
            mbar_expect_tx_only(mb, Int32(3 * 16 * CV_ROWS))
        crow_base = ld_s_s64(dsl)
        for j in cutlass.range_constexpr(3):
            row = A - 3 + j
            row = row if row > 0 else Int32(0)
            row = row if row < nT else nT - 1
            bulk_g2s(
                dst + j * 16 * CV_ROWS,
                crow_base + Int64(row) * Int64(C) * 2,
                Int32(16 * CV_ROWS),
                mb,
            )


@cute.jit
def _cv_commit_octet(stg, wbase, o, A):
    # new window of octet o of a staged chunk (A >= 3: three candidate rows only)
    addr = wbase + Int64(o) * 48
    crs = [ld_s_v4(stg + j * 16 * CV_ROWS + o * 16) for j in range(3)]
    if A >= 3:
        words = _conv_case_words(3, None, crs)
        st_g_v4(addr, words[0], words[1], words[2], words[3])
        st_g_v4(addr + 16, words[4], words[5], words[6], words[7])
        st_g_v4(addr + 32, words[8], words[9], words[10], words[11])
    else:
        one = Int32(1)
        w = ld_g_v4_pred(addr, one) + ld_g_v4_pred(addr + 16, one) + ld_g_v4_pred(addr + 32, one)
        words = _conv_words(A, w, crs)
        st_g_v4(addr, words[0], words[1], words[2], words[3])
        st_g_v4(addr + 16, words[4], words[5], words[6], words[7])
        st_g_v4(addr + 32, words[8], words[9], words[10], words[11])


@cute.jit
def _issue_entry(sb, stg, e):
    n = ld_s_u32(sb + SM_LN + e * 4)
    hb = ld_s_s64(sb + SM_LHB + e * 8)
    hk = ld_s_s64(sb + SM_LK + e * 8)
    hu = ld_s_s64(sb + SM_LU + e * 8)
    hg = ld_s_s64(sb + SM_LG + e * 8)
    stb = sb + stg * STAGE
    mb = sb + SM_MBAR + stg * 8
    mbar_expect_tx(mb, STATE_BYTES + n * 1024)
    bulk_g2s(stb + ST_STATE, hb, Int32(STATE_BYTES), mb)
    bulk_g2s(stb + ST_K, hk, n * 256, mb)
    bulk_g2s(stb + ST_G, hg, n * 512, mb)
    # U rows in U_REG region copies (region c: rows [U_RM c, U_RM c + U_RM)); the regions' bank offsets
    # differ, and the MMA k-slot order interleaves them -> conflict-free ldmatrix.trans
    for c in cutlass.range_constexpr(U_REG):
        nr = n - c * U_RM
        nr = nr if nr < U_RM else Int32(U_RM)
        if nr > 0:
            bulk_g2s(stb + ST_U + c * U_RP, hu + Int64(c * U_RM * 256), nr * 256, mb)


@cute.jit
def _commit_one(sb, stb, hb, n, tid, warp, lane, g, t, mi, rr, stg, jx, cnt):
    s_adj = sb + SM_ADJ
    s_dec = sb + SM_DEC
    s_red = sb + SM_RED + stg * 32
    gend = stb + ST_G + (n - 1) * 512

    # Adj = bf16(K * exp(Gend - G)) -> smem (column-permuted, swizzled); zero rows >= n
    for q in cutlass.range_constexpr(2):
        # rows >= 16 (second pass) go to warps 4-7: warps 0-3 compute the decay row instead.
        # Adj rows 24..31 are static zeros (n <= 23), cleared once per CTA.
        if q == 0 or tid >= 128:
            ci = tid if q == 0 else tid + 128
            i = ci // 16
            qc = ci % 16
            dst = s_adj + i * 256 + ((qc ^ (i % 8)) * 16)
            if i < n:
                ws = []
                for w in cutlass.range_constexpr(4):
                    pp = qc + 16 * w
                    kw = ld_s_u32(stb + ST_K + i * 256 + pp * 4)
                    g2 = ld_s_v2f(stb + ST_G + i * 512 + pp * 8)
                    e2 = ld_s_v2f(gend + pp * 8)
                    ws.append(adj_pair(kw, g2[0], g2[1], e2[0], e2[1]))
                st_s_v4(dst, ws[0], ws[1], ws[2], ws[3])
            else:
                st_s_v4(dst, Int32(0), Int32(0), Int32(0), Int32(0))
    if tid < 128:
        od = exp_f32(ld_s_f32(gend + tid * 4)) * ld_s_f32(stb + 18432)
        st_s_f32(s_dec + (((tid // 4) % 8) * 16 + (tid // 32) * 4 + tid % 4) * 4, od)

    # own state rows / columns
    r0 = warp * 16 + g
    r1 = r0 + 8
    wp0 = ld_s_v4(stb + r0 * 64 + t * 16)
    wr0 = ld_s_v4(stb + 8192 + r0 * 64 + t * 16)
    sp0 = ld_s_u16(stb + 16384 + r0 * 8 + t * 2)
    sr0 = ld_s_u16(stb + 17408 + r0 * 8 + t * 2)
    wp1 = ld_s_v4(stb + r1 * 64 + t * 16)
    wr1 = ld_s_v4(stb + 8192 + r1 * 64 + t * 16)
    sp1 = ld_s_u16(stb + 16384 + r1 * 8 + t * 2)
    sr1 = ld_s_u16(stb + 17408 + r1 * 8 + t * 2)
    cute.arch.barrier()

    # Decode old state * end decay directly into the MMA C fragments.  This
    # shortens packed-state/scale live ranges and avoids a second final-state
    # fragment beside the 64 FP32 MMA accumulators.
    acc = [[Float32(0.0) for _ in range(4)] for _ in range(16)]
    dec = []
    for q in cutlass.range_constexpr(8):
        dec += ld_s_v4f(s_dec + q * 64 + t * 16)
    wps = [wp0, wp1]
    wrs = [wr0, wr1]
    sps = [sp0, sp1]
    srs = [sr0, sr1]
    zeros = [Float32(0.0) for _ in range(8)]
    for hr in cutlass.range_constexpr(2):
        sph2 = e4m3x2_to_f16x2(sps[hr])
        srh2 = e4m3x2_to_f16x2(srs[hr])
        spab = [sph2, sph2 >> 16]
        srab = [srh2, srh2 >> 16]
        for q in cutlass.range_constexpr(4):
            old8 = deccomb8h(
                wps[hr][q], wrs[hr][q], spab[q // 2], srab[q // 2], dec[q * 8 : (q + 1) * 8], zeros
            )
            for m in cutlass.range_constexpr(8):
                acc[q * 4 + m // 2][m % 2 + 2 * hr] = old8[m]

    # update = U^T @ Adj   (warp: 16 rows x 128 cols), accumulating on C
    for s in cutlass.range_constexpr(2):
        ia = s * 16 + (mi // 2) * 8 + rr
        a = ldsm_x4_t(stb + ST_U + ia * U_PITCH + (warp * 2 + (mi % 2)) * 16)
        ib = s * 16 + (mi % 2) * 8 + rr
        for jp in cutlass.range_constexpr(8):
            jj = jp * 2 + mi // 2
            bb = ldsm_x4_t(s_adj + ib * 256 + ((jj ^ rr) * 16))
            acc[2 * jp] = mma_bf16(a, bb[0], bb[1], acc[2 * jp])
            acc[2 * jp + 1] = mma_bf16(a, bb[2], bb[3], acc[2 * jp + 1])

    # The MMA accumulators now are the final state; only remap them by row.
    fin = [[], []]
    bmaxs = []
    for hr in cutlass.range_constexpr(2):
        for q in cutlass.range_constexpr(4):
            for m in cutlass.range_constexpr(8):
                fin[hr].append(acc[q * 4 + m // 2][m % 2 + 2 * hr])
        # per 16-key block maxima (independent short trees), reused by the requantization
        for q in cutlass.range_constexpr(2):
            m0 = absmax8x(Float32(0.0), fin[hr][q * 16 : q * 16 + 8])
            m1 = absmax8x(Float32(0.0), fin[hr][q * 16 + 8 : q * 16 + 16])
            bmaxs.append(fmax_f32(m0, m1))
    amax = fmax_f32(fmax_f32(bmaxs[0], bmaxs[1]), fmax_f32(bmaxs[2], bmaxs[3]))
    amax = cute.arch.warp_redux_sync(value=amax, kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=True)
    if lane == 0:
        st_s_f32(s_red + warp * 4, amax)
    cute.arch.barrier()
    if tid == 0:
        if jx + 2 < cnt:
            _issue_entry(sb, stg, jx + 2)
    red = ld_s_v4f(s_red) + ld_s_v4f(s_red + 16)
    gmax = red[0]
    for q in cutlass.range_constexpr(1, 8):
        gmax = fmax_f32(gmax, red[q])
    outer, denom, rden = outer_and_denom(gmax)

    # requantize primary then residual and store
    rows = [r0, r1]
    for hr in cutlass.range_constexpr(2):
        row = rows[hr]
        qq = quant32x(fin[hr], outer, denom, rden, bmaxs[2 * hr], bmaxs[2 * hr + 1])
        st_g_v4(hb + Int64(row * 64 + t * 16), qq[0], qq[1], qq[2], qq[3])
        st_g_v4(hb + Int64(8192 + row * 64 + t * 16), qq[4], qq[5], qq[6], qq[7])
        st_g_u16(hb + Int64(16384 + row * 8 + t * 2), qq[8] & 65535)
        st_g_u16(hb + Int64(17408 + row * 8 + t * 2), (qq[8] >> 16) & 65535)
    if tid == 0:
        st_g_f32(hb + 18432, outer)
        st_g_s32(hb + 18436, Int32(0))
        st_g_s32(hb + 18440, Int32(0))
        st_g_s32(hb + 18444, Int32(0))


@cute.jit
def _commit_one_p(
    sb,
    stb,
    hb,
    n,
    tid,
    warp,
    lane,
    g,
    t,
    mi,
    rr,
    stg,
    jx,
    cnt,
    cv_k,
    cv_K,
    cv_c0,
    nP,
    cpl,
    nB,
    nT,
    C,
):
    s_adj = sb + SM_ADJ
    s_dec = sb + SM_DEC
    s_red = sb + SM_RED + stg * 32
    gend = stb + ST_G + (n - 1) * 512

    # Adj = bf16(K * exp(Gend - G)) -> smem (column-permuted, swizzled); zero rows >= n
    for q in cutlass.range_constexpr(2):
        # rows >= 16 (second pass) go to warps 4-7: warps 0-3 compute the decay row instead.
        # Adj rows 24..31 are static zeros (n <= 23), cleared once per CTA.
        if q == 0 or tid >= 128:
            ci = tid if q == 0 else tid + 128
            i = ci // 16
            # thread = one 16 B chunk qc of the (column-permuted) Adj row: its words w = 0..3 hold
            # key pairs pp = qc + 16 w -> conflict-free 8 B G loads and one 16 B store per thread
            qc = ci % 16
            ks = (i % U_RM) * U_REG + i // U_RM
            dst = s_adj + ks * 256 + ((qc ^ (ks % 8)) * 16)
            if i < n:
                ws = []
                for w in cutlass.range_constexpr(4):
                    pp = qc + 16 * w
                    kw = ld_s_u32(stb + ST_K + i * 256 + pp * 4)
                    g2 = ld_s_v2f(stb + ST_G + i * 512 + pp * 8)
                    e2 = ld_s_v2f(gend + pp * 8)
                    ws.append(adj_pair(kw, g2[0], g2[1], e2[0], e2[1]))
                st_s_v4(dst, ws[0], ws[1], ws[2], ws[3])
            else:
                st_s_v4(dst, Int32(0), Int32(0), Int32(0), Int32(0))
    if tid < 128:
        od = exp_f32(ld_s_f32(gend + tid * 4)) * ld_s_f32(stb + 18432)
        st_s_f32(s_dec + (((tid // 4) % 8) * 16 + (tid // 32) * 4 + tid % 4) * 4, od)

    # own state rows / columns
    r0 = warp * 16 + g
    r1 = r0 + 8
    wp0 = ld_s_v4(stb + r0 * 64 + t * 16)
    wr0 = ld_s_v4(stb + 8192 + r0 * 64 + t * 16)
    sp0 = ld_s_u16(stb + 16384 + r0 * 8 + t * 2)
    sr0 = ld_s_u16(stb + 17408 + r0 * 8 + t * 2)
    wp1 = ld_s_v4(stb + r1 * 64 + t * 16)
    wr1 = ld_s_v4(stb + 8192 + r1 * 64 + t * 16)
    sp1 = ld_s_u16(stb + 16384 + r1 * 8 + t * 2)
    sr1 = ld_s_u16(stb + 17408 + r1 * 8 + t * 2)
    cute.arch.barrier()
    if tid == 0:
        if cv_k < cv_K:
            _cv_stage(sb + SM_CVS, sb + SM_CVD + (cv_k % 2) * 32, sb + SM_MBAR + 16, nT, C, 1)

    # Decode old state * end decay directly into the MMA C fragments.  This
    # shortens packed-state/scale live ranges and avoids a second final-state
    # fragment beside the 64 FP32 MMA accumulators.
    acc = [[Float32(0.0) for _ in range(4)] for _ in range(16)]
    dec = []
    for q in cutlass.range_constexpr(8):
        dec += ld_s_v4f(s_dec + q * 64 + t * 16)
    wps = [wp0, wp1]
    wrs = [wr0, wr1]
    sps = [sp0, sp1]
    srs = [sr0, sr1]
    zeros = [Float32(0.0) for _ in range(8)]
    for hr in cutlass.range_constexpr(2):
        sph2 = e4m3x2_to_f16x2(sps[hr])
        srh2 = e4m3x2_to_f16x2(srs[hr])
        spab = [sph2, sph2 >> 16]
        srab = [srh2, srh2 >> 16]
        for q in cutlass.range_constexpr(4):
            old8 = deccomb8h(
                wps[hr][q], wrs[hr][q], spab[q // 2], srab[q // 2], dec[q * 8 : (q + 1) * 8], zeros
            )
            for m in cutlass.range_constexpr(8):
                acc[q * 4 + m // 2][m % 2 + 2 * hr] = old8[m]

    # update = U^T @ Adj   (warp: 16 rows x 128 cols), accumulating on C
    # k slots 0..15 as one K=16 step, slots 16..23 as a K=8 step (slots 24..31 hold no rows, n <= 23)
    ia = (mi // 2) * 8 + rr
    a = ldsm_x4_t(
        stb + ST_U + (ia % U_REG) * U_RP + (ia // U_REG) * 256 + (warp * 2 + (mi % 2)) * 16
    )
    ib = (mi % 2) * 8 + rr
    for jp in cutlass.range_constexpr(8):
        jj = jp * 2 + mi // 2
        bb = ldsm_x4_t(s_adj + ib * 256 + ((jj ^ rr) * 16))
        acc[2 * jp] = mma_bf16(a, bb[0], bb[1], acc[2 * jp])
        acc[2 * jp + 1] = mma_bf16(a, bb[2], bb[3], acc[2 * jp + 1])
    ia8 = 16 + rr
    a8 = ldsm_x2_t(
        stb + ST_U + (ia8 % U_REG) * U_RP + (ia8 // U_REG) * 256 + (warp * 2 + (mi % 2)) * 16
    )
    for jq in cutlass.range_constexpr(4):
        bb4 = ldsm_x4_t(s_adj + (16 + rr) * 256 + (((jq * 4 + mi) ^ rr) * 16))
        for u in cutlass.range_constexpr(4):
            acc[jq * 4 + u] = mma_bf16_k8(a8[0], a8[1], bb4[u], acc[jq * 4 + u])

    # The MMA accumulators now are the final state; only remap them by row.
    fin = [[], []]
    bmaxs = []
    for hr in cutlass.range_constexpr(2):
        for q in cutlass.range_constexpr(4):
            for m in cutlass.range_constexpr(8):
                fin[hr].append(acc[q * 4 + m // 2][m % 2 + 2 * hr])
        # per 16-key block maxima (independent short trees), reused by the requantization
        for q in cutlass.range_constexpr(2):
            m0 = absmax8x(Float32(0.0), fin[hr][q * 16 : q * 16 + 8])
            m1 = absmax8x(Float32(0.0), fin[hr][q * 16 + 8 : q * 16 + 16])
            bmaxs.append(fmax_f32(m0, m1))
    amax = fmax_f32(fmax_f32(bmaxs[0], bmaxs[1]), fmax_f32(bmaxs[2], bmaxs[3]))
    amax = cute.arch.warp_redux_sync(value=amax, kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=True)
    if lane == 0:
        st_s_f32(s_red + warp * 4, amax)
    cute.arch.barrier()
    if tid == 0:
        if jx + 2 < cnt:
            _issue_entry(sb, stg, jx + 2)
    red = ld_s_v4f(s_red) + ld_s_v4f(s_red + 16)
    gmax = red[0]
    for q in cutlass.range_constexpr(1, 8):
        gmax = fmax_f32(gmax, red[q])
    outer, denom, rden = outer_and_denom(gmax)

    # requantize primary then residual and store
    rows = [r0, r1]
    for hr in cutlass.range_constexpr(2):
        row = rows[hr]
        qq = quant32x(fin[hr], outer, denom, rden, bmaxs[2 * hr], bmaxs[2 * hr + 1])
        st_g_v4(hb + Int64(row * 64 + t * 16), qq[0], qq[1], qq[2], qq[3])
        st_g_v4(hb + Int64(8192 + row * 64 + t * 16), qq[4], qq[5], qq[6], qq[7])
        st_g_u16(hb + Int64(16384 + row * 8 + t * 2), qq[8] & 65535)
        st_g_u16(hb + Int64(17408 + row * 8 + t * 2), (qq[8] >> 16) & 65535)
    if tid == 0:
        st_g_f32(hb + 18432, outer)
        st_g_s32(hb + 18436, Int32(0))
        st_g_s32(hb + 18440, Int32(0))
        st_g_s32(hb + 18444, Int32(0))


# ---------------------------------------------------------------------------
# Small launches: one CTA per candidate (layer, row, head) item, no scan.  Each CTA commits its
# head's Conv window slice, the head-0 CTA advances the length, and the CTA folds the head when the
# row rolls over (same fold as the persistent kernel, one item, no ring).
# ---------------------------------------------------------------------------
@cute.kernel
def _commit_small_kernel(
    p_desc: Int64,
    p_slots: Int64,
    p_rslots: Int64,
    p_acc: Int64,
    p_dummy: Int64,
    nL: Int32,
    nB: Int32,
    nH: Int32,
    cap: Int32,
    nT: Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()
    sbuf = smem.allocate(SM_TOTAL, byte_alignment=1024)
    sb = sbuf.toint()
    warp = tid // 32
    lane = tid % 32
    # pair-fastest order: CTAs placed on one SM belong to different (layer, row) pairs, whose
    # rollovers are independent (all heads of a pair roll together)
    nLB = nL * nB
    h = bid // nLB
    lb = bid % nLB
    b = lb % nB
    layer = lb // nB
    row = p_desc + Int64(layer) * DESC_BYTES
    dm = ld_g_u8(p_dummy + Int64(b))
    A = ld_g_s32(p_acc + Int64(b) * 4)
    slot = ld_g_s32(p_slots + Int64(b) * 4)
    replay = Int64(ld_g_s32(p_rslots + Int64(b) * 4))
    oldp = ld_g_s64(row + 64)
    # layer descriptor constants for the bulk copies (metadata, not slot dereferences), fetched in
    # the first round trip together with the row's flag, slot and replay slot
    d_st = Int64(0)
    d_ss = Int64(0)
    d_hk = Int64(0)
    d_hks = Int64(0)
    d_hg = Int64(0)
    d_hgs = Int64(0)
    d_hu = Int64(0)
    d_hus = Int64(0)
    if tid == 0:
        d_st = ld_g_s64(row)
        d_ss = ld_g_s64(row + 8)
        d_hk = ld_g_s64(row + 32)
        d_hks = ld_g_s64(row + 80)
        d_hg = ld_g_s64(row + 48)
        d_hgs = ld_g_s64(row + 96)
        d_hu = ld_g_s64(row + 40)
        d_hus = ld_g_s64(row + 88)
    if dm == 0:
        # Speculative checkpoint stage: issued before the old length (one more dependent round trip)
        # is known; a rolling item then only waits for its history rows.  The barrier's single
        # arrival comes later (with the history bytes, or alone for a non-rolling item, which still
        # waits for the copy before its shared memory is released).
        mb = sb + SM_MBAR
        hist = Int64(h) * Int64(cap)
        if tid == 0:
            mbar_init(mb, 1)
            fence_mbar_init()
            fence_proxy_async()
            mbar_expect_tx_only(mb, Int32(STATE_BYTES))
            bulk_g2s(
                sb + ST_STATE,
                d_st + Int64(slot) * d_ss + Int64(h) * STATE_BYTES,
                Int32(STATE_BYTES),
                mb,
            )
        n = ld_g_s32(oldp + Int64(b) * 4) + A
        rl = Int32(1) if n >= 16 else Int32(0)
        if rl != 0:
            # zeroed U staging (rows >= n must stay finite), barrier, then the history bulk loads
            for q in cutlass.range_constexpr(3):
                zi = tid + q * THREADS
                if zi < 544:
                    st_s_v4(sb + ST_U + zi * 16, Int32(0), Int32(0), Int32(0), Int32(0))
            if tid < 128:
                st_s_v4(sb + SM_ADJ + 24 * 256 + tid * 16, Int32(0), Int32(0), Int32(0), Int32(0))
            if tid == 0:
                st_s_s64(sb + SM_LHB, d_hu + replay * d_hus + hist * 256)
                mbar_expect_tx(mb, n * 1024)
                bulk_g2s(sb + ST_K, d_hk + replay * d_hks + hist * 256, n * 256, mb)
                bulk_g2s(sb + ST_G, d_hg + replay * d_hgs + hist * 512, n * 512, mb)
            fence_proxy_async()
            cute.arch.barrier()
            # one U row per thread (rows land on the padded pitch for conflict-free ldmatrix.trans)
            if tid < n:
                hu = ld_s_s64(sb + SM_LHB)
                bulk_g2s(sb + ST_U + tid * U_PITCH, hu + Int64(tid) * 256, Int32(256), mb)
        if h == 0:
            if tid == 0:
                hlen = ld_g_s64(row + 56)
                hlstr = ld_g_s64(row + 104)
                st_g_s32(hlen + replay * hlstr, Int32(0) if rl != 0 else n)
        # Conv4 accepted prefix of this head's 3 x 128 channels: 48 octets
        if A != 0:
            if tid < 48:
                C = nH * 384
                o = (tid // 16) * (nH * 16) + h * 16 + tid % 16
                coff = ld_g_s64(row + 16)
                cstr = ld_g_s64(row + 24) * 2
                cand = ld_g_s64(row + 72)
                addr = coff + Int64(slot) * cstr + Int64(o) * 48
                need_old = Int32(1) if A < 3 else Int32(0)
                w = (
                    ld_g_v4_pred(addr, need_old)
                    + ld_g_v4_pred(addr + 16, need_old)
                    + ld_g_v4_pred(addr + 32, need_old)
                )
                crow_base = cand + (Int64(b) * Int64(nT)) * Int64(C) * 2 + Int64(o) * 16
                crs = []
                for j in cutlass.range_constexpr(3):
                    src = A + j
                    use_c = Int32(1) if src >= 3 else Int32(0)
                    crow = src - 3
                    crow = crow if crow > 0 else Int32(0)
                    crow = crow if crow < nT else nT - 1
                    crs.append(ld_g_v4_pred(crow_base + Int64(crow) * Int64(C) * 2, use_c))
                words = _conv_words(A, w, crs)
                st_g_v4(addr, words[0], words[1], words[2], words[3])
                st_g_v4(addr + 16, words[4], words[5], words[6], words[7])
                st_g_v4(addr + 32, words[8], words[9], words[10], words[11])
        if rl != 0:
            hb = ld_g_s64(row) + Int64(slot) * ld_g_s64(row + 8) + Int64(h) * STATE_BYTES
            done = Int32(0)
            while done == 0:
                done = mbar_try_wait(sb + SM_MBAR, Int32(0))
            _commit_one(
                sb,
                sb,
                hb,
                n,
                tid,
                warp,
                lane,
                lane // 4,
                lane % 4,
                lane // 8,
                lane % 8,
                Int32(0),
                Int32(0),
                Int32(1),
            )
        else:
            # non-rolling item: the speculative checkpoint copy must land before this CTA's shared
            # memory is released
            if tid == 0:
                mbar_arrive(mb)
                sdone = Int32(0)
                while sdone == 0:
                    sdone = mbar_try_wait(mb, Int32(0))


@cute.jit
def _host_small(
    descriptors: Int64,
    slots: Int64,
    replay_slots: Int64,
    accepted: Int64,
    dummy: Int64,
    layers: Int32,
    batch: Int32,
    heads: Int32,
    capacity: Int32,
    tokens: Int32,
    stream,
):
    _commit_small_kernel(
        descriptors,
        slots,
        replay_slots,
        accepted,
        dummy,
        layers,
        batch,
        heads,
        capacity,
        tokens,
    ).launch(
        grid=(layers * batch * heads, 1, 1),
        block=(THREADS, 1, 1),
        max_number_threads=[THREADS, 1, 1],
        min_blocks_per_mp=2,
        stream=stream,
    )


@cute.jit
def _host(
    descriptors: Int64,
    slots: Int64,
    replay_slots: Int64,
    accepted: Int64,
    dummy: Int64,
    layers: Int32,
    batch: Int32,
    heads: Int32,
    capacity: Int32,
    tokens: Int32,
    num_ctas: cutlass.Constexpr[int],
    stream,
):
    _commit_kernel(
        descriptors,
        slots,
        replay_slots,
        accepted,
        dummy,
        layers,
        batch,
        heads,
        capacity,
        tokens,
        num_ctas,
    ).launch(
        grid=(num_ctas, 1, 1),
        block=(THREADS, 1, 1),
        max_number_threads=[THREADS, 1, 1],
        min_blocks_per_mp=2,
        stream=stream,
    )


_compiled = {}
_compiled_small = {}
# candidate (layer, row, head) items up to which the scan-free one-CTA-per-item launch is used
SMALL_MAX_ITEMS = 400


def _get_compiled_small():
    device = torch.cuda.current_device()
    fn = _compiled_small.get(device)
    if fn is None:
        import cuda.bindings.driver as cuda

        fn = cute.compile(
            _host_small,
            *([Int64(0)] * 5),
            *([Int32(0)] * 5),
            cuda.CUstream(0),
        )
        _compiled_small[device] = fn
    return fn


def _get_compiled():
    device = torch.cuda.current_device()
    fn = _compiled.get(device)
    if fn is None:
        import cuda.bindings.driver as cuda

        num_ctas = 2 * torch.cuda.get_device_properties(device).multi_processor_count
        fn = cute.compile(
            _host,
            *([Int64(0)] * 5),
            *([Int32(0)] * 5),
            num_ctas,
            cuda.CUstream(0),
        )
        _compiled[device] = fn
    return fn


@torch.no_grad()
def commit(
    descriptors: torch.Tensor,
    slots: torch.Tensor,
    replay_slots: torch.Tensor,
    accepted_tokens: torch.Tensor,
    dummy: torch.Tensor,
    heads: int,
    capacity: int,
    max_tokens: int,
) -> None:
    """Commit accepted Conv inputs and roll over recurrent state across all layers.

    Each contiguous int64 ``[L, DESC_FIELDS]`` descriptor row holds the absolute
    checkpoint and Conv pointers and slot strides, the K/U/G/length pointers,
    the immutable old-length snapshot and candidate Conv pointers, then the
    K/U/G/length replay-slot strides. Conv slot strides count BF16 elements;
    the other strides count bytes. ``slots`` address the checkpoints and Conv
    windows, ``replay_slots`` the histories and lengths; both are int32 [B].
    Each slot's state and history payloads are dense and aligned. Runtime
    B/L/H and pool addresses are not compilation keys.
    """
    if not slots.numel() or not descriptors.shape[0]:
        return
    if (
        descriptors.dtype != torch.int64
        or descriptors.ndim != 2
        or descriptors.shape[1] != DESC_FIELDS
    ):
        raise ValueError(f"2FP4 MTP commit requires int64 [L,{DESC_FIELDS}] descriptors")
    if replay_slots.shape != slots.shape or replay_slots.dtype != torch.int32:
        raise ValueError("2FP4 MTP commit requires one int32 replay slot per batch row")
    if not descriptors.is_contiguous():
        raise ValueError("2FP4 MTP descriptors must be contiguous")
    if not 1 <= max_tokens <= 8 or not 16 + max_tokens <= capacity <= 24:
        raise ValueError("2FP4 MTP commit requires 1..8 tokens and sufficient history capacity")
    import cuda.bindings.driver as cuda

    n_items = descriptors.shape[0] * slots.numel() * heads
    if n_items <= SMALL_MAX_ITEMS:
        _get_compiled_small()(
            descriptors.data_ptr(),
            slots.data_ptr(),
            replay_slots.data_ptr(),
            accepted_tokens.data_ptr(),
            dummy.data_ptr(),
            descriptors.shape[0],
            slots.numel(),
            heads,
            capacity,
            max_tokens,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return
    fn = _get_compiled()
    fn(
        descriptors.data_ptr(),
        slots.data_ptr(),
        replay_slots.data_ptr(),
        accepted_tokens.data_ptr(),
        dummy.data_ptr(),
        descriptors.shape[0],
        slots.numel(),
        heads,
        capacity,
        max_tokens,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
