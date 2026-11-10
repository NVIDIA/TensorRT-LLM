# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-layer accepted-prefix Conv4 commit + native 2FP4 checkpoint commit (CuTe DSL).

One launch. Work item = (layer, batch row, head):
  * Conv4 history slice for the head's 384 channels (always, for non-dummy rows).
  * history_len update (head 0 only).
  * If the row rolls over (P+A>=16, or flush and P+A>0): reconstruct the 128x128
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

STATE_BYTES = 18448
THREADS = 256

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
            "{\n.reg .f32 t;\nmul.rn.f32 t, $1, 0f3FB8AA3B;\nex2.approx.f32 $0, t;\n}",
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
            "{\n.reg .f32 k0, k1, d0, d1;\n.reg .b32 t0, t1;\n"
            "shl.b32 t0, $1, 16;\nand.b32 t1, $1, 0xffff0000;\nmov.b32 k0, t0;\nmov.b32 k1, t1;\n"
            "sub.rn.f32 d0, $4, $2;\nsub.rn.f32 d1, $5, $3;\n"
            "mul.rn.f32 d0, d0, 0f3FB8AA3B;\nmul.rn.f32 d1, d1, 0f3FB8AA3B;\n"
            "ex2.approx.f32 d0, d0;\nex2.approx.f32 d1, d1;\n"
            "mul.rn.f32 k0, k0, d0;\nmul.rn.f32 k1, k1, d1;\n"
            "cvt.rn.bf16x2.f32 $0, k1, k0;\n}",
            "=r,r,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def bits_f32(x, *, loc=None, ip=None):
    return Float32(_asm([T.f32()], [_i32(x, loc, ip)], "mov.b32 $0, $1;", "=f,r", False, loc, ip))


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


def _qpass2(xs, p, L, outer, denom, rden="$23"):
    """xs: 16 scalar f32 names; packed pairs X{p}0..7 must already exist."""
    L.append(f"max.abs.f32 {p}a, {xs[0]}, {xs[1]}, {xs[2]};")
    for i in range(3, 16, 2):
        if i + 1 < 16:
            L.append(f"max.abs.f32 {p}a, {p}a, {xs[i]}, {xs[i + 1]};")
        else:
            L.append(f"max.abs.f32 {p}a, {p}a, {xs[i]}, {xs[i]};")
    # correctly rounded a / denom via reciprocal + Markstein correction
    L.append(f"mul.rn.f32 {p}sf, {p}a, {rden};")
    L.append(f"fma.rn.f32 {p}tt, {p}sf, nden, {p}a;")
    L.append(f"fma.rn.f32 {p}sf, {p}tt, {rden}, {p}sf;")
    L.append(f"min.f32 {p}sf, {p}sf, 0f43E00000;")
    L.append(f"cvt.rn.satfinite.e4m3x2.f32 {p}hs, zf, {p}sf;")
    L.append(f"cvt.rn.f16x2.e4m3x2 {p}hw, {p}hs;")
    L.append(f"mov.b32 {{{p}hl, {p}hh}}, {p}hw;")
    L.append(f"cvt.f32.f16 {p}sc, {p}hl;")
    L.append(f"mul.rn.f32 {p}e, {p}sc, {outer};")
    L.append(f"rcp.approx.f32 {p}rc, {p}e;")
    L.append(f"setp.eq.f32 {p}pz, {p}e, 0f00000000;")
    L.append(f"neg.f32 {p}ne, {p}e;")
    L.append(f"mov.b64 {p}RC, {{{p}rc, {p}rc}};")
    L.append(f"mov.b64 {p}_negative_scale, {{{p}ne, {p}ne}};")
    for i in range(8):
        L.append(f"mul.rn.f32x2 {p}Q{i}, {p}X{i}, {p}RC;")
        L.append(f"fma.rn.f32x2 {p}T{i}, {p}Q{i}, {p}_negative_scale, {p}X{i};")
        L.append(f"fma.rn.f32x2 {p}Q{i}, {p}T{i}, {p}RC, {p}Q{i};")
        L.append(f"mov.b64 {{{p}q{2 * i}, {p}q{2 * i + 1}}}, {p}Q{i};")
        L.append(f"cvt.rn.satfinite.e2m1x2.f32 {p}c{i}, {p}q{2 * i + 1}, {p}q{2 * i};")
    L.append(f"mov.b32 {p}w0, {{{p}c0, {p}c1, {p}c2, {p}c3}};")
    L.append(f"mov.b32 {p}w1, {{{p}c4, {p}c5, {p}c6, {p}c7}};")
    L.append(f"selp.b32 {p}w0, 0, {p}w0, {p}pz;")
    L.append(f"selp.b32 {p}w1, 0, {p}w1, {p}pz;")


def _quant16x_text():
    src = [f"${i + 5}" for i in range(16)]
    L = ["{", ".reg .f32 zf, nden;", "mov.f32 zf, 0f00000000;", "neg.f32 nden, $22;"]
    for p in ("A", "B"):
        L.append(f".reg .f32 {p}q<16>, {p}a, {p}sf, {p}sc, {p}e, {p}rc, {p}ne, {p}tt;")
        L.append(f".reg .b64 {p}X<8>, {p}Q<8>, {p}T<8>, {p}RC, {p}_negative_scale;")
        L.append(f".reg .b16 {p}hs, {p}hl, {p}hh;")
        L.append(f".reg .b32 {p}hw, {p}w0, {p}w1;")
        L.append(f".reg .b8 {p}c<8>;")
        L.append(f".reg .pred {p}pz;")
    L.append(".reg .b32 dh<8>, s0, s1;")
    L.append(".reg .b16 dl<16>;")
    L.append(".reg .f32 lv<16>, rv<16>;")
    L.append(".reg .b64 LV<8>;")
    for i in range(8):
        L.append(f"mov.b64 AX{i}, {{{src[2 * i]}, {src[2 * i + 1]}}};")
    _qpass2(src, "A", L, "$21", "$22")
    for i in range(8):
        L.append(f"cvt.rn.f16x2.e2m1x2 dh{i}, Ac{i};")
        L.append(f"mov.b32 {{dl{2 * i}, dl{2 * i + 1}}}, dh{i};")
        L.append(f"cvt.f32.f16 lv{2 * i}, dl{2 * i};")
        L.append(f"cvt.f32.f16 lv{2 * i + 1}, dl{2 * i + 1};")
        L.append(f"mov.b64 LV{i}, {{lv{2 * i}, lv{2 * i + 1}}};")
        L.append(f"fma.rn.f32x2 BX{i}, LV{i}, A_negative_scale, AX{i};")
        L.append(f"mov.b64 {{rv{2 * i}, rv{2 * i + 1}}}, BX{i};")
    _qpass2([f"rv{i}" for i in range(16)], "B", L, "$21", "$22")
    L += [
        "mov.b32 $0, Aw0;",
        "mov.b32 $1, Aw1;",
        "mov.b32 $2, Bw0;",
        "mov.b32 $3, Bw1;",
        "cvt.u32.u16 s0, Ahs;",
        "cvt.u32.u16 s1, Bhs;",
        "and.b32 s0, s0, 255;",
        "and.b32 s1, s1, 255;",
        "shl.b32 s1, s1, 8;",
        "or.b32 $4, s0, s1;",
        "}",
    ]
    return "\n".join(L)


_QUANT16X = _quant16x_text()


@dsl_user_op
def quant16x(xs, outer, denom, rden, *, loc=None, ip=None):
    args = [_f32(x, loc, ip) for x in xs] + [
        _f32(outer, loc, ip),
        _f32(denom, loc, ip),
        _f32(rden, loc, ip),
    ]
    r = _asm([T.i32()] * 5, args, _QUANT16X, "=r,=r,=r,=r,=r" + ",f" * 19, False, loc, ip)
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


# ---------------------------------------------------------------------------
# shared-memory map (bytes)
# ---------------------------------------------------------------------------
ST_STATE = 0  # 18448 B checkpoint head (bulk copy)
U_PITCH = 272  # padded U row pitch -> conflict-free ldmatrix.trans
ST_U = 18560  # 32 x 272 B  U rows   (per-row bulk copies, rows >= n stale/zero)
ST_K = 27264  # 24 x 256 B  K rows   (bulk copy)
ST_G = 33408  # 24 x 512 B  G rows   (bulk copy)
STAGE = 45696
SM_ADJ = 2 * STAGE  # 32 x 256 B  bf16 Adj, column permuted + swizzled
SM_DEC = SM_ADJ + 8192
SM_RED = SM_DEC + 512
SM_MBAR = SM_RED + 64
SM_CNT = SM_MBAR + 16
SM_LHB = SM_CNT + 16
SM_LK = SM_LHB + 2048
SM_LU = SM_LK + 2048
SM_LG = SM_LU + 2048
SM_LN = SM_LG + 2048
SM_SCAN = SM_LN + 1024
SM_TOTAL = SM_SCAN + 32


@cute.kernel
def _commit_kernel(
    p_desc: Int64,
    p_slots: Int64,
    p_smap: Int64,
    p_acc: Int64,
    p_dummy: Int64,
    flush: Int32,
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
    # issue bound: half of the CTAs run B first so that every SM overlaps the two.
    b_first = Int32(1) if bid * 2 < nP else Int32(0)
    for ph in cutlass.range(2, unroll=1):
        if ph == b_first:
            # ================= phase A: Conv4 accepted-prefix commit (8 channels / thread, 2 octets in flight)
            octs_lb = nH * 48
            total_oct = nL * nB * octs_lb
            gstride = nP * THREADS
            for idx0 in cutlass.range(bid * THREADS + tid, total_oct, 2 * gstride):
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
                    coff = ld_g_s64(p_desc + Int64(layer) * 80 + 16)
                    cstr = ld_g_s64(p_desc + Int64(layer) * 80 + 24) * 2
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
                    cand = ld_g_s64(p_desc + Int64(lb // nB) * 80 + 72)
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
                    outs = []
                    for j in cutlass.range_constexpr(3):
                        src = A + j
                        use_c = Int32(1) if src >= 3 else Int32(0)
                        is0 = Int32(1) if src == 0 else Int32(0)
                        is1 = Int32(1) if src == 1 else Int32(0)
                        cr = crs[j]
                        col = []
                        for c in cutlass.range_constexpr(8):
                            cv = (cr[c // 2] >> (16 * (c % 2))) & 0xFFFF
                            x0 = 3 * c
                            h0 = (w[x0 // 2] >> (16 * (x0 % 2))) & 0xFFFF
                            h1 = (w[(x0 + 1) // 2] >> (16 * ((x0 + 1) % 2))) & 0xFFFF
                            h2 = (w[(x0 + 2) // 2] >> (16 * ((x0 + 2) % 2))) & 0xFFFF
                            ov = sel_i32(is0, h0, sel_i32(is1, h1, h2))
                            col.append(sel_i32(use_c, cv, ov))
                        outs.append(col)
                    words = []
                    for m in cutlass.range_constexpr(12):
                        xa = 2 * m
                        xb = 2 * m + 1
                        words.append(outs[xa % 3][xa // 3] | (outs[xb % 3][xb // 3] << 16))
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
            cnt_t = Int32(0)
            scan_layer = p0 // nB
            scan_batch = p0 - scan_layer * nB
            for p in cutlass.range(p0, p1, unroll=4):
                b = scan_batch
                dm = ld_g_u8(p_dummy + Int64(b))
                A = ld_g_s32(p_acc + Int64(b) * 4)
                old = ld_g_s64(p_desc + Int64(p // nB) * 80 + 64)
                n = ld_g_s32(old + Int64(b) * 4) + A
                slot = ld_g_s32(p_slots + Int64(b) * 4)
                rl = Int32(1) if n >= 16 else Int32(0)
                if flush != 0:
                    rl = Int32(1) if n > 0 else Int32(0)
                rl = rl if dm == 0 else Int32(0)
                cnt_t = cnt_t + rl
                if p % nP == bid:
                    if dm == 0:
                        layer = scan_layer
                        live = ld_g_s32(p_smap + Int64(slot) * 4)
                        hlen = ld_g_s64(p_desc + Int64(layer) * 80 + 56)
                        st_g_s32(hlen + Int64(live) * 4, Int32(0) if rl != 0 else n)
                scan_batch = scan_batch + Int32(1)
                if scan_batch == nB:
                    scan_batch = Int32(0)
                    scan_layer = scan_layer + Int32(1)
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
                if tid == 0:
                    mbar_init(sb + SM_MBAR, 1)
                    mbar_init(sb + SM_MBAR + 8, 1)
                    fence_mbar_init()
                fence_proxy_async()
                cute.arch.barrier()
            for wbase in cutlass.range(0, my_total, THREADS):
                cute.arch.barrier()
                rho = excl
                if cnt_t > 0:
                    list_layer = p0 // nB
                    list_batch = p0 - list_layer * nB
                    for p in cutlass.range(p0, p1):
                        b = list_batch
                        dm = ld_g_u8(p_dummy + Int64(b))
                        A = ld_g_s32(p_acc + Int64(b) * 4)
                        old = ld_g_s64(p_desc + Int64(p // nB) * 80 + 64)
                        n = ld_g_s32(old + Int64(b) * 4) + A
                        rl = Int32(1) if n >= 16 else Int32(0)
                        if flush != 0:
                            rl = Int32(1) if n > 0 else Int32(0)
                        rl = rl if dm == 0 else Int32(0)
                        if rl != 0:
                            x = (rho * nH) % nP
                            h0 = (bid - x + nP) % nP
                            for h in cutlass.range(h0, nH, nP):
                                kk = (rho * nH + h - bid) // nP - wbase
                                if kk >= 0:
                                    if kk < THREADS:
                                        layer = list_layer
                                        slot = ld_g_s32(p_slots + Int64(b) * 4)
                                        live = ld_g_s32(p_smap + Int64(slot) * 4)
                                        state = ld_g_s64(p_desc + Int64(layer) * 80)
                                        sstr = ld_g_s64(p_desc + Int64(layer) * 80 + 8)
                                        hb = state + Int64(slot) * sstr + Int64(h) * STATE_BYTES
                                        hist = (Int64(live) * Int64(nH) + Int64(h)) * Int64(cap)
                                        hk = ld_g_s64(p_desc + Int64(layer) * 80 + 32)
                                        hu = ld_g_s64(p_desc + Int64(layer) * 80 + 40)
                                        hg = ld_g_s64(p_desc + Int64(layer) * 80 + 48)
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
                cnt = cnt if cnt < THREADS else Int32(THREADS)
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
                    _commit_one(sb, stb, hb, n, tid, warp, lane, g, t, mi, rr, stg, jx, cnt)
                seq = seq + cnt


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
    ub = hu
    for i in cutlass.range(n):
        bulk_g2s(stb + ST_U + i * U_PITCH, ub + Int64(i) * 256, Int32(256), mb)


@cute.jit
def _commit_one(sb, stb, hb, n, tid, warp, lane, g, t, mi, rr, stg, jx, cnt):
    s_adj = sb + SM_ADJ
    s_dec = sb + SM_DEC
    s_red = sb + SM_RED + stg * 32
    gend = stb + ST_G + (n - 1) * 512

    # Adj = bf16(K * exp(Gend - G)) -> smem (column-permuted, swizzled); zero rows >= n
    for q in cutlass.range_constexpr(2):
        ci = tid + q * THREADS
        i = ci // 16
        cch = ci % 16
        if i < n:
            kv = ld_s_v4(stb + ST_K + i * 256 + cch * 16)
            ga = ld_s_v4(stb + ST_G + i * 512 + cch * 32)
            gbv = ld_s_v4(stb + ST_G + i * 512 + cch * 32 + 16)
            ea = ld_s_v4(gend + cch * 32)
            eb = ld_s_v4(gend + cch * 32 + 16)
            gg = ga + gbv
            ee = ea + eb
            for m in cutlass.range_constexpr(4):
                w = adj_pair(
                    kv[m],
                    bits_f32(gg[2 * m]),
                    bits_f32(gg[2 * m + 1]),
                    bits_f32(ee[2 * m]),
                    bits_f32(ee[2 * m + 1]),
                )
                pp = cch * 4 + m
                qp = (pp % 16) * 4 + pp // 16
                qc = qp // 4
                st_s_u32(s_adj + i * 256 + ((qc ^ (i % 8)) * 16) + (qp % 4) * 4, w)
        else:
            for m in cutlass.range_constexpr(4):
                pp = cch * 4 + m
                qp = (pp % 16) * 4 + pp // 16
                qc = qp // 4
                st_s_u32(s_adj + i * 256 + ((qc ^ (i % 8)) * 16) + (qp % 4) * 4, Int32(0))
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
    amax = Float32(0.0)
    for hr in cutlass.range_constexpr(2):
        for q in cutlass.range_constexpr(4):
            for m in cutlass.range_constexpr(8):
                fin[hr].append(acc[q * 4 + m // 2][m % 2 + 2 * hr])
        for q in cutlass.range_constexpr(4):
            amax = absmax8x(amax, fin[hr][q * 8 : (q + 1) * 8])
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
        qa = quant16x(fin[hr][0:16], outer, denom, rden)
        qb = quant16x(fin[hr][16:32], outer, denom, rden)
        st_g_v4(hb + Int64(row * 64 + t * 16), qa[0], qa[1], qb[0], qb[1])
        st_g_v4(hb + Int64(8192 + row * 64 + t * 16), qa[2], qa[3], qb[2], qb[3])
        st_g_u16(hb + Int64(16384 + row * 8 + t * 2), (qa[4] & 255) | ((qb[4] & 255) << 8))
        st_g_u16(
            hb + Int64(17408 + row * 8 + t * 2), ((qa[4] >> 8) & 255) | (((qb[4] >> 8) & 255) << 8)
        )
    if tid == 0:
        st_g_f32(hb + 18432, outer)
        st_g_s32(hb + 18436, Int32(0))
        st_g_s32(hb + 18440, Int32(0))
        st_g_s32(hb + 18444, Int32(0))


@cute.jit
def _host(
    descriptors: Int64,
    slots: Int64,
    slot_map: Int64,
    accepted: Int64,
    dummy: Int64,
    flush: Int32,
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
        slot_map,
        accepted,
        dummy,
        flush,
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


def _get_compiled():
    device = torch.cuda.current_device()
    fn = _compiled.get(device)
    if fn is None:
        import cuda.bindings.driver as cuda

        num_ctas = 2 * torch.cuda.get_device_properties(device).multi_processor_count
        fn = cute.compile(
            _host,
            *([Int64(0)] * 5),
            *([Int32(0)] * 6),
            num_ctas,
            cuda.CUstream(0),
        )
        _compiled[device] = fn
    return fn


@torch.no_grad()
def commit(
    descriptors: torch.Tensor,
    slots: torch.Tensor,
    slot_map: torch.Tensor,
    accepted_tokens: torch.Tensor,
    dummy: torch.Tensor,
    heads: int,
    capacity: int,
    max_tokens: int,
    *,
    flush: bool = False,
) -> None:
    """Commit accepted Conv inputs and roll over recurrent state across all layers.

    The contiguous int64 [L,10] descriptor rows contain absolute checkpoint and
    Conv pointers/slot strides, K/U/G pointers, live lengths, immutable old
    lengths and candidate Conv pointers. Conv slot strides count BF16 elements;
    checkpoint strides count bytes. Inner state/history rows are dense and
    aligned. Runtime B/L/H and pool addresses are not compilation keys.
    """
    if not slots.numel() or not descriptors.shape[0]:
        return
    if descriptors.dtype != torch.int64 or descriptors.ndim != 2 or descriptors.shape[1] != 10:
        raise ValueError("2FP4 MTP commit requires int64 [L,10] descriptors")
    if not descriptors.is_contiguous():
        raise ValueError("2FP4 MTP descriptors must be contiguous")
    if not 1 <= max_tokens <= 8 or not 16 + max_tokens <= capacity <= 24:
        raise ValueError("2FP4 MTP commit requires 1..8 tokens and sufficient history capacity")
    import cuda.bindings.driver as cuda

    fn = _get_compiled()
    fn(
        descriptors.data_ptr(),
        slots.data_ptr(),
        slot_map.data_ptr(),
        accepted_tokens.data_ptr(),
        dummy.data_ptr(),
        int(flush),
        descriptors.shape[0],
        slots.numel(),
        heads,
        capacity,
        max_tokens,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
