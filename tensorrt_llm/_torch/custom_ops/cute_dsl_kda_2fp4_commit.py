# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed 2FP4 KDA checkpoint layout and the CuTe DSL codec helpers (SM100/SM103).

Each head of a checkpoint is ``STATE_BYTES`` bytes: primary and residual E2M1
values (``VALUES_BYTES`` each, 128 rows x 64 B), their E4M3 block-16 scales
at ``PRIMARY_SCALE`` and ``RESIDUAL_SCALE`` (128 rows x 8 B each), one FP32
outer scale at ``OUTER_SCALE`` and 12 zero bytes. The decode replay, the MTP
commit and the packed prefill share these PTX load/store, decode, quantize
and TMEM helpers.
"""

from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

STATE_BYTES = 18448
VALUES_BYTES = 8192
PRIMARY_SCALE = 16384
RESIDUAL_SCALE = 17408
OUTER_SCALE = 18432

# ---------------------------------------------------------------------------
# inline PTX helpers
# ---------------------------------------------------------------------------


def _ir(x, loc=None, ip=None):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ld_global_v4(addr: Int64, *, loc=None, ip=None):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_ir(Int64(addr), loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_global_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_s64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [_ir(Int64(addr), loc, ip)],
            "ld.global.b64 $0, [$1];",
            "=l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Int64(addr), loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_f32x4(addr: Int64, *, loc=None, ip=None):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [_ir(Int64(addr), loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_global_v4(addr: Int64, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            _ir(Int64(addr), loc, ip),
            _ir(Uint32(a), loc, ip),
            _ir(Uint32(b), loc, ip),
            _ir(Uint32(c), loc, ip),
            _ir(Uint32(d), loc, ip),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u32(addr: Int64, v: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Uint32(v), loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u16(addr: Int64, v: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Uint32(v), loc, ip)],
        "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.b16 [$0], t; }",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_outer_pad(addr: Int64, v: Float32, *, loc=None, ip=None):
    """float32 outer scale + 12 zero padding bytes as one 16B store."""
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Float32(v), loc, ip)],
        "{ .reg .b32 t; mov.b32 t, $1; st.global.v4.b32 [$0], {t, 0, 0, 0}; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _f_exp(x: Float32, *, loc=None, ip=None) -> Float32:
    """exp(x) exactly as Triton lowers tl.exp: mul by log2(e) then ex2.approx."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(x), loc, ip)],
            "{ .reg .f32 t; mul.f32 t, $1, 0f3FB8AA3B; ex2.approx.f32 $0, t; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f_div(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip)],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f_mul(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip)],
            "mul.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f_min(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f_max(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f_abs(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f_neg(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip)],
            "neg.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f_fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip), _ir(Float32(c), loc, ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_max_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(x), loc, ip)],
            "redux.sync.max.f32 $0, $1, -1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# -- fp4 / fp8 conversion blocks -------------------------------------------

_DEQ8_ASM = """
{
 .reg .b8  pb<4>;
 .reg .b8  rb<4>;
 .reg .b32 ph<4>;
 .reg .b32 rh<4>;
 .reg .b16 pq<8>;
 .reg .b16 rq<8>;
 .reg .f32 mp<8>;
 .reg .f32 mr<8>;
 .reg .f32 t<8>;
 mov.b32 {pb0,pb1,pb2,pb3}, $8;
 mov.b32 {rb0,rb1,rb2,rb3}, $9;
 cvt.rn.f16x2.e2m1x2 ph0, pb0;
 cvt.rn.f16x2.e2m1x2 ph1, pb1;
 cvt.rn.f16x2.e2m1x2 ph2, pb2;
 cvt.rn.f16x2.e2m1x2 ph3, pb3;
 cvt.rn.f16x2.e2m1x2 rh0, rb0;
 cvt.rn.f16x2.e2m1x2 rh1, rb1;
 cvt.rn.f16x2.e2m1x2 rh2, rb2;
 cvt.rn.f16x2.e2m1x2 rh3, rb3;
 mov.b32 {pq0,pq1}, ph0;
 mov.b32 {pq2,pq3}, ph1;
 mov.b32 {pq4,pq5}, ph2;
 mov.b32 {pq6,pq7}, ph3;
 mov.b32 {rq0,rq1}, rh0;
 mov.b32 {rq2,rq3}, rh1;
 mov.b32 {rq4,rq5}, rh2;
 mov.b32 {rq6,rq7}, rh3;
"""
for _i in range(8):
    _DEQ8_ASM += f" cvt.f32.f16 mp{_i}, pq{_i};\n cvt.f32.f16 mr{_i}, rq{_i};\n"
for _i in range(8):
    _DEQ8_ASM += f" fma.rn.f32 t{_i}, mr{_i}, $11, 0f00000000;\n"
for _i in range(8):
    _DEQ8_ASM += f" fma.rn.f32 ${_i}, mp{_i}, $10, t{_i};\n"
_DEQ8_ASM += "}\n"


@dsl_user_op
def _dequant8(pw: Uint32, rw: Uint32, ep: Float32, er: Float32, *, loc=None, ip=None):
    """8 state values = fma(mag_p, exp_p, RN(mag_r*exp_r)) -- Triton's contraction."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [
            _ir(Uint32(pw), loc, ip),
            _ir(Uint32(rw), loc, ip),
            _ir(Float32(ep), loc, ip),
            _ir(Float32(er), loc, ip),
        ],
        _DEQ8_ASM,
        "=f,=f,=f,=f,=f,=f,=f,=f,r,r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(8))


_DEC8_ASM = """
{
 .reg .b8  pb<4>;
 .reg .b32 ph<4>;
 .reg .b16 pq<8>;
 mov.b32 {pb0,pb1,pb2,pb3}, $8;
 cvt.rn.f16x2.e2m1x2 ph0, pb0;
 cvt.rn.f16x2.e2m1x2 ph1, pb1;
 cvt.rn.f16x2.e2m1x2 ph2, pb2;
 cvt.rn.f16x2.e2m1x2 ph3, pb3;
 mov.b32 {pq0,pq1}, ph0;
 mov.b32 {pq2,pq3}, ph1;
 mov.b32 {pq4,pq5}, ph2;
 mov.b32 {pq6,pq7}, ph3;
"""
for _i in range(8):
    _DEC8_ASM += f" cvt.f32.f16 ${_i}, pq{_i};\n"
_DEC8_ASM += "}\n"


@dsl_user_op
def _decode8(pw: Uint32, *, loc=None, ip=None):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [_ir(Uint32(pw), loc, ip)],
        _DEC8_ASM,
        "=f,=f,=f,=f,=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(8))


_EXP4_ASM = """
{
 .reg .b16 s<2>;
 .reg .b32 h<2>;
 .reg .b16 q<4>;
 .reg .f32 f<4>;
 mov.b32 {s0,s1}, $4;
 cvt.rn.f16x2.e4m3x2 h0, s0;
 cvt.rn.f16x2.e4m3x2 h1, s1;
 mov.b32 {q0,q1}, h0;
 mov.b32 {q2,q3}, h1;
 cvt.f32.f16 f0, q0;
 cvt.f32.f16 f1, q1;
 cvt.f32.f16 f2, q2;
 cvt.f32.f16 f3, q3;
 mul.f32 $0, f0, $5;
 mul.f32 $1, f1, $5;
 mul.f32 $2, f2, $5;
 mul.f32 $3, f3, $5;
}
"""


@dsl_user_op
def _expand4(scw: Uint32, outer: Float32, *, loc=None, ip=None):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_ir(Uint32(scw), loc, ip), _ir(Float32(outer), loc, ip)],
        _EXP4_ASM,
        "=f,=f,=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _pack_e4m3x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip)],
            "{ .reg .b16 s; cvt.rn.satfinite.e4m3x2.f32 s, $2, $1; cvt.u32.u16 $0, s; }",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_e4m3x4(a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                _ir(Float32(a), loc, ip),
                _ir(Float32(b), loc, ip),
                _ir(Float32(c), loc, ip),
                _ir(Float32(d), loc, ip),
            ],
            "{ .reg .b16 s0, s1;\n"
            " cvt.rn.satfinite.e4m3x2.f32 s0, $2, $1;\n"
            " cvt.rn.satfinite.e4m3x2.f32 s1, $4, $3;\n"
            " mov.b32 $0, {s0, s1}; }",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_e2m1x8(v, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Float32(x), loc, ip) for x in v],
            "{ .reg .b8 t<4>;\n"
            " cvt.rn.satfinite.e2m1x2.f32 t0, $2, $1;\n"
            " cvt.rn.satfinite.e2m1x2.f32 t1, $4, $3;\n"
            " cvt.rn.satfinite.e2m1x2.f32 t2, $6, $5;\n"
            " cvt.rn.satfinite.e2m1x2.f32 t3, $8, $7;\n"
            " mov.b32 $0, {t0, t1, t2, t3}; }",
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# -- TMEM raw access --------------------------------------------------------


def _tmem_regs_asm(op: str, n: int) -> str:
    regs = ", ".join(f"${i}" for i in range(n))
    if op == "st":
        return f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [${n}], {{{regs}}};\ntcgen05.wait::st.sync.aligned;"
    return f"tcgen05.ld.sync.aligned.32x32b.x{n}.b32 {{{regs}}}, [${n}];\ntcgen05.wait::ld.sync.aligned;"


@dsl_user_op
def _tmem_st32(addr: Uint32, vals, *, loc=None, ip=None):
    cons = ",".join(["f"] * 32) + ",r"
    llvm.inline_asm(
        None,
        [_ir(Float32(v), loc, ip) for v in vals] + [_ir(Uint32(addr), loc, ip)],
        _tmem_regs_asm("st", 32),
        cons,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tmem_ld32(addr: Uint32, *, loc=None, ip=None):
    cons = ",".join(["=f"] * 32) + ",r"
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 32),
        [_ir(Uint32(addr), loc, ip)],
        _tmem_regs_asm("ld", 32),
        cons,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def _fence_async_smem(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bf16x2_to_f32x2(x: Uint32, *, loc=None, ip=None):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_ir(Uint32(x), loc, ip)],
        "{ .reg .b32 a, b; shl.b32 a, $2, 16; and.b32 b, $2, -65536;"
        " mov.b32 $0, a; mov.b32 $1, b; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), r, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), r, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _sel_eq0(x: Float32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """(x == 0.0) ? a : b"""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(x), loc, ip), _ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip)],
            "{ .reg .pred p; setp.eq.f32 p, $1, 0f00000000; selp.f32 $0, $2, $3, p; }",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _tree_max(vals: list[Float32]) -> Float32:
    cur = list(vals)
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(_f_max(cur[i], cur[i + 1]))
        if len(cur) & 1:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]
