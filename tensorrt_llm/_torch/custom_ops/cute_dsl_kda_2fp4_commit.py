# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA native 2FP4 checkpoint commit, one CuTe DSL launch (SM100/SM103).

One persistent kernel: per (batch, head) work item it reconstructs the FP4
primary+residual checkpoint state, applies the cumulative log-decay, fuses the
BF16 history update through one tcgen05 UMMA (accumulator pre-seeded in TMEM),
repacks both FP4 components with a shared outer scale, and advances/resets the
shared [R] length counters from the immutable old_lengths snapshot.
"""

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

STATE_BYTES = 18448
VALUES_BYTES = 8192
PRIMARY_SCALE = 16384
RESIDUAL_SCALE = 17408
OUTER_SCALE = 18432
NTHREADS = 256
NWARPS = NTHREADS // 32

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
def _ld_global_u8(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "{ .reg .b16 t; ld.global.b8 t, [$1]; cvt.u32.u16 $0, t; }",
            "=r,l",
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
def _st_global_s32(addr: Int64, v: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Int32(v), loc, ip)],
        "st.global.b32 [$0], $1;",
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


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


@cute.kernel
def _commit_kernel(
    tiled_mma: cute.TiledMma,
    pool_p: Int64,
    slots_p: Int64,
    slotmap_p: Int64,
    hk_p: Int64,
    hu_p: Int64,
    hg_p: Int64,
    lengths_p: Int64,
    oldlen_p: Int64,
    dummy_p: Int64,
    slot_stride: Int64,
    total: Int32,
    H: cutlass.Constexpr,
    CAP: cutlass.Constexpr,
    FULL: cutlass.Constexpr,
):
    atom_a = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.MN_INTER, cutlass.BFloat16)
    a_shape = cute.append(tiled_mma.partition_shape_A(cute.dice((128, 128, 16), (1, None, 1))), 1)
    b_shape = cute.append(tiled_mma.partition_shape_B(cute.dice((128, 128, 16), (None, 1, 1))), 1)
    sA_layout = tcgen05.tile_to_mma_shape(atom_a, a_shape, order=(2, 1, 3))
    sB_layout = tcgen05.tile_to_mma_shape(atom_a, b_shape, order=(2, 1, 3))
    acc_layout = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, 128))).layout
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    warp = tidx // 32
    lane = tidx % 32

    smem = utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout.outer, 1024, swizzle=sA_layout.inner)
    sB = smem.allocate_tensor(cutlass.BFloat16, sB_layout.outer, 1024, swizzle=sB_layout.inner)
    sExp = smem.allocate_tensor(Float32, cute.make_layout(128), byte_alignment=16)
    sLg = smem.allocate_tensor(Float32, cute.make_layout(128), byte_alignment=16)
    sRed = smem.allocate_tensor(Float32, cute.make_layout(NWARPS), byte_alignment=16)
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=8)
    tmem_slot = smem.allocate_array(cutlass.Int32, 1, byte_alignment=16)

    # --- TMEM allocation (one 128-column accumulator reused by every head) ---
    if warp == 0:
        cute.arch.alloc_tmem(128, tmem_slot, is_two_cta=False)
    cute.arch.barrier()
    tmem_view = cute.make_tensor(tmem_slot, cute.make_layout(1))
    tmem_base = Uint32(tmem_view[0])
    if warp == 0:
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

    tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_slot)
    tAcc = cute.make_tensor(tmem_ptr, acc_layout)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

    v = (warp % 4) * 32 + lane
    cg = warp // 4
    kbase = cg * 64
    my_tmem = tmem_base + Uint32((warp % 4) * (32 * 65536)) + Uint32(cg * 64)

    kg = tidx % 16
    ii = tidx // 16
    k0 = kg * 8

    idx = Int32(bidx)
    while idx < total:
        b = idx // H
        h = idx % H
        slot = _ld_global_s32(slots_p + Int64(b) * 4)
        live = _ld_global_s32(slotmap_p + Int64(slot) * 4)
        if live >= Int32(0):
            old = _ld_global_s32(oldlen_p + Int64(b) * 4)
            is_dummy = _ld_global_u8(dummy_p + Int64(b))
            if h == Int32(0):
                if tidx == Int32(0):
                    if cutlass.const_expr(FULL):
                        if is_dummy == Int32(0):
                            _st_global_s32(lengths_p + Int64(live) * 4, (old + 1) % 16)
                    else:
                        _st_global_s32(lengths_p + Int64(live) * 4, Int32(0))

            if cutlass.const_expr(FULL):
                eligible = (is_dummy == Int32(0)) & (old == Int32(15))
                n = Int32(16)
            else:
                eligible = old > Int32(0)
                n = old

            if eligible:
                head_base = pool_p + Int64(slot) * slot_stride + Int64(h) * STATE_BYTES
                byteoff = Int64(v) * 64 + Int64(cg) * 32
                pbase = head_base + byteoff
                rbase = pbase + VALUES_BYTES

                pw = list(_ld_global_v4(pbase)) + list(_ld_global_v4(pbase + 16))
                rw = list(_ld_global_v4(rbase)) + list(_ld_global_v4(rbase + 16))
                scoff = Int64(v) * 8 + Int64(cg) * 4
                pscw = _ld_global_u32(head_base + PRIMARY_SCALE + scoff)
                rscw = _ld_global_u32(head_base + RESIDUAL_SCALE + scoff)
                outer0 = _ld_global_f32(head_base + OUTER_SCALE)
                ep = _expand4(pscw, outer0)
                er = _expand4(rscw, outer0)

                hist_base = (Int64(live) * H + Int64(h)) * CAP * 128

                cute.arch.barrier()
                if tidx < Int32(128):
                    lgv = _ld_global_f32(hg_p + (hist_base + Int64(n - 1) * 128 + Int64(tidx)) * 4)
                    sLg[tidx] = lgv
                    sExp[tidx] = _f_exp(lgv)
                cute.arch.barrier()

                # ---- B operand: adjusted_k[i, k] = bf16(hk * exp(last_g - hg)) ----
                act = cutlass.Boolean(True)
                if cutlass.const_expr(not FULL):
                    act = ii < n
                if act:
                    kw = _ld_global_v4(hk_p + (hist_base + Int64(ii) * 128 + Int64(k0)) * 2)
                    ga = _ld_global_f32x4(hg_p + (hist_base + Int64(ii) * 128 + Int64(k0)) * 4)
                    gb = _ld_global_f32x4(hg_p + (hist_base + Int64(ii) * 128 + Int64(k0) + 4) * 4)
                    gg = list(ga) + list(gb)
                    for m2 in cutlass.range_constexpr(4):
                        kv = _bf16x2_to_f32x2(kw[m2])
                        for s2 in cutlass.range_constexpr(2):
                            jj = m2 * 2 + s2
                            e = _f_exp(sLg[k0 + jj] - gg[jj])
                            sB[(k0 + jj, ii), 0, 0, 0] = _f_mul(kv[s2], e).to(cutlass.BFloat16)
                else:
                    for jj in cutlass.range_constexpr(8):
                        sB[(k0 + jj, ii), 0, 0, 0] = cutlass.BFloat16(0.0)

                # ---- A operand: cached_u[v, i] = hu[i, v] ----
                v0 = kg * 8
                if act:
                    uw = _ld_global_v4(hu_p + (hist_base + Int64(ii) * 128 + Int64(v0)) * 2)
                    for m2 in cutlass.range_constexpr(4):
                        uv = _bf16x2_to_f32x2(uw[m2])
                        for s2 in cutlass.range_constexpr(2):
                            sA[(v0 + m2 * 2 + s2, ii), 0, 0, 0] = uv[s2].to(cutlass.BFloat16)
                else:
                    for jj in cutlass.range_constexpr(8):
                        sA[(v0 + jj, ii), 0, 0, 0] = cutlass.BFloat16(0.0)

                # ---- seed the TMEM accumulator with state * exp(last_g) ----
                for half in cutlass.range_constexpr(2):
                    old_half = []
                    for jj in cutlass.range_constexpr(4):
                        j = half * 4 + jj
                        old_half.extend(_dequant8(pw[j], rw[j], ep[j // 2], er[j // 2]))
                    seed = [_f_mul(old_half[i], sExp[kbase + half * 32 + i]) for i in range(32)]
                    _tmem_st32(my_tmem + Uint32(half * 32), seed)
                cute.arch.barrier()
                _fence_async_smem()
                if warp == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                if warp == 0:
                    cute.gemm(
                        tiled_mma, tAcc, tCrA[None, None, None, 0], tCrB[None, None, None, 0], tAcc
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar)
                cute.arch.barrier()
                cute.arch.mbarrier_wait(mbar, 0)
                cute.arch.barrier()

                # ---- outer scale: two 32-value passes reduce register liveness ----
                sn0 = _tmem_ld32(my_tmem)
                aa0 = [_f_abs(x) for x in sn0]
                bam0 = _tree_max(aa0[0:16])
                bam1 = _tree_max(aa0[16:32])
                m0 = _f_max(bam0, bam1)
                sn1 = _tmem_ld32(my_tmem + Uint32(32))
                aa1 = [_f_abs(x) for x in sn1]
                bam2 = _tree_max(aa1[0:16])
                bam3 = _tree_max(aa1[16:32])
                m = _f_max(m0, _f_max(bam2, bam3))
                m = _warp_max_f32(m)
                cute.arch.barrier()
                if lane == Int32(0):
                    sRed[warp] = m
                cute.arch.barrier()
                gmax = _tree_max([sRed[w] for w in range(NWARPS)])
                outer_new = _f_div(gmax, Float32(2688.0))
                t6 = _f_mul(Float32(6.0), _sel_eq0(outer_new, Float32(1.0), outer_new))
                inv_t6 = _f_div(Float32(1.0), t6)

                # ---- primary repack ----
                bam = [bam0, bam1, bam2, bam3]
                psc = [_f_min(Float32(448.0), _f_mul(bam[g], inv_t6)) for g in range(4)]
                pscp = _pack_e4m3x4(psc[0], psc[1], psc[2], psc[3])
                _st_global_u32(head_base + PRIMARY_SCALE + scoff, pscp)
                pe = _expand4(pscp, outer_new)

                # ---- primary + residual repack, one 32-value half at a time ----
                for half in cutlass.range_constexpr(2):
                    sn = _tmem_ld32(my_tmem + Uint32(half * 32))
                    pinv = []
                    for pg in cutlass.range_constexpr(2):
                        e = pe[half * 2 + pg]
                        inv = Float32(0.0)
                        if e != Float32(0.0):
                            inv = _f_div(Float32(1.0), e)
                        pinv.append(inv)
                    pwords = []
                    for jj in cutlass.range_constexpr(4):
                        e = pe[half * 2 + jj // 2]
                        nv = [
                            _sel_eq0(
                                e,
                                Float32(0.0),
                                _f_mul(sn[8 * jj + t], pinv[jj // 2]),
                            )
                            for t in range(8)
                        ]
                        pwords.append(_pack_e2m1x8(nv))
                    _st_global_v4(
                        pbase + Int64(half * 16),
                        pwords[0],
                        pwords[1],
                        pwords[2],
                        pwords[3],
                    )

                    rem = []
                    for jj in cutlass.range_constexpr(4):
                        mg = _decode8(pwords[jj])
                        neg_e = _f_neg(pe[half * 2 + jj // 2])
                        for t in cutlass.range_constexpr(8):
                            rem.append(_f_fma(mg[t], neg_e, sn[8 * jj + t]))
                    ra = [_f_abs(x) for x in rem]
                    rb0 = _tree_max(ra[0:16])
                    rb1 = _tree_max(ra[16:32])
                    rs0 = _f_min(Float32(448.0), _f_mul(rb0, inv_t6))
                    rs1 = _f_min(Float32(448.0), _f_mul(rb1, inv_t6))
                    rscp = _pack_e4m3x2(rs0, rs1)
                    _st_global_u16(
                        head_base + RESIDUAL_SCALE + scoff + Int64(half * 2),
                        rscp,
                    )
                    re = _expand4(rscp, outer_new)
                    rinv = []
                    for rg in cutlass.range_constexpr(2):
                        e = re[rg]
                        inv = Float32(0.0)
                        if e != Float32(0.0):
                            inv = _f_div(Float32(1.0), e)
                        rinv.append(inv)
                    rwords = []
                    for jj in cutlass.range_constexpr(4):
                        e = re[jj // 2]
                        nv = [
                            _sel_eq0(
                                e,
                                Float32(0.0),
                                _f_mul(rem[8 * jj + t], rinv[jj // 2]),
                            )
                            for t in range(8)
                        ]
                        rwords.append(_pack_e2m1x8(nv))
                    _st_global_v4(
                        rbase + Int64(half * 16),
                        rwords[0],
                        rwords[1],
                        rwords[2],
                        rwords[3],
                    )

                if tidx == Int32(0):
                    _st_outer_pad(head_base + OUTER_SCALE, outer_new)
        idx = idx + Int32(gdim)

    cute.arch.barrier()
    if warp == 0:
        cute.arch.dealloc_tmem(tmem_ptr, 128, is_two_cta=False)


@cute.jit
def _commit_launch(
    pool_p: Int64,
    slots_p: Int64,
    slotmap_p: Int64,
    hk_p: Int64,
    hu_p: Int64,
    hg_p: Int64,
    lengths_p: Int64,
    oldlen_p: Int64,
    dummy_p: Int64,
    slot_stride: Int64,
    total: Int32,
    num_ctas: Int32,
    stream,
    H: cutlass.Constexpr,
    CAP: cutlass.Constexpr,
    FULL: cutlass.Constexpr,
    SMEM: cutlass.Constexpr,
):
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        OperandMajorMode.MN,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        (128, 128),
    )
    _commit_kernel(
        tiled_mma,
        pool_p,
        slots_p,
        slotmap_p,
        hk_p,
        hu_p,
        hg_p,
        lengths_p,
        oldlen_p,
        dummy_p,
        slot_stride,
        total,
        H,
        CAP,
        FULL,
    ).launch(grid=[num_ctas, 1, 1], block=[NTHREADS, 1, 1], smem=SMEM, stream=stream)


# ---------------------------------------------------------------------------
# host entry
# ---------------------------------------------------------------------------

_CACHE = {}
_SM = None


@torch.no_grad()
def commit(
    pool: torch.Tensor,
    slots: torch.Tensor,
    slot_map: torch.Tensor,
    hk: torch.Tensor,
    hu: torch.Tensor,
    hg: torch.Tensor,
    lengths: torch.Tensor,
    old_lengths: torch.Tensor,
    dummy: torch.Tensor,
    full_only: bool,
) -> None:
    """Commit packed checkpoints and advance/reset counters from a stable [B] snapshot."""
    global _SM
    B = slots.shape[0]
    H = pool.shape[1]
    total = B * H
    if total == 0:
        return
    if _SM is None:
        _SM = torch.cuda.get_device_properties(pool.device).multi_processor_count
    key = (int(H), int(hk.shape[2]), bool(full_only))
    fn = _CACHE.get(key)
    if fn is None:
        smem_bytes = 4096 + 4096 + 512 + 512 + 64 + 64 + 1024
        fn = cute.compile(
            _commit_launch,
            Int64(pool.data_ptr()),
            Int64(slots.data_ptr()),
            Int64(slot_map.data_ptr()),
            Int64(hk.data_ptr()),
            Int64(hu.data_ptr()),
            Int64(hg.data_ptr()),
            Int64(lengths.data_ptr()),
            Int64(old_lengths.data_ptr()),
            Int64(dummy.data_ptr()),
            Int64(pool.stride(0)),
            Int32(total),
            Int32(1),
            cutlass_torch.current_stream(),
            int(H),
            int(hk.shape[2]),
            bool(full_only),
            int(smem_bytes),
        )
        _CACHE[key] = fn
    num_ctas = min(total, _SM * 3)
    fn(
        Int64(pool.data_ptr()),
        Int64(slots.data_ptr()),
        Int64(slot_map.data_ptr()),
        Int64(hk.data_ptr()),
        Int64(hu.data_ptr()),
        Int64(hg.data_ptr()),
        Int64(lengths.data_ptr()),
        Int64(old_lengths.data_ptr()),
        Int64(dummy.data_ptr()),
        Int64(pool.stride(0)),
        Int32(total),
        Int32(num_ctas),
        cutlass_torch.current_stream(),
    )
