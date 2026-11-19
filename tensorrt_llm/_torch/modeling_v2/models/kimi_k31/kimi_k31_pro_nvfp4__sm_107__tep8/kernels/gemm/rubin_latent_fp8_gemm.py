# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline PTX helpers of the NVFP4 epilogues (SM100/SM103/SM107).

The gather CTAs of ``kernels/moe/rubin_ull_front.py`` use them: the global loads and stores, the
16-value amax, the E4M3 block scales of a 16-value group under the global scale (with its
reciprocal) and the E2M1 packing (two codes per byte, low nibble first), with exactly the
arithmetic of ``fp4_quantize`` (``quantize_with_block_size``).
"""

from __future__ import annotations

import cutlass
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm_pure(res, operands, text, constraints, *, loc=None, ip=None):
    """Inline PTX without side effects: the compiler may schedule or drop it freely."""
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return cutlass.Float32(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ldg_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _stg_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u16 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rcp_approx_ftz(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm_pure(
            T.f32(), [_fv(x, loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _abs_max_f32x16(vals, *, loc=None, ip=None):
    body = "".join(f"abs.f32 t{i}, ${i + 1};\n" for i in range(16))
    for step in (8, 4, 2, 1):
        body += "".join(f"max.f32 t{i}, t{i}, t{i + step};\n" for i in range(step))
    body += "mov.f32 $0, t0;\n"
    return cutlass.Float32(
        _asm_pure(
            T.f32(),
            [_fv(v, loc=loc, ip=ip) for v in vals],
            "{\n.reg .f32 " + ", ".join(f"t{i}" for i in range(16)) + ";\n" + body + "}",
            "=f," + ",".join(["f"] * 16),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nvfp4_scales_x2(amax0, amax1, gsf, rgsf, rcp6, *, loc=None, ip=None):
    """NVFP4 scales of two 16-element groups, as in ``cvt_warp_fp16_to_fp4`` (E4M3 scale).

    ``sf = e4m3(gsf * (amax * rcp(6)))`` and ``out_scale = amax != 0 ? rcp(f32(sf) * rcp(gsf)) : 0``
    with ``rcp`` = ``rcp.approx.ftz.f32``. Returns the two scale bytes packed in 16 bits (group
    0 in the low byte) and the two output scales.
    """
    res = _asm_pure(
        llvm.StructType.get_literal([T.i32(), T.f32(), T.f32()]),
        [_fv(x, loc=loc, ip=ip) for x in (amax0, amax1, gsf, rgsf, rcp6)],
        "{\n"
        ".reg .f32 s0, s1, f0, f1;\n"
        ".reg .b16 h, l16, h16;\n"
        ".reg .b32 hw;\n"
        ".reg .pred z0, z1;\n"
        "mul.rn.f32 s0, $3, $7;\n"
        "mul.rn.f32 s1, $4, $7;\n"
        "mul.rn.f32 s0, $5, s0;\n"
        "mul.rn.f32 s1, $5, s1;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h, s1, s0;\n"
        "cvt.rn.f16x2.e4m3x2 hw, h;\n"
        "mov.b32 {l16, h16}, hw;\n"
        "cvt.f32.f16 f0, l16;\n"
        "cvt.f32.f16 f1, h16;\n"
        "mul.rn.f32 f0, f0, $6;\n"
        "mul.rn.f32 f1, f1, $6;\n"
        "rcp.approx.ftz.f32 f0, f0;\n"
        "rcp.approx.ftz.f32 f1, f1;\n"
        "setp.eq.f32 z0, $3, 0f00000000;\n"
        "setp.eq.f32 z1, $4, 0f00000000;\n"
        "selp.f32 $1, 0f00000000, f0, z0;\n"
        "selp.f32 $2, 0f00000000, f1, z1;\n"
        "cvt.u32.u16 $0, h;\n"
        "}",
        "=r,=f,=f,f,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    sf = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    o0 = llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)
    o1 = llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)
    return Int32(sf), cutlass.Float32(o0), cutlass.Float32(o1)


@dsl_user_op
def _e2m1x8(vals, scale, *, loc=None, ip=None):
    """Scale eight FP32 values (``mul.rn``) and pack them as E2M1 into 32 bits, element 0 lowest."""
    body = "".join(f"mul.rn.f32 t{i}, ${i + 1}, $9;\n" for i in range(8))
    body += "".join(
        f"cvt.rn.satfinite.e2m1x2.f32 b{j}, t{2 * j + 1}, t{2 * j};\n" for j in range(4)
    )
    body += "mov.b32 $0, {b0, b1, b2, b3};\n"
    return Int32(
        _asm_pure(
            T.i32(),
            [_fv(v, loc=loc, ip=ip) for v in vals] + [_fv(scale, loc=loc, ip=ip)],
            "{\n.reg .f32 t0, t1, t2, t3, t4, t5, t6, t7;\n.reg .b8 b0, b1, b2, b3;\n" + body + "}",
            "=r," + ",".join(["f"] * 9),
            loc=loc,
            ip=ip,
        )
    )


__all__: list[str] = []
