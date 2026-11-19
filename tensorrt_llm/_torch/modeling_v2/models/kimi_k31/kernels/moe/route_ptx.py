# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constants and inline-PTX helpers of the Kimi K3.1 MoE routing kernel (SM100/SM103/SM107).

:mod:`route_tail` builds its top-16 kernels from these helpers. The top-k weights are
the ``sigmoid`` scores renormalized like ``noaux_tc_op``: :func:`_renormalize` evaluates its FP64
expression ``float(double(score) * scale / (double(total) + 1e-20))``, and FP32 division
(:func:`_div_rn_f32`) gives the same bits whenever the scale is 1 and the sum is >= 2^-12
(:func:`_unit_scale_fast_path`).
"""

from __future__ import annotations

import struct

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Kimi-K3 routed experts.
NUM_EXPERTS = 896
#: Experts selected per token.
TOP_K = 16
#: ``1e-20`` as a PTX double literal (the renormalization epsilon of ``noaux_tc_op``).
_EPS_F64 = "0d" + struct.pack(">d", 1e-20).hex().upper()


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


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _unpack(res, ty, cls, n, *, loc=None, ip=None):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _warp_inclusive_sum(value, *, loc=None, ip=None):
    body = "".join(
        f"shfl.sync.up.b32 t|p, x, {off}, 0, 0xffffffff;\n@p add.s32 x, x, t;\n"
        for off in (1, 2, 4, 8, 16)
    )
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "{\n.reg .b32 x, t;\n.reg .pred p;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_shared_or(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _warp_sum_f32(value, *, loc=None, ip=None):
    """Butterfly warp sum, same order as ``cooperative_groups::reduce(plus)``."""
    body = "".join(
        f"shfl.sync.bfly.b32 t, x, {off}, 0x1f, 0xffffffff;\nadd.rn.f32 x, x, t;\n"
        for off in (16, 8, 4, 2, 1)
    )
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            "{\n.reg .f32 x, t;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _renormalize(score, total, scale_bits, *, loc=None, ip=None):
    """``float(double(score) * scale / (double(total) + 1e-20))`` like ``noaux_tc_op``."""
    return Float32(
        _asm(
            T.f32(),
            [
                _fv(score, loc=loc, ip=ip),
                _fv(total, loc=loc, ip=ip),
                _lv(scale_bits, loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f64 a, b, s;\n"
            "mov.b64 s, $3;\n"
            "cvt.f64.f32 a, $1;\n"
            "mul.rn.f64 a, a, s;\n"
            "cvt.f64.f32 b, $2;\n"
            f"add.rn.f64 b, b, {_EPS_F64};\n"
            "div.rn.f64 a, a, b;\n"
            "cvt.rn.f32.f64 $0, a;\n"
            "}",
            "=f,f,f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unit_scale_fast_path(scale_bits, total, *, loc=None, ip=None):
    """1 if ``scale == 1.0`` and ``total >= 2^-12`` (see :func:`_renormalize`)."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(scale_bits, loc=loc, ip=ip), _fv(total, loc=loc, ip=ip)],
            "{\n.reg .pred p, q;\n"
            "setp.eq.u64 p, $1, 0x3FF0000000000000;\n"
            "setp.ge.f32 q, $2, 0f39800000;\n"
            "and.pred p, p, q;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,l,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_rn_f32(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _popc(value, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_iv(value, loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r", loc=loc, ip=ip)
    )
