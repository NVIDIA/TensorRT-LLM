# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline-PTX loads and stores on raw shared/global addresses for CuTe DSL kernels.

The MoE routing kernels (``noaux_tc_route_sort``, ``kimi_k3_router_logits``) address memory
through ``Int32`` shared-memory offsets and ``Int64`` global addresses. ``cute.arch`` has no
accessors for those (it provides the synchronization, shuffle, reduction and PDL primitives the
kernels use directly), so the PTX for them lives here, once. Every helper is a
``dsl_user_op``; values are ``Int32`` / ``Int64`` / ``Float32``.
"""

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def inline_asm(res, operands, text, constraints, *, loc=None, ip=None):
    """``llvm.inline_asm`` with side effects (AT&T dialect)."""
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


def i32(value, *, loc=None, ip=None):
    """``value`` as an ``i32`` IR operand."""
    return Int32(value).ir_value(loc=loc, ip=ip)


def i64(value, *, loc=None, ip=None):
    """``value`` as an ``i64`` IR operand."""
    return Int64(value).ir_value(loc=loc, ip=ip)


def f32(value, *, loc=None, ip=None):
    """``value`` as an ``f32`` IR operand."""
    return Float32(value).ir_value(loc=loc, ip=ip)


def unpack(res, ty, cls, n, *, loc=None, ip=None):
    """The ``n`` fields of an inline-asm struct result as ``cls`` values."""
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


# ---------------------------------------------------------------------------
# Global memory (64-bit addresses)
# ---------------------------------------------------------------------------


@dsl_user_op
def st_global_f32(addr, value, *, loc=None, ip=None):
    inline_asm(
        None,
        [i64(addr, loc=loc, ip=ip), f32(value, loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_b32(addr, value, *, loc=None, ip=None):
    inline_asm(
        None,
        [i64(addr, loc=loc, ip=ip), i32(value, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_f32(addr, *, loc=None, ip=None):
    return Float32(
        inline_asm(
            T.f32(),
            [i64(addr, loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_v4_f32(addr, *, loc=None, ip=None):
    res = inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [i64(addr, loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def ld_global_nc_v4_f32(addr, *, loc=None, ip=None):
    """Four FP32 values through the non-coherent (read-only) path."""
    res = inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [i64(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# Shared memory (32-bit offsets)
# ---------------------------------------------------------------------------


@dsl_user_op
def st_shared_b32(addr, value, *, loc=None, ip=None):
    inline_asm(
        None,
        [i32(addr, loc=loc, ip=ip), i32(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_b32(addr, *, loc=None, ip=None):
    return Int32(
        inline_asm(
            T.i32(),
            [i32(addr, loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_f32(addr, *, loc=None, ip=None):
    return Float32(
        inline_asm(
            T.f32(),
            [i32(addr, loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    inline_asm(
        None,
        [i32(addr, loc=loc, ip=ip)] + [f32(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_v4_f32(addr, *, loc=None, ip=None):
    res = inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [i32(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_v4_b32(addr, *, loc=None, ip=None):
    res = inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [i32(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def st_shared_zero_v4(addr, *, loc=None, ip=None):
    """Zero 16 bytes of shared memory."""
    inline_asm(
        None,
        [i32(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 z;\nmov.b32 z, 0;\nst.shared.v4.b32 [$0], {z, z, z, z};\n}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_b64_or_zero(valid, addr, *, loc=None, ip=None):
    """``*addr`` (64 bits) if ``valid != 0`` else 0."""
    return Int64(
        inline_asm(
            T.i64(),
            [i32(valid, loc=loc, ip=ip), i32(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b64 $0, 0;\n@p ld.shared.b64 $0, [$2];\n}",
            "=&l,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_shared_or_b32(addr, value, *, loc=None, ip=None):
    inline_asm(
        None,
        [i32(addr, loc=loc, ip=ip), i32(value, loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_dsmem_b32(addr, rank, value, *, loc=None, ip=None):
    """32-bit store into the shared memory of cluster CTA ``rank`` at its offset ``addr``."""
    inline_asm(
        None,
        [i32(addr, loc=loc, ip=ip), i32(rank, loc=loc, ip=ip), i32(value, loc=loc, ip=ip)],
        "{\n.reg .b32 r;\nmapa.shared::cluster.u32 r, $0, $1;\nst.shared::cluster.b32 [r], $2;\n}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )
