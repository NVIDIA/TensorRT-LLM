# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline-PTX operations of the CSA2 decode selection kernels, which import this module (and with
it the CuTe DSL) when they first build their kernel types."""

from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op


def asm(res, operands, text, constraints, *, loc=None, ip=None):
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


@dsl_user_op
def ld_global_v4(addr: Int64, *, loc=None, ip=None):
    out = asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ld_global_u8(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.s32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_s32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.s32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_v4_s32(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.global.v4.s32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_s32(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.s32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_shared_add(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_shared_u32(addr: Int32, value: Uint32, op: str, *, loc=None, ip=None) -> None:
    asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        f"red.shared.{op}.u32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_cluster_v4(addr: Int32, rank: Int32, *, loc=None, ip=None):
    """Load 4 consecutive 32-bit words from CTA ``rank``'s shared memory at ``addr``."""
    out = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(rank).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a;\n\tmapa.shared::cluster.u32 a, $4, $5;\n\t"
        "ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [a];\n\t}",
        "=r,=r,=r,=r,r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def st_cluster_s32(addr: Int32, rank: Int32, value: Int32, *, loc=None, ip=None) -> None:
    """Store to CTA ``rank``'s shared memory at this CTA's offset ``addr``."""
    asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(rank).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 a;\n\tmapa.shared::cluster.u32 a, $0, $1;\n\t"
        "st.shared::cluster.s32 [a], $2;\n\t}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shfl_up(value: Int32, delta: int, *, loc=None, ip=None) -> Int32:
    return Int32(
        asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.up.b32 $0, $1, {delta}, 0, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx(value: Int32, lane: int, *, loc=None, ip=None) -> Int32:
    return Int32(
        asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.idx.b32 $0, $1, {lane}, 31, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def reduce_warp(value: Uint32, op: str, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        asm(
            T.i32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            f"redux.sync.{op}.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def warp_sum(value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "redux.sync.add.s32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ordered(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Order-preserving key of an FP32 value: larger values, larger keys (never 0)."""
    return Uint32(
        asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 b, m;\n\t.reg .pred n;\n\t"
            "mov.b32 b, $1;\n\t"
            "setp.lt.s32 n, b, 0;\n\t"
            "selp.b32 m, 0xffffffff, 0x80000000, n;\n\t"
            "xor.b32 $0, b, m;\n\t}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def count_key(
    key,
    mask,
    prefix,
    hist,
    run_digit,
    run_count,
    low,
    high,
    shift: int,
    track: bool,
    *,
    loc=None,
    ip=None,
):
    """Count a nonzero ``key`` with ``key & mask == prefix`` into its digit's run (adding the
    previous run to ``hist`` on a new digit); ``track`` widens [low, high] to it."""
    out = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            Uint32(key).ir_value(loc=loc, ip=ip),
            Uint32(mask).ir_value(loc=loc, ip=ip),
            Uint32(prefix).ir_value(loc=loc, ip=ip),
            Int32(hist).ir_value(loc=loc, ip=ip),
            Int32(run_digit).ir_value(loc=loc, ip=ip),
            Int32(run_count).ir_value(loc=loc, ip=ip),
            Uint32(low).ir_value(loc=loc, ip=ip),
            Uint32(high).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred m, d, f;\n\t.reg .b32 t, g;\n\t"
        "and.b32 t, $4, $5;\n\t"
        "setp.eq.u32 m, t, $6;\n\t"
        "setp.ne.and.u32 m, $4, 0, m;\n\t"
        f"bfe.u32 g, $4, {shift}, 8;\n\t"
        "setp.ne.and.u32 d, g, $0, m;\n\t"
        "setp.ne.and.u32 f, $1, 0, d;\n\t"
        "@f mad.lo.u32 t, $0, 4, $7;\n\t"
        "@f red.shared.add.u32 [t], $1;\n\t"
        "@d mov.b32 $0, g;\n\t"
        "@d mov.b32 $1, 0;\n\t"
        "@m add.u32 $1, $1, 1;\n\t"
        + ("@m min.u32 $2, $2, $4;\n\t@m max.u32 $3, $3, $4;\n\t" if track else "")
        + "}",
        "=r,=r,=r,=r,r,r,r,r,0,1,2,3",
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), out, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), out, [3], loc=loc, ip=ip)),
    )
