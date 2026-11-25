# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Conv4, native 2FP4 KDA replay, gated output RMSNorm, and window commit.

One CTA per (batch row, head). The CTA that appends the sixteenth history
entry of its head also commits the full window into both checkpoint terms,
so ordinary decode needs a single launch. History lengths are advanced by
the last CTA of each batch row to arrive, which keeps every CTA's length
read ahead of the update without a separate launch.

With PDL, everything except the current step's projections (slot mapping,
history length, checkpoint and history copies, Conv window and constants)
is issued before ``griddepcontrol.wait``; all stores follow it.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir as mlir_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

from ..flashinfer_utils import get_env_enable_pdl
from .cute_dsl_kda_2fp4_commit import (
    OUTER_SCALE,
    PRIMARY_SCALE,
    RESIDUAL_SCALE,
    STATE_BYTES,
    VALUES_BYTES,
    _expand4,
    _f_abs,
    _f_div,
    _f_exp,
    _f_min,
    _f_mul,
    _ld_global_s32,
    _pack_e2m1x8,
    _pack_e4m3x2,
    _pack_e4m3x4,
    _sel_eq0,
    _st_global_u16,
    _st_global_u32,
    _st_global_v4,
    _st_outer_pad,
    _tmem_ld32,
    _tmem_st32,
    _tree_max,
    _warp_max_f32,
)

THREADS = 256
K = 128
N = 16
# Commit operand panels: K-major, no swizzle, 8-row x 16 B core matrices.
PANEL_LBO = 128
PANEL_SBO = 256
PANEL_BYTES = K * N * 2

_compiled = {}


def _ir(x, loc=None, ip=None):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def _rcp_n(x: Float32, *, loc=None, ip=None) -> Float32:
    """1/x, bit-identical to div.full.f32(1, x) for |x| in [2^-126, 2^126] (one MUFU.RCP, no range fixups)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(x), loc, ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_n(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """a/b, bit-identical to div.full.f32 for |b| in [2^-126, 2^126]: RN(a * MUFU.RCP(b))."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip)],
            "{ .reg .f32 r; rcp.approx.ftz.f32 r, $2; mul.f32 $0, $1, r; }",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_v2(addr: Int64, a: Uint32, b: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Uint32(a), loc, ip), _ir(Uint32(b), loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_v4(addr: Int32, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            _ir(Int32(addr), loc, ip),
            _ir(Uint32(a), loc, ip),
            _ir(Uint32(b), loc, ip),
            _ir(Uint32(c), loc, ip),
            _ir(Uint32(d), loc, ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """RN BF16 pair, ``lo`` in the low half (the lower K index)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Float32(lo), loc, ip), _ir(Float32(hi), loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# Batch indices, slot map and history lengths are a few bytes read by every
# layer; keep them in L2 across the weight traffic between KDA layers.
_EVICT_LAST = "createpolicy.fractional.L2::evict_last.b64 p, 1.0;"


@dsl_user_op
def _pol_ef(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _bulk_g(dst, src, mbar, size, ef=None):
    if cutlass.const_expr(False if ef is None else ef):
        prims.cp_async_bulk_shared_cluster_global(dst, src, mbar, size, l2_cache_hint=_pol_ef())
    else:
        prims.cp_async_bulk_shared_cluster_global(dst, src, mbar, size)


def _bulk_g_tp(dst, src, mbar, size):
    _bulk_g(dst, src, mbar, size, False)


def _bulk_g_ck(dst, src, mbar, size):
    _bulk_g(dst, src, mbar, size, False)


def _bulk_t_tp(dst, desc, coords, mbar):
    _bulk_t(dst, desc, coords, mbar, False)


@dsl_user_op
def _pol_en(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_normal.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _hist_pol(length):
    """L2 policy for a decode item's history windows: evict_first unless the row commits this step (the
    window commit kernel re-reads a committing row's history right after the decode)."""
    return cutlass.select_(length == N - 1, _pol_en(), _pol_ef())


def _bulk_g_rp(dst, src, mbar, size, pol):
    prims.cp_async_bulk_shared_cluster_global(dst, src, mbar, size, l2_cache_hint=pol)


def _bulk_g_lef(ef, dst, src, mbar, size):
    _bulk_g(dst, src, mbar, size, ef)


def _bulk_t_lef(ef, dst, desc, coords, mbar):
    _bulk_t(dst, desc, coords, mbar, ef)


def _bulk_t(dst, desc, coords, mbar, ef=None):
    if cutlass.const_expr(False if ef is None else ef):
        prims.cp_async_bulk_tensor_shared_cta_global(
            dst, desc, coords, mbar, l2_cache_hint=_pol_ef()
        )
    else:
        prims.cp_async_bulk_tensor_shared_cta_global(dst, desc, coords, mbar)


@dsl_user_op
def _ld_keep_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "{ .reg .b64 p; " + _EVICT_LAST + " ld.global.L2::cache_hint.b32 $0, [$1], p; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_u16(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "{ .reg .b16 t; ld.global.b16 t, [$1]; cvt.u32.u16 $0, t; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_u16_keep(addr: Int32, *, loc=None, ip=None) -> Uint32:
    """Shared u16 load into a 32-bit register, kept in program order (after mbarrier waits)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int32(addr), loc, ip)],
            "ld.shared.u16 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_v4_keep(addr: Int32, *, loc=None, ip=None):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_ir(Int32(addr), loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ld_keep_u8(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "{ .reg .b64 p; .reg .b16 t; "
            + _EVICT_LAST
            + " ld.global.L2::cache_hint.b8 t, [$1], p; cvt.u32.u16 $0, t; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_keep_s32(addr: Int64, v: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Int32(v), loc, ip)],
        "{ .reg .b64 p; " + _EVICT_LAST + " st.global.L2::cache_hint.b32 [$0], $1, p; }",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _named_bar(bar_id: int, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        f"bar.sync {bar_id}, {128};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _addr(tensor, index):
    return Int64((tensor.iterator + index).toint())


@dsl_user_op
def _grp16_absmax(x: Float32, lane: Int32, *, loc=None, ip=None):
    """(max |x| over this lane's 16-lane half of the warp, max |x| over the warp): two redux.sync
    (exact, like the butterfly of max.f32 it replaces, at a fraction of the latency)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_ir(Float32(x), loc, ip), _ir(Int32(lane), loc, ip)],
        "{ .reg .pred p; .reg .f32 a, l, h, lo, hi;\n"
        " abs.f32 a, $2;\n setp.lt.u32 p, $3, 16;\n"
        " selp.f32 l, a, 0f00000000, p;\n selp.f32 h, 0f00000000, a, p;\n"
        " redux.sync.max.f32 lo, l, -1;\n redux.sync.max.f32 hi, h, -1;\n"
        " selp.f32 $0, lo, hi, p;\n max.f32 $1, lo, hi; }",
        "=f,=f,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), r, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), r, [1], loc=loc, ip=ip)),
    )


@cute.jit
def _warp_sum_squares(value):
    partner = cute.arch.shuffle_sync_bfly(value * value, 16)
    total = cute.fma(value, value, partner)
    for offset in (8, 4, 2, 1):
        total = total + cute.arch.shuffle_sync_bfly(total, offset)
    return total


@cute.jit
def _warp_sum_products(lhs, rhs):
    partner = cute.arch.shuffle_sync_bfly(lhs * rhs, 16)
    total = cute.fma(lhs, rhs, partner)
    for offset in (8, 4, 2, 1):
        total = total + cute.arch.shuffle_sync_bfly(total, offset)
    return total


@cute.jit
def _smem_u32(ptr):
    return Int32(ptr.toint())


@cute.jit
def _panel_offset(row, chunk):
    return (row // 8) * PANEL_SBO + chunk * PANEL_LBO + (row % 8) * 16


@dsl_user_op
def _atom_add_if(pred: Int32, addr: Int64, val: Int32, *, loc=None, ip=None) -> Int32:
    """Predicated relaxed gpu-scope atomic add returning the old value (0 elsewhere)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int32(pred), loc, ip), _ir(Int64(addr), loc, ip), _ir(Int32(val), loc, ip)],
            "{ .reg .pred p; setp.ne.s32 p, $1, 0; mov.b32 $0, 0; @p atom.relaxed.gpu.global.add.u32 $0, [$2], $3; }",
            "=r,r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_as_s32(v: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Float32(v), loc, ip)],
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
def _s32_as_f32(v: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Int32(v), loc, ip)],
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
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE multiply that ptxas never contracts into an FMA."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(a), loc, ip), _ir(Float32(b), loc, ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
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
def _pin64(v: Int64, *, loc=None, ip=None) -> Int64:
    """Opaque copy: forces the operand to be materialized here (e.g. kernel-param loads early)."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [_ir(Int64(v), loc, ip)],
            "mov.b64 $0, $1;",
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_u16_ef(addr: Int64, v: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Uint32(v), loc, ip)],
        "{ .reg .b16 t; .reg .b64 p; createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
        " cvt.u16.u32 t, $1; st.global.L2::cache_hint.b16 [$0], t, p; }",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_bf16_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """BF16 global load returned as its exact FP32 value."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Int64(addr), loc, ip)],
            "{ .reg .b16 t; .reg .b32 u; ld.global.b16 t, [$1]; cvt.u32.u16 u, t;"
            " shl.b32 $0, u, 16; }",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_s64(addr: Int32, v: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int32(addr), loc, ip), _ir(Int64(v), loc, ip)],
        "st.shared.b64 [$0], $1;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bf16x2_lo_hi(x: Uint32, *, loc=None, ip=None):
    """Unpack a BF16 pair: (low half, high half) as FP32."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_ir(Uint32(x), loc, ip)],
        "{ .reg .b32 a, b; shl.b32 a, $2, 16; and.b32 b, $2, -65536; mov.b32 $0, a; mov.b32 $1, b; }",
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
def _scales4_f16(word: Uint32, *, loc=None, ip=None):
    """Four E4M3 scale bytes -> four f16 values (as raw 16-bit registers)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.i16()] * 4),
        [_ir(Uint32(word), loc, ip)],
        "{ .reg .b16 s<2>; .reg .b32 h<2>;\n"
        " mov.b32 {s0, s1}, $4;\n"
        " cvt.rn.f16x2.e4m3x2 h0, s0;\n"
        " cvt.rn.f16x2.e4m3x2 h1, s1;\n"
        " mov.b32 {$0, $1}, h0;\n"
        " mov.b32 {$2, $3}, h1; }",
        "=h,=h,=h,=h,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int16(llvm.extractvalue(T.i16(), r, [i], loc=loc, ip=ip)) for i in range(4)
    )


_SEED8_ASM = """
{
 .reg .b8 pb<4>, rb<4>;
 .reg .b32 ph<4>, rh<4>;
 .reg .f16 pq<8>, rq<8>, sp, sr;
 .reg .f32 t<8>;
 mov.b32 {pb0,pb1,pb2,pb3}, $8;
 mov.b32 {rb0,rb1,rb2,rb3}, $9;
 mov.b16 sp, $10;
 mov.b16 sr, $11;
"""
for _i in range(4):
    _SEED8_ASM += f" cvt.rn.f16x2.e2m1x2 ph{_i}, pb{_i};\n cvt.rn.f16x2.e2m1x2 rh{_i}, rb{_i};\n"
    _SEED8_ASM += f" mov.b32 {{pq{2 * _i},pq{2 * _i + 1}}}, ph{_i};\n mov.b32 {{rq{2 * _i},rq{2 * _i + 1}}}, rh{_i};\n"
for _i in range(8):
    _SEED8_ASM += f" fma.rn.f32.f16 t{_i}, rq{_i}, sr, 0f00000000;\n"
for _i in range(8):
    _SEED8_ASM += f" fma.rn.f32.f16 t{_i}, pq{_i}, sp, t{_i};\n"
for _i in range(8):
    _SEED8_ASM += f" mul.f32 ${_i}, t{_i}, ${12 + _i};\n"
_SEED8_ASM += "}\n"


@dsl_user_op
def _seed8(pw: Uint32, rw: Uint32, sp, sr, od, *, loc=None, ip=None):
    """8 decayed state values: (p*sp + r*sr) * od[t] with exact f16 products and FP32 sums."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [
            _ir(Uint32(pw), loc, ip),
            _ir(Uint32(rw), loc, ip),
            sp.ir_value(loc=loc, ip=ip),
            sr.ir_value(loc=loc, ip=ip),
        ]
        + [_ir(Float32(o), loc, ip) for o in od],
        _SEED8_ASM,
        "=f,=f,=f,=f,=f,=f,=f,=f,r,r,h,h," + ",".join(["f"] * 8),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(8)]


_REM8_ASM = """
{
 .reg .b8 b<4>;
 .reg .b32 h<4>;
 .reg .f16 q<8>, m1;
 mov.b32 {b0,b1,b2,b3}, $8;
 mov.b16 m1, 0xBC00;
"""
for _i in range(4):
    _REM8_ASM += (
        f" cvt.rn.f16x2.e2m1x2 h{_i}, b{_i};\n mov.b32 {{q{2 * _i},q{2 * _i + 1}}}, h{_i};\n"
    )
for _i in range(8):
    _REM8_ASM += f" fma.rn.f32.f16 ${_i}, q{_i}, m1, ${9 + _i};\n"
_REM8_ASM += "}\n"


@dsl_user_op
def _rem8(codes: Uint32, y, *, loc=None, ip=None):
    """Scaled residuals y - e2m1(code) for 8 values (exact f16 code, one FP32 rounding)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [_ir(Uint32(codes), loc, ip)] + [_ir(Float32(v), loc, ip) for v in y],
        _REM8_ASM,
        "=f,=f,=f,=f,=f,=f,=f,=f,r," + ",".join(["f"] * 8),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(8)]


@dsl_user_op
def _fma_bf16pair(u: Uint32, kw: Uint32, acc0: Float32, acc1: Float32, *, loc=None, ip=None):
    """(u.lo * kw.lo + acc0, u.lo * kw.hi + acc1): BF16 products are exact, one FP32 rounding."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            _ir(Uint32(u), loc, ip),
            _ir(Uint32(kw), loc, ip),
            _ir(Float32(acc0), loc, ip),
            _ir(Float32(acc1), loc, ip),
        ],
        "{ .reg .b16 ul, uh, kl, kh;\n mov.b32 {ul, uh}, $2;\n mov.b32 {kl, kh}, $3;\n"
        " fma.rn.f32.bf16 $0, ul, kl, $4;\n fma.rn.f32.bf16 $1, ul, kh, $5; }",
        "=f,=f,r,r,f,f",
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
def _tmem_st32_nw(addr: Uint32, vals, *, loc=None, ip=None):
    """tcgen05.st of 32 columns without the trailing wait (one _tmem_wait_st after the last store)."""
    regs = ", ".join(f"${i}" for i in range(32))
    llvm.inline_asm(
        None,
        [_ir(Float32(v), loc, ip) for v in vals] + [_ir(Uint32(addr), loc, ip)],
        f"tcgen05.st.sync.aligned.32x32b.x32.b32 [$32], {{{regs}}};",
        ",".join(["f"] * 32) + ",r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tmem_wait_st(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "tcgen05.wait::st.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tmem_ld64(addr: Uint32, *, loc=None, ip=None):
    """64 consecutive TMEM columns of this thread's lane: one load, one wait."""
    regs = ", ".join(f"${i}" for i in range(64))
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 64),
        [_ir(Uint32(addr), loc, ip)],
        f"tcgen05.ld.sync.aligned.32x32b.x64.b32 {{{regs}}}, [$64];\ntcgen05.wait::ld.sync.aligned;",
        ",".join(["=f"] * 64) + ",r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(64)]


@cute.jit
def _seed_commit(
    v,
    lane_addr,
    mma_a,
    res_vals,
    res_scales,
    mma_sfa,
    s_od,
):
    """TMEM row v <- dequant(primary + residual) * outer * exp(G_end): the decayed old state."""
    swizzle_s64 = cutlass.Swizzle.from_name("s64b")
    psw = [
        _ld_shared_u32(_smem_u32(mma_sfa.data_ptr()) + kb * 512 + (v % 32) * 16 + (v // 32) * 4)
        for kb in range(2)
    ]
    rsw = [_ld_shared_u32(_smem_u32(res_scales.data_ptr()) + v * 8 + 4 * j) for j in range(2)]
    sp = list(_scales4_f16(psw[0])) + list(_scales4_f16(psw[1]))
    sr = list(_scales4_f16(rsw[0])) + list(_scales4_f16(rsw[1]))
    for c in cutlass.range_constexpr(4):
        pw = (
            mma_a.subview(v * 64 + c * 16)
            .data_ptr()
            .apply_swizzle(swizzle_s64)
            .load(count=16, alignment=16)
            .bitcast(Uint32)
        )
        rw = (
            (res_vals.subview(v * 64 + c * 16).data_ptr())
            .load(count=16, alignment=16)
            .bitcast(Uint32)
        )
        od = [(s_od.data_ptr() + 32 * c + 4 * q).load(count=4, alignment=16) for q in range(8)]
        seed = []
        for jj in cutlass.range_constexpr(4):
            g = 2 * c + jj // 2
            seed = seed + _seed8(
                pw[jj], rw[jj], sp[g], sr[g], [od[2 * jj + t // 4][t % 4] for t in range(8)]
            )
        _tmem_st32_nw(lane_addr + Uint32(32 * c), seed)
    _tmem_wait_st()


@cute.jit
def _quant_pass2(sn, pe, inv_t6, outer_new, ng: cutlass.Constexpr = 4):
    """Both FP4 terms of four 16-key groups (sn: 64 values, pe: the groups' primary scales x outer),
    written phase by phase across the groups so their dependency chains overlap.
    Returns (8 primary words, 8 residual words, residual scale word)."""
    pinv = [_sel_eq0(pe[g], Float32(0.0), _rcp_n(pe[g])) for g in range(ng)]
    # Scaled values y = x / e (an all-zero group has e == 0 and x == 0, so y = 0 as well).
    y = [[_f_mul(sn[16 * g + t], pinv[g]) for t in range(16)] for g in range(ng)]
    pw = [[_pack_e2m1x8(y[g][8 * h2 : 8 * h2 + 8]) for h2 in range(2)] for g in range(ng)]
    rem = [_rem8(pw[g][0], y[g][0:8]) + _rem8(pw[g][1], y[g][8:16]) for g in range(ng)]
    rs = [
        _f_min(
            Float32(448.0), _f_mul(_f_mul(_tree_max([_f_abs(r) for r in rem[g]]), pe[g]), inv_t6)
        )
        for g in range(ng)
    ]
    # (each scale rounds on its own: packing two or four per conversion gives the same bytes)
    if cutlass.const_expr(ng == 4):
        rword = _pack_e4m3x4(rs[0], rs[1], rs[2], rs[3])
    else:
        rword = _pack_e4m3x2(rs[0], rs[1])
    re = list(_expand4(rword, outer_new))
    rmul = [_sel_eq0(re[g], Float32(0.0), _f_mul(pe[g], _rcp_n(re[g]))) for g in range(ng)]
    rw = [
        [_pack_e2m1x8([_f_mul(rem[g][8 * h2 + t], rmul[g]) for t in range(8)]) for h2 in range(2)]
        for g in range(ng)
    ]
    pw_all = [pw[g][h2] for g in range(ng) for h2 in range(2)]
    rw_all = [rw[g][h2] for g in range(ng) for h2 in range(2)]
    return pw_all, rw_all, rword


@cute.jit
def _commit_quantize(v, h, lane, warp, tidx, lane_addr, head_base, s_u15, s_k, s_cmax, commit_mbar):
    """Quantize half h (keys 64h..64h+63) of TMEM row v into both checkpoint terms.
    The current token's rank-1 update u15 (x) kn is added here (exact BF16 products)."""
    while not prims.mbarrier_try_wait_parity(commit_mbar, Int32(0)):
        pass
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    u15w = _ld_shared_u32(_smem_u32(s_u15.data_ptr()) + v * 4)
    sn = []
    for c in cutlass.range_constexpr(2):
        # (two 32-column loads: a 64-register load does not fit the 80-register variant)
        acc = _tmem_ld32(lane_addr + Uint32(64 * h + 32 * c))
        knw = [
            (s_k.data_ptr() + 64 * h + 32 * c + 8 * q).load(count=8, alignment=16).bitcast(Uint32)
            for q in range(4)
        ]
        for q in cutlass.range_constexpr(4):
            for p in cutlass.range_constexpr(4):
                x0, x1 = _fma_bf16pair(u15w, knw[q][p], acc[8 * q + 2 * p], acc[8 * q + 2 * p + 1])
                sn.append(x0)
                sn.append(x1)
    bam = [_tree_max([_f_abs(sn[16 * g + t]) for t in range(16)]) for g in range(4)]
    m = _warp_max_f32(_tree_max(bam))
    if lane == 0:
        s_cmax[warp] = m
    cute.arch.barrier()
    gmax = _tree_max([s_cmax[w] for w in range(8)])
    outer_new = _f_div(gmax, Float32(2688.0))
    t6 = _f_mul(Float32(6.0), _sel_eq0(outer_new, Float32(1.0), outer_new))
    inv_t6 = _rcp_n(t6)
    psc = [_f_min(Float32(448.0), _f_mul(bam[g], inv_t6)) for g in range(4)]
    pscp = _pack_e4m3x4(psc[0], psc[1], psc[2], psc[3])
    row_base = head_base + Int64(v) * 64
    _st_global_u32(head_base + PRIMARY_SCALE + Int64(v) * 8 + 4 * h, pscp)
    pe = list(_expand4(pscp, outer_new))
    pw_all, rw_all, rword = _quant_pass2(sn, pe, inv_t6, outer_new)
    # The half-row's 32 value bytes of each term as two 16-byte stores.
    for q in cutlass.range_constexpr(2):
        _st_global_v4(
            row_base + 32 * h + 16 * q,
            pw_all[4 * q],
            pw_all[4 * q + 1],
            pw_all[4 * q + 2],
            pw_all[4 * q + 3],
        )
        _st_global_v4(
            row_base + VALUES_BYTES + 32 * h + 16 * q,
            rw_all[4 * q],
            rw_all[4 * q + 1],
            rw_all[4 * q + 2],
            rw_all[4 * q + 3],
        )
    _st_global_u32(head_base + RESIDUAL_SCALE + Int64(v) * 8 + 4 * h, rword)
    if tidx == 0:
        _st_outer_pad(head_base + OUTER_SCALE, outer_new)


@dsl_user_op
def _ld_shared_u32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int32(addr), loc, ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_v2(addr: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int32(addr), loc, ip), _ir(Uint32(a), loc, ip), _ir(Uint32(b), loc, ip)],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_proxy_async_smem(*, loc=None, ip=None):
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
def _st_shared_u32(addr: Int32, v: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int32(addr), loc, ip), _ir(Uint32(v), loc, ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _lat_decode_mma(
    mma_a, mma_b, mma_sfa, mma_sfb, acc_addr, sfa_addr, sfb_addr, state_mbar, acc_mbar
):
    """Decode MMA (checkpoint x dynamic K/Q, MXF4NVF4 block16): SF copies, checkpoint wait, 2 MMAs, commit."""
    idesc = prims.Tcgen05MxOmmaInstrDesc.build(
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float4E2M1FN,
        scale_format=0,
        n_dim=8,
        m_dim=128,
    )
    desc_a = prims.Tcgen05SmemDesc.build(
        mma_a,
        leading_byte_offset=0,
        stride_byte_offset=512,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
    )
    desc_b = prims.Tcgen05SmemDesc.build(
        mma_b,
        leading_byte_offset=0,
        stride_byte_offset=512,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
    )
    desc_sfa = prims.Tcgen05SmemDesc.build(
        mma_sfa, leading_byte_offset=16, stride_byte_offset=128, layout=0
    )
    desc_sfb = prims.Tcgen05SmemDesc.build(
        mma_sfb, leading_byte_offset=16, stride_byte_offset=128, layout=0
    )
    acc_ptr = cutlass.inttoptr(acc_addr, 6, cutlass.Float32)
    sf_shape, sf_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
    if prims.elect_sync():
        for kblock in cutlass.range_constexpr(2):
            sfa_ptr = cutlass.inttoptr(sfa_addr + kblock * 4, 6, cutlass.Int32)
            sfb_ptr = cutlass.inttoptr(sfb_addr + kblock * 4, 6, cutlass.Int32)
            prims.tcgen05_cp(
                sf_shape,
                sfa_ptr,
                desc_sfa.advance_start_address(kblock * 512),
                multicast=sf_multicast,
            )
            prims.tcgen05_cp(
                sf_shape,
                sfb_ptr,
                desc_sfb.advance_start_address(kblock * 512),
                multicast=sf_multicast,
            )
        while not prims.mbarrier_try_wait_parity(state_mbar, cutlass.Int32(0)):
            pass
        for kblock in cutlass.range_constexpr(2):
            prims.tcgen05_mma_block_scale(
                prims.MMABlockScaleKind.MXF4NVF4,
                prims.CTAGroup.CTA_1,
                acc_ptr,
                desc_a.advance_start_address(kblock * 32),
                desc_b.advance_start_address(kblock * 32),
                idesc,
                enable_input_d=(kblock != 0),
                scale_a=cutlass.inttoptr(sfa_addr + kblock * 4, 6, cutlass.Int32),
                scale_b=cutlass.inttoptr(sfb_addr + kblock * 4, 6, cutlass.Int32),
                scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
            )
        prims.tcgen05_commit(acc_mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def _hist_window(cp_base, u_base, length, rows: cutlass.Constexpr, hist_k, hist_q):
    """Sequential history sum over rows 0..rows-1 (rows >= length add exact zeros); shared loads up front."""
    cps = []
    for i in cutlass.range_constexpr((rows + 3) // 4):
        cps = cps + _lds_v4_keep(cp_base + 16 * i)
    uws = [_lds_u16_keep(u_base + j * K * 2) for j in range(rows)]
    for j in cutlass.range_constexpr(rows):
        valid = cutlass.Int32(j) < length
        uj = cutlass.select_(valid, uws[j], Uint32(0))
        cj = cutlass.select_(valid, cps[j], Uint32(0))
        hist_k, hist_q = _hist_fma(uj, cj, hist_k, hist_q)
    return hist_k, hist_q


@cute.kernel
def _decode_kernel(
    tma_state: cutlass.GridConstant[cuda_exp.TensorMap],
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    replay_indices,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    arrivals,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    fast_silu: cutlass.Constexpr,
    tmem128: cutlass.Constexpr,
):
    """Warps 0-3 (Q group, thread t = key/value t) run the decode chain; warps 4-7 (C group)
    load the K/G history and build the history coefficients. Groups meet at three CTA barriers."""
    s_sink = cutlass.Array(cutlass.Int64, 32, space=cutlass.AddressSpace.smem, alignment=16)
    tidx_cta, _, _ = cute.arch.thread_idx()
    # L2 evict_first for this kernel's bulk copies with three CTAs per SM: the replay's working set then exceeds
    # L2, and these lines would otherwise push out the neighbouring kernels' data.
    lat_ef = tmem128
    block, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp_cta = cute.arch.warp_idx()
    tidx = tidx_cta
    warp = warp_cta
    item = block
    item_ok = cutlass.Int32(1)
    batch = item // heads
    head = item - batch * heads

    acc_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    state_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    u_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    conv_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    commit_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    accread_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    res_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    # Initialized by the thread that issues the first TMAs (warp 4, lane 0): every other use of
    # these barriers comes after a CTA barrier.
    if tidx == 128:
        prims.mbarrier_init(acc_mbar, 1)
        prims.mbarrier_init(state_mbar, 1)
        prims.mbarrier_init(u_mbar, 1)
        prims.mbarrier_init(conv_mbar, 1)
        prims.mbarrier_init(commit_mbar, 1)
        prims.mbarrier_init(res_mbar, 1)
        prims.mbarrier_init(accread_mbar, 128)
        prims.fence_mbarrier_init()

    # Warm the constant cache with every kernel-parameter line used later (a cold LDC costs ~0.25 us
    # on the dependent load chain): one warp folds the pointers/strides into a dummy shared store.
    if warp == 2:
        warm = (
            Int64(lengths.iterator.toint())
            ^ Int64(pool.iterator.toint())
            ^ Int64(conv_state.iterator.toint())
            ^ Int64(x.iterator.toint())
            ^ Int64(gate.iterator.toint())
            ^ Int64(z.iterator.toint())
            ^ Int64(beta.iterator.toint())
            ^ Int64(hk.iterator.toint())
            ^ Int64(hu.iterator.toint())
            ^ Int64(hg.iterator.toint())
            ^ Int64(out.iterator.toint())
            ^ Int64(arrivals.iterator.toint())
            ^ Int64(cute.crd2idx((1, 0), x.layout))
            ^ Int64(cute.crd2idx((1, 0), gate.layout))
            ^ Int64(cute.crd2idx((1, 0, 0), z.layout))
            ^ Int64(cute.crd2idx((1, 0), beta.layout))
            ^ Int64(cute.crd2idx((1, 0, 0), conv_state.layout))
            ^ Int64(cute.crd2idx((1, 0, 0), pool.layout))
        )
        if lane == 0:
            _st_shared_s64(_smem_u32(s_sink.data_ptr()) + 240, warm)
    # Layer constants first (independent of every other load), then the row metadata.
    # (Q group only: the C group needs just the row metadata.)
    t_q = tidx % K
    width_c = heads * K
    weight_raw = conv_weight.iterator.raw_ptr()
    wq = (weight_raw + (head * K + t_q) * 4).load(count=4, alignment=8).bitcast(Uint32)
    wk = (weight_raw + (width_c + head * K + t_q) * 4).load(count=4, alignment=8).bitcast(Uint32)
    wv = (
        (weight_raw + (2 * width_c + head * K + t_q) * 4).load(count=4, alignment=8).bitcast(Uint32)
    )
    norm_bf16 = norm_weight[t_q]
    dt_value = dt_bias[head * K + t_q]
    a_log_value = a_log[head]
    slot = _ld_keep_s32(_addr(slots, batch))
    replay32 = _ld_keep_s32(_addr(replay_indices, batch))
    is_dummy = _ld_keep_u8(_addr(dummy, batch))
    # Projection addresses (kernel params resolved now, while the row metadata is in flight;
    # the loads themselves follow the PDL wait).
    xq_addr = _pin64(Int64((x.iterator + cute.crd2idx((batch, head * K + t_q), x.layout)).toint()))
    xk_addr = _pin64(
        Int64((x.iterator + cute.crd2idx((batch, width_c + head * K + t_q), x.layout)).toint())
    )
    xv_addr = _pin64(
        Int64((x.iterator + cute.crd2idx((batch, 2 * width_c + head * K + t_q), x.layout)).toint())
    )
    gate_addr = _pin64(
        Int64((gate.iterator + cute.crd2idx((batch, head * K + t_q), gate.layout)).toint())
    )
    z_addr = _pin64(Int64((z.iterator + cute.crd2idx((batch, head, t_q), z.layout)).toint()))
    beta_addr = _pin64(Int64((beta.iterator + cute.crd2idx((batch, head), beta.layout)).toint()))
    cs_base = _pin64(Int64(conv_state.iterator.toint()))
    cs_stride = _pin64(Int64(cute.crd2idx((1, 0, 0), conv_state.layout)))

    smem = utils.SmemAllocator()
    s_q = cutlass.Array(cutlass.BFloat16, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_k = cutlass.Array(cutlass.BFloat16, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_estep = cutlass.Array(cutlass.Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16,)), byte_alignment=16)
    s_cpair = cutlass.Array(cutlass.Int32, N, space=cutlass.AddressSpace.smem, alignment=16)
    mma_a = cutlass.Array(cutlass.Int8, 8192, space=cutlass.AddressSpace.smem, alignment=128)
    mma_b = cutlass.Array(cutlass.Int8, 512, space=cutlass.AddressSpace.smem, alignment=128)
    mma_sfa = cutlass.Array(
        cutlass.Float8E4M3FN, 1024, space=cutlass.AddressSpace.smem, alignment=128
    )
    mma_sfb = cutlass.Array(
        cutlass.Float8E4M3FN, 1024, space=cutlass.AddressSpace.smem, alignment=128
    )
    tmem_holding = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    s_meta = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    cache_hu = cutlass.Array(
        cutlass.BFloat16, N * K, space=cutlass.AddressSpace.smem, alignment=128
    )
    cache_hg = cutlass.Array(cutlass.Float32, N * K, space=cutlass.AddressSpace.smem, alignment=128)
    if cutlass.const_expr(tmem128):
        s_hk = cutlass.Array(
            cutlass.BFloat16, N * K, space=cutlass.AddressSpace.smem, alignment=128
        )
    s_decay = cutlass.Array(cutlass.Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_u15 = cutlass.Array(cutlass.Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_od = cutlass.Array(cutlass.Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_cmax = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    res_vals = cutlass.Array(cutlass.Int8, 8192, space=cutlass.AddressSpace.smem, alignment=128)
    res_scales = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=128)
    # Commit operand panels reuse the G rows once the previous-G lookups are done (after B2).
    panel_a = cutlass.Array(cache_hg.data_ptr(), shape=PANEL_BYTES // 2, dtype=cutlass.BFloat16)
    panel_b = cutlass.Array(
        cache_hg.subview(PANEL_BYTES // 4).data_ptr(),
        shape=PANEL_BYTES // 2,
        dtype=cutlass.BFloat16,
    )

    if is_dummy != cutlass.Int32(0):
        if warp < 4:
            if cutlass.const_expr(pdl):
                prims.griddepcontrol("wait")
            if item_ok != 0:
                out[(batch, head, tidx)] = cutlass.BFloat16(0.0)
    else:
        replay_slot = cutlass.Int64(replay32)
        pool_off = cute.crd2idx((slot, head, 0), pool.layout)
        pool_base = pool.iterator + pool_off
        width = heads * K
        if warp == 4:
            if lane == 0:
                # The checkpoint and U (needed last) by the C group; the Q group goes straight
                # to its Conv-window loads and the PDL wait.
                prims.mbarrier_arrive_expect_tx(state_mbar, tma_state.global_tx_bytes())
                _bulk_t_lef(
                    lat_ef,
                    mma_a,
                    tma_state.get_ptr(),
                    (cutlass.Int32(0), cutlass.Int32(0), head, slot),
                    state_mbar,
                )
                prims.mbarrier_arrive_expect_tx(u_mbar, N * K * 2)
                _bulk_g_lef(
                    lat_ef,
                    cache_hu,
                    hu.iterator.raw_ptr() + cute.crd2idx((replay_slot, head, 0, 0), hu.layout),
                    u_mbar,
                    N * K * 2,
                )
        if warp == 5:
            # tmem128 (three CTAs per SM): the commit accumulator reuses the decode columns, so
            # 3 x 128 columns fit an SM and no CTA waits in tcgen05.alloc for another to exit.
            prims.tcgen05_alloc(tmem_holding, cutlass.Int32(128 if tmem128 else 256), group="cta_1")
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")

        if warp < 4:
            # ============================== Q group ==============================
            q_channel = head * K + tidx
            k_channel = width + head * K + tidx
            v_channel = 2 * width + head * K + tidx
            # The PDL wait, then the projections (the critical loads).
            if cutlass.const_expr(pdl):
                prims.griddepcontrol("wait")
            raw_qf = _ld_bf16_f32(xq_addr)
            raw_kf = _ld_bf16_f32(xk_addr)
            raw_vf = _ld_bf16_f32(xv_addr)
            gate_f = _ld_bf16_f32(gate_addr)
            z_f = _ld_bf16_f32(z_addr)
            beta_f = _ld_bf16_f32(beta_addr)
            row_cs = cs_base + Int64(slot) * cs_stride * 2
            q_state_a = row_cs + Int64(q_channel * 6)
            k_state_a = row_cs + Int64(k_channel * 6)
            v_state_a = row_cs + Int64(v_channel * 6)
            q_state = [_ld_global_u16(q_state_a + 2 * j) for j in range(3)]
            k_state = [_ld_global_u16(k_state_a + 2 * j) for j in range(3)]
            v_state = [_ld_global_u16(v_state_a + 2 * j) for j in range(3)]
            is_t0 = cutlass.Int32(0)
            if tidx == 0:
                is_t0 = cutlass.Int32(1)
            state_scale_words = (
                cute.recast_ptr(pool_base + PRIMARY_SCALE, dtype=cutlass.Int32).raw_ptr() + tidx * 2
            ).load(count=2, alignment=8)
            sfb_words = cutlass.inttoptr(
                mma_sfb.subview(tidx * 8).data_ptr().toint(),
                cutlass.AddressSpace.smem,
                cutlass.Int32,
            )
            sfb_words.store(
                cutlass.Vector.from_elements(
                    (cutlass.Int32(0), cutlass.Int32(0)), dtype=cutlass.Int32
                ),
                alignment=8,
            )
            if tidx < 24:
                zero = cutlass.Int32(0)
                cutlass.inttoptr(
                    mma_b.subview(128 + tidx * 16).data_ptr().toint(),
                    cutlass.AddressSpace.smem,
                    cutlass.Int32,
                ).store(
                    cutlass.Vector.from_elements((zero, zero, zero, zero), dtype=cutlass.Int32),
                    alignment=16,
                )

            # Gate step: needs only this step's projection and the layer constants.
            a = cute.exp(a_log_value, fastmath=True)
            shifted = gate_f + dt_value
            # cumulative = fma(step_scale, step_value, prev_g), the baseline's contraction.
            step_scale = -a
            step_value = shifted
            if bounded:
                step_scale = lower_bound
                step_value = cute.div(
                    cutlass.Float32(1.0),
                    cutlass.Float32(1.0) + cute.exp(-(a * shifted), fastmath=True),
                    full=True,
                )
            else:
                step_value = cutlass.select_(
                    shifted <= cutlass.Float32(20.0),
                    cute.log(
                        cutlass.Float32(1.0) + cute.exp(shifted, fastmath=True), fastmath=True
                    ),
                    shifted,
                )
            step = _mul_rn(step_scale, step_value)
            s_estep[tidx] = cute.exp(step, fastmath=True)
            zg = _rcp_approx(cutlass.Float32(1.0) + cute.exp(-z_f, fastmath=True))
            beta_sigmoid = _rcp_approx(cutlass.Float32(1.0) + cute.exp(-beta_f, fastmath=True))
            q1, q2 = q_state[1], q_state[2]
            k1, k2 = k_state[1], k_state[2]
            v1, v2 = v_state[1], v_state[2]
            raw_q = Uint32(_f32_as_s32(raw_qf)) >> Uint32(16)
            raw_k = Uint32(_f32_as_s32(raw_kf)) >> Uint32(16)
            raw_v = Uint32(_f32_as_s32(raw_vf)) >> Uint32(16)
            aq = _conv4_bf16(wq[0], wq[1], q_state[0], q1, q2, raw_q)
            ak = _conv4_bf16(wk[0], wk[1], k_state[0], k1, k2, raw_k)
            av = _conv4_bf16(wv[0], wv[1], v_state[0], v1, v2, raw_v)
            if cutlass.const_expr(fast_silu):
                aq = aq * _rcp_approx(cutlass.Float32(1.0) + cute.exp(-aq, fastmath=True))
                ak = ak * _rcp_approx(cutlass.Float32(1.0) + cute.exp(-ak, fastmath=True))
                av = av * _rcp_approx(cutlass.Float32(1.0) + cute.exp(-av, fastmath=True))
            else:
                aq = aq * cute.rcp(cutlass.Float32(1.0) + cute.exp(-aq, fastmath=True))
                ak = ak * cute.rcp(cutlass.Float32(1.0) + cute.exp(-ak, fastmath=True))
                av = av * cute.rcp(cutlass.Float32(1.0) + cute.exp(-av, fastmath=True))
            qf = aq.to(cutlass.BFloat16).to(cutlass.Float32)
            kf = ak.to(cutlass.BFloat16).to(cutlass.Float32)
            v_bf16 = av.to(cutlass.BFloat16)
            q_partial = _warp_sum_squares(qf)
            k_partial = _warp_sum_squares(kf)
            if lane == 0:
                s_red[8 + warp] = q_partial
                s_red[12 + warp] = k_partial
            cute.arch.barrier()  # B1

            length = s_meta[0]
            state_outer = _s32_as_f32(s_meta[1])
            # Decode accumulator / scale columns follow the commit's 128 columns.
            dcol = cutlass.Int32(0 if tmem128 else 128)
            # TMEM base (allocated before B1) and the decode MMA's column addresses, off the B3 path.
            tmem_raw = tmem_holding.load()
            base_col = tmem_raw & cutlass.Int32(0xFFFF)
            base_row = tmem_raw >> cutlass.Int32(16)
            acc_addr = (base_row << cutlass.Int32(16)) | (base_col + dcol)
            sfa_addr = (base_row << cutlass.Int32(16)) | (base_col + dcol + cutlass.Int32(8))
            sfb_addr = (base_row << cutlass.Int32(16)) | (base_col + dcol + cutlass.Int32(16))
            qr = (s_red[8] + s_red[10]) + (s_red[9] + s_red[11])
            kr = (s_red[12] + s_red[14]) + (s_red[13] + s_red[15])
            qden = cute.sqrt(qr + cutlass.Float32(1.0e-6), approx=True, ftz=True)
            kden = cute.sqrt(kr + cutlass.Float32(1.0e-6), approx=True, ftz=True)
            qn = _div_n(qf * cutlass.Float32(0.08838834764831845), qden).to(cutlass.BFloat16)
            kn = _div_n(kf, kden).to(cutlass.BFloat16)
            s_q[tidx] = qn
            s_k[tidx] = kn
            prev_g = cutlass.Float32(0.0)
            if cutlass.const_expr(tmem128):
                while not prims.mbarrier_try_wait_parity(conv_mbar, cutlass.Int32(0)):
                    pass
            if length > 0:
                prev_g = cache_hg[(length - 1) * K + tidx]
            cumulative = _fma_rn(step_scale, step_value, prev_g)
            decay = cute.exp(cumulative, fastmath=True)
            knf = kn.to(cutlass.Float32)
            qnf = qn.to(cutlass.Float32)
            dyn_k = knf * decay
            dyn_q = qnf * decay
            s_decay[tidx] = decay
            s_od[tidx] = decay * state_outer
            group_max_k = cute.abs(dyn_k)
            group_max_q = cute.abs(dyn_q)
            for offset in (8, 4, 2, 1):
                group_max_k = cute.arch.fmax(
                    group_max_k, cute.arch.shuffle_sync_bfly(group_max_k, offset)
                )
                group_max_q = cute.arch.fmax(
                    group_max_q, cute.arch.shuffle_sync_bfly(group_max_q, offset)
                )
            wmax = cute.arch.fmax(group_max_k, group_max_q)
            wmax = cute.arch.fmax(wmax, cute.arch.shuffle_sync_bfly(wmax, 16))
            wdot = _warp_sum_products(knf, qnf)
            if lane == 0:
                s_red[warp] = wmax
                s_red[4 + warp] = wdot
            cute.arch.barrier()  # B2

            mr = cute.arch.fmax(
                cute.arch.fmax(s_red[0], s_red[1]), cute.arch.fmax(s_red[2], s_red[3])
            )
            kq_dot = (s_red[4] + s_red[6]) + (s_red[5] + s_red[7])
            outer = mr * cutlass.Float32(1.0 / (6.0 * 448.0))
            safe_outer = _sel_eq0(outer, cutlass.Float32(1.0), outer)
            sfk_raw = _div_n(group_max_k, cutlass.Float32(6.0) * safe_outer)
            sfq_raw = _div_n(group_max_q, cutlass.Float32(6.0) * safe_outer)
            sfk_e4 = cute.arch.fmin(sfk_raw, cutlass.Float32(448.0)).to(cutlass.Float8E4M3FN)
            sfq_e4 = cute.arch.fmin(sfq_raw, cutlass.Float32(448.0)).to(cutlass.Float8E4M3FN)
            if (lane % 16) == 0:
                group = warp * 2 + lane // 16
                sf_offset_k = (group // 4) * 512 + (group % 4)
                mma_sfb.subview(sf_offset_k).store(sfk_e4)
                mma_sfb.subview(sf_offset_k + 16).store(sfq_e4)
            sfk = sfk_e4.to(cutlass.Float32)
            sfq = sfq_e4.to(cutlass.Float32)
            denom_k = sfk * outer
            denom_q = sfq * outer
            normalized_k = _sel_eq0(denom_k, cutlass.Float32(0.0), _div_n(dyn_k, denom_k))
            normalized_q = _sel_eq0(denom_q, cutlass.Float32(0.0), _div_n(dyn_q, denom_q))
            swizzle_s64 = cutlass.Swizzle.from_name("s64b")
            e2m1_type = mlir_ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type)
            odd_k = cute.arch.shuffle_sync_down(normalized_k, 1)
            odd_q = cute.arch.shuffle_sync_down(normalized_q, 1)
            if (lane % 2) == 0:
                packed_k_ir = prims.dialect.convert_f32x2_to_f4x2(odd_k, normalized_k, e2m1_type)
                packed_q_ir = prims.dialect.convert_f32x2_to_f4x2(odd_q, normalized_q, e2m1_type)
                byte = tidx // 2
                (mma_b.subview(byte)).data_ptr().apply_swizzle(swizzle_s64).store(
                    cutlass.Int8(packed_k_ir)
                )
                (mma_b.subview(64 + byte)).data_ptr().apply_swizzle(swizzle_s64).store(
                    cutlass.Int8(packed_q_ir)
                )
            for kblock in cutlass.range_constexpr(2):
                sf_offset = kblock * 512 + (tidx % 32) * 16 + (tidx // 32) * 4
                cutlass.inttoptr(
                    mma_sfa.subview(sf_offset).data_ptr().toint(),
                    cutlass.AddressSpace.smem,
                    cutlass.Int32,
                ).store(
                    cutlass.Vector.from_elements((state_scale_words[kblock],), dtype=cutlass.Int32)
                )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()  # B3

            mma_q = cutlass.Int32(1)
            if warp == 0:
                if mma_q != 0:
                    _lat_decode_mma(
                        mma_a,
                        mma_b,
                        mma_sfa,
                        mma_sfb,
                        acc_addr,
                        sfa_addr,
                        sfb_addr,
                        state_mbar,
                        acc_mbar,
                    )
            # Global stores only after the async-proxy fence (its MEMBAR would wait for them).
            for sa, w1_, w2_, w3_ in (
                (q_state_a, q1, q2, raw_q),
                (k_state_a, k1, k2, raw_k),
                (v_state_a, v1, v2, raw_v),
            ):
                _st_global_u16(sa, w1_)
                _st_global_u16(sa + 2, w2_)
                _st_global_u16(sa + 4, w3_)
            hk[(replay_slot, head, length, tidx)] = kn
            hg[(replay_slot, head, length, tidx)] = cumulative
            arrived = cutlass.Int32(0)
            if cutlass.const_expr(not fast_silu):
                # Every length read of this CTA (C group, before B1) precedes the arrival.
                is_w = is_t0
                arrived = _atom_add_if(is_w, _addr(arrivals, batch), cutlass.Int32(1))

            hist_k = cutlass.Float32(0.0)
            hist_q = cutlass.Float32(0.0)
            if length > 0:
                while not prims.mbarrier_try_wait_parity(u_mbar, cutlass.Int32(0)):
                    pass
                if cutlass.const_expr(_HIST_UNROLL > 0):
                    # Fully unrolled window: every shared load up front, rows >= length contribute
                    # u = c = 0 (exact zeros: the sequential sum is unchanged, bit for bit).
                    cp_base = _smem_u32(s_cpair.data_ptr())
                    u_base = _smem_u32(cache_hu.data_ptr()) + tidx * 2
                    if cutlass.const_expr(_HIST_UNROLL == 8):
                        if length <= 8:
                            hist_k, hist_q = _hist_window(
                                cp_base, u_base, length, 8, hist_k, hist_q
                            )
                        else:
                            hist_k, hist_q = _hist_window(
                                cp_base, u_base, length, N - 1, hist_k, hist_q
                            )
                    else:
                        hist_k, hist_q = _hist_window(
                            cp_base, u_base, length, N - 1, hist_k, hist_q
                        )
                else:
                    for hist in cutlass.range(length, unroll=4):
                        uw = _ld_shared_u16(_smem_u32(cache_hu.data_ptr()) + (hist * K + tidx) * 2)
                        cw2 = _ld_shared_u32(_smem_u32(s_cpair.data_ptr()) + hist * 4)
                        hist_k, hist_q = _hist_fma(uw, cw2, hist_k, hist_q)

            while not prims.mbarrier_try_wait_parity(acc_mbar, cutlass.Int32(0)):
                pass
            base_col = tmem_raw & cutlass.Int32(0xFFFF)
            base_row = tmem_raw >> cutlass.Int32(16)
            tmem_row = base_row + warp * cutlass.Int32(32)
            tmem_addr = (tmem_row << cutlass.Int32(16)) | (base_col + dcol)
            checkpoint_vec = prims.tcgen05_ld(
                "32x32b", cutlass.inttoptr(tmem_addr, 6, cutlass.Float32), num=8
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if cutlass.const_expr(tmem128):
                if length == N - 1:
                    # The decode columns are read: the commit may now seed TMEM over them.
                    prims.mbarrier_arrive(accread_mbar)
            checkpoint_rescale = state_outer * outer
            cp_k = checkpoint_vec[0] * checkpoint_rescale
            cp_q = checkpoint_vec[1] * checkpoint_rescale
            update = beta_sigmoid * (v_bf16.to(cutlass.Float32) - cp_k - hist_k)
            core = cp_q + hist_q + update * kq_dot
            core_bf16 = core.to(cutlass.BFloat16)
            core_f = core_bf16.to(cutlass.Float32)
            rms_partial = _warp_sum_squares(core_f)
            if lane == 0:
                s_red[8 + warp] = rms_partial
            _st_shared_u32(_smem_u32(s_u15.data_ptr()) + tidx * 4, _pack_bf16x2(update, update))
            _named_bar(1)
            commits = length == N - 1
            if not commits:
                if warp == 0:
                    prims.tcgen05_dealloc(
                        cutlass.inttoptr(tmem_raw, 6, cutlass.Float32),
                        cutlass.Int32(128 if tmem128 else 256),
                        group="cta_1",
                    )
            total = (s_red[8] + s_red[10]) + (s_red[9] + s_red[11])
            rstd = _rsqrt_approx(total * cutlass.Float32(1.0 / 128.0) + norm_eps)
            result = core_f * rstd * norm_bf16.to(cutlass.Float32) * zg
            out[(batch, head, tidx)] = result.to(cutlass.BFloat16)
            hu[(replay_slot, head, length, tidx)] = update.to(cutlass.BFloat16)
            if cutlass.const_expr(pdl):
                prims.griddepcontrol("launch_dependents")
            if cutlass.const_expr(not fast_silu):
                if tidx == 0:
                    if arrived == cutlass.Int32(heads - 1):
                        _st_keep_s32(_addr(arrivals, batch), cutlass.Int32(0))
                        next_length = length + 1
                        if commits:
                            next_length = cutlass.Int32(0)
                        _st_keep_s32(
                            _addr(lengths, cute.crd2idx((replay_slot,), lengths.layout)),
                            next_length,
                        )
            if commits:
                cute.arch.barrier()  # B4: decode done (u15 published), commit MMA issued
                lane_addr = Uint32(
                    (
                        ((tmem_raw >> cutlass.Int32(16)) + warp * cutlass.Int32(32))
                        << cutlass.Int32(16)
                    )
                    | (tmem_raw & cutlass.Int32(0xFFFF))
                )
                _commit_quantize(
                    tidx,
                    0,
                    lane,
                    warp,
                    tidx,
                    lane_addr,
                    Int64(pool_base.toint()),
                    s_u15,
                    s_k,
                    s_cmax,
                    commit_mbar,
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.barrier()  # all TMEM reads done
                if warp == 0:
                    prims.tcgen05_dealloc(
                        cutlass.inttoptr(tmem_raw, 6, cutlass.Float32),
                        cutlass.Int32(128 if tmem128 else 256),
                        group="cta_1",
                    )
        else:
            # ============================== C group ==============================
            # Warp cw owns history rows 4cw..4cw+3, lane l keys 4l..4l+3 (conflict-free rows).
            cw = warp - 4
            length_c = cutlass.Int32(0)
            outer_c = cutlass.Int32(0)
            htma = tmem128
            if cutlass.const_expr(htma):
                if tidx == 128:
                    # (conv_mbar is otherwise unused; initialized above by this thread)
                    prims.mbarrier_arrive_expect_tx(conv_mbar, N * K * 4 + N * K * 2)
                    _bulk_g_lef(
                        lat_ef,
                        cache_hg,
                        hg.iterator.raw_ptr() + cute.crd2idx((replay_slot, head, 0, 0), hg.layout),
                        conv_mbar,
                        N * K * 4,
                    )
                    _bulk_g_lef(
                        lat_ef,
                        s_hk,
                        hk.iterator.raw_ptr() + cute.crd2idx((replay_slot, head, 0, 0), hk.layout),
                        conv_mbar,
                        N * K * 2,
                    )
            if tidx == 128:
                length_c = _ld_global_s32(
                    _addr(lengths, cute.crd2idx((replay_slot,), lengths.layout))
                )
                outer_c = _ld_global_s32(Int64((pool_base + OUTER_SCALE).toint()))
            key0 = lane * 4
            if cutlass.const_expr(not htma):
                g_rows = [
                    (
                        hg.iterator.raw_ptr()
                        + cute.crd2idx((replay_slot, head, cw * 4 + r, key0), hg.layout)
                    ).load(count=4, alignment=16)
                    for r in range(4)
                ]
                k_rows = [
                    (
                        hk.iterator.raw_ptr()
                        + cute.crd2idx((replay_slot, head, cw * 4 + r, key0), hk.layout)
                    )
                    .load(count=4, alignment=8)
                    .bitcast(Uint32)
                    for r in range(4)
                ]
                # G rows to shared memory for the Q group's previous-G lookup.
                for r in cutlass.range_constexpr(4):
                    (cache_hg.data_ptr() + (cw * 4 + r) * K + key0).store(g_rows[r], alignment=16)
            arr_c = cutlass.Int32(0)
            if tidx == 128:
                s_meta[0] = length_c
                s_meta[1] = outer_c
                # Arrival as soon as this CTA's only length read is done (the row's last arriver
                # advances the length after B3), off the decode chain of the Q group.
                # (small grids only: with three CTAs per SM the Q-group arrival after B3 is faster)
                if cutlass.const_expr(fast_silu):
                    is_wc = cutlass.Int32(1)
                    arr_c = _atom_add_if(is_wc, _addr(arrivals, batch), cutlass.Int32(1))
                if length_c == N - 1:
                    # Commit: residual values and scales (read only by the commit).
                    prims.mbarrier_arrive_expect_tx(res_mbar, 9216)
                    _bulk_g_lef(
                        lat_ef,
                        res_vals,
                        cute.recast_ptr(pool_base + VALUES_BYTES, dtype=cutlass.Int8).raw_ptr(),
                        res_mbar,
                        8192,
                    )
                    _bulk_g_lef(
                        lat_ef,
                        res_scales,
                        cute.recast_ptr(pool_base + RESIDUAL_SCALE, dtype=cutlass.Int8).raw_ptr(),
                        res_mbar,
                        1024,
                    )
            cute.arch.barrier()  # B1
            if cutlass.const_expr(htma):
                while not prims.mbarrier_try_wait_parity(conv_mbar, cutlass.Int32(0)):
                    pass
                g_rows = [
                    (cache_hg.data_ptr() + (cw * 4 + r) * K + key0).load(count=4, alignment=16)
                    for r in range(4)
                ]
                k_rows = [
                    (s_hk.data_ptr() + (cw * 4 + r) * K + key0)
                    .load(count=4, alignment=8)
                    .bitcast(Uint32)
                    for r in range(4)
                ]
            length = s_meta[0]
            prev4 = cutlass.Vector.from_elements(
                (
                    cutlass.Float32(0.0),
                    cutlass.Float32(0.0),
                    cutlass.Float32(0.0),
                    cutlass.Float32(0.0),
                ),
                dtype=cutlass.Float32,
            )
            if length > 0:
                prev4 = (cache_hg.data_ptr() + (length - 1) * K + key0).load(count=4, alignment=16)
            # w = K * exp(prev_g - G): independent of this step's gate. A warp whose four rows are all
            # past the window's length skips its coefficients (they would only compete for issue slots
            # with the decode chain on the same schedulers).
            rows_live = cw * 4 < length
            w = [cutlass.Float32(0.0) for _ in range(16)]
            if rows_live:
                w = []
                for r in cutlass.range_constexpr(4):
                    ck = list(_bf16x2_lo_hi(k_rows[r][0])) + list(_bf16x2_lo_hi(k_rows[r][1]))
                    for j in cutlass.range_constexpr(4):
                        w.append(ck[j] * cute.exp(prev4[j] - g_rows[r][j], fastmath=True))
            cute.arch.barrier()  # B2
            adjs = [[cutlass.Float32(0.0) for _ in range(4)] for _ in range(4)]
            if rows_live:
                e4 = (s_estep.data_ptr() + key0).load(count=4, alignment=16)
                kn4 = (s_k.data_ptr() + key0).load(count=4, alignment=8).bitcast(Uint32)
                qn4 = (s_q.data_ptr() + key0).load(count=4, alignment=8).bitcast(Uint32)
                kn_f = list(_bf16x2_lo_hi(kn4[0])) + list(_bf16x2_lo_hi(kn4[1]))
                qn_f = list(_bf16x2_lo_hi(qn4[0])) + list(_bf16x2_lo_hi(qn4[1]))
                partial = []
                adjs = []
                for r in cutlass.range_constexpr(4):
                    adj = list(
                        _bf16x2_lo_hi(_pack_bf16x2(w[4 * r] * e4[0], w[4 * r + 1] * e4[1]))
                    ) + list(
                        _bf16x2_lo_hi(_pack_bf16x2(w[4 * r + 2] * e4[2], w[4 * r + 3] * e4[3]))
                    )
                    adjs.append(adj)
                    pk = adj[0] * kn_f[0]
                    pq = adj[0] * qn_f[0]
                    for j in cutlass.range_constexpr(1, 4):
                        pk = cute.fma(adj[j], kn_f[j], pk)
                        pq = cute.fma(adj[j], qn_f[j], pq)
                    valid = (cw * 4 + r) < length
                    partial.append(cutlass.select_(valid, pk, cutlass.Float32(0.0)))
                    partial.append(cutlass.select_(valid, pq, cutlass.Float32(0.0)))
                vals = partial
                width_ = 8
                for offset in (16, 8, 4):
                    half = width_ // 2
                    upper = (lane & offset) != 0
                    nxt = []
                    for i in cutlass.range_constexpr(half):
                        send = cutlass.select_(upper, vals[i], vals[half + i])
                        keep = cutlass.select_(upper, vals[half + i], vals[i])
                        nxt.append(keep + cute.arch.shuffle_sync_bfly(send, offset))
                    vals = nxt
                    width_ = half
                total = vals[0]
                total = total + cute.arch.shuffle_sync_bfly(total, 2)
                total = total + cute.arch.shuffle_sync_bfly(total, 1)
                sel = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1)
                if (lane & 3) == 0:
                    _st_shared_u16(
                        _smem_u32(s_cpair.data_ptr()) + (cw * 4 + sel // 2) * 4 + (sel % 2) * 2,
                        _pack_bf16x2(total, total),
                    )
            cute.arch.barrier()  # B3
            if cutlass.const_expr(fast_silu):
                if tidx == 128:
                    if arr_c == cutlass.Int32(heads - 1):
                        _st_keep_s32(_addr(arrivals, batch), cutlass.Int32(0))
                        next_length = cutlass.select_(length == N - 1, cutlass.Int32(0), length + 1)
                        _st_keep_s32(
                            _addr(lengths, cute.crd2idx((replay_slot,), lengths.layout)),
                            next_length,
                        )
            if length == N - 1:
                # ---- commit, phase 1 (overlaps the decode tail): panels, decayed seed, BF16 MMA ----
                v = cw * 32 + lane
                krow = key0
                for j in cutlass.range_constexpr(4):
                    b3 = cutlass.select_(cw == 3, cutlass.Float32(0.0), adjs[3][j])
                    off = _panel_offset(krow + j, cw // 2) + (cw % 2) * 8
                    _st_shared_v2(
                        _smem_u32(panel_b.data_ptr()) + off,
                        _pack_bf16x2(adjs[0][j], adjs[1][j]),
                        _pack_bf16x2(adjs[2][j], b3),
                    )
                # A operand: U_i[v] for i < 15, zero for the current token's row.
                while not prims.mbarrier_try_wait_parity(u_mbar, cutlass.Int32(0)):
                    pass
                uv = [cache_hu[i * K + v].to(cutlass.Float32) for i in range(N - 1)] + [
                    cutlass.Float32(0.0)
                ]
                for chunk in cutlass.range_constexpr(2):
                    _st_shared_v4(
                        _smem_u32(panel_a.data_ptr()) + _panel_offset(v, chunk),
                        _pack_bf16x2(uv[8 * chunk], uv[8 * chunk + 1]),
                        _pack_bf16x2(uv[8 * chunk + 2], uv[8 * chunk + 3]),
                        _pack_bf16x2(uv[8 * chunk + 4], uv[8 * chunk + 5]),
                        _pack_bf16x2(uv[8 * chunk + 6], uv[8 * chunk + 7]),
                    )
                while not prims.mbarrier_try_wait_parity(res_mbar, cutlass.Int32(0)):
                    pass
                while not prims.mbarrier_try_wait_parity(state_mbar, cutlass.Int32(0)):
                    pass
                tmem_raw_c = tmem_holding.load()
                lane_addr = Uint32(
                    (
                        ((tmem_raw_c >> cutlass.Int32(16)) + cw * cutlass.Int32(32))
                        << cutlass.Int32(16)
                    )
                    | (tmem_raw_c & cutlass.Int32(0xFFFF))
                )
                if cutlass.const_expr(tmem128):
                    while not prims.mbarrier_try_wait_parity(accread_mbar, cutlass.Int32(0)):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                _seed_commit(v, lane_addr, mma_a, res_vals, res_scales, mma_sfa, s_od)
                _fence_proxy_async_smem()
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                _named_bar(2)
                if cw == 0:
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    idesc_c = prims.Tcgen05InstrDesc.build(
                        c_dtype=Float32,
                        a_dtype=cutlass.BFloat16,
                        b_dtype=cutlass.BFloat16,
                        n_dim=K,
                        m_dim=K,
                    )
                    desc_pa = prims.Tcgen05SmemDesc.build(
                        panel_a,
                        leading_byte_offset=PANEL_LBO,
                        stride_byte_offset=PANEL_SBO,
                        layout=prims.Tcgen05SmemSwizzle.NONE,
                    )
                    desc_pb = prims.Tcgen05SmemDesc.build(
                        panel_b,
                        leading_byte_offset=PANEL_LBO,
                        stride_byte_offset=PANEL_SBO,
                        layout=prims.Tcgen05SmemSwizzle.NONE,
                    )
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tmem_raw_c, 6, Float32),
                            desc_pa,
                            desc_pb,
                            idesc_c,
                            True,
                        )
                        prims.tcgen05_commit(commit_mbar, group=prims.CTAGroup.CTA_1)
                cute.arch.barrier()  # B4
                # ---- commit, phase 2: both groups quantize; C group takes keys 64..127 ----
                _commit_quantize(
                    v,
                    1,
                    lane,
                    warp,
                    tidx,
                    lane_addr,
                    Int64(pool_base.toint()),
                    s_u15,
                    s_k,
                    s_cmax,
                    commit_mbar,
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.barrier()  # all TMEM reads done


@cute.jit
def _commit_rows_chunk(lane_addr, u15w, s_k, c):
    """32 keys (chunk c) of TMEM row: accumulator + u15 * kn (exact BF16 products)."""
    acc = _tmem_ld32(lane_addr + Uint32(32 * c))
    knw = [
        (s_k.data_ptr() + 32 * c + 8 * q).load(count=8, alignment=16).bitcast(Uint32)
        for q in range(4)
    ]
    sn = []
    for q in cutlass.range_constexpr(4):
        for p in cutlass.range_constexpr(4):
            x0, x1 = _fma_bf16pair(u15w, knw[q][p], acc[8 * q + 2 * p], acc[8 * q + 2 * p + 1])
            sn.append(x0)
            sn.append(x1)
    return sn


@cute.jit
def _commit_quantize_all(
    v, lane, warp, tidx, lane_addr, head_base, s_u15, s_k, s_cmax, commit_mbar, commit_phase
):
    """4-warp commit quantization: thread v owns TMEM row v (all 128 keys), two TMEM passes
    (block maxima first, then quantization) to keep the register footprint small."""
    while not prims.mbarrier_try_wait_parity(commit_mbar, commit_phase):
        pass
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    u15w = _ld_shared_u32(_smem_u32(s_u15.data_ptr()) + v * 4)
    bam = []
    for c in cutlass.range_constexpr(4):
        sn = _commit_rows_chunk(lane_addr, u15w, s_k, c)
        for h2 in cutlass.range_constexpr(2):
            bam.append(_tree_max([_f_abs(sn[16 * h2 + t]) for t in range(16)]))
    m = _warp_max_f32(_tree_max(bam))
    if lane == 0:
        s_cmax[warp] = m
    cute.arch.barrier()
    gmax = _tree_max([s_cmax[w] for w in range(4)])
    outer_new = _f_div(gmax, Float32(2688.0))
    t6 = _f_mul(Float32(6.0), _sel_eq0(outer_new, Float32(1.0), outer_new))
    inv_t6 = _rcp_n(t6)
    row_base = head_base + Int64(v) * 64
    for h in cutlass.range_constexpr(2):
        psc = [_f_min(Float32(448.0), _f_mul(bam[4 * h + g], inv_t6)) for g in range(4)]
        pscp = _pack_e4m3x4(psc[0], psc[1], psc[2], psc[3])
        _st_global_u32(head_base + PRIMARY_SCALE + Int64(v) * 8 + 4 * h, pscp)
        pe = list(_expand4(pscp, outer_new))
        rbytes = []
        for cc in cutlass.range_constexpr(2):
            c = 2 * h + cc
            sn = _commit_rows_chunk(lane_addr, u15w, s_k, c)
            for g2 in cutlass.range_constexpr(2):
                g = 2 * cc + g2
                e = pe[g]
                pinv = _sel_eq0(e, Float32(0.0), _rcp_n(e))
                y = [_f_mul(sn[16 * g2 + t], pinv) for t in range(16)]
                pwords = [_pack_e2m1x8(y[8 * hh : 8 * hh + 8]) for hh in range(2)]
                _st_global_v2(row_base + 8 * (4 * h + g), pwords[0], pwords[1])
                rem_s = _rem8(pwords[0], y[0:8]) + _rem8(pwords[1], y[8:16])
                rb = _f_mul(_tree_max([_f_abs(r) for r in rem_s]), e)
                rs = _f_min(Float32(448.0), _f_mul(rb, inv_t6))
                rbyte = _pack_e4m3x2(rs, rs) & Uint32(0xFF)
                rbytes.append(rbyte)
                re = _expand4(rbyte, outer_new)[0]
                rmul = _sel_eq0(re, Float32(0.0), _f_mul(e, _rcp_n(re)))
                rwords = [
                    _pack_e2m1x8([_f_mul(rem_s[8 * hh + t], rmul) for t in range(8)])
                    for hh in range(2)
                ]
                _st_global_v2(row_base + VALUES_BYTES + 8 * (4 * h + g), rwords[0], rwords[1])
        rword = (
            rbytes[0]
            | (rbytes[1] << Uint32(8))
            | (rbytes[2] << Uint32(16))
            | (rbytes[3] << Uint32(24))
        )
        _st_global_u32(head_base + RESIDUAL_SCALE + Int64(v) * 8 + 4 * h, rword)
    if tidx == 0:
        _st_outer_pad(head_base + OUTER_SCALE, outer_new)


# ------------------------------------------------------------------ throughput (persistent) kernel

# Share of each CTA's items assigned statically when the tail is claimed dynamically.
_TP_STATIC_PCT = 85  # share of a persistent CTA's items assigned statically; the rest are claimed
# Per-stage shared-memory layout (bytes, all 128-B aligned).
SG_CK = 0  # primary image 8 KB (SW64 TMA)
SG_SF = 8192  # primary scales 1 KB
SG_HK = 9216  # K history 4 KB
SG_HU = 13312  # U history 4 KB
SG_HG = 17408  # G history 8 KB
SG_CONV = 25600  # Conv windows 3 x 768 B
SG_W = 27904  # Conv weights 3 x 1 KB
SG_DT = 30976  # dt_bias 512 B
SG_X = 31488  # x q/k/v 3 x 256 B
SG_GATE = 32256  # gate 256 B
SG_Z = 32512  # z 256 B
SG_BYTES = 32768
# meta words per stage: 0 slot, 1 rslot, 2 length, 3 dummy, 4 batch, 5 head, 6 outer bits, 7 a_log bits, 8 beta bits
META_WORDS = 16


@cute.jit
def _tp_issue_pre(
    tma_state,
    stage_ptr,
    full_mbar,
    item,
    heads: cutlass.Constexpr,
    slot,
    replay_slot,
    length,
    nbytes,
    pool,
    conv_state,
    conv_weight,
    dt_bias,
    hk,
    hu,
    hg,
):
    """Pre-wait TMAs of one item into a stage (all allowed before griddepcontrol.wait)."""
    batch = item // heads
    head = item - batch * heads
    width = heads * K
    prims.mbarrier_arrive_expect_tx(full_mbar, nbytes)
    _bulk_t_tp(
        _arr8(stage_ptr, SG_CK, 8192),
        tma_state.get_ptr(),
        (cutlass.Int32(0), cutlass.Int32(0), head, slot),
        full_mbar,
    )
    pool_head = pool.iterator + cute.crd2idx((slot, head, 0), pool.layout)
    _bulk_g_tp(
        _arr8(stage_ptr, SG_SF, 1024),
        cute.recast_ptr(pool_head + PRIMARY_SCALE, dtype=cutlass.Int8).raw_ptr(),
        full_mbar,
        1024,
    )
    if length > 0:
        coord = (replay_slot, head, 0, 0)
        _bulk_g_tp(
            _arr8(stage_ptr, SG_HK, 4096),
            hk.iterator.raw_ptr() + cute.crd2idx(coord, hk.layout),
            full_mbar,
            length * (K * 2),
        )
        _bulk_g_tp(
            _arr8(stage_ptr, SG_HU, 4096),
            hu.iterator.raw_ptr() + cute.crd2idx(coord, hu.layout),
            full_mbar,
            length * (K * 2),
        )
        _bulk_g_tp(
            _arr8(stage_ptr, SG_HG, 8192),
            hg.iterator.raw_ptr() + cute.crd2idx(coord, hg.layout),
            full_mbar,
            length * (K * 4),
        )
    state_raw = conv_state.iterator.raw_ptr()
    weight_raw = conv_weight.iterator.raw_ptr()
    for part in cutlass.range_constexpr(3):
        _bulk_g_tp(
            _arr8(stage_ptr, SG_CONV + part * 768, 768),
            state_raw + cute.crd2idx((slot, part * width + head * K, 0), conv_state.layout),
            full_mbar,
            768,
        )
        _bulk_g_tp(
            _arr8(stage_ptr, SG_W + part * 1024, 1024),
            weight_raw + (part * width + head * K) * 4,
            full_mbar,
            1024,
        )
    _bulk_g_tp(
        _arr8(stage_ptr, SG_DT, 512),
        dt_bias.iterator.raw_ptr() + head * K,
        full_mbar,
        512,
    )


@cute.jit
def _tp_issue_post(stage_ptr, full_mbar, item, heads: cutlass.Constexpr, x, gate, z):
    batch = item // heads
    head = item - batch * heads
    width = heads * K
    for part in cutlass.range_constexpr(3):
        _bulk_g(
            _arr8(stage_ptr, SG_X + part * 256, 256),
            x.iterator.raw_ptr() + cute.crd2idx((batch, part * width + head * K), x.layout),
            full_mbar,
            256,
            True,
        )
    _bulk_g(
        _arr8(stage_ptr, SG_GATE, 256),
        gate.iterator.raw_ptr() + cute.crd2idx((batch, head * K), gate.layout),
        full_mbar,
        256,
        True,
    )
    _bulk_g(
        _arr8(stage_ptr, SG_Z, 256),
        z.iterator.raw_ptr() + cute.crd2idx((batch, head, 0), z.layout),
        full_mbar,
        256,
        True,
    )


@cute.jit
def _tp_issue_warp(
    warp,
    tma_state,
    stage_ptr,
    full_mbar,
    item,
    heads: cutlass.Constexpr,
    slot,
    replay_slot,
    length,
    dmy,
    pool,
    conv_state,
    conv_weight,
    dt_bias,
    hk,
    hu,
    hg,
    x,
    gate,
    z,
):
    """Refill of one stage, split over the four warps (elected lane of each)."""
    batch = item // heads
    head = item - batch * heads
    width = heads * K
    if warp == 0:
        if dmy != 0:
            prims.mbarrier_arrive(full_mbar)
        else:
            nbytes = 9216 + length * (K * 8) + 2304 + 3072 + 512 + 768 + 512
            prims.mbarrier_arrive_expect_tx(full_mbar, nbytes)
            _bulk_t_tp(
                _arr8(stage_ptr, SG_CK, 8192),
                tma_state.get_ptr(),
                (cutlass.Int32(0), cutlass.Int32(0), head, slot),
                full_mbar,
            )
            pool_head = pool.iterator + cute.crd2idx((slot, head, 0), pool.layout)
            _bulk_g_tp(
                _arr8(stage_ptr, SG_SF, 1024),
                cute.recast_ptr(pool_head + PRIMARY_SCALE, dtype=cutlass.Int8).raw_ptr(),
                full_mbar,
                1024,
            )
    if dmy == 0:
        if warp == 1:
            if length > 0:
                coord = (replay_slot, head, 0, 0)
                _bulk_g_tp(
                    _arr8(stage_ptr, SG_HK, 4096),
                    hk.iterator.raw_ptr() + cute.crd2idx(coord, hk.layout),
                    full_mbar,
                    length * (K * 2),
                )
                _bulk_g_tp(
                    _arr8(stage_ptr, SG_HU, 4096),
                    hu.iterator.raw_ptr() + cute.crd2idx(coord, hu.layout),
                    full_mbar,
                    length * (K * 2),
                )
                _bulk_g_tp(
                    _arr8(stage_ptr, SG_HG, 8192),
                    hg.iterator.raw_ptr() + cute.crd2idx(coord, hg.layout),
                    full_mbar,
                    length * (K * 4),
                )
        if warp == 2:
            state_raw = conv_state.iterator.raw_ptr()
            weight_raw = conv_weight.iterator.raw_ptr()
            for part in cutlass.range_constexpr(3):
                _bulk_g_tp(
                    _arr8(stage_ptr, SG_CONV + part * 768, 768),
                    state_raw + cute.crd2idx((slot, part * width + head * K, 0), conv_state.layout),
                    full_mbar,
                    768,
                )
                _bulk_g_tp(
                    _arr8(stage_ptr, SG_W + part * 1024, 1024),
                    weight_raw + (part * width + head * K) * 4,
                    full_mbar,
                    1024,
                )
        if warp == 3:
            _bulk_g_tp(
                _arr8(stage_ptr, SG_DT, 512),
                dt_bias.iterator.raw_ptr() + head * K,
                full_mbar,
                512,
            )
            for part in cutlass.range_constexpr(3):
                _bulk_g_tp(
                    _arr8(stage_ptr, SG_X + part * 256, 256),
                    x.iterator.raw_ptr() + cute.crd2idx((batch, part * width + head * K), x.layout),
                    full_mbar,
                    256,
                )
            _bulk_g_tp(
                _arr8(stage_ptr, SG_GATE, 256),
                gate.iterator.raw_ptr() + cute.crd2idx((batch, head * K), gate.layout),
                full_mbar,
                256,
            )
            _bulk_g_tp(
                _arr8(stage_ptr, SG_Z, 256),
                z.iterator.raw_ptr() + cute.crd2idx((batch, head, 0), z.layout),
                full_mbar,
                256,
            )


@cute.jit
def _tp_refill_b1(
    warp,
    stage_ptr,
    mbar,
    item,
    heads: cutlass.Constexpr,
    slot,
    conv_state,
    conv_weight,
    dt_bias,
    x,
    gate,
    z,
    go,
):
    """Early refill, part 1 (after B1: the Conv window, weights, dt and projections are consumed).
    Warps 1 and 2 each arrive once on the stage's start barrier (with their bytes unless padding)."""
    batch = item // heads
    head = item - batch * heads
    width = heads * K
    if warp == 1:
        if not go:
            prims.mbarrier_arrive(mbar)
        else:
            prims.mbarrier_arrive_expect_tx(mbar, 5888)
            state_raw = conv_state.iterator.raw_ptr()
            weight_raw = conv_weight.iterator.raw_ptr()
            for part in cutlass.range_constexpr(3):
                _bulk_g(
                    _arr8(stage_ptr, SG_CONV + part * 768, 768),
                    state_raw + cute.crd2idx((slot, part * width + head * K, 0), conv_state.layout),
                    mbar,
                    768,
                    True,
                )
                _bulk_g_tp(
                    _arr8(stage_ptr, SG_W + part * 1024, 1024),
                    weight_raw + (part * width + head * K) * 4,
                    mbar,
                    1024,
                )
            _bulk_g_tp(
                _arr8(stage_ptr, SG_DT, 512),
                dt_bias.iterator.raw_ptr() + head * K,
                mbar,
                512,
            )
    if warp == 2:
        if not go:
            prims.mbarrier_arrive(mbar)
        else:
            prims.mbarrier_arrive_expect_tx(mbar, 1280)
            for part in cutlass.range_constexpr(3):
                _bulk_g(
                    _arr8(stage_ptr, SG_X + part * 256, 256),
                    x.iterator.raw_ptr() + cute.crd2idx((batch, part * width + head * K), x.layout),
                    mbar,
                    256,
                    True,
                )
            _bulk_g(
                _arr8(stage_ptr, SG_GATE, 256),
                gate.iterator.raw_ptr() + cute.crd2idx((batch, head * K), gate.layout),
                mbar,
                256,
                True,
            )
            _bulk_g(
                _arr8(stage_ptr, SG_Z, 256),
                z.iterator.raw_ptr() + cute.crd2idx((batch, head, 0), z.layout),
                mbar,
                256,
                True,
            )


@cute.jit
def _tp_refill_mid(
    part: cutlass.Constexpr,
    stage_ptr,
    mbar,
    tma_state,
    pool,
    hk,
    hg,
    head,
    slot,
    replay_slot,
    length,
    go,
):
    """Early refill of the scales (part 0), checkpoint image (1) or K/G rows (2): one arrival on the
    stage's middle barrier each (with the bytes unless padding)."""
    if not go:
        prims.mbarrier_arrive(mbar)
    else:
        if cutlass.const_expr(part == 0):
            prims.mbarrier_arrive_expect_tx(mbar, 1024)
            e_pool = pool.iterator + cute.crd2idx((slot, head, 0), pool.layout)
            _bulk_g(
                _arr8(stage_ptr, SG_SF, 1024),
                cute.recast_ptr(e_pool + PRIMARY_SCALE, dtype=cutlass.Int8).raw_ptr(),
                mbar,
                1024,
                False,
            )
        if cutlass.const_expr(part == 1):
            prims.mbarrier_arrive_expect_tx(mbar, 8192)
            _bulk_t(
                _arr8(stage_ptr, SG_CK, 8192),
                tma_state.get_ptr(),
                (cutlass.Int32(0), cutlass.Int32(0), head, slot),
                mbar,
                False,
            )
        if cutlass.const_expr(part == 2):
            prims.mbarrier_arrive_expect_tx(mbar, length * (K * 6))
            if length > 0:
                coord = (replay_slot, head, 0, 0)
                hpol = _hist_pol(length)
                _bulk_g_rp(
                    _arr8(stage_ptr, SG_HK, 4096),
                    hk.iterator.raw_ptr() + cute.crd2idx(coord, hk.layout),
                    mbar,
                    length * (K * 2),
                    hpol,
                )
                _bulk_g_rp(
                    _arr8(stage_ptr, SG_HG, 8192),
                    hg.iterator.raw_ptr() + cute.crd2idx(coord, hg.layout),
                    mbar,
                    length * (K * 4),
                    hpol,
                )


@cute.jit
def _seed_commit_t(
    v,
    lane_addr,
    mma_a,
    sbase,
    res_off,
    res_scales,
    mma_sfa,
    s_od,
):
    """_seed_commit with the residual row v at sbase + res_off + 64 v."""
    swizzle_s64 = cutlass.Swizzle.from_name("s64b")
    psw = [
        _ld_shared_u32(_smem_u32(mma_sfa.data_ptr()) + kb * 512 + (v % 32) * 16 + (v // 32) * 4)
        for kb in range(2)
    ]
    rsw = [_ld_shared_u32(_smem_u32(res_scales.data_ptr()) + v * 8 + 4 * j) for j in range(2)]
    sp = list(_scales4_f16(psw[0])) + list(_scales4_f16(psw[1]))
    sr = list(_scales4_f16(rsw[0])) + list(_scales4_f16(rsw[1]))
    res_row = cutlass.Array(sbase + res_off + v * 64, shape=64, dtype=cutlass.Int8)
    for c in cutlass.range_constexpr(4):
        pw = (
            mma_a.subview(v * 64 + c * 16)
            .data_ptr()
            .apply_swizzle(swizzle_s64)
            .load(count=16, alignment=16)
            .bitcast(Uint32)
        )
        rw = (res_row.subview(c * 16).data_ptr()).load(count=16, alignment=16).bitcast(Uint32)
        od = [(s_od.data_ptr() + 32 * c + 4 * q).load(count=4, alignment=16) for q in range(8)]
        seed = []
        for jj in cutlass.range_constexpr(4):
            g = 2 * c + jj // 2
            seed = seed + _seed8(
                pw[jj], rw[jj], sp[g], sr[g], [od[2 * jj + t // 4][t % 4] for t in range(8)]
            )
        _tmem_st32(lane_addr + Uint32(32 * c), seed)


@dsl_user_op
def _dot4_bf16(
    a01: Uint32, a23: Uint32, k01: Uint32, k23: Uint32, acc: Float32, *, loc=None, ip=None
) -> Float32:
    """acc + sum_j a_j * k_j over 4 packed BF16 pairs (exact products, FP32 accumulation in order)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                _ir(Uint32(a01), loc, ip),
                _ir(Uint32(a23), loc, ip),
                _ir(Uint32(k01), loc, ip),
                _ir(Uint32(k23), loc, ip),
                _ir(Float32(acc), loc, ip),
            ],
            "{ .reg .b16 a0, a1, a2, a3, k0, k1, k2, k3; .reg .f32 t;\n"
            " mov.b32 {a0, a1}, $1; mov.b32 {a2, a3}, $2; mov.b32 {k0, k1}, $3; mov.b32 {k2, k3}, $4;\n"
            " fma.rn.f32.bf16 t, a0, k0, $5;\n fma.rn.f32.bf16 t, a1, k1, t;\n"
            " fma.rn.f32.bf16 t, a2, k2, t;\n fma.rn.f32.bf16 $0, a3, k3, t; }",
            "=f,r,r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _conv4_bf16(
    w01: Uint32, w23: Uint32, s0: Uint32, s1: Uint32, s2: Uint32, x: Uint32, *, loc=None, ip=None
) -> Float32:
    """w0*s0 + w1*s1 + w2*s2 + w3*x as one FP32 FMA chain on BF16 operands (exact products), i.e.
    the same roundings as converting to FP32 first."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                _ir(Uint32(w01), loc, ip),
                _ir(Uint32(w23), loc, ip),
                _ir(Uint32(s0), loc, ip),
                _ir(Uint32(s1), loc, ip),
                _ir(Uint32(s2), loc, ip),
                _ir(Uint32(x), loc, ip),
            ],
            "{ .reg .b16 w0, w1, w2, w3, a, b, c, d, z; .reg .f32 t;\n"
            " mov.b32 {w0, w1}, $1; mov.b32 {w2, w3}, $2;\n"
            " mov.b32 {a, z}, $3; mov.b32 {b, z}, $4; mov.b32 {c, z}, $5; mov.b32 {d, z}, $6;\n"
            " fma.rn.f32.bf16 t, w0, a, 0f00000000;\n fma.rn.f32.bf16 t, w1, b, t;\n"
            " fma.rn.f32.bf16 t, w2, c, t;\n fma.rn.f32.bf16 $0, w3, d, t; }",
            "=f,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _hist_fma(u: Uint32, cpair: Uint32, acc_k: Float32, acc_q: Float32, *, loc=None, ip=None):
    """(acc_k + u.lo * c.lo, acc_q + u.lo * c.hi) with BF16 operands."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            _ir(Uint32(u), loc, ip),
            _ir(Uint32(cpair), loc, ip),
            _ir(Float32(acc_k), loc, ip),
            _ir(Float32(acc_q), loc, ip),
        ],
        "{ .reg .b16 ul, uh, cl, ch;\n mov.b32 {ul, uh}, $2;\n mov.b32 {cl, ch}, $3;\n"
        " fma.rn.f32.bf16 $0, ul, cl, $4;\n fma.rn.f32.bf16 $1, ul, ch, $5; }",
        "=f,=f,r,r,f,f",
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
def _ld_shared_u16(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int32(addr), loc, ip)],
            "{ .reg .b16 t; ld.shared.b16 t, [$1]; cvt.u32.u16 $0, t; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_u16(addr: Int32, v: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int32(addr), loc, ip), _ir(Uint32(v), loc, ip)],
        "{ .reg .b16 t; cvt.u16.u32 t, $1; st.shared.b16 [$0], t; }",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rcp_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(x), loc, ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rsqrt_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_ir(Float32(x), loc, ip)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_u16_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Raw 16-bit global load, zero-extended (no FP conversion near the load)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "{ .reg .b16 t; ld.global.b16 t, [$1]; cvt.u32.u16 $0, t; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _arr8(ptr, off, n):
    return cutlass.Array(ptr + off, shape=n, dtype=cutlass.Int8)


@dsl_user_op
def _prefetch_l2(addr: Int64, nbytes: int, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip)],
        f"cp.async.bulk.prefetch.L2.global [$0], {nbytes};",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_inc_if(pred: Int32, addr: Int64, *, loc=None, ip=None) -> Int32:
    """Predicated atomic ticket (old value; +1). atom.inc is not warp-aggregated by ptxas (an
    aggregated add shuffles its result right after the atomic, stalling the warp for the round trip)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int32(pred), loc, ip), _ir(Int64(addr), loc, ip)],
            "{ .reg .pred p; setp.ne.s32 p, $1, 0; mov.b32 $0, 0; "
            "@p atom.relaxed.gpu.global.inc.u32 $0, [$2], 2147483647; }",
            "=r,r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def _tp_kernel(
    tma_state: cutlass.GridConstant[cuda_exp.TensorMap],
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    replay_indices,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    arrivals,
    sched,
    cflags,
    n_items: Int32,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    ext_commit: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    dyn: cutlass.Constexpr,
    l2pf: cutlass.Constexpr,
):
    """Persistent decode: CTA c handles items c, c+G, ... (item = row * H + head), two-stage TMA prefetch."""
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    ncta, _, _ = cute.arch.grid_dim()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    s_stages = cutlass.Array(
        cutlass.Int8, stages * SG_BYTES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    s_meta = cutlass.Array(
        cutlass.Int32, stages * META_WORDS, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_pre = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    s_next = cutlass.Array(
        cutlass.Int32, 2 * META_WORDS, space=cutlass.AddressSpace.smem, alignment=16
    )
    full_mbar = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    # ext_commit: the stage's parts complete on three barriers (start / after B1 / before the history loop)
    mid_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    late_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    commit_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    res_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    smem = utils.SmemAllocator()
    s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16,)), byte_alignment=16)
    s_q = cutlass.Array(cutlass.BFloat16, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_k = cutlass.Array(cutlass.BFloat16, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_cum = cutlass.Array(cutlass.Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
    # (commit-only buffers are not allocated in ext_commit mode: ~1.2 KB, the 6th CTA per SM)
    s_od = None
    s_u15 = None
    s_cmax = None
    if cutlass.const_expr(not ext_commit):
        s_od = cutlass.Array(cutlass.Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
        s_u15 = cutlass.Array(cutlass.Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
        s_cmax = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    s_cpair = cutlass.Array(cutlass.Int32, N, space=cutlass.AddressSpace.smem, alignment=16)
    mma_b = cutlass.Array(cutlass.Int8, 512, space=cutlass.AddressSpace.smem, alignment=128)
    mma_sfa = cutlass.Array(
        cutlass.Float8E4M3FN, 1024, space=cutlass.AddressSpace.smem, alignment=128
    )
    mma_sfb = cutlass.Array(
        cutlass.Float8E4M3FN, 1024, space=cutlass.AddressSpace.smem, alignment=128
    )
    tmem_holding = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    if warp == 0:
        if prims.elect_sync():
            for st0 in cutlass.range_constexpr(stages):
                prims.mbarrier_init(
                    cutlass.Array(full_mbar.subview(st0).data_ptr(), shape=1, dtype=cutlass.Int64),
                    3 if ext_commit else 1,
                )
            if cutlass.const_expr(ext_commit):
                prims.mbarrier_init(mid_mbar, 3)
                prims.mbarrier_init(late_mbar, 1)
            prims.mbarrier_init(acc_mbar, 1)
            prims.mbarrier_init(commit_mbar, 1)
            prims.mbarrier_init(res_mbar, 1)
            prims.fence_mbarrier_init()
    # (warp 3: thread 0 goes straight from the barrier inits to the prologue's metadata loads)
    if warp == 3:
        prims.tcgen05_alloc(tmem_holding, cutlass.Int32((32 if ext_commit else 128)), group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    norm_bf16 = norm_weight[tidx]
    width = heads * K
    # SFB atoms and the B operand's padding rows stay zero for every item (only the group
    # leaders' scale bytes and rows 0-1 change).
    sfb_words = cutlass.inttoptr(
        mma_sfb.subview(tidx * 8).data_ptr().toint(), cutlass.AddressSpace.smem, cutlass.Int32
    )
    sfb_words.store(
        cutlass.Vector.from_elements((cutlass.Int32(0), cutlass.Int32(0)), dtype=cutlass.Int32),
        alignment=8,
    )
    if tidx < 24:
        zero0 = cutlass.Int32(0)
        cutlass.inttoptr(
            mma_b.subview(128 + tidx * 16).data_ptr().toint(),
            cutlass.AddressSpace.smem,
            cutlass.Int32,
        ).store(
            cutlass.Vector.from_elements((zero0, zero0, zero0, zero0), dtype=cutlass.Int32),
            alignment=16,
        )

    # ---- prologue (thread 0): full metadata + pre-wait TMAs for the first stages items,
    # slot/replay/dummy of the item after them ----
    item2p = cta + stages * ncta
    pre2_slot = cutlass.Int32(0)
    pre2_rs = cutlass.Int32(0)
    pre2_dmy = cutlass.Int32(0)
    if tidx == 0:
        # (issued with the first item's loads; stored after the projections' TMA)
        if item2p < n_items:
            batch2p = item2p // heads
            pre2_slot = _ld_keep_s32(_addr(slots, batch2p))
            pre2_rs = _ld_keep_s32(_addr(replay_indices, batch2p))
            pre2_dmy = _ld_keep_u8(_addr(dummy, batch2p))
        for st0 in cutlass.range_constexpr(stages):
            item0 = cta + st0 * ncta
            if item0 < n_items:
                batch0 = item0 // heads
                head0 = item0 - batch0 * heads
                slot0 = _ld_keep_s32(_addr(slots, batch0))
                rs0 = _ld_keep_s32(_addr(replay_indices, batch0))
                dmy0 = _ld_keep_u8(_addr(dummy, batch0))
                mb0 = cutlass.Array(full_mbar.subview(st0).data_ptr(), shape=1, dtype=cutlass.Int64)
                base0 = st0 * META_WORDS
                s_meta[base0 + 3] = dmy0
                s_meta[base0 + 4] = batch0
                s_meta[base0 + 5] = head0
                if dmy0 == 0:
                    if cutlass.const_expr(ext_commit):
                        # (loads first, then everything that needs only the slot, then the length-sized parts:
                        # two dependent rounds instead of four)
                        alog0 = _ld_global_s32(_addr(a_log, head0))
                        len0 = _ld_keep_s32(
                            _addr(lengths, cute.crd2idx((cutlass.Int64(rs0),), lengths.layout))
                        )
                        pool0 = pool.iterator + cute.crd2idx((slot0, head0, 0), pool.layout)
                        outer0 = _ld_global_s32(Int64((pool0 + OUTER_SCALE).toint()))
                        sp0 = s_stages.subview(st0 * SG_BYTES).data_ptr()
                        prims.mbarrier_arrive_expect_tx(mb0, 5888)
                        state_raw0 = conv_state.iterator.raw_ptr()
                        weight_raw0 = conv_weight.iterator.raw_ptr()
                        for part in cutlass.range_constexpr(3):
                            _bulk_g(
                                _arr8(sp0, SG_CONV + part * 768, 768),
                                state_raw0
                                + cute.crd2idx(
                                    (slot0, part * width + head0 * K, 0), conv_state.layout
                                ),
                                mb0,
                                768,
                                True,
                            )
                            _bulk_g_tp(
                                _arr8(sp0, SG_W + part * 1024, 1024),
                                weight_raw0 + (part * width + head0 * K) * 4,
                                mb0,
                                1024,
                            )
                        _bulk_g_tp(
                            _arr8(sp0, SG_DT, 512),
                            dt_bias.iterator.raw_ptr() + head0 * K,
                            mb0,
                            512,
                        )
                        for mpart in cutlass.range_constexpr(2):
                            _tp_refill_mid(
                                mpart,
                                sp0,
                                mid_mbar,
                                tma_state,
                                pool,
                                hk,
                                hg,
                                head0,
                                slot0,
                                cutlass.Int64(rs0),
                                len0,
                                True,
                            )
                        s_meta[base0 + 0] = slot0
                        s_meta[base0 + 1] = rs0
                        s_meta[base0 + 7] = alog0
                        s_meta[base0 + 2] = len0
                        s_meta[base0 + 6] = outer0
                        _tp_refill_mid(
                            2,
                            sp0,
                            mid_mbar,
                            tma_state,
                            pool,
                            hk,
                            hg,
                            head0,
                            slot0,
                            cutlass.Int64(rs0),
                            len0,
                            True,
                        )
                    else:
                        len0 = _ld_keep_s32(
                            _addr(lengths, cute.crd2idx((cutlass.Int64(rs0),), lengths.layout))
                        )
                        pool0 = pool.iterator + cute.crd2idx((slot0, head0, 0), pool.layout)
                        s_meta[base0 + 0] = slot0
                        s_meta[base0 + 1] = rs0
                        s_meta[base0 + 2] = len0
                        s_meta[base0 + 6] = _ld_global_s32(Int64((pool0 + OUTER_SCALE).toint()))
                        s_meta[base0 + 7] = _f32_as_s32(a_log[head0])
                    if cutlass.const_expr(ext_commit):
                        sp0 = s_stages.subview(st0 * SG_BYTES).data_ptr()
                        prims.mbarrier_arrive_expect_tx(late_mbar, len0 * (K * 2))
                        if len0 > 0:
                            _bulk_g_rp(
                                _arr8(sp0, SG_HU, 4096),
                                hu.iterator.raw_ptr()
                                + cute.crd2idx((cutlass.Int64(rs0), head0, 0, 0), hu.layout),
                                late_mbar,
                                len0 * (K * 2),
                                _hist_pol(len0),
                            )
                    else:
                        nbytes0 = 9216 + len0 * (K * 8) + 2304 + 3072 + 512 + 768 + 512
                        _tp_issue_pre(
                            tma_state,
                            s_stages.subview(st0 * SG_BYTES).data_ptr(),
                            mb0,
                            item0,
                            heads,
                            slot0,
                            cutlass.Int64(rs0),
                            len0,
                            nbytes0,
                            pool,
                            conv_state,
                            conv_weight,
                            dt_bias,
                            hk,
                            hu,
                            hg,
                        )
                else:
                    prims.mbarrier_arrive(mb0)
                    if cutlass.const_expr(ext_commit):
                        for _a in cutlass.range_constexpr(3):
                            prims.mbarrier_arrive(mid_mbar)
                        prims.mbarrier_arrive(late_mbar)
    if cutlass.const_expr(pdl):
        prims.griddepcontrol("wait")
    if tidx == 0:
        for st0 in cutlass.range_constexpr(stages):
            item0 = cta + st0 * ncta
            if item0 < n_items:
                base0 = st0 * META_WORDS
                mb0 = cutlass.Array(full_mbar.subview(st0).data_ptr(), shape=1, dtype=cutlass.Int64)
                if s_meta[base0 + 3] == 0:
                    batch0 = s_meta[base0 + 4]
                    head0 = s_meta[base0 + 5]
                    # (the projections' TMA first: the beta load's store would stall this thread)
                    beta0 = beta[(batch0, head0)]
                    if cutlass.const_expr(ext_commit):
                        prims.mbarrier_arrive_expect_tx(mb0, 1280)
                    _tp_issue_post(
                        s_stages.subview(st0 * SG_BYTES).data_ptr(), mb0, item0, heads, x, gate, z
                    )
                    s_meta[base0 + 8] = _f32_as_s32(beta0.to(cutlass.Float32))
                else:
                    if cutlass.const_expr(ext_commit):
                        prims.mbarrier_arrive(mb0)
                if cutlass.const_expr(ext_commit):
                    # (s_pre before this thread's final arrive: read after the first stage wait)
                    s_pre[3] = item2p
                    if item2p < n_items:
                        s_pre[0] = pre2_slot
                        s_pre[1] = pre2_rs
                        s_pre[2] = pre2_dmy
                    prims.mbarrier_arrive(mb0)
        if cutlass.const_expr(not ext_commit):
            s_pre[3] = item2p
            if item2p < n_items:
                s_pre[0] = pre2_slot
                s_pre[1] = pre2_rs
                s_pre[2] = pre2_dmy
    cute.arch.barrier()
    tmem_raw = tmem_holding.load()
    base_col = tmem_raw & cutlass.Int32(0xFFFF)
    base_row = tmem_raw >> cutlass.Int32(16)
    lane_addr = Uint32(((base_row + warp * cutlass.Int32(32)) << cutlass.Int32(16)) | base_col)
    mma_count = cutlass.Int32(0)
    commit_count = cutlass.Int32(0)
    # Work distribution: items cta + k*ncta (k <= stages + 1) are static, later ones are claimed
    # from a global counter (sched[0]) two items ahead, so commits (3-4x slower items) do not unbalance
    # the CTAs. Every CTA that claims stops after its first failing claim, so a launch makes exactly
    # n_claims claims; the CTA that receives the last value resets the counter for the next call.
    sched_addr = _addr(sched, 0)
    n_static = cutlass.max((n_items // ncta) * _TP_STATIC_PCT // 100, cutlass.Int32(stages + 2))
    base_dyn = n_static * ncta
    n_claiming = cutlass.min(cutlass.max(n_items - (n_static - 1) * ncta, cutlass.Int32(0)), ncta)
    n_claims = cutlass.max(n_items - base_dyn, cutlass.Int32(0)) + n_claiming
    itq0 = cta
    itq1 = cta + ncta
    jj = cutlass.Int32(0)
    item2_t0 = cta + stages * ncta
    item3_t0 = cta + (stages + 1) * ncta
    pos3 = cutlass.Int32(stages + 1)  # static position of item3 (thread 0)
    # Runtime (not constant) predicate: a constant one lets ptxas warp-aggregate the claim atomic,
    # which shuffles its result right away and stalls warp 0 for the full round trip.
    t0flag = cutlass.Int32(0)
    if tidx == 0:
        t0flag = cutlass.Int32(1)
    while itq0 < n_items:
        j = jj
        st = j % stages
        phase = (j // stages) % 2
        sbase = s_stages.subview(st * SG_BYTES).data_ptr()
        mbase = st * META_WORDS
        mb = cutlass.Array(full_mbar.subview(st).data_ptr(), shape=1, dtype=cutlass.Int64)
        # Metadata pipeline (thread 0): loads issued now, consumed at the end of this item.
        item2 = item2_t0
        item3 = item3_t0
        p_slot = s_pre[0]
        p_rs = s_pre[1]
        p_dmy = s_pre[2]
        batch2 = item2 // heads
        head2 = item2 - batch2 * heads
        len2 = cutlass.Int32(0)
        outer2 = cutlass.Int32(0)
        alog2 = cutlass.Int32(0)
        beta2 = cutlass.Int32(0)
        n3_slot = cutlass.Int32(0)
        n3_rs = cutlass.Int32(0)
        n3_dmy = cutlass.Int32(0)
        if tidx == 0:
            if item2 < n_items:
                if p_dmy == 0:
                    len2 = _ld_keep_s32(
                        _addr(lengths, cute.crd2idx((cutlass.Int64(p_rs),), lengths.layout))
                    )
                    outer2 = _ld_global_s32(
                        Int64(
                            (
                                pool.iterator
                                + cute.crd2idx((p_slot, head2, 0), pool.layout)
                                + OUTER_SCALE
                            ).toint()
                        )
                    )
                    alog2 = _ld_global_s32(_addr(a_log, head2))
                    beta2 = _ld_u16_s32(_addr(beta, cute.crd2idx((batch2, head2), beta.layout)))
            if item3 < n_items:
                batch3 = item3 // heads
                n3_slot = _ld_keep_s32(_addr(slots, batch3))
                n3_rs = _ld_keep_s32(_addr(replay_indices, batch3))
                n3_dmy = _ld_keep_u8(_addr(dummy, batch3))
        # Claim the item after item3 (thread 0); the result is first used at the end of this item.
        claim = cutlass.Int32(0)
        # (only once the static items are used up: item3 is the last static one or a claimed one)
        if cutlass.const_expr(dyn):
            do_claim = cutlass.select_(pos3 >= n_static - 1, t0flag, cutlass.Int32(0))
            claim = _atom_inc_if(
                cutlass.select_(item3 < n_items, do_claim, cutlass.Int32(0)), sched_addr
            )
        while not prims.mbarrier_try_wait_parity(mb, phase):
            pass
        # Stage metadata: written by thread 0 before its arrive on this barrier (release/acquire).
        dmy = s_meta[mbase + 3]
        batch = s_meta[mbase + 4]
        head = s_meta[mbase + 5]
        # Next item (single-stage early refill): s_pre is also written before that arrive.
        e_item = s_pre[3]
        e_slot = s_pre[0]
        e_live = e_item < n_items
        e_go = e_live and (s_pre[2] == 0)
        e_head = e_item - (e_item // heads) * heads
        if dmy != 0:
            out[(batch, head, tidx)] = cutlass.BFloat16(0.0)
            if cutlass.const_expr(ext_commit):
                if tidx == 0:
                    if head == 0:
                        _st_keep_s32(_addr(cflags, batch), cutlass.Int32(0))
            if cutlass.const_expr(ext_commit):
                # (a padding item reads nothing from the stage: every early part of the next item now)
                if e_live:
                    if prims.elect_sync():
                        _tp_refill_b1(
                            warp,
                            sbase,
                            mb,
                            e_item,
                            heads,
                            e_slot,
                            conv_state,
                            conv_weight,
                            dt_bias,
                            x,
                            gate,
                            z,
                            e_go,
                        )
                        if warp == 2:
                            _tp_refill_mid(
                                0,
                                sbase,
                                mid_mbar,
                                tma_state,
                                pool,
                                hk,
                                hg,
                                e_head,
                                e_slot,
                                cutlass.Int64(0),
                                cutlass.Int32(0),
                                e_go,
                            )
                        if warp == 0:
                            _tp_refill_mid(
                                1,
                                sbase,
                                mid_mbar,
                                tma_state,
                                pool,
                                hk,
                                hg,
                                e_head,
                                e_slot,
                                cutlass.Int64(0),
                                cutlass.Int32(0),
                                e_go,
                            )
                    if tidx == 0:
                        _tp_refill_mid(
                            2,
                            sbase,
                            mid_mbar,
                            tma_state,
                            pool,
                            hk,
                            hg,
                            e_head,
                            e_slot,
                            cutlass.Int64(p_rs),
                            len2,
                            e_go,
                        )
        else:
            slot = s_meta[mbase + 0]
            replay_slot = cutlass.Int64(s_meta[mbase + 1])
            length = s_meta[mbase + 2]
            state_outer = _s32_as_f32(s_meta[mbase + 6])
            a_log_value = _s32_as_f32(s_meta[mbase + 7])
            beta_f = _s32_as_f32(s_meta[mbase + 8])
            commits = length == N - 1
            # ext_commit: the window commit kernel folds the window after this launch.
            if cutlass.const_expr(ext_commit):
                do_commit = False
            else:
                do_commit = commits
            pool_head = pool.iterator + cute.crd2idx((slot, head, 0), pool.layout)
            if cutlass.const_expr(ext_commit and l2pf):
                if commits:
                    if tidx == 32:
                        # The commit kernel reads this window's residual term next: warm L2 now.
                        _prefetch_l2(
                            Int64(pool_head.toint()) + VALUES_BYTES, STATE_BYTES - VALUES_BYTES
                        )
            s_ck = _arr8(sbase, SG_CK, 8192)
            s_hk = cutlass.Array(sbase + SG_HK, shape=N * K, dtype=cutlass.BFloat16)
            s_hu = cutlass.Array(sbase + SG_HU, shape=N * K, dtype=cutlass.BFloat16)
            s_hg = cutlass.Array(sbase + SG_HG, shape=N * K, dtype=cutlass.Float32)
            s_cv = cutlass.Array(sbase + SG_CONV, shape=9 * K, dtype=cutlass.BFloat16)
            s_w = cutlass.Array(sbase + SG_W, shape=12 * K, dtype=cutlass.BFloat16)
            s_dt = cutlass.Array(sbase + SG_DT, shape=K, dtype=cutlass.Float32)
            s_x = cutlass.Array(sbase + SG_X, shape=3 * K, dtype=cutlass.BFloat16)
            s_g = cutlass.Array(sbase + SG_GATE, shape=K, dtype=cutlass.BFloat16)
            s_z = cutlass.Array(sbase + SG_Z, shape=K, dtype=cutlass.BFloat16)

            # Gate step (bit-identical contraction with the baseline).
            a = cute.exp(a_log_value, fastmath=True)
            shifted = s_g[tidx].to(cutlass.Float32) + s_dt[tidx]
            step_scale = -a
            step_value = shifted
            if bounded:
                step_scale = lower_bound
                step_value = cute.div(
                    cutlass.Float32(1.0),
                    cutlass.Float32(1.0) + cute.exp(-(a * shifted), fastmath=True),
                    full=True,
                )
            else:
                step_value = cutlass.select_(
                    shifted <= cutlass.Float32(20.0),
                    cute.log(
                        cutlass.Float32(1.0) + cute.exp(shifted, fastmath=True), fastmath=True
                    ),
                    shifted,
                )
            zg = _rcp_approx(
                cutlass.Float32(1.0) + cute.exp(-(s_z[tidx].to(cutlass.Float32)), fastmath=True)
            )
            beta_sigmoid = _rcp_approx(cutlass.Float32(1.0) + cute.exp(-beta_f, fastmath=True))
            # Conv4 + SiLU.
            wq = (s_w.data_ptr() + tidx * 4).load(count=4, alignment=8).bitcast(Uint32)
            wk = (s_w.data_ptr() + 512 + tidx * 4).load(count=4, alignment=8).bitcast(Uint32)
            wv = (s_w.data_ptr() + 1024 + tidx * 4).load(count=4, alignment=8).bitcast(Uint32)
            cv_addr = _smem_u32(s_cv.data_ptr()) + tidx * 6
            x_addr = _smem_u32(s_x.data_ptr()) + tidx * 2
            q0, q1, q2 = (
                _ld_shared_u16(cv_addr),
                _ld_shared_u16(cv_addr + 2),
                _ld_shared_u16(cv_addr + 4),
            )
            k0, k1, k2 = (
                _ld_shared_u16(cv_addr + 6 * K),
                _ld_shared_u16(cv_addr + 6 * K + 2),
                _ld_shared_u16(cv_addr + 6 * K + 4),
            )
            v0, v1, v2 = (
                _ld_shared_u16(cv_addr + 12 * K),
                _ld_shared_u16(cv_addr + 12 * K + 2),
                _ld_shared_u16(cv_addr + 12 * K + 4),
            )
            raw_q, raw_k, raw_v = (
                _ld_shared_u16(x_addr),
                _ld_shared_u16(x_addr + 2 * K),
                _ld_shared_u16(x_addr + 4 * K),
            )
            aq = _conv4_bf16(wq[0], wq[1], q0, q1, q2, raw_q)
            ak = _conv4_bf16(wk[0], wk[1], k0, k1, k2, raw_k)
            av = _conv4_bf16(wv[0], wv[1], v0, v1, v2, raw_v)
            aq = aq * _rcp_approx(cutlass.Float32(1.0) + cute.exp(-aq, fastmath=True))
            ak = ak * _rcp_approx(cutlass.Float32(1.0) + cute.exp(-ak, fastmath=True))
            av = av * _rcp_approx(cutlass.Float32(1.0) + cute.exp(-av, fastmath=True))
            qf = aq.to(cutlass.BFloat16).to(cutlass.Float32)
            kf = ak.to(cutlass.BFloat16).to(cutlass.Float32)
            v_bf16 = av.to(cutlass.BFloat16)
            q_partial = _warp_sum_squares(qf)
            k_partial = _warp_sum_squares(kf)
            if lane == 0:
                s_red[8 + warp] = q_partial
                s_red[12 + warp] = k_partial
            cute.arch.barrier()  # B1
            # Single stage: refill the next item's parts as soon as this item has consumed them.
            if cutlass.const_expr(ext_commit):
                if e_live:
                    if prims.elect_sync():
                        _tp_refill_b1(
                            warp,
                            sbase,
                            mb,
                            e_item,
                            heads,
                            e_slot,
                            conv_state,
                            conv_weight,
                            dt_bias,
                            x,
                            gate,
                            z,
                            e_go,
                        )
                # this item's scales, checkpoint image and K/G rows (previous G is read next)
                while not prims.mbarrier_try_wait_parity(mid_mbar, phase):
                    pass

            qr = (s_red[8] + s_red[10]) + (s_red[9] + s_red[11])
            kr = (s_red[12] + s_red[14]) + (s_red[13] + s_red[15])
            qden = cute.sqrt(qr + cutlass.Float32(1.0e-6), approx=True, ftz=True)
            kden = cute.sqrt(kr + cutlass.Float32(1.0e-6), approx=True, ftz=True)
            qn = _div_n(qf * cutlass.Float32(0.08838834764831845), qden).to(cutlass.BFloat16)
            kn = _div_n(kf, kden).to(cutlass.BFloat16)
            s_q[tidx] = qn
            s_k[tidx] = kn
            prev_g = cutlass.Float32(0.0)
            if length > 0:
                prev_g = s_hg[(length - 1) * K + tidx]
            cumulative = _fma_rn(step_scale, step_value, prev_g)
            s_cum[tidx] = cumulative
            decay = cute.exp(cumulative, fastmath=True)
            if cutlass.const_expr(not ext_commit):
                s_od[tidx] = decay * state_outer
            knf = kn.to(cutlass.Float32)
            qnf = qn.to(cutlass.Float32)
            dyn_k = knf * decay
            dyn_q = qnf * decay
            group_max_k, wmax_k = _grp16_absmax(dyn_k, lane)
            group_max_q, wmax_q = _grp16_absmax(dyn_q, lane)
            wmax = cute.arch.fmax(wmax_k, wmax_q)
            if lane == 0:
                s_red[warp] = wmax
            cute.arch.barrier()  # B2

            mr = cute.arch.fmax(
                cute.arch.fmax(s_red[0], s_red[1]), cute.arch.fmax(s_red[2], s_red[3])
            )
            outer = mr * cutlass.Float32(1.0 / (6.0 * 448.0))
            safe_outer = _sel_eq0(outer, cutlass.Float32(1.0), outer)
            sfk_raw = _div_n(group_max_k, cutlass.Float32(6.0) * safe_outer)
            sfq_raw = _div_n(group_max_q, cutlass.Float32(6.0) * safe_outer)
            sfk_e4 = cute.arch.fmin(sfk_raw, cutlass.Float32(448.0)).to(cutlass.Float8E4M3FN)
            sfq_e4 = cute.arch.fmin(sfq_raw, cutlass.Float32(448.0)).to(cutlass.Float8E4M3FN)
            if (lane % 16) == 0:
                group = warp * 2 + lane // 16
                sf_offset_k = (group // 4) * 512 + (group % 4)
                mma_sfb.subview(sf_offset_k).store(sfk_e4)
                mma_sfb.subview(sf_offset_k + 16).store(sfq_e4)
            sfk = sfk_e4.to(cutlass.Float32)
            sfq = sfq_e4.to(cutlass.Float32)
            denom_k = sfk * outer
            denom_q = sfq * outer
            normalized_k = _sel_eq0(denom_k, cutlass.Float32(0.0), _div_n(dyn_k, denom_k))
            normalized_q = _sel_eq0(denom_q, cutlass.Float32(0.0), _div_n(dyn_q, denom_q))
            swizzle_s64 = cutlass.Swizzle.from_name("s64b")
            e2m1_type = mlir_ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type)
            odd_k = cute.arch.shuffle_sync_down(normalized_k, 1)
            odd_q = cute.arch.shuffle_sync_down(normalized_q, 1)
            if (lane % 2) == 0:
                packed_k_ir = prims.dialect.convert_f32x2_to_f4x2(odd_k, normalized_k, e2m1_type)
                packed_q_ir = prims.dialect.convert_f32x2_to_f4x2(odd_q, normalized_q, e2m1_type)
                byte = tidx // 2
                (mma_b.subview(byte)).data_ptr().apply_swizzle(swizzle_s64).store(
                    cutlass.Int8(packed_k_ir)
                )
                (mma_b.subview(64 + byte)).data_ptr().apply_swizzle(swizzle_s64).store(
                    cutlass.Int8(packed_q_ir)
                )
            sfw = (
                cutlass.Array(sbase + SG_SF, shape=256, dtype=cutlass.Int32).data_ptr() + tidx * 2
            ).load(count=2, alignment=8)
            for kblock in cutlass.range_constexpr(2):
                sf_offset = kblock * 512 + (tidx % 32) * 16 + (tidx // 32) * 4
                cutlass.inttoptr(
                    mma_sfa.subview(sf_offset).data_ptr().toint(),
                    cutlass.AddressSpace.smem,
                    cutlass.Int32,
                ).store(cutlass.Vector.from_elements((sfw[kblock],), dtype=cutlass.Int32))
            # k.q dot partials (needed after the MMA): off the B1-B2 critical path
            wdot = _warp_sum_products(knf, qnf)
            if lane == 0:
                s_red[4 + warp] = wdot
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()  # B3
            kq_dot = (s_red[4] + s_red[6]) + (s_red[5] + s_red[7])
            if cutlass.const_expr(ext_commit):
                if e_live:
                    if warp == 2:
                        if prims.elect_sync():
                            _tp_refill_mid(
                                0,
                                sbase,
                                mid_mbar,
                                tma_state,
                                pool,
                                hk,
                                hg,
                                e_head,
                                e_slot,
                                cutlass.Int64(0),
                                cutlass.Int32(0),
                                e_go,
                            )

            if warp == 3:
                idesc = prims.Tcgen05MxOmmaInstrDesc.build(
                    a_dtype=cutlass.Float4E2M1FN,
                    b_dtype=cutlass.Float4E2M1FN,
                    scale_format=0,
                    n_dim=8,
                    m_dim=128,
                )
                desc_a = prims.Tcgen05SmemDesc.build(
                    s_ck,
                    leading_byte_offset=0,
                    stride_byte_offset=512,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                )
                desc_b = prims.Tcgen05SmemDesc.build(
                    mma_b,
                    leading_byte_offset=0,
                    stride_byte_offset=512,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                )
                desc_sfa = prims.Tcgen05SmemDesc.build(
                    mma_sfa, leading_byte_offset=16, stride_byte_offset=128, layout=0
                )
                desc_sfb = prims.Tcgen05SmemDesc.build(
                    mma_sfb, leading_byte_offset=16, stride_byte_offset=128, layout=0
                )
                acc_addr = (base_row << cutlass.Int32(16)) | base_col
                sfa_addr = (base_row << cutlass.Int32(16)) | (base_col + cutlass.Int32(8))
                sfb_addr = (base_row << cutlass.Int32(16)) | (base_col + cutlass.Int32(16))
                acc_ptr = cutlass.inttoptr(acc_addr, 6, cutlass.Float32)
                sf_shape, sf_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
                if prims.elect_sync():
                    for kblock in cutlass.range_constexpr(2):
                        sfa_ptr = cutlass.inttoptr(sfa_addr + kblock * 4, 6, cutlass.Int32)
                        sfb_ptr = cutlass.inttoptr(sfb_addr + kblock * 4, 6, cutlass.Int32)
                        prims.tcgen05_cp(
                            sf_shape,
                            sfa_ptr,
                            desc_sfa.advance_start_address(kblock * 512),
                            multicast=sf_multicast,
                        )
                        prims.tcgen05_cp(
                            sf_shape,
                            sfb_ptr,
                            desc_sfb.advance_start_address(kblock * 512),
                            multicast=sf_multicast,
                        )
                    for kblock in cutlass.range_constexpr(2):
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF4NVF4,
                            prims.CTAGroup.CTA_1,
                            acc_ptr,
                            desc_a.advance_start_address(kblock * 32),
                            desc_b.advance_start_address(kblock * 32),
                            idesc,
                            enable_input_d=(kblock != 0),
                            scale_a=cutlass.inttoptr(sfa_addr + kblock * 4, 6, cutlass.Int32),
                            scale_b=cutlass.inttoptr(sfb_addr + kblock * 4, 6, cutlass.Int32),
                            scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                        )
                    prims.tcgen05_commit(acc_mbar, group=prims.CTAGroup.CTA_1)

            # Global stores after the async-proxy fence.
            state_raw = conv_state.iterator.raw_ptr()
            sq_a = Int64(
                (state_raw + cute.crd2idx((slot, head * K + tidx, 0), conv_state.layout)).toint()
            )
            sk_a = sq_a + Int64(cute.crd2idx((0, width, 0), conv_state.layout)) * 2
            sv_a = sk_a + Int64(cute.crd2idx((0, width, 0), conv_state.layout)) * 2
            is_t0 = cutlass.Int32(0)
            if tidx == 0:
                is_t0 = cutlass.Int32(1)
            arrived = cutlass.Int32(0)
            for sa, w1_, w2_, w3_ in (
                (sq_a, q1, q2, raw_q),
                (sk_a, k1, k2, raw_k),
                (sv_a, v1, v2, raw_v),
            ):
                _st_global_u16_ef(sa, w1_)
                _st_global_u16_ef(sa + 2, w2_)
                _st_global_u16_ef(sa + 4, w3_)
            hk[(replay_slot, head, length, tidx)] = kn
            hg[(replay_slot, head, length, tidx)] = cumulative
            arrived = _atom_add_if(is_t0, _addr(arrivals, batch), cutlass.Int32(1))

            # History coefficients (conflict-free: warp w rows 4w..4w+3, lane l keys 4l..4l+3).
            key0 = lane * 4
            adw = [Uint32(0) for _ in range(8)]
            if warp * 4 < length:
                cum4 = (s_cum.data_ptr() + key0).load(count=4, alignment=16)
                kn4 = (s_k.data_ptr() + key0).load(count=4, alignment=8).bitcast(Uint32)
                qn4 = (s_q.data_ptr() + key0).load(count=4, alignment=8).bitcast(Uint32)
                partial = []
                for r in cutlass.range_constexpr(4):
                    row = warp * 4 + r
                    g4 = (s_hg.data_ptr() + row * K + key0).load(count=4, alignment=16)
                    k4 = (
                        (s_hk.data_ptr() + row * K + key0)
                        .load(count=4, alignment=8)
                        .bitcast(Uint32)
                    )
                    ck = list(_bf16x2_lo_hi(k4[0])) + list(_bf16x2_lo_hi(k4[1]))
                    e = [cute.exp(cum4[jj] - g4[jj], fastmath=True) for jj in range(4)]
                    a01 = _pack_bf16x2(ck[0] * e[0], ck[1] * e[1])
                    a23 = _pack_bf16x2(ck[2] * e[2], ck[3] * e[3])
                    adw[2 * r] = a01
                    adw[2 * r + 1] = a23
                    pk = _dot4_bf16(a01, a23, kn4[0], kn4[1], cutlass.Float32(0.0))
                    pq = _dot4_bf16(a01, a23, qn4[0], qn4[1], cutlass.Float32(0.0))
                    valid = row < length
                    partial.append(cutlass.select_(valid, pk, cutlass.Float32(0.0)))
                    partial.append(cutlass.select_(valid, pq, cutlass.Float32(0.0)))
                vals = partial
                width_ = 8
                for offset in (16, 8, 4):
                    half = width_ // 2
                    upper = (lane & offset) != 0
                    nxt = []
                    for i in cutlass.range_constexpr(half):
                        send = cutlass.select_(upper, vals[i], vals[half + i])
                        keep = cutlass.select_(upper, vals[half + i], vals[i])
                        nxt.append(keep + cute.arch.shuffle_sync_bfly(send, offset))
                    vals = nxt
                    width_ = half
                total = vals[0]
                total = total + cute.arch.shuffle_sync_bfly(total, 2)
                total = total + cute.arch.shuffle_sync_bfly(total, 1)
                sel = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1)
                if (lane & 3) == 0:
                    # s_cpair[row] = (k, q) coefficients as two BF16 halves.
                    _st_shared_u16(
                        _smem_u32(s_cpair.data_ptr()) + (warp * 4 + sel // 2) * 4 + (sel % 2) * 2,
                        _pack_bf16x2(total, total),
                    )
            adjs = [(adw[2 * r], adw[2 * r + 1]) for r in range(4)]
            cute.arch.barrier()  # B4
            if cutlass.const_expr(ext_commit):
                if tidx == 0:
                    if e_live:
                        _tp_refill_mid(
                            2,
                            sbase,
                            mid_mbar,
                            tma_state,
                            pool,
                            hk,
                            hg,
                            e_head,
                            e_slot,
                            cutlass.Int64(p_rs),
                            len2,
                            e_go,
                        )
            if cutlass.const_expr(not ext_commit):
                if do_commit:
                    if tidx == 0:
                        prims.mbarrier_arrive_expect_tx(res_mbar, 9216)
                        res_src = cute.recast_ptr(
                            pool_head + VALUES_BYTES, dtype=cutlass.Int8
                        ).raw_ptr()
                        _bulk_g_tp(_arr8(sbase, SG_HK, 4096), res_src, res_mbar, 4096)
                        _bulk_g_tp(_arr8(sbase, SG_CONV, 4096), res_src + 4096, res_mbar, 4096)
                        _bulk_g_tp(
                            _arr8(sbase, SG_SF, 1024),
                            cute.recast_ptr(
                                pool_head + RESIDUAL_SCALE, dtype=cutlass.Int8
                            ).raw_ptr(),
                            res_mbar,
                            1024,
                        )
            hist_k = cutlass.Float32(0.0)
            hist_q = cutlass.Float32(0.0)
            if cutlass.const_expr(ext_commit):
                while not prims.mbarrier_try_wait_parity(late_mbar, phase):
                    pass
            for hist in cutlass.range(length, unroll=4):
                uw = _ld_shared_u16(_smem_u32(s_hu.data_ptr()) + (hist * K + tidx) * 2)
                cw2 = _ld_shared_u32(_smem_u32(s_cpair.data_ptr()) + hist * 4)
                hist_k, hist_q = _hist_fma(uw, cw2, hist_k, hist_q)

            while not prims.mbarrier_try_wait_parity(acc_mbar, mma_count % 2):
                pass
            mma_count = mma_count + 1
            if cutlass.const_expr(ext_commit):
                if e_live:
                    if warp == 0:
                        if prims.elect_sync():
                            _tp_refill_mid(
                                1,
                                sbase,
                                mid_mbar,
                                tma_state,
                                pool,
                                hk,
                                hg,
                                e_head,
                                e_slot,
                                cutlass.Int64(0),
                                cutlass.Int32(0),
                                e_go,
                            )
            checkpoint_vec = prims.tcgen05_ld(
                "32x32b", cutlass.inttoptr(Int32(lane_addr), 6, cutlass.Float32), num=8
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            checkpoint_rescale = state_outer * outer
            cp_k = checkpoint_vec[0] * checkpoint_rescale
            cp_q = checkpoint_vec[1] * checkpoint_rescale
            update = beta_sigmoid * (v_bf16.to(cutlass.Float32) - cp_k - hist_k)
            core = cp_q + hist_q + update * kq_dot
            core_bf16 = core.to(cutlass.BFloat16)
            core_f = core_bf16.to(cutlass.Float32)
            rms_partial = _warp_sum_squares(core_f)
            if lane == 0:
                s_red[8 + warp] = rms_partial
            if cutlass.const_expr(not ext_commit):
                _st_shared_u32(_smem_u32(s_u15.data_ptr()) + tidx * 4, _pack_bf16x2(update, update))
            cute.arch.barrier()  # B5
            total2 = (s_red[8] + s_red[10]) + (s_red[9] + s_red[11])
            rstd = _rsqrt_approx(total2 * cutlass.Float32(1.0 / 128.0) + norm_eps)
            result = core_f * rstd * norm_bf16.to(cutlass.Float32) * zg
            out[(batch, head, tidx)] = result.to(cutlass.BFloat16)
            hu[(replay_slot, head, length, tidx)] = update.to(cutlass.BFloat16)
            if tidx == 0:
                if arrived == cutlass.Int32(heads - 1):
                    _st_keep_s32(_addr(arrivals, batch), cutlass.Int32(0))
                    next_length = length + 1
                    if commits:
                        next_length = cutlass.Int32(0)
                    _st_keep_s32(
                        _addr(lengths, cute.crd2idx((replay_slot,), lengths.layout)), next_length
                    )
                    if cutlass.const_expr(ext_commit):
                        # Per-row commit flag for the window commit kernel (written every launch).
                        _st_keep_s32(
                            _addr(cflags, batch),
                            cutlass.select_(commits, cutlass.Int32(1), cutlass.Int32(0)),
                        )
            if cutlass.const_expr(not ext_commit):
                if do_commit:
                    # Panels in the stage's G rows (no longer needed): A = U^T (row 15 zero), B = adj (row 15 zero).
                    panel_a = cutlass.Array(sbase + SG_HG, shape=PANEL_BYTES, dtype=cutlass.Int8)
                    panel_b = cutlass.Array(
                        sbase + SG_HG + PANEL_BYTES, shape=PANEL_BYTES, dtype=cutlass.Int8
                    )
                    # Transpose the 4 rows x 4 keys BF16 block: key jk gets rows (0,1),(2,3) as two words.
                    for jk in cutlass.range_constexpr(4):
                        words = [adjs[r][jk // 2] for r in range(4)]
                        sh = 16 * (jk % 2)
                        lo01 = ((words[0] >> Uint32(sh)) & Uint32(0xFFFF)) | (
                            ((words[1] >> Uint32(sh)) & Uint32(0xFFFF)) << Uint32(16)
                        )
                        r3 = cutlass.select_(
                            warp == 3, Uint32(0), (words[3] >> Uint32(sh)) & Uint32(0xFFFF)
                        )
                        lo23 = ((words[2] >> Uint32(sh)) & Uint32(0xFFFF)) | (r3 << Uint32(16))
                        off = _panel_offset(key0 + jk, warp // 2) + (warp % 2) * 8
                        _st_shared_v2(_smem_u32(panel_b.data_ptr()) + off, lo01, lo23)
                    uv = [s_hu[i * K + tidx].to(cutlass.Float32) for i in range(N - 1)] + [
                        cutlass.Float32(0.0)
                    ]
                    for chunk in cutlass.range_constexpr(2):
                        _st_shared_v4(
                            _smem_u32(panel_a.data_ptr()) + _panel_offset(tidx, chunk),
                            _pack_bf16x2(uv[8 * chunk], uv[8 * chunk + 1]),
                            _pack_bf16x2(uv[8 * chunk + 2], uv[8 * chunk + 3]),
                            _pack_bf16x2(uv[8 * chunk + 4], uv[8 * chunk + 5]),
                            _pack_bf16x2(uv[8 * chunk + 6], uv[8 * chunk + 7]),
                        )
                    # Residual rows 0-63 in the K region, rows 64-127 in the Conv/x tail, scales in the SF region.
                    while not prims.mbarrier_try_wait_parity(res_mbar, commit_count % 2):
                        pass
                    res_off = cutlass.select_(
                        tidx < 64, cutlass.Int32(SG_HK), cutlass.Int32(SG_CONV - 4096)
                    )
                    _seed_commit_t(
                        tidx,
                        lane_addr,
                        s_ck,
                        sbase,
                        res_off,
                        _arr8(sbase, SG_SF, 1024),
                        mma_sfa,
                        s_od,
                    )
                    _fence_proxy_async_smem()
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    cute.arch.barrier()
                    if warp == 0:
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        idesc_c = prims.Tcgen05InstrDesc.build(
                            c_dtype=Float32,
                            a_dtype=cutlass.BFloat16,
                            b_dtype=cutlass.BFloat16,
                            n_dim=K,
                            m_dim=K,
                        )
                        desc_pa = prims.Tcgen05SmemDesc.build(
                            panel_a,
                            leading_byte_offset=PANEL_LBO,
                            stride_byte_offset=PANEL_SBO,
                            layout=prims.Tcgen05SmemSwizzle.NONE,
                        )
                        desc_pb = prims.Tcgen05SmemDesc.build(
                            panel_b,
                            leading_byte_offset=PANEL_LBO,
                            stride_byte_offset=PANEL_SBO,
                            layout=prims.Tcgen05SmemSwizzle.NONE,
                        )
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                cutlass.inttoptr(tmem_raw, 6, Float32),
                                desc_pa,
                                desc_pb,
                                idesc_c,
                                True,
                            )
                            prims.tcgen05_commit(commit_mbar, group=prims.CTAGroup.CTA_1)
                    _commit_quantize_all(
                        tidx,
                        lane,
                        warp,
                        tidx,
                        lane_addr,
                        Int64(pool_head.toint()),
                        s_u15,
                        s_k,
                        s_cmax,
                        commit_mbar,
                        commit_count % 2,
                    )
                    commit_count = commit_count + 1
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        # ---- end of item: publish item+2G's metadata, then refill stage st (split over the warps) ----
        # (this CTA barrier also bounds the warps' skew to one stage-barrier phase: without it, fast warps
        # crossing consecutive padding items can lap a slow warp's parity wait -> deadlock; it also orders
        # this item's s_red reads before the next item's writes)
        nb = (j % 2) * META_WORDS
        if tidx == 0:
            s_next[nb + 9] = item2
            if item2 < n_items:
                s_next[nb + 0] = p_slot
                s_next[nb + 1] = p_rs
                s_next[nb + 2] = len2
                s_next[nb + 3] = p_dmy
                s_next[nb + 6] = outer2
                s_next[nb + 7] = alog2
                s_next[nb + 8] = beta2 << cutlass.Int32(16)
        cute.arch.barrier()
        if tidx == 0:
            # (published before this thread's arrive below, i.e. visible after the next wait)
            s_pre[0] = n3_slot
            s_pre[1] = n3_rs
            s_pre[2] = n3_dmy
            s_pre[3] = item3
        item2_all = s_next[nb + 9]
        if item2_all < n_items:
            r_slot = s_next[nb + 0]
            r_rs = s_next[nb + 1]
            r_len = s_next[nb + 2]
            r_dmy = s_next[nb + 3]
            batch2a = item2_all // heads
            head2a = item2_all - batch2a * heads
            if tidx == 0:
                s_meta[mbase + 0] = r_slot
                s_meta[mbase + 1] = r_rs
                s_meta[mbase + 2] = r_len
                s_meta[mbase + 3] = r_dmy
                s_meta[mbase + 4] = batch2a
                s_meta[mbase + 5] = head2a
                s_meta[mbase + 6] = s_next[nb + 6]
                s_meta[mbase + 7] = s_next[nb + 7]
                s_meta[mbase + 8] = s_next[nb + 8]
            # (thread 0 is warp 0's elected lane, whose arrive_expect_tx releases the writes above)
            if cutlass.const_expr(ext_commit):
                # Early refill, last part: the U rows (read by the history loop), then the start
                # barrier's final arrival (after this thread's s_meta / s_pre writes).
                if tidx == 0:
                    if r_dmy != 0:
                        prims.mbarrier_arrive(late_mbar)
                    else:
                        prims.mbarrier_arrive_expect_tx(late_mbar, r_len * (K * 2))
                        if r_len > 0:
                            _bulk_g_rp(
                                _arr8(sbase, SG_HU, 4096),
                                hu.iterator.raw_ptr()
                                + cute.crd2idx((cutlass.Int64(r_rs), head2a, 0, 0), hu.layout),
                                late_mbar,
                                r_len * (K * 2),
                                _hist_pol(r_len),
                            )
                    prims.mbarrier_arrive(mb)
            else:
                if prims.elect_sync():
                    _tp_issue_warp(
                        warp,
                        tma_state,
                        sbase,
                        mb,
                        item2_all,
                        heads,
                        r_slot,
                        cutlass.Int64(r_rs),
                        r_len,
                        r_dmy,
                        pool,
                        conv_state,
                        conv_weight,
                        dt_bias,
                        hk,
                        hu,
                        hg,
                        x,
                        gate,
                        z,
                    )
        if cutlass.const_expr(dyn):
            if t0flag != 0:
                if item3 < n_items:
                    if pos3 >= n_static - 1:
                        if claim == n_claims - 1:
                            _st_keep_s32(sched_addr, cutlass.Int32(0))
        item2_t0 = item3
        if cutlass.const_expr(not dyn):
            item3_t0 = item3 + ncta
        else:
            item3_t0 = cutlass.select_(
                pos3 >= n_static - 1,
                cutlass.select_(item3 < n_items, base_dyn + claim, n_items),
                item3 + ncta,
            )
        pos3 = pos3 + 1
        if cutlass.const_expr(stages == 2):
            itq0 = itq1
            itq1 = item2_all
        else:
            itq0 = item2_all
        jj = jj + 1
    if cutlass.const_expr(pdl and ext_commit):
        # (dependents launch once every CTA is leaving)
        prims.griddepcontrol("launch_dependents")
    if warp == 3:
        prims.tcgen05_dealloc(
            cutlass.inttoptr(tmem_raw, 6, cutlass.Float32),
            cutlass.Int32((32 if ext_commit else 128)),
            group="cta_1",
        )


@cute.jit
def _launch_tp(
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    replay_indices,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    arrivals,
    sched,
    cflags,
    n_items: Int32,
    ncta: Int32,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    ext_commit: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    dyn: cutlass.Constexpr,
    l2pf: cutlass.Constexpr,
    cps: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    primary_layout = cute.make_layout(
        (64, 128, heads, pool.shape[0]),
        stride=(1, 64, STATE_BYTES, pool.layout.stride[0]),
    )
    primary_tensor = cute.make_tensor(pool.iterator, primary_layout)
    tma_state = cuda_exp.create_tensor_map_tiled_from_view(
        primary_tensor,
        box_dims=(64, 128, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_exp.TensorMapSwizzle.s64b,
    )
    _tp_kernel(
        tma_state,
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool,
        slots,
        dummy,
        replay_indices,
        hk,
        hu,
        hg,
        lengths,
        a_log,
        dt_bias,
        out,
        arrivals,
        sched,
        cflags,
        n_items,
        lower_bound,
        norm_eps,
        heads,
        bounded,
        pdl,
        ext_commit,
        stages,
        dyn,
        l2pf,
    ).launch(
        grid=[ncta, 1, 1], block=[128, 1, 1], stream=stream, use_pdl=pdl, min_blocks_per_mp=cps
    )


CS_STATE = (
    0  # primary values | residual values | primary scales | residual scales | outer (18448 B)
)
CS_HK = 18560
CS_HU = CS_HK + 4096
CS_HG = CS_HU + 4096
CS_BYTES = CS_HG + 8192
CS_TX = STATE_BYTES + 4096 + 4096 + 8192
CK_MAX_ROWS = 1024
CK_THREADS = 256


@dsl_user_op
def _ballot(pred: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int32(pred), loc, ip)],
            "{ .reg .pred p; setp.ne.s32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _popc(x: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Uint32(x), loc, ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lanemask_lt(*, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %lanemask_lt;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_v4u(addr: Int32, *, loc=None, ip=None):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(Int32(addr), loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)]


@cute.jit
def _ck_quantize(
    v, hh, lane, warp, tidx, tmem_row, head_base, s_cmax, halves: cutlass.Constexpr = False
):
    """Quantize keys 64hh..64hh+63 of TMEM row v into both checkpoint terms (shared outer scale).
    halves: the second pass reloads and quantizes 32 keys at a time (fewer live registers)."""
    if cutlass.const_expr(halves):
        bam = []
        for c in cutlass.range_constexpr(2):
            sn = _tmem_ld32(tmem_row + Uint32(64 * hh + 32 * c))
            bam = bam + [_tree_max([_f_abs(sn[16 * g + t]) for t in range(16)]) for g in range(2)]
    else:
        sn = _tmem_ld64(tmem_row + Uint32(64 * hh))
        bam = [_tree_max([_f_abs(sn[16 * g + t]) for t in range(16)]) for g in range(4)]
    m = _warp_max_f32(_tree_max(bam))
    if lane == 0:
        s_cmax[warp] = m
    cute.arch.barrier()
    gmax = _tree_max([s_cmax[w] for w in range(8)])
    outer_new = _f_div(gmax, Float32(2688.0))
    t6 = _f_mul(Float32(6.0), _sel_eq0(outer_new, Float32(1.0), outer_new))
    inv_t6 = _rcp_n(t6)
    psc = [_f_min(Float32(448.0), _f_mul(bam[g], inv_t6)) for g in range(4)]
    pscp = _pack_e4m3x4(psc[0], psc[1], psc[2], psc[3])
    row_base = head_base + Int64(v) * 64
    _st_global_u32(head_base + PRIMARY_SCALE + Int64(v) * 8 + 4 * hh, pscp)
    pe = list(_expand4(pscp, outer_new))
    if cutlass.const_expr(halves):
        rwh = []
        for q in cutlass.range_constexpr(2):
            sn2 = _tmem_ld32(tmem_row + Uint32(64 * hh + 32 * q))
            pw_h, rw_h, rword_h = _quant_pass2(sn2, pe[2 * q : 2 * q + 2], inv_t6, outer_new, 2)
            _st_global_v4(row_base + 32 * hh + 16 * q, pw_h[0], pw_h[1], pw_h[2], pw_h[3])
            _st_global_v4(
                row_base + VALUES_BYTES + 32 * hh + 16 * q, rw_h[0], rw_h[1], rw_h[2], rw_h[3]
            )
            rwh.append(rword_h & Uint32(0xFFFF))
        _st_global_u32(
            head_base + RESIDUAL_SCALE + Int64(v) * 8 + 4 * hh, rwh[0] | (rwh[1] << Uint32(16))
        )
    else:
        pw_all, rw_all, rword = _quant_pass2(sn, pe, inv_t6, outer_new)
        for q in cutlass.range_constexpr(2):
            _st_global_v4(
                row_base + 32 * hh + 16 * q,
                pw_all[4 * q],
                pw_all[4 * q + 1],
                pw_all[4 * q + 2],
                pw_all[4 * q + 3],
            )
            _st_global_v4(
                row_base + VALUES_BYTES + 32 * hh + 16 * q,
                rw_all[4 * q],
                rw_all[4 * q + 1],
                rw_all[4 * q + 2],
                rw_all[4 * q + 3],
            )
        _st_global_u32(head_base + RESIDUAL_SCALE + Int64(v) * 8 + 4 * hh, rword)
    if tidx == 0:
        _st_outer_pad(head_base + OUTER_SCALE, outer_new)


_DQ8_ASM = """
{
 .reg .b8 pb<4>, rb<4>;
 .reg .b32 ph<4>, rh<4>;
 .reg .f16 pq<8>, rq<8>, sp, sr;
 .reg .f32 t<8>;
 mov.b32 {pb0,pb1,pb2,pb3}, $8;
 mov.b32 {rb0,rb1,rb2,rb3}, $9;
 mov.b16 sp, $10;
 mov.b16 sr, $11;
"""
for _i in range(4):
    _DQ8_ASM += f" cvt.rn.f16x2.e2m1x2 ph{_i}, pb{_i};\n cvt.rn.f16x2.e2m1x2 rh{_i}, rb{_i};\n"
    _DQ8_ASM += f" mov.b32 {{pq{2 * _i},pq{2 * _i + 1}}}, ph{_i};\n mov.b32 {{rq{2 * _i},rq{2 * _i + 1}}}, rh{_i};\n"
for _i in range(8):
    _DQ8_ASM += f" fma.rn.f32.f16 t{_i}, rq{_i}, sr, 0f00000000;\n"
for _i in range(8):
    _DQ8_ASM += f" fma.rn.f32.f16 ${_i}, pq{_i}, sp, t{_i};\n"
_DQ8_ASM += "}\n"


@cute.jit
def _ck_issue(stage_ptr, mbar, pool, hk, hu, hg, slot, rs, head, mbar_h=None):
    """The full 16-entry history (first: the panels need it) and the state image of one (row, head)
    into a stage; with mbar_h the history completes on its own barrier. state_first: the image is
    copied first (its dequantization overlaps the history's arrival)."""
    mh = mbar
    if cutlass.const_expr(mbar_h is not None):
        mh = mbar_h
        prims.mbarrier_arrive_expect_tx(mh, 16384)
        prims.mbarrier_arrive_expect_tx(mbar, STATE_BYTES)
    else:
        prims.mbarrier_arrive_expect_tx(mbar, CS_TX)
    pool_head = pool.iterator + cute.crd2idx((slot, head, 0), pool.layout)
    coord = (rs, head, 0, 0)
    _bulk_g_ck(
        _arr8(stage_ptr, CS_HG, 8192),
        hg.iterator.raw_ptr() + cute.crd2idx(coord, hg.layout),
        mh,
        8192,
    )
    _bulk_g_ck(
        _arr8(stage_ptr, CS_HK, 4096),
        hk.iterator.raw_ptr() + cute.crd2idx(coord, hk.layout),
        mh,
        4096,
    )
    _bulk_g_ck(
        _arr8(stage_ptr, CS_HU, 4096),
        hu.iterator.raw_ptr() + cute.crd2idx(coord, hu.layout),
        mh,
        4096,
    )
    _bulk_g_ck(
        _arr8(stage_ptr, CS_STATE, STATE_BYTES),
        cute.recast_ptr(pool_head, dtype=cutlass.Int8).raw_ptr(),
        mbar,
        STATE_BYTES,
    )


@cute.kernel
def _commit_kernel(
    pool,
    slots,
    replay_indices,
    hk,
    hu,
    hg,
    cflags,
    batch_size: Int32,
    heads: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    nst: cutlass.Constexpr,
    early_refill: cutlass.Constexpr,
    halves: cutlass.Constexpr,
):
    """Persistent: CTA c folds commit items c, c+G, ... of the list (committed row, head).
    wide: 16 warps (one CTA per SM), each thread seeds / quantizes 32 keys of one row instead of 64."""
    NW = 8
    NT = 32 * NW
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    ncta, _, _ = cute.arch.grid_dim()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    s_stages = cutlass.Array(
        cutlass.Int8, nst * CS_BYTES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    panel_a = cutlass.Array(
        cutlass.BFloat16, PANEL_BYTES // 2, space=cutlass.AddressSpace.smem, alignment=128
    )
    panel_b = cutlass.Array(
        cutlass.BFloat16, PANEL_BYTES // 2, space=cutlass.AddressSpace.smem, alignment=128
    )
    s_od = cutlass.Array(Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_cmax = cutlass.Array(Float32, NW, space=cutlass.AddressSpace.smem, alignment=16)
    s_wcnt = cutlass.Array(Int32, NW, space=cutlass.AddressSpace.smem, alignment=16)
    s_slot = cutlass.Array(Int32, CK_MAX_ROWS, space=cutlass.AddressSpace.smem, alignment=16)
    s_rs = cutlass.Array(Int32, CK_MAX_ROWS, space=cutlass.AddressSpace.smem, alignment=16)
    full_mbar = cutlass.Array(cutlass.Int64, nst, space=cutlass.AddressSpace.smem, alignment=8)
    hist_mbar = cutlass.Array(cutlass.Int64, nst, space=cutlass.AddressSpace.smem, alignment=8)
    commit_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_holding = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    if warp == 0:
        if prims.elect_sync():
            for st0 in cutlass.range_constexpr(nst):
                prims.mbarrier_init(
                    cutlass.Array(full_mbar.subview(st0).data_ptr(), shape=1, dtype=cutlass.Int64),
                    1,
                )
                prims.mbarrier_init(
                    cutlass.Array(hist_mbar.subview(st0).data_ptr(), shape=1, dtype=cutlass.Int64),
                    1,
                )
            prims.mbarrier_init(commit_mbar, 1)
            prims.fence_mbarrier_init()
    # (the next CTA on this SM can only launch after this CTA's allocation permit is relinquished)
    if warp == 7:
        prims.tcgen05_alloc(tmem_holding, cutlass.Int32(128), group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    # Lengths and the sixteenth history entries come from the decode kernel.
    if cutlass.const_expr(pdl):
        prims.griddepcontrol("wait")

    # ---- committed rows (every CTA builds the same list), flagged by the decode kernel ----
    n_rows = Int32(0)
    for base in cutlass.range(0, batch_size, NT, unroll=1):
        b = base + tidx
        flag = Int32(0)
        slot_b = Int32(0)
        rs_b = Int32(0)
        if b < batch_size:
            flag = _ld_keep_s32(_addr(cflags, b))
            rs_b = _ld_keep_s32(_addr(replay_indices, b))
            slot_b = _ld_keep_s32(_addr(slots, b))
        bal = _ballot(flag)
        if lane == 0:
            s_wcnt[warp] = _popc(bal)
        cute.arch.barrier()
        before = Int32(0)
        total = Int32(0)
        for w in cutlass.range_constexpr(NW):
            cw = s_wcnt[w]
            total = total + cw
            before = before + cutlass.select_(w < warp, cw, Int32(0))
        pos = n_rows + before + _popc(bal & _lanemask_lt())
        if flag != 0:
            s_slot[pos] = slot_b
            s_rs[pos] = rs_b
        n_rows = n_rows + total
        cute.arch.barrier()
    n_items = n_rows * heads

    # Item rank of this CTA: with _CK_SPREAD, even CTAs take the first items (two CTAs per SM, so
    # a short item list puts one item per SM wherever the CTAs landed).
    crank = cta
    v = tidx % K
    hh = tidx // K
    tmem_raw = cutlass.Int32(0)
    tmem_row = Uint32(0)
    if tidx == 0:
        for st0 in cutlass.range_constexpr(nst):
            it0 = crank + st0 * ncta
            if it0 < n_items:
                i0 = it0 // heads
                h0 = it0 - i0 * heads
                _ck_issue(
                    s_stages.subview(st0 * CS_BYTES).data_ptr(),
                    cutlass.Array(full_mbar.subview(st0).data_ptr(), shape=1, dtype=cutlass.Int64),
                    pool,
                    hk,
                    hu,
                    hg,
                    s_slot[i0],
                    cutlass.Int64(s_rs[i0]),
                    h0,
                    cutlass.Array(hist_mbar.subview(st0).data_ptr(), shape=1, dtype=cutlass.Int64),
                )
    n_iter = Int32(0)
    if n_items > crank:
        n_iter = (n_items - crank + ncta - 1) // ncta
    for j in cutlass.range(n_iter, unroll=1):
        it = crank + j * ncta
        st = j % nst
        phase = (j // nst) % 2
        i = it // heads
        head = it - i * heads
        slot = s_slot[i]
        sbase = s_stages.subview(st * CS_BYTES).data_ptr()
        saddr = _smem_u32(sbase)
        mb = cutlass.Array(full_mbar.subview(st).data_ptr(), shape=1, dtype=cutlass.Int64)
        mbh = cutlass.Array(hist_mbar.subview(st).data_ptr(), shape=1, dtype=cutlass.Int64)
        head_base = Int64((pool.iterator + cute.crd2idx((slot, head, 0), pool.layout)).toint())
        hk_s = cutlass.Array(sbase + CS_HK, shape=N * K, dtype=cutlass.BFloat16)
        hu_s = cutlass.Array(sbase + CS_HU, shape=N * K, dtype=cutlass.BFloat16)
        hg_s = cutlass.Array(sbase + CS_HG, shape=N * K, dtype=Float32)
        while not prims.mbarrier_try_wait_parity(mbh, phase):
            pass
        cum_v = hg_s[(N - 1) * K + v]
        e_cum = _f_exp(cum_v)
        # Operand panels, rows 8hh..8hh+7: A = U^T, B = bf16(K * exp(G_end - G)) (row 15: kn, U_15).
        adj = []
        upd = []
        for t in cutlass.range_constexpr(8):
            r = 8 * hh + t
            adj.append(_f_mul(hk_s[r * K + v].to(Float32), _f_exp(cum_v - hg_s[r * K + v])))
            upd.append(hu_s[r * K + v].to(Float32))
        off = _panel_offset(v, hh)
        _st_shared_v4(
            _smem_u32(panel_b.data_ptr()) + off,
            _pack_bf16x2(adj[0], adj[1]),
            _pack_bf16x2(adj[2], adj[3]),
            _pack_bf16x2(adj[4], adj[5]),
            _pack_bf16x2(adj[6], adj[7]),
        )
        _st_shared_v4(
            _smem_u32(panel_a.data_ptr()) + off,
            _pack_bf16x2(upd[0], upd[1]),
            _pack_bf16x2(upd[2], upd[3]),
            _pack_bf16x2(upd[4], upd[5]),
            _pack_bf16x2(upd[6], upd[7]),
        )
        # (the state image follows the history)
        while not prims.mbarrier_try_wait_parity(mb, phase):
            pass
        if hh == 0:
            s_od[v] = _f_mul(e_cum, _s32_as_f32(Int32(_ld_shared_u32(saddr + OUTER_SCALE))))
        cute.arch.barrier()
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem_raw = tmem_holding.load()
        tmem_row = Uint32(
            (((tmem_raw >> Int32(16)) + (warp % 4) * Int32(32)) << Int32(16))
            | (tmem_raw & Int32(0xFFFF))
        )
        # Seed TMEM row v, keys 64hh..64hh+63: dequant(primary + residual) * outer * exp(G_end).
        psw = _ld_shared_u32(saddr + PRIMARY_SCALE + v * 8 + 4 * hh)
        rsw = _ld_shared_u32(saddr + RESIDUAL_SCALE + v * 8 + 4 * hh)
        sp = list(_scales4_f16(psw))
        sr = list(_scales4_f16(rsw))
        for cc in cutlass.range_constexpr(2):
            c = 2 * hh + cc
            pw = _ld_shared_v4u(saddr + v * 64 + c * 16)
            rw = _ld_shared_v4u(saddr + VALUES_BYTES + v * 64 + c * 16)
            seed = []
            for jj in cutlass.range_constexpr(4):
                od = (s_od.data_ptr() + 32 * c + 8 * jj).load(count=8, alignment=16)
                seed = seed + _seed8(
                    pw[jj],
                    rw[jj],
                    sp[2 * cc + jj // 2],
                    sr[2 * cc + jj // 2],
                    [od[t] for t in range(8)],
                )
            _tmem_st32_nw(tmem_row + Uint32(32 * c), seed)
        _tmem_wait_st()
        _fence_proxy_async_smem()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier()
        if cutlass.const_expr(early_refill):
            # (the stage's history and state image are consumed: panels and seed are built)
            if tidx == 32:
                it2e = it + nst * ncta
                if it2e < n_items:
                    i2e = it2e // heads
                    h2e = it2e - i2e * heads
                    _ck_issue(
                        sbase, mb, pool, hk, hu, hg, s_slot[i2e], cutlass.Int64(s_rs[i2e]), h2e, mbh
                    )
        if warp == 0:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=cutlass.BFloat16,
                b_dtype=cutlass.BFloat16,
                n_dim=K,
                m_dim=K,
            )
            desc_a = prims.Tcgen05SmemDesc.build(
                panel_a,
                leading_byte_offset=PANEL_LBO,
                stride_byte_offset=PANEL_SBO,
                layout=prims.Tcgen05SmemSwizzle.NONE,
            )
            desc_b = prims.Tcgen05SmemDesc.build(
                panel_b,
                leading_byte_offset=PANEL_LBO,
                stride_byte_offset=PANEL_SBO,
                layout=prims.Tcgen05SmemSwizzle.NONE,
            )
            if prims.elect_sync():
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F16,
                    prims.CTAGroup.CTA_1,
                    cutlass.inttoptr(tmem_raw, 6, Float32),
                    desc_a,
                    desc_b,
                    idesc,
                    True,
                )
                prims.tcgen05_commit(commit_mbar, group=prims.CTAGroup.CTA_1)
        while not prims.mbarrier_try_wait_parity(commit_mbar, j % 2):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

        _ck_quantize(v, hh, lane, warp, tidx, tmem_row, head_base, s_cmax, halves)
        # TMEM, panels and the stage are free once every thread is here.
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier()
        if cutlass.const_expr(not early_refill):
            if tidx == 0:
                it2 = it + nst * ncta
                if it2 < n_items:
                    i2 = it2 // heads
                    h2_ = it2 - i2 * heads
                    _ck_issue(
                        sbase, mb, pool, hk, hu, hg, s_slot[i2], cutlass.Int64(s_rs[i2]), h2_, mbh
                    )
    if cutlass.const_expr(pdl):
        prims.griddepcontrol("launch_dependents")
    # (CTAs without items never read the address: make warp 7's allocation visible before the dealloc)
    cute.arch.barrier()
    if warp == 7:
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(
            cutlass.inttoptr(tmem_holding.load(), 6, cutlass.Float32),
            cutlass.Int32(128),
            group="cta_1",
        )


@cute.jit
def _launch_commit(
    pool,
    slots,
    replay_indices,
    hk,
    hu,
    hg,
    cflags,
    batch_size: Int32,
    ncta: Int32,
    heads: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    nst: cutlass.Constexpr,
    early_refill: cutlass.Constexpr,
    halves: cutlass.Constexpr,
    minb: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _commit_kernel(
        pool,
        slots,
        replay_indices,
        hk,
        hu,
        hg,
        cflags,
        batch_size,
        heads,
        pdl,
        nst,
        early_refill,
        halves,
    ).launch(
        grid=[ncta, 1, 1],
        block=[CK_THREADS, 1, 1],
        stream=stream,
        use_pdl=pdl,
        min_blocks_per_mp=minb,
    )


@cute.jit
def _launch_decode(
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    replay_indices,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    arrivals,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    minb: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    primary_layout = cute.make_layout(
        (64, 128, heads, pool.shape[0]),
        stride=(1, 64, STATE_BYTES, pool.layout.stride[0]),
    )
    primary_tensor = cute.make_tensor(pool.iterator, primary_layout)
    tma_state = cuda_exp.create_tensor_map_tiled_from_view(
        primary_tensor,
        box_dims=(64, 128, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_exp.TensorMapSwizzle.s64b,
    )
    _decode_kernel(
        tma_state,
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool,
        slots,
        dummy,
        replay_indices,
        hk,
        hu,
        hg,
        lengths,
        a_log,
        dt_bias,
        out,
        arrivals,
        lower_bound,
        norm_eps,
        heads,
        bounded,
        pdl,
        minb == 0,
        minb != 0,
    ).launch(
        grid=[x.shape[0] * heads // 1, 1, 1],
        block=[THREADS * 1, 1, 1],
        stream=stream,
        use_pdl=pdl,
        min_blocks_per_mp=minb,
    )


def _fake_tensor_args(heads: int, history_strides: tuple[int, int, int, int]) -> tuple:
    b = cute.sym_int32(divisibility=1)
    s = cute.sym_int32(divisibility=1)
    r = cute.sym_int32(divisibility=1)
    a = cute.sym_int32(divisibility=1)

    def row(divisibility: int = 1) -> cute.SymInt:
        return cute.sym_int64(divisibility=divisibility)

    width = heads * K
    channels = 3 * width
    k_stride, u_stride, g_stride, length_stride = history_strides
    return (
        make_fake_tensor(cutlass.BFloat16, (b, channels), stride=(row(16), 1)),
        make_fake_tensor(cutlass.BFloat16, (s, channels, 3), stride=(row(16), 3, 1)),
        make_fake_tensor(cutlass.BFloat16, (channels, 4), stride=(4, 1)),
        make_fake_tensor(cutlass.BFloat16, (b, width), stride=(row(8), 1)),
        make_fake_tensor(cutlass.BFloat16, (b, heads), stride=(row(1), 1)),
        make_fake_tensor(cutlass.BFloat16, (b, heads, K), stride=(row(8), K, 1)),
        make_fake_tensor(cutlass.BFloat16, (K,), stride=(1,)),
        make_fake_tensor(cutlass.Uint8, (s, heads, STATE_BYTES), stride=(row(16), STATE_BYTES, 1)),
        make_fake_tensor(cutlass.Int32, (b,), stride=(1,)),
        make_fake_tensor(cutlass.Boolean, (b,), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (b,), stride=(1,)),
        make_fake_tensor(cutlass.BFloat16, (r, heads, N, K), stride=(k_stride, N * K, K, 1)),
        make_fake_tensor(cutlass.BFloat16, (r, heads, N, K), stride=(u_stride, N * K, K, 1)),
        make_fake_tensor(cutlass.Float32, (r, heads, N, K), stride=(g_stride, N * K, K, 1)),
        make_fake_tensor(cutlass.Int32, (r,), stride=(length_stride,)),
        make_fake_tensor(cutlass.Float32, (heads,), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (width,), stride=(1,)),
        make_fake_tensor(cutlass.BFloat16, (b, heads, K), stride=(width, K, 1)),
        make_fake_tensor(cutlass.Int32, (a,), stride=(1,)),
    )


# Launch variants by grid size (B * H items); every variant computes the same results.
_MID_ITEMS = 304  # above this, the decode kernel caps registers so three CTAs fit per SM
_MID_MIN_BLOCKS = 3
_PERSISTENT_MIN_ITEMS = (
    1536  # from here on, the persistent kernel decodes and a separate kernel commits
)
_PERSISTENT_CTAS_PER_SM = 6
_SEPARATE_COMMIT_MAX_ROWS = (
    CK_MAX_ROWS  # the separate commit kernel's row flags (in the workspace) hold this many rows
)
_DYNAMIC_TAIL_ITEMS = (
    6  # items per persistent CTA from which the tail items are claimed at run time
)
_L2_PREFETCH_MAX_ITEMS = (
    3072  # up to here, the persistent kernel warms L2 with the committing rows' residuals
)
_COMMIT_CTAS_PER_SM = 2
_COMMIT_MANY_ITEMS = 12288  # from here on, the commit kernel runs three single-stage CTAs per SM
_PTXAS_OPTIONS = "-regUsageLevel=8"
_HIST_UNROLL = 15  # decode kernel: fully unrolled, masked history window

REPLAY_WORKSPACE_INT32 = CK_MAX_ROWS + 2
"""int: Minimum ``workspace`` length of :func:`fused_replay` (per-row commit flags, then two counters)."""

_sm_counts: dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _sm_counts:
        _sm_counts[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _sm_counts[index]


def _compile_options(ptxas: str) -> str:
    return "--enable-tvm-ffi" + (f" --ptxas-options='{ptxas}'" if ptxas else "")


def _check_histories(*histories: torch.Tensor) -> None:
    for history in histories:
        if (
            history.stride()[1:] != (N * K, K, 1)
            or (history.stride(0) * history.element_size()) % 16
        ):
            raise ValueError(
                "2FP4 history slots must be dense, 16-byte-aligned [H, 16, 128] payloads"
            )


@torch.no_grad()
def fused_replay(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    dummy: torch.Tensor,
    replay_indices: torch.Tensor,
    hk: torch.Tensor,
    hu: torch.Tensor,
    hg: torch.Tensor,
    lengths: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    norm_eps: float,
    out: torch.Tensor,
    arrivals: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    """Write BF16 [B,H,128] output, append or commit the window, and advance lengths.

    ``arrivals`` holds at least B zero-initialized int32 counters, and
    ``workspace`` is a contiguous int32 tensor of at least
    ``REPLAY_WORKSPACE_INT32`` values whose last two are zero. Every launch
    returns those counters to zero, so both can be shared by all layers that
    run in one stream.
    """
    _check_histories(hk, hu, hg)
    if (
        workspace.dtype != torch.int32
        or not workspace.is_contiguous()
        or workspace.numel() < REPLAY_WORKSPACE_INT32
    ):
        raise ValueError(
            f"2FP4 replay needs a contiguous int32 workspace of {REPLAY_WORKSPACE_INT32} values"
        )
    batch, heads = x.shape[0], a_log.numel()
    items = batch * heads
    bounded = lower_bound is not None
    lower_bound = 0.0 if lower_bound is None else float(lower_bound)
    norm_eps = float(norm_eps)
    pdl = get_env_enable_pdl()
    history_strides = (hk.stride(0), hu.stride(0), hg.stride(0), lengths.stride(0))
    sms = _sm_count(x.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    tensors = (
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool,
        slots,
        dummy,
        replay_indices,
        hk,
        hu,
        hg,
        lengths,
        a_log,
        dt_bias,
        out,
        arrivals,
    )
    if items >= _PERSISTENT_MIN_ITEMS:
        _persistent_replay(
            tensors,
            workspace,
            batch,
            heads,
            bounded,
            lower_bound,
            norm_eps,
            pdl,
            history_strides,
            sms,
            stream,
        )
        return
    min_blocks = _MID_MIN_BLOCKS if items > _MID_ITEMS else 0
    key = ("decode", heads, bounded, pdl, history_strides, min_blocks)
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _launch_decode,
            *_fake_tensor_args(heads, history_strides),
            lower_bound,
            norm_eps,
            heads,
            bounded,
            pdl,
            min_blocks,
            stream,
            options=_compile_options(_PTXAS_OPTIONS),
        )
    _compiled[key](*tensors, lower_bound, norm_eps, stream)


def _persistent_replay(
    tensors,
    workspace,
    batch,
    heads,
    bounded,
    lower_bound,
    norm_eps,
    pdl,
    history_strides,
    sms,
    stream,
) -> None:
    """Large grids: a persistent decode over (row, head) items, then (up to CK_MAX_ROWS rows) the window commits
    of the rows that reached a full window in a separate persistent kernel."""
    items = batch * heads
    separate_commit = batch <= _SEPARATE_COMMIT_MAX_ROWS
    if separate_commit:
        stages, ctas_per_sm = 1, _PERSISTENT_CTAS_PER_SM
        dynamic = items >= _DYNAMIC_TAIL_ITEMS * ctas_per_sm * sms
    else:
        stages, ctas_per_sm, dynamic = 2, 3, False
    l2_prefetch = separate_commit and items <= _L2_PREFETCH_MAX_ITEMS
    flags, counters = workspace[:CK_MAX_ROWS], workspace[CK_MAX_ROWS:REPLAY_WORKSPACE_INT32]
    key = (
        "persistent",
        heads,
        bounded,
        pdl,
        history_strides,
        separate_commit,
        dynamic,
        l2_prefetch,
    )
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _launch_tp,
            *_fake_tp_args(heads, history_strides),
            1,
            1,
            lower_bound,
            norm_eps,
            heads,
            bounded,
            pdl,
            separate_commit,
            stages,
            dynamic,
            l2_prefetch,
            ctas_per_sm,
            stream,
            options=_compile_options(_PTXAS_OPTIONS),
        )
    _compiled[key](
        *tensors,
        counters,
        flags,
        items,
        min(items, ctas_per_sm * sms),
        lower_bound,
        norm_eps,
        stream,
    )
    if not separate_commit:
        return
    many = items >= _COMMIT_MANY_ITEMS
    commit_ctas_per_sm, commit_stages, early_refill, halves = (
        (3, 1, True, True) if many else (_COMMIT_CTAS_PER_SM, 2, False, False)
    )
    key = ("commit", heads, pdl, history_strides, many)
    if key not in _compiled:
        fake = _fake_tensor_args(heads, history_strides)
        _compiled[key] = cute.compile(
            _launch_commit,
            fake[7],
            fake[8],
            fake[10],
            fake[11],
            fake[12],
            fake[13],
            make_fake_tensor(cutlass.Int32, (CK_MAX_ROWS,), stride=(1,)),
            1,
            1,
            heads,
            pdl,
            commit_stages,
            early_refill,
            halves,
            commit_ctas_per_sm,
            stream,
            options=_compile_options(""),
        )
    pool, slots, replay_indices, hk, hu, hg = (tensors[i] for i in (7, 8, 10, 11, 12, 13))
    _compiled[key](
        pool,
        slots,
        replay_indices,
        hk,
        hu,
        hg,
        flags,
        batch,
        min(items, commit_ctas_per_sm * sms),
        stream,
    )


def _fake_tp_args(heads: int, history_strides: tuple[int, int, int, int]) -> tuple:
    return _fake_tensor_args(heads, history_strides) + (
        make_fake_tensor(cutlass.Int32, (2,), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (CK_MAX_ROWS,), stride=(1,)),
    )
