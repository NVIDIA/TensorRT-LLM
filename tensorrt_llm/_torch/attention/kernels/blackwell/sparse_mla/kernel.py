# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline-PTX and CuTe DSL helpers of the GB300 sparse MLA kernel (``kernel_gb300``)."""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(operands, text, constraints, result=None, loc=None, ip=None, pure=False):
    # ``pure``: a register-only computation the compiler may schedule, combine or drop freely.
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=not pure,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_16(dst, src, *, loc=None, ip=None):
    """16-byte cp.async (L2 only) from a global byte address to a shared byte address."""
    _asm(
        [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r, l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_16_zero(dst, src, *, loc=None, ip=None):
    """16 zero bytes to shared memory through the cp.async path (source size 0)."""
    _asm(
        [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16, 0;",
        "r, l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_arrive_noinc(mbar, *, loc=None, ip=None):
    """Arrive on ``mbar`` once this thread's earlier cp.async operations have landed."""
    _asm(
        [Int32(mbar).ir_value(loc=loc, ip=ip)],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        loc=loc,
        ip=ip,
    )


def _tma_desc(atom, loc=None, ip=None):
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(atom._trait.value, loc=loc, ip=ip)
    ty = _cute_ir.PtrType.get(_cute_nvgpu_ir.TmaDescriptorTiledType.get(), AddressSpace.generic, 64)
    return (
        _cute_nvgpu_ir.get_tma_desc_addr(ty, exec_atom, loc=loc, ip=ip)
        .toint()
        .ir_value(loc=loc, ip=ip)
    )


@dsl_user_op
def tma_load_tile(atom, dst, mbar, col, row, *, loc=None, ip=None):
    """One 2D TMA tile (the atom's box) into this CTA's shared memory."""
    _asm(
        [
            Int32(dst).ir_value(loc=loc, ip=ip),
            _tma_desc(atom, loc, ip),
            Int32(mbar).ir_value(loc=loc, ip=ip),
            Int32(col).ir_value(loc=loc, ip=ip),
            Int32(row).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4}], [$2];",
        "r, l, r, r, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_tile(atom, src, col, row, *, loc=None, ip=None):
    """One 2D TMA tile (the atom's box) from this CTA's shared memory to global (bulk group)."""
    _asm(
        [
            _tma_desc(atom, loc, ip),
            Int32(src).ir_value(loc=loc, ip=ip),
            Int32(col).ir_value(loc=loc, ip=ip),
            Int32(row).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$0, {$2, $3}], [$1];",
        "l, r, r, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_load(dst, src, size, mbar, *, loc=None, ip=None):
    """``size`` bytes (multiple of 16) from global ``src`` to this CTA's shared ``dst`` by the bulk copy engine."""
    _asm(
        [
            Int32(dst).ir_value(loc=loc, ip=ip),
            Int64(src).ir_value(loc=loc, ip=ip),
            Int32(size).ir_value(loc=loc, ip=ip),
            Int32(mbar).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r, l, r, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_before_thread_sync(*, loc=None, ip=None):
    _asm([], "tcgen05.fence::before_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def tcgen05_after_thread_sync(*, loc=None, ip=None):
    _asm([], "tcgen05.fence::after_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def pack_e4m3x4(a, b, c, d, *, loc=None, ip=None) -> Int32:
    """Four FP32 values as saturated E4M3 bytes, ``a`` in the lowest byte."""
    return Int32(
        _asm(
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
            "{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\nmov.b32 $0, {lo, hi};\n}",
            "=r, f, f, f, f",
            result=T.i32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


@dsl_user_op
def e4m3x2_to_f16x2(pair16, *, loc=None, ip=None) -> Int32:
    """Two E4M3 bytes (low 16 bits of ``pair16``) as an f16x2 word, the lower byte in the lower half."""
    return Int32(
        _asm(
            [Int32(pair16).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $1;\ncvt.rn.f16x2.e4m3x2 $0, lo;\n}",
            "=r, r",
            result=T.i32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


@dsl_user_op
def f16_half_to_f32(word, upper: bool, *, loc=None, ip=None) -> Float32:
    """One f16 half of ``word`` as FP32."""
    half = "hi" if upper else "lo"
    return Float32(
        _asm(
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "{\n.reg .f16 lo, hi;\nmov.b32 {lo, hi}, $1;\ncvt.f32.f16 $0, " + half + ";\n}",
            "=f, r",
            result=T.f32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


@dsl_user_op
def add_f16x2(a, b, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f16x2 $0, $1, $2;",
            "=r, r, r",
            result=T.i32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


@dsl_user_op
def pack_e4m3x4_f16x2(lo, hi, *, loc=None, ip=None) -> Int32:
    """Two f16x2 words as four saturated E4M3 bytes, ``lo``'s low half in the lowest byte."""
    return Int32(
        _asm(
            [Int32(lo).ir_value(loc=loc, ip=ip), Int32(hi).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 l, h;\ncvt.rn.satfinite.e4m3x2.f16x2 l, $1;\ncvt.rn.satfinite.e4m3x2.f16x2 h, $2;\n"
            "mov.b32 $0, {l, h};\n}",
            "=r, r, r",
            result=T.i32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


def tree_max(vals):
    """Maximum of a list of FP32 values as a balanced tree (trace-time Python, outside the DSL rewriting)."""
    while len(vals) > 1:
        vals = [cute.arch.fmax(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)] + (
            [vals[-1]] if len(vals) % 2 else []
        )
    return vals[0]


def e4m3_sum16(words):
    """FP32 sum of the 16 E4M3 values in four packed words: f16x2 partial sums (at most 16 x 2^8) added
    as a tree, then the two halves in FP32."""
    parts = [half for w in words for half in (e4m3x2_to_f16x2(w), e4m3x2_to_f16x2(w >> 16))]
    while len(parts) > 1:
        parts = [add_f16x2(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
    return f16_half_to_f32(parts[0], False) + f16_half_to_f32(parts[0], True)


@dsl_user_op
def pack_f32x2(a, b, *, loc=None, ip=None) -> Int64:
    """Two FP32 values as one f32x2 operand, ``a`` in the low half."""
    return Int64(
        _asm(
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, {$1, $2};",
            "=l, f, f",
            result=T.i64(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


@dsl_user_op
def scale_bf16x2(a, b, s2, *, loc=None, ip=None) -> Int32:
    """(a, b) times the f32x2 operand ``s2`` as BF16, ``a`` in the low half: one FMUL2 and one conversion."""
    return Int32(
        _asm(
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Int64(s2).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b64 x, y;\n.reg .f32 l, h;\nmov.b64 x, {$1, $2};\nmul.rn.f32x2 y, x, $3;\nmov.b64 {l, h}, y;\n"
            "cvt.rn.bf16x2.f32 $0, h, l;\n}",
            "=r, f, f, l",
            result=T.i32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


@dsl_user_op
def exp2_f16x2_fma2(a, b, s2, c2, *, loc=None, ip=None) -> Int32:
    """2^(a s + c), 2^(b s + c) as an f16x2 word (``a`` in the low half) from the f32x2 operands ``s2`` and
    ``c2``: one FFMA2 (two exact FP32 FMAs), one f16x2 conversion, one f16x2 exponential."""
    return Int32(
        _asm(
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Int64(s2).ir_value(loc=loc, ip=ip),
                Int64(c2).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b64 x, y;\n.reg .f32 l, h;\n.reg .b32 w;\nmov.b64 x, {$1, $2};\nfma.rn.f32x2 y, x, $3, $4;\n"
            "mov.b64 {l, h}, y;\ncvt.rn.f16x2.f32 w, h, l;\nex2.approx.f16x2 $0, w;\n}",
            "=r, f, f, l, l",
            result=T.i32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


@dsl_user_op
def st_shared_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        [Int32(v).ir_value(loc=loc, ip=ip) for v in (addr, w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r, r, r, r, r",
        loc=loc,
        ip=ip,
    )
