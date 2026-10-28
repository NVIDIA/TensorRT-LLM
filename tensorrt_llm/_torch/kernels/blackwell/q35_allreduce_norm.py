# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNNVL all-reduce + residual + RMSNorm (CuTe DSL) on the library's MNNVL workspace.

``Q35AllReduceNormKernel`` matches ``oneshotAllreduceFusionKernel`` bit for bit: same thread layout (a
cluster of ``ctas`` CTAs per token, one 16-byte vector per thread) and same arithmetic order (FP32
rank-order sum -> BF16, BF16 residual add, BF16 squares summed per thread / warp butterfly / CTA butterfly /
cluster in CTA order, ``bf16(x * rstd * w)``, FP8 = satfinite(norm * (1 / scale))).

It shares the library's Lamport triple buffer and ``buffer_flags`` (current / dirty buffer, bytes per
buffer, dirty stages, 4 x dirty bytes, arrival counter), so its launches may interleave with the library
kernels in one graph; only the byte counts in the flags cross launches, its data layout is its own.

Extensions: ``se_fold`` computes ``routed + sigmoid(gate) * shared`` in-kernel with
``fused_sigmoid_gate_mul_add``'s arithmetic; ``moe_topk`` finalizes ``routed`` from the un-finalized
trtllm-gen MoE output with the finalize kernel's arithmetic; ``two_shot`` is bitwise the library's two-shot
+ Lamport RMSNorm pair in one launch; ``fp8`` (with ``two_shot``) carries E4M3 plus one FP32 scale per 256
elements in both stages (half the bytes; two E4M3 roundings per element, so not bitwise the library);
``tpc`` > 1 (BF16 two-shot) gives each CTA ``tpc`` consecutive tokens, so a call larger than one wave of
CTAs still runs as one wave, with every token's output bits those of the one-token kernel.

PDL: only the workspace table and the norm weight are read before ``griddepcontrol.wait``; never the
input, the residual (the previous all-reduce's output) or the flags.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

VEC = 8  # BF16 per thread (one 16-byte vector)
_NEG_ZERO = -2147483648  # 0x80000000: Lamport "not written" word (+0.0, -0.0 as BF16 pair)


def _asm(
    res, operands, text: str, constraints: str, side_effects: bool = True, *, loc=None, ip=None
):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc=None, ip=None):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ld_v4(addr: Int64, *, loc=None, ip=None):
    """Four 32-bit words at a 16-byte aligned global address (ordered after earlier asm)."""
    words = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_volatile_v4(addr: Int64, *, loc=None, ip=None):
    words = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip)] + [_v(Int32(w), loc, ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_v2(addr: Int64, w0: Int32, w1: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(w0), loc, ip), _v(Int32(w1), loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_s64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [_v(Int64(addr), loc, ip)],
            "ld.global.b64 $0, [$1];",
            "=l,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_u16(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip)],
            "{\n\t.reg .b16 h;\n\tld.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_volatile_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip)],
            "ld.volatile.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_acquire_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_s32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(value), loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_release_add_s32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(value), loc, ip)],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sanitize(word: Int32, *, loc=None, ip=None) -> Int32:
    """The two BF16 of a word with each -0.0 replaced by +0.0 (no written word is the sentinel)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(word), loc, ip)],
            "{\n\t.reg .b32 t, r;\n\t.reg .pred p;\n\tmov.b32 r, $1;\n\t"
            "and.b32 t, r, 0xffff0000;\n\tsetp.eq.b32 p, t, 0x80000000;\n\t@p and.b32 r, r, 0x0000ffff;\n\t"
            "and.b32 t, r, 0x0000ffff;\n\tsetp.eq.b32 p, t, 0x00008000;\n\t@p and.b32 r, r, 0xffff0000;\n\t"
            "mov.b32 $0, r;\n\t}",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lo(word: Int32, *, loc=None, ip=None) -> Float32:
    """FP32 value of the BF16 in the low half of a word."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Int32(word), loc, ip)],
            "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _hi(word: Int32, *, loc=None, ip=None) -> Float32:
    """FP32 value of the BF16 in the high half of a word."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Int32(word), loc, ip)],
            "{\n\t.reg .b32 t;\n\tand.b32 t, $1, 0xffff0000;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_bf16(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """Two FP32 rounded to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Float32(lo), loc, ip), _v(Float32(hi), loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16r(x: Float32, *, loc=None, ip=None) -> Float32:
    """FP32 -> BF16 (nearest even) -> FP32."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 t;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b32 t, {0, h};\n\t"
            "mov.b32 $0, t;\n\t}",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_ge0(p: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a if p >= 0 else b`` (selp, no branch)."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Int32(p), loc, ip), _v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "{\n\t.reg .pred q;\n\tsetp.ge.s32 q, $1, 0;\n\tselp.f32 $0, $2, $3, q;\n\t}",
            "=f,r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max0(p: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [_v(Int32(p), loc, ip)], "max.s32 $0, $1, 0;", "=r,r", False, loc=loc, ip=ip)
    )


@dsl_user_op
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip), _v(Float32(c), loc, ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rsqrt(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sigmoid_triton(g: Float32, *, loc=None, ip=None) -> Float32:
    """Triton's fp32 ``tl.sigmoid``: 1 / (1 + ex2.approx(-g * log2(e))) with div.full."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(g), loc, ip)],
            "{\n\t.reg .f32 t, e;\n\tmul.f32 t, $1, 0fBFB8AA3B;\n\tex2.approx.f32 e, t;\n\t"
            "add.f32 e, e, 0f3F800000;\n\tdiv.full.f32 $0, 0f3F800000, e;\n\t}",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """Two FP32 -> two E4M3 (nearest even, saturate to finite) in the low 16 bits, ``lo`` first."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Float32(lo), loc, ip), _v(Float32(hi), loc, ip)],
            "{\n\t.reg .b16 h;\n\tcvt.rn.satfinite.e4m3x2.f32 h, $2, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x4_f32(word: Int32, *, loc=None, ip=None):
    """Four E4M3 (byte i = element i) -> four FP32 (exact, via FP16)."""
    out = _asm(
        ir.Type.parse("!llvm.struct<(f32,f32,f32,f32)>"),
        [_v(Int32(word), loc, ip)],
        "{\n\t.reg .b16 lo, hi, a, b, c, d;\n\t.reg .b32 h01, h23;\n\tmov.b32 {lo, hi}, $4;\n\t"
        "cvt.rn.f16x2.e4m3x2 h01, lo;\n\tcvt.rn.f16x2.e4m3x2 h23, hi;\n\tmov.b32 {a, b}, h01;\n\t"
        "mov.b32 {c, d}, h23;\n\tcvt.f32.f16 $0, a;\n\tcvt.f32.f16 $1, b;\n\tcvt.f32.f16 $2, c;\n\t"
        "cvt.f32.f16 $3, d;\n\t}",
        "=f,=f,=f,=f,r",
        False,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _sanitize_e4m3(word: Int32, *, loc=None, ip=None) -> Int32:
    """Four E4M3 with the one pattern equal to the sentinel (+0, +0, +0, -0) replaced by four +0."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(word), loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.eq.b32 p, $1, 0x80000000;\n\tselp.b32 $0, 0, $1, p;\n\t}",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_bits(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [_v(Float32(x), loc, ip)], "mov.b32 $0, $1;", "=r,f", False, loc=loc, ip=ip)
    )


@dsl_user_op
def _bits_f32(w: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_v(Int32(w), loc, ip)], "mov.b32 $0, $1;", "=f,r", False, loc=loc, ip=ip)
    )


@dsl_user_op
def _amax8(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    a4: Float32,
    a5: Float32,
    a6: Float32,
    a7: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """max |a_i| (NaN-free inputs)."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip) for a in (a0, a1, a2, a3, a4, a5, a6, a7)],
            "{\n\t.reg .f32 m, t;\n\tabs.f32 m, $1;\n\tabs.f32 t, $2;\n\tmax.f32 m, m, t;\n\tabs.f32 t, $3;\n\t"
            "max.f32 m, m, t;\n\tabs.f32 t, $4;\n\tmax.f32 m, m, t;\n\tabs.f32 t, $5;\n\tmax.f32 m, m, t;\n\t"
            "abs.f32 t, $6;\n\tmax.f32 m, m, t;\n\tabs.f32 t, $7;\n\tmax.f32 m, m, t;\n\tabs.f32 t, $8;\n\t"
            "max.f32 $0, m, t;\n\t}",
            "=f,f,f,f,f,f,f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_eq(p: Int32, q: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a if p == q else b`` (selp, no branch)."""
    return Float32(
        _asm(
            T.f32(),
            [
                _v(Int32(p), loc, ip),
                _v(Int32(q), loc, ip),
                _v(Float32(a), loc, ip),
                _v(Float32(b), loc, ip),
            ],
            "{\n\t.reg .pred c;\n\tsetp.eq.s32 c, $1, $2;\n\tselp.f32 $0, $3, $4, c;\n\t}",
            "=f,r,r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmax(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_volatile_v2(addr: Int64, *, loc=None, ip=None):
    words = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.volatile.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _shfl_bfly(x: Float32, mask: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip), _v(Int32(mask), loc, ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _smem_addr(ptr, *, loc=None, ip=None) -> Int32:
    return Int32(ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def _sink4(addr: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None) -> None:
    """Store a ^ b ^ c ^ d to shared ``addr``, forcing the loads of the four words to complete first
    (PDL: inputs consumed before launch_dependents)."""
    _asm(
        None,
        [_v(Int32(addr), loc, ip)] + [_v(Int32(w), loc, ip) for w in (a, b, c, d)],
        "{\n\t.reg .b32 t;\n\txor.b32 t, $1, $2;\n\txor.b32 t, t, $3;\n\txor.b32 t, t, $4;\n\t"
        "st.shared.b32 [$0], t;\n\t}",
        "r,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_f32(addr: Int32, value: Float32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Float32(value), loc, ip)],
        "st.shared.f32 [$0], $1;",
        "r,f,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(Int32(addr), loc, ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_v4(
    addr: Int32, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [_v(Int32(addr), loc, ip)] + [_v(Int32(w), loc, ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    words = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [_v(Int32(addr), loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _min_s32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(a), loc, ip), _v(Int32(b), loc, ip)],
            "min.s32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_dsmem_f32(addr: Int32, cta_rank: Int32, value: Float32, *, loc=None, ip=None) -> None:
    """Store at this CTA's shared address ``addr`` in cluster CTA ``cta_rank``."""
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Int32(cta_rank), loc, ip), _v(Float32(value), loc, ip)],
        "{\n\t.reg .b32 ra;\n\tmapa.shared::cluster.u32 ra, $0, $1;\n\tst.shared::cluster.f32 [ra], $2;\n\t}",
        "r,r,f,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None) -> None:
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;",
        "~{memory}",
        loc=loc,
        ip=ip,
    )


class Q35AllReduceNormKernel:
    """MNNVL all-reduce + residual + RMSNorm for ``m`` tokens.

    Tensors (contiguous): x [M, H] BF16 (unused with ``se_fold``), routed / shared [M, H] BF16 and
    gate [M] BF16 (``se_fold``), residual [M, H], weight [H] BF16 (``(1 + w)`` baked for gemma norms),
    scale [1] FP32 (``quant``), residual_out [M, H], norm_out [M, H] (``norm_out``), quant_out
    [M, H] E4M3 as uint8 (``quant``), table int64 [3 + W] = (local buffer, multicast buffer, flags,
    every rank's unicast buffer).
    """

    def __init__(
        self,
        *,
        hidden: int,
        norm_eps: float,
        nranks: int,
        ctas: int,
        use_pdl: bool,
        se_fold: bool,
        norm_out: bool,
        quant: bool,
        moe_topk: int = 0,
        fc2_row: int = 0,
        two_shot: bool = False,
        fp8: bool = False,
        tpc: int = 1,
    ):
        if hidden % (VEC * ctas) or (hidden // VEC // ctas) % 32 or hidden // VEC // ctas > 1024:
            raise ValueError(f"unsupported hidden size {hidden} for {ctas} CTAs per token")
        if nranks not in (2, 4, 8, 16) or ctas not in (1, 2, 4, 8):
            raise ValueError("unsupported group / cluster size")
        if fp8 and not two_shot:
            raise ValueError("the FP8 payload needs the two-shot mode")
        if tpc < 1 or (tpc > 1 and (not two_shot or fp8 or ctas != 1)):
            raise ValueError(
                "more than one token per CTA needs the BF16 two-shot mode with one CTA per token"
            )
        self.tpc = tpc  # tokens per CTA (see the module docstring)
        self.hidden = hidden
        self.norm_eps = float(norm_eps)
        self.nranks = nranks
        self.ctas = ctas
        self.threads = hidden // VEC // ctas
        self.num_warps = self.threads // 32
        # 512-thread CTAs (one CTA per token) are launched with two CTAs per SM as the bound (at most 64 registers):
        # one wave then covers 2 x SMs tokens (304 rows on GB300), the wave size tpc assumes. Without the bound the
        # compiler uses more registers and fits one CTA per SM, so a 448-row call needs two waves, one more
        # cross-rank latency chain.
        self.min_ctas_per_sm = 2 if self.threads == 512 else 0
        self.use_pdl = use_pdl
        self.se_fold = se_fold
        # moe_topk > 0: routed = sum_k w[t, k] * fc2[perm[t, k]] over local experts (perm < 0 skipped),
        # FP32 FMA in k order, BF16 once (the finalize kernel's math).
        self.moe_topk = moe_topk
        self.fc2_row = fc2_row or hidden  # elements per fc2 row (padded hidden)
        self.norm_out = norm_out
        self.quant = quant
        self.two_shot = two_shot
        # fp8: E4M3 payload in both two-shot stages, one FP32 scale per 256 elements (a warp's 32 vectors) after the
        # H data bytes of a row; bf16: H BF16.
        self.fp8 = fp8
        self.row_bytes = hidden + hidden // 64 if fp8 else hidden * 2

    @cute.jit
    def __call__(
        self,
        gX: cute.Tensor,
        gRouted: cute.Tensor,
        gShared: cute.Tensor,
        gGate: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        gWs: cute.Tensor,
        fc2_ptr: Int64,
        topw_ptr: Int64,
        perm_ptr: Int64,
        rank: Int32,
        stream,
    ) -> None:
        m = cute.size(gRes, mode=[0])
        if const_expr(self.tpc > 1):
            # smem: the CTA's residual vectors (tpc x H BF16), per-token warp sums, the flags
            self.kernel_mt(
                gX,
                gRouted,
                gShared,
                gGate,
                gRes,
                gW,
                gScale,
                gResOut,
                gOut,
                gQuant,
                gWs,
                fc2_ptr,
                topw_ptr,
                perm_ptr,
                rank,
            ).launch(
                grid=[(m + self.tpc - 1) // self.tpc, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[1, 1, 1],
                smem=1024 + self.tpc * self.hidden * 2,
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=self.min_ctas_per_sm,
            )
        else:
            self.kernel(
                gX,
                gRouted,
                gShared,
                gGate,
                gRes,
                gW,
                gScale,
                gResOut,
                gOut,
                gQuant,
                gWs,
                fc2_ptr,
                topw_ptr,
                perm_ptr,
                rank,
            ).launch(
                grid=[m * self.ctas, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[self.ctas, 1, 1],
                smem=512,
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=self.min_ctas_per_sm,
            )

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gRouted: cute.Tensor,
        gShared: cute.Tensor,
        gGate: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        gWs: cute.Tensor,
        fc2_ptr: Int64,
        topw_ptr: Int64,
        perm_ptr: Int64,
        rank: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        ctas: cutlass.Constexpr = self.ctas
        nranks: cutlass.Constexpr = self.nranks
        vecs: cutlass.Constexpr = self.hidden // VEC
        num_warps: cutlass.Constexpr = self.num_warps
        tok = bid // ctas
        part = bid % ctas
        m = grid // ctas
        vi = part * self.threads + tidx  # this thread's 16-byte vector within the token
        warp = tidx // 32
        lane = tidx % 32
        row_off = (Int64(tok) * Int64(vecs) + Int64(vi)) * Int64(16)

        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout((num_warps,)), byte_alignment=16)
        csum = smem.allocate_tensor(Float32, cute.make_layout((ctas,)), byte_alignment=16)
        fl = smem.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        red_addr = _smem_addr(red.iterator)
        csum_addr = _smem_addr(csum.iterator)

        # ---- before the dependency wait: only host-written addresses and the norm weight ----
        tbl = gWs.iterator.toint()
        own = _ld_s64(Int64(tbl))
        mcast = _ld_s64(Int64(tbl) + Int64(8))
        flags = _ld_s64(Int64(tbl) + Int64(16))
        w0, w1, w2, w3 = _ld_v4(Int64(gW.iterator.toint()) + Int64(vi) * Int64(16))
        owner = tok % Int32(nranks)  # two-shot: the rank that reduces this token
        peer = own
        if const_expr(self.two_shot):
            peer = _ld_s64(Int64(tbl) + Int64(24) + Int64(owner) * Int64(8))

        if const_expr(self.use_pdl):
            griddepcontrol_wait()

        # ---- this rank's inputs, issued ahead of the flag loads so the barrier below waits one latency ----
        r0, r1, r2, r3 = _ld_v4(Int64(gRes.iterator.toint()) + row_off)
        if const_expr(self.se_fold):
            if const_expr(self.moe_topk > 0):
                # Index / weight loads now, row loads after the barrier (non-local experts read row 0 and
                # are dropped by a select).
                topk: cutlass.Constexpr = self.moe_topk
                kbase = Int64(tok) * Int64(topk)
                pk = cute.make_rmem_tensor((topk,), Int32)
                wk = cute.make_rmem_tensor((topk,), Int32)
                for k in cutlass.range_constexpr(topk):
                    pk[k] = _ld_s32(perm_ptr + (kbase + Int64(k)) * Int64(4))
                    wk[k] = _ld_u16(topw_ptr + (kbase + Int64(k)) * Int64(2))
            else:
                f0, f1, f2, f3 = _ld_v4(Int64(gRouted.iterator.toint()) + row_off)
            h0, h1, h2, h3 = _ld_v4(Int64(gShared.iterator.toint()) + row_off)
            gw = _ld_u16(Int64(gGate.iterator.toint()) + Int64(tok) * Int64(2))
        else:
            x0, x1, x2, x3 = _ld_v4(Int64(gX.iterator.toint()) + row_off)

        # ---- Lamport state read by one thread per CTA (all warps polling one line is an L2 hot spot);
        # the CTA counts its arrival only after reading it ----
        if tidx == 0:
            lf0, lf1, lf2, lf3 = _ld_volatile_v4(flags)
            lf4, lf5, lf6, lf7 = _ld_volatile_v4(flags + Int64(16))
            fl[0] = lf0
            fl[1] = lf1
            fl[2] = lf2
            fl[3] = lf3
            fl[4] = lf4
            fl[5] = lf5
            fl[6] = lf6
            fl[7] = lf7
        cute.arch.sync_threads()
        cur = fl[0]
        dirty = fl[1]
        buf_bytes = Int64(fl[2])
        dirty_stages = fl[3]
        sb0 = fl[4]
        sb1 = fl[5]
        sb2 = fl[6]
        sb3 = fl[7]
        if tidx == 0:
            _red_release_add_s32(flags + Int64(32), Int32(1))

        if const_expr(self.se_fold):
            if const_expr(self.moe_topk > 0):
                ev = cute.make_rmem_tensor((topk * 4,), Int32)
                for k in cutlass.range_constexpr(topk):
                    e0, e1, e2, e3 = _ld_v4(
                        fc2_ptr
                        + (Int64(_max0(pk[k])) * Int64(self.fc2_row) + Int64(vi) * Int64(VEC))
                        * Int64(2)
                    )
                    ev[4 * k + 0] = e0
                    ev[4 * k + 1] = e1
                    ev[4 * k + 2] = e2
                    ev[4 * k + 3] = e3
                acc = cute.make_rmem_tensor((VEC,), Float32)
                for e in cutlass.range_constexpr(VEC):
                    acc[e] = Float32(0.0)
                for k in cutlass.range_constexpr(topk):
                    wf = _lo(wk[k])
                    for i in cutlass.range_constexpr(4):
                        acc[2 * i] = _sel_ge0(
                            pk[k], _fma(wf, _lo(ev[4 * k + i]), acc[2 * i]), acc[2 * i]
                        )
                        acc[2 * i + 1] = _sel_ge0(
                            pk[k], _fma(wf, _hi(ev[4 * k + i]), acc[2 * i + 1]), acc[2 * i + 1]
                        )
                f0 = _pack_bf16(acc[0], acc[1])
                f1 = _pack_bf16(acc[2], acc[3])
                f2 = _pack_bf16(acc[4], acc[5])
                f3 = _pack_bf16(acc[6], acc[7])
            g = _lo(gw)
            s = _sigmoid_triton(g)
            x0 = _pack_bf16(_fma(s, _lo(h0), _lo(f0)), _fma(s, _hi(h0), _hi(f0)))
            x1 = _pack_bf16(_fma(s, _lo(h1), _lo(f1)), _fma(s, _hi(h1), _hi(f1)))
            x2 = _pack_bf16(_fma(s, _lo(h2), _lo(f2)), _fma(s, _hi(h2), _hi(f2)))
            x3 = _pack_bf16(_fma(s, _lo(h3), _lo(f3)), _fma(s, _hi(h3), _hi(f3)))

        tot = Int64(m) * Int64(vecs)
        idx = Int64(tok) * Int64(vecs) + Int64(vi)
        data_off = Int64(cur) * buf_bytes
        lt = tok // Int32(nranks)
        rb: cutlass.Constexpr = self.row_bytes
        hb: cutlass.Constexpr = self.hidden  # FP8 rows: the scales start after the H data bytes
        wi = vi // Int32(32)  # FP8: this thread's 256-element scale block (its warp's 32 vectors)
        xq = cute.make_rmem_tensor(
            (VEC,), Float32
        )  # FP8: this rank's input as FP32 (exact, BF16 values)
        if const_expr(self.two_shot):
            if const_expr(self.fp8):
                # ---- scatter, E4M3: per-warp amax -> scale = amax / 448 (FP32 after the row's H data bytes) ----
                xw = (x0, x1, x2, x3)
                for i in cutlass.range_constexpr(4):
                    xq[2 * i] = _lo(xw[i])
                    xq[2 * i + 1] = _hi(xw[i])
                amax = _amax8(xq[0], xq[1], xq[2], xq[3], xq[4], xq[5], xq[6], xq[7])
                for k in cutlass.range_constexpr(5):
                    amax = _fmax(amax, _shfl_bfly(amax, Int32(16 >> k)))
                amax = _fmax(amax, Float32(1e-30))
                inv = _div_rn(Float32(448.0), amax)
                q0 = _sanitize_e4m3(
                    _e4m3x2(_mul(xq[0], inv), _mul(xq[1], inv))
                    | (_e4m3x2(_mul(xq[2], inv), _mul(xq[3], inv)) << Int32(16))
                )
                q1 = _sanitize_e4m3(
                    _e4m3x2(_mul(xq[4], inv), _mul(xq[5], inv))
                    | (_e4m3x2(_mul(xq[6], inv), _mul(xq[7], inv)) << Int32(16))
                )
                srow = peer + data_off + (Int64(lt) * Int64(nranks) + Int64(rank)) * Int64(rb)
                _st_v2(srow + Int64(vi) * Int64(8), q0, q1)
                if lane == 0:
                    _st_s32(
                        srow + Int64(hb) + Int64(wi) * Int64(4),
                        _f32_bits(_mul(amax, Float32(1.0 / 448.0))),
                    )
            else:
                # ---- scatter: unicast into the owner's SCATTER stage [t / W][rank][H] ----
                _st_v4(
                    peer
                    + data_off
                    + ((Int64(lt) * Int64(nranks) + Int64(rank)) * Int64(vecs) + Int64(vi))
                    * Int64(16),
                    _sanitize(x0),
                    _sanitize(x1),
                    _sanitize(x2),
                    _sanitize(x3),
                )
        else:
            # ---- one multicast store reaches every rank's current buffer ([rank][token][H]) ----
            _st_v4(
                mcast + data_off + (Int64(rank) * tot + idx) * Int64(16),
                _sanitize(x0),
                _sanitize(x1),
                _sanitize(x2),
                _sanitize(x3),
            )

        # ---- PDL: release dependents only after every allocator-recyclable input is consumed by the whole CTA
        # (a dependent's pre-wait write into a recycled block must not race these reads). After this only the
        # persistent workspace / flags and this call's outputs are touched. ----
        if const_expr(self.use_pdl):
            _sink4(red_addr, r0, r1, r2, r3)
            cute.arch.sync_threads()
            griddepcontrol_launch_dependents()

        # ---- clear this thread's share of the previous call's buffer (every dirty stage) ----
        neg0 = Int32(_NEG_ZERO)
        for s_i in cutlass.range_constexpr(4):
            if Int32(s_i) < dirty_stages:
                stage_bytes = buf_bytes // Int64(dirty_stages)
                stage = own + (Int64(dirty) * Int64(dirty_stages) + Int64(s_i)) * stage_bytes
                sb = sb0 if s_i == 0 else (sb1 if s_i == 1 else (sb2 if s_i == 2 else sb3))
                clear_vecs = (Int64(sb) + Int64(15)) // Int64(16)
                c = idx
                while c < clear_vecs:
                    _st_v4(stage + c * Int64(16), neg0, neg0, neg0, neg0)
                    c = c + tot

        rw = (r0, r1, r2, r3)
        ww = (w0, w1, w2, w3)
        res = cute.make_rmem_tensor((VEC,), Float32)  # residual out (BF16 values)
        got = cute.make_rmem_tensor((nranks * 4,), Int32)
        if const_expr(self.two_shot and self.fp8):
            # (names below are unique in this function: the DSL rejects a dynamic-if region that retypes a name)
            half8 = buf_bytes // Int64(2)  # BROADCAST stage offset within the current buffer
            oacc = cute.make_rmem_tensor((VEC,), Float32)  # owner: FP32 rank-order sum
            if owner == rank:
                # ---- owner: wait for every rank's E4M3 vector + scale; FP32 sum in rank order (own input exact) ----
                fpend_o = Int32(1)
                while fpend_o != 0:
                    fpend_o = Int32(0)
                    for r in cutlass.range_constexpr(nranks):
                        orow = own + data_off + (Int64(lt) * Int64(nranks) + Int64(r)) * Int64(rb)
                        oa, ob = _ld_volatile_v2(orow + Int64(vi) * Int64(8))
                        osw = _ld_volatile_s32(orow + Int64(hb) + Int64(wi) * Int64(4))
                        got[r * 3 + 0] = oa
                        got[r * 3 + 1] = ob
                        got[r * 3 + 2] = osw
                        if oa == neg0:
                            fpend_o = Int32(1)
                        if ob == neg0:
                            fpend_o = Int32(1)
                        if osw == neg0:
                            fpend_o = Int32(1)
                for e in cutlass.range_constexpr(VEC):
                    oacc[e] = Float32(0.0)
                for r in cutlass.range_constexpr(nranks):
                    osc = _bits_f32(got[r * 3 + 2])
                    d0, d1, d2, d3 = _e4m3x4_f32(got[r * 3 + 0])
                    d4, d5, d6, d7 = _e4m3x4_f32(got[r * 3 + 1])
                    dr = (d0, d1, d2, d3, d4, d5, d6, d7)
                    for e in cutlass.range_constexpr(VEC):
                        oacc[e] = _add(oacc[e], _sel_eq(rank, Int32(r), xq[e], _mul(dr[e], osc)))
                # ---- quantize the sum (per-warp scale) and multicast it into every rank's BROADCAST stage ----
                bmax = _amax8(
                    oacc[0], oacc[1], oacc[2], oacc[3], oacc[4], oacc[5], oacc[6], oacc[7]
                )
                for k in cutlass.range_constexpr(5):
                    bmax = _fmax(bmax, _shfl_bfly(bmax, Int32(16 >> k)))
                bmax = _fmax(bmax, Float32(1e-30))
                binv = _div_rn(Float32(448.0), bmax)
                bw0 = _sanitize_e4m3(
                    _e4m3x2(_mul(oacc[0], binv), _mul(oacc[1], binv))
                    | (_e4m3x2(_mul(oacc[2], binv), _mul(oacc[3], binv)) << Int32(16))
                )
                bw1 = _sanitize_e4m3(
                    _e4m3x2(_mul(oacc[4], binv), _mul(oacc[5], binv))
                    | (_e4m3x2(_mul(oacc[6], binv), _mul(oacc[7], binv)) << Int32(16))
                )
                brow = mcast + data_off + half8 + Int64(tok) * Int64(rb)
                _st_v2(brow + Int64(vi) * Int64(8), bw0, bw1)
                if lane == 0:
                    _st_s32(
                        brow + Int64(hb) + Int64(wi) * Int64(4),
                        _f32_bits(_mul(bmax, Float32(1.0 / 448.0))),
                    )
            # ---- every rank: wait for the reduced E4M3 vector + scale in the BROADCAST stage ----
            fpend_b = Int32(1)
            while fpend_b != 0:
                fpend_b = Int32(0)
                brow_o = own + data_off + half8 + Int64(tok) * Int64(rb)
                ba, bb = _ld_volatile_v2(brow_o + Int64(vi) * Int64(8))
                bsw = _ld_volatile_s32(brow_o + Int64(hb) + Int64(wi) * Int64(4))
                got[0] = ba
                got[1] = bb
                got[2] = bsw
                if ba == neg0:
                    fpend_b = Int32(1)
                if bb == neg0:
                    fpend_b = Int32(1)
                if bsw == neg0:
                    fpend_b = Int32(1)
            bs = _bits_f32(got[2])
            g0, g1, g2, g3 = _e4m3x4_f32(got[0])
            g4, g5, g6, g7 = _e4m3x4_f32(got[1])
            gr = (g0, g1, g2, g3, g4, g5, g6, g7)
            for i in cutlass.range_constexpr(4):
                res[2 * i] = _bf16r(_add(_bf16r(_mul(gr[2 * i], bs)), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_bf16r(_mul(gr[2 * i + 1], bs)), _hi(rw[i])))
        elif const_expr(self.two_shot):
            half = buf_bytes // Int64(2)  # BROADCAST stage offset within the current buffer
            if owner == rank:
                # ---- owner: wait for every rank's vector, FP32 sum in rank order -> BF16, multicast ----
                pend_o = Int32(1)
                while pend_o != 0:
                    pend_o = Int32(0)
                    for r in cutlass.range_constexpr(nranks):
                        a1, b1, c1, d1 = _ld_volatile_v4(
                            own
                            + data_off
                            + ((Int64(lt) * Int64(nranks) + Int64(r)) * Int64(vecs) + Int64(vi))
                            * Int64(16)
                        )
                        got[r * 4 + 0] = a1
                        got[r * 4 + 1] = b1
                        got[r * 4 + 2] = c1
                        got[r * 4 + 3] = d1
                        if a1 == neg0:
                            pend_o = Int32(1)
                        if b1 == neg0:
                            pend_o = Int32(1)
                        if c1 == neg0:
                            pend_o = Int32(1)
                        if d1 == neg0:
                            pend_o = Int32(1)
                rsum = cute.make_rmem_tensor((4,), Int32)
                for i in cutlass.range_constexpr(4):
                    slo = Float32(0.0)
                    shi = Float32(0.0)
                    for r in cutlass.range_constexpr(nranks):
                        slo = _add(slo, _lo(got[r * 4 + i]))
                        shi = _add(shi, _hi(got[r * 4 + i]))
                    rsum[i] = _sanitize(_pack_bf16(slo, shi))
                _st_v4(
                    mcast + data_off + half + idx * Int64(16), rsum[0], rsum[1], rsum[2], rsum[3]
                )
            # ---- every rank: wait for the reduced vector in the BROADCAST stage ----
            pend_b = Int32(1)
            while pend_b != 0:
                pend_b = Int32(0)
                a2, b2, c2, d2 = _ld_volatile_v4(own + data_off + half + idx * Int64(16))
                got[0] = a2
                got[1] = b2
                got[2] = c2
                got[3] = d2
                if a2 == neg0:
                    pend_b = Int32(1)
                if b2 == neg0:
                    pend_b = Int32(1)
                if c2 == neg0:
                    pend_b = Int32(1)
                if d2 == neg0:
                    pend_b = Int32(1)
            for i in cutlass.range_constexpr(4):
                res[2 * i] = _bf16r(_add(_lo(got[i]), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_hi(got[i]), _hi(rw[i])))
        else:
            # ---- wait for this vector of every rank ----
            pending = Int32(1)
            while pending != 0:
                pending = Int32(0)
                for r in cutlass.range_constexpr(nranks):
                    a, b, cc, d = _ld_volatile_v4(
                        own + data_off + (Int64(r) * tot + idx) * Int64(16)
                    )
                    got[r * 4 + 0] = a
                    got[r * 4 + 1] = b
                    got[r * 4 + 2] = cc
                    got[r * 4 + 3] = d
                    if a == neg0:
                        pending = Int32(1)
                    if b == neg0:
                        pending = Int32(1)
                    if cc == neg0:
                        pending = Int32(1)
                    if d == neg0:
                        pending = Int32(1)

            # ---- FP32 sum in rank order -> BF16; + residual (BF16) ----
            for i in cutlass.range_constexpr(4):
                lo = Float32(0.0)
                hi = Float32(0.0)
                for r in cutlass.range_constexpr(nranks):
                    lo = _add(lo, _lo(got[r * 4 + i]))
                    hi = _add(hi, _hi(got[r * 4 + i]))
                res[2 * i] = _bf16r(_add(_bf16r(lo), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_bf16r(hi), _hi(rw[i])))
        out_off = Int64(gResOut.iterator.toint()) + row_off
        _st_v4(
            out_off,
            _pack_bf16(res[0], res[1]),
            _pack_bf16(res[2], res[3]),
            _pack_bf16(res[4], res[5]),
            _pack_bf16(res[6], res[7]),
        )

        # ---- RMSNorm: BF16 squares, per-thread sum in element order, warp / CTA / cluster sums ----
        tsum = Float32(0.0)
        for e in cutlass.range_constexpr(VEC):
            tsum = _add(tsum, _bf16r(_mul(res[e], res[e])))
        for k in cutlass.range_constexpr(5):
            tsum = _add(tsum, _shfl_bfly(tsum, Int32(16 >> k)))
        if lane == 0:
            _st_shared_f32(red_addr + Int32(4) * warp, tsum)
        cute.arch.sync_threads()
        bsum = Float32(0.0)
        if lane < num_warps:
            bsum = _ld_shared_f32(red_addr + Int32(4) * lane)
        for k in cutlass.range_constexpr(5):
            bsum = _add(bsum, _shfl_bfly(bsum, Int32(16 >> k)))
        total = bsum
        if const_expr(ctas > 1):
            # Every CTA of the cluster gets every CTA's sum at its own index, then sums in CTA order.
            if tidx < ctas:
                _st_dsmem_f32(csum_addr + Int32(4) * part, Int32(tidx), bsum)
            _cluster_sync()
            total = Float32(0.0)
            for p in cutlass.range_constexpr(ctas):
                total = _add(total, _ld_shared_f32(csum_addr + Int32(4 * p)))
        rstd = _rsqrt(_add(_div_rn(total, Float32(float(self.hidden))), Float32(self.norm_eps)))
        y = cute.make_rmem_tensor((VEC,), Float32)
        for i in cutlass.range_constexpr(4):
            if const_expr(self.two_shot):  # rmsNormLamport: (w * x) * rstd
                y[2 * i] = _bf16r(_mul(_mul(_lo(ww[i]), res[2 * i]), rstd))
                y[2 * i + 1] = _bf16r(_mul(_mul(_hi(ww[i]), res[2 * i + 1]), rstd))
            else:  # one-shot: (x * rstd) * w
                y[2 * i] = _bf16r(_mul(_mul(res[2 * i], rstd), _lo(ww[i])))
                y[2 * i + 1] = _bf16r(_mul(_mul(res[2 * i + 1], rstd), _hi(ww[i])))
        if const_expr(self.norm_out):
            _st_v4(
                Int64(gOut.iterator.toint()) + row_off,
                _pack_bf16(y[0], y[1]),
                _pack_bf16(y[2], y[3]),
                _pack_bf16(y[4], y[5]),
                _pack_bf16(y[6], y[7]),
            )
        if const_expr(self.quant):
            inv = _div_rn(Float32(1.0), gScale[0])
            q01 = _e4m3x2(_mul(y[0], inv), _mul(y[1], inv))
            q23 = _e4m3x2(_mul(y[2], inv), _mul(y[3], inv))
            q45 = _e4m3x2(_mul(y[4], inv), _mul(y[5], inv))
            q67 = _e4m3x2(_mul(y[6], inv), _mul(y[7], inv))
            _st_v2(
                Int64(gQuant.iterator.toint()) + (Int64(tok) * Int64(vecs) + Int64(vi)) * Int64(8),
                q01 | (q23 << 16),
                q45 | (q67 << 16),
            )

        # ---- hand the triple buffer to the next call once every CTA has read the flags ----
        if bid == 0:
            if tidx == 0:
                arrived = _ld_acquire_s32(flags + Int64(32))
                while arrived != grid:
                    arrived = _ld_acquire_s32(flags + Int64(32))
                _st_s32(flags, (cur + Int32(1)) % Int32(3))
                _st_s32(flags + Int64(4), cur)
                if const_expr(self.two_shot):
                    # bytes written per stage (rows of row_bytes: H BF16, or H E4M3 + H / 256 FP32 scales)
                    _st_s32(flags + Int64(12), Int32(2))
                    _st_s32(
                        flags + Int64(16),
                        ((Int32(m) + Int32(nranks - 1)) // Int32(nranks))
                        * Int32(self.row_bytes * nranks),
                    )
                    _st_s32(flags + Int64(20), Int32(m) * Int32(self.row_bytes))
                else:
                    _st_s32(flags + Int64(12), Int32(1))
                    _st_s32(flags + Int64(16), Int32(m) * Int32(self.hidden * nranks * 2))
                    _st_s32(flags + Int64(20), Int32(0))
                _st_s32(flags + Int64(24), Int32(0))
                _st_s32(flags + Int64(28), Int32(0))
                _st_s32(flags + Int64(32), Int32(0))

    # ---- tpc > 1: several tokens per CTA (BF16 two-shot, ctas == 1) ----

    def _mt_load(self, tok, vi, gX, gRouted, gShared, gGate, gRes, topw_ptr, perm_ptr):
        """Issue token ``tok``'s input loads (consumed later): residual, and x or the fold operands."""
        row_off = (Int64(tok) * Int64(self.hidden // VEC) + Int64(vi)) * Int64(16)
        d = {"r": _ld_v4(Int64(gRes.iterator.toint()) + row_off)}
        if self.se_fold:
            if self.moe_topk > 0:
                kbase = Int64(tok) * Int64(self.moe_topk)
                pk = cute.make_rmem_tensor((self.moe_topk,), Int32)
                wk = cute.make_rmem_tensor((self.moe_topk,), Int32)
                for k in range(self.moe_topk):
                    pk[k] = _ld_s32(perm_ptr + (kbase + Int64(k)) * Int64(4))
                    wk[k] = _ld_u16(topw_ptr + (kbase + Int64(k)) * Int64(2))
                d["pk"] = pk
                d["wk"] = wk
            else:
                d["f"] = _ld_v4(Int64(gRouted.iterator.toint()) + row_off)
            d["h"] = _ld_v4(Int64(gShared.iterator.toint()) + row_off)
            d["gw"] = _ld_u16(Int64(gGate.iterator.toint()) + Int64(tok) * Int64(2))
        else:
            d["x"] = _ld_v4(Int64(gX.iterator.toint()) + row_off)
        return d

    def _mt_x(self, d, vi, fc2_ptr):
        """A token's all-reduce input (four BF16x2 words) from its loaded operands: the one-token kernel's MoE
        finalize and shared-expert combine, same operations in the same order."""
        if not self.se_fold:
            return d["x"]
        if self.moe_topk > 0:
            topk = self.moe_topk
            pk = d["pk"]
            wk = d["wk"]
            ev = cute.make_rmem_tensor((topk * 4,), Int32)
            for k in range(topk):
                e0, e1, e2, e3 = _ld_v4(
                    fc2_ptr
                    + (Int64(_max0(pk[k])) * Int64(self.fc2_row) + Int64(vi) * Int64(VEC))
                    * Int64(2)
                )
                ev[4 * k + 0] = e0
                ev[4 * k + 1] = e1
                ev[4 * k + 2] = e2
                ev[4 * k + 3] = e3
            acc = cute.make_rmem_tensor((VEC,), Float32)
            for e in range(VEC):
                acc[e] = Float32(0.0)
            for k in range(topk):
                wf = _lo(wk[k])
                for i in range(4):
                    acc[2 * i] = _sel_ge0(
                        pk[k], _fma(wf, _lo(ev[4 * k + i]), acc[2 * i]), acc[2 * i]
                    )
                    acc[2 * i + 1] = _sel_ge0(
                        pk[k], _fma(wf, _hi(ev[4 * k + i]), acc[2 * i + 1]), acc[2 * i + 1]
                    )
            f = (
                _pack_bf16(acc[0], acc[1]),
                _pack_bf16(acc[2], acc[3]),
                _pack_bf16(acc[4], acc[5]),
                _pack_bf16(acc[6], acc[7]),
            )
        else:
            f = d["f"]
        h = d["h"]
        s = _sigmoid_triton(_lo(d["gw"]))
        return tuple(
            _pack_bf16(_fma(s, _lo(h[i]), _lo(f[i])), _fma(s, _hi(h[i]), _hi(f[i])))
            for i in range(4)
        )

    @cute.kernel
    def kernel_mt(
        self,
        gX: cute.Tensor,
        gRouted: cute.Tensor,
        gShared: cute.Tensor,
        gGate: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        gWs: cute.Tensor,
        fc2_ptr: Int64,
        topw_ptr: Int64,
        perm_ptr: Int64,
        rank: Int32,
    ) -> None:
        """BF16 two-shot, ``tpc`` consecutive tokens per CTA (module docstring); per token the ``kernel`` path's
        arithmetic (two-shot, BF16, ctas == 1)."""
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        nranks: cutlass.Constexpr = self.nranks
        vecs: cutlass.Constexpr = self.hidden // VEC
        num_warps: cutlass.Constexpr = self.num_warps
        tpc: cutlass.Constexpr = self.tpc
        m = cute.size(gRes, mode=[0])
        last = Int32(m - 1)
        vi = tidx  # this thread's 16-byte vector of every token
        warp = tidx // 32
        lane = tidx % 32
        tok0 = bid * Int32(tpc)  # the CTA's first token; token j = min(tok0 + j, m - 1)

        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout((tpc * num_warps,)), byte_alignment=16)
        fl = smem.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        stash = smem.allocate_tensor(Int32, cute.make_layout((tpc * vecs * 4,)), byte_alignment=16)
        red_addr = _smem_addr(red.iterator)
        stash_addr = _smem_addr(stash.iterator) + vi * Int32(
            16
        )  # token j's residual vector at + j x vecs x 16

        # ---- ahead of the dependency wait: host-written addresses and the norm weight (parameters) ----
        tbl = gWs.iterator.toint()
        own = _ld_s64(Int64(tbl))
        mcast = _ld_s64(Int64(tbl) + Int64(8))
        flags = _ld_s64(Int64(tbl) + Int64(16))
        w0, w1, w2, w3 = _ld_v4(Int64(gW.iterator.toint()) + Int64(vi) * Int64(16))
        first = _min_s32(tok0, last)
        peer0 = _ld_s64(Int64(tbl) + Int64(24) + Int64(first % Int32(nranks)) * Int64(8))

        if const_expr(self.use_pdl):
            griddepcontrol_wait()

        # ---- the first token's inputs ahead of the flag loads (as ``kernel``) ----
        ins0 = self._mt_load(first, vi, gX, gRouted, gShared, gGate, gRes, topw_ptr, perm_ptr)
        if tidx == 0:
            lf0, lf1, lf2, lf3 = _ld_volatile_v4(flags)
            lf4, lf5, lf6, lf7 = _ld_volatile_v4(flags + Int64(16))
            fl[0] = lf0
            fl[1] = lf1
            fl[2] = lf2
            fl[3] = lf3
            fl[4] = lf4
            fl[5] = lf5
            fl[6] = lf6
            fl[7] = lf7
        cute.arch.sync_threads()
        cur = fl[0]
        dirty = fl[1]
        buf_bytes = Int64(fl[2])
        dirty_stages = fl[3]
        sb0 = fl[4]
        sb1 = fl[5]
        sb2 = fl[6]
        sb3 = fl[7]
        if tidx == 0:
            _red_release_add_s32(flags + Int64(32), Int32(1))
        data_off = Int64(cur) * buf_bytes
        half = buf_bytes // Int64(2)  # BROADCAST stage offset within the current buffer

        # ---- every token: input -> unicast into its owner's SCATTER stage [t / W][rank][H]; residual -> smem ----
        for j in cutlass.range_constexpr(tpc):
            tj = _min_s32(tok0 + Int32(j), last)
            pj = (
                peer0
                if j == 0
                else _ld_s64(Int64(tbl) + Int64(24) + Int64(tj % Int32(nranks)) * Int64(8))
            )
            ins = (
                ins0
                if j == 0
                else self._mt_load(tj, vi, gX, gRouted, gShared, gGate, gRes, topw_ptr, perm_ptr)
            )
            r0, r1, r2, r3 = ins["r"]
            _st_shared_v4(stash_addr + Int32(j * vecs * 16), r0, r1, r2, r3)
            x0, x1, x2, x3 = self._mt_x(ins, vi, fc2_ptr)
            lt = tj // Int32(nranks)
            _st_v4(
                pj
                + data_off
                + ((Int64(lt) * Int64(nranks) + Int64(rank)) * Int64(vecs) + Int64(vi)) * Int64(16),
                _sanitize(x0),
                _sanitize(x1),
                _sanitize(x2),
                _sanitize(x3),
            )

        # ---- PDL: every input consumed (the residual words went to smem, x to the scatter) by every thread ----
        if const_expr(self.use_pdl):
            cute.arch.sync_threads()
            griddepcontrol_launch_dependents()

        # ---- clear this thread's share of the previous call's buffer (every dirty stage) ----
        neg0 = Int32(_NEG_ZERO)
        fidx = Int64(bid) * Int64(self.threads) + Int64(tidx)
        fstride = Int64(grid) * Int64(self.threads)
        for s_i in cutlass.range_constexpr(4):
            if Int32(s_i) < dirty_stages:
                stage_bytes = buf_bytes // Int64(dirty_stages)
                stage = own + (Int64(dirty) * Int64(dirty_stages) + Int64(s_i)) * stage_bytes
                sb = sb0 if s_i == 0 else (sb1 if s_i == 1 else (sb2 if s_i == 2 else sb3))
                clear_vecs = (Int64(sb) + Int64(15)) // Int64(16)
                c = fidx
                while c < clear_vecs:
                    _st_v4(stage + c * Int64(16), neg0, neg0, neg0, neg0)
                    c = c + fstride

        # ---- owner (consecutive tokens: at most one per CTA up to 8 tokens): wait for every rank's vector, FP32 sum
        # in rank order -> BF16, multicast into every rank's BROADCAST stage ----
        got = cute.make_rmem_tensor((nranks * 4,), Int32)
        for j in cutlass.range_constexpr(tpc):
            to = _min_s32(tok0 + Int32(j), last)
            if to % Int32(nranks) == rank:
                lto = to // Int32(nranks)
                pend_o = Int32(1)
                while pend_o != 0:
                    pend_o = Int32(0)
                    for r in cutlass.range_constexpr(nranks):
                        a1, b1, c1, d1 = _ld_volatile_v4(
                            own
                            + data_off
                            + ((Int64(lto) * Int64(nranks) + Int64(r)) * Int64(vecs) + Int64(vi))
                            * Int64(16)
                        )
                        got[r * 4 + 0] = a1
                        got[r * 4 + 1] = b1
                        got[r * 4 + 2] = c1
                        got[r * 4 + 3] = d1
                        if a1 == neg0:
                            pend_o = Int32(1)
                        if b1 == neg0:
                            pend_o = Int32(1)
                        if c1 == neg0:
                            pend_o = Int32(1)
                        if d1 == neg0:
                            pend_o = Int32(1)
                rsum = cute.make_rmem_tensor((4,), Int32)
                for i in cutlass.range_constexpr(4):
                    slo = Float32(0.0)
                    shi = Float32(0.0)
                    for r in cutlass.range_constexpr(nranks):
                        slo = _add(slo, _lo(got[r * 4 + i]))
                        shi = _add(shi, _hi(got[r * 4 + i]))
                    rsum[i] = _sanitize(_pack_bf16(slo, shi))
                _st_v4(
                    mcast + data_off + half + (Int64(to) * Int64(vecs) + Int64(vi)) * Int64(16),
                    rsum[0],
                    rsum[1],
                    rsum[2],
                    rsum[3],
                )

        # ---- every token: wait for the reduced vector, + residual, RMSNorm (rmsNormLamport: (w * x) * rstd) ----
        ww = (w0, w1, w2, w3)
        for j in cutlass.range_constexpr(tpc):
            tb = _min_s32(tok0 + Int32(j), last)
            bidx = Int64(tb) * Int64(vecs) + Int64(vi)
            pend_b = Int32(1)
            while pend_b != 0:
                pend_b = Int32(0)
                a2, b2, c2, d2 = _ld_volatile_v4(own + data_off + half + bidx * Int64(16))
                got[0] = a2
                got[1] = b2
                got[2] = c2
                got[3] = d2
                if a2 == neg0:
                    pend_b = Int32(1)
                if b2 == neg0:
                    pend_b = Int32(1)
                if c2 == neg0:
                    pend_b = Int32(1)
                if d2 == neg0:
                    pend_b = Int32(1)
            q0, q1, q2, q3 = _ld_shared_v4(stash_addr + Int32(j * vecs * 16))
            rw = (q0, q1, q2, q3)
            res = cute.make_rmem_tensor((VEC,), Float32)
            for i in cutlass.range_constexpr(4):
                res[2 * i] = _bf16r(_add(_lo(got[i]), _lo(rw[i])))
                res[2 * i + 1] = _bf16r(_add(_hi(got[i]), _hi(rw[i])))
            _st_v4(
                Int64(gResOut.iterator.toint()) + bidx * Int64(16),
                _pack_bf16(res[0], res[1]),
                _pack_bf16(res[2], res[3]),
                _pack_bf16(res[4], res[5]),
                _pack_bf16(res[6], res[7]),
            )
            tsum = Float32(0.0)
            for e in cutlass.range_constexpr(VEC):
                tsum = _add(tsum, _bf16r(_mul(res[e], res[e])))
            for k in cutlass.range_constexpr(5):
                tsum = _add(tsum, _shfl_bfly(tsum, Int32(16 >> k)))
            if lane == 0:
                _st_shared_f32(red_addr + Int32(4 * j * num_warps) + Int32(4) * warp, tsum)
            cute.arch.sync_threads()
            bsum = Float32(0.0)
            if lane < num_warps:
                bsum = _ld_shared_f32(red_addr + Int32(4 * j * num_warps) + Int32(4) * lane)
            for k in cutlass.range_constexpr(5):
                bsum = _add(bsum, _shfl_bfly(bsum, Int32(16 >> k)))
            rstd = _rsqrt(_add(_div_rn(bsum, Float32(float(self.hidden))), Float32(self.norm_eps)))
            y = cute.make_rmem_tensor((VEC,), Float32)
            for i in cutlass.range_constexpr(4):
                y[2 * i] = _bf16r(_mul(_mul(_lo(ww[i]), res[2 * i]), rstd))
                y[2 * i + 1] = _bf16r(_mul(_mul(_hi(ww[i]), res[2 * i + 1]), rstd))
            if const_expr(self.norm_out):
                _st_v4(
                    Int64(gOut.iterator.toint()) + bidx * Int64(16),
                    _pack_bf16(y[0], y[1]),
                    _pack_bf16(y[2], y[3]),
                    _pack_bf16(y[4], y[5]),
                    _pack_bf16(y[6], y[7]),
                )
            if const_expr(self.quant):
                inv = _div_rn(Float32(1.0), gScale[0])
                q01 = _e4m3x2(_mul(y[0], inv), _mul(y[1], inv))
                q23 = _e4m3x2(_mul(y[2], inv), _mul(y[3], inv))
                q45 = _e4m3x2(_mul(y[4], inv), _mul(y[5], inv))
                q67 = _e4m3x2(_mul(y[6], inv), _mul(y[7], inv))
                _st_v2(
                    Int64(gQuant.iterator.toint()) + bidx * Int64(8),
                    q01 | (q23 << 16),
                    q45 | (q67 << 16),
                )

        # ---- hand the triple buffer to the next call once every CTA has read the flags ----
        if bid == 0:
            if tidx == 0:
                arrived = _ld_acquire_s32(flags + Int64(32))
                while arrived != grid:
                    arrived = _ld_acquire_s32(flags + Int64(32))
                _st_s32(flags, (cur + Int32(1)) % Int32(3))
                _st_s32(flags + Int64(4), cur)
                _st_s32(flags + Int64(12), Int32(2))
                _st_s32(
                    flags + Int64(16),
                    ((Int32(m) + Int32(nranks - 1)) // Int32(nranks))
                    * Int32(self.row_bytes * nranks),
                )
                _st_s32(flags + Int64(20), Int32(m) * Int32(self.row_bytes))
                _st_s32(flags + Int64(24), Int32(0))
                _st_s32(flags + Int64(28), Int32(0))
                _st_s32(flags + Int64(32), Int32(0))
