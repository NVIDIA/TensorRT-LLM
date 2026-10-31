# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch MoE decode front (M <= 32 tokens, 512 experts, top-10).

Router GEMM + top-k routing + trtllm-gen pre-routed tables + NVFP4 quantization of the MoE input. Outputs are
bitwise those of the low-M router GEMM + ``RouteTables``, except the row order inside an expert's block (token
order here; the expert GEMM rows are independent and the finalize gathers by expanded index).

* GEMM CTAs (``num_experts / 8``): :class:`LowMMmaGemmKernel`'s BF16 router with the same tactic, so the logits
  are bit-identical. Each also writes its logits as valid-tagged words (``0x10000 | bf16``) into a persistent
  staging buffer that must be all zero between launches.
* Router CTA (block 0): polls its token rows (``ld.volatile``) until every word carries the tag, then zeroes them.
  Keys are ``twiddled bf16 << 16 | 65535 - expert`` (ties -> smaller expert, as ``reduceTopK``); renormalization
  copies the Triton route kernel's arithmetic (FTZ libdevice ``expf``, its summation tree, ``div.rn.ftz``).
* Quant CTAs: NVFP4 quantization of x (linear scale layout), ``fp4_quantize`` arithmetic.

PDL: before ``griddepcontrol.wait`` only static data (router weights, global scale) is read and nothing global
is written; every CTA triggers after a CTA barrier that follows its last global access. Above 32 tokens use
:mod:`moe_route_front`.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .low_m_mma_gemm import CHUNK, N_TILE, LowMMmaGemmKernel, _g2r_atom

try:
    from cutlass.memory import SmemAllocator as _SmemAllocator
except ImportError:  # older CuTe DSL
    from cutlass.utils import SmemAllocator as _SmemAllocator


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


def _i32(x, loc=None, ip=None):
    return Int32(x).ir_value(loc=loc, ip=ip)


def _i64(x, loc=None, ip=None):
    return Int64(x).ir_value(loc=loc, ip=ip)


def _f32(x, loc=None, ip=None):
    return Float32(x).ir_value(loc=loc, ip=ip)


def _struct(n: int):
    return llvm.StructType.get_literal([T.i32()] * n)


def _unpack(out, n: int, loc=None, ip=None):
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(n))


# PDL ops clobber memory so no memory access moves across them.
@dsl_user_op
def _gdc_wait(*, loc=None, ip=None) -> None:
    _asm(None, [], "griddepcontrol.wait;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _gdc_launch(*, loc=None, ip=None) -> None:
    _asm(None, [], "griddepcontrol.launch_dependents;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _ld_v4(addr: Int64, *, loc=None, ip=None):
    out = _asm(
        _struct(4),
        [_i64(addr, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, loc, ip)


@dsl_user_op
def _ld_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_i64(addr, loc, ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_b32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(value, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_b16(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(value, loc, ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_u8(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(value, loc, ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u8 [$0], h;\n\t}",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_v2(addr: Int64, w0: Int32, w1: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(w0, loc, ip), _i32(w1, loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _smem_addr(ptr, *, loc=None, ip=None) -> Int32:
    return Int32(ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def _ld_shared(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(addr, loc, ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(value, loc, ip)],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_or_shared(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(value, loc, ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _redux_max_u32(x: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_u32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(a, loc, ip), _i32(b, loc, ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_eq(x: Int32, y: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``a if x == y else b``."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip), _i32(y, loc, ip), _i32(a, loc, ip), _i32(b, loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.eq.b32 p, $1, $2;\n\tselp.b32 $0, $3, $4, p;\n\t}",
            "=r,r,r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_eq_f(x: Int32, y: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a if x == y else b`` (FP32 operands)."""
    return Float32(
        _asm(
            T.f32(),
            [_i32(x, loc, ip), _i32(y, loc, ip), _f32(a, loc, ip), _f32(b, loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.eq.b32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n\t}",
            "=f,r,r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_lt(x: Int32, y: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """``a if x < y else b`` (unsigned compare: a negative x is never below y)."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip), _i32(y, loc, ip), _i32(a, loc, ip), _i32(b, loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.lt.u32 p, $1, $2;\n\tselp.b32 $0, $3, $4, p;\n\t}",
            "=r,r,r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _popc(x: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [_i32(x, loc, ip)], "popc.b32 $0, $1;", "=r,r", False, loc=loc, ip=ip)
    )


@dsl_user_op
def _bit(t: Int32, *, loc=None, ip=None) -> Int32:
    """``1 << t`` (t in [0, 31])."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(t, loc, ip)],
            "{\n\t.reg .b32 one;\n\tmov.b32 one, 1;\n\tshl.b32 $0, one, $1;\n\t}",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _below(t: Int32, *, loc=None, ip=None) -> Int32:
    """``(1 << t) - 1`` as a bit mask (t in [0, 31]): the tokens before t."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(t, loc, ip)],
            "{\n\t.reg .b32 a;\n\tmov.b32 a, 0xffffffff;\n\tshl.b32 a, a, $1;\n\t"
            "not.b32 $0, a;\n\t}",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _scan_add(x: Int32, delta: int, *, loc=None, ip=None) -> Int32:
    """One step of a warp inclusive scan: ``x + x[lane - delta]`` for lanes >= delta, else ``x``."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip)],
            "{\n\t.reg .b32 y;\n\t.reg .pred p;\n\t"
            f"shfl.sync.up.b32 y|p, $1, {delta}, 0, 0xffffffff;\n\t"
            "@p add.s32 y, y, $1;\n\t@!p mov.b32 y, $1;\n\tmov.b32 $0, y;\n\t}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


# Routing arithmetic: bitwise that of the Triton route kernel.
@dsl_user_op
def _key_val(key: Int32, *, loc=None, ip=None) -> Float32:
    """The FP32 value of a key's BF16 logit (inverse twiddle)."""
    return Float32(
        _asm(
            T.f32(),
            [_i32(key, loc, ip)],
            "{\n\t.reg .b32 tw, m;\n\t.reg .pred p;\n\tshr.u32 tw, $1, 16;\n\t"
            "and.b32 m, tw, 32768;\n\tsetp.ne.u32 p, m, 0;\n\tselp.b32 m, 32768, 65535, p;\n\t"
            "xor.b32 tw, tw, m;\n\tshl.b32 tw, tw, 16;\n\tmov.b32 $0, tw;\n\t}",
            "=f,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _exp_tr(x: Float32, *, loc=None, ip=None) -> Float32:
    """libdevice ``__nv_expf`` with FTZ, as Triton compiles ``libdevice.exp`` (route kernel PTX)."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(x, loc, ip)],
            "{\n\t.reg .b32 t, j, k, r, e, s, c0, c1, c2, c3, c4, c5, c6;\n\t"
            "mov.b32 c0, 0f3BBB989D;\n\tmov.b32 c1, 0f3F000000;\n\t"
            "mov.b32 c2, 0f437C0000;\n\tmov.b32 c3, 0f4B400001;\n\t"
            "mov.b32 c4, 0fCB40007F;\n\tmov.b32 c5, 0f3FB8AA3B;\n\tmov.b32 c6, 0f32A57060;\n\t"
            "fma.rn.ftz.f32 t, $1, c0, c1;\n\t"
            "cvt.ftz.sat.f32.f32 t, t;\n\t"
            "fma.rm.ftz.f32 j, t, c2, c3;\n\t"
            "add.f32 k, j, c4;\n\t"
            "neg.f32 k, k;\n\t"
            "fma.rn.ftz.f32 r, $1, c5, k;\n\t"
            "fma.rn.ftz.f32 r, $1, c6, r;\n\t"
            "shl.b32 s, j, 23;\n\t"
            "ex2.approx.ftz.f32 e, r;\n\t"
            "mul.f32 $0, e, s;\n\t}",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sub(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "sub.f32 $0, $1, $2;",
            "=f,f,f",
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
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "add.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_bf16(a: Float32, b: Float32, *, loc=None, ip=None) -> Int32:
    """BF16 bits of ``a / b`` (``div.rn.ftz.f32``, then round to nearest even)."""
    return Int32(
        _asm(
            T.i32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "{\n\t.reg .b32 q;\n\t.reg .b16 h;\n\tdiv.rn.ftz.f32 q, $1, $2;\n\t"
            "cvt.rn.bf16.f32 h, q;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nvfp4_block(
    w0: Int32,
    w1: Int32,
    w2: Int32,
    w3: Int32,
    w4: Int32,
    w5: Int32,
    w6: Int32,
    w7: Int32,
    gs: Float32,
    *,
    loc=None,
    ip=None,
):
    """NVFP4 quantization of one 16-element BF16 block (element 2i in the low half of word i): the 8 packed
    E2M1 bytes as two words (byte j = elements 2j | 2j + 1 << 4) and the E4M3 scale byte. Arithmetic of
    ``_nvfp4_quant_row`` (route kernel PTX): vmax = max |x|, sf = e4m3(gs * (vmax * rcp(6))),
    scale = rcp(f32(sf) * rcp(gs)) (0 if vmax == 0), cvt.rn.satfinite.e2m1x2(x * scale)."""
    unpack = []
    for i in range(8):
        unpack.append(f"shl.b32 f{2 * i}, ${3 + i}, 16;")
        unpack.append(f"and.b32 f{2 * i + 1}, ${3 + i}, 0xffff0000;")
    amax = ["abs.f32 m, f0;"]
    for j in range(1, 16):
        amax.append(f"abs.f32 a, f{j};")
        amax.append("max.f32 m, m, a;")
    pack = []
    for j in range(8):
        word = "u" if j < 4 else "v"
        pack.append(f"mul.f32 y0, osc, f{2 * j};")
        pack.append(f"mul.f32 y1, osc, f{2 * j + 1};")
        pack.append("cvt.rn.satfinite.e2m1x2.f32 q, y1, y0;")
        pack.append("cvt.u16.u8 hq, q;")
        pack.append("cvt.u32.u16 c, hq;")
        if j % 4 == 0:
            pack.append(f"mov.b32 {word}, c;")
        else:
            pack.append(f"shl.b32 c, c, {8 * (j % 4)};")
            pack.append(f"or.b32 {word}, {word}, c;")
    text = "\n\t".join(
        [
            "{",
            ".reg .b32 " + ", ".join(f"f{j}" for j in range(16)) + ";",
            ".reg .b32 a, m, t, s, r6, rg, sff, osc, y0, y1, c, u, v, sfw;",
            ".reg .b16 sfh, h0, h1, hq;",
            ".reg .b8 q;",
            ".reg .pred nz;",
        ]
        + unpack
        + amax
        + [
            "mov.b32 r6, 0f40C00000;",
            "rcp.approx.ftz.f32 r6, r6;",
            "mul.f32 t, r6, m;",
            "mul.f32 s, t, $11;",
            "cvt.rn.satfinite.e4m3x2.f32 sfh, 0f00000000, s;",
            "cvt.rn.f16x2.e4m3x2 sfw, sfh;",
            "mov.b32 {h0, h1}, sfw;",
            "cvt.f32.f16 sff, h0;",
            "rcp.approx.ftz.f32 rg, $11;",
            "mul.f32 t, rg, sff;",
            "rcp.approx.ftz.f32 osc, t;",
            "setp.neu.f32 nz, m, 0f00000000;",
            "selp.f32 osc, osc, 0f00000000, nz;",
        ]
        + pack
        + ["mov.b32 $0, u;", "mov.b32 $1, v;", "cvt.u32.u16 c, sfh;", "and.b32 $2, c, 255;", "}"]
    )
    out = _asm(
        _struct(3),
        [_i32(w, loc, ip) for w in (w0, w1, w2, w3, w4, w5, w6, w7)] + [_f32(gs, loc, ip)],
        text,
        "=r,=r,=r,r,r,r,r,r,r,r,r,f",
        False,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 3, loc, ip)


@dsl_user_op
def _min_u32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(a, loc, ip), _i32(b, loc, ip)],
            "min.u32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _key1(word: Int32, idx: Int32, *, loc=None, ip=None) -> Int32:
    """Top-k key of the BF16 logit in the low half of ``word`` (expert ``idx``): ``twiddle(bits) << 16 |
    (65535 - expert)``, unsigned order = value order, ties -> smaller expert. twiddle: negative -> bits ^ 0xFFFF,
    else bits | 0x8000."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(word, loc, ip), _i32(idx, loc, ip)],
            "{\n\t.reg .b32 bv, t, m;\n\t.reg .pred p;\n\t"
            "and.b32 bv, $1, 65535;\n\t"
            "and.b32 t, bv, 32768;\n\tsetp.ne.u32 p, t, 0;\n\tselp.b32 m, 65535, 32768, p;\n\t"
            "xor.b32 bv, bv, m;\n\tshl.b32 bv, bv, 16;\n\t"
            "mov.b32 t, 65535;\n\tsub.u32 t, t, $2;\n\tor.b32 $0, bv, t;\n\t}",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _vote_all_bit16(x: Int32, *, loc=None, ip=None) -> Int32:
    """1 if bit 16 of ``x`` is set in every lane of the warp, else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip)],
            "{\n\t.reg .b32 a;\n\t.reg .pred p, q;\n\tand.b32 a, $1, 65536;\n\tsetp.ne.u32 p, a, 0;\n\t"
            "vote.sync.all.pred q, p, 0xffffffff;\n\tselp.u32 $0, 1, 0, q;\n\t}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_volatile_v4(addr: Int64, *, loc=None, ip=None):
    out = _asm(
        _struct(4),
        [_i64(addr, loc, ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, loc, ip)


@dsl_user_op
def _st_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_i64(addr, loc, ip)] + [_i32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tagged_bf16(x: Float32, *, loc=None, ip=None) -> Int32:
    """Staging word of a logit: ``0x10000 | bf16_rn(x)`` (bit 16 = written in this launch)."""
    return Int32(
        _asm(
            T.i32(),
            [_f32(x, loc, ip)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 w;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.u32.u16 w, h;\n\t"
            "or.b32 $0, w, 65536;\n\t}",
            "=r,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


def _bitonic_pairs(n: int):
    """Compare-exchange stages of a descending bitonic sort of n (power of two) values: (i, j, i_gets_max)."""
    pairs = []
    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            for i in range(n):
                partner = i ^ j
                if partner > i:
                    pairs.append((i, partner, (i & k) == 0))
            j //= 2
        k *= 2
    return pairs


_SORT16 = _bitonic_pairs(16)


def max_num_tiles(num_tokens: int, top_k: int, num_local_experts: int, tile: int) -> int:
    """Mirror of getMaxNumCgasInBatchDim (trtllmGen blockScaleMoe runner.h)."""
    remaining = num_tokens * top_k
    filled = min(num_local_experts, remaining)
    remaining -= filled
    tiles = filled
    if remaining > 0:
        tiles += remaining // tile
    return tiles


class MoeDecodeFrontKernel(LowMMmaGemmKernel):
    """``num_rows`` (<= 32) tokens: router logits [M, E] (BF16), NVFP4 x, top-k ids / BF16 weights and the
    trtllm-gen pre-routed tables (flat int32 buffer in ``prerouted_table_sizes`` order) in one launch (module
    doc). ``k_warps``: the low-M router tactic of ``num_rows`` (32 for M <= 8, else 16), which fixes the GEMM's
    summation order. ``a_stride``: row stride of x in elements. Local experts [local_start, local_start +
    num_local) (local_start is a runtime argument). ``sync_ptr``: a persistent int32 [32, E] staging buffer, all
    zero between launches; block 0 is the router CTA."""

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        k_warps: int,
        use_pdl: bool,
        num_experts: int,
        top_k: int,
        num_local: int,
        tile: int,
        a_stride: int,
    ) -> None:
        super().__init__(
            num_rows=num_rows,
            k_extent=k_extent,
            k_warps=k_warps,
            use_pdl=use_pdl,
            n_bf16=num_experts,
            n_tiles_bf16=1,
        )
        threads = 32 * k_warps
        if num_rows > 32:
            raise ValueError("the token masks hold at most 32 tokens")
        if top_k != 10:
            raise ValueError("the renormalization tree is the route kernel's top-10 tree")
        if num_experts != 32 * 16:
            raise ValueError("one warp per token holds 16 logits per lane: 512 experts")
        if not 1 <= num_local <= min(num_experts, threads) or num_local % 32:
            raise ValueError(
                f"num_local={num_local} must be a multiple of 32 and <= {min(num_experts, threads)}"
            )
        if tile & (tile - 1) or not 8 <= tile <= 256:
            raise ValueError(f"tile={tile} must be a power of two in [8, 256]")
        if num_rows * top_k > threads:
            raise ValueError("one thread per expanded pick")
        if a_stride % 8 or a_stride < k_extent:
            raise ValueError("x rows must be 16-byte aligned")
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local = num_local
        self.tile = tile
        self.a_stride = a_stride
        self.threads = threads
        self.tok_per_warp = -(-num_rows // k_warps)
        if self.tok_per_warp > 2:
            raise ValueError("at most two tokens per warp")
        self.gemm_ctas = num_experts // N_TILE
        self.gemm_cta0 = 1  # block 0 is the router CTA
        self.blocks_per_row = k_extent // 16
        self.quant_ctas = -(-(num_rows * self.blocks_per_row) // threads)
        self.quant_cta0 = self.gemm_cta0 + self.gemm_ctas
        self.ctas = self.quant_cta0 + self.quant_ctas
        self.max_ctas = max_num_tiles(num_rows, top_k, num_local, tile)
        self.max_ctas_per_expert = -(-num_rows // tile)
        # prerouted_table_sizes order: e2p [M*K], p2t [(max_ctas + 1) * tile], cta -> expert, cta -> limit,
        # CTA count, padded rows.
        self.off_p2t = num_rows * top_k
        self.off_cte = self.off_p2t + (self.max_ctas + 1) * tile
        self.off_ctl = self.off_cte + self.max_ctas
        self.off_nct = self.off_ctl + self.max_ctas
        self.off_npd = self.off_nct + 1
        self.picks_pad = ((num_rows * top_k + 31) // 32) * 32

    @cute.jit
    def __call__(
        self,
        gA16: cute.Tensor,
        gB2: cute.Tensor,
        gC: cute.Tensor,
        gs_ptr: Int64,
        xq_ptr: Int64,
        xsf_ptr: Int64,
        ids_ptr: Int64,
        wts_ptr: Int64,
        tab_ptr: Int64,
        sync_ptr: Int64,
        local_start: Int32,
        stream,
    ) -> None:
        copy_a16 = _g2r_atom(BFloat16, streaming=False)
        copy_b2 = _g2r_atom(BFloat16, streaming=True)
        smem = (
            self.k_warps * self.row_tiles * 16 * N_TILE * 4
            + 2 * self.num_local * 4
            + self.picks_pad * 4
            + 256
        )
        self.kernel(
            gA16,
            gB2,
            gC,
            gs_ptr,
            xq_ptr,
            xsf_ptr,
            ids_ptr,
            wts_ptr,
            tab_ptr,
            sync_ptr,
            local_start,
            copy_a16,
            copy_b2,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            smem=smem,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA16: cute.Tensor,
        gB2: cute.Tensor,
        gC: cute.Tensor,
        gs_ptr: Int64,
        xq_ptr: Int64,
        xsf_ptr: Int64,
        ids_ptr: Int64,
        wts_ptr: Int64,
        tab_ptr: Int64,
        sync_ptr: Int64,
        local_start: Int32,
        copy_a16: cute.CopyAtom,
        copy_b2: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = _SmemAllocator()
        red = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (self.k_warps, self.row_tiles * 16, N_TILE),
                stride=(self.row_tiles * 16 * N_TILE, N_TILE, 1),
            ),
            byte_alignment=16,
        )
        s_mask = smem.allocate_tensor(Int32, cute.make_layout((self.num_local,)), byte_alignment=16)
        s_offs = smem.allocate_tensor(Int32, cute.make_layout((self.num_local,)), byte_alignment=16)
        s_pick = smem.allocate_tensor(Int32, cute.make_layout((self.picks_pad,)), byte_alignment=16)
        sm = (
            _smem_addr(s_mask.iterator),
            _smem_addr(s_offs.iterator),
            _smem_addr(s_pick.iterator),
        )
        if bidx == 0:
            self._router_cta(ids_ptr, wts_ptr, tab_ptr, sync_ptr, local_start, sm, tidx)
        elif bidx < self.quant_cta0:
            self._gemm_cta(
                gA16, gB2, gC, sync_ptr, red, copy_a16, copy_b2, tidx, bidx - self.gemm_cta0
            )
        else:
            self._quant_cta(gA16, gs_ptr, xq_ptr, xsf_ptr, tidx, bidx - self.quant_cta0)

    @cute.jit
    def _gemm_cta(
        self,
        gA16: cute.Tensor,
        gB2: cute.Tensor,
        gC: cute.Tensor,
        sync_ptr: Int64,
        red: cute.Tensor,
        copy_a16: cute.CopyAtom,
        copy_b2: cute.CopyAtom,
        tidx,
        cta,
    ) -> None:
        """The low-M router's BF16 CTA (LowMMmaGemmKernel._cta, BF16 segment, one tile, no split-K / swap), then
        the hand-off of its logits: valid-tagged words into the staging buffer (the router CTA polls them)."""
        warp = tidx // 32
        lane = tidx % 32
        r = lane // 4
        q = lane % 4
        num_rows: cutlass.Constexpr = self.num_rows
        row_tiles: cutlass.Constexpr = self.row_tiles
        k_warps: cutlass.Constexpr = self.k_warps
        chunks: cutlass.Constexpr = self.chunks_per_warp
        n0 = cta * N_TILE
        k_warp0 = warp * (chunks * CHUNK)

        # Router weights are static: loaded before the PDL wait.
        w8 = cute.make_rmem_tensor(cute.make_layout((chunks, 16), stride=(16, 1)), Uint8)
        w16 = cute.make_rmem_tensor(cute.make_layout((2 * chunks, 8), stride=(8, 1)), BFloat16)
        gB2_vec = cute.logical_divide(gB2, (None, 8))
        b_row = n0 + r
        for c in cutlass.range_constexpr(chunks):
            k0 = k_warp0 + c * CHUNK + 16 * q
            cute.copy(copy_b2, gB2_vec[b_row, (None, k0 // 8)], w16[2 * c, None])
            cute.copy(copy_b2, gB2_vec[b_row, (None, k0 // 8 + 1)], w16[2 * c + 1, None])
        alpha = Float32(1.0)

        if const_expr(self.use_pdl):
            _gdc_wait()

        acc = cute.make_rmem_tensor(cute.make_layout((row_tiles, 4), stride=(4, 1)), Float32)
        acc.fill(0.0)
        self._mainloop_bf16(acc, gA16, w8, w16, copy_a16, k_warp0, r, q, 1, False)

        # Split-K reduction in warp order, as the low-M kernel (keeps the logits bitwise).
        for t in cutlass.range_constexpr(row_tiles):
            col = 2 * q
            red[warp, 16 * t + r, col] = acc[t, 0]
            red[warp, 16 * t + r, col + 1] = acc[t, 1]
            red[warp, 16 * t + r + 8, col] = acc[t, 2]
            red[warp, 16 * t + r + 8, col + 1] = acc[t, 3]
        cute.arch.sync_threads()
        for idx in cutlass.range(tidx, num_rows * N_TILE, 32 * k_warps):
            out_row = idx // N_TILE
            out_col = idx % N_TILE
            value = Float32(0.0)
            for w in cutlass.range_constexpr(k_warps):
                value = value + red[w, out_row, out_col]
            value = value * alpha
            gC[out_row, n0 + out_col] = value.to(gC.element_type)
            _st_b32(
                sync_ptr
                + (Int64(out_row) * Int64(self.num_experts) + Int64(n0 + out_col)) * Int64(4),
                _tagged_bf16(value),
            )
        # This barrier follows the CTA's last global access.
        cute.arch.sync_threads()
        if const_expr(self.use_pdl):
            _gdc_launch()

    @cute.jit
    def _router_cta(
        self,
        ids_ptr: Int64,
        wts_ptr: Int64,
        tab_ptr: Int64,
        sync_ptr: Int64,
        local_start: Int32,
        sm,
        tidx,
    ) -> None:
        """Block 0 routes from the staging buffer as the GEMM CTAs publish their logits."""
        if tidx < self.num_local:
            _st_shared(sm[0] + tidx * 4, Int32(0))  # token masks (shared memory, before the wait)
        if const_expr(self.use_pdl):
            _gdc_wait()
        self._route(ids_ptr, wts_ptr, tab_ptr, sync_ptr, local_start, sm, tidx)
        if const_expr(self.use_pdl):
            _gdc_launch()

    @cute.jit
    def _route(
        self,
        ids_ptr: Int64,
        wts_ptr: Int64,
        tab_ptr: Int64,
        sync_ptr: Int64,
        local_start: Int32,
        sm,
        tidx,
    ) -> None:
        """Routes every token and writes the tables."""
        mask_addr, offs_addr, pick_addr = sm[0], sm[1], sm[2]
        warp = tidx // 32
        lane = tidx % 32
        num_rows: cutlass.Constexpr = self.num_rows
        top_k: cutlass.Constexpr = self.top_k
        num_local: cutlass.Constexpr = self.num_local
        tile: cutlass.Constexpr = self.tile

        # (1) per token: top-k, renormalized weights, local picks into the token masks
        self._tokens(ids_ptr, wts_ptr, sync_ptr, local_start, mask_addr, pick_addr, warp, lane)
        cute.arch.sync_threads()

        # (2) warp 0: per-expert rows / CTAs, exclusive scan of the CTA counts, CTA tables
        if warp == 0:
            self._tables_warp(tab_ptr, mask_addr, offs_addr, lane)
        cute.arch.sync_threads()

        # (3) one thread per expanded pick: expanded -> permuted, permuted -> token
        if tidx < num_rows * top_k:
            p_lid = _ld_shared(pick_addr + tidx * 4)
            p_tok = tidx // top_k
            p_lid0 = cutlass.max(p_lid, Int32(0))
            p_rank = _popc(_ld_shared(mask_addr + p_lid0 * 4) & _below(p_tok))
            p_row = _ld_shared(offs_addr + p_lid0 * 4) * Int32(tile) + p_rank
            perm = _sel_lt(p_lid, Int32(num_local), p_row, Int32(-1))
            if perm >= Int32(0):
                _st_b32(tab_ptr + Int64(self.off_p2t + perm) * Int64(4), p_tok)
            _st_b32(tab_ptr + Int64(tidx) * Int64(4), perm)
        cute.arch.sync_threads()

    @cute.jit
    def _tokens(
        self,
        ids_ptr: Int64,
        wts_ptr: Int64,
        sync_ptr: Int64,
        local_start: Int32,
        mask_addr: Int32,
        pick_addr: Int32,
        warp,
        lane,
    ) -> None:
        """Token phase: tokens warp and warp + k_warps (when M > k_warps) interleaved; per token and lane the 16
        keys are bitonic-sorted once and each round pops the head of the lane that owned the maximum."""
        num_rows: cutlass.Constexpr = self.num_rows
        top_k: cutlass.Constexpr = self.top_k
        num_experts: cutlass.Constexpr = self.num_experts
        tpw: cutlass.Constexpr = self.tok_per_warp
        if warp < num_rows:
            # tokens of this warp (the second one clamped to the first when it does not exist; never stored)
            tok1 = warp
            if const_expr(tpw == 2):
                tok1 = _sel_lt(warp + self.k_warps, Int32(num_rows), warp + self.k_warps, warp)
            words = cute.make_rmem_tensor(cute.make_layout((16 * tpw,)), Int32)
            # Poll until every word carries the valid bit (bit 16), then clear the rows for the next launch.
            src0 = sync_ptr + (Int64(warp) * Int64(num_experts) + Int64(lane) * Int64(16)) * Int64(
                4
            )
            src1 = sync_ptr + (Int64(tok1) * Int64(num_experts) + Int64(lane) * Int64(16)) * Int64(
                4
            )
            pending = Int32(1)
            while pending != 0:
                for v in cutlass.range_constexpr(4):
                    b0, b1, b2, b3 = _ld_volatile_v4(src0 + Int64(16 * v))
                    words[4 * v] = b0
                    words[4 * v + 1] = b1
                    words[4 * v + 2] = b2
                    words[4 * v + 3] = b3
                if const_expr(tpw == 2):
                    for v in cutlass.range_constexpr(4):
                        c0, c1, c2, c3 = _ld_volatile_v4(src1 + Int64(16 * v))
                        words[16 + 4 * v] = c0
                        words[16 + 4 * v + 1] = c1
                        words[16 + 4 * v + 2] = c2
                        words[16 + 4 * v + 3] = c3
                acc_and = words[0]
                for j in cutlass.range_constexpr(16 * tpw - 1):
                    acc_and = acc_and & words[j + 1]
                pending = Int32(1) - _vote_all_bit16(acc_and)
            zero = Int32(0)
            for v in cutlass.range_constexpr(4):
                _st_v4(src0 + Int64(16 * v), zero, zero, zero, zero)
            if const_expr(tpw == 2):
                if tok1 != warp:
                    for v in cutlass.range_constexpr(4):
                        _st_v4(src1 + Int64(16 * v), zero, zero, zero, zero)
            e0 = lane * 16
            keys = cute.make_rmem_tensor(cute.make_layout((16 * tpw,)), Int32)
            for r in cutlass.range_constexpr(tpw):
                for j in cutlass.range_constexpr(16):
                    keys[16 * r + j] = _key1(words[16 * r + j], e0 + j)
                for pi in cutlass.range_constexpr(len(_SORT16)):
                    i, jj, imax = _SORT16[pi]
                    x = keys[16 * r + i]
                    y = keys[16 * r + jj]
                    hi = _max_u32(x, y)
                    lo = _min_u32(x, y)
                    if const_expr(imax):
                        keys[16 * r + i] = hi
                        keys[16 * r + jj] = lo
                    else:
                        keys[16 * r + i] = lo
                        keys[16 * r + jj] = hi
            sel = cute.make_rmem_tensor(cute.make_layout((top_k * tpw,)), Int32)
            val = cute.make_rmem_tensor(cute.make_layout((top_k * tpw,)), Float32)
            for k in cutlass.range_constexpr(top_k):
                for r in cutlass.range_constexpr(tpw):
                    head = keys[16 * r]
                    g = _redux_max_u32(head)
                    sel[top_k * r + k] = Int32(65535) - (g & Int32(65535))
                    val[top_k * r + k] = _key_val(g)
                    if const_expr(k < top_k - 1):
                        # pop: the owner lane's sorted keys move up by one (keys are unique)
                        for j in cutlass.range_constexpr(top_k - 1):
                            keys[16 * r + j] = _sel_eq(
                                head, g, keys[16 * r + j + 1], keys[16 * r + j]
                            )
            for r in cutlass.range_constexpr(tpw):
                if const_expr(r == 0):
                    self._finish_token(
                        sel,
                        val,
                        0,
                        ids_ptr,
                        wts_ptr,
                        local_start,
                        mask_addr,
                        pick_addr,
                        warp,
                        lane,
                        Int32(1),
                    )
                else:
                    self._finish_token(
                        sel,
                        val,
                        top_k,
                        ids_ptr,
                        wts_ptr,
                        local_start,
                        mask_addr,
                        pick_addr,
                        tok1,
                        lane,
                        _sel_eq(tok1, warp, Int32(0), Int32(1)),
                    )

    @cute.jit
    def _finish_token(
        self,
        sel: cute.Tensor,
        val: cute.Tensor,
        base: cutlass.Constexpr,
        ids_ptr: Int64,
        wts_ptr: Int64,
        local_start: Int32,
        mask_addr: Int32,
        pick_addr: Int32,
        tok,
        lane,
        active,
    ) -> None:
        """Renormalization (route kernel arithmetic) of one token's picks sel / val [base, base + K), lane k stores
        pick k (id, BF16 weight) and records a local pick in the token masks; nothing when not ``active``."""
        top_k: cutlass.Constexpr = self.top_k
        num_local: cutlass.Constexpr = self.num_local
        vmax = val[base]
        ex = cute.make_rmem_tensor(cute.make_layout((top_k,)), Float32)
        for k in cutlass.range_constexpr(top_k):
            ex[k] = _exp_tr(_sub(val[base + k], vmax))
        ssum = _add(
            _add(_add(_add(ex[0], ex[8]), ex[4]), _add(ex[2], ex[6])),
            _add(_add(_add(ex[1], ex[9]), ex[5]), _add(ex[3], ex[7])),
        )
        my_sel = sel[base]
        my_e = ex[0]
        for k in cutlass.range_constexpr(top_k - 1):
            my_sel = _sel_eq(lane, Int32(k + 1), sel[base + k + 1], my_sel)
            my_e = _sel_eq_f(lane, Int32(k + 1), ex[k + 1], my_e)
        if active != Int32(0):
            if lane < top_k:
                o = Int64(tok) * Int64(top_k) + Int64(lane)
                _st_b32(ids_ptr + o * Int64(4), my_sel)
                _st_b16(wts_ptr + o * Int64(2), _div_bf16(my_e, ssum))
                lid = my_sel - local_start
                pick = _sel_lt(lid, Int32(num_local), lid, Int32(-1))
                if pick >= Int32(0):
                    _red_or_shared(mask_addr + pick * 4, _bit(tok))
                _st_shared(pick_addr + (tok * top_k + lane) * 4, pick)

    @cute.jit
    def _tables_warp(self, tab_ptr: Int64, mask_addr: Int32, offs_addr: Int32, lane) -> None:
        """Warp 0: lane l owns the local experts [l * EPL, (l + 1) * EPL): rows (popc of the token mask), CTAs,
        a lane-local then warp-wide exclusive scan of the CTA counts, the CTA -> expert / row-limit tables, the
        experts' first CTA in shared memory, and the CTA / padded-row totals."""
        num_local: cutlass.Constexpr = self.num_local
        tile: cutlass.Constexpr = self.tile
        epl: cutlass.Constexpr = num_local // 32
        cnt = cute.make_rmem_tensor(cute.make_layout((epl,)), Int32)
        nct = cute.make_rmem_tensor(cute.make_layout((epl,)), Int32)
        e0 = lane * epl
        total = Int32(0)
        for j in cutlass.range_constexpr(epl):
            cnt[j] = _popc(_ld_shared(mask_addr + (e0 + j) * 4))
            nct[j] = (cnt[j] + Int32(tile - 1)) // Int32(tile)
            total = total + nct[j]
        incl = total
        for d in cutlass.range_constexpr(5):
            incl = _scan_add(incl, 1 << d)
        offs = incl - total
        for j in cutlass.range_constexpr(epl):
            e = e0 + j
            _st_shared(offs_addr + e * 4, offs)
            for c in cutlass.range_constexpr(self.max_ctas_per_expert):
                if Int32(c) < nct[j]:
                    cta_i = offs + Int32(c)
                    _st_b32(tab_ptr + Int64(self.off_cte + cta_i) * Int64(4), e)
                    _st_b32(
                        tab_ptr + Int64(self.off_ctl + cta_i) * Int64(4),
                        cutlass.min((cta_i + 1) * Int32(tile), offs * Int32(tile) + cnt[j]),
                    )
            offs = offs + nct[j]
        if lane == 31:
            _st_b32(tab_ptr + Int64(self.off_nct) * Int64(4), incl)
            _st_b32(tab_ptr + Int64(self.off_npd) * Int64(4), incl * Int32(tile))

    @cute.jit
    def _quant_cta(
        self,
        gA16: cute.Tensor,
        gs_ptr: Int64,
        xq_ptr: Int64,
        xsf_ptr: Int64,
        tidx,
        qcta,
    ) -> None:
        """NVFP4 quantization of x, one 16-element block per thread (linear scale layout)."""
        bpr: cutlass.Constexpr = self.blocks_per_row
        gs = _ld_f32(gs_ptr)  # static (the experts' input scale): before the wait
        if const_expr(self.use_pdl):
            _gdc_wait()
        b = qcta * self.threads + tidx
        if b < self.num_rows * bpr:
            row = b // bpr
            cb = b % bpr
            src = Int64(gA16.iterator.toint()) + (
                Int64(row) * Int64(self.a_stride) + Int64(cb) * Int64(16)
            ) * Int64(2)
            w0, w1, w2, w3 = _ld_v4(src)
            w4, w5, w6, w7 = _ld_v4(src + Int64(16))
            lo, hi, sf = _nvfp4_block(w0, w1, w2, w3, w4, w5, w6, w7, gs)
            _st_v2(xq_ptr + Int64(row) * Int64(self.k_extent // 2) + Int64(cb) * Int64(8), lo, hi)
            _st_u8(xsf_ptr + Int64(row) * Int64(bpr) + Int64(cb), sf)
        cute.arch.sync_threads()
        if const_expr(self.use_pdl):
            _gdc_launch()
