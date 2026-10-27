# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode producer of one packed head, for the QKV GEMV's producer epilogue (``QKV_PRODUCER_FOLD``): q / k /
index-q / index-k RMSNorm + NeoX partial RoPE, NVFP4 K / V and FP8 index-K cache insert, FP8 q / index-q out.

Bit for bit ``torch.ops.trtllm.minimax_m3_sm107_tp2_producer_pw`` (cpp/tensorrt_llm/kernels/minimaxM3Sm107Tp2/
producerPw2.cu, m3_producer_kernel) at the TP2 geometry: 32 query heads, 2 KV heads, 2 index-q heads and one index-K
head, head_dim 128, rotary_dim 64, page 128. The sm_107 TP8 target's m3_layer.phase_producer at these heads: lane l of
the calling warp holds elements [4l, 4l + 4) of the head exactly as the C++ kernel's warp, so every reduction (the RMS
sum of squares, the RoPE pair exchange, the NVFP4 block max) runs the same shuffle tree in the same order. Every
floating-point operation is written as its PTX instruction (mul.rn / add.rn / fma.rn / rsqrt.approx / rcp.approx.ftz /
cvt) so the DSL compiler cannot contract or reassociate differently from the library build (--use_fast_math). The
places where that build contracted are marked CONTRACT and follow the shipped SASS (msa/results/sassdump-*).

Packed row layout per token: [32 q | 2 k | 2 v | 2 index_q | index_k] x 128 BF16 (the QKV GEMV output rounded to
BF16).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op

HEADS_Q = 32  # capacity: fixed: the tp2 query heads per rank
HEADS_KV = 2  # capacity: fixed: the tp2 KV heads per rank
HEADS_INDEX = 2  # capacity: fixed: the tp2 index-q heads per rank
PACKED_HEADS = (
    HEADS_Q + 2 * HEADS_KV + HEADS_INDEX + 1
)  # q x 32, k x 2, v x 2, index_q x 2, index_k
HEAD_DIM = 128
ROTARY_DIM = 64  # capacity: fixed: the checkpoint's rotary dim
PAGE = 128
ELEMS = HEAD_DIM // 32  # capacity: fixed: elements per lane (per lane)
# The first packed head of each kind (producerPw2.cu kBegin / vBegin / indexQBegin / indexKHead).
K_BEGIN = HEADS_Q
V_BEGIN = K_BEGIN + HEADS_KV
IQ_BEGIN = V_BEGIN + HEADS_KV
IK_HEAD = IQ_BEGIN + HEADS_INDEX
# Head kinds (``producer_head``'s ``kind``).
KIND_Q, KIND_K, KIND_V, KIND_IQ, KIND_IK = 0, 1, 2, 3, 4


def _asm(ret, args, ptx, cons, side=False, loc=None, ip=None):
    return _llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        ptx,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---- exact float arithmetic (no contraction) ---------------------------------------------------------------------
@dsl_user_op
def f_mul(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f_add(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f_fma(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f_max(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f_abs(a, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Float32(a)], "abs.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def f_neg(a, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Float32(a)], "neg.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def f_rsqrt(a, *, loc=None, ip=None):
    # NOT ftz: both producer builds this one is held to (the library producerPw and the target's own) run
    # rsqrt.approx.f32 with denormal scaling; .ftz differed on a denormal RMS argument (eps 0).
    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Float32(a)], "rsqrt.approx.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def f_rcp(a, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Float32(a)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def bf16_bits_to_f32(bits16, *, loc=None, ip=None):
    """The float value of a BF16 given in the low 16 bits of an Int32."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Int32(bits16)],
            "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def round_bf16(a, *, loc=None, ip=None):
    """__bfloat162float(__float2bfloat16_rn(a))."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 t;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b32 t, {0, h};\n\tmov.b32 $0, t;\n\t}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3x4_word(e0, e1, e2, e3, *, loc=None, ip=None):
    """store_fp8x4: bytes [e0, e1, e2, e3] (cvt.rn.satfinite.e4m3x2.f32 puts its first source in the high byte)."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(e0), cutlass.Float32(e1), cutlass.Float32(e2), cutlass.Float32(e3)],
            "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
            "=r,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3_byte_and_value(x, *, loc=None, ip=None):
    """(byte, float value) of __nv_fp8_e4m3(x): the encoded scale and static_cast<float> of it."""
    byte = cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n\t.reg .b16 p;\n\t.reg .b32 w;\n\tcvt.rn.satfinite.e4m3x2.f32 p, 0f00000000, $1;\n\t"
            "cvt.u32.u16 w, p;\n\tand.b32 $0, w, 0xFF;\n\t}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )
    value = cutlass.Float32(
        _asm(
            _T.f32(),
            [byte],
            "{\n\t.reg .b16 p;\n\t.reg .b32 h2;\n\t.reg .b16 h;\n\tcvt.u16.u32 p, $1;\n\t"
            "cvt.rn.f16x2.e4m3x2 h2, p;\n\tmov.b32 {h, _}, h2;\n\tcvt.f32.f16 $0, h;\n\t}",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )
    return byte, value


@dsl_user_op
def e2m1x4_halfword(q0, q1, q2, q3, *, loc=None, ip=None):
    """The low 16 bits of fp32x8_to_e2m1 for q[0..3] (q[4..7] = 0): byte0 = (q1, q0), byte1 = (q3, q2)."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(q0), cutlass.Float32(q1), cutlass.Float32(q2), cutlass.Float32(q3)],
            "{\n\t.reg .b8 b0, b1;\n\t.reg .b32 w;\n\tcvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n\tmov.b32 w, {b0, b1, 0, 0};\n\tmov.b32 $0, w;\n\t}",
            "=r,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


# ---- global memory -----------------------------------------------------------------------------------------------
@dsl_user_op
def ld_u64(addr, *, loc=None, ip=None):
    return cutlass.Int64(
        _asm(_T.i64(), [cutlass.Int64(addr)], "ld.global.u64 $0, [$1];", "=l,l", True, loc, ip)
    )


@dsl_user_op
def ld_s32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(_T.i32(), [cutlass.Int64(addr)], "ld.global.s32 $0, [$1];", "=r,l", True, loc, ip)
    )


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Int64(addr)], "ld.global.f32 $0, [$1];", "=f,l", True, loc, ip)
    )


@dsl_user_op
def st_u32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(v)],
        "st.global.u32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_u16(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(v)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u16 [$0], h;\n\t}",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_u8(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(v)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u8 [$0], h;\n\t}",
        "l,r",
        True,
        loc,
        ip,
    )


# ---- CONTRACT points (follow the shipped SASS; see the module docstring) -------------------------------------------
def sumsq_pair(acc, x, y):
    """sumSquares += x * x + y * y  (C++: one pair of the packed row). CONTRACT: fma(y, y, x * x) then add."""
    return f_add(acc, f_fma(y, y, f_mul(x, x)))


def rms_reciprocal(sum_squares, eps):
    """rsqrtf(sumSquares / 128 + eps). CONTRACT: the / 128 is an exact * 2^-7; fused with + eps into one fma."""
    return f_rsqrt(f_fma(sum_squares, cutlass.Float32(1.0 / HEAD_DIM), eps))


class ProducerArgs:
    """Raw device addresses (int64) + strides, as the C++ Params (producerPw2.h) carries them."""

    def __init__(
        self,
        packed,
        q_out,
        index_q_out,
        index_k,
        kv_data,
        kv_scale,
        out_cache_loc,
        kv_quant_scale,
        position_ids,
        rotary_cos_sin,
        q_weight,
        k_weight,
        index_q_weight,
        index_k_weight,
        eps,
        num_pages,
        index_page_stride,
        index_token_stride,
        data_page_stride,
        data_plane_stride,
        data_head_stride,
        data_token_stride,
        scale_page_stride,
        scale_plane_stride,
        scale_head_stride,
    ):
        self.__dict__.update(locals())
        del self.__dict__["self"]


@cute.jit
def producer_head(tok, head, lane, x0, x1, x2, x3, a: ProducerArgs):
    """Everything after the packed-row load for packed head ``head`` (warp-uniform; dynamic Int32 or Constexpr) of
    decode token ``tok``, on the calling warp: lane ``lane`` holds the head's elements [4 lane, 4 lane + 4) as the
    float values x0..x3 of the BF16 packed row (the QKV GEMV output rounded to BF16). Norm weights, KV quant scale,
    position and RoPE loads, the RMS tree, RoPE, and the q / index-q / index-K / K-V NVFP4 stores, at the TP2 heads
    (K and V rows into their head's block of the HND pools, index-q into its head of ``index_q_out``).

    Data flow is branch-free (every load is in bounds for every head: V reads the q norm weight, a non-rotating lane
    reads its token's RoPE row, and the results are then discarded with select); only the stores branch, on
    warp-uniform conditions, so every shuffle runs on the full warp as in the C++ kernel."""
    l4 = cutlass.Int64(lane * ELEMS)
    # The head's kind: the packed row's heads run q, k, v, index-q, index-k in order.
    kind = cutlass.select_(
        head >= IK_HEAD,
        cutlass.Int32(KIND_IK),
        cutlass.select_(
            head >= IQ_BEGIN,
            cutlass.Int32(KIND_IQ),
            cutlass.select_(
                head >= V_BEGIN,
                cutlass.Int32(KIND_V),
                cutlass.select_(head >= K_BEGIN, cutlass.Int32(KIND_K), cutlass.Int32(KIND_Q)),
            ),
        ),
    )
    is_v = kind == KIND_V
    zero = cutlass.Float32(0.0)

    # Norm weights (BF16): q / k / index-q / index-k weight by head kind (V: q's, unused).
    w_base = cutlass.select_(
        kind == KIND_K,
        a.k_weight,
        cutlass.select_(
            kind == KIND_IQ,
            a.index_q_weight,
            cutlass.select_(kind == KIND_IK, a.index_k_weight, a.q_weight),
        ),
    )
    ww = ld_u64(w_base + l4 * 2)
    w0 = bf16_bits_to_f32(cutlass.Int32(ww & cutlass.Int64(0xFFFF)))
    w1 = bf16_bits_to_f32(cutlass.Int32((ww >> cutlass.Int64(16)) & cutlass.Int64(0xFFFF)))
    w2 = bf16_bits_to_f32(cutlass.Int32((ww >> cutlass.Int64(32)) & cutlass.Int64(0xFFFF)))
    w3 = bf16_bits_to_f32(cutlass.Int32((ww >> cutlass.Int64(48)) & cutlass.Int64(0xFFFF)))
    quant = ld_f32(a.kv_quant_scale + cutlass.select_(is_v, cutlass.Int64(8), cutlass.Int64(4)))

    # Position id, RoPE row.
    pos = ld_s32(a.position_ids + cutlass.Int64(tok) * 4)
    row = cutlass.Int64(pos) * ROTARY_DIM
    coef = cutlass.Int64((lane * ELEMS) % (ROTARY_DIM // 2))
    c0 = ld_f32(a.rotary_cos_sin + (row + coef) * 4)
    c1 = ld_f32(a.rotary_cos_sin + (row + coef + 1) * 4)
    c2 = ld_f32(a.rotary_cos_sin + (row + coef + 2) * 4)
    c3 = ld_f32(a.rotary_cos_sin + (row + coef + 3) * 4)
    s0 = ld_f32(a.rotary_cos_sin + (row + ROTARY_DIM // 2 + coef) * 4)
    s1 = ld_f32(a.rotary_cos_sin + (row + ROTARY_DIM // 2 + coef + 1) * 4)
    s2 = ld_f32(a.rotary_cos_sin + (row + ROTARY_DIM // 2 + coef + 2) * 4)
    s3 = ld_f32(a.rotary_cos_sin + (row + ROTARY_DIM // 2 + coef + 3) * 4)

    # RMSNorm (all heads compute it; V keeps x).
    ss = sumsq_pair(zero, x0, x1)
    ss = sumsq_pair(ss, x2, x3)
    ss = f_add(ss, cute.arch.shuffle_sync_bfly(ss, offset=16))
    ss = f_add(ss, cute.arch.shuffle_sync_bfly(ss, offset=8))
    ss = f_add(ss, cute.arch.shuffle_sync_bfly(ss, offset=4))
    ss = f_add(ss, cute.arch.shuffle_sync_bfly(ss, offset=2))
    ss = f_add(ss, cute.arch.shuffle_sync_bfly(ss, offset=1))
    r = rms_reciprocal(ss, a.eps)
    one = cutlass.Float32(1.0)
    n0 = f_mul(x0, f_mul(r, f_add(one, w0)))
    n1 = f_mul(x1, f_mul(r, f_add(one, w1)))
    n2 = f_mul(x2, f_mul(r, f_add(one, w2)))
    n3 = f_mul(x3, f_mul(r, f_add(one, w3)))
    # NeoX partial RoPE: lane l's dim 4l + i pairs with lane l ^ 8 (dim +-32) within the first 64 dims.
    pair_off = (ROTARY_DIM // 2) // ELEMS
    p0 = cute.arch.shuffle_sync_bfly(n0, offset=pair_off)
    p1 = cute.arch.shuffle_sync_bfly(n1, offset=pair_off)
    p2 = cute.arch.shuffle_sync_bfly(n2, offset=pair_off)
    p3 = cute.arch.shuffle_sync_bfly(n3, offset=pair_off)
    rot = lane * ELEMS < ROTARY_DIM
    low = lane * ELEMS < ROTARY_DIM // 2
    p0 = cutlass.select_(low, f_neg(p0), p0)
    p1 = cutlass.select_(low, f_neg(p1), p1)
    p2 = cutlass.select_(low, f_neg(p2), p2)
    p3 = cutlass.select_(low, f_neg(p3), p3)
    r0 = cutlass.select_(rot, f_fma(p0, s0, f_mul(n0, c0)), n0)
    r1 = cutlass.select_(rot, f_fma(p1, s1, f_mul(n1, c1)), n1)
    r2 = cutlass.select_(rot, f_fma(p2, s2, f_mul(n2, c2)), n2)
    r3 = cutlass.select_(rot, f_fma(p3, s3, f_mul(n3, c3)), n3)
    e0 = cutlass.select_(is_v, x0, r0)
    e1 = cutlass.select_(is_v, x1, r1)
    e2 = cutlass.select_(is_v, x2, r2)
    e3 = cutlass.select_(is_v, x3, r3)

    if kind == KIND_Q:
        st_u32(
            a.q_out + (cutlass.Int64(tok) * HEADS_Q + cutlass.Int64(head)) * HEAD_DIM + l4,
            e4m3x4_word(e0, e1, e2, e3),
        )
    if kind == KIND_IQ:
        st_u32(
            a.index_q_out
            + (cutlass.Int64(tok) * HEADS_INDEX + cutlass.Int64(head - IQ_BEGIN)) * HEAD_DIM
            + l4,
            e4m3x4_word(e0, e1, e2, e3),
        )
    if kind != KIND_Q:
        if kind != KIND_IQ:
            slot = ld_s32(a.out_cache_loc + cutlass.Int64(tok) * 4)
            page = slot // PAGE
            if slot >= 0:
                if page < a.num_pages:
                    within = slot & (PAGE - 1)
                    if kind == KIND_IK:
                        st_u32(
                            a.index_k
                            + cutlass.Int64(page) * a.index_page_stride
                            + cutlass.Int64(within) * a.index_token_stride
                            + l4,
                            e4m3x4_word(e0, e1, e2, e3),
                        )
                    else:
                        plane = cutlass.select_(is_v, cutlass.Int64(1), cutlass.Int64(0))
                        kv_head = cutlass.Int64(
                            cutlass.select_(is_v, head - V_BEGIN, head - K_BEGIN)
                        )
                        col = lane // 4
                        sw = cutlass.select_(
                            is_v,
                            cutlass.Int64((within // 4) * 32 + col * 4 + within % 4),
                            cutlass.Int64(within * 8 + col),
                        )
                        s_off = (
                            cutlass.Int64(page) * a.scale_page_stride
                            + plane * a.scale_plane_stride
                            + kv_head * a.scale_head_stride
                            + sw
                        )
                        # quantize_fp4x4: BF16-rounded values, 16-element block max over the 4-lane group.
                        q0 = round_bf16(e0)
                        q1 = round_bf16(e1)
                        q2 = round_bf16(e2)
                        q3 = round_bf16(e3)
                        lmax = f_max(
                            f_max(f_max(f_max(zero, f_abs(q0)), f_abs(q1)), f_abs(q2)), f_abs(q3)
                        )
                        lmax = f_max(lmax, cute.arch.shuffle_sync_bfly(lmax, offset=1))
                        lmax = f_max(lmax, cute.arch.shuffle_sync_bfly(lmax, offset=2))
                        sbyte, sval = e4m3_byte_and_value(
                            f_mul(quant, f_mul(lmax, f_rcp(cutlass.Float32(6.0))))
                        )
                        if lane % 4 == 0:
                            st_u8(a.kv_scale + s_off, sbyte)
                        outs = cutlass.select_(lmax != zero, f_rcp(f_mul(sval, f_rcp(quant))), zero)
                        hw = e2m1x4_halfword(
                            f_mul(q0, outs), f_mul(q1, outs), f_mul(q2, outs), f_mul(q3, outs)
                        )
                        d_off = (
                            cutlass.Int64(page) * a.data_page_stride
                            + plane * a.data_plane_stride
                            + kv_head * a.data_head_stride
                            + cutlass.Int64(within) * a.data_token_stride
                            + cutlass.Int64(lane * 2)
                        )
                        st_u16(a.kv_data + d_off, hw)
