# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""PTX helpers for the combine's MXFP8 epilogue: the exact instructions of the CUDA quantizer
(quantization.cuh cvt_warp_fp16_to_mxfp8), so the epilogue's E4M3 values and UE8M0 scales are bitwise those of
mxfp8_quantize / quantize_mxfp8_rows on the same bf16 values."""

import cutlass
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


def _asm(ret, args, text, cons):
    return llvm.inline_asm(
        ret, args, text, cons, has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
    )


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """reciprocal_approximate_ftz."""
    return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f"))


@dsl_user_op
def ue8m0_round_up(a: Float32, *, loc=None, ip=None) -> Int32:
    """__nv_cvt_float_to_e8m0(a, __NV_SATFINITE, cudaRoundPosInf): the low byte of the packed pair."""
    return Int32(
        _asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.rp.satfinite.ue8m0x2.f32 h, $1, $1;\ncvt.u32.u16 $0, h;\n}\n",
            "=r,f",
        )
    ) & Int32(0xFF)


@dsl_user_op
def as_f32(a: Int32, *, loc=None, ip=None) -> Float32:
    """Reinterpret 32 bits as fp32."""
    return Float32(_asm(T.f32(), [Int32(a).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r"))


@dsl_user_op
def bf16x2_lo_f32(w: Int32, *, loc=None, ip=None) -> Float32:
    """The low bf16 of a packed pair as fp32 (exact)."""
    return Float32(_asm(T.f32(), [Int32(w).ir_value(loc=loc, ip=ip)], "shl.b32 $0, $1, 16;", "=f,r"))


@dsl_user_op
def bf16x2_hi_f32(w: Int32, *, loc=None, ip=None) -> Float32:
    """The high bf16 of a packed pair as fp32 (exact)."""
    return Float32(_asm(T.f32(), [Int32(w).ir_value(loc=loc, ip=ip)], "and.b32 $0, $1, 0xFFFF0000;", "=f,r"))


@dsl_user_op
def fabs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "abs.f32 $0, $1;", "=f,f"))


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
        )
    )


@dsl_user_op
def ue8m0_to_f32(sf_bits: Int32, *, loc=None, ip=None) -> Float32:
    """static_cast<float>(__nv_fp8_e8m0): 0 -> 2^-127 (0x00400000), 0xFF -> NaN (0x7FFFFFFF), else bits << 23."""
    return Float32(
        _asm(
            T.f32(),
            [Int32(sf_bits).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 s;\n.reg .pred z, n;\n"
            "shl.b32 s, $1, 23;\n"
            "setp.eq.u32 z, $1, 0;\n"
            "setp.eq.u32 n, $1, 255;\n"
            "selp.b32 s, 0x00400000, s, z;\n"
            "selp.b32 s, 0x7FFFFFFF, s, n;\n"
            "mov.b32 $0, s;\n}\n",
            "=f,r",
        )
    )


def sf_offset_128x4(m_idx, k_idx, num_k_tiles):
    """get_sf_out_offset_128x4 (batch 0): [m tiles, k tiles, 32, 4, 4]."""
    return (
        (m_idx // 128) * (num_k_tiles * 512)
        + (k_idx // 4) * 512
        + (m_idx % 32) * 16
        + ((m_idx % 128) // 32) * 4
        + (k_idx % 4)
    )
