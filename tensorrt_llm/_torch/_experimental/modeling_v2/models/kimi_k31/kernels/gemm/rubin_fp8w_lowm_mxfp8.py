# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 quantization PTX helpers (SM107) of the attention-residual kernel's MXFP8 side output.

``kernels/norm/rubin_attn_res.py`` takes from here the 1x32 group amax of BF16 words, its UE8M0
exponent and the E4M3 codes, the in-kernel quantization of the engine's low-M FP8 GEMM
(``fp8_block_scale_low_m_gemm``, whose helpers these are): per 1x32 group, ``e`` is
the biased exponent of ``2^ceil(log2(amax / 448))`` with ``amax`` floored at BF16 ``0x38D2``
(~1e-4), and the codes are ``satfinite_rn_e4m3(x * 2^(127 - e))``.
"""

from __future__ import annotations

from cutlass import Int32
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.kernels.blackwell.fp8_block_scale_low_m_gemm import (
    _asm,
    _bf16x2_inv_scale,
    _iv,
    _quant_bf16x8_to_e4m3,
    _ue8m0_of_amax,
)


@dsl_user_op
def _amax_bf16x8_words(w0, w1, w2, w3, *, loc=None, ip=None):
    """|x| max of eight BF16 values (four packed words) as a packed BF16 pair (garbage sign)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "{\n.reg .b32 m0, m1;\n"
            "max.xorsign.abs.bf16x2 m0, $1, $2;\n"
            "max.xorsign.abs.bf16x2 m1, $3, $4;\n"
            "max.xorsign.abs.bf16x2 $0, m0, m1;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _amax_pair(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


# Producer-side helpers (bitwise the low-M kernel's in-kernel quantization).
amax_bf16x8_words = _amax_bf16x8_words
amax_pair = _amax_pair
ue8m0_of_amax = _ue8m0_of_amax
bf16x2_inv_scale = _bf16x2_inv_scale
quant_bf16x8_to_e4m3 = _quant_bf16x8_to_e4m3

__all__ = [
    "amax_bf16x8_words",
    "amax_pair",
    "bf16x2_inv_scale",
    "quant_bf16x8_to_e4m3",
    "ue8m0_of_amax",
]
