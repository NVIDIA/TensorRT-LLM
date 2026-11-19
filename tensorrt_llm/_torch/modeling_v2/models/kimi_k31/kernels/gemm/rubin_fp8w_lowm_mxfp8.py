# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 quantization PTX helpers (SM107) of the attention-residual kernel's MXFP8 side output.

``kernels/norm/rubin_attn_res.py`` takes from here the 1x32 group amax of BF16 words (packed BF16
magnitudes, NaN skipped by ``max.xorsign.abs``).
"""

from __future__ import annotations

from cutlass import Int32
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.kernels.blackwell.fp8_block_scale_low_m_gemm import _asm, _iv


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


amax_bf16x8_words = _amax_bf16x8_words
amax_pair = _amax_pair

__all__ = [
    "amax_bf16x8_words",
    "amax_pair",
]
