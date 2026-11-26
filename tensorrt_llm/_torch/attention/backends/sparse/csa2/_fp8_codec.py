# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The E2M1 and E4M3 conversions native staging and PrimTS both decode NVFP4 MAIN rows with.

Native staging (``kernel.py``, compiled with the classic CuTe DSL frontend) and the PrimTS kernels
(``FrontendNext``) call these ``dsl_user_op`` inline-PTX ops, which compile under both, so a MAIN
record turns into the same E4M3 bytes on either provider at the same power-of-two scale
(``test_native_staging_matches_prims_ts_main_bytes``). The module imports cutlass; ``kernel.py``
imports it where it loads cutlass.
"""

from __future__ import annotations

import cutlass
from cutlass import Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

__all__ = ["e4m3_f16x2", "f16x2_pow2", "scaled_e2m1x4"]


def f16x2_pow2(exponent: int) -> int:
    """The F16x2 bits of ``2**exponent`` in both halves, for a normal F16 power of two."""
    if not -14 <= exponent <= 15:
        raise ValueError("The F16 power of two must be normal")
    bits = (exponent + 15) << 10
    return bits | (bits << 16)


def _u32_asm(sources: list, body: str, constraints: str, loc, ip) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip) for value in sources],
            body,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.dsl_user_op
def e4m3_f16x2(byte_pair, *, loc=None, ip=None) -> Uint32:
    """An E4M3 byte widened exactly to F16 in both halves of a word.

    ``byte_pair`` is a 32-bit DSL integer holding the byte in bits 0-7 and again in bits 8-15.
    """
    return _u32_asm(
        [byte_pair],
        "{\n.reg .b16 s;\ncvt.u16.u32 s, $1;\ncvt.rn.f16x2.e4m3x2 $0, s;\n}",
        "=r,r",
        loc,
        ip,
    )


@cutlass.dsl_user_op
def scaled_e2m1x4(source, group_scale, factor, *, loc=None, ip=None) -> Uint32:
    """The four E2M1 values in the low 16 bits of ``source`` times the F16 ``group_scale`` pair
    (``e4m3_f16x2`` of the record's E4M3 group scale), then times the F16 ``factor`` pair (a power
    of two, ``f16x2_pow2``), rounded RN/satfinite to four E4M3 bytes.

    The three arguments are 32-bit DSL integers. E2M1 times an E4M3 scale has at most five
    significant bits and lies in [2**-10, 2688], so the factor is exact unless the product
    overflows F16, which the rounding saturates.
    """
    return _u32_asm(
        [source, group_scale, factor],
        "{\n.reg .b8 a, b;\n.reg .b16 x, y;\n.reg .b32 h, l, t;\n"
        "cvt.u8.u32 a, $1;\nshr.b32 t, $1, 8;\ncvt.u8.u32 b, t;\n"
        "cvt.rn.f16x2.e2m1x2 l, a;\ncvt.rn.f16x2.e2m1x2 h, b;\n"
        "mul.rn.f16x2 l, l, $2;\nmul.rn.f16x2 h, h, $2;\n"
        "mul.rn.f16x2 l, l, $3;\nmul.rn.f16x2 h, h, $3;\n"
        "cvt.rn.satfinite.e4m3x2.f16x2 x, l;\ncvt.rn.satfinite.e4m3x2.f16x2 y, h;\n"
        "mov.b32 $0, {x, y};\n}",
        "=r,r,r,r",
        loc,
        ip,
    )
