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
"""Bit-exact scalar reference for the MegaMoE-CuteDSL gen-phase output chain.

Pure Python (no torch / numpy) so it can run anywhere. Every function models
one IEEE-754 binary32 or bfloat16 operation with round-to-nearest-even and no
flush-to-zero, matching ``mul.rn.f32x2`` / ``fma.rn.f32x2`` / ``add.rn.f32`` /
``cvt.rn.bf16.f32``.

The modelled chain is the stock MegaMoE TopkReduce followed by the M3 layer
tail::

    acc   = t[0] * s[0]                      (fp32 mul)
    acc   = fma(t[k], s[k], acc), k = 1..K-1 (fp32 fma)
    routed = bf16(acc)
    moe   = bf16(f32(shared) + f32(routed))  (torch bf16 add)
    h     = f32(moe) + f32(residual)         (fp32 add)
    residual_out = bf16(h)
    out   = bf16((h * rstd) * weight)        (fp32 muls)

``rstd`` is an input: it depends on the FlashInfer reduction order and on
``rsqrt.approx``, neither of which is reproducible on the host.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from typing import Sequence

F32_MANTISSA_BITS = 23
F32_MIN_NORMAL_EXP = -126
F32_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
BF16_NAN_BITS = 0x7FC0


def f32_bits(x: float) -> int:
    """Return the binary32 bit pattern of an fp32-representable value."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _round_exact_to_f32(q: Fraction) -> float:
    """Round a nonzero exact rational to binary32 (RN-even, gradual underflow)."""
    negative = q < 0
    q = -q if negative else q
    num, den = q.numerator, q.denominator
    exp = num.bit_length() - den.bit_length()
    if (num << max(0, -exp)) < (den << max(0, exp)):
        exp -= 1
    quantum_exp = max(exp, F32_MIN_NORMAL_EXP) - F32_MANTISSA_BITS
    if quantum_exp < 0:
        scaled_num, scaled_den = num << -quantum_exp, den
    else:
        scaled_num, scaled_den = num, den << quantum_exp
    mantissa, remainder = divmod(scaled_num, scaled_den)
    twice = 2 * remainder
    if twice > scaled_den or (twice == scaled_den and mantissa & 1):
        mantissa += 1
    value = math.ldexp(float(mantissa), quantum_exp)
    if value > F32_MAX:
        value = math.inf
    return -value if negative else value


def f32(x: float) -> float:
    """Round an fp64 value to binary32 (RN-even); overflow goes to +-inf."""
    if not math.isfinite(x) or x == 0.0:
        return x
    return _round_exact_to_f32(Fraction(x))


def add_f32(a: float, b: float) -> float:
    """binary32 a + b.

    Rounding the fp64 sum is innocuous: 53 >= 2 * 24 + 2.
    """
    return f32(a + b)


def mul_f32(a: float, b: float) -> float:
    """binary32 a * b; the fp64 product of two binary32 values is exact."""
    return f32(a * b)


def fma_f32(a: float, b: float, c: float) -> float:
    """binary32 fma(a, b, c) with a single rounding."""
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return f32(a * b + c)
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        product_negative = math.copysign(1.0, a) * math.copysign(1.0, b) < 0
        product_is_zero = a == 0.0 or b == 0.0
        if product_is_zero and c == 0.0 and product_negative and math.copysign(1.0, c) < 0:
            return -0.0
        return 0.0
    return _round_exact_to_f32(exact)


def bf16_bits(x: float) -> int:
    """cvt.rn.bf16.f32 of a binary32 value, returned as a 16-bit pattern."""
    if math.isnan(x):
        return BF16_NAN_BITS
    bits = f32_bits(x)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16_value(bits: int) -> float:
    return f32_from_bits((bits & 0xFFFF) << 16)


def bf16(x: float) -> float:
    """Round a binary32 value to the nearest bfloat16 value."""
    return bf16_value(bf16_bits(x))


def topk_reduce_element(terms: Sequence[float], scores: Sequence[float] | None) -> float:
    """Stock TopkReduce for one hidden element.

    ``terms`` are bf16 values (one per top-k slot, slot order); ``scores`` are
    binary32 routing weights, or None for the unscaled plane sum.
    """
    if not terms:
        raise ValueError("TopkReduce needs at least one top-k slot")
    if scores is not None and len(scores) != len(terms):
        raise ValueError("terms and scores must have the same top-k length")
    acc = terms[0] if scores is None else mul_f32(terms[0], scores[0])
    for k in range(1, len(terms)):
        acc = fma_f32(terms[k], 1.0 if scores is None else scores[k], acc)
    return bf16(acc)


def topk_reduce_row(
    terms: Sequence[Sequence[float]], scores: Sequence[float] | None
) -> list[float]:
    """Stock TopkReduce for one token: ``terms[k][i]`` is slot k, hidden i."""
    hidden = len(terms[0])
    if any(len(row) != hidden for row in terms):
        raise ValueError("every top-k slot must have the same hidden size")
    return [topk_reduce_element([row[i] for row in terms], scores) for i in range(hidden)]


def layer_tail_element(
    routed: float, shared: float, residual: float, rstd: float, weight: float
) -> tuple[float, float, float]:
    """M3 MoE tail for one element.

    Returns ``(moe_output, residual_out, normed_out)``, each a bf16 value.
    """
    moe = bf16(add_f32(shared, routed))
    h = add_f32(moe, residual)
    return moe, bf16(h), bf16(mul_f32(mul_f32(h, rstd), weight))
