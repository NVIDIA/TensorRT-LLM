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
"""Arithmetic of the softmax, the epilogues and the preparations in CuTe DSL operations: the exp2 landing on the
f32 subnormal grid with its anchor bias and its multiplier, f16x2 / E4M3 and BF16 packing, the scaling of a BF16
pair, the guarded reciprocal, the power-of-two ceiling of a scale, the rounding of an NVFP4 block scale, float bit
casts, the pairwise summation tree, the sum and the maximum over a warp, and the hashed start values of the
eigensolvers. The arithmetic that is inline PTX is in the PTX modules of common/ and of the families."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import (
    BFloat16,
    Float8E4M3FN,
    Float16,
    Float32,
    Int16,
    Int32,
    Uint16,
    Uint32,
)
from cutlass.cutlass_dsl import T, dsl_user_op

from ..constants import SUBNORMAL_ULPS
from .dsl import (
    FPRoundingMode,
    SaturationMode,
    convert,
    convert_and_pack_integer,
    cvt_f32x2_to_f8x2,
)


def subnormal_anchor_bias(m, a, prescale_log2: int):
    """-m * a + (15 + prescale_log2) * 2^-139 on the f32 subnormal grid: the anchor score m lands on 2^prescale_log2."""
    bias = Float32((15 + prescale_log2) * SUBNORMAL_ULPS * 2.0**-149)
    return cute.math.fma(
        cute.math.neg(Float32(m)), Float32(a), bias, rounding=cute.math.RoundingMode.NEAREST_EVEN
    )


def subnormal_scale(a):
    """a * 2^-139 (a * SUBNORMAL_ULPS * 2^-149), rounded to nearest onto the f32 subnormal grid: the multiplier of
    the exp2 landing on the f16 pattern of 2^t."""
    return cute.math.mul(
        Float32(a),
        Float32(SUBNORMAL_ULPS * 2.0**-149),
        rounding=cute.math.RoundingMode.NEAREST_EVEN,
    )


def ex2_sub_f16x2(x0, x1, a, b):
    """f32 x0, x1, a, b -> i32 word of two f16, 2^t of x0 in the low half and of x1 in the high half. The packed FMA
    (fma.rn.f32x2) forms x0 a + b and x1 a + b, and the saturating pack (cvt.pack.sat.u16.s32) packs their bit
    patterns, read as s32, into two u16. a = scale * 2^-139 and b = (bias + 15) * 2^-139 are f32 subnormals (a
    power-of-two scale, or one that keeps ~11 significant bits there), so x a + b lands on the subnormal grid with
    the bits round(1024 * (t + 15)), the f16 pattern of 2^t with a linear mantissa; the pack saturates unsigned
    (t < -15, -inf -> +0)."""
    z = cute.arch.fma_packed_f32x2((x0, x1), (a, a), (b, b))
    return convert_and_pack_integer(z[1].bitcast(Int32), z[0].bitcast(Int32), Uint16)


def f16x2x2_to_e4m3x4(a, b):
    """Two words of two f16 each -> one word of four E4M3 bytes (RNE, satfinite), those of a in the low half."""
    lo, hi = (
        convert(
            cutlass.Vector.from_elements((w,), Int32).bitcast(Float16),
            Float8E4M3FN,
            rnd=FPRoundingMode.RN,
            sat=SaturationMode.SATFINITE,
        )
        for w in (a, b)
    )
    return cutlass.Vector.from_elements((lo, hi), Int16).bitcast(Int32)[0]


def bf16x2(lo, hi):
    """Two FP32 -> one word of two BF16 (RNE), lo in the low half."""
    pair = cutlass.Vector.from_elements((hi, lo), Float32)
    return convert(pair, BFloat16, rnd=FPRoundingMode.RN).bitcast(Uint32)[0]


def e4m3x4(a, b, c, d):
    """Four FP32 -> one word of four E4M3 bytes (RNE, satfinite), a in the low byte."""
    lo, hi = (
        cvt_f32x2_to_f8x2(y, x, Float8E4M3FN, rnd=FPRoundingMode.RN, sat=SaturationMode.SATFINITE)
        for x, y in ((a, b), (c, d))
    )
    return cutlass.Vector.from_elements((lo, hi), Int16).bitcast(Uint32)[0]


def rcp_or_zero(x):
    """The approximate reciprocal of the f32 x (rcp.approx.f32) when x > 0, else 0."""
    return Float32(cutlass.select_(x > Float32(0.0), cute.math.rcp(x, approx=True), Float32(0.0)))


def tree_sum(vals):
    """Packed pairwise tree sum of a list of FP32 values (a power of two of them), halves added as f32x2 pairs."""
    acc = list(vals)
    while len(acc) > 2:
        half = len(acc) // 2
        nxt = []
        for j in range(0, half, 2):
            pa = cute.arch.add_packed_f32x2(
                (acc[j], acc[j + 1]), (acc[half + j], acc[half + j + 1])
            )
            nxt += [pa[0], pa[1]]
        acc = nxt
    return acc[0] + acc[1]


@dsl_user_op
def f2u(x, *, loc=None, ip=None):
    """The bit pattern of the f32 x as a Uint32."""
    return Uint32(llvm.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def u2f(x, *, loc=None, ip=None):
    """The f32 with the bit pattern of the Uint32 x."""
    return Float32(llvm.bitcast(T.f32(), Uint32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def pow2_ceil(x):
    """The power of two at or above max(x, 1e-30), for a finite f32 x >= 0: the exponent field of the maximum,
    raised by one when its mantissa is nonzero."""
    bits = f2u(cute.arch.fmax(x, Float32(1e-30)))
    exponent = bits & Uint32(0x7F800000)
    return u2f(
        Uint32(
            cutlass.select_(
                (bits & Uint32(0x007FFFFF)) != Uint32(0), exponent + Uint32(0x00800000), exponent
            )
        )
    )


def pow2_ceil_448(m):
    """The power of two at or above max(m / 448, 1e-30), for a finite f32 m >= 0: the scale that brings values of
    magnitude up to m into the E4M3 range."""
    return pow2_ceil(cute.math.div(m, Float32(448.0), rounding=cute.math.RoundingMode.NEAREST_EVEN))


def round_m3(x):
    """x (FP32, >= 0) rounded to 3 mantissa bits, nearest even, at any exponent."""
    b = f2u(x)
    b = b + Uint32(0x7FFFF) + ((b >> 20) & Uint32(1))
    return u2f(b & Uint32(0xFFF00000))


def scale_bf16x2(x, a):
    """i32 word x of two BF16, f32 a -> i32 word of the two halves times a, rounded back to BF16: each half in the
    high 16 bits of an f32, a multiply that keeps subnormal results (mul.rn.f32), then the packing of bf16x2."""
    lo = u2f(Uint32(x) << Uint32(16))
    hi = u2f(Uint32(x) & Uint32(0xFFFF0000))
    nearest = cute.math.RoundingMode.NEAREST_EVEN
    return Int32(
        bf16x2(cute.math.mul(lo, a, rounding=nearest), cute.math.mul(hi, a, rounding=nearest))
    )


def warp_sum(x):
    """The sum of the f32 x over the lanes of the warp, by five butterfly shuffles."""
    for o in (16, 8, 4, 2, 1):
        x = x + cute.arch.shuffle_sync_bfly(x, offset=Int32(o))
    return x


def warp_max(x):
    """The maximum of the f32 x over the lanes of the warp, by five butterfly shuffles."""
    for o in (16, 8, 4, 2, 1):
        x = cute.arch.fmax(x, cute.arch.shuffle_sync_bfly(x, offset=Int32(o)))
    return x


def hash01(i, key):
    """A value in [-0.5, 0.5) from the integers i and key by integer avalanche mixing: the deterministic dense start
    vectors of the eigensolvers."""
    h = Uint32(i + 1) ^ (Uint32(key) * Uint32(0x9E3779B9))
    h = (h ^ (h >> Uint32(16))) * Uint32(0x7FEB352D)
    h = (h ^ (h >> Uint32(15))) * Uint32(0x846CA68B)
    h = h ^ (h >> Uint32(16))
    return Float32(h >> Uint32(8)) * Float32(1.0 / 16777216.0) - Float32(0.5)
