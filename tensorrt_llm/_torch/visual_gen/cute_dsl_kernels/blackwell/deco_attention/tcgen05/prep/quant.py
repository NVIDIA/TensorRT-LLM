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
"""E2M1 block quantizers of the preparation passes: NVFP4 blocks of 16 values with a UE4M3 scale.

* A block takes the scale max |x| / c rounded to three mantissa bits (round_m3): c = E2M1_MAX, which puts the largest
  magnitude on the largest code, or c = WIDE_SCALE_MAX.
* A compensation block takes c = E2M1_MAX (quant16). A head block takes its scale by the scale rule of the call
  (constants.SCALE_RULES; quant16_dq_knownmax):
    - "max": c = E2M1_MAX.
    - "band": c = WIDE_SCALE_MAX where a value of the block lies in the band below the largest magnitude
      (_in_band), else c = E2M1_MAX.
    - "search": the block is quantized with both scales and keeps the one with the smaller squared error, the sum
      of (x - code * scale)^2 over the 16 values in FP32 (two partial sums, of the even and of the odd values); the
      scale of WIDE_SCALE_MAX only with a smaller error (_search).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32

from ...common.expmath import f2u, rcp_or_zero, round_m3
from ...common.ptx import f32x8_to_e2m1x8
from ...constants import SCALE_BAND_HIGH, SCALE_BAND_LOW, WIDE_SCALE_MAX
from ..ptx import e2m1x8_to_f32x8

_SIXTH = 0.16666667163372040
_WIDE = 1.0 / WIDE_SCALE_MAX


def _mul2_list(x, s):
    """Return [x_j * s], by packed FP32x2 multiplies."""
    out = []
    for j in range(0, len(x), 2):
        p = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (s, s))
        out += [p[0], p[1]]
    return out


def _absmax(x):
    """Return max |x_j|, as a chain of maxima with the absolute values."""
    am = cute.math.absf(x[0])
    for j in range(1, len(x)):
        am = cute.arch.fmax(am, cute.math.absf(x[j]))
    return am


def _nvfp4_codes(x, am):
    """Return the UE4M3 scale of a 16-value block whose max |x| is am (am / 6 rounded to three mantissa bits, round_m3)
    and the block's two words of E2M1 codes."""
    sc = round_m3(am * Float32(_SIXTH))
    inv = rcp_or_zero(sc)
    y = _mul2_list(x, inv)
    return sc, f32x8_to_e2m1x8(y[0:8]), f32x8_to_e2m1x8(y[8:16])


def _residual(x, sc, w0, w1):
    """Return the quantization residual x - dequantized x of a block with the scale sc and the two words of codes
    (one FMA per value)."""
    vals = e2m1x8_to_f32x8(w0) + e2m1x8_to_f32x8(w1)
    nsc = -sc
    dq = []
    for j in range(0, 16, 2):
        p = cute.arch.fma_packed_f32x2((vals[j], vals[j + 1]), (nsc, nsc), (x[j], x[j + 1]))
        dq += [p[0], p[1]]
    return dq


def _in_band(x, am):
    """Return whether a value of a block with max |x| am lies in the band of the scale rule "band":
    lo <= |x| < hi, lo = am * SCALE_BAND_LOW and hi = am * SCALE_BAND_HIGH.

    * The bounds are the rounded products of one packed multiply.
    * A difference |x| - lo is exact for a value near the band, and a negative one has the sign bit: as unsigned
      numbers, the bit patterns of the differences of the values inside the band are those below that of hi - lo.
    """
    bounds = cute.arch.mul_packed_f32x2(
        (am, am), (Float32(SCALE_BAND_LOW), Float32(SCALE_BAND_HIGH))
    )
    lo = bounds[0]
    d = [f2u(cute.math.absf(v) - lo) for v in x]
    while len(d) > 1:
        d = [Uint32(cutlass.select_(d[i] < d[i + 1], d[i], d[i + 1])) for i in range(0, len(d), 2)]
    return d[0] < f2u(bounds[1] - lo)


def _candidate(x, am, factor):
    """Return (the scale, the two words of codes, the residual, the squared error) of a block of 16 values with
    max |x| am at the scale am * factor."""
    sc = round_m3(am * factor)
    inv = rcp_or_zero(sc)
    y = _mul2_list(x, inv)
    w0, w1 = f32x8_to_e2m1x8(y[0:8]), f32x8_to_e2m1x8(y[8:16])
    dq = _residual(x, sc, w0, w1)
    e0 = Float32(0.0)
    e1 = Float32(0.0)
    for j in range(0, 16, 2):
        e = cute.arch.fma_packed_f32x2((dq[j], dq[j + 1]), (dq[j], dq[j + 1]), (e0, e1))
        e0, e1 = e[0], e[1]
    return sc, w0, w1, dq, e0 + e1


def _search(x, am):
    """Return (the scale, the two words of codes, the residual) of a block of 16 values with max |x| am at the scale
    with the smaller squared error of am / E2M1_MAX and am / WIDE_SCALE_MAX, the first when the errors are equal."""
    sc, w0, w1, dq, err = _candidate(x, am, Float32(_SIXTH))
    sc_w, v0, v1, dq_w, err_w = _candidate(x, am, Float32(_WIDE))
    wide = err_w < err
    sc = Float32(cutlass.select_(wide, sc_w, sc))
    w0 = Uint32(cutlass.select_(wide, v0, w0))
    w1 = Uint32(cutlass.select_(wide, v1, w1))
    return sc, w0, w1, [Float32(cutlass.select_(wide, a, b)) for a, b in zip(dq_w, dq)]


def quant16_dq_knownmax(x, am, rule="max"):
    """Quantize an NVFP4 head block of 16 values whose max |x| is am under a scale rule (constants.SCALE_RULES):
    return the UE4M3 scale, two words of E2M1 codes and the quantization residual x - dequantized x at that scale."""
    if rule == "search":
        return _search(x, am)
    if rule == "max":
        sc, w0, w1 = _nvfp4_codes(x, am)
    else:
        factor = Float32(cutlass.select_(_in_band(x, am), Float32(_WIDE), Float32(_SIXTH)))
        sc = round_m3(am * factor)
        inv = rcp_or_zero(sc)
        y = _mul2_list(x, inv)
        w0, w1 = f32x8_to_e2m1x8(y[0:8]), f32x8_to_e2m1x8(y[8:16])
    return sc, w0, w1, _residual(x, sc, w0, w1)


def quant16(x, am=None):
    """Quantize an NVFP4 block of 16 values (am: their max |x| where it is known): return the UE4M3 scale and two
    words of E2M1 codes."""
    return _nvfp4_codes(x, _absmax(x) if am is None else am)
