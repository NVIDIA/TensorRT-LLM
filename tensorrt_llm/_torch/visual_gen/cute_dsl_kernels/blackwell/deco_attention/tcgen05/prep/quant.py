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
"""E2M1 block quantizers of the preparation passes. An NVFP4 block holds 16 values with a UE4M3 scale, max |x| / 6
rounded to three mantissa bits (round_m3)."""

import cutlass.cute as cute
from cutlass import Float32

from ...common.expmath import rcp_or_zero, round_m3
from ...common.ptx import f32x8_to_e2m1x8
from ..ptx import e2m1x8_to_f32x8


def _mul2_list(x, s):
    """[x_j * s] with packed FP32x2 multiplies."""
    out = []
    for j in range(0, len(x), 2):
        p = cute.arch.mul_packed_f32x2((x[j], x[j + 1]), (s, s))
        out += [p[0], p[1]]
    return out


def _absmax(x):
    """max |x_j| as a chain of maxima with the absolute values."""
    am = cute.math.abs(x[0])
    for j in range(1, len(x)):
        am = cute.arch.fmax(am, cute.math.absf(x[j]))
    return am


def _nvfp4_codes(x, am):
    """The UE4M3 scale of a 16-value block whose max |x| is am, and the block's two words of E2M1 codes."""
    sc = round_m3(am * Float32(0.16666667163372040))
    inv = rcp_or_zero(sc)
    y = _mul2_list(x, inv)
    return sc, f32x8_to_e2m1x8(y[0:8]), f32x8_to_e2m1x8(y[8:16])


def quant16_dq_knownmax(x, am):
    """NVFP4 block of 16 values whose max |x| is am: the UE4M3 scale, two words of E2M1 codes and the quantization
    residual x - dequantized x (one FMA per value)."""
    sc, w0, w1 = _nvfp4_codes(x, am)
    vals = e2m1x8_to_f32x8(w0) + e2m1x8_to_f32x8(w1)
    nsc = -sc
    dq = []
    for j in range(0, 16, 2):
        p = cute.arch.fma_packed_f32x2((vals[j], vals[j + 1]), (nsc, nsc), (x[j], x[j + 1]))
        dq += [p[0], p[1]]
    return sc, w0, w1, dq


def quant16(x):
    """NVFP4 block of 16 values: the UE4M3 scale and two words of E2M1 codes."""
    return _nvfp4_codes(x, _absmax(x))
