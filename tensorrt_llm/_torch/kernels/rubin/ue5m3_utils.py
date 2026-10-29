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
"""CuTe DSL helpers for UE5M3 block scales (``cutlass.FloatNV8E5M3FNU``).

UE5M3 has 5 exponent bits (bias 15), 3 mantissa bits and subnormals: codes
0x00-0xFE are finite (0 to 114688) and 0xFF is NaN. Block-scaled MMA on SM107
reads every finite code as such.

Encoding uses the hardware conversion (``cvt.rp`` to ``ue5m3x2``), which
rounds up and does not saturate: inputs above 114688 become NaN, so they are
clamped first. Decoding uses integer arithmetic, because the DSL's
UE5M3 -> f32 conversion maps the codes with exponent 31 (65536 to 114688) to
infinity.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm.quantization.utils.fp4_utils import UE5M3_MAX


@dsl_user_op
def _bitcast_u32_f32(bits, *, loc=None, ip=None):
    return cutlass.Float32(llvm.bitcast(T.f32(), cutlass.Uint32(bits).ir_value(loc=loc, ip=ip)))


@cute.jit
def ue5m3_to_f32(code: cutlass.Uint32) -> cutlass.Float32:
    """Decode a finite UE5M3 code (0x00-0xFE)."""
    exponent = code >> 3
    mantissa = code & 7
    value = cutlass.Float32(mantissa) * cutlass.Float32(2.0**-17)
    if exponent != 0:
        value = _bitcast_u32_f32(((exponent + 112) << 23) | (mantissa << 20))
    return value


@cute.jit
def f32_to_ue5m3(scales: cute.Tensor, codes: cute.Tensor):
    """Round the non-negative FP32 ``scales`` up into the UE5M3 ``codes``.

    Both are register tensors of the same size; values above the largest
    finite UE5M3 value saturate to it.
    """
    count = cute.size(scales)
    code_bytes = cute.recast_tensor(codes, cutlass.Uint8)
    for start in cutlass.range_constexpr(0, count, 4):
        chunk = min(4, count - start)
        clamped = cute.make_rmem_tensor((4,), cutlass.Float32)
        clamped.fill(0.0)
        for i in cutlass.range_constexpr(chunk):
            clamped[i] = cutlass.min(scales[start + i], cutlass.Float32(UE5M3_MAX))
        converted = cute.make_rmem_tensor((4,), cutlass.FloatNV8E5M3FNU)
        # A 4-wide conversion lowers to two cvt.rp ue5m3x2 instructions.
        converted.store(clamped.load().to(cutlass.FloatNV8E5M3FNU))
        converted_bytes = cute.recast_tensor(converted, cutlass.Uint8)
        for i in cutlass.range_constexpr(chunk):
            code_bytes[start + i] = converted_bytes[i]
