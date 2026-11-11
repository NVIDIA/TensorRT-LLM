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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Four head chains with compile-time FP32 or packed native arithmetic."""
import cutlass
import cutlass.cute as cute
from .native_lowp_math import (
    native_pack, native_pack_relu, native_fp8_pack, native_fp8_pack_relu,
    native_fma, native_add, native_sum_fp32,
)


@cute.jit
def head_zero(dtype: cutlass.Constexpr = None):
    if cutlass.const_expr(dtype is None):
        result = cute.full((4,), 0.0, dtype=cutlass.Float32)
    else:
        result = cute.make_rmem_tensor((2,), cutlass.Uint32)
        result.fill(0)
    return result


@cute.jit
def head_sum(acc, dtype: cutlass.Constexpr = None):
    if cutlass.const_expr(dtype is None):
        result = acc[0] + acc[1] + acc[2] + acc[3]
    else:
        result = native_sum_fp32(dtype, native_add(dtype, acc[0], acc[1]))
    return result


@cute.jit
def native_head_fma4(x, w, acc, dtype: cutlass.Constexpr, fp8: cutlass.Constexpr):
    for pair in cutlass.range_constexpr(2):
        if cutlass.const_expr(fp8):
            xx = native_fp8_pack_relu(dtype, x[2 * pair], x[2 * pair + 1])
            ww = native_fp8_pack(dtype, w[2 * pair], w[2 * pair + 1])
        else:
            xx = native_pack_relu(dtype, x[2 * pair], x[2 * pair + 1])
            ww = native_pack(dtype, w[2 * pair], w[2 * pair + 1])
        acc[pair] = native_fma(dtype, xx, ww, acc[pair])
    return acc


@cute.jit
def native_weight_pair(weights, index, stage, dtype: cutlass.Constexpr,
                       fp8: cutlass.Constexpr = False):
    # Match DeepGEMM: one packed 32-bit shared load for two BF16 weights.
    # Weight storage and head arithmetic remain independent specializations.
    if cutlass.const_expr(weights.element_type == cutlass.BFloat16 and dtype == cutlass.BFloat16):
        pointer = weights.iterator + index * weights.stride[0] + stage * weights.stride[1]
        packed = cute.make_tensor(cute.recast_ptr(pointer, dtype=cutlass.Uint32), cute.make_layout(1))
        result = packed[0]
    else:
        a = cutlass.Float32(weights[(index, stage)])
        b = cutlass.Float32(weights[(index + 1, stage)])
        if cutlass.const_expr(fp8):
            result = native_fp8_pack(dtype, a, b)
        else:
            result = native_pack(dtype, a, b)
    return result
