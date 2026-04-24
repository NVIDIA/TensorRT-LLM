/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Fast dequantization for NVFP4 (FP4 E2M1 weights -> BF16) and
 * FP8 E4M3 scale dequantization -> BF16.
 *
 * The FP4->BF16 dequant shifts the 3-bit FP4 value (sign + 2-bit exponent +
 * 1-bit mantissa) into the BF16 exponent/mantissa fields via bitwise ops.
 * A subsequent multiply by 2^(bias_offset) corrects the exponent bias
 * (skip_flop=false) or is deferred to fuse with scale multiply (skip_flop=true).
 */

#pragma once

#include "marlin_dtypes.cuh"

namespace MARLIN_NAMESPACE_NAME
{

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

// Lookup-table based 3-input logical operation; the compiler does not always
// recognize the pattern automatically.
template <int lut>
__device__ inline int lop3(int a, int b, int c)
{
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// =========================================================================
// FP4 E2M1 -> BF16 dequantization
// =========================================================================

// skip_flop=true: just place bits, caller multiplies exponent bias later.
template <bool skip_flop>
__device__ inline void dequant_fp4(int q, nv_bfloat162* frag_b)
{
    // Constants for FP4 (E2M1) -> BF16 (E8M7)
    constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
    constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
    constexpr int MASK = 0x70007000;

    // Extract and shift FP4 values to BF16 format
    int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
    q <<= 4;
    int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<nv_bfloat162 const*>(&Out1);
    frag_b[0] = *reinterpret_cast<nv_bfloat162 const*>(&Out2);

    if constexpr (!skip_flop)
    {
        // Apply exponent bias correction
        constexpr int BIAS_OFFSET = (1 << (BF16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
        constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
        const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<float const*>(&BIAS));

        frag_b[1] = __hmul2(frag_b[1], bias_reg);
        frag_b[0] = __hmul2(frag_b[0], bias_reg);
    }
}

// =========================================================================
// FP8 E4M3 scale -> BF16 dequantization
// =========================================================================

__device__ inline void dequant_fp8_scales(int q, nv_bfloat162* frag_b)
{
    constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
    constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
    constexpr int MASK = 0x7F007F00;

    // Extract and shift FP8 values to BF16 format
    int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
    q <<= 8;
    int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<nv_bfloat162 const*>(&Out1);
    frag_b[0] = *reinterpret_cast<nv_bfloat162 const*>(&Out2);
}

#endif

} // namespace MARLIN_NAMESPACE_NAME
