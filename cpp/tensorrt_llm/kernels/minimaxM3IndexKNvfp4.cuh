/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// MiniMax-M3 NVFP4-E index-K page of 128 tokens: 128 rows of 64 B E2M1 codes, then 128 rows of 8 B E4M3 scales
// (one per 16 dims). Byte j of a row holds dim 2j in its low nibble and dim 2j + 1 in its high nibble; a code is
// sign << 3 | index into {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Scales are restricted to {1, 1.25, 1.5} x 2^e with e in
// [-6, 5], so every dequantized value is exactly an E4M3 value.
constexpr int kMinimaxM3Nvfp4EIndexKRowBytes = 64;
constexpr int kMinimaxM3Nvfp4EIndexKScaleOffset = 128 * kMinimaxM3Nvfp4EIndexKRowBytes;

//! Quantize one token's 128-dim index-K row to NVFP4-E and store it at `row` of the page at `pageBase`.
//!
//! All 32 lanes of the warp must call this; lane `lane` holds dims 4 * lane .. 4 * lane + 3 as the FP32 output
//! of norm and RoPE. The bytes are bit-exact with the reference quantizer: the values are rounded to BF16, and each
//! 16-dim group (four lanes) keeps the first of nine scales around amax / 6 that minimizes the squared error,
//! summed as a binary tree in dim order.
__device__ __forceinline__ void minimaxM3StoreNvfp4EIndexK(
    float const (&elements)[4], uint8_t* pageBase, int row, int lane)
{
    float values[4];
    float amax = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        values[i] = __bfloat162float(__float2bfloat16_rn(elements[i]));
        amax = fmaxf(amax, fabsf(values[i]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    // floor(log2(amax / 6)), read from the exponent field of the IEEE quotient.
    int const exponent0 = static_cast<int>(__float_as_uint(__fdiv_rn(amax, 6.0F)) >> 23) - 127;

    uint32_t bestCodes = 0;
    float bestScale = 0.0F;
    float bestError = 0.0F;
#pragma unroll
    for (int d = -1; d <= 1; ++d)
    {
        // Clamping the exponent to [-7, 6] keeps 2^exponent a normal float and does not change the clamped scale.
        int const exponent = min(max(exponent0 + d, -7), 6);
        float const powerOfTwo = __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
#pragma unroll
        for (int m = 0; m < 3; ++m)
        {
            float const scale = fminf(fmaxf((1.0F + 0.25F * m) * powerOfTwo, 0.015625F), 48.0F);
            uint32_t codes = 0;
            float squares[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                // |x| / scale to the nearest E2M1 magnitude, ties to even, saturating at 6. For a BF16 x, the
                // IEEE quotient lies on the same side of (or on) each midpoint t as x does of the exact t * scale,
                // so comparing x itself is identical and needs no division.
                float const a = fabsf(values[i]);
                int const index = (a > 0.25F * scale) + (a >= 0.75F * scale) + (a > 1.25F * scale)
                    + (a >= 1.75F * scale) + (a > 2.5F * scale) + (a >= 3.5F * scale) + (a > 5.0F * scale);
                float const magnitude
                    = index < 4 ? 0.5F * index : ((index & 1) ? 1.5F : 1.0F) * (index < 6 ? 2.0F : 4.0F);
                // The error decides the scale, so its rounding must not change: the _rn intrinsics are never
                // contracted into FMAs.
                float const diff = __fsub_rn(__fmul_rn(magnitude, scale), fabsf(values[i]));
                squares[i] = __fmul_rn(diff, diff);
                // Zero is stored as +0.
                codes |= static_cast<uint32_t>(index | (index != 0 && values[i] < 0.0F ? 8 : 0)) << (4 * i);
            }
            float error = __fadd_rn(__fadd_rn(squares[0], squares[1]), __fadd_rn(squares[2], squares[3]));
            error = __fadd_rn(error, __shfl_xor_sync(0xffffffff, error, 1));
            error = __fadd_rn(error, __shfl_xor_sync(0xffffffff, error, 2));
            if ((d == -1 && m == 0) || error < bestError)
            {
                bestCodes = codes;
                bestScale = scale;
                bestError = error;
            }
        }
    }

    *reinterpret_cast<uint16_t*>(pageBase + row * kMinimaxM3Nvfp4EIndexKRowBytes + 2 * lane)
        = static_cast<uint16_t>(bestCodes);
    if (lane % 4 == 0)
    {
        pageBase[kMinimaxM3Nvfp4EIndexKScaleOffset + row * 8 + lane / 4]
            = amax == 0.0F ? 0 : __nv_fp8_e4m3(bestScale).__x;
    }
}

//! Warps that store one token's index-K row with minimaxM3StoreNvfp4EIndexKQuarter, one quarter (two 16-dim groups)
//! each. Producers use them for launches of at most kMinimaxM3Nvfp4EIndexKSplitMaxTokens tokens, where the
//! nine-scale search of a one-warp store would be the launch's critical path; larger launches keep one warp per row.
constexpr int kMinimaxM3Nvfp4EIndexKWarps = 4;
constexpr int kMinimaxM3Nvfp4EIndexKSplitMaxTokens = 512;

//! Quantize dims [32 * quarter, 32 * quarter + 32) of one token's 128-dim index-K row to NVFP4-E and store them at
//! `row` of the page at `pageBase`.
//!
//! All 32 lanes of the warp must call this; lane `lane` holds dims 4 * lane .. 4 * lane + 3 of the whole row as the
//! FP32 output of norm and RoPE. The bytes are bit-exact with the reference quantizer: the values are rounded to BF16,
//! and each 16-dim group keeps the first of nine scales around amax / 6 that minimizes the squared error, summed as a
//! binary tree in dim order. One dim per lane shortens the nine-scale search.
__device__ __forceinline__ void minimaxM3StoreNvfp4EIndexKQuarter(
    float const (&elements)[4], uint8_t* pageBase, int row, int lane, int quarter)
{
    // Lane l takes dim 32 * quarter + l, held as element l % 4 by lane 8 * quarter + l / 4; lanes 0-15 and 16-31
    // are the two groups.
    int const sourceLane = 8 * quarter + lane / 4;
    float value = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float const shuffled = __shfl_sync(0xffffffff, elements[i], sourceLane);
        value = lane % 4 == i ? shuffled : value;
    }
    value = __bfloat162float(__float2bfloat16_rn(value));
    float const magnitudeIn = fabsf(value);
    float amax = magnitudeIn;
#pragma unroll
    for (int offset = 1; offset < 16; offset *= 2)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }
    // floor(log2(amax / 6)), read from the exponent field of the IEEE quotient.
    int const exponent0 = static_cast<int>(__float_as_uint(__fdiv_rn(amax, 6.0F)) >> 23) - 127;

    // |x| / scale to the nearest E2M1 magnitude index, ties to even, saturating at 6. For a BF16 x, the IEEE quotient
    // lies on the same side of (or on) each midpoint t as x does of the exact t * scale, so comparing x itself is
    // identical and needs no division.
    auto const e2m1Index = [magnitudeIn](float scale)
    {
        return (magnitudeIn > 0.25F * scale) + (magnitudeIn >= 0.75F * scale) + (magnitudeIn > 1.25F * scale)
            + (magnitudeIn >= 1.75F * scale) + (magnitudeIn > 2.5F * scale) + (magnitudeIn >= 3.5F * scale)
            + (magnitudeIn > 5.0F * scale);
    };

    float bestScale = 0.0F;
    float bestError = 0.0F;
#pragma unroll
    for (int d = -1; d <= 1; ++d)
    {
        // Clamping the exponent to [-7, 6] keeps 2^exponent a normal float and does not change the clamped scale.
        int const exponent = min(max(exponent0 + d, -7), 6);
        float const powerOfTwo = __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
#pragma unroll
        for (int m = 0; m < 3; ++m)
        {
            float const scale = fminf(fmaxf((1.0F + 0.25F * m) * powerOfTwo, 0.015625F), 48.0F);
            int const index = e2m1Index(scale);
            float const magnitude = index < 4 ? 0.5F * index : ((index & 1) ? 1.5F : 1.0F) * (index < 6 ? 2.0F : 4.0F);
            // The error decides the scale, so its rounding must not change: the _rn intrinsics are never contracted
            // into FMAs. Lanes 2j and 2j + 1 hold adjacent dims, so the butterfly sums the group's tree in dim order.
            float const diff = __fsub_rn(__fmul_rn(magnitude, scale), magnitudeIn);
            float error = __fmul_rn(diff, diff);
#pragma unroll
            for (int offset = 1; offset < 16; offset *= 2)
            {
                error = __fadd_rn(error, __shfl_xor_sync(0xffffffff, error, offset));
            }
            if ((d == -1 && m == 0) || error < bestError)
            {
                bestScale = scale;
                bestError = error;
            }
        }
    }

    // Zero is stored as +0. Byte j of the row holds dims 2j and 2j + 1, so lane 4k (k = 0..7) stores the two bytes
    // of lanes 4k .. 4k + 3.
    int const index = e2m1Index(bestScale);
    uint32_t const code = static_cast<uint32_t>(index | (index != 0 && value < 0.0F ? 8 : 0));
    uint32_t const byte = code | (__shfl_xor_sync(0xffffffff, code, 1) << 4);
    uint32_t const twoBytes = byte | (__shfl_down_sync(0xffffffff, byte, 2) << 8);
    if (lane % 4 == 0)
    {
        *reinterpret_cast<uint16_t*>(pageBase + row * kMinimaxM3Nvfp4EIndexKRowBytes + 16 * quarter + lane / 2)
            = static_cast<uint16_t>(twoBytes);
    }
    if (lane % 16 == 0)
    {
        pageBase[kMinimaxM3Nvfp4EIndexKScaleOffset + row * 8 + 2 * quarter + lane / 16]
            = amax == 0.0F ? 0 : __nv_fp8_e4m3(bestScale).__x;
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
