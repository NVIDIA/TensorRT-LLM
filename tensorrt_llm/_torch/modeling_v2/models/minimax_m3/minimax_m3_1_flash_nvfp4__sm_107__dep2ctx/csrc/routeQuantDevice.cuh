/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// DEP_ROUTER_FOLD boundary: the device pieces of noaux_tc_fp4_quantize_stage_out that the
// post-attention add + RMSNorm runs on the rows it holds. Each piece reproduces the stock code it names bit for bit
// (the UT compares the staged bytes): the single-group noaux top-k with its routing weight (noAuxTcKernels.cu
// deepseek_v3_topk_block, non-group branch, and noAuxTcRoutingWeight) and the NVFP4 conversion of one 8-element
// vector with a 16-element block shared by an adjacent thread pair (quantization.cuh cvt_warp_fp16_to_fp4<bf16, 16,
// false> with the linear scale layout).
#pragma once

#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace route_quant
{
namespace cg = cooperative_groups;

//! noAuxTcKernels.cu sigmoid_accurate.
__device__ __forceinline__ float sigmoidAccurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

//! noAuxTcKernels.cu scalingFactorAsShortFloat.
__device__ __forceinline__ bool scalingFactorAsShortFloat(double factor, float& factorFloat)
{
    uint64_t const bits = static_cast<uint64_t>(__double_as_longlong(factor));
    int const exponent = static_cast<int>((bits >> 52) & 0x7FFU) - 1023;
    uint64_t const mantissa = bits & ((uint64_t{1} << 52) - 1);
    factorFloat
        = __uint_as_float((static_cast<uint32_t>(exponent + 127) << 23) | static_cast<uint32_t>(mantissa >> 29));
    bool const shortMantissa = (mantissa & ((uint64_t{1} << 33) - 1)) == 0;
    bool const inRange = exponent >= -20 && (exponent < 20 || (exponent == 20 && mantissa == 0));
    return (bits >> 63) == 0 && shortMantissa && inRange;
}

//! noAuxTcKernels.cu noAuxTcRoutingWeightFp64.
__device__ __noinline__ float routingWeightFp64(float s, float S, double c)
{
    return static_cast<float>(s * c / (S + 1e-20));
}

//! noAuxTcKernels.cu noAuxTcRoutingWeight (the FP64 expression without FP64 instructions on the common path).
__device__ __forceinline__ float routingWeight(float s, float S, double c)
{
    float cFloat;
    if (scalingFactorAsShortFloat(c, cFloat) && S >= 0x1p-13f)
    {
        float const pHi = __fmul_rn(s, cFloat);
        if (pHi >= 0x1p-60f)
        {
            float const pLo = __fmaf_rn(s, cFloat, -pHi);
            float const q = __fdiv_rn(pHi, S);
            float const r = __fmaf_rn(-q, S, pHi);
            float const e = __fdiv_rn(__fadd_rn(r, pLo), S);
            float const y = __fadd_rn(q, e);
            uint32_t const bits = __float_as_uint(y);
            if ((bits & 0x7FFFFFU) != 0 && (bits & 0x7F800000U) != 0x7F800000U)
            {
                float const ulp = __uint_as_float(bits & 0x7F800000U) * 0x1p-23f;
                float const g = __fsub_rn(e, __fsub_rn(y, q));
                if (fabsf(g) < 0.5F * ulp - 0x1p-20f * ulp)
                {
                    return y;
                }
            }
        }
    }
    return routingWeightFp64(s, S, c);
}

//! deepseek_v3_topk_block's single-group selection for one row whose sigmoid scores and biased scores are in shared
//! memory (sig / biased, numExperts entries), run by warp 0 of the CTA: the top-k ids and their weights
//! s * c / (S + 1e-20) (Fp32Weight: noAuxTcRoutingWeight, else the float expression), lanes < topk storing
//! ids[lane] / weights[lane]. MaxNumExperts is the stock expert tier (128 or 256), MaxNumTopExperts 32 as the
//! route-quant launch instantiates it.
template <int MaxNumExperts, bool Fp32Weight, int MaxNumTopExperts = 32>
__device__ __forceinline__ void topkWarp0(cg::thread_block_tile<32> const& warp, float const* sig, float const* biased,
    int numExperts, int topk, double routedScalingFactor, int32_t* ids, float* weights)
{
    int32_t const laneIdx = threadIdx.x % 32;
    static constexpr float invalidScoreFloat = float{-INFINITY};
    constexpr int NumChunks = (MaxNumExperts + 32 - 1) / 32;
    float topScores[MaxNumTopExperts];
    int32_t topExperts[MaxNumTopExperts];
    float localScores[NumChunks];
    int32_t localIdx[NumChunks];
#pragma unroll
    for (int ii = 0; ii < NumChunks; ++ii)
    {
        auto expertIdx = ii * 32 + laneIdx;
        localIdx[ii] = expertIdx;
        localScores[ii] = expertIdx < numExperts ? biased[expertIdx] : invalidScoreFloat;
    }
    tensorrt_llm::kernels::reduce_topk::reduceTopK(
        warp, topScores, topExperts, localScores, localIdx, invalidScoreFloat, topk);
    int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
    float scoreNorm = laneIdx < topk ? sig[expertIdx] : 0.F;
    auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
    if constexpr (Fp32Weight)
    {
        if (laneIdx < topk)
        {
            weights[laneIdx] = routingWeight(scoreNorm, redNorm, routedScalingFactor);
            ids[laneIdx] = expertIdx;
        }
    }
    else
    {
        auto finalScore = static_cast<float>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
        if (laneIdx < topk)
        {
            weights[laneIdx] = finalScore;
            ids[laneIdx] = expertIdx;
        }
    }
}

//! quantization.cuh cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false> of the 8 BF16 values packed in v, the 16-value
//! block shared with the adjacent lane (lane ^ 1, same code path); returns the 8 E2M1 codes, writes the E4M3 block
//! scale to sfOut when it is not null (the even lane of the pair).
__device__ __forceinline__ uint32_t cvtBf16x8ToFp4(uint4 v, float sfScaleVal, uint8_t* sfOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __nv_bfloat162 const* e = reinterpret_cast<__nv_bfloat162 const*>(&v);
    __nv_bfloat162 localMax = __habs2(e[0]);
#pragma unroll
    for (int i = 1; i < 4; i++)
    {
        localMax = __hmax2(localMax, __habs2(e[i]));
    }
    uint32_t laneId = 0;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    uint32_t const sfGroupMask = 0x3U << (laneId & ~1U);
    localMax = __hmax2(__shfl_xor_sync(sfGroupMask, localMax, 1), localMax);
    float vecMax = float(__hmax(localMax.x, localMax.y));
    float rcp6;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(rcp6) : "f"(6.0f));
    auto SFValue = sfScaleVal * (vecMax * rcp6);
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    uint8_t const fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
    float rcpScale, outputScale = 0.0f;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(rcpScale) : "f"(sfScaleVal));
    if (vecMax != 0)
    {
        float const prod = SFValue * rcpScale;
        asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(outputScale) : "f"(prod));
    }
    if (sfOut)
    {
        *sfOut = fp8SFVal;
    }
    float2 f[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        f[i] = __bfloat1622float2(e[i]);
        f[i].x *= outputScale;
        f[i].y *= outputScale;
    }
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(f[0].x), "f"(f[0].y), "f"(f[1].x), "f"(f[1].y), "f"(f[2].x), "f"(f[2].y), "f"(f[3].x), "f"(f[3].y));
    return val;
#else
    return 0;
#endif
}

} // namespace route_quant
} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
