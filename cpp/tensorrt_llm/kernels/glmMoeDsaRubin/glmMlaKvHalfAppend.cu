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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmMlaKvHalfAppend.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <algorithm>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{

constexpr int kNopeDim = kGlmKvHalfLatentDim;
constexpr int kRopeDim = kGlmKvHalfRopeDim;
// BF16 values per 16-byte vector.
constexpr int kEltsPerVec = 8;
constexpr int kNopeVecs = kNopeDim / kEltsPerVec;
constexpr int kRopeVecs = kRopeDim / kEltsPerVec;
// The cache row in 8-element units.
constexpr int kRowVecs = kNopeVecs + kRopeVecs;
constexpr int kWarpSize = 32;
constexpr int kNopeVecsPerLane = kNopeVecs / kWarpSize;
// Four token rows per block, one warp per row.
constexpr int kBlockSize = 128;
static_assert(kNopeVecs % kWarpSize == 0 && kRopeVecs <= kWarpSize, "the row splits over one warp");

inline __device__ float2 bf16x2ToFloat2(uint32_t word)
{
    return make_float2(__uint_as_float(word << 16), __uint_as_float(word & 0xFFFF0000u));
}

// Eight FP32 values to saturated E4M3.
inline __device__ uint2 toFp8x8(float const (&v)[kEltsPerVec])
{
    uint2 out;
    auto* pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
#pragma unroll
    for (int j = 0; j < kEltsPerVec / 2; ++j)
    {
        pairs[j] = __nv_cvt_float2_to_fp8x2(make_float2(v[2 * j], v[2 * j + 1]), __NV_SATFINITE, __NV_E4M3);
    }
    return out;
}

// One warp per token row. Lane l holds latent vectors l and l + 32 and, for l < 8, K-RoPE vector l.
// kTriggerAtEntry: the PDL dependents may launch at kernel entry instead of after the stores; they read this kernel's
// outputs only after their own grid-dependency wait.
template <int kBlockThreads, bool kTriggerAtEntry>
__global__ void __launch_bounds__(kBlockThreads)
    glmMlaKvHalfAppendKernel(__nv_bfloat16 const* __restrict__ latent, int latentRowStride, KVBlockArray kvCache,
        float2 const* __restrict__ cosSinCache, __nv_bfloat16 const* __restrict__ kvNormWeight, float kvNormEps,
        int numTokens, int seqLenQ, int32_t const* __restrict__ kvLens, float* __restrict__ bmm1Scale,
        float* __restrict__ bmm2Scale, float softmaxScale)
{
    constexpr int kRowsPerBlock = kBlockThreads / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    int const warpId = threadIdx.x / kWarpSize;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (kTriggerAtEntry)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    // The latent row is the previous kernel's output: wait before any global read.
    cudaGridDependencySynchronize();
#endif

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // The scales of the generation attention over this cache (unit KV scales).
        constexpr float kLog2e = 1.4426950408889634074f;
        bmm1Scale[0] = softmaxScale;
        bmm1Scale[1] = softmaxScale * kLog2e;
        bmm2Scale[0] = 1.f;
    }

    // The norm weight is row-invariant: two 8-element slices per lane, kept in FP32.
    float weight[kNopeVecsPerLane][kEltsPerVec];
#pragma unroll
    for (int i = 0; i < kNopeVecsPerLane; ++i)
    {
        uint4 const w = *reinterpret_cast<uint4 const*>(kvNormWeight + (lane + i * kWarpSize) * kEltsPerVec);
        uint32_t const words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float2 const f = bf16x2ToFloat2(words[j]);
            weight[i][2 * j] = f.x;
            weight[i][2 * j + 1] = f.y;
        }
    }

    for (int token = warpId + blockIdx.x * kRowsPerBlock; token < numTokens; token += kRowsPerBlock * gridDim.x)
    {
        int const seq = token / seqLenQ;
        int const tokenInSeq = token - seq * seqLenQ;
        // Cache slot and RoPE position, as the generation RoPE kernel derives them.
        int const tokenKvIdx = kvLens[seq] - seqLenQ + tokenInSeq;
        if (tokenKvIdx < 0)
        {
            continue;
        }
        __nv_bfloat16 const* row = latent + static_cast<size_t>(token) * static_cast<size_t>(latentRowStride);
        // Page-table read first: it does not depend on the reduction.
        auto* kDst = reinterpret_cast<uint8_t*>(kvCache.getKBlockPtr(seq, tokenKvIdx));

        uint4 nope[kNopeVecsPerLane];
#pragma unroll
        for (int i = 0; i < kNopeVecsPerLane; ++i)
        {
            nope[i] = *reinterpret_cast<uint4 const*>(row + (lane + i * kWarpSize) * kEltsPerVec);
        }
        uint4 rope = make_uint4(0, 0, 0, 0);
        if (lane < kRopeVecs)
        {
            rope = *reinterpret_cast<uint4 const*>(row + kNopeDim + lane * kEltsPerVec);
        }

        float x[kNopeVecsPerLane][kEltsPerVec];
        float sumSquares = 0.f;
#pragma unroll
        for (int i = 0; i < kNopeVecsPerLane; ++i)
        {
            uint32_t const words[4] = {nope[i].x, nope[i].y, nope[i].z, nope[i].w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float2 const f = bf16x2ToFloat2(words[j]);
                x[i][2 * j] = f.x;
                x[i][2 * j + 1] = f.y;
                sumSquares += f.x * f.x + f.y * f.y;
            }
        }
#pragma unroll
        for (int mask = kWarpSize / 2; mask > 0; mask >>= 1)
        {
            sumSquares += __shfl_xor_sync(0xFFFFFFFFu, sumSquares, mask);
        }
        float const rstd = rsqrtf(sumSquares / static_cast<float>(kNopeDim) + kvNormEps);

#pragma unroll
        for (int i = 0; i < kNopeVecsPerLane; ++i)
        {
            float scaled[kEltsPerVec];
#pragma unroll
            for (int j = 0; j < kEltsPerVec; ++j)
            {
                scaled[j] = x[i][j] * (rstd * weight[i][j]);
            }
            int const vec = lane + i * kWarpSize;
            *reinterpret_cast<uint2*>(kDst + kvCache.getKVLocalIdx(tokenKvIdx, 0, kRowVecs, vec) * kEltsPerVec)
                = toFp8x8(scaled);
        }

        if (lane < kRopeVecs)
        {
            // Pair p of the RoPE slice uses entry kRopeDim * position + p of the table.
            float2 const* coef = cosSinCache + static_cast<size_t>(kRopeDim) * tokenKvIdx + lane * (kEltsPerVec / 2);
            uint32_t const words[4] = {rope.x, rope.y, rope.z, rope.w};
            float rotated[kEltsPerVec];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float2 const v = bf16x2ToFloat2(words[j]);
                float2 const cs = coef[j];
                rotated[2 * j] = cs.x * v.x - cs.y * v.y;
                rotated[2 * j + 1] = cs.x * v.y + cs.y * v.x;
            }
            *reinterpret_cast<uint2*>(
                kDst + kvCache.getKVLocalIdx(tokenKvIdx, 0, kRowVecs, kNopeVecs + lane) * kEltsPerVec)
                = toFp8x8(rotated);
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (!kTriggerAtEntry)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

} // namespace

void invokeGlmMlaKvHalfAppend(__nv_bfloat16 const* latent, int32_t latentRowStride, KVBlockArray const& kvCache,
    float2 const* cosSinCache, __nv_bfloat16 const* kvNormWeight, float kvNormEps, int32_t numTokens, int32_t seqLenQ,
    int32_t const* kvLens, float* bmm1Scale, float* bmm2Scale, float softmaxScale, bool triggerAtEntry,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(latent != nullptr && cosSinCache != nullptr && kvNormWeight != nullptr && kvLens != nullptr
            && bmm1Scale != nullptr && bmm2Scale != nullptr,
        "The GLM MLA KV append needs the latent rows, the rope table, the norm weight, the KV lengths and the scales");
    TLLM_CHECK_WITH_INFO(latentRowStride >= kNopeDim + kRopeDim && latentRowStride % kEltsPerVec == 0,
        "latent row stride %d must cover a %d-column row and be a multiple of %d", latentRowStride, kNopeDim + kRopeDim,
        kEltsPerVec);
    TLLM_CHECK_WITH_INFO(numTokens >= 0 && seqLenQ > 0 && numTokens % seqLenQ == 0,
        "MLA can only support input sequences with the same sequence length.");

    constexpr int kRowsPerBlock = kBlockSize / kWarpSize;
    dim3 const grid(static_cast<int>(tensorrt_llm::common::divUp(std::max(numTokens, 1), kRowsPerBlock)));
    auto const kernel
        = triggerAtEntry ? glmMlaKvHalfAppendKernel<kBlockSize, true> : glmMlaKvHalfAppendKernel<kBlockSize, false>;
    tensorrt_llm::common::launchWithPdlWhenEnabled("glmMlaKvHalfAppend", kernel, grid, kBlockSize, 0, stream, latent,
        static_cast<int>(latentRowStride), kvCache, cosSinCache, kvNormWeight, kvNormEps, static_cast<int>(numTokens),
        static_cast<int>(seqLenQ), kvLens, bmm1Scale, bmm2Scale, softmaxScale);
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
