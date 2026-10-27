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

// The TP2 dense layers' QK norm + RoPE with FP8 output. The norm and RoPE are the generic fusedQKNormRopeKernel's at
// head_dim 128 without YaRN or mRoPE, expression for expression; the store rounds Q/K to BF16 and then to E4M3
// (saturating), the two roundings of the stock BF16 kernel followed by its FP8 cast.

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp2/denseQkNormRopeFp8.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp2
{
namespace denseQkNormRopeFp8
{
namespace
{

constexpr int kElemsPerThread = kHeadDim / 32;
constexpr int kVecSize = kElemsPerThread * sizeof(__nv_bfloat16) / 4;
constexpr int kBlockSize = 256;
static_assert(kVecSize == 2, "a lane loads 4 BF16 values as one uint2");

// tensorrt_llm::common::warpReduceSum<float>: the same butterfly, so the same sum.
__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val = val + __shfl_xor_sync(0xffffffff, val, mask, 32);
    }
    return val;
}

__device__ __forceinline__ void storeFp8ViaBf16(__nv_fp8_e4m3* out, long long offsetThread, float const (&elements)[4])
{
#pragma unroll
    for (int i = 0; i < kElemsPerThread; i += 2)
    {
        float2 const rounded = __bfloat1622float2(__float22bfloat162_rn(make_float2(elements[i], elements[i + 1])));
        __nv_fp8x2_e4m3 const packed(rounded);
        reinterpret_cast<__nv_fp8x2_storage_t*>(&out[offsetThread])[i / 2] = packed.__x;
    }
}

// The K (kv 0) or V (kv 1) row of one head of token tokenIdx in the HND paged pool, at the token's cache slot.
__device__ __forceinline__ __nv_fp8_e4m3* cacheRow(KvCacheView const& cache, int kv, int headIdx, int tokenIdx)
{
    long long const slot = static_cast<long long>(cache.outCacheLoc[tokenIdx]);
    long long const page = slot / cache.tokensPerBlock;
    long long const within = slot % cache.tokensPerBlock;
    return cache.base + page * cache.pageStride + kv * cache.kvStride + headIdx * cache.headStride
        + within * cache.tokenStride;
}

template <bool kInterleave, bool kWriteCache>
__global__ void denseQkNormRopeFp8Kernel(__nv_bfloat16 const* qkvIn, __nv_fp8_e4m3* qkvOut, int numHeadsQ,
    int numHeadsK, int numHeadsV, int rotaryDim, float eps, __nv_bfloat16 const* qWeight, __nv_bfloat16 const* kWeight,
    float base, int const* positionIds, int numTokens, bool useGemma, KvCacheView cache)
{
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;
    int const totalQkHeads = numHeadsQ + numHeadsK;
    int const totalHeads = totalQkHeads + numHeadsV;
    int const tokenIdx = globalWarpIdx / totalHeads;
    int const localHeadIdx = globalWarpIdx % totalHeads;
    if (tokenIdx >= numTokens)
    {
        return;
    }
    bool const isQ = localHeadIdx < numHeadsQ;
    bool const isV = localHeadIdx >= totalQkHeads;
    int headIdx;
    int segStart;
    if (isQ)
    {
        headIdx = localHeadIdx;
        segStart = 0;
    }
    else if (!isV)
    {
        headIdx = localHeadIdx - numHeadsQ;
        segStart = numHeadsQ * kHeadDim;
    }
    else
    {
        headIdx = localHeadIdx - totalQkHeads;
        segStart = totalQkHeads * kHeadDim;
    }

    using vec_T = uint2;
    int const offsetThread
        = tokenIdx * totalHeads * kHeadDim + segStart + headIdx * kHeadDim + laneId * kElemsPerThread;
    float elements[kElemsPerThread];
    float sumOfSquares = 0.0f;
    {
        vec_T vec = *reinterpret_cast<vec_T const*>(&qkvIn[offsetThread]);
#pragma unroll
        for (int i = 0; i < kVecSize; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
            sumOfSquares += vals.x * vals.x;
            sumOfSquares += vals.y * vals.y;
            elements[2 * i] = vals.x;
            elements[2 * i + 1] = vals.y;
        }
    }
    if (isV)
    {
        // V is BF16 already: its single E4M3 rounding is the stock cast's.
        storeFp8ViaBf16(qkvOut, offsetThread, elements);
        if (kWriteCache && tokenIdx < cache.numLiveTokens && cache.outCacheLoc[tokenIdx] >= 0)
        {
            storeFp8ViaBf16(cacheRow(cache, 1, headIdx, tokenIdx), laneId * kElemsPerThread, elements);
        }
        return;
    }

    sumOfSquares = warpReduceSum(sumOfSquares);
    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(kHeadDim) + eps);
    for (int i = 0; i < kElemsPerThread; i++)
    {
        int dim = laneId * kElemsPerThread + i;
        float weight = isQ ? __bfloat162float(qWeight[dim]) : __bfloat162float(kWeight[dim]);
        elements[i] *= rms_rcp * (useGemma ? (1.0f + weight) : weight);
    }

    float elements2[kElemsPerThread];
    float cos_vals[kElemsPerThread];
    float sin_vals[kElemsPerThread];
    float const neg2_log2base_over_rd = -2.0f * __log2f(base) / static_cast<float>(rotaryDim);
    int const rd_mask = rotaryDim - 1;
    bool const rd_is_pow2 = ((rotaryDim & rd_mask) == 0);
    float const pos_id = static_cast<float>(positionIds[tokenIdx]);
    if constexpr (kInterleave)
    {
        for (int i = 0; i < kElemsPerThread; i++)
        {
            if (i % 2 == 0)
            {
                elements2[i] = -elements[i + 1];
            }
            else
            {
                elements2[i] = elements[i - 1];
            }
            int dim_idx = laneId * kElemsPerThread + i;
            int half_dim = dim_idx / 2;
            float freq = exp2f(static_cast<float>(half_dim) * neg2_log2base_over_rd);
            float theta = pos_id * freq;
            __sincosf(theta, &sin_vals[i], &cos_vals[i]);
        }
    }
    else
    {
        __syncwarp();
        int pairOffset = (rotaryDim / 2) / kElemsPerThread;
        for (int i = 0; i < kElemsPerThread; i++)
        {
            elements2[i] = __shfl_xor_sync(0xffffffff, elements[i], pairOffset);
            if (laneId < pairOffset)
            {
                elements2[i] = -elements2[i];
            }
            int dim_idx = laneId * kElemsPerThread + i;
            dim_idx = rd_is_pow2 ? ((dim_idx * 2) & rd_mask) : ((dim_idx * 2) % rotaryDim);
            int half_dim = dim_idx / 2;
            float freq = exp2f(static_cast<float>(half_dim) * neg2_log2base_over_rd);
            float theta = pos_id * freq;
            __sincosf(theta, &sin_vals[i], &cos_vals[i]);
        }
        __syncwarp();
    }

    // attention_factor is 1 without YaRN: (x * cos + x2 * sin) * 1 rounds as the generic kernel's product does.
    float const attention_factor = 1.0f;
    if (rotaryDim == kHeadDim)
    {
        for (int i = 0; i < kElemsPerThread; i++)
        {
            elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
        }
    }
    else
    {
        for (int i = 0; i < kElemsPerThread; i++)
        {
            int dim_idx = laneId * kElemsPerThread + i;
            if (dim_idx < rotaryDim)
            {
                elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
            }
        }
    }
    storeFp8ViaBf16(qkvOut, offsetThread, elements);
    if constexpr (kWriteCache)
    {
        // Rows past the live count (padded graph rows) own no slot, nor does a negative (sentinel) slot.
        if (!isQ && tokenIdx < cache.numLiveTokens && cache.outCacheLoc[tokenIdx] >= 0)
        {
            storeFp8ViaBf16(cacheRow(cache, 0, headIdx, tokenIdx), laneId * kElemsPerThread, elements);
        }
        // The compact query [tokens, q heads, 128] the decode reads, beside the packed row.
        if (isQ && cache.qCompact != nullptr)
        {
            storeFp8ViaBf16(cache.qCompact,
                (static_cast<long long>(tokenIdx) * numHeadsQ + headIdx) * kHeadDim + laneId * kElemsPerThread,
                elements);
        }
    }
}

} // namespace

void launch(void const* qkvIn, void* qkvOut, int numTokens, int numHeadsQ, int numHeadsK, int numHeadsV, int rotaryDim,
    float eps, void const* qWeight, void const* kWeight, float base, bool interleave, int const* positionIds,
    bool useGemma, KvCacheView const* cache, cudaStream_t stream)
{
    int const totalWarps = numTokens * (numHeadsQ + numHeadsK + numHeadsV);
    int const warpsPerBlock = kBlockSize / 32;
    dim3 const grid((totalWarps + warpsPerBlock - 1) / warpsPerBlock);
    auto const* in = static_cast<__nv_bfloat16 const*>(qkvIn);
    auto* out = static_cast<__nv_fp8_e4m3*>(qkvOut);
    auto const* qw = static_cast<__nv_bfloat16 const*>(qWeight);
    auto const* kw = static_cast<__nv_bfloat16 const*>(kWeight);
    KvCacheView const view = cache != nullptr ? *cache : KvCacheView{};
    auto const run = [&](auto kernel)
    {
        kernel<<<grid, kBlockSize, 0, stream>>>(in, out, numHeadsQ, numHeadsK, numHeadsV, rotaryDim, eps, qw, kw, base,
            positionIds, numTokens, useGemma, view);
    };
    if (cache != nullptr)
    {
        interleave ? run(denseQkNormRopeFp8Kernel<true, true>) : run(denseQkNormRopeFp8Kernel<false, true>);
    }
    else
    {
        interleave ? run(denseQkNormRopeFp8Kernel<true, false>) : run(denseQkNormRopeFp8Kernel<false, false>);
    }
}

} // namespace denseQkNormRopeFp8
} // namespace minimaxM3Sm107Tp2
} // namespace kernels

TRTLLM_NAMESPACE_END
