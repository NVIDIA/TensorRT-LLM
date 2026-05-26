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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

constexpr int kReferenceHeadDim = 128;
constexpr int kTurboQuant4AttentionTileSize = 128;
constexpr int kTurboQuant4MaxTiledQueryTokens = 256;
constexpr int64_t kTurboQuant4MaxGridX = std::numeric_limits<int>::max();

__constant__ float kTurboQuant4Centroids[16] = {-0.1739F, -0.1172F, -0.0895F, -0.0688F, -0.0513F, -0.0356F, -0.0210F,
    -0.0069F, 0.0069F, 0.0210F, 0.0356F, 0.0513F, 0.0688F, 0.0895F, 0.1172F, 0.1739F};

__constant__ float kTurboQuant4Boundaries[15] = {(-0.1739F + -0.1172F) / 2.0F, (-0.1172F + -0.0895F) / 2.0F,
    (-0.0895F + -0.0688F) / 2.0F, (-0.0688F + -0.0513F) / 2.0F, (-0.0513F + -0.0356F) / 2.0F,
    (-0.0356F + -0.0210F) / 2.0F, (-0.0210F + -0.0069F) / 2.0F, (-0.0069F + 0.0069F) / 2.0F, (0.0069F + 0.0210F) / 2.0F,
    (0.0210F + 0.0356F) / 2.0F, (0.0356F + 0.0513F) / 2.0F, (0.0513F + 0.0688F) / 2.0F, (0.0688F + 0.0895F) / 2.0F,
    (0.0895F + 0.1172F) / 2.0F, (0.1172F + 0.1739F) / 2.0F};

template <typename T>
__device__ float toFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ float toFloat<half>(half value)
{
    return __half2float(value);
}

#ifdef ENABLE_BF16
template <>
__device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}
#endif

template <typename T>
__device__ T fromFloat(float value)
{
    return static_cast<T>(value);
}

template <>
__device__ half fromFloat<half>(float value)
{
    return __float2half_rn(value);
}

#ifdef ENABLE_BF16
template <>
__device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}
#endif

__device__ int bucketizeTurboQuant4(float value, float gridScale)
{
    int index = 0;
#pragma unroll
    for (int i = 0; i < 15; ++i)
    {
        if (value > kTurboQuant4Boundaries[i] * gridScale)
        {
            ++index;
        }
    }
    return index;
}

__device__ float turboQuant4Centroid(int index, float gridScale)
{
    return kTurboQuant4Centroids[index] * gridScale;
}

__device__ void fwhtInShared(float* values, int headDim)
{
    int const tid = threadIdx.x;
    for (int h = 1; h < headDim; h *= 2)
    {
        int const pairCount = headDim / 2;
        for (int pair = tid; pair < pairCount; pair += blockDim.x)
        {
            int const group = pair / h;
            int const j = pair - group * h;
            int const first = group * 2 * h + j;
            int const second = first + h;
            float const a = values[first];
            float const b = values[second];
            values[first] = a + b;
            values[second] = a - b;
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void turboQuant4QuantizeKernel(
    T const* input, uint8_t* codes, float* scales, int64_t numVectors, int headDim)
{
    extern __shared__ float values[];
    __shared__ float sumSq;
    __shared__ float sumDot;
    __shared__ float sumCodeSq;

    int64_t const vectorIdx = blockIdx.x;
    if (vectorIdx >= numVectors)
    {
        return;
    }

    int const tid = threadIdx.x;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        values[i] = toFloat(input[vectorIdx * headDim + i]);
    }
    __syncthreads();

    fwhtInShared(values, headDim);

    float const invSqrtHeadDim = rsqrtf(static_cast<float>(headDim));
    float localSumSq = 0.0F;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        values[i] *= invSqrtHeadDim;
        localSumSq += values[i] * values[i];
    }
    if (tid == 0)
    {
        sumSq = 0.0F;
    }
    __syncthreads();
    atomicAdd(&sumSq, localSumSq);
    __syncthreads();

    float const normalizationScale = sqrtf(sumSq) + 1.0e-10F;
    float const gridScale = sqrtf(static_cast<float>(kReferenceHeadDim) / static_cast<float>(headDim));
    float localDot = 0.0F;
    float localCodeSq = 0.0F;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        int const index = bucketizeTurboQuant4(values[i] / normalizationScale, gridScale);
        float const centroid = turboQuant4Centroid(index, gridScale);
        localDot += values[i] * centroid;
        localCodeSq += centroid * centroid;
    }
    if (tid == 0)
    {
        sumDot = 0.0F;
        sumCodeSq = 0.0F;
    }
    __syncthreads();
    atomicAdd(&sumDot, localDot);
    atomicAdd(&sumCodeSq, localCodeSq);
    __syncthreads();

    float const vectorScale = fmaxf(sumDot / fmaxf(sumCodeSq, 1.0e-10F), 1.0e-10F);
    if (tid == 0)
    {
        scales[vectorIdx] = vectorScale;
    }

    int const halfHeadDim = headDim / 2;
    for (int i = tid; i < halfHeadDim; i += blockDim.x)
    {
        int const low = bucketizeTurboQuant4(values[2 * i] / normalizationScale, gridScale);
        int const high = bucketizeTurboQuant4(values[2 * i + 1] / normalizationScale, gridScale);
        codes[vectorIdx * halfHeadDim + i] = static_cast<uint8_t>(low | (high << 4));
    }
}

template <typename T>
__global__ void turboQuant4DequantizeKernel(
    uint8_t const* codes, float const* scales, T* output, int64_t numVectors, int headDim)
{
    extern __shared__ float values[];

    int64_t const vectorIdx = blockIdx.x;
    if (vectorIdx >= numVectors)
    {
        return;
    }

    int const tid = threadIdx.x;
    int const halfHeadDim = headDim / 2;
    float const vectorScale = scales[vectorIdx];
    float const gridScale = sqrtf(static_cast<float>(kReferenceHeadDim) / static_cast<float>(headDim));
    for (int i = tid; i < halfHeadDim; i += blockDim.x)
    {
        uint8_t const packed = codes[vectorIdx * halfHeadDim + i];
        values[2 * i] = turboQuant4Centroid(packed & 0x0F, gridScale) * vectorScale;
        values[2 * i + 1] = turboQuant4Centroid((packed >> 4) & 0x0F, gridScale) * vectorScale;
    }
    __syncthreads();

    fwhtInShared(values, headDim);

    float const invSqrtHeadDim = rsqrtf(static_cast<float>(headDim));
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        output[vectorIdx * headDim + i] = fromFloat<T>(values[i] * invSqrtHeadDim);
    }
}

template <typename T>
__global__ void turboQuant4UpdateCacheKernel(T const* input, uint8_t* cache, float* scaleCache, int32_t const* blockIds,
    int64_t numVectors, int numHeads, int headDim, int kvFactor, int kvIndex, int startPos, int tokensPerBlock)
{
    extern __shared__ float values[];
    __shared__ float sumSq;
    __shared__ float sumDot;
    __shared__ float sumCodeSq;

    int64_t const vectorIdx = blockIdx.x;
    if (vectorIdx >= numVectors)
    {
        return;
    }

    int const tokenIdx = static_cast<int>(vectorIdx / numHeads);
    int const headIdx = static_cast<int>(vectorIdx - static_cast<int64_t>(tokenIdx) * numHeads);
    int const tokenPos = startPos + tokenIdx;
    int const blockListPos = tokenPos / tokensPerBlock;
    int const blockOffset = tokenPos % tokensPerBlock;
    int const blockId = blockIds[blockListPos];

    int const tid = threadIdx.x;
    int64_t const inputOffset = (static_cast<int64_t>(tokenIdx) * numHeads + headIdx) * headDim;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        values[i] = toFloat(input[inputOffset + i]);
    }
    __syncthreads();

    fwhtInShared(values, headDim);

    float const invSqrtHeadDim = rsqrtf(static_cast<float>(headDim));
    float localSumSq = 0.0F;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        values[i] *= invSqrtHeadDim;
        localSumSq += values[i] * values[i];
    }
    if (tid == 0)
    {
        sumSq = 0.0F;
    }
    __syncthreads();
    atomicAdd(&sumSq, localSumSq);
    __syncthreads();

    float const normalizationScale = sqrtf(sumSq) + 1.0e-10F;
    float const gridScale = sqrtf(static_cast<float>(kReferenceHeadDim) / static_cast<float>(headDim));
    float localDot = 0.0F;
    float localCodeSq = 0.0F;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        int const index = bucketizeTurboQuant4(values[i] / normalizationScale, gridScale);
        float const centroid = turboQuant4Centroid(index, gridScale);
        localDot += values[i] * centroid;
        localCodeSq += centroid * centroid;
    }
    if (tid == 0)
    {
        sumDot = 0.0F;
        sumCodeSq = 0.0F;
    }
    __syncthreads();
    atomicAdd(&sumDot, localDot);
    atomicAdd(&sumCodeSq, localCodeSq);
    __syncthreads();

    float const vectorScale = fmaxf(sumDot / fmaxf(sumCodeSq, 1.0e-10F), 1.0e-10F);
    int const halfHeadDim = headDim / 2;
    int64_t const cacheVectorOffset
        = (((static_cast<int64_t>(blockId) * kvFactor + kvIndex) * tokensPerBlock + blockOffset) * numHeads + headIdx)
        * halfHeadDim;
    int64_t const scaleOffset
        = ((static_cast<int64_t>(blockId) * kvFactor + kvIndex) * tokensPerBlock + blockOffset) * numHeads + headIdx;
    if (tid == 0)
    {
        scaleCache[scaleOffset] = vectorScale;
    }

    for (int i = tid; i < halfHeadDim; i += blockDim.x)
    {
        int const low = bucketizeTurboQuant4(values[2 * i] / normalizationScale, gridScale);
        int const high = bucketizeTurboQuant4(values[2 * i + 1] / normalizationScale, gridScale);
        cache[cacheVectorOffset + i] = static_cast<uint8_t>(low | (high << 4));
    }
}

template <typename T>
__global__ void turboQuant4DequantizeCacheKernel(uint8_t const* cache, float const* scaleCache, int32_t const* blockIds,
    T* output, int64_t numVectors, int numHeads, int headDim, int kvFactor, int kvIndex, int tokensPerBlock)
{
    extern __shared__ float values[];

    int64_t const vectorIdx = blockIdx.x;
    if (vectorIdx >= numVectors)
    {
        return;
    }

    int const tokenIdx = static_cast<int>(vectorIdx / numHeads);
    int const headIdx = static_cast<int>(vectorIdx - static_cast<int64_t>(tokenIdx) * numHeads);
    int const blockListPos = tokenIdx / tokensPerBlock;
    int const blockOffset = tokenIdx % tokensPerBlock;
    int const blockId = blockIds[blockListPos];

    int const tid = threadIdx.x;
    int const halfHeadDim = headDim / 2;
    int64_t const cacheVectorOffset
        = (((static_cast<int64_t>(blockId) * kvFactor + kvIndex) * tokensPerBlock + blockOffset) * numHeads + headIdx)
        * halfHeadDim;
    int64_t const scaleOffset
        = ((static_cast<int64_t>(blockId) * kvFactor + kvIndex) * tokensPerBlock + blockOffset) * numHeads + headIdx;
    float const vectorScale = scaleCache[scaleOffset];
    float const gridScale = sqrtf(static_cast<float>(kReferenceHeadDim) / static_cast<float>(headDim));
    for (int i = tid; i < halfHeadDim; i += blockDim.x)
    {
        uint8_t const packed = cache[cacheVectorOffset + i];
        values[2 * i] = turboQuant4Centroid(packed & 0x0F, gridScale) * vectorScale;
        values[2 * i + 1] = turboQuant4Centroid((packed >> 4) & 0x0F, gridScale) * vectorScale;
    }
    __syncthreads();

    fwhtInShared(values, headDim);

    float const invSqrtHeadDim = rsqrtf(static_cast<float>(headDim));
    int64_t const outputOffset = (static_cast<int64_t>(tokenIdx) * numHeads + headIdx) * headDim;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        output[outputOffset + i] = fromFloat<T>(values[i] * invSqrtHeadDim);
    }
}

__device__ float blockReduceSum(float value, float* scratch)
{
    int const tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ void loadTurboQuant4CacheVector(uint8_t const* cache, float const* scaleCache, int32_t const* blockIds,
    int kvFactor, int kvIndex, int tokenIdx, int headIdx, int numHeads, int headDim, int tokensPerBlock, float* values)
{
    int const blockListPos = tokenIdx / tokensPerBlock;
    int const blockOffset = tokenIdx % tokensPerBlock;
    int const blockId = blockIds[blockListPos];
    int const tid = threadIdx.x;
    int const halfHeadDim = headDim / 2;
    int64_t const cacheVectorOffset
        = (((static_cast<int64_t>(blockId) * kvFactor + kvIndex) * tokensPerBlock + blockOffset) * numHeads + headIdx)
        * halfHeadDim;
    int64_t const scaleOffset
        = ((static_cast<int64_t>(blockId) * kvFactor + kvIndex) * tokensPerBlock + blockOffset) * numHeads + headIdx;
    float const vectorScale = scaleCache[scaleOffset];
    float const gridScale = sqrtf(static_cast<float>(kReferenceHeadDim) / static_cast<float>(headDim));
    for (int i = tid; i < halfHeadDim; i += blockDim.x)
    {
        uint8_t const packed = cache[cacheVectorOffset + i];
        values[2 * i] = turboQuant4Centroid(packed & 0x0F, gridScale) * vectorScale;
        values[2 * i + 1] = turboQuant4Centroid((packed >> 4) & 0x0F, gridScale) * vectorScale;
    }
    __syncthreads();

    fwhtInShared(values, headDim);

    float const invSqrtHeadDim = rsqrtf(static_cast<float>(headDim));
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        values[i] *= invSqrtHeadDim;
    }
    __syncthreads();
}

__device__ bool turboQuant4AttentionTokenAllowed(
    int kvTokenIdx, int queryPosition, bool isCausal, int attentionWindowSize)
{
    if ((isCausal || attentionWindowSize > 0) && kvTokenIdx > queryPosition)
    {
        return false;
    }
    if (attentionWindowSize > 0 && kvTokenIdx <= queryPosition - attentionWindowSize)
    {
        return false;
    }
    return true;
}

template <typename T>
__global__ void turboQuant4AttentionKernel(T const* q, uint8_t const* cache, float const* scaleCache,
    int32_t const* blockIds, T* output, int seqLen, int numHeads, int numKvHeads, int headDim, int kvFactor,
    int qStartPos, float qScaling, bool isCausal, int attentionWindowSize, int tokensPerBlock)
{
    extern __shared__ float shared[];
    float* qValues = shared;
    float* values = qValues + headDim;
    float* outValues = values + headDim;
    float* scratch = outValues + headDim;
    __shared__ float sharedMax;
    __shared__ float sharedSum;
    __shared__ float sharedWeight;

    int const qTokenIdx = blockIdx.x;
    int const qHeadIdx = blockIdx.y;
    int const tid = threadIdx.x;
    int const kvGroupSize = numHeads / numKvHeads;
    int const kvHeadIdx = qHeadIdx / kvGroupSize;
    int const queryPosition = qStartPos + qTokenIdx;
    float const qkScale = 1.0F / (sqrtf(static_cast<float>(headDim)) * qScaling);

    int64_t const qOffset = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * headDim;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        qValues[i] = toFloat(q[qOffset + i]);
        outValues[i] = 0.0F;
    }
    if (tid == 0)
    {
        sharedMax = -FLT_MAX;
        sharedSum = 0.0F;
    }
    __syncthreads();

    for (int kvTokenIdx = 0; kvTokenIdx < seqLen; ++kvTokenIdx)
    {
        if (!turboQuant4AttentionTokenAllowed(kvTokenIdx, queryPosition, isCausal, attentionWindowSize))
        {
            continue;
        }
        loadTurboQuant4CacheVector(cache, scaleCache, blockIds, kvFactor, 0, kvTokenIdx, kvHeadIdx, numKvHeads, headDim,
            tokensPerBlock, values);
        float localDot = 0.0F;
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            localDot += qValues[i] * values[i];
        }
        float const dot = blockReduceSum(localDot, scratch) * qkScale;
        if (tid == 0)
        {
            sharedMax = fmaxf(sharedMax, dot);
        }
        __syncthreads();
    }

    for (int kvTokenIdx = 0; kvTokenIdx < seqLen; ++kvTokenIdx)
    {
        if (!turboQuant4AttentionTokenAllowed(kvTokenIdx, queryPosition, isCausal, attentionWindowSize))
        {
            continue;
        }
        loadTurboQuant4CacheVector(cache, scaleCache, blockIds, kvFactor, 0, kvTokenIdx, kvHeadIdx, numKvHeads, headDim,
            tokensPerBlock, values);
        float localDot = 0.0F;
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            localDot += qValues[i] * values[i];
        }
        float const dot = blockReduceSum(localDot, scratch) * qkScale;
        if (tid == 0)
        {
            sharedWeight = expf(dot - sharedMax);
            sharedSum += sharedWeight;
        }
        __syncthreads();
        loadTurboQuant4CacheVector(cache, scaleCache, blockIds, kvFactor, 1, kvTokenIdx, kvHeadIdx, numKvHeads, headDim,
            tokensPerBlock, values);
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            outValues[i] += sharedWeight * values[i];
        }
        __syncthreads();
    }

    int64_t const outputOffset = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * headDim;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        output[outputOffset + i] = fromFloat<T>(sharedSum > 0.0F ? outValues[i] / sharedSum : 0.0F);
    }
}

template <typename T>
__global__ void turboQuant4AttentionPartialKernel(T const* q, uint8_t const* cache, float const* scaleCache,
    int32_t const* blockIds, float* partialMax, float* partialSum, float* partialOut, int seqLen, int numHeads,
    int numKvHeads, int headDim, int kvFactor, int qStartPos, float qScaling, bool isCausal, int attentionWindowSize,
    int tokensPerBlock, int numTiles, int tileSize)
{
    extern __shared__ float shared[];
    float* qValues = shared;
    float* values = qValues + headDim;
    float* outValues = values + headDim;
    float* scratch = outValues + headDim;
    __shared__ float sharedMax;
    __shared__ float sharedSum;
    __shared__ float sharedWeight;

    int const qTokenIdx = blockIdx.x;
    int const qHeadIdx = blockIdx.y;
    int const tileIdx = blockIdx.z;
    int const tid = threadIdx.x;
    int const kvGroupSize = numHeads / numKvHeads;
    int const kvHeadIdx = qHeadIdx / kvGroupSize;
    int const queryPosition = qStartPos + qTokenIdx;
    int const tileStart = tileIdx * tileSize;
    int const tileEndCandidate = tileStart + tileSize;
    int const tileEnd = tileEndCandidate < seqLen ? tileEndCandidate : seqLen;
    float const qkScale = 1.0F / (sqrtf(static_cast<float>(headDim)) * qScaling);

    int64_t const qOffset = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * headDim;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        qValues[i] = toFloat(q[qOffset + i]);
        outValues[i] = 0.0F;
    }
    if (tid == 0)
    {
        sharedMax = -FLT_MAX;
        sharedSum = 0.0F;
    }
    __syncthreads();

    for (int kvTokenIdx = tileStart; kvTokenIdx < tileEnd; ++kvTokenIdx)
    {
        if (!turboQuant4AttentionTokenAllowed(kvTokenIdx, queryPosition, isCausal, attentionWindowSize))
        {
            continue;
        }
        loadTurboQuant4CacheVector(cache, scaleCache, blockIds, kvFactor, 0, kvTokenIdx, kvHeadIdx, numKvHeads, headDim,
            tokensPerBlock, values);
        float localDot = 0.0F;
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            localDot += qValues[i] * values[i];
        }
        float const dot = blockReduceSum(localDot, scratch) * qkScale;
        if (tid == 0)
        {
            sharedMax = fmaxf(sharedMax, dot);
        }
        __syncthreads();
    }

    for (int kvTokenIdx = tileStart; kvTokenIdx < tileEnd; ++kvTokenIdx)
    {
        if (!turboQuant4AttentionTokenAllowed(kvTokenIdx, queryPosition, isCausal, attentionWindowSize))
        {
            continue;
        }
        loadTurboQuant4CacheVector(cache, scaleCache, blockIds, kvFactor, 0, kvTokenIdx, kvHeadIdx, numKvHeads, headDim,
            tokensPerBlock, values);
        float localDot = 0.0F;
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            localDot += qValues[i] * values[i];
        }
        float const dot = blockReduceSum(localDot, scratch) * qkScale;
        if (tid == 0)
        {
            sharedWeight = expf(dot - sharedMax);
            sharedSum += sharedWeight;
        }
        __syncthreads();
        loadTurboQuant4CacheVector(cache, scaleCache, blockIds, kvFactor, 1, kvTokenIdx, kvHeadIdx, numKvHeads, headDim,
            tokensPerBlock, values);
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            outValues[i] += sharedWeight * values[i];
        }
        __syncthreads();
    }

    int64_t const partialBase = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * numTiles + tileIdx;
    if (tid == 0)
    {
        partialMax[partialBase] = sharedMax;
        partialSum[partialBase] = sharedSum;
    }
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        partialOut[partialBase * headDim + i] = outValues[i];
    }
}

template <typename T>
__global__ void turboQuant4AttentionFinalizeKernel(float const* partialMax, float const* partialSum,
    float const* partialOut, T* output, int numHeads, int headDim, int numTiles)
{
    __shared__ float globalMax;
    __shared__ float globalSum;

    int const qTokenIdx = blockIdx.x;
    int const qHeadIdx = blockIdx.y;
    int const tid = threadIdx.x;
    int64_t const partialBase = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * numTiles;

    if (tid == 0)
    {
        globalMax = -FLT_MAX;
        for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx)
        {
            globalMax = fmaxf(globalMax, partialMax[partialBase + tileIdx]);
        }
        globalSum = 0.0F;
        for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx)
        {
            float const tileSum = partialSum[partialBase + tileIdx];
            if (tileSum > 0.0F)
            {
                globalSum += tileSum * expf(partialMax[partialBase + tileIdx] - globalMax);
            }
        }
    }
    __syncthreads();

    int64_t const outputOffset = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * headDim;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        float accumulator = 0.0F;
        for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx)
        {
            float const tileSum = partialSum[partialBase + tileIdx];
            if (tileSum > 0.0F)
            {
                float const tileScale = expf(partialMax[partialBase + tileIdx] - globalMax);
                accumulator += tileScale * partialOut[(partialBase + tileIdx) * headDim + i];
            }
        }
        output[outputOffset + i] = fromFloat<T>(globalSum > 0.0F ? accumulator / globalSum : 0.0F);
    }
}

template <typename T>
__global__ void turboQuant4BatchAttentionKernel(T const* q, uint8_t const* cache, float const* scaleCache,
    int32_t const* blockIds, int32_t const* qBatchIndices, int32_t const* queryPositions, int32_t const* seqLens,
    T* output, int batchSize, int numHeads, int numKvHeads, int headDim, int kvFactor, int blockIdStride,
    float qScaling, bool isCausal, int attentionWindowSize, int tokensPerBlock)
{
    extern __shared__ float shared[];
    float* qValues = shared;
    float* values = qValues + headDim;
    float* outValues = values + headDim;
    float* scratch = outValues + headDim;
    __shared__ float sharedMax;
    __shared__ float sharedSum;
    __shared__ float sharedWeight;

    int const qTokenIdx = blockIdx.x;
    int const qHeadIdx = blockIdx.y;
    int const tid = threadIdx.x;
    int const sampleIdx = qBatchIndices[qTokenIdx];
    int64_t const outputOffset = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * headDim;
    if (sampleIdx < 0 || sampleIdx >= batchSize)
    {
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            output[outputOffset + i] = fromFloat<T>(0.0F);
        }
        return;
    }

    int const seqLen = seqLens[sampleIdx];
    int const kvGroupSize = numHeads / numKvHeads;
    int const kvHeadIdx = qHeadIdx / kvGroupSize;
    int const queryPosition = queryPositions[qTokenIdx];
    int32_t const* sampleBlockIds = blockIds + static_cast<int64_t>(sampleIdx) * blockIdStride;
    float const qkScale = 1.0F / (sqrtf(static_cast<float>(headDim)) * qScaling);

    int64_t const qOffset = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * headDim;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        qValues[i] = toFloat(q[qOffset + i]);
        outValues[i] = 0.0F;
    }
    if (tid == 0)
    {
        sharedMax = -FLT_MAX;
        sharedSum = 0.0F;
    }
    __syncthreads();

    for (int kvTokenIdx = 0; kvTokenIdx < seqLen; ++kvTokenIdx)
    {
        if (!turboQuant4AttentionTokenAllowed(kvTokenIdx, queryPosition, isCausal, attentionWindowSize))
        {
            continue;
        }
        loadTurboQuant4CacheVector(cache, scaleCache, sampleBlockIds, kvFactor, 0, kvTokenIdx, kvHeadIdx, numKvHeads,
            headDim, tokensPerBlock, values);
        float localDot = 0.0F;
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            localDot += qValues[i] * values[i];
        }
        float const dot = blockReduceSum(localDot, scratch) * qkScale;
        if (tid == 0)
        {
            sharedMax = fmaxf(sharedMax, dot);
        }
        __syncthreads();
    }

    for (int kvTokenIdx = 0; kvTokenIdx < seqLen; ++kvTokenIdx)
    {
        if (!turboQuant4AttentionTokenAllowed(kvTokenIdx, queryPosition, isCausal, attentionWindowSize))
        {
            continue;
        }
        loadTurboQuant4CacheVector(cache, scaleCache, sampleBlockIds, kvFactor, 0, kvTokenIdx, kvHeadIdx, numKvHeads,
            headDim, tokensPerBlock, values);
        float localDot = 0.0F;
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            localDot += qValues[i] * values[i];
        }
        float const dot = blockReduceSum(localDot, scratch) * qkScale;
        if (tid == 0)
        {
            sharedWeight = expf(dot - sharedMax);
            sharedSum += sharedWeight;
        }
        __syncthreads();
        loadTurboQuant4CacheVector(cache, scaleCache, sampleBlockIds, kvFactor, 1, kvTokenIdx, kvHeadIdx, numKvHeads,
            headDim, tokensPerBlock, values);
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            outValues[i] += sharedWeight * values[i];
        }
        __syncthreads();
    }

    for (int i = tid; i < headDim; i += blockDim.x)
    {
        output[outputOffset + i] = fromFloat<T>(sharedSum > 0.0F ? outValues[i] / sharedSum : 0.0F);
    }
}

template <typename T>
__global__ void turboQuant4BatchAttentionPartialKernel(T const* q, uint8_t const* cache, float const* scaleCache,
    int32_t const* blockIds, int32_t const* qBatchIndices, int32_t const* queryPositions, int32_t const* seqLens,
    float* partialMax, float* partialSum, float* partialOut, int batchSize, int numHeads, int numKvHeads, int headDim,
    int kvFactor, int blockIdStride, float qScaling, bool isCausal, int attentionWindowSize, int tokensPerBlock,
    int numTiles, int tileSize)
{
    extern __shared__ float shared[];
    float* qValues = shared;
    float* values = qValues + headDim;
    float* outValues = values + headDim;
    float* scratch = outValues + headDim;
    __shared__ float sharedMax;
    __shared__ float sharedSum;
    __shared__ float sharedWeight;

    int const qTokenIdx = blockIdx.x;
    int const qHeadIdx = blockIdx.y;
    int const tileIdx = blockIdx.z;
    int const tid = threadIdx.x;
    int const sampleIdx = qBatchIndices[qTokenIdx];
    int64_t const partialBase = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * numTiles + tileIdx;
    if (sampleIdx < 0 || sampleIdx >= batchSize)
    {
        if (tid == 0)
        {
            partialMax[partialBase] = -FLT_MAX;
            partialSum[partialBase] = 0.0F;
        }
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            partialOut[partialBase * headDim + i] = 0.0F;
        }
        return;
    }

    int const seqLen = seqLens[sampleIdx];
    int const kvGroupSize = numHeads / numKvHeads;
    int const kvHeadIdx = qHeadIdx / kvGroupSize;
    int const queryPosition = queryPositions[qTokenIdx];
    int32_t const* sampleBlockIds = blockIds + static_cast<int64_t>(sampleIdx) * blockIdStride;
    int const tileStart = tileIdx * tileSize;
    int const tileEndCandidate = tileStart + tileSize;
    int const tileEnd = tileEndCandidate < seqLen ? tileEndCandidate : seqLen;
    float const qkScale = 1.0F / (sqrtf(static_cast<float>(headDim)) * qScaling);

    int64_t const qOffset = (static_cast<int64_t>(qTokenIdx) * numHeads + qHeadIdx) * headDim;
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        qValues[i] = toFloat(q[qOffset + i]);
        outValues[i] = 0.0F;
    }
    if (tid == 0)
    {
        sharedMax = -FLT_MAX;
        sharedSum = 0.0F;
    }
    __syncthreads();

    for (int kvTokenIdx = tileStart; kvTokenIdx < tileEnd; ++kvTokenIdx)
    {
        if (!turboQuant4AttentionTokenAllowed(kvTokenIdx, queryPosition, isCausal, attentionWindowSize))
        {
            continue;
        }
        loadTurboQuant4CacheVector(cache, scaleCache, sampleBlockIds, kvFactor, 0, kvTokenIdx, kvHeadIdx, numKvHeads,
            headDim, tokensPerBlock, values);
        float localDot = 0.0F;
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            localDot += qValues[i] * values[i];
        }
        float const dot = blockReduceSum(localDot, scratch) * qkScale;
        if (tid == 0)
        {
            sharedMax = fmaxf(sharedMax, dot);
        }
        __syncthreads();
    }

    for (int kvTokenIdx = tileStart; kvTokenIdx < tileEnd; ++kvTokenIdx)
    {
        if (!turboQuant4AttentionTokenAllowed(kvTokenIdx, queryPosition, isCausal, attentionWindowSize))
        {
            continue;
        }
        loadTurboQuant4CacheVector(cache, scaleCache, sampleBlockIds, kvFactor, 0, kvTokenIdx, kvHeadIdx, numKvHeads,
            headDim, tokensPerBlock, values);
        float localDot = 0.0F;
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            localDot += qValues[i] * values[i];
        }
        float const dot = blockReduceSum(localDot, scratch) * qkScale;
        if (tid == 0)
        {
            sharedWeight = expf(dot - sharedMax);
            sharedSum += sharedWeight;
        }
        __syncthreads();
        loadTurboQuant4CacheVector(cache, scaleCache, sampleBlockIds, kvFactor, 1, kvTokenIdx, kvHeadIdx, numKvHeads,
            headDim, tokensPerBlock, values);
        for (int i = tid; i < headDim; i += blockDim.x)
        {
            outValues[i] += sharedWeight * values[i];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partialMax[partialBase] = sharedMax;
        partialSum[partialBase] = sharedSum;
    }
    for (int i = tid; i < headDim; i += blockDim.x)
    {
        partialOut[partialBase * headDim + i] = outValues[i];
    }
}

int getTurboQuant4ThreadCount(int headDim)
{
    int threads = 32;
    while (threads < headDim / 2)
    {
        threads *= 2;
    }
    return std::min(threads, 1024);
}

void checkTurboQuant4HeadDim(int64_t headDim)
{
    TORCH_CHECK(headDim > 0, "TurboQuant4 head_dim must be positive.");
    TORCH_CHECK((headDim & (headDim - 1)) == 0, "TurboQuant4 head_dim must be a power of 2, got ", headDim);
    TORCH_CHECK(headDim % 2 == 0, "TurboQuant4 head_dim must be even, got ", headDim);
    TORCH_CHECK(headDim <= 1024, "TurboQuant4 native kernels currently support head_dim <= 1024, got ", headDim);
}

void checkTurboQuant4VectorCount(int64_t numVectors)
{
    TORCH_CHECK(numVectors <= kTurboQuant4MaxGridX, "TurboQuant4 native kernels support at most ", kTurboQuant4MaxGridX,
        " vectors in one launch, got ", numVectors);
}

void checkTurboQuant4IntRange(int64_t value, char const* name)
{
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), "TurboQuant4 ", name, " must be in [0, ",
        std::numeric_limits<int>::max(), "], got ", value);
}

void checkTurboQuant4SameDevice(at::Tensor const& reference, at::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.device() == reference.device(), "TurboQuant4 ", name, " must be on the same device as ",
        reference.device(), ", got ", tensor.device());
}

void checkTurboQuant4CacheTensors(
    at::Tensor const& cache, at::Tensor const& scales, at::Tensor const& blockIds, int64_t kvIndex)
{
    CHECK_INPUT(cache, at::ScalarType::Byte);
    CHECK_INPUT(scales, at::ScalarType::Float);
    CHECK_INPUT(blockIds, at::ScalarType::Int);
    TORCH_CHECK(cache.dim() == 5, "TurboQuant4 cache must have shape [blocks, kv, tokens, heads, head_dim / 2].");
    TORCH_CHECK(scales.dim() == 5, "TurboQuant4 scale cache must have shape [blocks, kv, tokens, heads, 1].");
    TORCH_CHECK(blockIds.dim() == 1, "TurboQuant4 block ids must be a 1D int32 tensor.");
    TORCH_CHECK(scales.sizes()[0] == cache.sizes()[0], "TurboQuant4 cache/scales block count mismatch.");
    TORCH_CHECK(scales.sizes()[1] == cache.sizes()[1], "TurboQuant4 cache/scales KV factor mismatch.");
    TORCH_CHECK(scales.sizes()[2] == cache.sizes()[2], "TurboQuant4 cache/scales tokens_per_block mismatch.");
    TORCH_CHECK(scales.sizes()[3] == cache.sizes()[3], "TurboQuant4 cache/scales head count mismatch.");
    TORCH_CHECK(scales.sizes()[4] == 1, "TurboQuant4 scale cache last dimension must be 1.");
    TORCH_CHECK(kvIndex >= 0 && kvIndex < cache.sizes()[1], "TurboQuant4 kv_index is out of range.");
}

void checkTurboQuant4BatchCacheTensors(at::Tensor const& cache, at::Tensor const& scales, at::Tensor const& blockIds)
{
    CHECK_INPUT(cache, at::ScalarType::Byte);
    CHECK_INPUT(scales, at::ScalarType::Float);
    CHECK_INPUT(blockIds, at::ScalarType::Int);
    TORCH_CHECK(cache.dim() == 5, "TurboQuant4 cache must have shape [blocks, kv, tokens, heads, head_dim / 2].");
    TORCH_CHECK(scales.dim() == 5, "TurboQuant4 scale cache must have shape [blocks, kv, tokens, heads, 1].");
    TORCH_CHECK(blockIds.dim() == 2, "TurboQuant4 batched block ids must be a 2D int32 tensor.");
    TORCH_CHECK(scales.sizes()[0] == cache.sizes()[0], "TurboQuant4 cache/scales block count mismatch.");
    TORCH_CHECK(scales.sizes()[1] == cache.sizes()[1], "TurboQuant4 cache/scales KV factor mismatch.");
    TORCH_CHECK(scales.sizes()[2] == cache.sizes()[2], "TurboQuant4 cache/scales tokens_per_block mismatch.");
    TORCH_CHECK(scales.sizes()[3] == cache.sizes()[3], "TurboQuant4 cache/scales head count mismatch.");
    TORCH_CHECK(scales.sizes()[4] == 1, "TurboQuant4 scale cache last dimension must be 1.");
    TORCH_CHECK(cache.sizes()[1] >= 2, "TurboQuant4 attention requires key and value cache entries.");
}

void checkTurboQuant4BlockIds(at::Tensor const& blockIds, int64_t blockCount, int64_t maxBlocks)
{
    TORCH_CHECK(blockIds.numel() >= blockCount, "TurboQuant4 block id tensor is shorter than the requested sequence.");
    TORCH_CHECK(maxBlocks >= 0, "TurboQuant4 cache block count must be non-negative.");
    if (blockCount == 0)
    {
        return;
    }
    at::Tensor requiredBlockIds = blockIds.narrow(0, 0, blockCount);
    int64_t const minBlockId = requiredBlockIds.min().item<int32_t>();
    int64_t const maxBlockId = requiredBlockIds.max().item<int32_t>();
    TORCH_CHECK(minBlockId >= 0 && maxBlockId < maxBlocks, "TurboQuant4 block ids must be in [0, ", maxBlocks,
        "), got range [", minBlockId, ", ", maxBlockId, "].");
}

void checkTurboQuant4BatchBlockIds(at::Tensor const& blockIds, at::Tensor const& seqLens, int64_t maxSeqLen,
    int64_t tokensPerBlock, int64_t maxBlocks)
{
    TORCH_CHECK(maxBlocks >= 0, "TurboQuant4 cache block count must be non-negative.");
    if (seqLens.numel() == 0)
    {
        return;
    }

    int64_t const minSeqLen = seqLens.min().item<int32_t>();
    int64_t const actualMaxSeqLen = seqLens.max().item<int32_t>();
    TORCH_CHECK(minSeqLen >= 0, "TurboQuant4 batch attention seq_lens must be non-negative.");
    TORCH_CHECK(actualMaxSeqLen <= maxSeqLen, "TurboQuant4 batch attention seq_lens must not exceed max_seq_len.");

    int64_t const maxRequiredBlocks
        = actualMaxSeqLen == 0 ? 0 : (actualMaxSeqLen + tokensPerBlock - 1) / tokensPerBlock;
    TORCH_CHECK(blockIds.sizes()[1] >= maxRequiredBlocks,
        "TurboQuant4 batch attention block id tensor is shorter than the requested sequence.");
    if (maxRequiredBlocks == 0)
    {
        return;
    }

    at::Tensor requiredBlocksBySeq = at::floor_divide(seqLens.to(at::kLong) + (tokensPerBlock - 1), tokensPerBlock);
    at::Tensor candidateBlockIds = blockIds.narrow(1, 0, maxRequiredBlocks);
    at::Tensor columns = at::arange(maxRequiredBlocks, blockIds.options().dtype(at::kLong));
    at::Tensor validBlockMask = columns.unsqueeze(0).lt(requiredBlocksBySeq.unsqueeze(1));
    at::Tensor requiredBlockIds = at::masked_select(candidateBlockIds, validBlockMask);
    if (requiredBlockIds.numel() == 0)
    {
        return;
    }
    int64_t const minBlockId = requiredBlockIds.min().item<int32_t>();
    int64_t const maxBlockId = requiredBlockIds.max().item<int32_t>();
    TORCH_CHECK(minBlockId >= 0 && maxBlockId < maxBlocks, "TurboQuant4 batch attention block ids must be in [0, ",
        maxBlocks, "), got range [", minBlockId, ", ", maxBlockId, "].");
}

void checkTurboQuant4BatchQueryMetadata(
    at::Tensor const& qBatchIndices, at::Tensor const& queryPositions, at::Tensor const& seqLens)
{
    if (qBatchIndices.numel() == 0)
    {
        return;
    }

    int64_t const batchSize = seqLens.numel();
    int64_t const minBatchIndex = qBatchIndices.min().item<int32_t>();
    int64_t const maxBatchIndex = qBatchIndices.max().item<int32_t>();
    TORCH_CHECK(minBatchIndex >= 0 && maxBatchIndex < batchSize,
        "TurboQuant4 batch attention batch indices must be in [0, ", batchSize, "), got range [", minBatchIndex,
        ", ", maxBatchIndex, "].");

    int64_t const minQueryPosition = queryPositions.min().item<int32_t>();
    TORCH_CHECK(minQueryPosition >= 0,
        "TurboQuant4 batch attention query positions must be non-negative, got ", minQueryPosition, ".");

    at::Tensor seqLensForQueries = at::index_select(seqLens, 0, qBatchIndices.to(at::kLong));
    at::Tensor invalidQueryPositions = queryPositions >= seqLensForQueries;
    if (invalidQueryPositions.any().item<bool>())
    {
        at::Tensor invalidPositions = at::masked_select(queryPositions, invalidQueryPositions);
        at::Tensor invalidSeqLens = at::masked_select(seqLensForQueries, invalidQueryPositions);
        int64_t const maxInvalidPosition = invalidPositions.max().item<int32_t>();
        int64_t const minInvalidSeqLen = invalidSeqLens.min().item<int32_t>();
        TORCH_CHECK(false,
            "TurboQuant4 batch attention query positions must be within the KV sequence length; got position up to ",
            maxInvalidPosition, " for seq_len as low as ", minInvalidSeqLen, ".");
    }
}

} // namespace

std::tuple<at::Tensor, at::Tensor> turboquant4_quantize(at::Tensor const& input)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.dim() >= 1, "TurboQuant4 input must have at least one dimension.");

    auto const headDim = input.sizes().back();
    checkTurboQuant4HeadDim(headDim);
    auto const numVectors = input.numel() / headDim;
    checkTurboQuant4VectorCount(numVectors);
    at::cuda::CUDAGuard deviceGuard{static_cast<signed char>(input.get_device())};

    std::vector<int64_t> codeShape(input.sizes().begin(), input.sizes().end());
    codeShape.back() = headDim / 2;
    std::vector<int64_t> scaleShape(input.sizes().begin(), input.sizes().end());
    scaleShape.back() = 1;

    at::Tensor codes = at::detail::empty_cuda(codeShape, at::ScalarType::Byte, input.device(), std::nullopt);
    at::Tensor scales = at::detail::empty_cuda(scaleShape, at::ScalarType::Float, input.device(), std::nullopt);
    if (numVectors == 0)
    {
        return {codes, scales};
    }

    int const threads = getTurboQuant4ThreadCount(static_cast<int>(headDim));
    size_t const sharedBytes = static_cast<size_t>(headDim) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dim3 const grid(static_cast<unsigned int>(numVectors));

#define LAUNCH_TURBOQUANT4_QUANTIZE(T)                                                                                 \
    turboQuant4QuantizeKernel<<<grid, threads, sharedBytes, stream>>>(reinterpret_cast<T const*>(input.data_ptr()),    \
        reinterpret_cast<uint8_t*>(codes.data_ptr()), reinterpret_cast<float*>(scales.data_ptr()), numVectors,         \
        static_cast<int>(headDim));

    if (input.scalar_type() == at::ScalarType::Half)
    {
        LAUNCH_TURBOQUANT4_QUANTIZE(half)
    }
    else if (input.scalar_type() == at::ScalarType::Float)
    {
        LAUNCH_TURBOQUANT4_QUANTIZE(float)
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_TURBOQUANT4_QUANTIZE(__nv_bfloat16)
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to quantize a BF16 tensor with TurboQuant4.");
#endif
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "TurboQuant4 quantize supports FP16, BF16, and FP32 input tensors.");
    }

#undef LAUNCH_TURBOQUANT4_QUANTIZE

    sync_check_cuda_error(stream);
    return {codes, scales};
}

at::Tensor turboquant4_dequantize(
    at::Tensor const& codes, at::Tensor const& scales, std::optional<c10::ScalarType> outputDtype)
{
    CHECK_INPUT(codes, at::ScalarType::Byte);
    CHECK_INPUT(scales, at::ScalarType::Float);
    checkTurboQuant4SameDevice(codes, scales, "scales");
    TORCH_CHECK(codes.dim() >= 1, "TurboQuant4 codes must have at least one dimension.");
    TORCH_CHECK(codes.dim() == scales.dim(), "TurboQuant4 codes/scales rank mismatch.");
    TORCH_CHECK(scales.sizes().back() == 1, "TurboQuant4 scales last dimension must be 1.");

    auto const halfHeadDim = codes.sizes().back();
    auto const headDim = halfHeadDim * 2;
    checkTurboQuant4HeadDim(headDim);

    for (int64_t i = 0; i < codes.dim() - 1; ++i)
    {
        TORCH_CHECK(codes.sizes()[i] == scales.sizes()[i], "TurboQuant4 codes/scales shape mismatch.");
    }
    at::cuda::CUDAGuard deviceGuard{static_cast<signed char>(codes.get_device())};

    auto dtype = outputDtype.value_or(at::ScalarType::Half);
    std::vector<int64_t> outputShape(codes.sizes().begin(), codes.sizes().end());
    outputShape.back() = headDim;
    at::Tensor output = at::detail::empty_cuda(outputShape, dtype, codes.device(), std::nullopt);

    auto const numVectors = codes.numel() / halfHeadDim;
    checkTurboQuant4VectorCount(numVectors);
    if (numVectors == 0)
    {
        return output;
    }
    int const threads = getTurboQuant4ThreadCount(static_cast<int>(headDim));
    size_t const sharedBytes = static_cast<size_t>(headDim) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(codes.get_device());
    dim3 const grid(static_cast<unsigned int>(numVectors));

#define LAUNCH_TURBOQUANT4_DEQUANTIZE(T)                                                                               \
    turboQuant4DequantizeKernel<<<grid, threads, sharedBytes, stream>>>(                                               \
        reinterpret_cast<uint8_t const*>(codes.data_ptr()), reinterpret_cast<float const*>(scales.data_ptr()),         \
        reinterpret_cast<T*>(output.data_ptr()), numVectors, static_cast<int>(headDim));

    if (dtype == at::ScalarType::Half)
    {
        LAUNCH_TURBOQUANT4_DEQUANTIZE(half)
    }
    else if (dtype == at::ScalarType::Float)
    {
        LAUNCH_TURBOQUANT4_DEQUANTIZE(float)
    }
    else if (dtype == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_TURBOQUANT4_DEQUANTIZE(__nv_bfloat16)
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to dequantize to BF16 with TurboQuant4.");
#endif
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "TurboQuant4 dequantize supports FP16, BF16, and FP32 output tensors.");
    }

#undef LAUNCH_TURBOQUANT4_DEQUANTIZE

    sync_check_cuda_error(stream);
    return output;
}

void turboquant4_update_cache(at::Tensor const& input, at::Tensor const& cache, at::Tensor const& scales,
    at::Tensor const& blockIds, int64_t kvIndex, int64_t startPos, int64_t tokensPerBlock)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.dim() == 3, "TurboQuant4 cache update input must have shape [seq_len, heads, head_dim].");
    TORCH_CHECK(startPos >= 0, "TurboQuant4 cache update start_pos must be non-negative.");
    TORCH_CHECK(tokensPerBlock > 0, "TurboQuant4 tokens_per_block must be positive.");
    checkTurboQuant4CacheTensors(cache, scales, blockIds, kvIndex);
    checkTurboQuant4SameDevice(input, cache, "cache");
    checkTurboQuant4SameDevice(input, scales, "scales");
    checkTurboQuant4SameDevice(input, blockIds, "block ids");
    at::cuda::CUDAGuard deviceGuard{static_cast<signed char>(input.get_device())};

    auto const seqLen = input.sizes()[0];
    auto const numHeads = input.sizes()[1];
    auto const headDim = input.sizes()[2];
    checkTurboQuant4HeadDim(headDim);
    checkTurboQuant4IntRange(seqLen, "seq_len");
    checkTurboQuant4IntRange(numHeads, "head count");
    checkTurboQuant4IntRange(headDim, "head_dim");
    checkTurboQuant4IntRange(cache.sizes()[1], "KV factor");
    checkTurboQuant4IntRange(kvIndex, "kv_index");
    checkTurboQuant4IntRange(startPos, "start_pos");
    checkTurboQuant4IntRange(tokensPerBlock, "tokens_per_block");
    TORCH_CHECK(cache.sizes()[2] == tokensPerBlock, "TurboQuant4 cache tokens_per_block mismatch.");
    TORCH_CHECK(cache.sizes()[3] == numHeads, "TurboQuant4 cache head count mismatch.");
    TORCH_CHECK(cache.sizes()[4] == headDim / 2, "TurboQuant4 cache head_dim mismatch.");

    int64_t const blockCount = seqLen == 0 ? 0 : (startPos + seqLen + tokensPerBlock - 1) / tokensPerBlock;
    checkTurboQuant4BlockIds(blockIds, blockCount, cache.sizes()[0]);

    int64_t const numVectors = seqLen * numHeads;
    checkTurboQuant4VectorCount(numVectors);
    if (numVectors == 0)
    {
        return;
    }

    int const threads = getTurboQuant4ThreadCount(static_cast<int>(headDim));
    size_t const sharedBytes = static_cast<size_t>(headDim) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dim3 const grid(static_cast<unsigned int>(numVectors));

#define LAUNCH_TURBOQUANT4_UPDATE_CACHE(T)                                                                             \
    turboQuant4UpdateCacheKernel<<<grid, threads, sharedBytes, stream>>>(reinterpret_cast<T const*>(input.data_ptr()), \
        reinterpret_cast<uint8_t*>(cache.data_ptr()), reinterpret_cast<float*>(scales.data_ptr()),                     \
        reinterpret_cast<int32_t const*>(blockIds.data_ptr()), numVectors, static_cast<int>(numHeads),                 \
        static_cast<int>(headDim), static_cast<int>(cache.sizes()[1]), static_cast<int>(kvIndex),                      \
        static_cast<int>(startPos), static_cast<int>(tokensPerBlock));

    if (input.scalar_type() == at::ScalarType::Half)
    {
        LAUNCH_TURBOQUANT4_UPDATE_CACHE(half)
    }
    else if (input.scalar_type() == at::ScalarType::Float)
    {
        LAUNCH_TURBOQUANT4_UPDATE_CACHE(float)
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_TURBOQUANT4_UPDATE_CACHE(__nv_bfloat16)
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to update TurboQuant4 KV cache.");
#endif
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "TurboQuant4 cache update supports FP16, BF16, and FP32 input tensors.");
    }

#undef LAUNCH_TURBOQUANT4_UPDATE_CACHE

    sync_check_cuda_error(stream);
}

at::Tensor turboquant4_dequantize_cache(at::Tensor const& cache, at::Tensor const& scales, at::Tensor const& blockIds,
    int64_t kvIndex, int64_t seqLen, int64_t tokensPerBlock, std::optional<c10::ScalarType> outputDtype)
{
    TORCH_CHECK(seqLen >= 0, "TurboQuant4 cache dequantize seq_len must be non-negative.");
    TORCH_CHECK(tokensPerBlock > 0, "TurboQuant4 tokens_per_block must be positive.");
    checkTurboQuant4CacheTensors(cache, scales, blockIds, kvIndex);
    checkTurboQuant4SameDevice(cache, scales, "scales");
    checkTurboQuant4SameDevice(cache, blockIds, "block ids");
    at::cuda::CUDAGuard deviceGuard{static_cast<signed char>(cache.get_device())};

    auto const numHeads = cache.sizes()[3];
    auto const headDim = cache.sizes()[4] * 2;
    checkTurboQuant4HeadDim(headDim);
    TORCH_CHECK(cache.sizes()[2] == tokensPerBlock, "TurboQuant4 cache tokens_per_block mismatch.");
    checkTurboQuant4IntRange(seqLen, "seq_len");
    checkTurboQuant4IntRange(numHeads, "head count");
    checkTurboQuant4IntRange(headDim, "head_dim");
    checkTurboQuant4IntRange(cache.sizes()[1], "KV factor");
    checkTurboQuant4IntRange(kvIndex, "kv_index");
    checkTurboQuant4IntRange(tokensPerBlock, "tokens_per_block");

    int64_t const blockCount = (seqLen + tokensPerBlock - 1) / tokensPerBlock;
    checkTurboQuant4BlockIds(blockIds, blockCount, cache.sizes()[0]);

    auto dtype = outputDtype.value_or(at::ScalarType::Half);
    at::Tensor output = at::detail::empty_cuda({seqLen, numHeads, headDim}, dtype, cache.device(), std::nullopt);

    int64_t const numVectors = seqLen * numHeads;
    checkTurboQuant4VectorCount(numVectors);
    if (numVectors == 0)
    {
        return output;
    }

    int const threads = getTurboQuant4ThreadCount(static_cast<int>(headDim));
    size_t const sharedBytes = static_cast<size_t>(headDim) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(cache.get_device());
    dim3 const grid(static_cast<unsigned int>(numVectors));

#define LAUNCH_TURBOQUANT4_DEQUANTIZE_CACHE(T)                                                                         \
    turboQuant4DequantizeCacheKernel<<<grid, threads, sharedBytes, stream>>>(                                          \
        reinterpret_cast<uint8_t const*>(cache.data_ptr()), reinterpret_cast<float const*>(scales.data_ptr()),         \
        reinterpret_cast<int32_t const*>(blockIds.data_ptr()), reinterpret_cast<T*>(output.data_ptr()), numVectors,    \
        static_cast<int>(numHeads), static_cast<int>(headDim), static_cast<int>(cache.sizes()[1]),                     \
        static_cast<int>(kvIndex), static_cast<int>(tokensPerBlock));

    if (dtype == at::ScalarType::Half)
    {
        LAUNCH_TURBOQUANT4_DEQUANTIZE_CACHE(half)
    }
    else if (dtype == at::ScalarType::Float)
    {
        LAUNCH_TURBOQUANT4_DEQUANTIZE_CACHE(float)
    }
    else if (dtype == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_TURBOQUANT4_DEQUANTIZE_CACHE(__nv_bfloat16)
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to dequantize TurboQuant4 KV cache.");
#endif
    }
    else
    {
        C10_THROW_ERROR(
            NotImplementedError, "TurboQuant4 cache dequantize supports FP16, BF16, and FP32 output tensors.");
    }

#undef LAUNCH_TURBOQUANT4_DEQUANTIZE_CACHE

    sync_check_cuda_error(stream);
    return output;
}

at::Tensor turboquant4_attention(at::Tensor const& q, at::Tensor const& cache, at::Tensor const& scales,
    at::Tensor const& blockIds, int64_t seqLen, int64_t qStartPos, int64_t tokensPerBlock, double qScaling,
    bool isCausal, int64_t attentionWindowSize)
{
    CHECK_TH_CUDA(q);
    CHECK_CONTIGUOUS(q);
    TORCH_CHECK(q.dim() == 3, "TurboQuant4 attention q must have shape [q_len, heads, head_dim].");
    TORCH_CHECK(seqLen >= 0, "TurboQuant4 attention seq_len must be non-negative.");
    TORCH_CHECK(qStartPos >= 0, "TurboQuant4 attention q_start_pos must be non-negative.");
    TORCH_CHECK(tokensPerBlock > 0, "TurboQuant4 tokens_per_block must be positive.");
    TORCH_CHECK(qScaling > 0.0, "TurboQuant4 attention q_scaling must be positive.");
    TORCH_CHECK(attentionWindowSize >= 0, "TurboQuant4 attention_window_size must be non-negative.");
    checkTurboQuant4CacheTensors(cache, scales, blockIds, 0);
    checkTurboQuant4SameDevice(q, cache, "cache");
    checkTurboQuant4SameDevice(q, scales, "scales");
    checkTurboQuant4SameDevice(q, blockIds, "block ids");
    at::cuda::CUDAGuard deviceGuard{static_cast<signed char>(q.get_device())};
    TORCH_CHECK(cache.sizes()[1] >= 2, "TurboQuant4 attention requires key and value cache entries.");

    auto const qLen = q.sizes()[0];
    auto const numHeads = q.sizes()[1];
    auto const numKvHeads = cache.sizes()[3];
    auto const headDim = q.sizes()[2];
    checkTurboQuant4HeadDim(headDim);
    checkTurboQuant4IntRange(qLen, "q_len");
    checkTurboQuant4IntRange(seqLen, "seq_len");
    checkTurboQuant4IntRange(numHeads, "head count");
    checkTurboQuant4IntRange(numKvHeads, "KV head count");
    checkTurboQuant4IntRange(headDim, "head_dim");
    checkTurboQuant4IntRange(cache.sizes()[1], "KV factor");
    checkTurboQuant4IntRange(qStartPos, "q_start_pos");
    checkTurboQuant4IntRange(tokensPerBlock, "tokens_per_block");
    checkTurboQuant4IntRange(attentionWindowSize, "attention_window_size");
    TORCH_CHECK(cache.sizes()[2] == tokensPerBlock, "TurboQuant4 cache tokens_per_block mismatch.");
    TORCH_CHECK(cache.sizes()[4] == headDim / 2, "TurboQuant4 cache head_dim mismatch.");
    TORCH_CHECK(numHeads > 0, "TurboQuant4 attention requires at least one query head.");
    TORCH_CHECK(numKvHeads > 0, "TurboQuant4 attention requires at least one KV head.");
    TORCH_CHECK(numHeads % numKvHeads == 0, "TurboQuant4 attention heads must be divisible by KV heads.");
    TORCH_CHECK(qLen == 0 || qStartPos + qLen <= seqLen,
        "TurboQuant4 attention query positions must be within the KV sequence length.");

    int64_t const blockCount = seqLen == 0 ? 0 : (seqLen + tokensPerBlock - 1) / tokensPerBlock;
    checkTurboQuant4BlockIds(blockIds, blockCount, cache.sizes()[0]);

    at::Tensor output = at::empty_like(q);
    if (qLen == 0)
    {
        return output;
    }

    int const threads = getTurboQuant4ThreadCount(static_cast<int>(headDim));
    size_t const sharedBytes = static_cast<size_t>(3 * headDim + threads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    bool const useTiledAttention = seqLen > kTurboQuant4AttentionTileSize && qLen <= kTurboQuant4MaxTiledQueryTokens;
    if (useTiledAttention)
    {
        int64_t const numTiles64 = (seqLen + kTurboQuant4AttentionTileSize - 1) / kTurboQuant4AttentionTileSize;
        checkTurboQuant4IntRange(numTiles64, "tile count");
        int const numTiles = static_cast<int>(numTiles64);
        at::Tensor partialMax
            = at::detail::empty_cuda({qLen, numHeads, numTiles}, at::ScalarType::Float, q.device(), std::nullopt);
        at::Tensor partialSum
            = at::detail::empty_cuda({qLen, numHeads, numTiles}, at::ScalarType::Float, q.device(), std::nullopt);
        at::Tensor partialOut = at::detail::empty_cuda(
            {qLen, numHeads, numTiles, headDim}, at::ScalarType::Float, q.device(), std::nullopt);
        dim3 const partialGrid(
            static_cast<unsigned int>(qLen), static_cast<unsigned int>(numHeads), static_cast<unsigned int>(numTiles));
        dim3 const finalizeGrid(static_cast<unsigned int>(qLen), static_cast<unsigned int>(numHeads));

#define LAUNCH_TURBOQUANT4_TILED_ATTENTION(T)                                                                          \
    turboQuant4AttentionPartialKernel<<<partialGrid, threads, sharedBytes, stream>>>(                                  \
        reinterpret_cast<T const*>(q.data_ptr()), reinterpret_cast<uint8_t const*>(cache.data_ptr()),                  \
        reinterpret_cast<float const*>(scales.data_ptr()), reinterpret_cast<int32_t const*>(blockIds.data_ptr()),      \
        reinterpret_cast<float*>(partialMax.data_ptr()), reinterpret_cast<float*>(partialSum.data_ptr()),              \
        reinterpret_cast<float*>(partialOut.data_ptr()), static_cast<int>(seqLen), static_cast<int>(numHeads),         \
        static_cast<int>(numKvHeads), static_cast<int>(headDim), static_cast<int>(cache.sizes()[1]),                   \
        static_cast<int>(qStartPos), static_cast<float>(qScaling), isCausal, static_cast<int>(attentionWindowSize),    \
        static_cast<int>(tokensPerBlock), numTiles, kTurboQuant4AttentionTileSize);                                    \
    turboQuant4AttentionFinalizeKernel<<<finalizeGrid, threads, 0, stream>>>(                                          \
        reinterpret_cast<float const*>(partialMax.data_ptr()), reinterpret_cast<float const*>(partialSum.data_ptr()),  \
        reinterpret_cast<float const*>(partialOut.data_ptr()), reinterpret_cast<T*>(output.data_ptr()),                \
        static_cast<int>(numHeads), static_cast<int>(headDim), numTiles);

        if (q.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_TURBOQUANT4_TILED_ATTENTION(half)
        }
        else if (q.scalar_type() == at::ScalarType::Float)
        {
            LAUNCH_TURBOQUANT4_TILED_ATTENTION(float)
        }
        else if (q.scalar_type() == at::ScalarType::BFloat16)
        {
#ifdef ENABLE_BF16
            LAUNCH_TURBOQUANT4_TILED_ATTENTION(__nv_bfloat16)
#else
            C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled for TurboQuant4 attention.");
#endif
        }
        else
        {
            C10_THROW_ERROR(NotImplementedError, "TurboQuant4 attention supports FP16, BF16, and FP32 query tensors.");
        }

#undef LAUNCH_TURBOQUANT4_TILED_ATTENTION
    }
    else
    {
        dim3 const grid(static_cast<unsigned int>(qLen), static_cast<unsigned int>(numHeads));

#define LAUNCH_TURBOQUANT4_ATTENTION(T)                                                                                \
    turboQuant4AttentionKernel<<<grid, threads, sharedBytes, stream>>>(reinterpret_cast<T const*>(q.data_ptr()),       \
        reinterpret_cast<uint8_t const*>(cache.data_ptr()), reinterpret_cast<float const*>(scales.data_ptr()),         \
        reinterpret_cast<int32_t const*>(blockIds.data_ptr()), reinterpret_cast<T*>(output.data_ptr()),                \
        static_cast<int>(seqLen), static_cast<int>(numHeads), static_cast<int>(numKvHeads), static_cast<int>(headDim), \
        static_cast<int>(cache.sizes()[1]), static_cast<int>(qStartPos), static_cast<float>(qScaling), isCausal,       \
        static_cast<int>(attentionWindowSize), static_cast<int>(tokensPerBlock));

        if (q.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_TURBOQUANT4_ATTENTION(half)
        }
        else if (q.scalar_type() == at::ScalarType::Float)
        {
            LAUNCH_TURBOQUANT4_ATTENTION(float)
        }
        else if (q.scalar_type() == at::ScalarType::BFloat16)
        {
#ifdef ENABLE_BF16
            LAUNCH_TURBOQUANT4_ATTENTION(__nv_bfloat16)
#else
            C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled for TurboQuant4 attention.");
#endif
        }
        else
        {
            C10_THROW_ERROR(NotImplementedError, "TurboQuant4 attention supports FP16, BF16, and FP32 query tensors.");
        }

#undef LAUNCH_TURBOQUANT4_ATTENTION
    }

    sync_check_cuda_error(stream);
    return output;
}

at::Tensor turboquant4_batch_attention(at::Tensor const& q, at::Tensor const& cache, at::Tensor const& scales,
    at::Tensor const& blockIds, at::Tensor const& qBatchIndices, at::Tensor const& queryPositions,
    at::Tensor const& seqLens, int64_t maxSeqLen, int64_t tokensPerBlock, double qScaling, bool isCausal,
    int64_t attentionWindowSize)
{
    CHECK_TH_CUDA(q);
    CHECK_CONTIGUOUS(q);
    CHECK_INPUT(qBatchIndices, at::ScalarType::Int);
    CHECK_INPUT(queryPositions, at::ScalarType::Int);
    CHECK_INPUT(seqLens, at::ScalarType::Int);
    TORCH_CHECK(q.dim() == 3, "TurboQuant4 batch attention q must have shape [q_len, heads, head_dim].");
    TORCH_CHECK(qBatchIndices.dim() == 1, "TurboQuant4 batch attention q_batch_indices must be a 1D tensor.");
    TORCH_CHECK(queryPositions.dim() == 1, "TurboQuant4 batch attention query_positions must be a 1D tensor.");
    TORCH_CHECK(seqLens.dim() == 1, "TurboQuant4 batch attention seq_lens must be a 1D tensor.");
    TORCH_CHECK(maxSeqLen >= 0, "TurboQuant4 batch attention max_seq_len must be non-negative.");
    TORCH_CHECK(tokensPerBlock > 0, "TurboQuant4 tokens_per_block must be positive.");
    TORCH_CHECK(qScaling > 0.0, "TurboQuant4 attention q_scaling must be positive.");
    TORCH_CHECK(attentionWindowSize >= 0, "TurboQuant4 attention_window_size must be non-negative.");
    checkTurboQuant4BatchCacheTensors(cache, scales, blockIds);
    checkTurboQuant4SameDevice(q, cache, "cache");
    checkTurboQuant4SameDevice(q, scales, "scales");
    checkTurboQuant4SameDevice(q, blockIds, "block ids");
    checkTurboQuant4SameDevice(q, qBatchIndices, "q batch indices");
    checkTurboQuant4SameDevice(q, queryPositions, "query positions");
    checkTurboQuant4SameDevice(q, seqLens, "sequence lengths");
    at::cuda::CUDAGuard deviceGuard{static_cast<signed char>(q.get_device())};

    auto const qLen = q.sizes()[0];
    auto const numHeads = q.sizes()[1];
    auto const numKvHeads = cache.sizes()[3];
    auto const headDim = q.sizes()[2];
    checkTurboQuant4HeadDim(headDim);
    checkTurboQuant4IntRange(qLen, "q_len");
    checkTurboQuant4IntRange(seqLens.numel(), "batch size");
    checkTurboQuant4IntRange(maxSeqLen, "max_seq_len");
    checkTurboQuant4IntRange(numHeads, "head count");
    checkTurboQuant4IntRange(numKvHeads, "KV head count");
    checkTurboQuant4IntRange(headDim, "head_dim");
    checkTurboQuant4IntRange(cache.sizes()[1], "KV factor");
    checkTurboQuant4IntRange(blockIds.sizes()[1], "block id stride");
    checkTurboQuant4IntRange(tokensPerBlock, "tokens_per_block");
    checkTurboQuant4IntRange(attentionWindowSize, "attention_window_size");
    TORCH_CHECK(qBatchIndices.numel() == qLen, "TurboQuant4 batch attention q_batch_indices length mismatch.");
    TORCH_CHECK(queryPositions.numel() == qLen, "TurboQuant4 batch attention query_positions length mismatch.");
    TORCH_CHECK(blockIds.sizes()[0] == seqLens.numel(), "TurboQuant4 batch attention batch size mismatch.");
    TORCH_CHECK(cache.sizes()[2] == tokensPerBlock, "TurboQuant4 cache tokens_per_block mismatch.");
    TORCH_CHECK(cache.sizes()[4] == headDim / 2, "TurboQuant4 cache head_dim mismatch.");
    TORCH_CHECK(numHeads > 0, "TurboQuant4 attention requires at least one query head.");
    TORCH_CHECK(numKvHeads > 0, "TurboQuant4 attention requires at least one KV head.");
    TORCH_CHECK(numHeads % numKvHeads == 0, "TurboQuant4 attention heads must be divisible by KV heads.");

    checkTurboQuant4BatchBlockIds(blockIds, seqLens, maxSeqLen, tokensPerBlock, cache.sizes()[0]);
    checkTurboQuant4BatchQueryMetadata(qBatchIndices, queryPositions, seqLens);

    at::Tensor output = at::empty_like(q);
    if (qLen == 0)
    {
        return output;
    }

    int const threads = getTurboQuant4ThreadCount(static_cast<int>(headDim));
    size_t const sharedBytes = static_cast<size_t>(3 * headDim + threads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    bool const useTiledAttention = maxSeqLen > kTurboQuant4AttentionTileSize && qLen <= kTurboQuant4MaxTiledQueryTokens;
    if (useTiledAttention)
    {
        int64_t const numTiles64 = (maxSeqLen + kTurboQuant4AttentionTileSize - 1) / kTurboQuant4AttentionTileSize;
        checkTurboQuant4IntRange(numTiles64, "tile count");
        int const numTiles = static_cast<int>(numTiles64);
        at::Tensor partialMax
            = at::detail::empty_cuda({qLen, numHeads, numTiles}, at::ScalarType::Float, q.device(), std::nullopt);
        at::Tensor partialSum
            = at::detail::empty_cuda({qLen, numHeads, numTiles}, at::ScalarType::Float, q.device(), std::nullopt);
        at::Tensor partialOut = at::detail::empty_cuda(
            {qLen, numHeads, numTiles, headDim}, at::ScalarType::Float, q.device(), std::nullopt);
        dim3 const partialGrid(
            static_cast<unsigned int>(qLen), static_cast<unsigned int>(numHeads), static_cast<unsigned int>(numTiles));
        dim3 const finalizeGrid(static_cast<unsigned int>(qLen), static_cast<unsigned int>(numHeads));

#define LAUNCH_TURBOQUANT4_BATCH_TILED_ATTENTION(T)                                                                    \
    turboQuant4BatchAttentionPartialKernel<<<partialGrid, threads, sharedBytes, stream>>>(                             \
        reinterpret_cast<T const*>(q.data_ptr()), reinterpret_cast<uint8_t const*>(cache.data_ptr()),                  \
        reinterpret_cast<float const*>(scales.data_ptr()), reinterpret_cast<int32_t const*>(blockIds.data_ptr()),      \
        reinterpret_cast<int32_t const*>(qBatchIndices.data_ptr()),                                                    \
        reinterpret_cast<int32_t const*>(queryPositions.data_ptr()),                                                   \
        reinterpret_cast<int32_t const*>(seqLens.data_ptr()), reinterpret_cast<float*>(partialMax.data_ptr()),         \
        reinterpret_cast<float*>(partialSum.data_ptr()), reinterpret_cast<float*>(partialOut.data_ptr()),              \
        static_cast<int>(seqLens.numel()), static_cast<int>(numHeads), static_cast<int>(numKvHeads),                   \
        static_cast<int>(headDim), static_cast<int>(cache.sizes()[1]), static_cast<int>(blockIds.sizes()[1]),          \
        static_cast<float>(qScaling), isCausal, static_cast<int>(attentionWindowSize),                                 \
        static_cast<int>(tokensPerBlock), numTiles, kTurboQuant4AttentionTileSize);                                    \
    turboQuant4AttentionFinalizeKernel<<<finalizeGrid, threads, 0, stream>>>(                                          \
        reinterpret_cast<float const*>(partialMax.data_ptr()), reinterpret_cast<float const*>(partialSum.data_ptr()),  \
        reinterpret_cast<float const*>(partialOut.data_ptr()), reinterpret_cast<T*>(output.data_ptr()),                \
        static_cast<int>(numHeads), static_cast<int>(headDim), numTiles);

        if (q.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_TURBOQUANT4_BATCH_TILED_ATTENTION(half)
        }
        else if (q.scalar_type() == at::ScalarType::Float)
        {
            LAUNCH_TURBOQUANT4_BATCH_TILED_ATTENTION(float)
        }
        else if (q.scalar_type() == at::ScalarType::BFloat16)
        {
#ifdef ENABLE_BF16
            LAUNCH_TURBOQUANT4_BATCH_TILED_ATTENTION(__nv_bfloat16)
#else
            C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled for TurboQuant4 batch attention.");
#endif
        }
        else
        {
            C10_THROW_ERROR(
                NotImplementedError, "TurboQuant4 batch attention supports FP16, BF16, and FP32 query tensors.");
        }

#undef LAUNCH_TURBOQUANT4_BATCH_TILED_ATTENTION
    }
    else
    {
        dim3 const grid(static_cast<unsigned int>(qLen), static_cast<unsigned int>(numHeads));

#define LAUNCH_TURBOQUANT4_BATCH_ATTENTION(T)                                                                          \
    turboQuant4BatchAttentionKernel<<<grid, threads, sharedBytes, stream>>>(reinterpret_cast<T const*>(q.data_ptr()),  \
        reinterpret_cast<uint8_t const*>(cache.data_ptr()), reinterpret_cast<float const*>(scales.data_ptr()),         \
        reinterpret_cast<int32_t const*>(blockIds.data_ptr()),                                                         \
        reinterpret_cast<int32_t const*>(qBatchIndices.data_ptr()),                                                    \
        reinterpret_cast<int32_t const*>(queryPositions.data_ptr()),                                                   \
        reinterpret_cast<int32_t const*>(seqLens.data_ptr()), reinterpret_cast<T*>(output.data_ptr()),                 \
        static_cast<int>(seqLens.numel()), static_cast<int>(numHeads), static_cast<int>(numKvHeads),                   \
        static_cast<int>(headDim), static_cast<int>(cache.sizes()[1]), static_cast<int>(blockIds.sizes()[1]),          \
        static_cast<float>(qScaling), isCausal, static_cast<int>(attentionWindowSize),                                 \
        static_cast<int>(tokensPerBlock));

        if (q.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_TURBOQUANT4_BATCH_ATTENTION(half)
        }
        else if (q.scalar_type() == at::ScalarType::Float)
        {
            LAUNCH_TURBOQUANT4_BATCH_ATTENTION(float)
        }
        else if (q.scalar_type() == at::ScalarType::BFloat16)
        {
#ifdef ENABLE_BF16
            LAUNCH_TURBOQUANT4_BATCH_ATTENTION(__nv_bfloat16)
#else
            C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled for TurboQuant4 batch attention.");
#endif
        }
        else
        {
            C10_THROW_ERROR(
                NotImplementedError, "TurboQuant4 batch attention supports FP16, BF16, and FP32 query tensors.");
        }

#undef LAUNCH_TURBOQUANT4_BATCH_ATTENTION
    }

    sync_check_cuda_error(stream);
    return output;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("turboquant4_quantize(Tensor input) -> (Tensor, Tensor)");
    m.def("turboquant4_dequantize(Tensor codes, Tensor scales, ScalarType? output_dtype=None) -> Tensor");
    m.def(
        "turboquant4_update_cache(Tensor input, Tensor(a!) cache, Tensor(b!) scales, Tensor block_ids, int kv_index, "
        "int start_pos, int tokens_per_block) -> ()");
    m.def(
        "turboquant4_dequantize_cache(Tensor cache, Tensor scales, Tensor block_ids, int kv_index, int seq_len, "
        "int tokens_per_block, ScalarType? output_dtype=None) -> Tensor");
    m.def(
        "turboquant4_attention(Tensor q, Tensor cache, Tensor scales, Tensor block_ids, int seq_len, int q_start_pos, "
        "int tokens_per_block, float q_scaling, bool is_causal, int attention_window_size) -> Tensor");
    m.def(
        "turboquant4_batch_attention(Tensor q, Tensor cache, Tensor scales, Tensor block_ids, "
        "Tensor q_batch_indices, Tensor query_positions, Tensor seq_lens, int max_seq_len, int tokens_per_block, "
        "float q_scaling, bool is_causal, int attention_window_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("turboquant4_quantize", &tensorrt_llm::torch_ext::turboquant4_quantize);
    m.impl("turboquant4_dequantize", &tensorrt_llm::torch_ext::turboquant4_dequantize);
    m.impl("turboquant4_update_cache", &tensorrt_llm::torch_ext::turboquant4_update_cache);
    m.impl("turboquant4_dequantize_cache", &tensorrt_llm::torch_ext::turboquant4_dequantize_cache);
    m.impl("turboquant4_attention", &tensorrt_llm::torch_ext::turboquant4_attention);
    m.impl("turboquant4_batch_attention", &tensorrt_llm::torch_ext::turboquant4_batch_attention);
}
