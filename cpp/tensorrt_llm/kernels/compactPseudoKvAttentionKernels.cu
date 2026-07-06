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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int32_t kCompactPseudoKvThreadsPerBlock = 128;
constexpr float kCompactPseudoKvMaskedScore = -3.4028234663852886e38F;

__device__ float compactPseudoKvReduceMax(float value)
{
    __shared__ float shared[kCompactPseudoKvThreadsPerBlock];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float compactPseudoKvReduceSum(float value)
{
    __shared__ float shared[kCompactPseudoKvThreadsPerBlock];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float compactPseudoKvScore(CompactPseudoKvAttentionLaunchParams const& params, char const* queryBase,
    float const* key, int32_t const compactIdx, int32_t const headIdx)
{
    auto const keyBase = reinterpret_cast<char const*>(key)
        + compactIdx * params.compact_pseudokv_params.key_stride_token_in_bytes
        + headIdx * params.compact_pseudokv_params.key_stride_head_in_bytes;
    float score = 0.0F;
    for (int32_t dim = 0; dim < params.compact_pseudokv_params.head_size; ++dim)
    {
        score += reinterpret_cast<float const*>(queryBase)[dim] * reinterpret_cast<float const*>(keyBase)[dim];
    }
    return score;
}

__global__ void compactPseudoKvAttentionFloatKernel(CompactPseudoKvAttentionLaunchParams params)
{
    extern __shared__ float scoreCache[];
    int32_t const queryIdx = blockIdx.x;
    int32_t const headIdx = blockIdx.y;
    if (queryIdx >= params.query_token_count || headIdx >= params.compact_pseudokv_params.num_heads)
    {
        return;
    }

    auto const* query = static_cast<float const*>(params.query);
    auto* output = static_cast<float*>(params.output);
    auto const* key = static_cast<float const*>(params.compact_pseudokv_params.key);
    auto const* value = static_cast<float const*>(params.compact_pseudokv_params.value);
    auto const* mask = params.compact_pseudokv_params.causal_mask;
    int32_t const compactTokens = params.compact_pseudokv_params.compact_token_count;
    int32_t const headSize = params.compact_pseudokv_params.head_size;
    float const scale = rsqrtf(static_cast<float>(headSize));
    auto const queryBase = reinterpret_cast<char const*>(query) + queryIdx * params.query_stride_token_in_bytes
        + headIdx * params.query_stride_head_in_bytes;

    float localMax = kCompactPseudoKvMaskedScore;
    for (int32_t compactIdx = threadIdx.x; compactIdx < compactTokens; compactIdx += blockDim.x)
    {
        if (mask != nullptr && mask[queryIdx * compactTokens + compactIdx])
        {
            scoreCache[compactIdx] = kCompactPseudoKvMaskedScore;
            continue;
        }
        scoreCache[compactIdx] = compactPseudoKvScore(params, queryBase, key, compactIdx, headIdx) * scale;
        localMax = fmaxf(localMax, scoreCache[compactIdx]);
    }
    float const maxScore = compactPseudoKvReduceMax(localMax);

    auto const outputBase = reinterpret_cast<char*>(output) + queryIdx * params.output_stride_token_in_bytes
        + headIdx * params.output_stride_head_in_bytes;
    if (!isfinite(maxScore))
    {
        for (int32_t dim = threadIdx.x; dim < headSize; dim += blockDim.x)
        {
            reinterpret_cast<float*>(outputBase)[dim] = 0.0F;
        }
        return;
    }

    float localDenom = 0.0F;
    for (int32_t compactIdx = threadIdx.x; compactIdx < compactTokens; compactIdx += blockDim.x)
    {
        if (mask != nullptr && mask[queryIdx * compactTokens + compactIdx])
        {
            scoreCache[compactIdx] = 0.0F;
            continue;
        }
        scoreCache[compactIdx] = expf(scoreCache[compactIdx] - maxScore);
        localDenom += scoreCache[compactIdx];
    }
    float const denom = compactPseudoKvReduceSum(localDenom);

    for (int32_t dim = threadIdx.x; dim < headSize; dim += blockDim.x)
    {
        float weightedValue = 0.0F;
        for (int32_t compactIdx = 0; compactIdx < compactTokens; ++compactIdx)
        {
            if (mask != nullptr && mask[queryIdx * compactTokens + compactIdx])
            {
                continue;
            }
            auto const valueBase = reinterpret_cast<char const*>(value)
                + compactIdx * params.compact_pseudokv_params.value_stride_token_in_bytes
                + headIdx * params.compact_pseudokv_params.value_stride_head_in_bytes;
            weightedValue += scoreCache[compactIdx] * reinterpret_cast<float const*>(valueBase)[dim];
        }
        reinterpret_cast<float*>(outputBase)[dim] = weightedValue / denom;
    }
}

size_t compactPseudoKvScoreCacheBytes(CompactPseudoKvAttentionLaunchParams const& params)
{
    return sizeof(float) * params.compact_pseudokv_params.compact_token_count;
}

int32_t compactPseudoKvMaxDynamicSharedMemoryBytes()
{
    int32_t device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int32_t maxSharedMemory = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (maxSharedMemory == 0)
    {
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, device));
    }
    return maxSharedMemory;
}

} // namespace

void invokeCompactPseudoKvAttention(CompactPseudoKvAttentionLaunchParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.isEnabled(), "Compact pseudo-KV attention launch params are incomplete.");
    TLLM_CHECK_WITH_INFO(
        params.data_type == nvinfer1::DataType::kFLOAT, "Compact pseudo-KV native kernel currently supports FP32.");
    size_t const scoreCacheBytes = compactPseudoKvScoreCacheBytes(params);
    int32_t const maxSharedMemory = compactPseudoKvMaxDynamicSharedMemoryBytes();
    TLLM_CHECK_WITH_INFO(scoreCacheBytes <= static_cast<size_t>(maxSharedMemory),
        "Compact pseudo-KV score cache exceeds per-block dynamic shared-memory capacity.");
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(compactPseudoKvAttentionFloatKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int32_t>(scoreCacheBytes)));
    dim3 const grid(params.query_token_count, params.compact_pseudokv_params.num_heads, 1);
    dim3 const block(kCompactPseudoKvThreadsPerBlock, 1, 1);
    compactPseudoKvAttentionFloatKernel<<<grid, block, scoreCacheBytes, stream>>>(params);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
