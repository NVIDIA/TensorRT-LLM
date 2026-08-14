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

#include "nvfp4MlaKvCacheGather.h"

#include "tensorrt_llm/common/assert.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

__device__ __forceinline__ float e2m1ToFloat(uint8_t value)
{
    constexpr float kMagnitude[8] = {0.F, 0.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
    float const magnitude = kMagnitude[value & 0x7U];
    return (value & 0x8U) == 0 ? magnitude : -magnitude;
}

__global__ void nvFp4MlaKvCacheGatherKernel(uint8_t const* __restrict__ dataPool,
    __nv_fp8_e4m3 const* __restrict__ scalePool, int32_t const* __restrict__ globalIndices,
    __nv_fp8_e4m3* __restrict__ output, int32_t* __restrict__ compactIndices,
    float const* __restrict__ globalDequantScale, int32_t topK, int32_t headDim, int64_t numPoolTokens)
{
    int32_t const row = blockIdx.x;
    int32_t const topKIdx = blockIdx.y;
    int32_t const globalIdx = globalIndices[static_cast<int64_t>(row) * topK + topKIdx];
    int64_t const compactIdx = static_cast<int64_t>(row) * topK + topKIdx;
    bool const valid = globalIdx >= 0 && static_cast<int64_t>(globalIdx) < numPoolTokens;

    if (threadIdx.x == 0)
    {
        compactIndices[compactIdx] = valid ? static_cast<int32_t>(compactIdx) : -1;
    }
    if (!valid)
    {
        return;
    }

    int32_t const packedHeadDim = headDim / 2;
    int32_t const scalesPerToken = headDim / 16;
    float const dequantScale = globalDequantScale == nullptr ? 1.F : globalDequantScale[0];
    for (int32_t dim = threadIdx.x; dim < headDim; dim += blockDim.x)
    {
        uint8_t const packed = dataPool[static_cast<int64_t>(globalIdx) * packedHeadDim + dim / 2];
        uint8_t const fp4 = (dim & 1) == 0 ? packed & 0xFU : packed >> 4;
        float const blockScale
            = static_cast<float>(scalePool[static_cast<int64_t>(globalIdx) * scalesPerToken + dim / 16]);
        output[compactIdx * headDim + dim] = __nv_fp8_e4m3(e2m1ToFloat(fp4) * blockScale * dequantScale);
    }
}

} // namespace

void invokeNvFp4MlaKvCacheGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool, int32_t const* globalIndices,
    __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale, int32_t numRows, int32_t topK,
    int32_t headDim, int64_t numPoolTokens, cudaStream_t stream)
{
    if (numRows == 0 || topK == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(headDim > 0 && headDim % 16 == 0,
        "NVFP4 MLA gather requires head_dim to be a positive multiple of 16, got %d", headDim);

    constexpr int32_t kThreads = 256;
    nvFp4MlaKvCacheGatherKernel<<<dim3(numRows, topK), kThreads, 0, stream>>>(
        dataPool, scalePool, globalIndices, output, compactIndices, globalDequantScale, topK, headDim, numPoolTokens);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
