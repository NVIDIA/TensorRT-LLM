/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/logitsBitmask.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace kernels
{
namespace
{

int32_t constexpr kBitsPerMaskElement = 32;
int32_t constexpr kThreadsPerBlock = 256;

template <typename T>
__device__ T negativeInfinity()
{
    return -INFINITY;
}

template <>
__device__ half negativeInfinity<half>()
{
    return -CUDART_INF_FP16;
}

template <>
__device__ __nv_bfloat16 negativeInfinity<__nv_bfloat16>()
{
    return -CUDART_INF_BF16;
}

template <typename T, typename PackedT>
__global__ void __launch_bounds__(kThreadsPerBlock) logitsBitmaskKernel(
    T** __restrict__ logits, uint32_t const** __restrict__ bitmask, int32_t vocabSizePadded, int32_t bitmaskSize)
{
    int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
    int const batchIdx = blockIdx.y;

    int const logitsGmemOffset = kThreadsPerBlock * blockIdx.x * kBitsPerMaskElement;
    T* logitsGmemPtr = logits[batchIdx] + logitsGmemOffset;
    __shared__ T logitsSmem[kThreadsPerBlock * kBitsPerMaskElement];

#pragma unroll
    for (int offset = 0; offset < kThreadsPerBlock * kBitsPerMaskElement; offset += kThreadsPerBlock * kAlignment)
    {
        int localOffset = offset + threadIdx.x * kAlignment;
        if (logitsGmemOffset + localOffset >= vocabSizePadded)
        {
            break;
        }
        *reinterpret_cast<PackedT*>(logitsSmem + localOffset)
            = *reinterpret_cast<PackedT*>(logitsGmemPtr + localOffset);
    }
    __syncthreads();

    int const bitmaskIdx = kThreadsPerBlock * blockIdx.x + threadIdx.x;
    uint32_t const bitmaskVal = bitmask[batchIdx][bitmaskIdx];

#pragma unroll
    for (int i = 0; i < kBitsPerMaskElement; ++i)
    {
        int offset = (i + threadIdx.x) % warpSize;
        if (bitmaskIdx * kBitsPerMaskElement + offset >= vocabSizePadded)
        {
            continue;
        }
        if (!((bitmaskVal >> offset) & 1))
        {
            logitsSmem[threadIdx.x * kBitsPerMaskElement + offset] = negativeInfinity<T>();
        }
    }
    __syncthreads();

#pragma unroll
    for (int offset = 0; offset < kThreadsPerBlock * kBitsPerMaskElement; offset += kThreadsPerBlock * kAlignment)
    {
        int localOffset = offset + threadIdx.x * kAlignment;
        if (logitsGmemOffset + localOffset >= vocabSizePadded)
        {
            break;
        }
        *reinterpret_cast<PackedT*>(logitsGmemPtr + localOffset)
            = *reinterpret_cast<PackedT*>(logitsSmem + localOffset);
    }
}
} // namespace

template <typename T>
void invokeLogitsBitmask(
    T** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream)
{
    int bitmaskSize = ceilDiv(vocabSizePadded, kBitsPerMaskElement);
    dim3 grid(ceilDiv(bitmaskSize, kThreadsPerBlock), batchSize);
    dim3 block(kThreadsPerBlock);

    if (vocabSizePadded % (sizeof(float4) / sizeof(T)) == 0)
    {
        logitsBitmaskKernel<T, float4><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
    }
    else if (vocabSizePadded % (sizeof(float2) / sizeof(T)) == 0)
    {
        logitsBitmaskKernel<T, float2><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
    }
    else if (vocabSizePadded % (sizeof(float) / sizeof(T)) == 0)
    {
        logitsBitmaskKernel<T, float><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
    }
    else
    {
        logitsBitmaskKernel<T, T><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
    }
}

template void invokeLogitsBitmask<float>(
    float** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream);
template void invokeLogitsBitmask<half>(
    half** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeLogitsBitmask<__nv_bfloat16>(
    __nv_bfloat16** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream);
#endif
} // namespace kernels
} // namespace tensorrt_llm
