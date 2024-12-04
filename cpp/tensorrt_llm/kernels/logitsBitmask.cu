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

constexpr int32_t kBitsPerMaskElement = 32;
constexpr int32_t kThreadsPerBlock = 512;

template <typename T>
__device__ T GetNegativeInfinity()
{
    return -INFINITY;
}

template <>
__device__ half GetNegativeInfinity<half>()
{
    return __float2half(-INFINITY);
}

template <typename T>
__global__ void __launch_bounds__(512) logitsBitmaskKernel(
    T** __restrict__ logits, uint32_t const** __restrict__ bitmask, int32_t vocabSizePadded, int32_t bitmaskSize)
{
    int batchIdx = blockIdx.y;
    int bitmaskIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (bitmaskIdx >= bitmaskSize)
    {
        return;
    }

    uint32_t bitmaskVal = bitmask[batchIdx][bitmaskIdx];
    T* logitsPtr = logits[batchIdx] + bitmaskIdx * kBitsPerMaskElement;
    for (int i = 0; i < kBitsPerMaskElement; ++i)
    {
        if (bitmaskIdx * kBitsPerMaskElement + i >= vocabSizePadded)
        {
            break;
        }
        if (!(bitmaskVal & 1))
        {
            // TODO(enweiz): Fix uncoalesced global memory access here.
            logitsPtr[i] = GetNegativeInfinity<T>();
        }
        bitmaskVal >>= 1;
    }
}

template <typename T>
void invokeLogitsBitmask(
    T** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream)
{
    int bitmaskSize = ceilDiv(vocabSizePadded, kBitsPerMaskElement);
    dim3 grid(ceilDiv(bitmaskSize, kThreadsPerBlock), batchSize);
    dim3 block(kThreadsPerBlock);

    logitsBitmaskKernel<T><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
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
