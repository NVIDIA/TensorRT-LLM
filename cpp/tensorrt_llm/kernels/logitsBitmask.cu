/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
__device__ PackedT packedNegativeInfinity()
{
    int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
    T packed[kAlignment];
#pragma unroll
    for (int i = 0; i < kAlignment; i++)
    {
        packed[i] = negativeInfinity<T>();
    }
    return *reinterpret_cast<PackedT*>(packed);
}
} // namespace

template <typename T, typename PackedT, int32_t kBitsPerThread>
__global__ void __launch_bounds__(kThreadsPerBlock) logitsBitmaskKernel(
    T** __restrict__ logits, uint32_t const** __restrict__ bitmask, int32_t vocabSizePadded, int32_t bitmaskSize)
{
    int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
    uint32_t constexpr kPackedMask = (1 << kAlignment) - 1;

    int const batchIdx = blockIdx.y;

    int const blockOffset = blockIdx.x * kThreadsPerBlock * kBitsPerThread;
    T* logitsGmemPtr = logits[batchIdx] + blockOffset;

    uint32_t const* bitmaskGmemPtr = bitmask[batchIdx] + blockOffset / kBitsPerMaskElement;
    int const bitmaskInnerIdx = threadIdx.x % (kBitsPerMaskElement / kAlignment);
    T logitsReg[kAlignment];

#pragma unroll
    for (int offset = threadIdx.x * kAlignment; offset < kThreadsPerBlock * kBitsPerThread;
         offset += kThreadsPerBlock * kAlignment)
    {
        if (blockOffset + offset >= vocabSizePadded)
        {
            break;
        }

        uint32_t const bitmaskVal
            = (~bitmaskGmemPtr[offset / kBitsPerMaskElement] >> (bitmaskInnerIdx * kAlignment)) & kPackedMask;

        if (bitmaskVal == 0)
        {
            continue;
        }

        if (bitmaskVal == kPackedMask)
        {
            *reinterpret_cast<PackedT*>(logitsGmemPtr + offset) = packedNegativeInfinity<T, PackedT>();
            continue;
        }

        *reinterpret_cast<PackedT*>(logitsReg) = *reinterpret_cast<PackedT*>(logitsGmemPtr + offset);
#pragma unroll
        for (int i = 0; i < kAlignment; i++)
        {
            if (((bitmaskVal >> i) & 1))
            {
                logitsReg[i] = negativeInfinity<T>();
            }
        }
        *reinterpret_cast<PackedT*>(logitsGmemPtr + offset) = *reinterpret_cast<PackedT*>(logitsReg);
    }
}

template <typename T, typename PackedT>
void logitsBitmaskDispatchToBitsPerThread(
    T** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream)
{
    int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const numBlocksPerRow = ceilDiv(2048 / kThreadsPerBlock * smCount, batchSize);
    int32_t const numBitsPerThread = ceilDiv(vocabSizePadded, kThreadsPerBlock * numBlocksPerRow);
    int32_t bitmaskSize = ceilDiv(vocabSizePadded, kBitsPerMaskElement);

    dim3 const block(kThreadsPerBlock);

    if (numBitsPerThread <= 4 && kAlignment <= 4)
    {
        dim3 const grid(ceilDiv(vocabSizePadded, kThreadsPerBlock * 4), batchSize);
        logitsBitmaskKernel<T, PackedT, 4><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
    }
    else if (numBitsPerThread <= 8 && kAlignment <= 8)
    {
        dim3 const grid(ceilDiv(vocabSizePadded, kThreadsPerBlock * 8), batchSize);
        logitsBitmaskKernel<T, PackedT, 8><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
    }
    else if (numBitsPerThread <= 16 && kAlignment <= 16)
    {
        dim3 const grid(ceilDiv(vocabSizePadded, kThreadsPerBlock * 16), batchSize);
        logitsBitmaskKernel<T, PackedT, 16><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
    }
    else
    {
        dim3 const grid(ceilDiv(vocabSizePadded, kThreadsPerBlock * 32), batchSize);
        logitsBitmaskKernel<T, PackedT, 32><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
    }
}

template <typename T>
void invokeLogitsBitmask(
    T** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream)
{
    // Dispatch to PackedT
    if (vocabSizePadded % (sizeof(float4) / sizeof(T)) == 0)
    {
        logitsBitmaskDispatchToBitsPerThread<T, float4>(logits, bitmask, batchSize, vocabSizePadded, stream);
    }
    else if (vocabSizePadded % (sizeof(float2) / sizeof(T)) == 0)
    {
        logitsBitmaskDispatchToBitsPerThread<T, float2>(logits, bitmask, batchSize, vocabSizePadded, stream);
    }
    else if (vocabSizePadded % (sizeof(float) / sizeof(T)) == 0)
    {
        logitsBitmaskDispatchToBitsPerThread<T, float>(logits, bitmask, batchSize, vocabSizePadded, stream);
    }
    else
    {
        logitsBitmaskDispatchToBitsPerThread<T, T>(logits, bitmask, batchSize, vocabSizePadded, stream);
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

template <typename T, typename PackedT, int32_t kBitsPerThread>
__global__ void __launch_bounds__(kThreadsPerBlock) contiguousLogitsBitmaskKernel(T* __restrict__ logits,
    uint32_t const* __restrict__ bitmask, int32_t const* __restrict__ tokenMask, int32_t const* __restrict__ d2t,
    int32_t vocabSizePadded, int32_t bitmaskSize)
{
    int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
    uint32_t constexpr kPackedMask = (1 << kAlignment) - 1;

    int const batchIdx = blockIdx.y;
    if (tokenMask != nullptr && !tokenMask[batchIdx])
    {
        return;
    }

    int const blockOffset = blockIdx.x * kThreadsPerBlock * kBitsPerThread;
    T* logitsGmemPtr = logits + batchIdx * vocabSizePadded + blockOffset;

    uint32_t const* bitmaskGmemPtr = bitmask + batchIdx * bitmaskSize + blockOffset / kBitsPerMaskElement;
    int const bitmaskInnerIdx = threadIdx.x % (kBitsPerMaskElement / kAlignment);
    T logitsReg[kAlignment];

#pragma unroll
    for (int offset = threadIdx.x * kAlignment; offset < kThreadsPerBlock * kBitsPerThread;
         offset += kThreadsPerBlock * kAlignment)
    {
        if (blockOffset + offset >= vocabSizePadded)
        {
            break;
        }

        uint32_t bitmaskVal = 0;
        if (d2t == nullptr)
        {
            bitmaskVal
                = (~bitmaskGmemPtr[offset / kBitsPerMaskElement] >> (bitmaskInnerIdx * kAlignment)) & kPackedMask;
        }
        else
        {
#pragma unroll
            for (int i = 0; i < kAlignment; i++)
            {
                int const d2tOffset = blockOffset + offset + i + d2t[blockOffset + offset + i];
                bitmaskVal |= ((~bitmask[batchIdx * bitmaskSize + d2tOffset / kBitsPerMaskElement]
                                   >> (d2tOffset % kBitsPerMaskElement))
                                  & 1)
                    << i;
            }
        }

        if (bitmaskVal == 0)
        {
            continue;
        }

        if (bitmaskVal == kPackedMask)
        {
            *reinterpret_cast<PackedT*>(logitsGmemPtr + offset) = packedNegativeInfinity<T, PackedT>();
            continue;
        }

        *reinterpret_cast<PackedT*>(logitsReg) = *reinterpret_cast<PackedT*>(logitsGmemPtr + offset);
#pragma unroll
        for (int i = 0; i < kAlignment; i++)
        {
            if (((bitmaskVal >> i) & 1))
            {
                logitsReg[i] = negativeInfinity<T>();
            }
        }
        *reinterpret_cast<PackedT*>(logitsGmemPtr + offset) = *reinterpret_cast<PackedT*>(logitsReg);
    }
}

template <typename T, typename PackedT>
void contiguousLogitsBitmaskDispatchToBitsPerThread(T* logits, uint32_t const* bitmask, int32_t const* tokenMask,
    int32_t const* d2t, int32_t batchSize, int32_t vocabSizePadded, int32_t bitmaskSize, cudaStream_t stream)
{
    int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const numBlocksPerRow = ceilDiv(2048 / kThreadsPerBlock * smCount, batchSize);
    int32_t const numBitsPerThread = ceilDiv(vocabSizePadded, kThreadsPerBlock * numBlocksPerRow);

    dim3 const block(kThreadsPerBlock);

    if (numBitsPerThread <= 4 && kAlignment <= 4)
    {
        dim3 const grid(ceilDiv(vocabSizePadded, kThreadsPerBlock * 4), batchSize);
        contiguousLogitsBitmaskKernel<T, PackedT, 4>
            <<<grid, block, 0, stream>>>(logits, bitmask, tokenMask, d2t, vocabSizePadded, bitmaskSize);
    }
    else if (numBitsPerThread <= 8 && kAlignment <= 8)
    {
        dim3 const grid(ceilDiv(vocabSizePadded, kThreadsPerBlock * 8), batchSize);
        contiguousLogitsBitmaskKernel<T, PackedT, 8>
            <<<grid, block, 0, stream>>>(logits, bitmask, tokenMask, d2t, vocabSizePadded, bitmaskSize);
    }
    else if (numBitsPerThread <= 16 && kAlignment <= 16)
    {
        dim3 const grid(ceilDiv(vocabSizePadded, kThreadsPerBlock * 16), batchSize);
        contiguousLogitsBitmaskKernel<T, PackedT, 16>
            <<<grid, block, 0, stream>>>(logits, bitmask, tokenMask, d2t, vocabSizePadded, bitmaskSize);
    }
    else
    {
        dim3 const grid(ceilDiv(vocabSizePadded, kThreadsPerBlock * 32), batchSize);
        contiguousLogitsBitmaskKernel<T, PackedT, 32>
            <<<grid, block, 0, stream>>>(logits, bitmask, tokenMask, d2t, vocabSizePadded, bitmaskSize);
    }
}

template <typename T>
void invokeContiguousLogitsBitmask(T* logits, uint32_t const* bitmask, int32_t const* tokenMask, int32_t const* d2t,
    int32_t batchSize, int32_t vocabSizePadded, int32_t bitmaskSize, cudaStream_t stream)
{
    // Dispatch to PackedT
    if (vocabSizePadded % (sizeof(float4) / sizeof(T)) == 0)
    {
        contiguousLogitsBitmaskDispatchToBitsPerThread<T, float4>(
            logits, bitmask, tokenMask, d2t, batchSize, vocabSizePadded, bitmaskSize, stream);
    }
    else if (vocabSizePadded % (sizeof(float2) / sizeof(T)) == 0)
    {
        contiguousLogitsBitmaskDispatchToBitsPerThread<T, float2>(
            logits, bitmask, tokenMask, d2t, batchSize, vocabSizePadded, bitmaskSize, stream);
    }
    else if (vocabSizePadded % (sizeof(float) / sizeof(T)) == 0)
    {
        contiguousLogitsBitmaskDispatchToBitsPerThread<T, float>(
            logits, bitmask, tokenMask, d2t, batchSize, vocabSizePadded, bitmaskSize, stream);
    }
    else
    {
        contiguousLogitsBitmaskDispatchToBitsPerThread<T, T>(
            logits, bitmask, tokenMask, d2t, batchSize, vocabSizePadded, bitmaskSize, stream);
    }
}

template void invokeContiguousLogitsBitmask<float>(float* logits, uint32_t const* bitmask, int32_t const* tokenMask,
    int32_t const* d2t, int32_t batchSize, int32_t vocabSizePadded, int32_t bitmaskSize, cudaStream_t stream);
template void invokeContiguousLogitsBitmask<half>(half* logits, uint32_t const* bitmask, int32_t const* tokenMask,
    int32_t const* d2t, int32_t batchSize, int32_t vocabSizePadded, int32_t bitmaskSize, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeContiguousLogitsBitmask<__nv_bfloat16>(__nv_bfloat16* logits, uint32_t const* bitmask,
    int32_t const* tokenMask, int32_t const* d2t, int32_t batchSize, int32_t vocabSizePadded, int32_t bitmaskSize,
    cudaStream_t stream);
#endif

} // namespace kernels
} // namespace tensorrt_llm
