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

#include "tensorrt_llm/kernels/megaMoePrepareKernel.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kSfVecSize = 32;

template <typename T>
__device__ __forceinline__ float toFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename ExpertT, typename ScaleT>
__global__ void megaMoePrepareKernel(__nv_bfloat16 const* __restrict__ input,
    ExpertT const* __restrict__ tokenSelectedExperts, ScaleT const* __restrict__ tokenFinalScales,
    uint32_t* __restrict__ xOut, uint32_t* __restrict__ xSfOut, int64_t* __restrict__ topkIdxOut,
    float* __restrict__ topkWeightsOut, int numTokens, int hiddenSize, int topK)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int kEltsPerThread = CVT_ELTS_PER_THREAD;
    static constexpr int kThreadsPerSf = kSfVecSize / kEltsPerThread;
    using InputVec = PackedVec<__nv_bfloat16>;

    int const numColThreads = hiddenSize / kEltsPerThread;
    int const numSfCols = hiddenSize / kSfVecSize;

    for (int rowIdx = blockIdx.x; rowIdx < numTokens; rowIdx += gridDim.x)
    {
        for (int topk = threadIdx.x; topk < topK; topk += blockDim.x)
        {
            int64_t const offset = static_cast<int64_t>(rowIdx) * topK + topk;
            topkIdxOut[offset] = static_cast<int64_t>(tokenSelectedExperts[offset]);
            topkWeightsOut[offset] = toFloat(tokenFinalScales[offset]);
        }

        for (int colIdx = threadIdx.x; colIdx < numColThreads; colIdx += blockDim.x)
        {
            uint8_t* sfOut = nullptr;
            if (colIdx % kThreadsPerSf == 0)
            {
                int64_t const sfOffset = static_cast<int64_t>(rowIdx) * numSfCols + colIdx / kThreadsPerSf;
                sfOut = reinterpret_cast<uint8_t*>(xSfOut) + sfOffset;
            }

            int64_t const inOffset = static_cast<int64_t>(rowIdx) * numColThreads + colIdx;
            InputVec inVec = reinterpret_cast<InputVec const*>(input)[inOffset];
            reinterpret_cast<uint64_t*>(xOut)[inOffset]
                = cvt_warp_fp16_to_mxfp8<__nv_bfloat16, kSfVecSize>(inVec, sfOut);
        }
    }
#endif
}

template <typename ExpertT, typename ScaleT>
void launchMegaMoePrepare(void const* input, void const* tokenSelectedExperts, void const* tokenFinalScales, void* xOut,
    void* xSfOut, int64_t* topkIdxOut, float* topkWeightsOut, int numTokens, int hiddenSize, int topK,
    int multiProcessorCount, cudaStream_t stream)
{
    int const blockX = std::min(hiddenSize / CVT_ELTS_PER_THREAD, 512);
    int const numBlocksPerSm = std::max(1, 2048 / blockX);
    int const gridX = std::min(numTokens, multiProcessorCount * numBlocksPerSm);

    megaMoePrepareKernel<ExpertT, ScaleT><<<gridX, blockX, 0, stream>>>(static_cast<__nv_bfloat16 const*>(input),
        static_cast<ExpertT const*>(tokenSelectedExperts), static_cast<ScaleT const*>(tokenFinalScales),
        static_cast<uint32_t*>(xOut), static_cast<uint32_t*>(xSfOut), topkIdxOut, topkWeightsOut, numTokens, hiddenSize,
        topK);
}

template <typename ExpertT>
void dispatchMegaMoePrepareScale(void const* input, void const* tokenSelectedExperts, void const* tokenFinalScales,
    void* xOut, void* xSfOut, int64_t* topkIdxOut, float* topkWeightsOut, int numTokens, int hiddenSize, int topK,
    MegaMoePrepareScaleType scaleType, int multiProcessorCount, cudaStream_t stream)
{
    switch (scaleType)
    {
    case MegaMoePrepareScaleType::FP32:
        launchMegaMoePrepare<ExpertT, float>(input, tokenSelectedExperts, tokenFinalScales, xOut, xSfOut, topkIdxOut,
            topkWeightsOut, numTokens, hiddenSize, topK, multiProcessorCount, stream);
        break;
    case MegaMoePrepareScaleType::FP16:
        launchMegaMoePrepare<ExpertT, half>(input, tokenSelectedExperts, tokenFinalScales, xOut, xSfOut, topkIdxOut,
            topkWeightsOut, numTokens, hiddenSize, topK, multiProcessorCount, stream);
        break;
    case MegaMoePrepareScaleType::BF16:
        launchMegaMoePrepare<ExpertT, __nv_bfloat16>(input, tokenSelectedExperts, tokenFinalScales, xOut, xSfOut,
            topkIdxOut, topkWeightsOut, numTokens, hiddenSize, topK, multiProcessorCount, stream);
        break;
    }
}

} // namespace

void invokeMegaMoePrepare(void const* input, void const* tokenSelectedExperts, void const* tokenFinalScales, void* xOut,
    void* xSfOut, int64_t* topkIdxOut, float* topkWeightsOut, int numTokens, int hiddenSize, int topK,
    MegaMoePrepareExpertType expertType, MegaMoePrepareScaleType scaleType, int multiProcessorCount,
    cudaStream_t stream)
{
    if (numTokens == 0)
    {
        return;
    }

    switch (expertType)
    {
    case MegaMoePrepareExpertType::INT32:
        dispatchMegaMoePrepareScale<int32_t>(input, tokenSelectedExperts, tokenFinalScales, xOut, xSfOut, topkIdxOut,
            topkWeightsOut, numTokens, hiddenSize, topK, scaleType, multiProcessorCount, stream);
        break;
    case MegaMoePrepareExpertType::INT64:
        dispatchMegaMoePrepareScale<int64_t>(input, tokenSelectedExperts, tokenFinalScales, xOut, xSfOut, topkIdxOut,
            topkWeightsOut, numTokens, hiddenSize, topK, scaleType, multiProcessorCount, stream);
        break;
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
