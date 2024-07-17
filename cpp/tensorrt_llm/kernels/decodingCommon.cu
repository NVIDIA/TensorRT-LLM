/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/common.h"
#include <stdio.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace kernels
{

__global__ void curandInitialize(curandState_t* state, int const* batchSlots, int const size, const uint64_t randomSeed)
{
    int const idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        auto const batchSlot = batchSlots != nullptr ? batchSlots[idx] : idx;
        curand_init(randomSeed, 0, 0, &state[batchSlot]);
    }
}

void invokeCurandInitialize(
    curandState_t* state, int const* batchSlots, const size_t batchSize, const uint64_t randomSeed, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((int) (ceil(batchSize * 1.0 / 256)));
    curandInitialize<<<grid, block, 0, stream>>>(state, batchSlots, batchSize, randomSeed);
}

__global__ void curandBatchInitialize(
    curandState_t* states, SizeType32 const* batchSlots, SizeType32 const size, uint64_t const* randomSeeds)
{
    SizeType32 const bid = threadIdx.x + blockIdx.x * blockDim.x;
    if (bid < size)
    {
        auto const batchSlot = batchSlots != nullptr ? batchSlots[bid] : bid;
        curand_init(randomSeeds[bid], 0, 0, &states[batchSlot]);
    }
}

void invokeCurandBatchInitialize(curandState_t* states, SizeType32 const* batchSlots, const size_t batchSize,
    uint64_t const* randomSeeds, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(static_cast<SizeType32>(ceil(batchSize * 1.0 / 256)));
    curandBatchInitialize<<<grid, block, 0, stream>>>(states, batchSlots, batchSize, randomSeeds);
}

template <typename T>
__global__ void addBiasSoftMax(T* logits, T** logitsPtrs, T* probs, T const* bias, int32_t const* endIds,
    FinishedState const* finished, int32_t const* batchSlots, int32_t batchSize, int32_t maxBatchSize,
    int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded, bool skipSoftMax, bool batchSlotsLogits)
{
    auto const batchIdx = blockIdx.x;
    auto const beamIdx = blockIdx.y;
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;
    auto const batchIdxLogits = batchSlotsLogits ? batchSlot : batchIdx;
    FinishedState const finishState
        = finished != nullptr ? finished[beamIdx * maxBatchSize + batchSlot] : FinishedState::empty();
    if (finishState.isSkipDecoding())
    {
        return;
    }

    auto logitsPtr = logitsPtrs ? logitsPtrs[batchIdx] + beamIdx * vocabSizePadded
                                : logits + (batchIdxLogits * beamWidth + beamIdx) * vocabSizePadded;

    bool finish = finishState.isFinished();
    int offset = (batchIdxLogits * beamWidth + beamIdx) * vocabSizePadded;

    float maxVal = -1 * FLT_MAX;
    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    __shared__ float sMaxVal;
    __shared__ float sSumVal;

    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        auto logit = logitsPtr[tid];
        if (tid < vocabSize)
        {
            if (finish && endIds != nullptr)
            {
                logit = (tid == endIds[batchSlot]) ? MAX_T_VAL : -MAX_T_VAL;
            }
            else
            {
                T bias_val = (bias != nullptr) ? bias[tid] : (T) 0.0f;
                logit += bias_val;
            }
        }
        else
        {
            logit = -MAX_T_VAL;
        }
        maxVal = max(maxVal, (float) logit);
        logitsPtr[tid] = logit;
    }

    if (!skipSoftMax)
    {
        maxVal = blockReduceMax<float>((float) maxVal);
        if (threadIdx.x == 0)
        {
            sMaxVal = maxVal;
        }
        __syncthreads();

        float sumVal = 0.0f;
        for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
        {
            probs[offset + tid] = __expf((float) logitsPtr[tid] - sMaxVal);
            sumVal += (float) probs[offset + tid];
        }

        sumVal = blockReduceSum<float>(sumVal);
        if (threadIdx.x == 0)
        {
            sSumVal = sumVal;
        }
        __syncthreads();

        for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
        {
            probs[offset + tid] = ((float) probs[offset + tid] / (sSumVal + 1e-6f));
        }
    }
}

template <typename T>
void invokeAddBiasSoftMax(T* logits, T** logitsPtrs, T* probs, T const* bias, int32_t const* endIds,
    FinishedState const* finished, int32_t const* batchSlots, int32_t batchSize, int32_t maxBatchSize,
    int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded, bool skipSoftMax, bool batchSlotsLogits,
    cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    dim3 grid(batchSize, beamWidth);
    auto const vocabRoundedToWarp = roundUp(vocabSize, 32);
    dim3 block(min(vocabRoundedToWarp, 1024));
    // vocabSize, e.g., 30000, 7000.... vocabSize is usually very big.
    addBiasSoftMax<<<grid, block, 0, stream>>>(logits, logitsPtrs, probs, bias, endIds, finished, batchSlots, batchSize,
        maxBatchSize, beamWidth, vocabSize, vocabSizePadded, skipSoftMax, batchSlotsLogits);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template void invokeAddBiasSoftMax(float* logits, float** logitsPtrs, float* probs, float const* bias,
    int32_t const* endIds, FinishedState const* finished, int32_t const* batchSlots, int32_t batchSize,
    int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded, bool skipSoftMax,
    bool batchSlotsLogits, cudaStream_t stream);

template void invokeAddBiasSoftMax(half* logits, half** logitsPtrs, half* probs, half const* bias,
    int32_t const* endIds, FinishedState const* finished, int32_t const* batchSlots, int32_t batchSize,
    int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded, bool skipSoftMax,
    bool batchSlotsLogits, cudaStream_t stream);

template <typename T>
__global__ void scatterDecodingParamsKernel(T const* src, T* dst, int const* batchSlots, int batchSize)
{
    auto const batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize)
    {
        return;
    }
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    dst[batchSlot] = src[batchIdx];
}

template <typename T>
void invokeScatterDecodingParams(T const* src, T* dst, int const* batchSlots, int batchSize, cudaStream_t stream)
{
    constexpr int THREADS_PER_CTA = 256;
    dim3 grid(divUp(batchSize, THREADS_PER_CTA));
    scatterDecodingParamsKernel<<<grid, THREADS_PER_CTA, 0, stream>>>(src, dst, batchSlots, batchSize);
}

template void invokeScatterDecodingParams(
    float const* src, float* dst, int const* batchSlots, int batchSize, cudaStream_t stream);
template void invokeScatterDecodingParams(
    uint32_t const* src, uint32_t* dst, int const* batchSlots, int batchSize, cudaStream_t stream);
template void invokeScatterDecodingParams(
    int32_t const* src, int32_t* dst, int const* batchSlots, int batchSize, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
