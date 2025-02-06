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

#include "tensorrt_llm/kernels/decodingCommon.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/runtime/common.h"

#include <cstdint>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels
{

__global__ void curandInitialize(curandState_t* state, int const* batchSlots, int const size, uint64_t const randomSeed)
{
    int const idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        auto const batchSlot = batchSlots != nullptr ? batchSlots[idx] : idx;
        curand_init(randomSeed, 0, 0, &state[batchSlot]);
    }
}

void invokeCurandInitialize(
    curandState_t* state, int const* batchSlots, size_t const batchSize, uint64_t const randomSeed, cudaStream_t stream)
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

void invokeCurandBatchInitialize(curandState_t* states, SizeType32 const* batchSlots, size_t const batchSize,
    uint64_t const* randomSeeds, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(static_cast<SizeType32>(ceil(batchSize * 1.0 / 256)));
    curandBatchInitialize<<<grid, block, 0, stream>>>(states, batchSlots, batchSize, randomSeeds);
}

template <typename T>
__global__ void addBiasSoftMax(T* logits, T** logitsPtrs, T* probs, float* outputEntropy, T const* bias,
    float const* temperatures, int32_t const* endIds, FinishedState const* finished, int32_t const* beamWidths,
    int32_t const* batchSlots, float const* minPs, int32_t maxBatchSize, int32_t maxBeamWidth, int32_t vocabSize,
    int32_t vocabSizePadded, bool skipSoftMax, bool batchSlotsLogits, bool ptrsForBeams, bool const* skipDecode)
{
    auto const batchIdx = blockIdx.x;
    auto const beamIdx = blockIdx.y;
    auto const batchSlot = batchSlots ? batchSlots[batchIdx] : batchIdx;
    if (beamWidths && beamIdx >= beamWidths[batchSlot])
    {
        return;
    }
    if ((skipDecode != nullptr && skipDecode[batchSlot]))
    {
        return;
    }

    auto const batchIdxLogits = batchSlotsLogits ? batchSlot : batchIdx;
    FinishedState const finishState
        = finished != nullptr ? finished[beamIdx * maxBatchSize + batchSlot] : FinishedState::empty();
    if (finishState.isSkipDecoding())
    {
        return;
    }
    bool const finish = finishState.isFinished();

    auto logitsPtr = logitsPtrs ? (ptrsForBeams ? logitsPtrs[batchIdx * maxBeamWidth + beamIdx]
                                                : logitsPtrs[batchIdx] + beamIdx * vocabSizePadded)
                                : logits + (batchIdxLogits * maxBeamWidth + beamIdx) * vocabSizePadded;

    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;
    float const EPSILON = (std::is_same<T, half>::value) ? 1e-3f : 1e-6f;
    float maxVal = -FLT_MAX;
    __shared__ float sMaxVal, sSumVal;

    auto const tempInv = temperatures ? T{1.f / (temperatures[batchSlot] + EPSILON)} : T{1.f};

    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        auto logit = logitsPtr[tid];
        logit = temperatures ? logit * tempInv : logit;
        if (tid < vocabSize)
        {
            if (finish && endIds != nullptr)
            {
                // Prefer token EOS if the request has finished
                logit = (tid == endIds[batchSlot]) ? MAX_T_VAL : -MAX_T_VAL;
            }
            else
            {
                // Compute biased logit if the request has not finished, or `endIds` is nullptr
                logit += (bias != nullptr) ? bias[tid] : T{0.0f};
            }
        }
        else
        {
            logit = -MAX_T_VAL;
        }
        maxVal = max(maxVal, static_cast<float>(logit));
        logitsPtr[tid] = logit; // Write back biased logits
    }

    float minP = minPs != nullptr ? minPs[batchSlot] : 0.0f;

    if (!skipSoftMax)
    {
        maxVal = blockReduceMax<float>(static_cast<float>(maxVal));
        if (threadIdx.x == 0)
        {
            sMaxVal = maxVal;
        }
        __syncthreads();

        // `probs == nullptr` is specialization for Beam-Search, which needs log and writes output to`logitsPtrs`
        float sumVal = 0.0f;
        int const offset = (probs != nullptr) ? ((batchIdxLogits * maxBeamWidth + beamIdx) * vocabSizePadded) : 0;
        T* dst = (probs != nullptr) ? probs : logitsPtr;
        for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
        {
            auto value = __expf(static_cast<float>(logitsPtr[tid]) - sMaxVal);
            // minP : probability of token proportional to the max token
            // compare minP against exp(logit - maxVal) / exp(maxVal - maxVal) = exp(logit - maxVal)
            if (value < minP)
            {
                value = 0.0;
                logitsPtr[tid] = -MAX_T_VAL;
            }
            dst[offset + tid] = value;
            sumVal += value;
        }

        sumVal = blockReduceSum<float>(sumVal);
        if (threadIdx.x == 0)
        {
            sSumVal = sumVal;
        }
        __syncthreads();

        float entropy{0.f};
        for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
        {
            auto const softmaxValue = static_cast<float>(dst[offset + tid]) / (sSumVal + EPSILON);
            auto const probValue = (probs != nullptr) ? softmaxValue : __logf(softmaxValue);
            if (outputEntropy)
            {
                entropy += probValue * __logf(probValue + EPSILON);
            }
            dst[offset + tid] = probValue;
        }

        if (outputEntropy)
        {
            entropy = blockReduceSum<float>(entropy);

            if (threadIdx.x == 0)
            {
                outputEntropy[batchSlot * maxBeamWidth + beamIdx] = -entropy;
            }
        }
    }
}

template <typename T>
void invokeAddBiasSoftMax(BiasSoftmaxParams<T> const params, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    dim3 grid(params.batchSize, params.maxBeamWidth);
    auto const vocabRoundedToWarp = roundUp(params.vocabSize, 32);
    dim3 block(std::min(vocabRoundedToWarp, 1024)); // vocabSize is usually larger than 1024
    addBiasSoftMax<<<grid, block, 0, stream>>>(params.logits, params.logitsPtrs, params.probs, params.outputEntropy,
        params.bias, params.temperatures, params.endIds, params.finished, params.beamWidths, params.batchSlots,
        params.minPs, params.maxBatchSize, params.maxBeamWidth, params.vocabSize, params.vocabSizePadded,
        params.skipSoftMax, params.batchSlotsLogits, params.ptrsForBeams, params.skipDecode);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template void invokeAddBiasSoftMax(BiasSoftmaxParams<float> const params, cudaStream_t stream);
template void invokeAddBiasSoftMax(BiasSoftmaxParams<half> const params, cudaStream_t stream);

template <typename T>
__global__ void scatterDecodingParamsKernel(T const* src, T scalar, T* dst, int const* batchSlots, int batchSize)
{
    auto const batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize)
    {
        return;
    }
    auto const batchSlot = batchSlots[batchIdx];
    dst[batchSlot] = (src == nullptr ? scalar : src[batchIdx]);
}

template <typename T>
void invokeScatterDecodingParams(
    T const* src, T scalar, T* dst, int const* batchSlots, int batchSize, cudaStream_t stream)
{
    constexpr int THREADS_PER_CTA = 256;
    dim3 grid(divUp(batchSize, THREADS_PER_CTA));
    scatterDecodingParamsKernel<<<grid, THREADS_PER_CTA, 0, stream>>>(src, scalar, dst, batchSlots, batchSize);
}

template void invokeScatterDecodingParams(
    float const* src, float scalar, float* dst, int const* batchSlots, int batchSize, cudaStream_t stream);
template void invokeScatterDecodingParams(
    uint32_t const* src, uint32_t scalar, uint32_t* dst, int const* batchSlots, int batchSize, cudaStream_t stream);
template void invokeScatterDecodingParams(
    int32_t const* src, int32_t scalar, int32_t* dst, int const* batchSlots, int batchSize, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
