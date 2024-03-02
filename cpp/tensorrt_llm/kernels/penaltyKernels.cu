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

#include <assert.h>
#include <float.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/penaltyKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void batchApplyPenalty(T const* const* inputLogits, T* outputLogits, T const* biases,
    int32_t* penaltyWorkspace, int32_t const* penaltyWorkspacePrev, float const* temperatures,
    float const* repetitionPenalties, float const* presencePenalties, float const* frequencyPenalties,
    const bool accumulateVocab, int32_t const maxSeqLen, int32_t const vocabSize, int32_t const vocabSizePadded,
    int32_t const** outputIdsPtr, int32_t const** parentIdsPtr, int32_t const* inputLengths,
    int32_t const* sequenceLengths, int32_t const* minLengths, int32_t const* endIds, int32_t const* batchSlots)
{
    int32_t const beamWidth = gridDim.y;
    int32_t const batchIdx = blockIdx.x;
    int32_t const beamIdx = blockIdx.y;
    int32_t const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    int32_t const batchBeamIdx = batchIdx * beamWidth + beamIdx;
    int32_t const batchSlotBeamIdx = batchSlot * beamWidth + beamIdx;
    int32_t const inputLen = inputLengths == nullptr ? 0 : inputLengths[batchSlotBeamIdx];
    int32_t const currentStep = sequenceLengths == nullptr ? 0 : sequenceLengths[batchSlotBeamIdx];
    T const* biasBase = biases + batchSlot * vocabSizePadded;
    // Initialize or update the number of occurrences of tokens
    if (accumulateVocab)
    {
        penaltyWorkspace += batchBeamIdx * vocabSize;
        if (currentStep <= inputLen)
        { // Context phase
            for (int32_t index = threadIdx.x; index < vocabSize; index += blockDim.x)
            {
                penaltyWorkspace[index] = 0;
            }
            __syncthreads();
            for (int32_t step = threadIdx.x; step < inputLen; step += blockDim.x)
            {
                // All beams in the context phase are identical
                int32_t penaltyIndex = outputIdsPtr[batchSlot][beamIdx * maxSeqLen + step];
                if (penaltyIndex < vocabSize)
                {
                    atomicAdd(&penaltyWorkspace[penaltyIndex], 1);
                }
            }
        }
        else
        { // Generation phase
            if (beamWidth > 1)
            {
                int32_t parentBeam = parentIdsPtr[batchSlot][beamIdx * maxSeqLen + currentStep - 2];
                penaltyWorkspacePrev += (batchIdx * beamWidth + parentBeam) * vocabSize;
                for (int32_t index = threadIdx.x; index < vocabSize; index += blockDim.x)
                {
                    penaltyWorkspace[index] = penaltyWorkspacePrev[index];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                int32_t penaltyIndex = outputIdsPtr[batchSlot][beamIdx * maxSeqLen + currentStep - 1];
                if (penaltyIndex < vocabSize)
                {
                    penaltyWorkspace[penaltyIndex] += 1;
                }
            }
        }
        __syncthreads();
    }
    // Apply bias and penalties
    auto const inLogitsPtr = inputLogits[batchIdx] + beamIdx * vocabSizePadded;
    auto outLogitsPtr = outputLogits + batchBeamIdx * vocabSizePadded;
    const T MASK_VAL = (std::is_same<T, half>::value) ? -HALF_FLT_MAX : -FLT_MAX;
    float invTemperature, repetitionPenalty, presencePenalty, frequencyPenalty;
    if (temperatures != nullptr)
    {
        invTemperature = 1.0f / (temperatures[batchSlot] + 1e-6f);
    }
    if (repetitionPenalties != nullptr)
    {
        repetitionPenalty = repetitionPenalties[batchSlot];
    }
    if (presencePenalties != nullptr)
    {
        presencePenalty = presencePenalties[batchSlot];
    }
    if (frequencyPenalties != nullptr)
    {
        frequencyPenalty = frequencyPenalties[batchSlot];
    }
    for (int32_t index = threadIdx.x; index < vocabSizePadded; index += blockDim.x)
    {
        if (index < vocabSize)
        {
            float logit = (float) inLogitsPtr[index];
            // Bias
            if (biases != nullptr)
            {
                logit += (float) biasBase[index];
            }
            // Temperature
            if (temperatures != nullptr)
            {
                logit *= invTemperature;
            }
            int32_t numOccurences = penaltyWorkspace[index];
            if (numOccurences > 0)
            {
                // Repetition
                if (repetitionPenalties != nullptr)
                {
                    logit = logit < 0.0f ? logit * repetitionPenalty : logit / repetitionPenalty;
                }
                // Presence
                if (presencePenalties != nullptr)
                {
                    logit -= presencePenalty;
                }
                // Frequency
                if (frequencyPenalties != nullptr)
                {
                    logit -= frequencyPenalty * numOccurences;
                }
            }
            outLogitsPtr[index] = logit;
        }
        else
        {
            outLogitsPtr[index] = MASK_VAL;
        }
    }
    if (minLengths != nullptr)
    {
        __syncthreads();
        // Min length
        if ((threadIdx.x == 0) && (currentStep - inputLen < minLengths[batchSlot]))
        {
            outLogitsPtr[endIds[batchSlot]] = MASK_VAL;
        }
    }
}

template <typename T>
void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<T>& params)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    dim3 block(256);
    dim3 grid(params.batchSize, params.beamWidth);
    batchApplyPenalty<T><<<grid, block, 0, params.stream>>>(params.inputLogits, params.outputLogits, params.biases,
        params.penaltyWorkspace, params.penaltyWorkspacePrev, params.temperatures, params.repetitionPenalties,
        params.presencePenalties, params.frequencyPenalties, params.accumulateVocab, params.maxSeqLen, params.vocabSize,
        params.vocabSizePadded, params.outputIdsPtr, params.parentIdsPtr, params.inputLengths, params.sequenceLengths,
        params.minLengths, params.endIds, params.batchSlots);
}

template void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<float>& params);

template void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<half>& params);

} // namespace kernels
} // namespace tensorrt_llm
