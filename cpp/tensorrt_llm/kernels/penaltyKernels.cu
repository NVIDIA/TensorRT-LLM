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

#include "tensorrt_llm/kernels/penaltyKernels.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"

#include <cassert>
#include <cfloat>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels
{

__device__ bool almostEqual(float a, float b, float epsilon)
{
    return fabs(a - b) < epsilon;
}

template <typename T>
__global__ void batchApplyPenalty(T const* const* inputLogits, T* outputLogits, T const* biases,
    TokenIdType* penaltyWorkspace, TokenIdType const* penaltyWorkspacePrev, float const* temperatures,
    float const* repetitionPenalties, float const* presencePenalties, float const* frequencyPenalties,
    SizeType32 maxSeqLen, SizeType32 vocabSize, SizeType32 vocabSizePadded, TokenIdType const** outputIdsPtr,
    SizeType32 const** parentIdsPtr, SizeType32 const* inputLengths, SizeType32 const* sequenceLengths,
    SizeType32 const* minLengths, TokenIdType const* endIds, SizeType32 const* batchSlots,
    SizeType32 const* tokensPerStep, FinishedState const* finished)
{
    auto const beamWidth = static_cast<SizeType32>(gridDim.y);
    auto const maxTokensPerStep = static_cast<SizeType32>(gridDim.z);
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const beamIdx = static_cast<SizeType32>(blockIdx.y);
    auto const stepIdx = static_cast<SizeType32>(blockIdx.z);
    auto const batchSlot = batchSlots[batchIdx];

    FinishedState const finishState = finished != nullptr ? finished[batchSlot] : FinishedState::empty();
    if (finishState.isSkipDecoding())
    {
        return;
    }

    auto const batchBeamStepIdx = (batchIdx * beamWidth + beamIdx) * maxTokensPerStep + stepIdx;
    auto const batchSlotBeamIdx = batchSlot * beamWidth + beamIdx;
    auto const inputLen = inputLengths == nullptr ? SizeType32{0} : inputLengths[batchSlotBeamIdx];
    auto const currentStep = sequenceLengths == nullptr ? SizeType32{0} : sequenceLengths[batchSlotBeamIdx];
    T const* biasBase = biases + batchSlot * vocabSizePadded;

    if (tokensPerStep != nullptr && stepIdx >= tokensPerStep[batchSlot])
    {
        return;
    }

    float invTemperature{layers::DefaultDecodingParams::getTemperature()};
    float repetitionPenalty{layers::DefaultDecodingParams::getRepetitionPenalty()};
    float presencePenalty{layers::DefaultDecodingParams::getPresencePenalty()};
    float frequencyPenalty{layers::DefaultDecodingParams::getFrequencyPenalty()};
    SizeType32 minLength{layers::DefaultDecodingParams::getMinLength()};
    bool accumulateVocab{false};
    bool hasTemperature{false};
    bool hasMinLength{false};
    if (temperatures != nullptr)
    {
        float temperature = temperatures[batchSlot];
        invTemperature = 1.0f / (temperature + 1e-6f);
        hasTemperature |= (!almostEqual(temperature, layers::DefaultDecodingParams::getTemperature(), 1e-9));
    }
    if (repetitionPenalties != nullptr)
    {
        repetitionPenalty = repetitionPenalties[batchSlot];
        accumulateVocab
            |= (!almostEqual(repetitionPenalty, layers::DefaultDecodingParams::getRepetitionPenalty(), 1e-9));
    }
    if (presencePenalties != nullptr)
    {
        presencePenalty = presencePenalties[batchSlot];
        accumulateVocab |= (!almostEqual(presencePenalty, layers::DefaultDecodingParams::getPresencePenalty(), 1e-9));
    }
    if (frequencyPenalties != nullptr)
    {
        frequencyPenalty = frequencyPenalties[batchSlot];
        accumulateVocab |= (!almostEqual(frequencyPenalty, layers::DefaultDecodingParams::getFrequencyPenalty(), 1e-9));
    }
    if (minLengths != nullptr)
    {
        minLength = minLengths[batchSlot];
        hasMinLength |= (minLength > 0);
    }

    // Initialize or update the number of occurrences of tokens
    if (accumulateVocab)
    {
        penaltyWorkspace += batchBeamStepIdx * vocabSize;
        if (currentStep <= inputLen)
        { // Context phase
            for (auto index = static_cast<SizeType32>(threadIdx.x); index < vocabSize;
                 index += static_cast<SizeType32>(blockDim.x))
            {
                penaltyWorkspace[index] = 0;
            }
            __syncthreads();
            for (auto step = static_cast<SizeType32>(threadIdx.x); step < inputLen;
                 step += static_cast<SizeType32>(blockDim.x))
            {
                // All beams in the context phase are identical
                auto penaltyIndex = outputIdsPtr[batchSlot][beamIdx * maxSeqLen + step];
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
                auto parentBeam = parentIdsPtr[batchSlot][beamIdx * maxSeqLen + currentStep - 1];
                penaltyWorkspacePrev += ((batchIdx * beamWidth + parentBeam) * maxTokensPerStep + stepIdx) * vocabSize;
                for (auto index = static_cast<SizeType32>(threadIdx.x); index < vocabSize;
                     index += static_cast<SizeType32>(blockDim.x))
                {
                    penaltyWorkspace[index] = penaltyWorkspacePrev[index];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                auto penaltyIndex = outputIdsPtr[batchSlot][beamIdx * maxSeqLen + currentStep - 1];
                if (penaltyIndex < vocabSize)
                {
                    penaltyWorkspace[penaltyIndex] += 1;
                }
            }
        }
        __syncthreads();
    }

    // Apply bias and penalties
    auto const inLogitsPtr = inputLogits[batchIdx] + (beamIdx * maxTokensPerStep + stepIdx) * vocabSizePadded;
    auto outLogitsPtr = outputLogits + batchBeamStepIdx * vocabSizePadded;
    T const MASK_VAL = (std::is_same<T, half>::value) ? -HALF_FLT_MAX : -FLT_MAX;
    for (auto index = static_cast<SizeType32>(threadIdx.x); index < vocabSizePadded;
         index += static_cast<SizeType32>(blockDim.x))
    {
        if (index < vocabSize)
        {
            auto logit = static_cast<float>(inLogitsPtr[index]);
            // Bias
            if (biases != nullptr)
            {
                logit += static_cast<float>(biasBase[index]);
            }
            // Temperature
            if (hasTemperature)
            {
                logit *= invTemperature;
            }
            if (accumulateVocab)
            {
                SizeType32 numOccurences = penaltyWorkspace[index];
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
            }
            // do clamp to prevent overflow
            if (logit > static_cast<float>(-MASK_VAL))
            {
                logit = static_cast<float>(-MASK_VAL);
            }
            else if (logit < static_cast<float>(MASK_VAL))
            {
                logit = static_cast<float>(MASK_VAL);
            }
            outLogitsPtr[index] = logit;
        }
        else
        {
            outLogitsPtr[index] = MASK_VAL;
        }
    }
    if (hasMinLength)
    {
        __syncthreads();
        // If current generation length is too short, make sure EOS doesn't have high probability.
        // This check is not needed when endId is already -1 as generation won't stop on EOS anyway.
        if ((threadIdx.x == 0) && (currentStep - inputLen < minLength) && endIds[batchSlot] > -1)
        {
            outLogitsPtr[endIds[batchSlot]] = MASK_VAL;
        }
    }
}

template <typename T>
void invokeBatchApplyPenalty(InvokeBatchApplyPenaltyParams<T> const& params)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    dim3 block(512);
    dim3 grid(params.batchSize, params.beamWidth, params.maxTokensPerStep);
    batchApplyPenalty<T><<<grid, block, 0, params.stream>>>(params.inputLogits, params.outputLogits, params.biases,
        params.penaltyWorkspace, params.penaltyWorkspacePrev, params.temperatures, params.repetitionPenalties,
        params.presencePenalties, params.frequencyPenalties, params.maxSeqLen, params.vocabSize, params.vocabSizePadded,
        params.outputIdsPtr, params.parentIdsPtr, params.inputLengths, params.sequenceLengths, params.minLengths,
        params.endIds, params.batchSlots, params.tokensPerStep, params.finished);
}

template void invokeBatchApplyPenalty(InvokeBatchApplyPenaltyParams<float> const& params);

template void invokeBatchApplyPenalty(InvokeBatchApplyPenaltyParams<half> const& params);

} // namespace tensorrt_llm::kernels
