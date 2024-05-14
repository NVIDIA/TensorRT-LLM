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
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void batchApplyPenalty(T const* const* inputLogits, T* outputLogits, T const* biases,
    TokenIdType* penaltyWorkspace, TokenIdType const* penaltyWorkspacePrev, float const* temperatures,
    float const* repetitionPenalties, float const* presencePenalties, float const* frequencyPenalties,
    bool accumulateVocab, SizeType32 maxSeqLen, SizeType32 vocabSize, SizeType32 vocabSizePadded,
    TokenIdType const** outputIdsPtr, SizeType32 const** parentIdsPtr, SizeType32 const* inputLengths,
    SizeType32 const* sequenceLengths, SizeType32 const* minLengths, TokenIdType const* endIds,
    SizeType32 const* batchSlots, SizeType32 const* tokensPerStep)
{
    auto const beamWidth = static_cast<SizeType32>(gridDim.y);
    auto const maxTokensPerStep = static_cast<SizeType32>(gridDim.z);
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const beamIdx = static_cast<SizeType32>(blockIdx.y);
    auto const stepIdx = static_cast<SizeType32>(blockIdx.z);
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    auto const batchBeamStepIdx = (batchIdx * beamWidth + beamIdx) * maxTokensPerStep + stepIdx;
    auto const batchSlotBeamIdx = batchSlot * beamWidth + beamIdx;
    auto const inputLen = inputLengths == nullptr ? SizeType32{0} : inputLengths[batchSlotBeamIdx];
    auto const currentStep = sequenceLengths == nullptr ? SizeType32{0} : sequenceLengths[batchSlotBeamIdx];
    T const* biasBase = biases + batchSlot * vocabSizePadded;

    if (tokensPerStep != nullptr && stepIdx >= tokensPerStep[batchSlot])
    {
        return;
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
                auto parentBeam = parentIdsPtr[batchSlot][beamIdx * maxSeqLen + currentStep - 2];
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
            if (temperatures != nullptr)
            {
                logit *= invTemperature;
            }
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
void invokeBatchApplyPenalty(InvokeBatchApplyPenaltyParams<T> const& params)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    dim3 block(256);
    dim3 grid(params.batchSize, params.beamWidth, params.maxTokensPerStep);
    batchApplyPenalty<T><<<grid, block, 0, params.stream>>>(params.inputLogits, params.outputLogits, params.biases,
        params.penaltyWorkspace, params.penaltyWorkspacePrev, params.temperatures, params.repetitionPenalties,
        params.presencePenalties, params.frequencyPenalties, params.accumulateVocab, params.maxSeqLen, params.vocabSize,
        params.vocabSizePadded, params.outputIdsPtr, params.parentIdsPtr, params.inputLengths, params.sequenceLengths,
        params.minLengths, params.endIds, params.batchSlots, params.tokensPerStep);
}

template void invokeBatchApplyPenalty(InvokeBatchApplyPenaltyParams<float> const& params);

template void invokeBatchApplyPenalty(InvokeBatchApplyPenaltyParams<half> const& params);

} // namespace kernels
} // namespace tensorrt_llm
