/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
__global__ void batchApplyPenalty(T* logits, const T* biases, int* penaltyWorkspace, const int* penaltyWorkspacePrev,
    const float* temperatures, const float* repetitionPenalties, const float* presencePenalties,
    const float* frequencyPenalties, const bool accumulateVocab, const int maxSeqLen, const int vocabSize,
    const int vocabSizePadded, const int** outputIdsPtr, const int** parentIdsPtr, const int* inputLengths,
    const int* sequenceLengths, const int* minLengths, const int* endIds, const int* batchSlots)
{
    const int beamWidth = gridDim.y;
    const int batchIdx = blockIdx.x;
    const int beamIdx = blockIdx.y;
    const int batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    const int batchBeamIdx = batchIdx * beamWidth + beamIdx;
    const int batchSlotBeamIdx = batchSlot * beamWidth + beamIdx;
    const int inputLen = inputLengths == nullptr ? 0 : inputLengths[batchSlotBeamIdx];
    const int currentStep = sequenceLengths == nullptr ? 0 : sequenceLengths[batchSlotBeamIdx];
    // Initialize or update the number of occurrences of tokens
    if (accumulateVocab)
    {
        penaltyWorkspace += batchBeamIdx * vocabSize;
        if (currentStep <= inputLen)
        { // Context phase
            for (int index = threadIdx.x; index < vocabSize; index += blockDim.x)
            {
                penaltyWorkspace[index] = 0;
            }
            __syncthreads();
            for (int step = threadIdx.x; step < inputLen; step += blockDim.x)
            {
                // All beams in the context phase are identical
                int penaltyIndex = outputIdsPtr[batchSlot][beamIdx * maxSeqLen + step];
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
                int parentBeam = parentIdsPtr[batchSlot][beamIdx * maxSeqLen + currentStep - 2];
                penaltyWorkspacePrev += (batchIdx * beamWidth + parentBeam) * vocabSize;
                for (int index = threadIdx.x; index < vocabSize; index += blockDim.x)
                {
                    penaltyWorkspace[index] = penaltyWorkspacePrev[index];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                int penaltyIndex = outputIdsPtr[batchSlot][beamIdx * maxSeqLen + currentStep - 1];
                if (penaltyIndex < vocabSize)
                {
                    penaltyWorkspace[penaltyIndex] += 1;
                }
            }
        }
        __syncthreads();
    }
    // Apply bias and penalties
    logits += batchBeamIdx * vocabSizePadded;
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
    for (int index = threadIdx.x; index < vocabSizePadded; index += blockDim.x)
    {
        if (index < vocabSize)
        {
            float logit = (float) logits[index];
            // Bias
            if (biases != nullptr)
            {
                logit += (float) biases[index];
            }
            // Temperature
            if (temperatures != nullptr)
            {
                logit *= invTemperature;
            }
            int numOccurences = penaltyWorkspace[index];
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
            logits[index] = logit;
        }
        else
        {
            logits[index] = MASK_VAL;
        }
    }
    if (minLengths != nullptr)
    {
        __syncthreads();
        // Min length
        if ((threadIdx.x == 0) && (currentStep - inputLen < minLengths[batchSlot]))
        {
            logits[endIds[batchSlot]] = MASK_VAL;
        }
    }
}

template <typename T>
void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<T>& params)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    dim3 block(256);
    dim3 grid(params.batchSize, params.beamWidth);
    batchApplyPenalty<T><<<grid, block, 0, params.stream>>>(params.logits, params.biases, params.penaltyWorkspace,
        params.penaltyWorkspacePrev, params.temperatures, params.repetitionPenalties, params.presencePenalties,
        params.frequencyPenalties, params.accumulateVocab, params.maxSeqLen, params.vocabSize, params.vocabSizePadded,
        params.outputIdsPtr, params.parentIdsPtr, params.inputLengths, params.sequenceLengths, params.minLengths,
        params.endIds, params.batchSlots);
}

template void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<float>& params);

template void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<half>& params);

} // namespace kernels
} // namespace tensorrt_llm
