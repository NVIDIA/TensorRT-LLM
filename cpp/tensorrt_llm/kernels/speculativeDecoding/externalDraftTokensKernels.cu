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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/speculativeDecoding/externalDraftTokensKernels.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels::speculative_decoding
{
namespace
{
__global__ void acceptDraftTokensByIds(TokenIdType const* draftIds, TokenIdType const* targetIds,
    SizeType32 const* contextLengths, SizeType32 const* numsDraftTokens, SizeType32* sequenceLengths,
    FinishedState const* finished, FinishedState* finishedFinal, SizeType32* finishedSum, SizeType32 const* batchSlots,
    SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 maxSeqLen, SizeType32 maxDraftTokens)
{
    for (auto batchIdx = static_cast<SizeType32>(threadIdx.x); batchIdx < batchSize; batchIdx += blockDim.x)
    {
        auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
        auto const numDraftTokens = numsDraftTokens[batchSlot];

        auto const contextLength = contextLengths[batchSlot];
        auto& sequenceLength = sequenceLengths[batchSlot];
        SizeType32 finishedDraftIdx = 0;
        for (auto ti = contextLength; ti < min(sequenceLength, contextLength + numDraftTokens);
             ++ti, ++finishedDraftIdx)
        {
            auto const draftIdx = ti - contextLength;
            auto const targetTokenIdx = batchSlot * maxSeqLen + ti;
            auto const draftTokenIdx = batchSlot * maxDraftTokens + draftIdx;
            // Check if draft tokens are the same as target tokens
            bool const accepted = draftIds[draftTokenIdx] == targetIds[targetTokenIdx];
            if (!accepted)
            {
                // Set sequence length to the numAcceptedTokens + 1
                sequenceLength = min(ti + 1, maxSeqLen);
                // FIXME(nkorobov): do we need to set endIds here?
                break;
            }
        }
        FinishedState finishState = finished[finishedDraftIdx * maxBatchSize + batchSlot];
        finishedFinal[batchSlot] = finishState;

        if (finishedSum)
        {
            finishedSum[batchSlot] = static_cast<int>(finishState.isFinished());
        }
    }
}
} // namespace

void invokeAcceptDraftTokensByIds(TokenIdType const* draftIds, TokenIdType const* targetIds,
    SizeType32 const* contextLengths, SizeType32 const* numsDraftTokens, SizeType32* sequenceLengths,
    FinishedState const* finished, FinishedState* finishedFinal, SizeType32* finishedSum, SizeType32 const* batchSlots,
    SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth, SizeType32 maxSeqLen,
    SizeType32 maxDraftTokens, cudaStream_t stream)
{
    TLLM_CHECK(beamWidth == 1);
    dim3 block(min(1024, batchSize));
    dim3 grid(1);
    acceptDraftTokensByIds<<<grid, block, 0, stream>>>(draftIds, targetIds, contextLengths, numsDraftTokens,
        sequenceLengths, finished, finishedFinal, finishedSum, batchSlots, batchSize, maxBatchSize, maxSeqLen,
        maxDraftTokens);
}

namespace
{
template <typename T>
__global__ void acceptDraftTokensByLogitsKernel(T const* draftProbs, T* targetProbs, SizeType32 const* numsDraftTokens,
    FinishedState* finished, curandState_t* curandState, SizeType32 const* batchSlots, SizeType32 batchSize,
    SizeType32 maxBatchSize, SizeType32 maxDraftTokens, SizeType32 beamWidth, SizeType32 vocabSize,
    bool randomThreshold, float constantThreshold)
{
    auto const bid = blockIdx.x;
    auto const draftTokenIdx = blockIdx.y;
    auto const batchIdx = bid / beamWidth;
    auto const beamIdx = bid % beamWidth;
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    auto const batchSlotBeamWidth = batchSlot * beamWidth + beamIdx;

    auto const numDraftTokens = numsDraftTokens[batchSlotBeamWidth];

    if (draftTokenIdx >= numDraftTokens)
    {
        return;
    }

    auto const logitsOffset = (batchSlot * maxDraftTokens + draftTokenIdx) * beamWidth * vocabSize;
    auto const draftProbsBatch = draftProbs + logitsOffset;
    auto const targetProbsBatch = targetProbs + logitsOffset;

    SizeType32 rejected = 0;
    auto vocabSizePadded = static_cast<SizeType32>((vocabSize + blockDim.x - 1) / blockDim.x) * blockDim.x;

    for (auto vIdx = static_cast<SizeType32>(threadIdx.x); vIdx < vocabSizePadded;
         vIdx += static_cast<SizeType32>(blockDim.x))
    {
        if (rejected > 0)
        {
            break;
        }

        // FIXME(nkorobov): We compare probability distributions, but it might make sense to compare probabilities of
        // the selected tokens based on the https://arxiv.org/pdf/2302.01318.pdf
        bool const pred = vIdx < vocabSize;
        auto const threshold
            = pred ? (randomThreshold ? curand_uniform(curandState + batchSlot) : constantThreshold) : 0.f;
        auto const targetProb = pred ? static_cast<float>(targetProbsBatch[vIdx]) : 1.f;
        auto const draftProb = pred ? static_cast<float>(draftProbsBatch[vIdx]) : 0.f;

        rejected = __syncthreads_count(targetProb < threshold * draftProb);
    }
    if (threadIdx.x == 0)
    {
        finished[draftTokenIdx * maxBatchSize * beamWidth + batchSlotBeamWidth]
            = rejected > 0 ? FinishedState::skipDecoding() : FinishedState::empty();
    }
}

template <typename T>
__global__ void correctAcceptedStatesAndLogits(T const* draftProbs, T* targetProbs, T** targetLogits,
    SizeType32 const* numsDraftTokens, FinishedState* finished, SizeType32 const* batchSlots, SizeType32 batchSize,
    SizeType32 maxBatchSize, SizeType32 maxDraftTokens, SizeType32 beamWidth, SizeType32 vocabSize)
{
    auto const bid = blockIdx.x;
    auto const batchIdx = bid / beamWidth;
    auto const beamIdx = bid % beamWidth;
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    auto const batchSlotBeamWidth = batchSlot * beamWidth + beamIdx;
    auto const numDraftTokens = numsDraftTokens[batchSlotBeamWidth];

    __shared__ SizeType32 numAcceptedTokens;
    if (threadIdx.x == 0)
    {
        numAcceptedTokens = numDraftTokens;
        bool cummulativeSkipDecoding = false;
        for (SizeType32 ti = 0; ti < numDraftTokens + 1; ++ti)
        {
            auto& finishedState = finished[ti * maxBatchSize * beamWidth + batchSlotBeamWidth];
            bool localSkipDecoding = finishedState.isSkipDecoding();
            if (cummulativeSkipDecoding == false && localSkipDecoding == true)
            {
                numAcceptedTokens = ti;
            }

            finishedState = cummulativeSkipDecoding ? FinishedState::skipDecoding() : FinishedState::empty();
            cummulativeSkipDecoding |= localSkipDecoding;
        }
    }
    __syncthreads();

    if (numAcceptedTokens < numDraftTokens)
    {
        auto const logitsIdx = (batchSlot * maxDraftTokens + numAcceptedTokens) * beamWidth * vocabSize;
        auto const draftProbBatch = draftProbs + logitsIdx;
        auto targetProbBatch = targetProbs + logitsIdx;
        auto targetLogitsBatch = targetLogits[bid] + numAcceptedTokens * beamWidth * vocabSize;

        float sumProbs = 0.f;
        for (SizeType32 vIdx = static_cast<SizeType32>(threadIdx.x); vIdx < vocabSize;
             vIdx += static_cast<SizeType32>(blockDim.x))
        {
            auto const correctedProb = max(static_cast<float>(targetProbBatch[vIdx] - draftProbBatch[vIdx]), 0.f);
            sumProbs += correctedProb;
            targetProbBatch[vIdx] = correctedProb;
        }

        __shared__ float sumProbsShared;
        sumProbs = blockReduceSum<float>((float) sumProbs);
        if (threadIdx.x == 0)
        {
            sumProbsShared = max(sumProbs, 1e-6f);
        }
        __syncthreads();

        for (SizeType32 vIdx = static_cast<SizeType32>(threadIdx.x); vIdx < vocabSize;
             vIdx += static_cast<SizeType32>(blockDim.x))
        {
            auto const correctedNormProb = static_cast<float>(targetProbBatch[vIdx]) / sumProbsShared;
            targetLogitsBatch[vIdx] = __logf(correctedNormProb / (1.f - correctedNormProb));
        }
    }
}
} // namespace

template <typename T>
void acceptDraftTokensByLogits(T* draftLogits, T** targetLogits, T* draftProbs, T* targetProbs,
    SizeType32 const* numsDraftTokens, FinishedState* finished, curandState_t* curandState,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth,
    SizeType32 vocabSize, SizeType32 vocabSizePadded, SizeType32 maxDraftTokens, bool randomThreshold,
    float constantThreshold, cudaStream_t stream)
{
    TLLM_CHECK(beamWidth == 1);
    {
        invokeAddBiasSoftMax(draftLogits, static_cast<T**>(nullptr), draftProbs, static_cast<T*>(nullptr), nullptr,
            finished, batchSlots, batchSize, maxBatchSize, beamWidth * maxDraftTokens, vocabSize, vocabSizePadded,
            /* skip softmax */ false,
            /* batchSlotLogits */ true, stream);
        invokeAddBiasSoftMax(static_cast<T*>(nullptr), targetLogits, targetProbs, static_cast<T*>(nullptr), nullptr,
            finished, batchSlots, batchSize, maxBatchSize, beamWidth * maxDraftTokens, vocabSize, vocabSizePadded,
            /* skip softmax */ false,
            /* batchSlotLogits */ true, stream);
    }
    {
        dim3 block(1024);
        dim3 grid(batchSize * beamWidth, maxDraftTokens);
        acceptDraftTokensByLogitsKernel<<<grid, block, 0, stream>>>(draftProbs, targetProbs, numsDraftTokens, finished,
            curandState, batchSlots, batchSize, maxBatchSize, maxDraftTokens, beamWidth, vocabSizePadded,
            randomThreshold, constantThreshold);
    }
    {
        dim3 block(1024);
        dim3 grid(batchSize * beamWidth);
        correctAcceptedStatesAndLogits<<<grid, block, 0, stream>>>(draftProbs, targetProbs, targetLogits,
            numsDraftTokens, finished, batchSlots, batchSize, maxBatchSize, maxDraftTokens, beamWidth, vocabSizePadded);
    }
}

template void acceptDraftTokensByLogits(float* draftLogits, float** targetLogits, float* draftProbs, float* targetProbs,
    SizeType32 const* numsDraftTokens, FinishedState* finished, curandState_t* curandState,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth,
    SizeType32 vocabSize, SizeType32 vocabSizePadded, SizeType32 maxDraftTokens, bool randomThreshold,
    float constantThreshold, cudaStream_t stream);
template void acceptDraftTokensByLogits(half* draftLogits, half** targetLogits, half* draftProbs, half* targetProbs,
    SizeType32 const* numsDraftTokens, FinishedState* finished, curandState_t* curandState,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth,
    SizeType32 vocabSize, SizeType32 vocabSizePadded, SizeType32 maxDraftTokens, bool randomThreshold,
    float constantThreshold, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::speculative_decoding
