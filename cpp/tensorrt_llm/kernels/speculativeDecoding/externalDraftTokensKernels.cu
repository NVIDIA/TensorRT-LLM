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

template <typename T>
__global__ void maskTargetLogitsKernel(T* targetLogits, SizeType32 const* batchSlots, SizeType32 beamWidth,
    SizeType32 vocabSize, FinishedState const* finishedInput, SizeType32 maxBatchSize,
    SizeType32* outputIdsAfterSampling, SizeType32* runtimeTopKDevicePtr, bool* maskBuffer)
{
    /**
     * @brief Masking the selected token to -inf as was done in Huggingface TopK/TopP Logits Warper
     * https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/generation/logits_process.py#L533
     */

    auto const bid = blockIdx.x;
    auto const batchIdx = bid / beamWidth;
    auto const tid = static_cast<SizeType32>(threadIdx.x);
    auto const batchSlot = batchSlots[batchIdx];

    constexpr bool IS_HALF = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_HALF) ? HALF_FLT_MAX : FLT_MAX;

    auto targetLogitsBatch = targetLogits + batchIdx * vocabSize;
    auto& finishedState = finishedInput[batchSlot];

    auto* outputIdsAfterSamplingPtr = outputIdsAfterSampling + batchSlot * vocabSize;
    auto* maskBufferBatch = maskBuffer + batchSlot * vocabSize;

    if (finishedState.isSkipDecoding() || finishedState.isFinished())
    {
        return;
    }

    __shared__ SizeType32 tokensToMask;

    if (tid == 0)
    {
        tokensToMask = runtimeTopKDevicePtr[batchSlot];
    }
    __syncthreads();

    for (SizeType32 vIdx = tid; vIdx < vocabSize; vIdx += static_cast<SizeType32>(blockDim.x))
    {
        if (outputIdsAfterSamplingPtr[vIdx] == -1)
        { // we need to find the -1 boundary from returnAllTopP outputIds if topK == 0 or number of topP indices < topK
            tokensToMask = vIdx;
        }
        maskBufferBatch[vIdx] = false;
    }

    __syncthreads();
    if (tid == 0 && tokensToMask == 0)
    {
        // all tokens are selected if topK == 0 && topP ~= 1.0f
        // in this case tokensToMask = vocabSize
        tokensToMask = vocabSize;
    }
    __syncthreads();

    for (SizeType32 vIdx = tid; vIdx < tokensToMask; vIdx += static_cast<SizeType32>(blockDim.x))
    {
        auto tokenToMask = outputIdsAfterSamplingPtr[vIdx];
        maskBufferBatch[tokenToMask] = true;
    }

    __syncthreads();

    for (SizeType32 vIdx = tid; vIdx < vocabSize; vIdx += static_cast<SizeType32>(blockDim.x))
    {
        if (!maskBufferBatch[vIdx])
        {
            targetLogitsBatch[vIdx] = -MAX_T_VAL;
        }
    }
}

template <typename T>
__global__ void acceptDraftTokensKernel(T const* draftProbs, T* targetProbs, SizeType32 const* numsDraftTokens,
    bool const* batchUseDraftLogits, TokenIdType const* draftIds, FinishedState const* finishedInput,
    FinishedState* finishedOutput, curandState_t* curandState, SizeType32 const* batchSlots, SizeType32 maxDraftTokens,
    SizeType32 beamWidth, SizeType32 vocabSize, bool randomThreshold, float constantThreshold, SizeType32 step,
    bool* batchIsAccepted, SizeType32* targetOutputIds)
{
    auto const bid = blockIdx.x;
    auto const draftTokenIdx = step;
    auto const batchIdx = bid / beamWidth;
    auto const beamIdx = bid % beamWidth;
    auto const batchSlot = batchSlots[batchIdx];
    auto const batchSlotBeamWidth = batchSlot * beamWidth + beamIdx;
    auto const tid = static_cast<SizeType32>(threadIdx.x);

    auto const numDraftTokens = numsDraftTokens[batchSlotBeamWidth];
    auto const useDraftLogits = batchUseDraftLogits[batchSlotBeamWidth];

    if (numDraftTokens == 0 || draftTokenIdx > numDraftTokens || finishedInput[batchSlot].isSkipDecoding()
        || finishedInput[batchSlot].isFinished())
    {
        if (tid == 0)
        {
            batchIsAccepted[batchSlot] = true;

            // either finished or skip decode in previous step, this step don't need decoding
            finishedOutput[batchSlot].setSkipDecoding();

            // if previous step is finished, write the state to next step too
            if (finishedInput[batchSlot].isFinished())
            {
                finishedOutput[batchSlot] = finishedInput[batchSlot];
            }
        }
        return;
    }

    if (draftTokenIdx == numDraftTokens)
    {
        if (tid == 0)
        {
            batchIsAccepted[batchSlot] = false;
            finishedOutput[batchSlot].setSkipDecoding();
        }
        return;
    }
    // else (draftTokenIdx < numDraftTokens)

    auto const logitsOffset = (batchSlot * maxDraftTokens + draftTokenIdx) * beamWidth * vocabSize;
    auto const draftProbsBatch = draftProbs + logitsOffset;
    auto const targetProbsBatch = targetProbs + (batchIdx * beamWidth * vocabSize);

    __shared__ bool isAccepted;
    __shared__ T sSumVal;
    if (tid == 0)
    {
        auto const draftOutputTokenId = draftIds[batchSlot * maxDraftTokens + draftTokenIdx];
        if (useDraftLogits)
        {
            float threshold = randomThreshold ? curand_uniform(curandState + batchSlot) : constantThreshold;
            auto const targetProb = static_cast<float>(targetProbsBatch[draftOutputTokenId]);
            auto const draftProb = static_cast<float>(draftProbsBatch[draftOutputTokenId]);
            isAccepted = targetProb > threshold * draftProb;
        }
        else
        {
            // Check if draft tokens are the same as target tokens
            isAccepted = targetOutputIds[batchSlot] == draftOutputTokenId;
        }
        if (!isAccepted)
        {
            finishedOutput[batchSlot].setSkipDecoding();
        }
        batchIsAccepted[batchSlot] = isAccepted;
    }

    __syncthreads();

    if (useDraftLogits && !isAccepted)
    {
        // correct target distribution
        T const zeroVal = static_cast<T>(0.0F);
        T sumVal = zeroVal;
        for (SizeType32 vIdx = tid; vIdx < vocabSize; vIdx += static_cast<SizeType32>(blockDim.x))
        {
            targetProbsBatch[vIdx] -= draftProbsBatch[vIdx];
            targetProbsBatch[vIdx] = targetProbsBatch[vIdx] >= zeroVal ? targetProbsBatch[vIdx] : zeroVal;
            sumVal += targetProbsBatch[vIdx];
        }
        sumVal = blockReduceSum<T>(sumVal);
        if (tid == 0)
        {
            sSumVal = sumVal;
        }
        __syncthreads();

        for (SizeType32 vIdx = tid; vIdx < vocabSize; vIdx += static_cast<SizeType32>(blockDim.x))
        {
            targetProbsBatch[vIdx] /= sSumVal;
        }
    }
}

__global__ void forwardAcceptedTokensKernel(SizeType32 batchSize, SizeType32 const* batchSlots, bool* batchIsAccepted,
    SizeType32* sequenceLengths, TokenIdType const* draftIds, TokenIdType** idsPtrs, SizeType32 step,
    SizeType32 maxDraftTokens, TokenIdType const* endIds, FinishedState* finishedOutput)
{
    auto index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    for (SizeType32 bi = index; bi < batchSize; bi += static_cast<SizeType32>(gridDim.x * blockDim.x))
    {
        auto const batchSlot = batchSlots[bi];
        if (batchIsAccepted[batchSlot] && !finishedOutput[batchSlot].isSkipDecoding()
            && !finishedOutput[batchSlot].isFinished())
        {
            auto const curSeqLen = sequenceLengths[batchSlot];
            auto const draftTokenIdx = step;
            auto const draftOutputTokenId = draftIds[batchSlot * maxDraftTokens + draftTokenIdx];
            auto* outputIdsRequestPtr = idsPtrs[batchSlot];
            auto const outIdx = curSeqLen;
            outputIdsRequestPtr[outIdx] = draftOutputTokenId;
            if (outputIdsRequestPtr[outIdx] == endIds[batchSlot])
            {
                finishedOutput[batchSlot].setFinishedEOS();
                // Do not increase seq len when EOS is generated. Seq len should always contain only tokens to be
                // outputted
            }
            else
            {
                // We don't need to set output finished state as it is assumed to be in non finished state
                sequenceLengths[batchSlot] += 1;
            }
        }
    }
} // namespace

} // namespace

template <typename T>
void invokeMaskTargetLogits(SizeType32 batchSize, T* targetLogits, SizeType32 const* batchSlots, SizeType32 beamWidth,
    SizeType32 vocabSizePadded, FinishedState const* finishedInput, SizeType32 maxBatchSize,
    SizeType32* outputIdsAfterSampling, SizeType32* runtimeTopKDevicePtr, bool* maskBuffer, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(beamWidth == 1);
    {
        dim3 block(1024);
        dim3 grid(batchSize * beamWidth);
        maskTargetLogitsKernel<<<grid, block, 0, stream>>>(targetLogits, batchSlots, beamWidth, vocabSizePadded,
            finishedInput, maxBatchSize, outputIdsAfterSampling, runtimeTopKDevicePtr, maskBuffer);
    }
    sync_check_cuda_error(stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void invokeAcceptDraftTokens(SizeType32 batchSize, T* draftProbs, T* targetProbs, SizeType32 const* numsDraftTokens,
    bool const* batchUseDraftLogits, TokenIdType const* draftIds, FinishedState const* finishedInput,
    FinishedState* finishedOutput, curandState_t* curandState, SizeType32 const* batchSlots, SizeType32 maxDraftTokens,
    SizeType32 beamWidth, SizeType32 vocabSizePadded, bool randomThreshold, float constantThreshold, SizeType32 step,
    bool* batchIsAccepted, SizeType32* targetOutputIds, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(beamWidth == 1);
    {
        dim3 block(1024);
        dim3 grid(batchSize * beamWidth);
        acceptDraftTokensKernel<<<grid, block, 0, stream>>>(draftProbs, targetProbs, numsDraftTokens,
            batchUseDraftLogits, draftIds, finishedInput, finishedOutput, curandState, batchSlots, maxDraftTokens,
            beamWidth, vocabSizePadded, randomThreshold, constantThreshold, step, batchIsAccepted, targetOutputIds);
    }
    sync_check_cuda_error(stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template void invokeMaskTargetLogits(SizeType32 batchSize, float* targetLogits, SizeType32 const* batchSlots,
    SizeType32 beamWidth, SizeType32 vocabSizePadded, FinishedState const* finishedInput, SizeType32 maxBatchSize,
    SizeType32* outputIdsAfterSampling, SizeType32* runtimeTopKDevicePtr, bool* maskBuffer, cudaStream_t stream);
template void invokeMaskTargetLogits(SizeType32 batchSize, half* targetLogits, SizeType32 const* batchSlots,
    SizeType32 beamWidth, SizeType32 vocabSizePadded, FinishedState const* finishedInput, SizeType32 maxBatchSize,
    SizeType32* outputIdsAfterSampling, SizeType32* runtimeTopKDevicePtr, bool* maskBuffer, cudaStream_t stream);

template void invokeAcceptDraftTokens(SizeType32 batchSize, float* draftProbs, float* targetProbs,
    SizeType32 const* numsDraftTokens, bool const* batchUseDraftLogits, TokenIdType const* draftIds,
    FinishedState const* finishedInput, FinishedState* finishedOutput, curandState_t* curandState,
    SizeType32 const* batchSlots, SizeType32 maxDraftTokens, SizeType32 beamWidth, SizeType32 vocabSizePadded,
    bool randomThreshold, float constantThreshold, SizeType32 step, bool* batchIsAccepted, SizeType32* targetOutputIds,
    cudaStream_t stream);
template void invokeAcceptDraftTokens(SizeType32 batchSize, half* draftProbs, half* targetProbs,
    SizeType32 const* numsDraftTokens, bool const* batchUseDraftLogits, TokenIdType const* draftIds,
    FinishedState const* finishedInput, FinishedState* finishedOutput, curandState_t* curandState,
    SizeType32 const* batchSlots, SizeType32 maxDraftTokens, SizeType32 beamWidth, SizeType32 vocabSizePadded,
    bool randomThreshold, float constantThreshold, SizeType32 step, bool* batchIsAccepted, SizeType32* targetOutputIds,
    cudaStream_t stream);

void invokeForwardAcceptedTokens(SizeType32 batchSize, SizeType32 const* batchSlots, bool* batchIsAccepted,
    SizeType32* outputSequenceLengths, TokenIdType const* draftIds, TokenIdType** idsPtrs, SizeType32 step,
    SizeType32 maxDraftTokens, TokenIdType const* endIds, FinishedState* finishedOutput, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
    dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
    forwardAcceptedTokensKernel<<<grid, block, 0, stream>>>(batchSize, batchSlots, batchIsAccepted,
        outputSequenceLengths, draftIds, idsPtrs, step, maxDraftTokens, endIds, finishedOutput);
    sync_check_cuda_error(stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
} // namespace tensorrt_llm::kernels::speculative_decoding
