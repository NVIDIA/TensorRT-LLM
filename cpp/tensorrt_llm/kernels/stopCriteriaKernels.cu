/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/stopCriteriaKernels.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace kernels
{

namespace
{
__global__ void stopWordsCriterion(TokenIdType const** outputIds, SizeType32 const** parentIds,
    TokenIdType const* const* stopWords, FinishedState* finished, SizeType32* sequenceLengths,
    SizeType32 const* batchSlots, SizeType32 const* stopWordsLens, SizeType32* numNewTokens, SizeType32 batchSize,
    SizeType32 beamWidth, SizeType32 maxSeqLen)
{
    auto const id = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    auto const batchIdx = blockIdx.y / beamWidth;
    auto const beamIdx = blockIdx.y % beamWidth;
    auto const batchSlot = batchSlots[batchIdx];
    auto const batchBeamIdx = batchSlot * beamWidth + beamIdx;
    auto const newTokens = numNewTokens ? numNewTokens[batchSlot] : 1;

    auto const* baseStopWords = stopWords[batchSlot];
    auto const stopWordsLen = stopWordsLens[batchSlot];
    auto const* baseOffsets = baseStopWords + stopWordsLen;

    if (id >= stopWordsLen || baseOffsets[id] < 0)
    {
        return;
    }

    auto const itemEnd = baseOffsets[id];
    auto const itemStart = (id > 0) ? baseOffsets[id - 1] : 0;
    auto const itemSize = itemEnd - itemStart;

    // The single-token case unconditionally bans the token
    bool shouldStop = false;
    SizeType32 stopLen = INT_MAX;
    SizeType32 step = 0;

    for (; step < newTokens; ++step)
    {
        // Need to minus newTokens because the sequenceLengths is already updated in this point
        auto const currentStep = sequenceLengths[batchBeamIdx] - newTokens + step;
        // Is sequence larger than stop word to look for a match?
        if (currentStep + 1 >= itemSize)
        {
            shouldStop = true;
            stopLen = currentStep + 1;
            auto parentId = static_cast<SizeType32>(beamIdx);
            bool const gatherBeam = beamWidth > 1;

            // Start from the last token
            for (auto tokenIdx = itemSize - 1; tokenIdx >= 0; tokenIdx--)
            {
                auto const previousToken
                    = outputIds[batchSlot][parentId * maxSeqLen + currentStep - (itemSize - 1) + tokenIdx];
                // If token does not match already, stop comparison
                if (previousToken != baseStopWords[itemStart + tokenIdx])
                {
                    shouldStop = false;
                    break;
                }
                if (gatherBeam)
                {
                    parentId = parentIds == nullptr
                        ? SizeType32{0}
                        : parentIds[batchSlot][parentId * maxSeqLen + currentStep - (itemSize - 1) + tokenIdx];

                    if (parentId < 0 || parentId >= beamWidth)
                    {
                        shouldStop = false;
                        break;
                    }
                }
            }
        }
        if (shouldStop)
        {
            finished[batchSlot * beamWidth + beamIdx].setFinishedStopWords();
            // When more than 1 token is predicted per step, find the first match with the stop word
            if (newTokens > 1)
            {
                // Update num of new tokens up to stopped word (including).
                atomicMin(numNewTokens + batchSlot, step + 1);
                // Update seq lengths up to stopped word (including).
                atomicMin(sequenceLengths + batchBeamIdx, stopLen);
            }
            break;
        }
    }
}
} // namespace

void invokeStopWordsCriterion(TokenIdType const** outputIds, SizeType32 const** parentIds,
    TokenIdType const* const* stopWords, FinishedState* finished, SizeType32* sequenceLengths,
    SizeType32 const* batchSlots, SizeType32 const* stopWordsLen, SizeType32* numNewTokens, SizeType32 maxStopWordsLen,
    SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSeqLen, cudaStream_t stream)
{
    // Check if we have sampled a word from the stopWords list. If so, stop the sequence.
    dim3 block, grid;
    constexpr SizeType32 maxBlockSize{256};

    block.x = min(((maxStopWordsLen + 32 - 1) / 32) * 32, maxBlockSize);
    grid.x = (maxStopWordsLen + block.x - 1) / block.x;
    grid.y = batchSize * beamWidth;

    stopWordsCriterion<<<grid, block, 0, stream>>>(outputIds, parentIds, stopWords, finished, sequenceLengths,
        batchSlots, stopWordsLen, numNewTokens, batchSize, beamWidth, maxSeqLen);
    sync_check_cuda_error(stream);
}

__global__ void lengthCriterion(FinishedState* finished, SizeType32* finishedSum, SizeType32 const* sequenceLimitLength,
    SizeType32* sequenceLengths, SizeType32* numNewTokens, SizeType32 const* batchSlots, SizeType32 batchSize,
    SizeType32 beamWidth)
{
    SizeType32 threadFinishedCount = 0;
    auto const batchIdx = blockIdx.x;
    auto const batchSlot = batchSlots[batchIdx];

    for (auto beamIdx = static_cast<SizeType32>(threadIdx.x); beamIdx < beamWidth;
         beamIdx += static_cast<SizeType32>(blockDim.x))
    {
        auto const batchSlotBeamWidthIdx = batchSlot * beamWidth + beamIdx;

        auto finishState = finished[batchSlotBeamWidthIdx];

        auto const numTokensToLimit = sequenceLimitLength[batchSlot] - sequenceLengths[batchSlotBeamWidthIdx];
        if (numTokensToLimit <= 0)
        {
            finishState.setFinishedMaxLength();
            sequenceLengths[batchSlotBeamWidthIdx] = sequenceLimitLength[batchSlot];
            if (numNewTokens)
            {
                numNewTokens[batchSlot] = numNewTokens[batchSlot] + numTokensToLimit;
            }
        }
        threadFinishedCount += finishState.isFinished() ? 1 : 0;
        finished[batchSlotBeamWidthIdx] = finishState;
    }

    if (finishedSum)
    {
        SizeType32 blockFinishedCount = 0;
        if (blockDim.x <= 32)
        {
            blockFinishedCount = warpReduceSum(threadFinishedCount);
        }
        else
        {
            blockFinishedCount = blockReduceSum(threadFinishedCount);
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            finishedSum[batchSlot] = blockFinishedCount;
        }
    }
}

void invokeLengthCriterion(FinishedState* finished, SizeType32* finishedSum, SizeType32 const* sequenceLimitLength,
    SizeType32* sequenceLengths, SizeType32* numNewTokens, SizeType32 const* batchSlots, SizeType32 batchSize,
    SizeType32 beamWidth, cudaStream_t stream)
{
    // Check if we have attained the sequence length limit. If so, stop the
    // sequence. In addition, check if all sequences are stopped and return the
    // result in shouldStop
    dim3 block{min(512, static_cast<uint32_t>(beamWidth))};
    dim3 grid{static_cast<uint32_t>(batchSize)};

    lengthCriterion<<<grid, block, 0, stream>>>(
        finished, finishedSum, sequenceLimitLength, sequenceLengths, numNewTokens, batchSlots, batchSize, beamWidth);
    sync_check_cuda_error(stream);
}

__global__ void explicitEOSCriterion(TokenIdType const** outputIds, TokenIdType const* endIds, FinishedState* finished,
    SizeType32* sequenceLengths, SizeType32* numNewTokens, SizeType32 const* batchSlots, SizeType32 batchSize,
    SizeType32 maxTokensPerStep)
{
    auto const batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize)
    {
        return;
    }

    auto const batchSlot = batchSlots[batchIdx];
    if (finished[batchSlot].isFinished())
    {
        return;
    }

    auto const numTokens = numNewTokens != nullptr ? numNewTokens[batchSlot] : maxTokensPerStep;
    auto const endId = endIds[batchSlot];
    auto const sequenceLength = sequenceLengths[batchSlot];

    auto const posStart = max(0, sequenceLength - numTokens);
    auto const posEnd = sequenceLength;
    for (SizeType32 pos = posStart; pos < posEnd; ++pos)
    {
        auto const token = outputIds[batchSlot][pos];
        if (token == endId)
        {
            finished[batchSlot].setFinishedEOS();
            sequenceLengths[batchSlot] = max(0, pos);
            if (numNewTokens)
            {
                numNewTokens[batchSlot] = pos - posStart;
            }
            return;
        }
    }
}

void invokeExplicitEOSCriterion(TokenIdType const** outputIds, TokenIdType const* endIds, FinishedState* finished,
    SizeType32* sequenceLengths, SizeType32* numNewTokens, SizeType32 const* batchSlots, SizeType32 batchSize,
    SizeType32 beamWidth, SizeType32 maxTokensPerStep, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(beamWidth == 1, "Explicit EOS criterion does not support beam search");
    // Check if we have sampled an end id token. If so, stop the sequence.
    SizeType32 constexpr blockSize{256};

    dim3 grid;
    grid.x = divUp(batchSize, blockSize);

    explicitEOSCriterion<<<grid, blockSize, 0, stream>>>(
        outputIds, endIds, finished, sequenceLengths, numNewTokens, batchSlots, batchSize, maxTokensPerStep);
    sync_check_cuda_error(stream);
}

} // namespace kernels
} // namespace tensorrt_llm
