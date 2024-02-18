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

namespace tensorrt_llm
{
namespace kernels
{
__global__ void stopWordsCriterion(int32_t const** outputIds, int32_t const** parentIds, int32_t const** stopWords,
    FinishedState* finished, int32_t const* sequenceLengths, int32_t const* batchSlots, int32_t const* stopWordsLens,
    int32_t batchSize, int32_t beamWidth, int32_t maxSeqLen)
{
    int32_t const id = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const batchIdx = blockIdx.y / beamWidth;
    int32_t const beamIdx = blockIdx.y % beamWidth;
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;
    auto const batchBeamIdx = batchSlot * beamWidth + beamIdx;

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

    // Need to minus 1 because the sequenceLengths is updated in this step
    auto const currentStep = sequenceLengths[batchBeamIdx] - 1;
    // Enough previously generated tokens to look for a match
    if (currentStep + 1 >= itemSize)
    {
        shouldStop = true;
        auto parentId = beamIdx;
        bool const gatherBeam = beamWidth > 1;

        for (int32_t tokenIdx = itemSize - 1; tokenIdx >= 0; tokenIdx--)
        {
            auto const previousToken
                = outputIds[batchSlot][parentId * maxSeqLen + currentStep - (itemSize - 1) + tokenIdx];
            if (previousToken != baseStopWords[itemStart + tokenIdx])
            {
                shouldStop = false;
                break;
            }
            if (gatherBeam)
            {
                parentId = parentIds == nullptr
                    ? 0
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
    }
}

void invokeStopWordsCriterion(int32_t const** outputIds, int32_t const** parentIds, int32_t const** stopWords,
    FinishedState* finished, int32_t const* sequenceLengths, int32_t const* batchSlots, int32_t const* stopWordsLen,
    int32_t maxStopWordsLen, int32_t batchSize, int32_t beamWidth, int32_t maxSeqLen, cudaStream_t stream)
{
    // Check if we have sampled a word from the stopWords list. If so, stop the sequence.
    dim3 block, grid;
    constexpr int32_t maxBlockSize{256};

    block.x = min(((maxStopWordsLen + 32 - 1) / 32) * 32, maxBlockSize);
    grid.x = (maxStopWordsLen + block.x - 1) / block.x;
    grid.y = batchSize * beamWidth;

    stopWordsCriterion<<<grid, block, 0, stream>>>(outputIds, parentIds, stopWords, finished, sequenceLengths,
        batchSlots, stopWordsLen, batchSize, beamWidth, maxSeqLen);
    sync_check_cuda_error();
}

__global__ void lengthCriterion(FinishedState* finished, int32_t* finishedSum, uint32_t const* sequenceLimitLength,
    int32_t const* sequenceLengths, int32_t const* batchSlots, int32_t batchSize, int32_t beamWidth)
{
    int32_t threadFinishedCount = 0;
    auto const batchIdx = blockIdx.x;
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;

    for (int32_t beamIdx = threadIdx.x; beamIdx < beamWidth; beamIdx += blockDim.x)
    {
        auto const batchSlotBeamWidthIdx = batchSlot * beamWidth + beamIdx;

        auto finishState = finished[batchSlotBeamWidthIdx];

        if (sequenceLengths[batchSlotBeamWidthIdx] >= sequenceLimitLength[batchSlot])
        {
            finishState.setFinishedMaxLength();
        }
        threadFinishedCount += finishState.isFinished() ? 1 : 0;
        finished[batchSlotBeamWidthIdx] = finishState;
    }

    if (finishedSum)
    {
        int blockFinishedCount = 0;
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

void invokeLengthCriterion(FinishedState* finished, int32_t* finishedSum, uint32_t const* sequenceLimitLength,
    int32_t const* sequenceLengths, int32_t const* batchSlots, int32_t batchSize, int32_t beamWidth,
    cudaStream_t stream)
{
    // Check if we have attained the sequence length limit. If so, stop the
    // sequence. In addition, check if all sequences are stopped and return the
    // result in shouldStop
    dim3 block{min(512, uint32_t(beamWidth))};
    dim3 grid{uint32_t(batchSize)};

    lengthCriterion<<<grid, block, 0, stream>>>(
        finished, finishedSum, sequenceLimitLength, sequenceLengths, batchSlots, batchSize, beamWidth);
    sync_check_cuda_error();
}

} // namespace kernels
} // namespace tensorrt_llm
