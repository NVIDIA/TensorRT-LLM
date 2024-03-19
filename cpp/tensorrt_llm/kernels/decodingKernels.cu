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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingKernels.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{

namespace kernels
{

__global__ void gatherTree(gatherTreeParam param)
{
    for (int batchbeamIdx = blockIdx.x * blockDim.x + threadIdx.x; batchbeamIdx < param.batchSize * param.beamWidth;
         batchbeamIdx += gridDim.x * blockDim.x)
    {
        int const batch = batchbeamIdx / param.beamWidth;
        int const beam = batchbeamIdx % param.beamWidth;
        int const inputLen = param.inputLengths == nullptr ? 0 : param.inputLengths[batchbeamIdx];

        int const* parentIds = param.parentIds;
        int const* stepIds = param.stepIds;

        // TODO optimize the reduce_max operation for large beamWidth
        int maxLen = -1;
        bool updateResponseInputLength = param.responseInputLengths != nullptr;
        // int selected_beam_index = 0;
        for (int beamIdx = 0; beamIdx < param.beamWidth; beamIdx++)
        {
            int tmpLen
                = param.sequenceLengths[batch * param.beamWidth + beamIdx] + param.maxSequenceLengthFinalStep - 1;
            param.sequenceLengths[batch * param.beamWidth + beamIdx] = tmpLen;
            if (updateResponseInputLength)
            {
                param.responseInputLengths[batch * param.beamWidth + beamIdx] = inputLen;
            }
            if (tmpLen > maxLen)
            {
                maxLen = tmpLen;
            }
        }
        int const maxSeqLenB = min(param.maxSeqLen, maxLen);
        if (maxSeqLenB <= 0)
        {
            continue;
        }

        int const initialTgtIx = batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + maxSeqLenB - 1;
        int const initialParentIx = batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + maxSeqLenB - 1;
        param.outputIds[initialTgtIx] = __ldg(stepIds + initialParentIx);
        int parent = parentIds == nullptr ? 0 : __ldg(parentIds + initialParentIx) % param.beamWidth;
        bool foundBad = false;

        for (int level = maxSeqLenB - 2; level >= 0; --level)
        {
            int const levelBeamIx = batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + level;
            int const levelParentIx = batch * param.beamWidth * param.maxSeqLen + parent * param.maxSeqLen + level;
            if (parent < 0 || parent > param.beamWidth)
            {
                param.outputIds[levelBeamIx] = param.endTokens[batch];
                parent = -1;
                foundBad = true;
            }
            else
            {
                param.outputIds[levelBeamIx] = __ldg(stepIds + levelParentIx);
                parent = parentIds == nullptr ? 0 : __ldg(parentIds + levelParentIx) % param.beamWidth;
            }
        }
        // set the padded part as end_token
        // inputLen
        for (int index = maxLen; index < param.maxSeqLen; ++index)
        {
            param.outputIds[batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + index]
                = param.endTokens[batch];
        }

        // Not necessary when using a BeamSearchDecoder, but necessary
        // when a user feeds in possibly broken trajectory (i.e., non-eos
        // entries in a beam following eos entries).
        if (!foundBad)
        {
            bool finished = false;
            // skip the step 0 because it is often the start token
            int startStep = 1;
            for (int time = startStep; time < maxSeqLenB; ++time)
            {
                int const levelBeamIx = batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + time;
                if (finished)
                {
                    param.outputIds[levelBeamIx] = param.endTokens[batch];
                }
                else if (param.outputIds[levelBeamIx] == param.endTokens[batch])
                {
                    finished = true;
                }
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ T applyLengthPenalty(T logProb, int length, float lengthPenalty)
{
    // score = log(prob) / (length ^ lengthPenalty)
    if (lengthPenalty == 0.0f || length == 1)
    {
        return logProb;
    }
    return logProb / static_cast<T>(powf(length, lengthPenalty));
}

struct RankNorm
{
    int rank;
    float norm;
};

inline __device__ RankNorm swap(RankNorm const& rankNorm, int mask, int dir)
{
    // Exchange the rank and norm inside the warp.
    RankNorm other;
    other.rank = __shfl_xor_sync(unsigned(-1), rankNorm.rank, mask);
    other.norm = __shfl_xor_sync(unsigned(-1), rankNorm.norm, mask);

    // Update the sorted values.
    bool doSwap = (rankNorm.norm != other.norm) && ((rankNorm.norm > other.norm) == dir);
    RankNorm res;
    res.rank = doSwap ? other.rank : rankNorm.rank;
    res.norm = doSwap ? other.norm : rankNorm.norm;

    return res;
}

inline __device__ uint32_t bfe(uint32_t a, uint32_t start, uint32_t len = 1)
{
    uint32_t d;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(start), "r"(len));
    return d;
}

__global__ void finalized(gatherTreeParam param)
{
    int const beamIdx = static_cast<int>(threadIdx.x);
    int const beamWidth{param.beamWidth};

    extern __shared__ char array[];
    int* sRank = (int*) (array);
    int* sLength = (int*) (sRank + beamWidth);
    float* sScores = (float*) (sLength + beamWidth);
    float* sNormedScores = (float*) (sScores + beamWidth);
    int* sIds = (int*) (sNormedScores + beamWidth);

    if (beamIdx < beamWidth)
    {
        int const idx = blockIdx.x * param.beamWidth + beamIdx;
        int const numGeneratedToken{param.sequenceLengths[idx] - param.inputLengths[idx]};
        sNormedScores[beamIdx] = applyLengthPenalty(param.cumLogProbs[idx], numGeneratedToken, param.lengthPenalty);
        sLength[beamIdx] = param.sequenceLengths[idx];
        sScores[beamIdx] = param.cumLogProbs[idx];
    }
    for (int idx = beamIdx; idx < beamWidth * param.maxSeqLen; idx += blockDim.x)
    {
        sIds[idx] = param.outputIds[blockIdx.x * param.beamWidth * param.maxSeqLen + idx];
    }
    __syncthreads();

    RankNorm rankNorm;
    rankNorm.rank = beamIdx;
    rankNorm.norm = beamIdx < beamWidth ? sNormedScores[beamIdx] : -FLT_MAX;

    if (beamWidth < 32)
    {
        int warpid = threadIdx.x / 32;
        int laneid = threadIdx.x % 32;

        if (warpid == 0 && beamWidth > 1)
        {
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0)); //  2
        }

        if (warpid == 0 && beamWidth > 2)
        {
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1)); //  3~4
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && beamWidth > 4)
        {
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2)); //  5~8
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 3) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 3) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && beamWidth > 8)
        {
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3)); // 9~16
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && beamWidth > 16)
        {
            rankNorm = swap(rankNorm, 0x10, bfe(laneid, 4) ^ bfe(laneid, 4)); // 17~32
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3));
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }
    }
    else
    {
        // Not supported! We must have a check before calling that kernel.
    }

    if (beamIdx < beamWidth)
    {
        sRank[beamIdx] = rankNorm.rank;
    }

    __syncthreads();

    if (beamIdx < beamWidth)
    {
        auto srcIdx{rankNorm.rank};
        auto tgtIdx{blockIdx.x * param.beamWidth + beamIdx};
        param.sequenceLengths[tgtIdx] = sLength[srcIdx];
        param.cumLogProbs[tgtIdx] = sScores[srcIdx];
    }

    for (int beamIdx = 0; beamIdx < beamWidth; beamIdx++)
    {
        for (int i = threadIdx.x; i < sLength[sRank[beamIdx]]; i += blockDim.x)
        {
            param.outputIds[blockIdx.x * beamWidth * param.maxSeqLen + beamIdx * param.maxSeqLen + i]
                = sIds[sRank[beamIdx] * param.maxSeqLen + i];
        }
    }
}

void invokeGatherTree(gatherTreeParam param)
{
    int batchbeam = param.batchSize * param.beamWidth;
    dim3 grid(1), block(batchbeam);
    // though decoder do not support > 1024 for now
    if (batchbeam > 1024)
    {
        grid.x = ceil(param.batchSize * param.beamWidth / 1024.);
        block.x = 1024;
    }
    gatherTree<<<grid, block, 0, param.stream>>>(param);
    sync_check_cuda_error();

    if (param.beamWidth > 1)
    {
        TLLM_CHECK_WITH_INFO(param.beamWidth <= 32, "TRT-LLM does not support beam width > 32 now");
        // sort results by normalized cumLogProbs
        dim3 grid(param.batchSize);
        dim3 block(divUp(param.beamWidth, 32) * 32);

        auto shm_size = param.beamWidth * (sizeof(float) * 2 + sizeof(int) * 2 + sizeof(int) * param.maxSeqLen);
        finalized<<<grid, block, shm_size, param.stream>>>(param);
    }
}

__global__ void finalize(int* outputIds, int* sequenceLengths, float* cumLogProbs, float* outputLogProbs,
    int const* topKOutputIds, int const* topKSequenceLengths, float const* scores, float const* topKCumLogProbs,
    float const* topKLogProbs, int const* numBeams, int const* inputLengths, int const beamWidth, int const maxSeqLen)
{
    // outputIds: [bs, beamWidth, maxSeqLen]
    // sequenceLengths: [bs, beamWidth]
    // cumLogProbs: [bs, beamWidth]
    // outputLogProbs: [bs, beamWidth, maxSeqLen]
    // topKOutputIds: [bs, 2 * beamWidth, maxSeqLen + 1]
    // topKSequenceLengths: [bs, 2 * beamWidth]
    // scores: [bs, 2 * beamWidth]
    // topKCumLogProbs: [bs, 2 * beamWidth]
    // topKLogProbs: [bs, 2 * beamWidth, maxSeqLen + 1]
    // numBeams: [bs]

    // This kernel do a sorting for scores first, and then put the topKOutputIds
    // into outputIds by the rank of scores.
    // Note that we remove the start_token (the id at first position) from topKOutputIds

    extern __shared__ char array[];
    int* sRank = (int*) (array);                              // [beamWidth]
    float* sScores = (float*) (sRank + beamWidth);            // [2 * beamWidth]
    int* sSequenceLengths = (int*) (sScores + beamWidth * 2); // [beamWidth]
    int const numBeam = numBeams[blockIdx.x];
    if (threadIdx.x < numBeam)
    {
        sScores[threadIdx.x] = scores[blockIdx.x * beamWidth * 2 + threadIdx.x];
    }
    __syncthreads();

    if (numBeam < 32)
    {
        int const beamIdx = threadIdx.x;
        RankNorm rankNorm;
        rankNorm.rank = beamIdx;
        rankNorm.norm = beamIdx < numBeam ? sScores[beamIdx] : -FLT_MAX;

        int warpid = threadIdx.x / 32;
        int laneid = threadIdx.x % 32;

        if (warpid == 0 && numBeam > 1)
        {
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0)); //  2
        }

        if (warpid == 0 && numBeam > 2)
        {
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1)); //  3~4
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && numBeam > 4)
        {
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2)); //  5~8
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 3) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 3) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && numBeam > 8)
        {
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3)); // 9~16
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && numBeam > 16)
        {
            rankNorm = swap(rankNorm, 0x10, bfe(laneid, 4) ^ bfe(laneid, 4)); // 17~32
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3));
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }

        if (beamIdx < beamWidth)
        {
            sRank[beamIdx] = rankNorm.rank;
        }

        __syncthreads();
    }
    else
    {
        for (int i = 0; i < beamWidth; i++)
        {
            float score = threadIdx.x < numBeams[blockIdx.x] ? sScores[threadIdx.x] : -FLT_MAX;
            float maxScore = blockReduceMax<float>(score);

            if (threadIdx.x == 0)
            {
                for (int j = 0; j < beamWidth * 2; j++)
                {
                    if (sScores[j] == maxScore)
                    {
                        sRank[i] = j;
                        sScores[j] = -FLT_MAX;
                        break;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x < beamWidth)
    {
        sSequenceLengths[threadIdx.x] = topKSequenceLengths[blockIdx.x * beamWidth * 2 + sRank[threadIdx.x]];
        sequenceLengths[blockIdx.x * beamWidth + threadIdx.x] = sSequenceLengths[threadIdx.x];

        if (cumLogProbs != nullptr)
        {
            cumLogProbs[blockIdx.x * beamWidth + threadIdx.x]
                = topKCumLogProbs[blockIdx.x * beamWidth * 2 + sRank[threadIdx.x]];
        }
    }
    __syncthreads();

    for (int beamIdx = 0; beamIdx < beamWidth; beamIdx++)
    {
        // start from step 1 to skip the start token
        for (int i = threadIdx.x; i < sSequenceLengths[beamIdx]; i += blockDim.x)
        {
            outputIds[blockIdx.x * beamWidth * maxSeqLen + beamIdx * maxSeqLen + i]
                = topKOutputIds[blockIdx.x * (beamWidth * 2) * maxSeqLen + sRank[beamIdx] * maxSeqLen + i];
            if (outputLogProbs != nullptr)
            {
                int inputLen = inputLengths[blockIdx.x * beamWidth + beamIdx];
                if (i >= inputLen)
                {
                    outputLogProbs[blockIdx.x * beamWidth * maxSeqLen + beamIdx * maxSeqLen + i - inputLen]
                        = topKLogProbs[blockIdx.x * (beamWidth * 2) * maxSeqLen + sRank[beamIdx] * maxSeqLen + i];
                }
            }
        }
    }
}

void invokeFinalize(int* outputIds, int* sequenceLengths, float* cumLogProbs, float* outputLogProbs,
    int const* topKOutputIds, int const* topKSequenceLengths, float const* scores, float const* topKCumLogProbs,
    float const* topKLogProbs, int const* numBeams, int const* inputLengths, int const beamWidth, int const maxSeqLen,
    int const batchSize, cudaStream_t stream)
{
    TLLM_LOG_DEBUG("%s %s start", __FILE__, __PRETTY_FUNCTION__);
    dim3 block(beamWidth * 2);
    block.x = (block.x + 31) / 32 * 32;
    TLLM_CHECK(block.x < 1024);
    finalize<<<batchSize, block, beamWidth * sizeof(int) * 2 + (beamWidth * 2) * sizeof(float), stream>>>(outputIds,
        sequenceLengths, cumLogProbs, outputLogProbs, topKOutputIds, topKSequenceLengths, scores, topKCumLogProbs,
        topKLogProbs, numBeams, inputLengths, beamWidth, maxSeqLen);
}

__global__ void initializeOutput(int* outputIds, int const* endIds, int const maxSeqLen)
{
    for (int i = threadIdx.x; i < maxSeqLen; i += blockDim.x)
    {
        outputIds[blockIdx.x * maxSeqLen + i] = endIds[blockIdx.x];
    }
}

void invokeInitializeOutput(int* outputIds, int const* endIds, int batchBeam, int maxSeqLen, cudaStream_t stream)
{
    initializeOutput<<<batchBeam, 256, 0, stream>>>(outputIds, endIds, maxSeqLen);
}

__global__ void copyNextStepIds(TokenIdType* nextStepIds, TokenIdType const* const* outputIdsPtr,
    SizeType const* sequenceLengths, SizeType const* numNewTokens, SizeType const* batchSlots, SizeType batchSize,
    SizeType maxBatchSize, SizeType beamWidth, SizeType maxSeqLen, SizeType maxTokensPerStep)
{
    for (auto index = static_cast<SizeType>(blockIdx.x * blockDim.x + threadIdx.x);
         index < batchSize * beamWidth * maxTokensPerStep; index += static_cast<SizeType>(blockDim.x * gridDim.x))
    {
        auto const batchIdx{index / (beamWidth * maxTokensPerStep)};
        auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;
        auto const remainder{index % (beamWidth * maxTokensPerStep)};
        auto const beamIdx{remainder / maxTokensPerStep};
        auto const tokenIdx{remainder % maxTokensPerStep};
        auto const newTokens = numNewTokens == nullptr ? 1 : numNewTokens[batchSlot];
        auto const batchBeamIdx = batchSlot * beamWidth + beamIdx;
        auto const tokenBatchBeamIdx = tokenIdx * maxBatchSize * beamWidth + batchSlot * beamWidth + beamIdx;
        if (tokenIdx >= newTokens)
        {
            continue;
        }
        nextStepIds[tokenBatchBeamIdx]
            = outputIdsPtr[batchSlot][beamIdx * maxSeqLen + sequenceLengths[batchBeamIdx] - newTokens + tokenIdx];
    }
}

void invokeCopyNextStepIds(TokenIdType* nextStepIds, TokenIdType const* const* outputIdsPtr,
    SizeType const* sequenceLengths, SizeType const* numNewTokens, SizeType const* batchSlots, SizeType batchSize,
    SizeType maxBatchSize, SizeType beamWidth, SizeType maxSeqLen, SizeType maxTokensPerStep, cudaStream_t stream)
{
    auto const numElems = batchSize * beamWidth * maxTokensPerStep;
    dim3 block(min(256, numElems));
    dim3 grid(divUp(numElems, block.x));
    copyNextStepIds<<<grid, block, 0, stream>>>(nextStepIds, outputIdsPtr, sequenceLengths, numNewTokens, batchSlots,
        batchSize, maxBatchSize, beamWidth, maxSeqLen, maxTokensPerStep);
}

__global__ void transposeLogProbs(float* outputLogProbs, float* outputLogProbsTiled, int const* sequenceLengths,
    int const* batchSlots, int batchSize, int maxBatchSize, int beamWidth, int maxSeqLen)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int const batchIdx = index / (beamWidth * maxSeqLen);
    int const tmpIdx = index % (beamWidth * maxSeqLen);
    int const beamIdx = tmpIdx / maxSeqLen;
    int const pos = tmpIdx % maxSeqLen;
    if (batchIdx >= batchSize)
    {
        return;
    }

    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;
    if (pos < sequenceLengths[batchSlot])
    {
        auto const batchBeamIdx = batchSlot * beamWidth * maxSeqLen + beamIdx * maxSeqLen + pos;
        outputLogProbs[batchBeamIdx]
            = outputLogProbsTiled[pos * maxBatchSize * beamWidth + batchSlot * beamWidth + beamIdx];
    }
}

void invokeTransposeLogProbs(float* outputLogProbs, float* outputLogProbsTiled, int const* sequenceLengths,
    int const* batchSlots, int batchSize, int maxBatchSize, int beamWidth, int maxSeqLen, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(divUp(batchSize * beamWidth * maxSeqLen, block.x));
    transposeLogProbs<<<grid, block, 0, stream>>>(outputLogProbs, outputLogProbsTiled, sequenceLengths, batchSlots,
        batchSize, maxBatchSize, beamWidth, maxSeqLen);
}

__global__ void acceptDraftTokensByIds(int32_t const* draftIds, int32_t const* targetIds, int32_t const* contextLengths,
    int32_t const* numsDraftTokens, int32_t* sequenceLengths, FinishedState const* finished,
    FinishedState* finishedFinal, int32_t* finishedSum, int32_t const* batchSlots, int32_t batchSize,
    int32_t maxBatchSize, int32_t maxSeqLen, int32_t maxDraftTokens)
{
    for (int batchIdx = threadIdx.x; batchIdx < batchSize; batchIdx += blockDim.x)
    {
        auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
        auto const numDraftTokens = numsDraftTokens[batchSlot];

        auto const contextLength = contextLengths[batchSlot];
        auto& sequenceLength = sequenceLengths[batchSlot];
        int finishedDraftIdx = 0;
        for (int ti = contextLength; ti < min(sequenceLength, contextLength + numDraftTokens); ++ti, ++finishedDraftIdx)
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

void invokeAcceptDraftTokensByIds(int32_t const* draftIds, int32_t const* targetIds, int32_t const* contextLengths,
    int32_t const* numsDraftTokens, int32_t* sequenceLengths, FinishedState const* finished,
    FinishedState* finishedFinal, int32_t* finishedSum, int32_t const* batchSlots, int32_t batchSize,
    int32_t maxBatchSize, int32_t beamWidth, int32_t maxSeqLen, int32_t maxDraftTokens, cudaStream_t stream)
{
    TLLM_CHECK(beamWidth == 1);
    dim3 block(min(1024, batchSize));
    dim3 grid(1);
    acceptDraftTokensByIds<<<grid, block, 0, stream>>>(draftIds, targetIds, contextLengths, numsDraftTokens,
        sequenceLengths, finished, finishedFinal, finishedSum, batchSlots, batchSize, maxBatchSize, maxSeqLen,
        maxDraftTokens);
}

template <typename T>
__global__ void acceptDraftTokensByLogitsKernel(T const* draftProbs, T* targetProbs, int32_t const* numsDraftTokens,
    FinishedState* finished, curandState_t* curandState, int32_t const* batchSlots, int32_t batchSize,
    int32_t maxBatchSize, int32_t maxDraftTokens, int32_t beamWidth, int32_t vocabSize, bool randomThreshold,
    float constantThreshold)
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

    int32_t rejected = 0;
    auto vocabSizePadded = static_cast<int32_t>((vocabSize + blockDim.x - 1) / blockDim.x) * blockDim.x;

    for (int32_t vIdx = threadIdx.x; vIdx < vocabSizePadded; vIdx += blockDim.x)
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
    int32_t const* numsDraftTokens, FinishedState* finished, int32_t const* batchSlots, int32_t batchSize,
    int32_t maxBatchSize, int32_t maxDraftTokens, int32_t beamWidth, int32_t vocabSize)
{
    auto const bid = blockIdx.x;
    auto const batchIdx = bid / beamWidth;
    auto const beamIdx = bid % beamWidth;
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    auto const batchSlotBeamWidth = batchSlot * beamWidth + beamIdx;
    auto const numDraftTokens = numsDraftTokens[batchSlotBeamWidth];

    __shared__ int32_t numAcceptedTokens;
    if (threadIdx.x == 0)
    {
        numAcceptedTokens = numDraftTokens;
        bool cummulativeSkipDecoding = false;
        for (int32_t ti = 0; ti < numDraftTokens + 1; ++ti)
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
        for (int32_t vIdx = threadIdx.x; vIdx < vocabSize; vIdx += blockDim.x)
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

        for (int32_t vIdx = threadIdx.x; vIdx < vocabSize; vIdx += blockDim.x)
        {
            auto const correctedNormProb = static_cast<float>(targetProbBatch[vIdx]) / sumProbsShared;
            targetLogitsBatch[vIdx] = __logf(correctedNormProb / (1.f - correctedNormProb));
        }
    }
}

template <typename T>
void acceptDraftTokensByLogits(T* draftLogits, T** targetLogits, T* draftProbs, T* targetProbs,
    int32_t const* numsDraftTokens, FinishedState* finished, curandState_t* curandState, int32_t const* batchSlots,
    int32_t batchSize, int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded,
    int32_t maxDraftTokens, bool randomThreshold, float constantThreshold, cudaStream_t stream)
{
    TLLM_CHECK(beamWidth == 1);
    {
        invokeAddBiasSoftMax(draftLogits, (T**) (nullptr), draftProbs, (T*) (nullptr), nullptr, finished, batchSlots,
            batchSize, maxBatchSize, beamWidth * maxDraftTokens, vocabSize, vocabSizePadded, /* skip softmax */ false,
            /* batchSlotLogits */ true, stream);
        invokeAddBiasSoftMax((T*) (nullptr), targetLogits, targetProbs, (T*) (nullptr), nullptr, finished, batchSlots,
            batchSize, maxBatchSize, beamWidth * maxDraftTokens, vocabSize, vocabSizePadded, /* skip softmax */ false,
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
    int32_t const* numsDraftTokens, FinishedState* finished, curandState_t* curandState, int32_t const* batchSlots,
    int32_t batchSize, int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded,
    int32_t maxDraftTokens, bool randomThreshold, float constantThreshold, cudaStream_t stream);
template void acceptDraftTokensByLogits(half* draftLogits, half** targetLogits, half* draftProbs, half* targetProbs,
    int32_t const* numsDraftTokens, FinishedState* finished, curandState_t* curandState, int32_t const* batchSlots,
    int32_t batchSize, int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded,
    int32_t maxDraftTokens, bool randomThreshold, float constantThreshold, cudaStream_t stream);

__device__ __forceinline__ int4 reduceMaxInt4(int4 const& a, int4 const& b)
{
    return a.x >= b.x ? a : b;
}

template <typename T, SizeType BLOCK_SIZE>
__global__ void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* targetIds,
    SizeType* sequenceLengths, SizeType* acceptedLengths, FinishedState* finishedFinal, SizeType const* batchSlots,
    SizeType const* paths, TokenIdType const* endIds, T const* medusaLogits, T const** logitsPtrs, SizeType batchSize,
    SizeType vocabSize, SizeType maxBatchSize, SizeType maxDraftSeqLen, SizeType maxTargetSeqLen, SizeType maxNumHeads,
    SizeType maxTokensPerStep)
{
    auto const batchIdx = static_cast<SizeType>(blockIdx.x);
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    auto const inputLength = sequenceLengths[batchSlot];
    auto const endId = endIds[batchSlot];
    auto const maxNumDraftTokens = maxNumHeads + 1;

    int4 partialMax{-1, -1, 0, 0};
    // Go over different paths and construct implicit sequences
    for (auto pathIdx = static_cast<SizeType>(threadIdx.x); pathIdx < maxTokensPerStep;
         pathIdx += static_cast<SizeType>(blockDim.x))
    {
        auto acceptedLength = maxNumDraftTokens;
        auto const pathOffset = flat_index3(batchSlot, pathIdx, 0, maxTokensPerStep, maxNumDraftTokens);
        bool hasEnd = false;

        auto const tokenId = paths[pathOffset];
        // Continue if path does not exist
        if (tokenId == -1)
        {
            continue;
        }
        auto const targetTokenIdx = batchSlot * maxTargetSeqLen + tokenId;
        auto targetToken = targetIds[targetTokenIdx];
        auto nextIdx = tokenId;

        // Go along the path
        for (SizeType ti = 1; ti < maxNumDraftTokens; ++ti)
        {
            auto const tokenId = paths[pathOffset + ti];
            // Break if path terminates
            if (tokenId == -1)
            {
                acceptedLength = ti;
                break;
            }
            auto const targetTokenIdx = batchSlot * maxTargetSeqLen + tokenId;
            auto const draftTokenIdx = batchSlot * maxDraftSeqLen + inputLength + tokenId;
            auto const draftToken = outputIds[draftTokenIdx];

            // Check if draft tokens are the same as target tokens
            bool const accepted = draftToken == targetToken;
            hasEnd = targetToken == endId;
            if (!accepted || hasEnd)
            {
                acceptedLength = hasEnd ? ti - 1 : ti;
                nextIdx = tokenId;
                break;
            }
            targetToken = targetIds[targetTokenIdx];
            nextIdx = tokenId;
        }
        // Get longest path of the thread
        if (partialMax.x < acceptedLength)
        {
            partialMax.x = acceptedLength;
            partialMax.y = pathIdx;
            partialMax.z = hasEnd;
            partialMax.w = nextIdx;
        }
    }

    // Get the longest path of the block (request)
    typedef cub::BlockReduce<int4, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    int4 total = BlockReduce(tempStorage).Reduce(partialMax, reduceMaxInt4);

    __shared__ int4 totalShared;
    if (threadIdx.x == 0)
    {
        totalShared = total;
    }

    __syncthreads();

    auto const acceptedLength = totalShared.x;
    auto const bestPathIdx = totalShared.y;
    auto const bestNextIdx = totalShared.w;
    auto const pathOffset = flat_index3(batchSlot, bestPathIdx, 0, maxTokensPerStep, maxNumDraftTokens);
    for (auto ti = static_cast<SizeType>(threadIdx.x); ti < acceptedLength; ti += static_cast<SizeType>(blockDim.x))
    {
        auto const tokenId = paths[pathOffset + ti];
        auto const targetSrcTokenIdx = batchSlot * maxTargetSeqLen + tokenId;
        auto const draftDstTokenIdx = batchSlot * maxDraftSeqLen + inputLength + ti;
        auto const targetToken = targetIds[targetSrcTokenIdx];
        // Copy accepted tokens to the sequence with draft tokens (outputIds === outputIds)
        outputIds[draftDstTokenIdx] = targetToken;
    }

    // Leading thread reconstructs winning path and sets new data
    if (threadIdx.x == 0)
    {
        auto const hasEnd = totalShared.z;
        // Set end condition
        if (hasEnd)
        {
            finishedFinal[batchSlot].setFinishedEOS();
        }
        // Make correction to the sequence length
        sequenceLengths[batchSlot] += acceptedLength;
        acceptedLengths[batchSlot] = acceptedLength;
    }

    // Prepare logits pointers to respective logits from Medusa Heads for the all-top-K sampling kernel
    for (auto hi = static_cast<SizeType>(threadIdx.x); hi < maxNumHeads; hi += static_cast<SizeType>(blockDim.x))
    {
        logitsPtrs[batchIdx * maxNumHeads + hi]
            = medusaLogits + flat_index4(hi, batchIdx, bestNextIdx, 0, maxBatchSize, maxTokensPerStep, vocabSize);
    }
}

template <typename T>
void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* targetIds, SizeType* sequenceLengths,
    SizeType* acceptedLengths, FinishedState* finishedFinal, SizeType const* batchSlots, SizeType const* paths,
    TokenIdType const* endIds, T const* medusaLogits, T const** logitsPtrs, SizeType batchSize, SizeType vocabSize,
    SizeType maxBatchSize, SizeType maxDraftSeqLen, SizeType maxTargetSeqLen, SizeType maxNumHeads,
    SizeType maxTokensPerStep, cudaStream_t stream)
{
    constexpr SizeType BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(batchSize);
    acceptDraftTokensByIdsWithPaths<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(outputIds, targetIds, sequenceLengths,
        acceptedLengths, finishedFinal, batchSlots, paths, endIds, medusaLogits, logitsPtrs, batchSize, vocabSize,
        maxBatchSize, maxDraftSeqLen, maxTargetSeqLen, maxNumHeads, maxTokensPerStep);
}

template void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* targetIds,
    SizeType* sequenceLengths, SizeType* acceptedLengths, FinishedState* finishedFinal, SizeType const* batchSlots,
    SizeType const* paths, TokenIdType const* endIds, float const* medusaLogits, float const** logitsPtrs,
    SizeType batchSize, SizeType vocabSize, SizeType maxBatchSize, SizeType maxDraftSeqLen, SizeType maxTargetSeqLen,
    SizeType maxNumHeads, SizeType maxTokensPerStep, cudaStream_t stream);
template void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* targetIds,
    SizeType* sequenceLengths, SizeType* acceptedLengths, FinishedState* finishedFinal, SizeType const* batchSlots,
    SizeType const* paths, TokenIdType const* endIds, half const* medusaLogits, half const** logitsPtrs,
    SizeType batchSize, SizeType vocabSize, SizeType maxBatchSize, int32_t maxDraftSeqLen, SizeType maxTargetSeqLen,
    SizeType maxNumHeads, SizeType maxTokensPerStep, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
