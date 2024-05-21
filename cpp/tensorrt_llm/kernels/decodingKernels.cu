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

__global__ void insertUnfinishedPathKernel(BeamHypotheses bh)
{
    // Move ALL unfinished beams from bh.outputIdsUnfinish to bh.outputIdsCBA
    // So here might be more than `nBM` beams in bh.outputIdsCBA after the call
    // bh.outputIdsUnfinish -> bh.outputIdsCBA
    // bh.sequenceLengths   -> bh.sequenceLengthsCBA
    // bh.cumLogProbs       -> bh.cumLogProbsCBA
    // bh.logProbsTiled     -> bh.logProbsCBA
    // update bh.normedScoresCBA
    // update bh.numBeamsCBA

    int const bid = blockIdx.x;       // Index of Batch
    int const nBM{bh.nBeamWidth};
    int const nMBS{bh.nMaxBatchSize}; // Only for bh.logProbsTiled
    int const nMSL{bh.nMaxSeqLen};
    bool const bOutputLogProbs{bh.logProbsCBA != nullptr && bh.logProbsTiled != nullptr};
    int const indexDstStart{bh.numBeamsCBA[bid]};

    if (bh.batchDones[bid])
    {
        return;
    }

    for (int i = 0; i < nBM; ++i)
    {
        int const srcBeam = bid * nBM + i;
        int const dstBeam = bid * nBM * 2 + i + indexDstStart;
        int const step = bh.sequenceLengths[srcBeam] - 1;

        // The last token
        int const srcId = srcBeam * nMSL + step;
        int const dstId = dstBeam * nMSL + step;
        bh.outputIdsCBA[dstId] = bh.outputIdsUnfinish[srcId];
        if (bOutputLogProbs)
        {
            bh.logProbsCBA[dstId] = bh.logProbsTiled[step * nMBS * nBM + srcBeam];
        }
        // Previous tokens
        int prevId = bh.parentIdsUnfinish[srcId];
        for (int j = step - 1; j >= 0; --j)
        {
            int const index = bid * nBM * nMSL + prevId * nMSL + j;
            bh.outputIdsCBA[dstBeam * nMSL + j] = bh.outputIdsUnfinish[index];
            prevId = bh.parentIdsUnfinish[index];
        }
        if (bOutputLogProbs)
        {
            prevId = bh.parentIdsUnfinish[srcId];
            for (int j = step - 1; j >= 0; --j)
            {
                int const index = bid * nBM * nMSL + prevId * nMSL + j;
                bh.logProbsCBA[dstBeam * nMSL + j] = bh.logProbsTiled[j * nMBS * nBM + bid * nBM + prevId];
                prevId = bh.parentIdsUnfinish[index];
            }
        }
        // Other parameters
        bh.sequenceLengthsCBA[dstBeam] = bh.sequenceLengths[srcBeam];
        bh.normedScoresCBA[dstBeam]
            = applyLengthPenalty(bh.cumLogProbs[srcBeam], step - bh.inputLengths[srcBeam], bh.lengthPenalties[bid]);
        bh.cumLogProbsCBA[dstBeam] = bh.cumLogProbs[srcBeam];
        bh.numBeamsCBA[bid]++;
    }
}

void invokeInsertUnfinishedPath(BeamHypotheses& bh, cudaStream_t stream)
{
    insertUnfinishedPathKernel<<<bh.nBatchSize, 1, 0, stream>>>(bh);
}

__global__ void finalizeKernel(BeamHypotheses bh)
{
    // Do index sort on bh.normedScoresCBA, then move buffers from CBA to output by the order of index
    // bh.outputIdsCBA       -> bh.outputIds
    // bh.sequenceLengthsCBA -> bh.sequenceLengths
    // bh.cumLogProbsCBA     -> bh.cumLogProbs
    // bh.logProbsCBA        -> bh.logProbs

    int const bid = blockIdx.x;          // Index of Batch
    int const tid = threadIdx.x;         // Index of Beam
    int const nBM{bh.nBeamWidth};
    int const nCBA{bh.numBeamsCBA[bid]}; // count of candidate beams in CBA, nBM <= nCBA <= 2*nBM
    int const nMSL{bh.nMaxSeqLen};

    extern __shared__ char smem[];
    int* smemRank = (int*) (smem);                // [nBM]
    float* smemScore = (float*) (smemRank + nBM); // [2*nBM]
    int* smemSL = (int*) (smemScore + nBM * 2);   // [nBM]

    // Sort
    if (tid < nCBA)
    {
        smemScore[tid] = bh.normedScoresCBA[bid * nBM * 2 + tid];
    }
    __syncthreads();
    if (nCBA < 32)
    {
        int const warpid = tid / 32;
        int const laneid = tid % 32;
        RankNorm rankNorm{tid, tid < nCBA ? smemScore[tid] : -FLT_MAX};

        if (warpid == 0 && nCBA > 1)
        {
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0)); //  2
        }

        if (warpid == 0 && nCBA > 2)
        {
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1)); //  3~4
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && nCBA > 4)
        {
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2)); //  5~8
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 3) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 3) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && nCBA > 8)
        {
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3)); // 9~16
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && nCBA > 16)
        {
            rankNorm = swap(rankNorm, 0x10, bfe(laneid, 4) ^ bfe(laneid, 4)); // 17~32
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3));
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }

        if (tid < nBM)
        {
            smemRank[tid] = rankNorm.rank;
        }
        __syncthreads();
    }
    else
    {
        for (int i = 0; i < nBM; ++i)
        {
            float const score = tid < bh.numBeamsCBA[bid] ? smemScore[tid] : -FLT_MAX;
            float const maxScore = blockReduceMax<float>(score);
            if (tid == 0)
            {
                for (int j = 0; j < nBM * 2; ++j)
                {
                    if (smemScore[j] == maxScore)
                    {
                        smemRank[i] = j;
                        smemScore[j] = -FLT_MAX;
                        break;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Move bh.sequenceLengths, bh.cumLogProbs
    if (tid < nBM)
    {
        smemSL[tid] = bh.sequenceLengthsCBA[bid * nBM * 2 + smemRank[tid]];
        bh.sequenceLengths[bid * nBM + tid] = smemSL[tid];
        if (bh.cumLogProbs != nullptr)
        {
            bh.cumLogProbs[bid * nBM + tid] = bh.cumLogProbsCBA[bid * nBM * 2 + smemRank[tid]];
        }
    }
    __syncthreads();

    // Move bh.outputIds, bh.logProbs
    for (int beamIdx = 0; beamIdx < nBM; beamIdx++)
    {
        for (int i = tid; i < smemSL[beamIdx]; i += blockDim.x)
        {
            int const dst = bid * nBM * nMSL + beamIdx * nMSL + i;
            int const src = bid * nBM * 2 * nMSL + smemRank[beamIdx] * nMSL + i;
            bh.outputIds[dst] = bh.outputIdsCBA[src];
        }
        if (bh.logProbs != nullptr)
        {
            for (int i = tid; i < smemSL[beamIdx]; i += blockDim.x)
            {
                if (int const inputLength = bh.inputLengths[bid * nBM + beamIdx]; i >= inputLength)
                {
                    int const dst = bid * nBM * nMSL + beamIdx * nMSL + i;
                    int const src = bid * nBM * 2 * nMSL + smemRank[beamIdx] * nMSL + i;
                    bh.logProbs[dst - inputLength] = bh.logProbsCBA[src];
                }
            }
        }
    }
}

void invokeFinalize(BeamHypotheses& bh, cudaStream_t stream)
{
    TLLM_LOG_DEBUG("%s %s start", __FILE__, __PRETTY_FUNCTION__);

    int const nBM = bh.nBeamWidth;
    size_t const smem_size = sizeof(int) * nBM * 2 + sizeof(float) * nBM * 2;
    finalizeKernel<<<bh.nBatchSize, roundUp(nBM * 2, 32), smem_size, stream>>>(bh);
}

__global__ void initializeOutput(TokenIdType* finalOutputIds, TokenIdType const* endIds, SizeType32 const nMaxSeqLen)
{
    for (int i = threadIdx.x; i < nMaxSeqLen; i += blockDim.x)
    {
        finalOutputIds[blockIdx.x * nMaxSeqLen + i] = endIds[blockIdx.x];
    }
}

void invokeInitializeOutput(TokenIdType* finalOutputIds, TokenIdType const* endIds, SizeType32 const batchBeam,
    SizeType32 const nMaxSeqLen, cudaStream_t stream)
{
    initializeOutput<<<batchBeam, 256, 0, stream>>>(finalOutputIds, endIds, nMaxSeqLen);
}

__global__ void copyNextStepIds(TokenIdType* nextStepIds, TokenIdType const* const* outputIdsPtr,
    SizeType32 const* sequenceLengths, SizeType32 const* numNewTokens, SizeType32 const* batchSlots,
    SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth, SizeType32 maxSeqLen,
    SizeType32 maxTokensPerStep)
{
    for (auto index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
         index < batchSize * beamWidth * maxTokensPerStep; index += static_cast<SizeType32>(blockDim.x * gridDim.x))
    {
        // batchSlots == nullptr both in Python/C++ workflow yet
        // numNewTokens == nullptr when Medusa is disabled
        auto const batchIdx{index / (beamWidth * maxTokensPerStep)};
        auto const batchSlot{batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx};
        auto const remainder{index % (beamWidth * maxTokensPerStep)};
        auto const beamIdx{remainder / maxTokensPerStep};
        auto const tokenIdx{remainder % maxTokensPerStep};
        auto const newTokens{numNewTokens == nullptr ? 1 : numNewTokens[batchSlot]};
        auto const batchBeamIdx = batchSlot * beamWidth + beamIdx;
        auto const tokenBatchBeamIdx = tokenIdx * maxBatchSize * beamWidth + batchSlot * beamWidth + beamIdx;
        auto const indexSrc = sequenceLengths[batchBeamIdx] - newTokens + tokenIdx;
        if (tokenIdx >= newTokens || indexSrc < 0)
        {
            continue;
        }
        nextStepIds[tokenBatchBeamIdx] = outputIdsPtr[batchSlot][beamIdx * maxSeqLen + indexSrc];
    }
}

void invokeCopyNextStepIds(TokenIdType* nextStepIds, TokenIdType const* const* outputIdsPtr,
    SizeType32 const* sequenceLengths, SizeType32 const* numNewTokens, SizeType32 const* batchSlots,
    SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth, SizeType32 maxSeqLen,
    SizeType32 maxTokensPerStep, cudaStream_t stream)
{
    int const numElems = batchSize * beamWidth * maxTokensPerStep;
    dim3 block(min(256, numElems));
    dim3 grid(divUp(numElems, block.x));
    copyNextStepIds<<<grid, block, 0, stream>>>(nextStepIds, outputIdsPtr, sequenceLengths, numNewTokens, batchSlots,
        batchSize, maxBatchSize, beamWidth, maxSeqLen, maxTokensPerStep);
}

__global__ void transposeLogProbs(float* outputLogProbs, float* outputLogProbsTiled, SizeType32 const* sequenceLengths,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth,
    SizeType32 maxSeqLen)
{
    auto index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);

    auto const batchIdx = index / (beamWidth * maxSeqLen);
    auto const tmpIdx = index % (beamWidth * maxSeqLen);
    auto const beamIdx = tmpIdx / maxSeqLen;
    auto const pos = tmpIdx % maxSeqLen;
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

void invokeTransposeLogProbs(float* outputLogProbs, float* outputLogProbsTiled, SizeType32 const* sequenceLengths,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth,
    SizeType32 maxSeqLen, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(divUp(batchSize * beamWidth * maxSeqLen, block.x));
    transposeLogProbs<<<grid, block, 0, stream>>>(outputLogProbs, outputLogProbsTiled, sequenceLengths, batchSlots,
        batchSize, maxBatchSize, beamWidth, maxSeqLen);
}

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

__device__ __forceinline__ int4 reduceMaxInt4(int4 const& a, int4 const& b)
{
    return a.x >= b.x ? a : b;
}

template <typename T, SizeType32 BLOCK_SIZE>
__global__ void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* draftIds,
    TokenIdType const* targetIds, SizeType32* sequenceLengths, SizeType32* acceptedLengths,
    FinishedState* finishedFinal, SizeType32 const* batchSlots, SizeType32 const* paths, TokenIdType const* endIds,
    T const** medusaLogits, T const** logitsPtrs, SizeType32* curTokensPerStep, SizeType32 const* targetTokensPerStep,
    SizeType32* bestPathIds, SizeType32 batchSize, SizeType32 vocabSize, SizeType32 maxBatchSize,
    SizeType32 maxDraftTokens, SizeType32 maxSeqLen, SizeType32 maxNumHeads, SizeType32 maxTokensPerStep)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    auto const inputLength = sequenceLengths[batchSlot];
    auto const endId = endIds[batchSlot];
    auto const numTokensPerStep = curTokensPerStep[batchSlot];
    auto const maxNumDraftTokens = maxNumHeads + 1;

    int4 partialMax{-1, -1, 0, 0};
    // Go over different paths and construct implicit sequences
    for (auto pathIdx = static_cast<SizeType32>(threadIdx.x); pathIdx < maxTokensPerStep;
         pathIdx += static_cast<SizeType32>(blockDim.x))
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
        auto const targetTokenIdx = batchSlot * maxDraftTokens + tokenId;
        auto targetToken = targetIds[targetTokenIdx];
        auto nextIdx = tokenId;

        // Go along the path
        for (SizeType32 ti = 1; ti < maxNumDraftTokens; ++ti)
        {
            auto const tokenId = paths[pathOffset + ti];
            // Break if path terminates
            if (tokenId == -1)
            {
                acceptedLength = ti;
                break;
            }
            auto const targetTokenIdx = batchSlot * maxDraftTokens + tokenId;
            auto const draftTokenIdx = batchSlot * (maxDraftTokens - 1) + tokenId - 1;
            // In context phase, no draft tokens are given. Set draft token to -1 to get guaranteed rejection
            auto const draftToken = tokenId >= numTokensPerStep ? -1 : draftIds[draftTokenIdx];
            // Check if draft tokens are the same as target tokens
            bool const accepted = draftToken == targetToken;
            hasEnd = targetToken == endId;
            if (!accepted || hasEnd)
            {
                acceptedLength = hasEnd ? ti - 1 : ti;
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
    auto const bestNextIdx = numTokensPerStep == 1 ? 0 : totalShared.w;
    auto const pathOffset = flat_index3(batchSlot, bestPathIdx, 0, maxTokensPerStep, maxNumDraftTokens);
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < acceptedLength; ti += static_cast<SizeType32>(blockDim.x))
    {
        auto const tokenId = paths[pathOffset + ti];
        auto const targetSrcTokenIdx = batchSlot * maxDraftTokens + tokenId;
        auto const outputTokenIdx = batchSlot * maxSeqLen + inputLength + ti;
        auto const targetToken = targetIds[targetSrcTokenIdx];
        // Copy accepted tokens to the sequence with draft tokens (outputIds === outputIds)
        outputIds[outputTokenIdx] = targetToken;
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
        // In Medusa decoding step, number of draft tokens is 0 and must be updated for the next steps
        if (numTokensPerStep == 1)
        {
            curTokensPerStep[batchSlot] = targetTokensPerStep[batchSlot];
        }
        bestPathIds[batchSlot] = bestPathIdx;
    }

    // Prepare logits pointers to respective logits from Medusa Heads for the all-top-K sampling kernel
    for (auto hi = static_cast<SizeType32>(threadIdx.x); hi < maxNumHeads; hi += static_cast<SizeType32>(blockDim.x))
    {
        logitsPtrs[batchIdx * maxNumHeads + hi]
            = medusaLogits[batchSlot * maxNumHeads + hi] + flat_index2(bestNextIdx, 0, vocabSize);
    }
}

template <typename T>
void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* draftIds, TokenIdType const* targetIds,
    SizeType32* sequenceLengths, SizeType32* acceptedLengths, FinishedState* finishedFinal,
    SizeType32 const* batchSlots, SizeType32 const* paths, TokenIdType const* endIds, T const** medusaLogits,
    T const** logitsPtrs, SizeType32* curTokensPerStep, SizeType32 const* targetTokensPerStep, SizeType32* bestPathIds,
    SizeType32 batchSize, SizeType32 vocabSize, SizeType32 maxBatchSize, SizeType32 maxDraftTokens,
    SizeType32 maxSeqLen, SizeType32 maxNumHeads, SizeType32 maxTokensPerStep, cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(batchSize);
    acceptDraftTokensByIdsWithPaths<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(outputIds, draftIds, targetIds,
        sequenceLengths, acceptedLengths, finishedFinal, batchSlots, paths, endIds, medusaLogits, logitsPtrs,
        curTokensPerStep, targetTokensPerStep, bestPathIds, batchSize, vocabSize, maxBatchSize, maxDraftTokens,
        maxSeqLen, maxNumHeads, maxTokensPerStep);
}

template void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* draftIds,
    TokenIdType const* targetIds, SizeType32* sequenceLengths, SizeType32* acceptedLengths,
    FinishedState* finishedFinal, SizeType32 const* batchSlots, SizeType32 const* paths, TokenIdType const* endIds,
    float const** medusaLogits, float const** logitsPtrs, SizeType32* curTokensPerStep,
    SizeType32 const* targetTokensPerStep, SizeType32* bestPathIds, SizeType32 batchSize, SizeType32 vocabSize,
    SizeType32 maxBatchSize, SizeType32 maxDraftTokens, SizeType32 maxSeqLen, SizeType32 maxNumHeads,
    SizeType32 maxTokensPerStep, cudaStream_t stream);
template void acceptDraftTokensByIdsWithPaths(TokenIdType* outputIds, TokenIdType const* draftIds,
    TokenIdType const* targetIds, SizeType32* sequenceLengths, SizeType32* acceptedLengths,
    FinishedState* finishedFinal, SizeType32 const* batchSlots, SizeType32 const* paths, TokenIdType const* endIds,
    half const** medusaLogits, half const** logitsPtrs, SizeType32* curTokensPerStep,
    SizeType32 const* targetTokensPerStep, SizeType32* bestPathIds, SizeType32 batchSize, SizeType32 vocabSize,
    SizeType32 maxBatchSize, SizeType32 maxDraftTokens, SizeType32 maxSeqLen, SizeType32 maxNumHeads,
    SizeType32 maxTokensPerStep, cudaStream_t stream);

__global__ void scatterMedusaDraftTokens(TokenIdType* treeDraftIds, TokenIdType const* sourceDraftIds,
    SizeType32 const* treeIds, SizeType32 const* tokensPerStepData, SizeType32 const* batchSlots,
    SizeType32 maxTokensPerStep)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    auto const tokensPerStep = tokensPerStepData[batchSlot];
    auto const maxDraftTokens = maxTokensPerStep - 1;
    for (auto index = static_cast<SizeType32>(threadIdx.x); index < tokensPerStep - 1;
         index += static_cast<SizeType32>(blockDim.x))
    {
        auto const indexInTree = treeIds[batchSlot * maxDraftTokens + index];
        auto const treeDraftIdx = batchSlot * maxDraftTokens + index;
        auto const sourceDraftIdx = batchSlot * maxTokensPerStep + indexInTree;
        treeDraftIds[treeDraftIdx] = sourceDraftIds[sourceDraftIdx];
    }
}

void scatterMedusaDraftTokens(TokenIdType* treeDraftIds, TokenIdType const* sourceDraftIds, SizeType32 const* treeIds,
    SizeType32 const* tokensPerStep, SizeType32 const* batchSlots, SizeType32 maxDraftTokens, SizeType32 batchSize,
    cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 256;
    scatterMedusaDraftTokens<<<batchSize, BLOCK_SIZE, 0, stream>>>(
        treeDraftIds, sourceDraftIds, treeIds, tokensPerStep, batchSlots, maxDraftTokens);
}

template <int32_t BLOCK_SIZE>
__global__ void packAcceptedPaths(SizeType32* acceptedLengthsCumSum, SizeType32* pathsOffsets,
    SizeType32 const* acceptedLengths, SizeType32 const* bestPathIds, SizeType32 const* paths,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 maxTokensPerStep, SizeType32 maxNumDraftTokens)
{
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;

    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage tempStorage;
    auto const batchSizeRounded = ((batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    __shared__ SizeType32 currentCumSum;
    if (threadIdx.x == 0)
    {
        currentCumSum = 0;
    }

    __syncthreads();

    for (auto bi = static_cast<SizeType32>(threadIdx.x); bi < batchSizeRounded;
         bi += static_cast<SizeType32>(blockDim.x))
    {
        auto const valid = bi < batchSize;
        auto const batchSlot = valid ? batchSlots[bi] : 0;
        auto const acceptedLen = valid ? acceptedLengths[batchSlot] - 1 : 0;
        SizeType32 cumSum;
        BlockScan(tempStorage).ExclusiveSum(acceptedLen + currentCumSum, cumSum);
        if (threadIdx.x == blockDim.x - 1)
        {
            currentCumSum = cumSum;
        }
        __syncthreads();

        if (valid)
        {
            acceptedLengthsCumSum[bi] = cumSum;
            auto const bestPathIdx = bestPathIds[batchSlot];
            auto const pathIdx = flat_index3(batchSlot, bestPathIdx, 0, maxTokensPerStep, maxNumDraftTokens);
            for (SizeType32 ti = 0; ti < acceptedLen; ++ti)
            {
                pathsOffsets[cumSum + ti] = paths[pathIdx + ti + 1] - 1;
            }
        }
    }
    if (threadIdx.x == 0)
    {
        acceptedLengthsCumSum[batchSize] = currentCumSum;
    }
}

void invokePackAcceptedPaths(SizeType32* acceptedLengthsCumSum, SizeType32* pathsOffsets,
    SizeType32 const* acceptedLengths, SizeType32 const* bestPathIds, SizeType32 const* paths,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 maxTokensPerStep, SizeType32 maxNumDraftTokens,
    cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 1024;
    packAcceptedPaths<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(acceptedLengthsCumSum, pathsOffsets, acceptedLengths,
        bestPathIds, paths, batchSlots, batchSize, maxTokensPerStep, maxNumDraftTokens);
}
} // namespace kernels
} // namespace tensorrt_llm
