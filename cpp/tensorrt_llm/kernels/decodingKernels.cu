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
#include "tensorrt_llm/common/cudaUtils.h"
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

class CopyBeamHypothesesStruct
{
public:
    TokenIdType const* srcOutputIdsCBA; // [BS, BM*2, MSL]
    TokenIdType* dstOutputIdsCBA;       // [BS, BM*2, MSL]
    SizeType32 outputIdsNumElts;

    float const* srcLogProbsCBA; // [BS, BM*2, MSL]
    float* dstLogProbsCBA;       // [BS, BM*2, MSL]
    SizeType32 logProbsNumElts;

    SizeType32 const* srcSequenceLengthsCBA; // [BS, BM*2]
    SizeType32* dstSequenceLengthsCBA;       // [BS, BM*2]
    SizeType32 sequenceLengthsNumElts;

    float const* srcCumLogProbsCBA; // [BS, BM*2]
    float* dstCumLogProbsCBA;       // [BS, BM*2]
    SizeType32 cumLogProbsCBANumElts;

    float const* srcNormedScoresCBA; // [BS, BM*2]
    float* dstNormedScoresCBA;       // [BS, BM*2]
    SizeType32 normedScoresNumElts;

    SizeType32 const* srcNumBeamsCBA; // [BS]
    SizeType32* dstNumBeamsCBA;       // [BS]
    SizeType32 numBeamsNumElts;

    float const* srcMinNormedScoresCBA; // [BS]
    float* dstMinNormedScoresCBA;       // [BS]
    SizeType32 minNormedScoresNumElts;

    bool const* srcBatchDones; // [BS]
    bool* dstBatchDones;       // [BS]
    SizeType32 batchDonesNumElts;

    float const* srcCumLogProbs; // [BS, BM]
    float* dstCumLogProbs;       // [BS, BM]
    SizeType32 cumLogProbsNumElts;
};

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
    // Exchange RankNorm data inside the warp
    RankNorm other;
    other.rank = __shfl_xor_sync(unsigned(-1), rankNorm.rank, mask);
    other.norm = __shfl_xor_sync(unsigned(-1), rankNorm.norm, mask);
    // dir == 0 -> return larger one
    // dir == 1 -> return smaller one
    bool doSwap = (rankNorm.norm != other.norm) && ((rankNorm.norm > other.norm) == dir);
    return doSwap ? other : rankNorm;
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
            rankNorm = swap(rankNorm, 0x10, bfe(laneid, 5) ^ bfe(laneid, 4)); // 17~32
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 5) ^ bfe(laneid, 3));
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 5) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 5) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 5) ^ bfe(laneid, 0));
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
    sync_check_cuda_error(param.stream);

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
    // So here might be more than `nBM` beams in bh.outputIdsCBA after this kernel
    // Data movement:
    // bh.outputIdsUnfinish -> bh.outputIdsCBA
    // bh.sequenceLengths   -> bh.sequenceLengthsCBA
    // bh.cumLogProbs       -> bh.cumLogProbsCBA
    // bh.logProbsTiled     -> bh.logProbsCBA
    // update bh.normedScoresCBA
    // update bh.numBeamsCBA

    size_t const bid = blockIdx.x;       // Index of Batch
    size_t const nBM{bh.nBeamWidth};
    size_t const nMBS{bh.nMaxBatchSize}; // Only for bh.logProbsTiled
    size_t const nMSL{bh.nMaxSeqLen};
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
            = applyLengthPenalty(bh.cumLogProbs[srcBeam], step - bh.inputLengths[srcBeam] + 1, bh.lengthPenalties[bid]);
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
    // Data movement:
    // bh.outputIdsCBA       -> bh.outputIds
    // bh.sequenceLengthsCBA -> bh.sequenceLengths
    // bh.cumLogProbsCBA     -> bh.cumLogProbs
    // bh.logProbsCBA        -> bh.logProbs

    int const bid = blockIdx.x;  // Index of Batch
    int const tid = threadIdx.x; // Index of Beam
    size_t const nBM{bh.nBeamWidth};
    size_t const nMSL{bh.nMaxSeqLen};
    int const nCBA{bh.numBeamsCBA[bid]}; // Count of candidates in CBA, nBM <= nCBA <= 2*nBM

    extern __shared__ char smem[];
    int* smemRank = (int*) (smem);                // [nBM]
    float* smemScore = (float*) (smemRank + nBM); // [2*nBM]
    int* smemSL = (int*) (smemScore + nBM * 2);   // [nBM]

    // Sort
    for (int i = tid; i < nCBA; i += blockDim.x)
    {
        smemScore[i] = bh.normedScoresCBA[bid * nBM * 2 + i];
    }
    __syncthreads();

    if (nCBA <= 32)
    {
        int const warpid = tid / 32;
        int const laneid = tid % 32;
        RankNorm rankNorm{tid, tid < nCBA ? smemScore[tid] : -FLT_MAX};

        if (warpid == 0 && nCBA > 1)
        {
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0)); // 2
        }
        if (warpid == 0 && nCBA > 2)
        {
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1)); // 3~4
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));
        }
        if (warpid == 0 && nCBA > 4)
        {
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2)); // 5~8
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
            rankNorm = swap(rankNorm, 0x10, bfe(laneid, 5) ^ bfe(laneid, 4)); // 17~32
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 5) ^ bfe(laneid, 3));
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 5) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 5) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 5) ^ bfe(laneid, 0));
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
            float maxScore = -FLT_MAX;
            for (int j = 0; j < (nCBA + 1024 - 1) / 1024; ++j)
            {
                int const index = tid + 1024 * j;
                float const score = (index < bh.numBeamsCBA[bid]) ? smemScore[index] : -FLT_MAX;
                float const maxScore1 = blockReduceMax<float>(score);
                maxScore = max(maxScore, maxScore1);
            }
            if (tid == 0)
            {
                for (int j = 0; j < nCBA; ++j)
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
    int const nBM = bh.nBeamWidth;
    int const nThread = min(roundUp(nBM * 2, 32), 1024);
    size_t const nByteSharedMemory = (sizeof(int) + sizeof(float)) * nBM * 2;
    finalizeKernel<<<bh.nBatchSize, nThread, nByteSharedMemory, stream>>>(bh);
    sync_check_cuda_error(stream);
}

__global__ void copyBeamHypotheses(CopyBeamHypothesesStruct copyStruct)
{
    auto const idx = static_cast<SizeType32>(threadIdx.x + blockIdx.x * blockDim.x);
    auto const stride = static_cast<SizeType32>(blockDim.x * gridDim.x);

    for (SizeType32 ii = idx; ii < copyStruct.outputIdsNumElts; ii += stride)
    {
        copyStruct.dstOutputIdsCBA[ii] = copyStruct.srcOutputIdsCBA[ii];
    }

    for (SizeType32 ii = idx; ii < copyStruct.logProbsNumElts; ii += stride)
    {
        copyStruct.dstLogProbsCBA[ii] = copyStruct.srcLogProbsCBA[ii];
    }

    for (SizeType32 ii = idx; ii < copyStruct.cumLogProbsNumElts; ii += stride)
    {
        copyStruct.dstCumLogProbs[ii] = copyStruct.srcCumLogProbs[ii];
    }

    for (SizeType32 ii = idx; ii < copyStruct.sequenceLengthsNumElts; ii += stride)
    {
        copyStruct.dstSequenceLengthsCBA[ii] = copyStruct.srcSequenceLengthsCBA[ii];
    }

    for (SizeType32 ii = idx; ii < copyStruct.cumLogProbsCBANumElts; ii += stride)
    {
        copyStruct.dstCumLogProbsCBA[ii] = copyStruct.srcCumLogProbsCBA[ii];
    }

    for (SizeType32 ii = idx; ii < copyStruct.normedScoresNumElts; ii += stride)
    {
        copyStruct.dstNormedScoresCBA[ii] = copyStruct.srcNormedScoresCBA[ii];
    }

    for (SizeType32 ii = idx; ii < copyStruct.numBeamsNumElts; ii += stride)
    {
        copyStruct.dstNumBeamsCBA[ii] = copyStruct.srcNumBeamsCBA[ii];
    }

    for (SizeType32 ii = idx; ii < copyStruct.minNormedScoresNumElts; ii += stride)
    {
        copyStruct.dstMinNormedScoresCBA[ii] = copyStruct.srcMinNormedScoresCBA[ii];
    }

    for (SizeType32 ii = idx; ii < copyStruct.batchDonesNumElts; ii += stride)
    {
        copyStruct.dstBatchDones[ii] = copyStruct.srcBatchDones[ii];
    }
}

void invokeCopyBeamHypotheses(DecodingOutput::BeamHypotheses const& src, DecodingOutput::BeamHypotheses const& dst,
    ITensor& srcCumLogProbs, ITensor& dstCumLogProbs, runtime::CudaStream const& stream, SizeType32 numSMs)
{
    CopyBeamHypothesesStruct copyStruct = {};

    copyStruct.srcOutputIdsCBA = bufferCast<TokenIdType>(*(src.outputIdsCBA));
    copyStruct.dstOutputIdsCBA = bufferCast<TokenIdType>(*(dst.outputIdsCBA));
    copyStruct.outputIdsNumElts = dst.outputIdsCBA->getSize();

    copyStruct.srcLogProbsCBA = bufferCast<float>(*(src.logProbsCBA));
    copyStruct.dstLogProbsCBA = bufferCast<float>(*(dst.logProbsCBA));
    copyStruct.logProbsNumElts = dst.logProbsCBA->getSize();

    copyStruct.srcSequenceLengthsCBA = bufferCast<SizeType32>(*(src.sequenceLengthsCBA));
    copyStruct.dstSequenceLengthsCBA = bufferCast<SizeType32>(*(dst.sequenceLengthsCBA));
    copyStruct.sequenceLengthsNumElts = dst.sequenceLengthsCBA->getSize();

    copyStruct.srcCumLogProbsCBA = bufferCast<float>(*(src.cumLogProbsCBA));
    copyStruct.dstCumLogProbsCBA = bufferCast<float>(*(dst.cumLogProbsCBA));
    copyStruct.cumLogProbsCBANumElts = dst.cumLogProbsCBA->getSize();

    copyStruct.srcNormedScoresCBA = bufferCast<float>(*(src.normedScoresCBA));
    copyStruct.dstNormedScoresCBA = bufferCast<float>(*(dst.normedScoresCBA));
    copyStruct.normedScoresNumElts = dst.normedScoresCBA->getSize();

    copyStruct.srcNumBeamsCBA = bufferCast<SizeType32>(*(src.numBeamsCBA));
    copyStruct.dstNumBeamsCBA = bufferCast<SizeType32>(*(dst.numBeamsCBA));
    copyStruct.numBeamsNumElts = dst.numBeamsCBA->getSize();

    copyStruct.srcMinNormedScoresCBA = bufferCast<float>(*(src.minNormedScoresCBA));
    copyStruct.dstMinNormedScoresCBA = bufferCast<float>(*(dst.minNormedScoresCBA));
    copyStruct.minNormedScoresNumElts = dst.minNormedScoresCBA->getSize();

    copyStruct.srcBatchDones = bufferCast<bool>(*(src.batchDones));
    copyStruct.dstBatchDones = bufferCast<bool>(*(dst.batchDones));
    copyStruct.batchDonesNumElts = dst.batchDones->getSize();

    copyStruct.srcCumLogProbs = bufferCast<float>(srcCumLogProbs);
    copyStruct.dstCumLogProbs = bufferCast<float>(dstCumLogProbs);
    copyStruct.cumLogProbsNumElts = srcCumLogProbs.getSize();

    copyBeamHypotheses<<<numSMs, 256, 0, stream.get()>>>(copyStruct);
}

__global__ void initializeOutput(
    TokenIdType* finalOutputIds, TokenIdType const* endIds, SizeType32 const beam, SizeType32 const nMaxSeqLen)
{
    for (int i = threadIdx.x; i < nMaxSeqLen; i += blockDim.x)
    {
        finalOutputIds[blockIdx.x * nMaxSeqLen + i] = endIds[blockIdx.x / beam];
    }
}

void invokeInitializeOutput(TokenIdType* finalOutputIds, TokenIdType const* endIds, SizeType32 const batch,
    SizeType32 const beam, SizeType32 const nMaxSeqLen, cudaStream_t stream)
{
    initializeOutput<<<batch * beam, 256, 0, stream>>>(finalOutputIds, endIds, beam, nMaxSeqLen);
}

__global__ void copyNextStepIds(TokenIdType* nextStepIds, TokenIdType const* const* outputIdsPtr,
    SizeType32 const* sequenceLengths, SizeType32 const* numNewTokens, SizeType32 const* batchSlots,
    SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth, SizeType32 maxSeqLen,
    SizeType32 maxTokensPerStep)
{
    for (auto index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
         index < batchSize * beamWidth * maxTokensPerStep; index += static_cast<SizeType32>(blockDim.x * gridDim.x))
    {
        // numNewTokens == nullptr when Medusa is disabled
        auto const batchIdx{index / (beamWidth * maxTokensPerStep)};
        auto const batchSlot{batchSlots[batchIdx]};
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

    auto const batchSlot = batchSlots[batchIdx];
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

} // namespace kernels

namespace runtime::kernels
{
// Must be similar to [cpp/tensorrt_llm/thop/gatherTreeOp.cpp] gatherTree
void gatherTree(DecodingOutput const& decodingOutput, DecodingInput const& decodingInput,
    SamplingConfig const& samplingConfig, runtime::CudaStream const& cudaStream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& stream = cudaStream.get();
    BufferManager manager{std::make_shared<CudaStream>(stream)};

    auto& finalOutputIds = *decodingOutput.gatheredIds;
    auto const& finalOutputIdsShape = finalOutputIds.getShape();
    auto const& decodingOutputIdsShape = decodingOutput.ids->getShape();
    auto const batchSize = finalOutputIdsShape.d[0];
    auto const beamWidth = finalOutputIdsShape.d[1];
    auto const maxSeqLength = finalOutputIdsShape.d[2];

    TLLM_CHECK_WITH_INFO(beamWidth > 1, "gatherTree is only needed for beam search.");

    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[0] == batchSize,
        common::fmtstr("Decoder batch size (" FMT_DIM ") does not match final batch size (" FMT_DIM ")",
            decodingOutputIdsShape.d[0], batchSize));
    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[1] == beamWidth,
        common::fmtstr("Decoder beam width (" FMT_DIM ") does not match final beam width (" FMT_DIM ")",
            decodingOutputIdsShape.d[1], beamWidth));
    TLLM_CHECK_WITH_INFO(decodingOutputIdsShape.d[2] <= maxSeqLength,
        common::fmtstr("Decoder seq length size (" FMT_DIM ") is too large for final seq length (" FMT_DIM ")",
            decodingOutputIdsShape.d[2], maxSeqLength));

    // prefill finalOutputIds with the EOS tokens from decodingInput.endIds
    tensorrt_llm::kernels::invokeInitializeOutput(bufferCast<TokenIdType>(finalOutputIds),
        bufferCast<TokenIdType>(*decodingInput.endIds), batchSize, beamWidth, maxSeqLength, stream);
    sync_check_cuda_error(stream);

    std::vector<float> lengthPenaltyVec;
    auto lengthPenaltyPtr = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize}), TRTDataType<float>::value));
    if (!samplingConfig.lengthPenalty.has_value() || samplingConfig.lengthPenalty.value().size() == 0)
    {
        lengthPenaltyVec = std::vector<float>(batchSize, 1.0f);
    }
    else if (long int const size = samplingConfig.lengthPenalty.value().size(); size == 1)
    {
        lengthPenaltyVec = std::vector<float>(batchSize, samplingConfig.lengthPenalty.value()[0]);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(size == batchSize,
            common::fmtstr("Size of lengthPenalty in SamplingConfig (" FMT_DIM ") is different from batchSize (" FMT_DIM
                           ")",
                size, batchSize));
        lengthPenaltyVec = samplingConfig.lengthPenalty.value();
    }

    lengthPenaltyPtr = manager.copyFrom(lengthPenaltyVec, ITensor::makeShape({batchSize}), runtime::MemoryType::kGPU);

    tensorrt_llm::kernels::BeamHypotheses bh;
    bh.nMaxBatchSize = batchSize;
    bh.nBatchSize = batchSize;
    bh.nBeamWidth = beamWidth;
    bh.nMaxSeqLen = maxSeqLength;
    bh.lengthPenalties = bufferCast<float>(*lengthPenaltyPtr);
    bh.inputLengths = bufferCast<SizeType32>(*decodingInput.lengths);
    bh.outputIds = bufferCast<TokenIdType>(finalOutputIds);
    bh.logProbs = bufferCastOrNull<float>(decodingOutput.logProbs);
    bh.logProbsTiled = bufferCast<float>(*decodingOutput.logProbsTiled);
    bh.sequenceLengths = bufferCast<SizeType32>(*decodingOutput.lengths);
    bh.cumLogProbs = bufferCast<float>(*decodingOutput.cumLogProbs);
    bh.outputIdsCBA = bufferCast<TokenIdType>(*decodingOutput.beamHypotheses.outputIdsCBA);
    bh.logProbsCBA = bufferCast<float>(*decodingOutput.beamHypotheses.logProbsCBA);
    bh.sequenceLengthsCBA = bufferCast<SizeType32>(*decodingOutput.beamHypotheses.sequenceLengthsCBA);
    bh.cumLogProbsCBA = bufferCast<float>(*decodingOutput.beamHypotheses.cumLogProbsCBA);
    bh.normedScoresCBA = bufferCast<float>(*decodingOutput.beamHypotheses.normedScoresCBA);
    bh.numBeamsCBA = bufferCast<SizeType32>(*decodingOutput.beamHypotheses.numBeamsCBA);
    bh.minNormedScoresCBA = bufferCast<float>(*decodingOutput.beamHypotheses.minNormedScoresCBA);
    bh.batchDones = bufferCast<bool>(*decodingOutput.beamHypotheses.batchDones);
    bh.finished = bufferCast<tensorrt_llm::kernels::FinishedState>(*decodingOutput.finishReasons);
    bh.outputIdsUnfinish = bufferCast<TokenIdType>(*decodingOutput.ids);
    bh.parentIdsUnfinish = bufferCast<TokenIdType>(*decodingOutput.parentIds);

    // This is where transpose is done
    tensorrt_llm::kernels::invokeInsertUnfinishedPath(bh, stream);
    sync_check_cuda_error(stream);

    tensorrt_llm::kernels::invokeFinalize(bh, stream);
    sync_check_cuda_error(stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace runtime::kernels

} // namespace tensorrt_llm
