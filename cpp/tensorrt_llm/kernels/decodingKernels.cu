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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{

namespace kernels
{

__global__ void gatherTree(gatherTreeParam param)
{
    for (int batchbeamIdx = blockIdx.x * blockDim.x + threadIdx.x; batchbeamIdx < param.batchSize * param.beamWidth;
         batchbeamIdx += gridDim.x * blockDim.x)
    {
        const int batch = batchbeamIdx / param.beamWidth;
        const int beam = batchbeamIdx % param.beamWidth;
        const int inputLen = param.inputLengths == nullptr ? 0 : param.inputLengths[batchbeamIdx];

        const int* parentIds = param.parentIds;
        const int* stepIds = param.stepIds;

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
        const int maxSeqLenB = min(param.maxSeqLen, maxLen);
        if (maxSeqLenB <= 0)
        {
            continue;
        }

        const int initialTgtIx = batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + maxSeqLenB - 1;
        const int initialParentIx = batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + maxSeqLenB - 1;
        param.outputIds[initialTgtIx] = __ldg(stepIds + initialParentIx);
        int parent = parentIds == nullptr ? 0 : __ldg(parentIds + initialParentIx) % param.beamWidth;
        bool foundBad = false;

        for (int level = maxSeqLenB - 2; level >= 0; --level)
        {
            const int levelBeamIx = batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + level;
            const int levelParentIx = batch * param.beamWidth * param.maxSeqLen + parent * param.maxSeqLen + level;
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
                const int levelBeamIx = batch * param.beamWidth * param.maxSeqLen + beam * param.maxSeqLen + time;
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
    // score = log(prob) / (length)^lengthPenalty.
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

inline __device__ RankNorm swap(const RankNorm& rankNorm, int mask, int dir)
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
    const int beamIdx = static_cast<int>(threadIdx.x);
    const int beamWidth{param.beamWidth};

    extern __shared__ char array[];
    int* sRank = (int*) (array);
    int* sLength = (int*) (sRank + beamWidth);
    float* sScores = (float*) (sLength + beamWidth);
    float* sNormedScores = (float*) (sScores + beamWidth);
    int* sIds = (int*) (sNormedScores + beamWidth);

    if (beamIdx < beamWidth)
    {
        const int idx = blockIdx.x * param.beamWidth + beamIdx;
        const int numGeneratedToken{param.sequenceLengths[idx] - param.inputLengths[idx]};
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
    const int* topKOutputIds, const int* topKSequenceLengths, const float* scores, const float* topKCumLogProbs,
    const float* topKLogProbs, const int* numBeams, const int* inputLengths, const int beamWidth, const int maxSeqLen)
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
    const int numBeam = numBeams[blockIdx.x];
    if (threadIdx.x < numBeam)
    {
        sScores[threadIdx.x] = scores[blockIdx.x * beamWidth * 2 + threadIdx.x];
    }
    __syncthreads();

    if (numBeam < 32)
    {
        const int beamIdx = threadIdx.x;
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
    const int* topKOutputIds, const int* topKSequenceLengths, const float* scores, const float* topKCumLogProbs,
    const float* topKLogProbs, const int* numBeams, const int* inputLengths, const int beamWidth, const int maxSeqLen,
    const int batchSize, cudaStream_t stream)
{
    TLLM_LOG_DEBUG("%s %s start", __FILE__, __PRETTY_FUNCTION__);
    dim3 block(beamWidth * 2);
    block.x = (block.x + 31) / 32 * 32;
    TLLM_CHECK(block.x < 1024);
    finalize<<<batchSize, block, beamWidth * sizeof(int) * 2 + (beamWidth * 2) * sizeof(float), stream>>>(outputIds,
        sequenceLengths, cumLogProbs, outputLogProbs, topKOutputIds, topKSequenceLengths, scores, topKCumLogProbs,
        topKLogProbs, numBeams, inputLengths, beamWidth, maxSeqLen);
}

__global__ void initializeOutput(int* outputIds, const int* endIds, const int maxSeqLen)
{
    for (int i = threadIdx.x; i < maxSeqLen; i += blockDim.x)
    {
        outputIds[blockIdx.x * maxSeqLen + i] = endIds[blockIdx.x];
    }
}

void invokeInitializeOutput(int* outputIds, const int* endIds, int batchBeam, int maxSeqLen, cudaStream_t stream)
{
    initializeOutput<<<batchBeam, 256, 0, stream>>>(outputIds, endIds, maxSeqLen);
}

__global__ void copyNextStepIds(
    int* nextStepIds, int** outputIdsPtr, const int* sequenceLengths, int batchSize, int beamWidth, int maxSeqLen)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batchSize * beamWidth;
         index += blockDim.x * gridDim.x)
    {
        const int batchIdx{index / beamWidth};
        const int beamIdx{index % beamWidth};
        nextStepIds[index] = outputIdsPtr[batchIdx][beamIdx * maxSeqLen + sequenceLengths[index] - 1];
    }
}

void invokeCopyNextStepIds(int* nextStepIds, int** outputIdsPtr, const int* sequenceLengths, int batchSize,
    int beamWidth, int maxSeqLen, cudaStream_t stream)
{
    dim3 block(min(256, batchSize * beamWidth));
    dim3 grid(divUp(batchSize * beamWidth, block.x));
    copyNextStepIds<<<grid, block, 0, stream>>>(
        nextStepIds, outputIdsPtr, sequenceLengths, batchSize, beamWidth, maxSeqLen);
}

__global__ void transposeLogProbs(float* output_log_probs, float* output_log_probs_tiled, const int* sequence_lengths,
    int batch_size, int beam_width, int max_seq_len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int batch_idx = index / (beam_width * max_seq_len);
    const int tmp_idx = index % (beam_width * max_seq_len);
    const int beam_idx = tmp_idx / max_seq_len;
    const int pos = tmp_idx % max_seq_len;

    if (batch_idx < batch_size && pos < sequence_lengths[batch_idx])
    {

        output_log_probs[index]
            = output_log_probs_tiled[pos * batch_size * beam_width + batch_idx * beam_width + beam_idx];
    }
}

void invokeTransposeLogProbs(float* output_log_probs, float* output_log_probs_tiled, const int* sequence_lengths,
    int batch_size, int beam_width, int max_seq_len, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(divUp(batch_size * beam_width * max_seq_len, block.x));
    transposeLogProbs<<<grid, block, 0, stream>>>(
        output_log_probs, output_log_probs_tiled, sequence_lengths, batch_size, beam_width, max_seq_len);
}

__global__ void acceptDraftTokensByIds(const int* draftIds, const int* targetIds, const int* contextLengths,
    const int* numsDraftTokens, int* sequenceLengths, const FinishedState* finished, FinishedState* finishedFinal,
    int* finishedSum, int batchSize, int beamWidth, int maxSeqLen, int maxDraftTokens)
{
    int threadFinishedCount = 0;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batchSize * beamWidth;
         index += blockDim.x * gridDim.x)
    {
        const auto numDraftTokens = numsDraftTokens[index];

        const auto contextLength = contextLengths[index];
        auto& sequenceLength = sequenceLengths[index];
        int finishedDraftIdx = 0;
        for (int ti = contextLength; ti < min(sequenceLength, contextLength + numDraftTokens); ++ti, ++finishedDraftIdx)
        {
            const auto draftIdx = ti - contextLength;
            const auto targetTokenIdx = index * maxSeqLen + ti;
            const auto draftTokenIdx = index * maxDraftTokens + draftIdx;
            // Check if draft tokens are the same as target tokens
            const bool accepted = draftIds[draftTokenIdx] == targetIds[targetTokenIdx];
            if (!accepted)
            {
                // Set sequence length to the numAcceptedTokens + 1
                sequenceLength = min(ti + 1, maxSeqLen);
                // FIXME(nkorobov): do we need to set endIds here?
                break;
            }
        }
        FinishedState finishState = finished[finishedDraftIdx * batchSize * beamWidth + index];
        finishedFinal[index] = finishState;
        threadFinishedCount += static_cast<int>(finishState.isFinished());
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
            finishedSum[0] = blockFinishedCount;
        }
    }
}

void invokeAcceptDraftTokensByIds(const int* draftIds, const int* targetIds, const int* contextLengths,
    const int* numsDraftTokens, int* sequenceLengths, const FinishedState* finished, FinishedState* finishedFinal,
    int* finishedSum, int batchSize, int beamWidth, int maxSeqLen, int maxDraftTokens, cudaStream_t stream)
{
    TLLM_CHECK(beamWidth == 1);
    dim3 block(min(256, batchSize * beamWidth));
    dim3 grid(1);
    acceptDraftTokensByIds<<<grid, block, 0, stream>>>(draftIds, targetIds, contextLengths, numsDraftTokens,
        sequenceLengths, finished, finishedFinal, finishedSum, batchSize, beamWidth, maxSeqLen, maxDraftTokens);
}

template <typename T>
__global__ void acceptDraftTokensByLogitsKernel(const T* draftProbs, T* targetProbs, const int* numsDraftTokens,
    FinishedState* finished, curandState_t* curandState, int batchSize, int beamWidth, int vocabSize,
    bool randomThreshold, float constantThreshold)
{
    const auto bid = blockIdx.x;
    const auto draftTokenIdx = blockIdx.y;
    const auto batchIdx = bid / beamWidth;

    const auto numDraftTokens = numsDraftTokens[bid];

    if (draftTokenIdx >= numDraftTokens)
    {
        return;
    }

    const auto logitsOffset = draftTokenIdx * batchSize * beamWidth * vocabSize + bid * vocabSize;
    const auto draftProbsBatch = draftProbs + logitsOffset;
    const auto targetProbsBatch = targetProbs + logitsOffset;

    int rejected = 0;

    for (int vIdx = threadIdx.x; vIdx < vocabSize; vIdx += blockDim.x)
    {
        if (rejected > 0)
        {
            break;
        }

        // FIXME(nkorobov): We compare probability distributions, but it might make sense to compare probabilities of
        // the selected tokens based on the https://arxiv.org/pdf/2302.01318.pdf
        const auto threshold = randomThreshold ? curand_uniform(curandState + batchIdx) : constantThreshold;

        const auto targetProb = static_cast<float>(targetProbsBatch[vIdx]);
        const auto draftProb = static_cast<float>(draftProbsBatch[vIdx]);

        rejected = __syncthreads_count(targetProb < threshold * draftProb);
    }
    if (threadIdx.x == 0)
    {
        finished[draftTokenIdx * batchSize * beamWidth + bid]
            = rejected > 0 ? FinishedState::skipDecoding() : FinishedState::empty();
    }
}

template <typename T>
__global__ void correctAcceptedStatesAndLogits(const T* draftProbs, T* targetProbs, T* targetLogits,
    const int* numsDraftTokens, FinishedState* finished, int batchSize, int beamWidth, int vocabSize)
{
    const auto bid = blockIdx.x;
    const auto numDraftTokens = numsDraftTokens[bid];

    __shared__ int numAcceptedTokens;
    if (threadIdx.x == 0)
    {
        numAcceptedTokens = numDraftTokens;
        bool cummulativeSkipDecoding = false;
        for (int ti = 0; ti < numDraftTokens + 1; ++ti)
        {
            auto& finishedState = finished[ti * batchSize * beamWidth + bid];
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
        const auto logitsIdx = numAcceptedTokens * batchSize * beamWidth * vocabSize + bid * vocabSize;
        const auto draftProbBatch = draftProbs + logitsIdx;
        auto targetProbBatch = targetProbs + logitsIdx;
        auto targetLogitsBatch = targetLogits + logitsIdx;

        float sumProbs = 0.f;
        for (int vIdx = threadIdx.x; vIdx < vocabSize; vIdx += blockDim.x)
        {
            const auto correctedProb = max(static_cast<float>(targetProbBatch[vIdx] - draftProbBatch[vIdx]), 0.f);
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

        for (int vIdx = threadIdx.x; vIdx < vocabSize; vIdx += blockDim.x)
        {
            const auto correctedNormProb = static_cast<float>(targetProbBatch[vIdx]) / sumProbsShared;
            targetLogitsBatch[vIdx] = __logf(correctedNormProb / (1.f - correctedNormProb));
        }
    }
}

template <typename T>
void acceptDraftTokensByLogits(T* draftLogits, T* targetLogits, T* draftProbs, T* targetProbs,
    const int* numsDraftTokens, FinishedState* finished, curandState_t* curandState, int batchSize, int beamWidth,
    int vocabSize, int vocabSizePadded, int maxDraftTokens, bool randomThreshold, float constantThreshold,
    cudaStream_t stream)
{
    TLLM_CHECK(beamWidth == 1);
    {
        invokeAddBiasSoftMax(draftLogits, draftProbs, (T*) (nullptr), nullptr, finished,
            batchSize * beamWidth * maxDraftTokens, vocabSize, vocabSizePadded, stream);
        invokeAddBiasSoftMax(targetLogits, targetProbs, (T*) (nullptr), nullptr, finished,
            batchSize * beamWidth * maxDraftTokens, vocabSize, vocabSizePadded, stream);
    }
    {
        dim3 block(1024);
        dim3 grid(batchSize * beamWidth, maxDraftTokens);
        acceptDraftTokensByLogitsKernel<<<grid, block, 0, stream>>>(draftProbs, targetProbs, numsDraftTokens, finished,
            curandState, batchSize, beamWidth, vocabSizePadded, randomThreshold, constantThreshold);
    }
    {
        dim3 block(1024);
        dim3 grid(batchSize * beamWidth);
        correctAcceptedStatesAndLogits<<<grid, block, 0, stream>>>(
            draftProbs, targetProbs, targetLogits, numsDraftTokens, finished, batchSize, beamWidth, vocabSizePadded);
    }
}

template void acceptDraftTokensByLogits(float* draftLogits, float* targetLogits, float* draftProbs, float* targetProbs,
    const int* numsDraftTokens, FinishedState* finished, curandState_t* curandState, int batchSize, int beamWidth,
    int vocabSize, int vocabSizePadded, int maxDraftTokens, bool randomThreshold, float constantThreshold,
    cudaStream_t stream);
template void acceptDraftTokensByLogits(half* draftLogits, half* targetLogits, half* draftProbs, half* targetProbs,
    const int* numsDraftTokens, FinishedState* finished, curandState_t* curandState, int batchSize, int beamWidth,
    int vocabSize, int vocabSizePadded, int maxDraftTokens, bool randomThreshold, float constantThreshold,
    cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
