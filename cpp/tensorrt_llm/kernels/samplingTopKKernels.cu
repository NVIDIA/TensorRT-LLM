/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <stdexcept>
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topKStage1(const T* __restrict logProbs, T* tmpLogProbs, int* topKTmpIdBuf, T* topKTmpValBuf,
    const FinishedState* finished, const int maxTopK, const int* topKs, const int vocabSize, const int* endIds,
    const bool* skipDecode, const int* batchSlots)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    int const tid = threadIdx.x;
    int const bid = blockIdx.x;

    auto const batchId = bid / BLOCKS_PER_BEAM_; // row id for logProbs
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchId] : batchId;
    FinishedState const finishState = finished != nullptr ? finished[batchSlot] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchSlot]) || (finishState.isSkipDecoding()))
    {
        return;
    }
    const int blockLane = bid % BLOCKS_PER_BEAM_;                  // block id for a beam
    const int k = (topKs != nullptr) ? topKs[batchSlot] : maxTopK; // batchId = batch index

    const int logBufIndex = batchId * vocabSize;
    const int tmpLogBufIndex = batchId * vocabSize;
    const int tmpTopKBufIndex = batchId * BLOCKS_PER_BEAM_ * maxTopK + blockLane * k;

    TopK_2<T> partial;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    if (finished != nullptr && finishState.isFinished())
    {
        if (tid < k)
        {
            const int index = tmpTopKBufIndex + tid;
            if (blockLane == 0 && tid == 0)
            {
                const int endId = endIds[batchSlot];
                topKTmpIdBuf[index] = tmpLogBufIndex + endId;
                topKTmpValBuf[index] = logProbs[logBufIndex + endId];
            }
            else
            {
                topKTmpIdBuf[index] = -1;
                topKTmpValBuf[index] = -MAX_T_VAL;
            }
        }
        return;
    }

    for (int elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        int localIndex = elemId + tmpLogBufIndex;
        int globalIndex = elemId + logBufIndex;
        tmpLogProbs[localIndex] = logProbs[globalIndex];
    }

    for (int ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (int elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            int index = elemId + tmpLogBufIndex;
            partial.insert(tmpLogProbs[index], index);
        }

        TopK_2<T> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmpTopKBufIndex + ite;
            topKTmpIdBuf[index] = total.p;
            topKTmpValBuf[index] = total.u;
            if (total.p >= 0)
            {
                tmpLogProbs[total.p] = -MAX_T_VAL;
            }
        }
        __syncthreads();
    }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topKStage2Sampling(const int* __restrict topKTmpIdBuf, T* topKTmpValBuf, int** ids,
    int* sequenceLengths, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, const int maxTopK, const int* topKs, const float topP, const float* topPs,
    curandState_t* curandstate, const int* endIds, const int vocabSize, const bool* skipDecode, const int* batchSlots,
    int maxBatchSize, const bool normalizeLogProbs, const bool logitHasProbs)
{
    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    int const tid = threadIdx.x;
    auto const batchIdx = blockIdx.x;
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;
    FinishedState const finishState = finishedInput != nullptr ? finishedInput[batchSlot] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchSlot]) || (finishState.isSkipDecoding()))
    {
        return;
    }

    const int k = (topKs != nullptr) ? topKs[batchSlot] : maxTopK;
    const float probThreshold = (topPs != nullptr) ? topPs[batchSlot] : topP;
    const int size = k * BLOCKS_PER_BEAM_;
    const int stride = maxTopK * BLOCKS_PER_BEAM_;

    typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    extern __shared__ char array[];
    __shared__ float s_sum;
    T* s_val = topKTmpValBuf + batchIdx * stride;
    int* s_id = reinterpret_cast<int*>(array);
    if (tid == 0)
    {
        s_sum = 0.0f;
    }
    TopK_2<float> partial;

    if (finishState.isFinished())
    {
        if (finishedOutput != nullptr)
        {
            finishedOutput[batchSlot] = finishState;
        }
        return;
    }

    float* s_val2 = reinterpret_cast<float*>(s_id + k);
    float maxLogit;
    for (int ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE_)
        {
            partial.insert((float) s_val[i], i);
        }

        TopK_2<float> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<float>);

        if (tid == 0)
        {
            if (ite == 0)
            {
                maxLogit = total.u;
            }
            s_id[ite] = total.p;
            s_val[total.p] = -MAX_T_VAL;

            // when cumLogProbs are computed, topKTmpValBuf (logits_buf_) are
            // already pre-processed by softmax_kernel
            if (!logitHasProbs)
            {
                total.u = __expf(total.u - maxLogit);
            }
            s_val2[ite] = total.u;
            s_sum += total.u;
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float randNum = (float) curand_uniform(curandstate + batchSlot) * probThreshold * s_sum;
        for (int i = 0; i < k; i++)
        {
            float expLogit = s_val2[i];
            randNum = randNum - expLogit;
            if (randNum <= 0.0f || i == k - 1)
            {
                int idx = s_id[i];
                // If s_id is -1 here we force output token to the last from vocabulary to get vivid indicator of smth
                // going wrong for the debug
                auto outputId = idx != -1 ? topKTmpIdBuf[batchIdx * stride + idx] % vocabSize : vocabSize - 1;
                auto const curSeqLen = sequenceLengths[batchSlot];
                ids[batchSlot][curSeqLen] = outputId;
                if (cumLogProbs != nullptr || outputLogProbs != nullptr)
                {
                    float logProb = logf(expLogit);
                    if (cumLogProbs != nullptr)
                    {
                        cumLogProbs[batchSlot] += logProb;
                    }
                    if (outputLogProbs != nullptr)
                    {
                        // 'outputLogProbs' is the probability induced by the top-k sampling:
                        // NOT normalized (same way as OpenAI does):
                        // log_prob = log P(i | i is in top-k) = log(expLogit)
                        // normalized:
                        // log_prob = log P(i | i is in top-k) = log(expLogit / sum)
                        outputLogProbs[curSeqLen * maxBatchSize + batchSlot]
                            = normalizeLogProbs ? logProb - logf(s_sum) : logProb;
                    }
                }
                break;
            }
        }
        if (sequenceLengths != nullptr && finishedOutput != nullptr)
        {
            const int seqLen = sequenceLengths[batchSlot];
            if (ids[batchSlot][seqLen] == endIds[batchSlot])
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
}

#define CASE_K(K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_, normalizeLogProbs)                               \
    topKStage1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                                                                     \
        <<<batchSize * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>(logProbs, tempLogProbs, topKTmpIdBuf,             \
            topKTmpValBuf, finishedInput, maxTopK, topKs, vocabSize, endIds, skipDecode, batchSlots);                  \
    topKStage2Sampling<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                                             \
        <<<batchSize, BLOCK_SIZE_2_, K_MAX * sizeof(int) + K_MAX * sizeof(float), stream>>>(topKTmpIdBuf,              \
            topKTmpValBuf, ids, sequenceLengths, finishedInput, finishedOutput, cumLogProbs, outputLogProbs, maxTopK,  \
            topKs, topP, topPs, curandstate, endIds, vocabSize, skipDecode, batchSlots, maxBatchSize,                  \
            normalizeLogProbs, logitsHasProbs);                                                                        \
    break;

template <typename T>
void invokeBatchTopKSampling(void* workspace, size_t& workspaceSize, const T* logProbs, int** ids, int* sequenceLengths,
    const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    curandState_t* curandstate, const int maxTopK, const int* topKs, const float topP, const float* topPs,
    const int vocabSizePadded, const int* endIds, const int* batchSlots, cudaStream_t stream, const int batchSize,
    int maxBatchSize, const bool* skipDecode, const bool normalizeLogProbs, const bool logitsHasProbs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // Not allow an ambiguous inputs topP and topPs.
    assert(topP == 1.0f || topPs == nullptr);
    const int vocabSize = vocabSizePadded;
    const int maxBlockPerBeam = 8;
    int tempLogProbsBufSize = batchSize * vocabSize;                // type float
    int topKTmpIdsBufSize = batchSize * maxTopK * maxBlockPerBeam;  // type int
    int topKTmpValBuf_size = batchSize * maxTopK * maxBlockPerBeam; // type float

    // prevent memory misaligned address
    tempLogProbsBufSize = (int) (ceil(tempLogProbsBufSize / 4.)) * 4;
    topKTmpIdsBufSize = (int) (ceil(topKTmpIdsBufSize / 4.)) * 4;
    topKTmpValBuf_size = (int) (ceil(topKTmpValBuf_size / 4.)) * 4;

    if (workspace == nullptr)
    {
        workspaceSize
            = sizeof(T) * tempLogProbsBufSize + sizeof(int) * topKTmpIdsBufSize + sizeof(T) * topKTmpValBuf_size;
        return;
    }

    if (maxTopK == 0)
    {
        return;
    }

    T* tempLogProbs = (T*) workspace;
    int* topKTmpIdBuf = (int*) (tempLogProbs + tempLogProbsBufSize);
    T* topKTmpValBuf = (T*) (topKTmpIdBuf + topKTmpIdsBufSize);

    int logMaxTopK(0);
    int recursor(maxTopK - 1);
    while (recursor >>= 1)
        ++logMaxTopK;
    switch (logMaxTopK)
    {
    case 0:
    case 1:
    case 2:
    case 3: // 0 < maxTopK <= 16
        CASE_K(16, 128, 128, 8, normalizeLogProbs);
    case 4: // 16 < maxTopK <= 32
        CASE_K(32, 256, 128, 8, normalizeLogProbs);
    case 5: // 32 < maxTopK <= 64
        CASE_K(64, 256, 256, 8, normalizeLogProbs);
    case 6:
    case 7:
    case 8:
    case 9: // 64 < maxTopK <= 1024
        CASE_K(1024, 256, 256, 8, normalizeLogProbs);
    default: throw std::domain_error(fmtstr("top-k kernel supports 1<=k<=1024 but got k=%d", maxTopK));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

#undef CASE_K

template void invokeBatchTopKSampling(void* workspace, size_t& workspaceSize, const float* logProbs, int** ids,
    int* sequenceLengths, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, curandState_t* curandstate, const int maxTopK, const int* topKs, const float topP,
    const float* topPs, const int vocabSizePadded, const int* endIds, const int* batchSlots, cudaStream_t stream,
    const int batchSize, int maxBatchSize, const bool* skipDecode, const bool normalizeLogProbs,
    const bool logitsHasProbs);

template void invokeBatchTopKSampling(void* workspace, size_t& workspaceSize, const half* logProbs, int** ids,
    int* sequenceLengths, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, curandState_t* curandstate, const int maxTopK, const int* topKs, const float topP,
    const float* topPs, const int vocabSizePadded, const int* endIds, const int* batchSlots, cudaStream_t stream,
    const int batchSize, int maxBatchSize, const bool* skipDecode, const bool normalizeLogProbs,
    const bool logitsHasProbs);

template <typename T>
void invokeTopKSampling(void* workspace, size_t& workspaceSize, const T* logProbs, int** ids, int* sequenceLengths,
    const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    curandState_t* curandstate, const int topK, const float topP, const int vocabSizePadded, const int* endIds,
    const int* batchSlots, cudaStream_t stream, const int batchSize, int maxBatchSize, const bool* skipDecode,
    const bool normalizeLogProbs, const bool logitsHasProbs)
{
    invokeBatchTopKSampling(workspace, workspaceSize, logProbs, ids, sequenceLengths, finishedInput, finishedOutput,
        cumLogProbs, outputLogProbs, curandstate, topK, nullptr, topP, nullptr, vocabSizePadded, endIds, batchSlots,
        stream, batchSize, maxBatchSize, skipDecode, normalizeLogProbs, logitsHasProbs);
}

template void invokeTopKSampling(void* workspace, size_t& workspaceSize, const float* logProbs, int** ids,
    int* sequenceLengths, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, curandState_t* curandstate, const int topK, const float topP, const int vocabSizePadded,
    const int* endIds, const int* batchSlots, cudaStream_t stream, const int batchSize, int maxBatchSize,
    const bool* skipDecode, const bool normalizeLogProbs, const bool logitsHasProbs);

template void invokeTopKSampling(void* workspace, size_t& workspaceSize, const half* logProbs, int** ids,
    int* sequenceLengths, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, curandState_t* curandstate, const int topK, const float topP, const int vocabSizePadded,
    const int* endIds, const int* batchSlots, cudaStream_t stream, const int batchSize, int maxBatchSize,
    const bool* skipDecode, const bool normalizeLogProbs, const bool logitsHasProbs);

} // namespace kernels
} // namespace tensorrt_llm
