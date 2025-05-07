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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/kernels/decodingCommon.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

#pragma nv_diag_suppress static_var_with_dynamic_init

template <typename T, int PBM, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void beamStage1Kernel(T const* __restrict logProbs, T const* __restrict bias,
    float* __restrict pStage3, int const* __restrict endIds, FinishedState const* __restrict finished, int const nV,
    runtime::SizeType32 const* batchSlots)
{
    int const nBM = gridDim.y;
    int const tid = threadIdx.x;
    int const slot = batchSlots[blockIdx.x];
    int const nVLocal = (nV + gridDim.z - 1) / gridDim.z;
    int const indexLeft = nVLocal * blockIdx.z;
    int const indexRight = std::min(indexLeft + nVLocal, nV);
    int const nVOffset = (blockIdx.x * nBM + blockIdx.y) * nV;
    int const nVChunk = indexRight - indexLeft;
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    using KVPair = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<KVPair, BLOCK_SIZE>;
    cub::ArgMax argmax;

    __shared__ float smemOutput[PBM * 4];
    __shared__ int threadToUpdate;
    __shared__ typename BlockReduceTopK::TempStorage smemReduceBuffer;
    extern __shared__ char smem[];
    T* smemLogProbs = reinterpret_cast<T*>(smem);

    // Load element from logProbs to smemLogProbs and do argmax meanwhile
    // Each thread is responsible for `nVLocal / BLOCK_SIZE` elements
    // Dynamic shared memory size: sizeof(T) * (nV + nVPart - 1) / nVPart
    KVPair kvLocal{-1, -MAX_T_VAL};
    for (int i = indexLeft + tid; i < indexRight; i += BLOCK_SIZE)
    {
        T const b{bias == nullptr ? (T) 0.0f : bias[i]};
        int const index = i - indexLeft;
        T const value = (finished[slot * nBM + blockIdx.y].isFinished()) ? (i == endIds[slot] ? MAX_T_VAL : -MAX_T_VAL)
                                                                         : (logProbs[nVOffset + i] + b);
        smemLogProbs[index] = value;
        kvLocal = argmax(kvLocal, {index, value});
    }
    __syncthreads();

    // Search the top 2K elements among `nVLocal` elements of this ThreadBlock and write into smemOutput
    for (int i = 0; i < 2 * nBM; ++i)
    {
        // Pop the element with largest value to "smemOutput" per iteration
        KVPair kv = BlockReduceTopK(smemReduceBuffer).Reduce(kvLocal, argmax);
        if (tid == 0)
        {
            int const index = nVOffset + indexLeft + kv.key;
            reinterpret_cast<int*>(smemOutput)[i] = index;
            smemOutput[PBM * 2 + i] = kv.value;
            smemLogProbs[kv.key] = -MAX_T_VAL; // Invalidate the value of the popped element
            threadToUpdate = kv.key % BLOCK_SIZE;
        }
        __syncthreads();

        if (tid == threadToUpdate && i < 2 * nBM - 1)
        {
            // The thread popped the element need to update its kvLocal
            // No need to do this in the last iteration
            kvLocal.key = nV - 1;
            kvLocal.value = -MAX_T_VAL;
            for (int index = tid; index < nVChunk; index += BLOCK_SIZE)
            {
                kvLocal = argmax(kvLocal, {index, smemLogProbs[index]});
            }
        }
        __syncthreads();
    }
    // Write the smemOutput into pStage3
    pStage3 += (blockIdx.x * nBM + blockIdx.y) * gridDim.z * PBM * 4 + blockIdx.z * PBM * 4;
    for (int i = tid; i < PBM * 4; i += BLOCK_SIZE)
    {
        pStage3[i] = smemOutput[i];
    }
}

template <typename T, int PBM, int BLOCK_SIZE, bool IS_FAST>
__launch_bounds__(BLOCK_SIZE) __global__
    void beamStage2Kernel(int* __restrict pStage2Ids, T* __restrict pStage2LogProbs, float* __restrict pStage3,
        float const* __restrict cumLogProbs, runtime::SizeType32 const* batchSlots, int const nV, int const nVPart)
{
    int const nBM = gridDim.y;
    int const gbid = blockIdx.x * gridDim.y + blockIdx.y;
    int const tid = threadIdx.x;
    int const slot = batchSlots[blockIdx.x];
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    using KVPair = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<KVPair, BLOCK_SIZE>;
    cub::ArgMax argmax;

    __shared__ KVPair smemOutput[PBM * 2];
    __shared__ typename BlockReduceTopK::TempStorage smemReduceBuffer;

    // Load data from stage 1
    float* pStage2Temp = pStage3 + PBM * 4 * gbid * nVPart;
    if constexpr (IS_FAST)
    {
        // Use shared memory instead of global memory
        extern __shared__ char smem[];
        float* smemVal = reinterpret_cast<float*>(smem);
        for (int idx = tid; idx < PBM * 4 * nVPart; idx += BLOCK_SIZE)
        {
            smemVal[idx] = pStage2Temp[idx];
        }
        pStage2Temp = smemVal;
        __syncthreads();
    }

    // Find the top 2K across all nVPart
    for (int k = 0; k < 2 * nBM; ++k)
    {
        KVPair kvLocal{nV - 1, -MAX_T_VAL};
        if (tid < nVPart)
        {
            for (int i = 0; i < 2 * nBM; ++i)
            {
                int const index = tid * PBM * 4 + i;
                T const topValue = pStage2Temp[index + PBM * 2];
                kvLocal = argmax(kvLocal, {index, topValue});
            }
        }
        KVPair kv = BlockReduceTopK(smemReduceBuffer).Reduce(kvLocal, argmax);
        if (tid == 0)
        {
            // Replace local offset into global offset and store kv pairs in shared memory
            int const offsetLocal = kv.key;
            kv.key = reinterpret_cast<int*>(pStage2Temp)[offsetLocal];
            smemOutput[k] = kv;
            // Invalidate the maximum value within the chunk
            reinterpret_cast<int*>(pStage2Temp)[offsetLocal] = nV - 1; // id in shared memory
            pStage2Temp[offsetLocal + PBM * 2] = -MAX_T_VAL;           // value in shared memory
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        auto const cumLogProb = cumLogProbs[slot * nBM + blockIdx.y];
        for (int i = 0; i < 2 * nBM; ++i)
        {
            pStage2Ids[gbid * 2 * nBM + i] = smemOutput[i].key;
            pStage2LogProbs[gbid * 2 * nBM + i] = (float) smemOutput[i].value + cumLogProb;
        }
    }
}

template <typename T, int PBM, int BLOCK_SIZE, bool IS_FAST, bool IS_V2>
__launch_bounds__(BLOCK_SIZE) __global__ void beamStage3Kernel(
    int const* __restrict pStage2Ids, T const* __restrict pStage2LogProbs, float* __restrict pStage3, BeamHypotheses bh)
{
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;
    int const bid = blockIdx.x; // Index of Batch
    int const tid = threadIdx.x;
    int const slot = bh.batchSlots[bid];
    size_t const nMBS{bh.nMaxBatchSize}; // Only for bh.logProbsTiled
    size_t const nBM{bh.nBeamWidth};
    // size_t const nBMIn{bh.bVBWS ? bh.nBeamWidthIn : bh.nBeamWidth};
    size_t const nBMOut{bh.bVBWS ? bh.nBeamWidthOut : bh.nBeamWidth};
    size_t const nMSL{bh.nMaxSeqLen};
    size_t const nV{bh.nVocabSize};
    float const diversityRate{bh.diversityRates[slot]};
    float const lengthPenalty{bh.lengthPenalties[slot]};
    int const earlyStopping{bh.earlyStoppings[slot]};

    using KVPair = cub::KeyValuePair<int, T>;
    __shared__ float smemCumLogProbs[PBM];
    __shared__ int smemSeqLen[PBM];
    __shared__ KVPair smemTopKV[(IS_V2) ? 1 : PBM * 2]; // Just a placeholder in V2 workflow

    if (bh.numBeamsCBA != nullptr)
    {
        // Beam search is enabled
        if (bh.numBeamsCBA[slot] == 0 && tid == 0)
        {
            // Initialize worst score in the first call
            bh.minNormedScoresCBA[slot] = 0.0f; // logProbs is in range (-inf, 0]
        }
        else if (earlyStopping == 1 && bh.numBeamsCBA[slot] == nBM
            || earlyStopping != 1 && bh.finished[slot * nBM].isFinished())
        {
            // Condition of early return:
            // 1. In EarlyStopping mode, and we have got enough beams
            // 2. In NonEarlyStopping mode, and this batch has been marked as done
            // TODO: improve the condition like below
            // earlyStopping == 1 && bh.numBeamsCBA[slot] == nBM || earlyStopping != 1 && bh.batchDones[slot]
            return;
        }
    }

    // This TopK is needless in V2 workflow
    if constexpr (IS_V2)
    {
        pStage2Ids += bid * nBMOut * 2;
        pStage2LogProbs += bid * nBMOut * 2;
    }
    else
    {
        int const nCandidate = nBM * nBM * 2;
        pStage2Ids += bid * nCandidate;
        pStage2LogProbs += bid * nCandidate;
        KVPair kvLocal{nCandidate - 1, -MAX_T_VAL};
        cub::ArgMax argmax;
        extern __shared__ char smem[];
        T* smemVal = nullptr;
        if constexpr (IS_FAST)
        {
            smemVal = reinterpret_cast<T*>(smem);
        }
        else
        {
            smemVal = reinterpret_cast<T*>(pStage3);
        }

        for (int i = tid; i < nCandidate; i += BLOCK_SIZE)
        {
            int const index = bh.numBeamsCBA == nullptr ? i % nBM : i / 2 / nBM;
            T const value = pStage2LogProbs[i] + static_cast<T>(diversityRate * index);
            kvLocal = argmax(kvLocal, {i, value});
            smemVal[i] = value;
        }
        __syncthreads();

        using BlockReduce = cub::BlockReduce<KVPair, BLOCK_SIZE>;
        __shared__ typename BlockReduce::TempStorage smemReduceBuffer;
        __shared__ int threadToUpdate;

        for (int i = 0; i < 2 * nBM; ++i)
        {
            KVPair kv = BlockReduce(smemReduceBuffer).Reduce(kvLocal, argmax);
            if (tid == 0)
            {
                smemTopKV[i] = kv;
                smemVal[kv.key] = -MAX_T_VAL;
                threadToUpdate = kv.key % BLOCK_SIZE;
            }
            __syncthreads();
            // Only one thread needs to update the old partial before the next block reduce.
            // No need to do this in the last iteration.
            if (tid == threadToUpdate && i < 2 * nBM - 1)
            {
                kvLocal.key = nCandidate - 1;
                kvLocal.value = -MAX_T_VAL;
                for (int index = tid; index < nCandidate; index += BLOCK_SIZE)
                {
                    kvLocal = argmax(kvLocal, {index, smemVal[index]});
                }
            }
        }
    }

    if (tid < nBM) // Prepare cumLogProbs for later use
    {
        smemCumLogProbs[tid] = bh.cumLogProbs[slot * nBM + tid];
    }
    __syncthreads();

    if (tid == 0)
    {
        int nBeamForNextStep{0};
        // Select finished beams into CBA or select tokens for next step sequentially
        // Reference (might be changed along HF in the future):
        // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py#L272
        for (int i = 0; i < 2 * nBMOut; ++i)
        {
            int topId;
            T topLogProb;
            if constexpr (IS_V2)
            {
                // Get top token and correspongding logProb sequentially from pStage2Ids / pStage2LogProbs
                topId = pStage2Ids[i];
                topLogProb = pStage2LogProbs[i];
            }
            else
            {
                // Get top token and correspongding logProb by index of smemTopKV
                int const key = smemTopKV[i].key;
                topId = pStage2Ids[key];
                topLogProb = pStage2LogProbs[key];
            }
            bool const isEndToken = (topId % nV == bh.endIds[slot]);
            if (i < nBM && bh.numBeamsCBA != nullptr && isEndToken)
            {
                // Condition of this branch:
                // This token is end-token and belongs to top nBM range in Beam search mode
                int const nSeqLen = bh.sequenceLengths[slot * nBM + i] + 1 - bh.inputLengths[slot * nBM + i];
                float const score = applyLengthPenalty(topLogProb, nSeqLen, lengthPenalty);
                int nCBA = bh.numBeamsCBA[slot];
                if (nCBA == nBM)
                {
                    // There are already nBM beams
                    if (score < bh.minNormedScoresCBA[slot])
                    {
                        // Current score is worse than the worst one in candidate beams
                        if (earlyStopping)
                        {
                            // Stop since we have got enough beams
                            break;
                        }
                        else
                        {
                            // Continue since there might be longer but better beams
                            continue;
                        }
                    }
                    else
                    {
                        // Current score is better than the worst one in candidate beams
                        // Find the candidate beam index with the worst score and erase it
                        for (int j = 0; j < nBM; j++)
                        {
                            if (bh.normedScoresCBA[slot * (nBM * 2) + j] == bh.minNormedScoresCBA[slot])
                            {
                                nCBA = j;
                                bh.numBeamsCBA[slot]--;
                                bh.minNormedScoresCBA[slot] = FLT_MAX;
                                bh.normedScoresCBA[slot * (nBM * 2) + j] = score;
                                for (int l = 0; l < nBM; l++)
                                {
                                    bh.minNormedScoresCBA[slot]
                                        = min(bh.minNormedScoresCBA[slot], bh.normedScoresCBA[slot * (nBM * 2) + l]);
                                }
                                break;
                            }
                        }
                    }
                }
                // Copy finished beam from work tree to CBA
                // The last token
                int indexPrev = (topId / nV) % nBM;
                int const step = bh.sequenceLengths[slot * nBM + indexPrev];
                int const offsetCBA = (slot * nBM * 2 + nCBA) * nMSL;
                bh.outputIdsCBA[offsetCBA + step] = bh.endIds[slot];
                if (bh.logProbsCBA != nullptr)
                {
                    bh.logProbsCBA[offsetCBA + step] = (float) topLogProb - smemCumLogProbs[(topId / nV) % nBM];
                }
                // Previous tokens
                for (int j = step - 1; j >= 0; j--)
                {
                    bh.outputIdsCBA[offsetCBA + j] = bh.outputIdsPtr[slot][indexPrev * nMSL + j];
                    indexPrev = bh.parentIdsPtr[slot][indexPrev * nMSL + j];
                }
                if (bh.logProbsCBA != nullptr && bh.logProbsTiled != nullptr)
                {
                    indexPrev = (topId / nV) % nBM;
                    for (int j = step - 1; j >= 0; j--)
                    {
                        int const index = (j * nMBS + slot) * nBM + indexPrev;
                        bh.logProbsCBA[offsetCBA + j] = bh.logProbsTiled[index];
                        indexPrev = bh.parentIdsPtr[slot][indexPrev * nMSL + j];
                    }
                }
                // Other parameters
                int const index = slot * (nBM * 2) + nCBA;
                bh.sequenceLengthsCBA[index] = step;
                bh.normedScoresCBA[index] = score;
                bh.minNormedScoresCBA[slot] = min(bh.minNormedScoresCBA[slot], bh.normedScoresCBA[index]);
                bh.numBeamsCBA[slot]++;
                bh.cumLogProbsCBA[index] = (float) topLogProb;
            }
            else if (i < nBM || bh.numBeamsCBA != nullptr && !isEndToken)
            {
                // Condition of this branch
                // 1. bh.numBeamsCBA == nullptr && i <  nBM, i.e., beam search is disable
                // 2. bh.numBeamsCBA != nullptr && i <  nBM && isEndToken == false, i.e., add token at the end
                // 3. bh.numBeamsCBA != nullptr && i >= nBM && isEndToken == false, i.e., add token at the end
                int const step = bh.sequenceLengths[slot * nBM + nBeamForNextStep];
                // Copy the selected token to work tree
                bh.outputIdsPtr[slot][nBeamForNextStep * nMSL + step] = topId;
                if (bh.logProbsTiled != nullptr)
                {
                    int const index = step * nMBS * nBM + slot * nBM + nBeamForNextStep;
                    int const indexBeam = topId / nV % nBM;
                    bh.logProbsTiled[index] = (float) topLogProb - smemCumLogProbs[indexBeam];
                }
                bh.cumLogProbs[slot * nBM + nBeamForNextStep] = (float) topLogProb;
                nBeamForNextStep++;
            }
            else
            {
                // Condition of this branch, which we do nothing for it
                // 1. bh.numBeamsCBA == nullptr && i >= nBM, i.e., beam search is disable
                // 2. bh.numBeamsCBA != nullptr && i >= nBM && isEndToken == true, i.e., ignore the worse beams
            }

            if (nBeamForNextStep >= nBMOut)
            {
                // Condition of this branch
                // 1. In EarlyStopping mode, and get enough candidate beams
                // 2. In EarlyStopping mode, and get enough tokens for the next generation step
                // 3. In NonEarlyStopping mode, and get enough tokens for the next generation step
                // TODO: improve the condition like below
                // earlyStopping == 1 && bh.numBeamsCBA[slot] >= nBM || nBeamForNextStep >= nBM
                break;
            }
        }
    }

    // Update bh.batchDones
    if (tid == 0 && bh.numBeamsCBA != nullptr)
    {
        if (bh.numBeamsCBA[slot] < nBM)
        {
            // no enough beams
            bh.batchDones[slot] = false;
        }
        else if (earlyStopping == 1)
        {
            // enough candidate beams in EarlyStopping mode
            bh.batchDones[slot] = true;
        }
        else
        {
            // enough beams in NonEarlyStopping mode
            int nSeqLen = bh.sequenceLengths[slot * nBM] + 1 - bh.inputLengths[slot * nBM];
            float const bestCumLogProbs = (IS_V2) ? pStage2LogProbs[0] : smemTopKV[0].value;
            // According to semantics of HF, smemTopKV[0].value is used as bestCumLogProbs
            // But maybe bh.cumLogProbs[slot * nBM + i] is more suitable?
            // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py#L307
            if (earlyStopping != 0 && lengthPenalty > 0.0f)
            {
                // Specialization for earlyStopping == "never" and lengthPenalty > 0 in HF
                nSeqLen = nMSL - bh.inputLengths[slot * nBM];
            }
            float const bestAttainableScore = applyLengthPenalty(bestCumLogProbs, nSeqLen, lengthPenalty);
            bh.batchDones[slot] = bh.minNormedScoresCBA[slot] >= bestAttainableScore;
        }
    }
    __syncthreads();

    // Update sequenceLengths, parentIdsPtr, outputIdsPtr and finished
    if (tid < nBM)
    {
        smemSeqLen[tid] = bh.sequenceLengths[slot * nBM + tid];
    }
    __syncthreads();

    if (tid < nBMOut)
    {
        int const indexBatchBeam = slot * nBM + tid;
        int const step = smemSeqLen[tid];
        if (!bh.finished[indexBatchBeam].isFinished())
        {
            smemSeqLen[tid]++;
        }
        int const newId = bh.outputIdsPtr[slot][tid * nMSL + step];
        int const newBeamId = (newId / nV) % nBM;
        int const newTokenId = newId % nV;
        bh.sequenceLengths[indexBatchBeam] = smemSeqLen[newBeamId];
        if (newTokenId == bh.endIds[slot])
        {
            bh.finished[indexBatchBeam].setFinishedEOS();
        }
        bh.parentIdsPtr[slot][tid * nMSL + step] = newBeamId;
        bh.outputIdsPtr[slot][tid * nMSL + step] = newTokenId;

        if ((earlyStopping == 1) && (bh.numBeamsCBA != nullptr && bh.numBeamsCBA[slot] == nBM)
            || (earlyStopping != 1) && bh.batchDones[slot])
        {
            bh.batchDones[slot] = true;
            bh.finished[indexBatchBeam].setFinished();
        }
    }
}

template <typename T, int PBM, bool IS_V2>
void beamSearchKernelLauncher(
    T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream)
{
    // clang-format off
    /*
    V1 Workflow (reference: https://github.com/NVIDIA/online-softmax):
    logProbs.shape = [nBS, nBM, nV]
             nV               |<- nVChunk ->|<- nVChunk ->| <- ... ->|          |<- nBM*4 ->|<- nBM*4 ->|<- ... ->| ■
        ┏━━━━━━━━━━┓          ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓          ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃nBM       ┃          ┃nBM                                   ┃          ┃nBM                              ┃
        ┣━━━━━━━━━━┫          ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  A       ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  B
    nBS ┃nBM       ┃ ---> nBS ┃nBM                                   ┃ ---> nBS ┃nBM                              ┃ --->
        ┣━━━━━━━━━━┫          ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫          ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
        ┃nBM       ┃          ┃nBM                                   ┃          ┃nBM                              ┃
        ┗━━━━━━━━━━┛          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
          logProbs            divide `nV` elements into `nVPart` parts          pStage3 with `nVPart` tiles per row

             |<- nBm*2 ->|  |<- nBm*2 ->|
             ┏━━━━━━━━━━━┓  ┏━━━━━━━━━━━┓
             ┃nBM        ┃  ┃nBM        ┃
     B       ┣━━━━━━━━━━━┫  ┣━━━━━━━━━━━┫  C
    ---> nBS ┃nBM        ┃  ┃nBM        ┃ --->
             ┣━━━━━━━━━━━┫  ┣━━━━━━━━━━━┫
             ┃nBM        ┃  ┃nBM        ┃
             ┗━━━━━━━━━━━┛  ┗━━━━━━━━━━━┛
               pStage2Ids  pStage2LogProbs

    ■: Each "tile" in pStage3 with shape [`nBM*4`] contains `nBM*2` top ids and corresponding `nBM*2` log probs.
        |<- nBm*2 ->|<- nBm*2 ->|
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
      1 ┃  top ids  | log probs ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━┛

    A: beamStage1Kernel: gridDim(BS,BM,nVPart), blockDim(nThreadStage1,1,1)
                         Each Block takes `nVChunk` contiguous elements from `logProbs`, does TopK and writes output to `pStage3`
    B: beamStage2Kernel: gridDim(BS,BM,1), blockDim(32/64/128,1,1)
                         Each Block takes `nVPart` contiguous tiles from pStage3, add `cumLogProbs`, does TopK` and writes output to `pStage2Ids` and `pStage2LogProbs`
    C: beamStage3Kernel: gridDim(BS,1,1), blockDim(128,1,1)
                         Main logic of Beam-Search, each Block is responsible for one batch, doing work below:
                             + moves one beam into candidate-beam-array if it is finished (gemerated end_id in this step).
                             + selects BM elements for the next generation step if not.
                             + maintains related score array, min_normed_score / batchDones / finished, etc..

    ===================================================================================================================================

    V2 Workflow (use Air-TopK for better performance, https://dl.acm.org/doi/pdf/10.1145/3581784.3607062)
    logProbs.shape = [nBS, nBM, nV]
        |<- nV ->|          |<- nBM*2 ->|  |<- nBM*2 ->|          |<- nBM*2 ->|          |<- nBM*2 ->|          |<- nBM*2 ->|
        ┏━━━━━━━━┓          ┏━━━━━━━━━━━┓  ┏━━━━━━━━━━━┓          ┏━━━━━━━━━━━┓          ┏━━━━━━━━━━━┓  D       ┏━━━━━━━━━━━┓
        ┃nBM     ┃          ┃nBM        ┃  ┃nBM        ┃          ┃nBM        ┃      nBS ┃           ┃ ---> nBS ┃           ┃ ---\
        ┣━━━━━━━━┫  A       ┣━━━━━━━━━━━┫  ┣━━━━━━━━━━━┫  B       ┣━━━━━━━━━━━┫  C       ┗━━━━━━━━━━━┛          ┗━━━━━━━━━━━┛    | E
    nBS ┃nBM     ┃ ---> nBS ┃nBM        ┃  ┃nBM        ┃ ---> nBS ┃nBM        ┃ --->       pStage2Id              pStage2Id      |--->
        ┣━━━━━━━━┫          ┣━━━━━━━━━━━┫  ┣━━━━━━━━━━━┫          ┣━━━━━━━━━━━┫          ┏━━━━━━━━━━━┓                           |
        ┃nBM     ┃          ┃nBM        ┃  ┃nBM        ┃          ┃nBM        ┃      nBS ┃           ┃ --------------------------/
        ┗━━━━━━━━┛          ┗━━━━━━━━━━━┛  ┗━━━━━━━━━━━┛          ┗━━━━━━━━━━━┛          ┗━━━━━━━━━━━┛
         logProbs             pStage1Id   pStage1LogProbs        pStage1LogProbs        pStage2LogProbs

    A: TopK            : Get top `nBM*2` elements in `nBS*nBM` groups (`nV` elements per group)
    B: addCumLogProbs  : Add `cumLogProbs` to the elements in each beam
    C: TopK            : Get top `nBM*2` elements in `nBS` group (`nBM*nBM*2` elements per group)
    D: gatherIds       : Combine stage1Id and stage2Id to get ids of the top `nBM*2` elements in input logProbs
    E: beamStage3Kernel: Main logic of Beam-Search, each Block is responsible for one batch, doing work below:
                             + moves one beam into candidate-beam-array if it is finished (gemerated end_id in this step).
                             + selects BM elements for the next generation step if not.
                             + maintains related score array, min_normed_score / batchDones / finished, etc..

    ===================================================================================================================================

    V2 Workflow for VBWS, similar to V2 workflow above, but `nBMIn` and `nBMOut` might be different from `nBM`
    logProbs.shape = [nBS, nBMIn, nV]
        |<- nV ->|          |<- nBMOut*2 ->|  |<- nBMOut*2 ->|          |<- nBMOut*2 ->|          |<- nBMOut*2 ->|          |<- nBMOut*2 ->|
        ┏━━━━━━━━┓          ┏━━━━━━━━━━━━━━┓  ┏━━━━━━━━━━━━━━┓          ┏━━━━━━━━━━━━━━┓          ┏━━━━━━━━━━━━━━┓  D       ┏━━━━━━━━━━━━━━┓
        ┃nBMIn   ┃          ┃nBMIn         ┃  ┃nBMIn         ┃          ┃nBMIn         ┃      nBS ┃              ┃ ---> nBS ┃              ┃ ---\
        ┣━━━━━━━━┫  A       ┣━━━━━━━━━━━━━━┫  ┣━━━━━━━━━━━━━━┫  B       ┣━━━━━━━━━━━━━━┫  C       ┗━━━━━━━━━━━━━━┛          ┗━━━━━━━━━━━━━━┛    | E
    nBS ┃nBMIn   ┃ ---> nBS ┃nBMIn         ┃  ┃nBMIn         ┃ ---> nBS ┃nBMIn         ┃ --->         pStage2Id                 pStage2Id       |--->
        ┣━━━━━━━━┫          ┣━━━━━━━━━━━━━━┫  ┣━━━━━━━━━━━━━━┫          ┣━━━━━━━━━━━━━━┫          ┏━━━━━━━━━━━━━━┓                              |
        ┃nBMIn   ┃          ┃nBMIn         ┃  ┃nBMIn         ┃          ┃nBMIn         ┃      nBS ┃              ┃ -----------------------------/
        ┗━━━━━━━━┛          ┗━━━━━━━━━━━━━━┛  ┗━━━━━━━━━━━━━━┛          ┗━━━━━━━━━━━━━━┛          ┗━━━━━━━━━━━━━━┛
         logProbs               pStage1Id      pStage1LogProbs           pStage1LogProbs           pStage2LogProbs
    */
    // clang-format on

    size_t const nBS{bh.nBatchSize};
    size_t const nBM{bh.nBeamWidth};
    size_t const nV{bh.nVocabSize};
    size_t const nVPart{bh.nVPart};
    size_t const nByteMaxSharedMemoryPerBlock{bh.nByteMaxSharedMemoryPerBlock};
    int* pStage2Ids{nullptr};
    T* pStage2LogProbs{nullptr};
    float* pStage3{nullptr};

    // VBWS:
    //     + `nBMIn` / `nBMOut` is the beam width in the last / next network forward computation respectively
    //     + `nBM` is the max value of the beam width array, which is used for memory allocatation
    // Normal Beam Search:
    //     + `nBMIn` / `nBMOut` / `nBM` share the same value
    // TODO: now `nBMIn` and `nBMOut` of request 0 is used for the whole batch,
    //     change to corresponding BMs if Diverse-Beam-Width-Search is supported
    size_t const nBMIn = bh.bVBWS ? bh.nBeamWidthInHost[0] : nBM;
    size_t const nBMOut = bh.bVBWS ? bh.nBeamWidthOutHost[0] : nBM;
    bh.nBeamWidthIn = nBMIn;   // Save nBMIn back to bh
    bh.nBeamWidthOut = nBMOut; // Save nBMOut back to bh

    if constexpr (IS_V2)
    {
        // see `BeamSearchLayer<T>::configureBeamSearchLayer()` for the workspace structure
        size_t const offsetStage1 = roundUp(nBS * nBM * nBM * 2, 4);
        size_t const offsetStage2 = roundUp(nBS * nBM * 2, 4);
        pStage2Ids = reinterpret_cast<int*>(workspace);
        int offset = sizeof(int) * offsetStage2;
        pStage2LogProbs = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + offset);
        offset += sizeof(T) * offsetStage2;
        int* pStage1Ids = reinterpret_cast<int*>(reinterpret_cast<char*>(workspace) + offset);
        pStage3 = reinterpret_cast<float*>(reinterpret_cast<char*>(workspace) + offset);
        offset += sizeof(int) * offsetStage1;
        T* pStage1LogProbs = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + offset);
        offset += sizeof(T) * offsetStage1;
        void* pTopK = reinterpret_cast<void*>(reinterpret_cast<char*>(workspace) + offset);

        // Stage 1
        invokeTopkLastDim<T>(nBS * nBMIn, nV, nBMOut * 2, true, logProbs, pStage1LogProbs, pStage1Ids, pTopK, stream);
        sync_check_cuda_error(stream);

        int nThread = min(roundUp(nBMIn * nBMOut * 2, 32), 1024);
        addCumLogProbs<<<nBS, nThread, 0, stream>>>(pStage1LogProbs, bh.cumLogProbs, bh.finished, bh.endIds,
            bh.diversityRates, bh.batchSlots, nBS, nBMIn, nBMOut, nBM);
        sync_check_cuda_error(stream);

        // Stage 2
        invokeTopkLastDim<T>(
            nBS, nBMIn * nBMOut * 2, nBMOut * 2, true, pStage1LogProbs, pStage2LogProbs, pStage2Ids, pTopK, stream);
        sync_check_cuda_error(stream);

        nThread = min(roundUp(nBMOut * 2, 32), 1024);
        gatherId<<<nBS, nThread, 0, stream>>>(pStage1Ids, pStage2Ids, nBS, nBMIn, nBMOut, nV);
        sync_check_cuda_error(stream);
    }
    else // V1
    {
        // see `BeamSearchLayer<T>::configureBeamSearchLayer()` for the workspace structure
        int const offset = roundUp(nBS * nBM * nBM * 2, 4);
        pStage2Ids = reinterpret_cast<int*>(workspace);
        pStage2LogProbs = reinterpret_cast<T*>(pStage2Ids + offset);
        pStage3 = reinterpret_cast<float*>(pStage2LogProbs + offset);

        // Stage 1
        size_t constexpr nThreadStage1 = (PBM < 16) ? ((PBM < 8) ? kThreadForSmallBeamWidth : 128) : 64;
        dim3 grid(nBS, nBM, bh.nVPart), block(nThreadStage1);
        beamStage1Kernel<T, PBM, nThreadStage1><<<grid, block, bh.nByteSharedMemoryStage1, stream>>>(
            logProbs, bias, pStage3, bh.endIds, bh.finished, nV, bh.batchSlots);
        sync_check_cuda_error(stream);

// Stage 2
#define BEAM_STAGE2_KERNEL(N_VOCAB_PART, IS_FAST)                                                                      \
    {                                                                                                                  \
        if (IS_FAST && nByteRuntimeSharedMemory > (48 << 10))                                                          \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(beamStage2Kernel<T, PBM, N_VOCAB_PART, IS_FAST>,                      \
                cudaFuncAttributeMaxDynamicSharedMemorySize, nByteRuntimeSharedMemory));                               \
        }                                                                                                              \
        beamStage2Kernel<T, PBM, N_VOCAB_PART, IS_FAST>                                                                \
            <<<dim3(nBS, nBM), N_VOCAB_PART, IS_FAST * nByteRuntimeSharedMemory, stream>>>(                            \
                pStage2Ids, pStage2LogProbs, pStage3, bh.cumLogProbs, bh.batchSlots, nV, nVPart);                      \
    }
        // TODO: rewrite kernel to remove dependence of constant block size to reduce compilation time
        size_t nByteRuntimeSharedMemory
            = sizeof(float) * nVPart * (PBM * 4) + sizeof(cub::KeyValuePair<int, T>) * PBM * 2;
        if (nByteRuntimeSharedMemory <= nByteMaxSharedMemoryPerBlock && nVPart <= 32)
        {
            BEAM_STAGE2_KERNEL(32, true)
        }
        else if (nByteRuntimeSharedMemory <= nByteMaxSharedMemoryPerBlock && nVPart <= 64)
        {
            BEAM_STAGE2_KERNEL(64, true)
        }
        else if (nByteRuntimeSharedMemory <= nByteMaxSharedMemoryPerBlock)
        {
            BEAM_STAGE2_KERNEL(128, true)
            // No branch with larger `N_VOCAB_PART` since nVPart <= kMaxVPartStage1 == 128
        }
        else
        {
            TLLM_LOG_TRACE("Use slow Beam Search stage 2 kernel due to large beam_width or vocab_size");
            BEAM_STAGE2_KERNEL(128, false)
        }
        sync_check_cuda_error(stream);
#undef BEAM_STAGE2_KERNEL
    }

    // Stage 3 in common
    size_t constexpr nThreadStage3 = (PBM + 31) / 32 * 32;
    size_t const nByteStaticSharedMemory = bh.nByteSharedMemoryStage3;
    size_t const nByteDynamicSharedMemory = (IS_V2) ? 0 : sizeof(T) * nBM * nBM * 2;
    size_t const nByteRuntimeSharedMemory = nByteStaticSharedMemory + nByteDynamicSharedMemory;

    if (nByteRuntimeSharedMemory <= nByteMaxSharedMemoryPerBlock)
    {
        if (nByteRuntimeSharedMemory > (48 << 10))
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(beamStage3Kernel<T, PBM, nThreadStage3, true, IS_V2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, nByteRuntimeSharedMemory));
        }
        beamStage3Kernel<T, PBM, nThreadStage3, true, IS_V2>
            <<<nBS, nThreadStage3, nByteDynamicSharedMemory, stream>>>(pStage2Ids, pStage2LogProbs, pStage3, bh);
    }
    else
    {
        if (nByteStaticSharedMemory > (48 << 10))
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(beamStage3Kernel<T, PBM, nThreadStage3, false, IS_V2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, nByteStaticSharedMemory));
        }
        beamStage3Kernel<T, PBM, nThreadStage3, false, IS_V2>
            <<<nBS, nThreadStage3, 0, stream>>>(pStage2Ids, pStage2LogProbs, pStage3, bh);
    }
    sync_check_cuda_error(stream);

    return;
}

#undef BEAM_STAGE2_KERNEL

#define INSTANTIATE_BEAM_SEARCH(T, PBM, IS_V2)                                                                         \
    template void beamSearchKernelLauncher<T, PBM, IS_V2>(                                                             \
        T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
