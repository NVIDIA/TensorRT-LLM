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

#define DO_SPLIT_SMALL_TOP_K_SOFTMAX

#define TOPK_FP16_STORAGE 0

#pragma nv_diag_suppress static_var_with_dynamic_init

template <typename T, int PAD_2K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beamStage3Kernel(int const* __restrict pTempId, T const* __restrict pTempVal, BeamHypotheses bh)
{
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const gbid{bh.nIte * bh.nBatchSizeLocal + bid}; // global batch index
    int const nBM{bh.nBeamWidth};
    int const nV{bh.nVocabSize};
    int const nCandidate{nBM * nBM * 2}; //  We extract top 2K candidates from each beam output
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    float const diversity_rate{bh.diversityRates[gbid]};
    float const length_penalty{bh.lengthPenalties[gbid]};
    int const early_stopping{bh.earlyStoppings[gbid]};

    __shared__ int nBeamForNextStep;
    __shared__ float smem_cum_log_probs[PAD_2K / 2];

    if (tid == 0)
    {
        nBeamForNextStep = 0;
    }
    if (tid < nBM)
    {
        smem_cum_log_probs[tid] = bh.cumLogProbs[bid * nBM + tid];
    }
    __syncthreads();

    if (bh.numBeamsCBA != nullptr)
    {
        // Beam search is enabled
        if (bh.numBeamsCBA[gbid] == 0 && tid == 0)
        {
            // Initialize worst_score in the first time
            bh.minNormedScoresCBA[gbid] = FLT_MAX;
        }
        else if (early_stopping == 1 && bh.numBeamsCBA[gbid] == nBM
            || early_stopping != 1 && bh.finished[bid * nBM].isFinished())
        {
            // Condition of early return:
            // 1. In EarlyStopping mode, and we have got enough beams
            // 2. In NonEarlyStopping mode, and this batch has been marked as done
            // TODO: improve the condition like below:
            // else if (early_stopping == 1 && bh.numBeamsCBA[gbid] == nBM || early_stopping != 1 && bh.batchDones[bid])
            return;
        }
    }

    // Get top 2K tokens from candidates
    pTempId += bid * nCandidate;
    pTempVal += bid * nCandidate;

    using cub_kvp = cub::KeyValuePair<int, T>;
    cub_kvp partial_topk{nCandidate - 1, -MAX_T_VAL};
    cub::ArgMax arg_max;
    extern __shared__ char smem[];
    T* smem_topk = reinterpret_cast<T*>(smem);

    for (int id = tid; id < nCandidate; id += THREADBLOCK_SIZE)
    {
        int const index = bh.numBeamsCBA == nullptr ? id % nBM : id / 2 / nBM;
        T val = pTempVal[id] + static_cast<T>(diversity_rate * index);
        cub_kvp new_elem{id, val};
        partial_topk = arg_max(partial_topk, new_elem);
        smem_topk[id] = val;
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage reduce_buffer;
    __shared__ cub_kvp cta_topk[PAD_2K];
    __shared__ int thread_requiring_update;

    for (int i = 0; i < 2 * nBM; ++i)
    {
        cub_kvp total_topk = BlockReduce(reduce_buffer).Reduce(partial_topk, arg_max);
        if (tid == 0)
        {
            cta_topk[i] = total_topk;
            smem_topk[total_topk.key] = -MAX_T_VAL;
            thread_requiring_update = total_topk.key % THREADBLOCK_SIZE;
        }
        __syncthreads();
        // Only one thread needs to update the old partial before the next block reduce.
        // No need to do this in the last iteration.
        if (tid == thread_requiring_update && i < (2 * nBM - 1))
        {
            partial_topk.key = nCandidate - 1;
            partial_topk.value = -MAX_T_VAL;
            for (int index = tid; index < nCandidate; index += THREADBLOCK_SIZE)
            {
                cub_kvp new_elem{index, smem_topk[index]};
                partial_topk = arg_max(partial_topk, new_elem);
            }
        }
    }

    if (tid == 0)
    {
        // Select finished beams into CBA or select tokens for next step sequentially
        // Reference (might be changed along HF in the future):
        // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py#L272
        for (int i = 0; i < 2 * nBM; ++i)
        {
            int const current_key = cta_topk[i].key;
            T const current_value = cta_topk[i].value;
            bool const is_end_token = pTempId[current_key] % nV == bh.endIds[bid];
            if (i < nBM && bh.numBeamsCBA != nullptr && is_end_token)
            {
                // Condition of this branch
                // This token is end-token and belongs to top nBM range in Beam search mode
                int const nSeqLen = bh.sequenceLengths[bid * nBM + i] + 1 - bh.inputLengths[gbid * nBM + i];
                float const normed_score = applyLengthPenalty(current_value, nSeqLen, length_penalty);
                int beam_idx = bh.numBeamsCBA[gbid];
                if (beam_idx == nBM)
                {
                    // There are already nBM beams
                    if (normed_score < bh.minNormedScoresCBA[gbid])
                    {
                        // Current score is worse than the worst one in candidate beams
                        if (early_stopping)
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
                            if (bh.normedScoresCBA[gbid * (nBM * 2) + j] == bh.minNormedScoresCBA[gbid])
                            {
                                beam_idx = j;
                                bh.numBeamsCBA[gbid]--;
                                bh.minNormedScoresCBA[gbid] = FLT_MAX;
                                bh.normedScoresCBA[gbid * (nBM * 2) + j] = normed_score;
                                for (int l = 0; l < nBM; l++)
                                {
                                    bh.minNormedScoresCBA[gbid]
                                        = min(bh.minNormedScoresCBA[gbid], bh.normedScoresCBA[gbid * (nBM * 2) + l]);
                                }
                                break;
                            }
                        }
                    }
                }
                int prev_id = (pTempId[current_key] / nV) % nBM;
                int const current_step = bh.sequenceLengths[bid * nBM + prev_id];
                int const id_offset_cba = (gbid * nBM * 2 + beam_idx) * bh.nMaxSeqLen;
                bh.outputIdsCBA[id_offset_cba + current_step] = bh.endIds[bid];
                if (bh.logProbsCBA != nullptr)
                {
                    bh.logProbsCBA[id_offset_cba + current_step]
                        = (float) pTempVal[current_key] - smem_cum_log_probs[(pTempId[current_key] / nV) % nBM];
                }
                // Copy finished beam from work tree to CBA
                for (int j = current_step - 1; j >= 0; j--)
                {
                    bh.outputIdsCBA[id_offset_cba + j] = bh.outputIdsPtr[bid][prev_id * bh.nMaxSeqLen + j];
                    prev_id = bh.parentIdsPtr[bid][prev_id * bh.nMaxSeqLen + j];
                }
                if (bh.logProbsCBA != nullptr && bh.logProbs != nullptr)
                {
                    prev_id = (pTempId[current_key] / nV) % nBM;
                    for (int j = current_step - 1; j >= 0; j--)
                    {
                        int const index = (j * bh.nBatchSize + gbid) * nBM + prev_id;
                        bh.logProbsCBA[id_offset_cba + j] = bh.logProbs[index];
                        prev_id = bh.parentIdsPtr[bid][prev_id * bh.nMaxSeqLen + j];
                    }
                }
                int const beam_idx_cba = gbid * (nBM * 2) + beam_idx;
                bh.sequenceLengthsCBA[beam_idx_cba] = current_step;
                bh.normedScoresCBA[beam_idx_cba] = normed_score;
                bh.minNormedScoresCBA[gbid] = min(bh.minNormedScoresCBA[gbid], bh.normedScoresCBA[beam_idx_cba]);
                bh.numBeamsCBA[gbid]++;
                bh.cumLogProbsCBA[beam_idx_cba] = (float) pTempVal[current_key];
            }
            else if (i < nBM || bh.numBeamsCBA != nullptr && !is_end_token)
            {
                // Condition of this branch
                // 1. bh.numBeamsCBA == nullptr && i <  nBM, i.e., beam search is disable
                // 2. bh.numBeamsCBA != nullptr && i <  nBM && is_end_token == false, i.e., add token at the end
                // 3. bh.numBeamsCBA != nullptr && i >= nBM && is_end_token == false, i.e., add token at the end
                int const current_step = bh.sequenceLengths[bid * nBM + nBeamForNextStep];
                // Copy the selected token to work tree
                bh.outputIdsPtr[bid][nBeamForNextStep * bh.nMaxSeqLen + current_step] = pTempId[current_key];
                if (bh.logProbs != nullptr)
                {
                    bh.logProbs[current_step * bh.nBatchSize * nBM + bid * nBM + nBeamForNextStep]
                        = (float) pTempVal[current_key] - smem_cum_log_probs[(pTempId[current_key] / nV) % nBM];
                }
                bh.cumLogProbs[bid * nBM + nBeamForNextStep] = (float) pTempVal[current_key];
                nBeamForNextStep++;
            }
            else
            {
                // Condition of this branch, which we do nothing for it
                // 1. bh.numBeamsCBA == nullptr && i >= nBM, i.e., beam search is disable
                // 2. bh.numBeamsCBA != nullptr && i >= nBM && is_end_token == true, i.e., ignore the worse beams
            }

            // if (early_stopping == 1 && bh.numBeamsCBA[gbid] >= nBM || nBeamForNextStep >= nBM)
            if (nBeamForNextStep >= nBM)
            {
                // Condition of this branch
                // 1. In EarlyStopping mode, and get enough candidate beams
                // 2. In EarlyStopping mode, and get enough tokens for the next generation step
                // 3. In NonEarlyStopping mode, and get enough tokens for the next generation step
                break;
            }
        }
    }

    // Update bh.batchDones
    if (tid == 0 && bh.numBeamsCBA != nullptr)
    {
        if (bh.numBeamsCBA[bid] < nBM)
        {
            // no enough beams
            bh.batchDones[bid] = false;
        }
        else if (early_stopping == 1)
        {
            // enough candidate beams in EarlyStopping mode
            bh.batchDones[bid] = true;
        }
        else
        {
            // enough beams in NonEarlyStopping mode
            int nSeqLen = bh.sequenceLengths[bid * nBM] + 1 - bh.inputLengths[gbid * nBM];
            float const best_sum_logprobs = cta_topk[0].value;
            // According to semantics of HF, cta_topk[0].value is used as best_sum_logprobs
            // But maybe bh.cumLogProbs[bid * nBM + i] is more suitable?
            // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py#L307
            if (early_stopping != 0 && length_penalty > 0.0f)
            {
                // Specialization for early_stopping == "never" and length_penalty > 0 in HF
                nSeqLen = bh.nMaxSeqLen - bh.inputLengths[gbid * nBM];
            }
            float const highest_attainable_score = applyLengthPenalty(best_sum_logprobs, nSeqLen, length_penalty);
            bh.batchDones[bid] = bh.minNormedScoresCBA[gbid] >= highest_attainable_score;
        }
    }
    __syncthreads();

    // Update sequence_lengths, parent_ids, output_ids and finished
    __shared__ int s_sequence_lengths[PAD_2K / 2];
    if (tid < nBM)
    {
        s_sequence_lengths[tid] = bh.sequenceLengths[bid * nBM + tid];
    }
    __syncthreads();

    if (tid < nBM)
    {
        int const bb_index = bid * nBM + tid;
        int const current_step = s_sequence_lengths[tid];
        if (!bh.finished[bb_index].isFinished())
        {
            s_sequence_lengths[tid]++;
        }
        int const new_id = bh.outputIdsPtr[bid][tid * bh.nMaxSeqLen + current_step];
        int const new_beam_id = (new_id / nV) % nBM;
        int const new_word_id = new_id % nV;
        bh.sequenceLengths[bb_index] = s_sequence_lengths[new_beam_id];
        if (new_word_id == bh.endIds[bid])
        {
            bh.finished[bb_index].setFinishedEOS();
        }
        bh.parentIdsPtr[bid][tid * bh.nMaxSeqLen + current_step] = new_beam_id;
        bh.outputIdsPtr[bid][tid * bh.nMaxSeqLen + current_step] = new_word_id;
        if ((early_stopping == 1) && (bh.numBeamsCBA != nullptr && bh.numBeamsCBA[gbid] == nBM)
            || (early_stopping != 1) && bh.batchDones[bid])
        {
            bh.batchDones[bid] = true;
            bh.finished[bb_index].setFinished();
        }
    }
}

struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool const is_a_bigger = a.m > b.m;
    MD const bigger = is_a_bigger ? a : b;
    MD const smaller = is_a_bigger ? b : a;
    MD res{bigger.m, bigger.d + smaller.d * __expf(smaller.m - bigger.m)};
    return res;
}

template <typename T, int PAD_K>
struct TopKMD
{
    MD md;
    TopK<T, PAD_K> topk;
};

template <typename T, int PAD_K>
__device__ __forceinline__ TopKMD<T, PAD_K> reduce_topk_md_op(TopKMD<T, PAD_K> const& a, TopKMD<T, PAD_K> const& b)
{
    TopKMD<T, PAD_K> res;
    res.md = reduce_md_op(a.md, b.md);
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

template <typename T, int ITEMS_PER_THREAD, int PAD_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beamKernel(T const* __restrict logits, T const* __restrict bias,
    int* __restrict pTempId, T* __restrict pTempVal, BeamHypotheses bh)
{
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const nBM{bh.nBeamWidth};
    int const nV{bh.nVocabSize};
    int const* endIds{bh.endIds};
    float const* cum_log_probs{bh.cumLogProbs};
    FinishedState const* finished{bh.finished};
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    TopKMD<float, PAD_K> partial;
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;
    partial.topk.init();

    if (finished[bid].isFinished())
    {
        for (int id = tid; id < nV; id += THREADBLOCK_SIZE)
        {
            float const val = id == endIds[bid / nBM] ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(val, id);
        }
    }
    else
    {
        T const* local_logits = logits + bid * nV;
        for (int id = tid; id < nV; id += THREADBLOCK_SIZE)
        {
            float const val = local_logits[id] + bias[id];
            MD new_elem{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(val, id);
        }
    }

    typedef cub::BlockReduce<TopKMD<float, PAD_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_buffer;

    TopKMD<float, PAD_K> total = BlockReduce(reduce_buffer).Reduce(partial, reduce_topk_md_op<float, PAD_K>);

    if (tid == 0)
    {
        int* local_topk_id = pTempId + bid * nBM;
        T const* local_topk_val = pTempVal + bid * nBM;
        float const total_m = total.md.m;
        float const total_d = logf(total.md.d);
        float local_cum_log_probs = cum_log_probs[bid];
        for (int i = 0; i < nBM; ++i)
        {
            local_topk_id[i] = total.topk.p[i] + bid * nV;
            local_topk_val[i] = total.topk.u[i] - total_m - total_d + local_cum_log_probs;
        }
    }
}

template <typename T, int ITEMS_PER_THREAD, int PAD_2K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__
    void beamStage1BaseKernel(T const* __restrict logits, T const* __restrict bias, float* __restrict pTemp,
        int const* __restrict endIds, FinishedState const* __restrict finished, int nBM, int nV)
{
    // Compare to beamStage1FastKernel, here is no share memory for storage of logits,
    // and each ThreadBlock is responsible for `nV / nVPart` elements
    constexpr int PACKED_TOP_KMD_SIZE = 2 * PAD_2K + 2;
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const nVLocal = (nV + gridDim.y - 1) / gridDim.y;
    int const section_start = nVLocal * blockIdx.y;
    int const section_end = std::min(section_start + nVLocal, nV);
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    // Load element from logits to do reduce_md and arg_max meanwhile
#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, PAD_2K> partial;
#else
    TopKMD<T, PAD_2K> partial;
#endif
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;
    partial.topk.init();

    if (finished[bid].isFinished())
    {
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            float const val = (id == endIds[bid / nBM]) ? MAX_T_VAL : -MAX_T_VAL;
            MD const new_elem_md{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem_md);
            partial.topk.insert(val, id);
        }
    }
    else
    {
        T const* local_logits = logits + bid * nV;
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            T const b = bias == nullptr ? (T) 0.0f : bias[id];
            T const val = local_logits[id] + b;
            MD new_elem_md{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem_md);
            partial.topk.insert(val, id);
        }
    }

    // Search the top 2K elements among `nV` elements and write into smem_output
#if TOPK_FP16_STORAGE == 1
    typedef cub::BlockReduce<TopKMD<__half, PAD_2K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_buffer;
    TopKMD<__half, PAD_2K> total = BlockReduce(reduce_buffer).Reduce(partial, reduce_topk_md_op<__half, PAD_2K>);
#else
    typedef cub::BlockReduce<TopKMD<T, PAD_2K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_buffer;
    TopKMD<T, PAD_2K> total = BlockReduce(reduce_buffer).Reduce(partial, reduce_topk_md_op<T, PAD_2K>);
#endif
    __shared__ float smem_output[PACKED_TOP_KMD_SIZE];

    if (tid == 0)
    {
        for (int i = 0; i < 2 * nBM; i++)
        {
            int const index = bid * nV + total.topk.p[i];
            reinterpret_cast<int*>(smem_output)[i] = index;
            smem_output[PAD_2K + i] = total.topk.u[i];
        }
        smem_output[2 * PAD_2K] = total.md.d;
        smem_output[2 * PAD_2K + 1] = total.md.m;
    }
    __syncthreads();

    // Write the smem_output into pTemp
    float* local_temp_buffer = pTemp + bid * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE;
#pragma unroll
    for (int id = tid; id < PACKED_TOP_KMD_SIZE; id += THREADBLOCK_SIZE)
    {
        local_temp_buffer[id] = smem_output[id];
    }
}

template <typename T, int ITEMS_PER_THREAD, int PAD_2K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__ void beamStage1FastKernel(T const* __restrict logits,
    T const* __restrict bias, float* __restrict pTemp, int const* __restrict endIds,
    FinishedState const* __restrict finished, int const nBM, int const nV, int const nVLocal)
{
    constexpr int PACKED_TOP_KMD_SIZE = 2 * PAD_2K + 2;
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const section_start = nVLocal * blockIdx.y;
    int const section_end = std::min(section_start + nVLocal, nV);
    int const valid_smem_length = section_end - section_start;
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    // Load element from logits to smem_logprobs, doing reduce_md and arg_max meanwhile
    // Each thread is responsible for `nVLocal / THREADBLOCK_SIZE` elements
    extern __shared__ char smem_[];
    T* smem_logprobs = reinterpret_cast<T*>(smem_);

    MD partial_md{-MAX_T_VAL, 0.0f};

#if TOPK_FP16_STORAGE == 1
    using cub_kvp = cub::KeyValuePair<int, __half>;
#else
    using cub_kvp = cub::KeyValuePair<int, T>;
#endif
    cub_kvp partial_topk{nV - 1, -MAX_T_VAL};
    cub::ArgMax arg_max;

    if (finished[bid].isFinished())
    {
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            float const val = (id == endIds[bid / nBM]) ? MAX_T_VAL : -MAX_T_VAL;
            int const smem_index = id - section_start;
            smem_logprobs[smem_index] = val;
            MD const new_elem_md{val, 1.0F};
            partial_md = reduce_md_op(partial_md, new_elem_md);
            cub_kvp const new_elem_topk{smem_index, val};
            partial_topk = arg_max(partial_topk, new_elem_topk);
        }
    }
    else
    {
        T const* local_logits = logits + bid * nV;
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            T const b = bias == nullptr ? (T) 0.0f : bias[id];
            T const val = local_logits[id] + b;
            int const smem_index = id - section_start;
            smem_logprobs[smem_index] = val;
            MD new_elem_md{val, 1.0F};
            partial_md = reduce_md_op(partial_md, new_elem_md);
            cub_kvp new_elem_topk{smem_index, val};
            partial_topk = arg_max(partial_topk, new_elem_topk);
        }
    }
    __syncthreads();

    // Search the top 2K elements among `nVLocal` elements of this ThreadBlock and write into smem_output
    __shared__ float smem_output[PACKED_TOP_KMD_SIZE];
    __shared__ int thread_requiring_update;

    using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;
    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;

    __shared__ union
    {
        typename BlockReduceMD::TempStorage md_smem;
        typename BlockReduceTopK::TempStorage topk_smem;
    } reduce_buffer;

    for (int i = 0; i < 2 * nBM; ++i)
    {
        // Pop the element with largest value to "smem_output" per iteration
        cub_kvp total_topk = BlockReduceTopK(reduce_buffer.topk_smem).Reduce(partial_topk, arg_max);
        if (tid == 0)
        {
            int const index = bid * nV + section_start + total_topk.key;
            reinterpret_cast<int*>(smem_output)[i] = index;
            smem_output[PAD_2K + i] = total_topk.value;
            smem_logprobs[total_topk.key] = -MAX_T_VAL; // pollute the value of the popped element
            thread_requiring_update = total_topk.key % THREADBLOCK_SIZE;
        }
        __syncthreads();

        if (tid == thread_requiring_update && i < 2 * nBM - 1)
        {
            // The thread popped the element need to update its partial_topk
            // No need to do this in the last iteration
            partial_topk.key = nV - 1;
            partial_topk.value = -MAX_T_VAL;
            for (int index = tid; index < valid_smem_length; index += THREADBLOCK_SIZE)
            {
                cub_kvp new_elem{index, smem_logprobs[index]};
                partial_topk = arg_max(partial_topk, new_elem);
            }
        }
    }

    // Do reduce_md among the top 2K elements in the smem_output and write into tail of smem_output
    auto reduce_md_func = [](const MD& a, const MD& b) { return reduce_md_op(a, b); };
    MD total_md = BlockReduceMD(reduce_buffer.md_smem).Reduce(partial_md, reduce_md_func);
    if (tid == 0)
    {
        smem_output[2 * PAD_2K] = total_md.d;
        smem_output[2 * PAD_2K + 1] = total_md.m;
    }
    __syncthreads();

    // Write the smem_output into pTemp
    float* local_temp_buffer = pTemp + bid * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE;
#pragma unroll
    for (int id = tid; id < PACKED_TOP_KMD_SIZE; id += THREADBLOCK_SIZE)
    {
        local_temp_buffer[id] = smem_output[id];
    }
}

template <typename T, int PAD_2K, int THREADBLOCK_SIZE, bool IS_FAST_KERNEL>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beamStage2Kernel(int* __restrict pTempId, T* __restrict pTempVal,
    float* __restrict pTemp, float const* __restrict cum_log_probs, int const nBM, int const nV, int const nVPart)
{
    constexpr int PACKED_TOP_KMD_SIZE = 2 * PAD_2K + 2;
    int const bid = blockIdx.x;
    int const tid = threadIdx.x;
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    using cub_kvp = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
    using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;

    __shared__ cub_kvp buf_smem_kv[PAD_2K];

    __shared__ union
    {
        typename BlockReduceTopK::TempStorage topk_smem;
        typename BlockReduceMD::TempStorage md_smem;

    } shared_temp_storage;

    cub::ArgMax arg_max;
    MD partial_md{-MAX_T_VAL, 0.0f};
    cub_kvp total_topk{nV - 1, -MAX_T_VAL};

    auto reduce_md_func = [](const MD& a, const MD& b) { return reduce_md_op(a, b); };

    // Load and unpack into registers through smem
    float* local_temp_storage = pTemp + PACKED_TOP_KMD_SIZE * bid * nVPart;
    if constexpr (IS_FAST_KERNEL) // Use share memory instead of global memory
    {
        extern __shared__ char smem[];
        float* smem_topk = reinterpret_cast<float*>(smem);
        for (int idx = tid; idx < PACKED_TOP_KMD_SIZE * nVPart; idx += THREADBLOCK_SIZE)
        {
            smem_topk[idx] = local_temp_storage[idx];
        }
        local_temp_storage = smem_topk;
        __syncthreads();
    }

    // Find the top 2K across all nVPart
    for (int k = 0; k < 2 * nBM; ++k)
    {
        cub_kvp partial_topk{nV - 1, -MAX_T_VAL};
        // Only threads responsible for a chunk will do the computation
        if (tid < nVPart)
        {
            for (int i = 0; i < 2 * nBM; ++i)
            {
                int const current_index = tid * PACKED_TOP_KMD_SIZE + i;
                T current_value = local_temp_storage[current_index + PAD_2K];
                cub_kvp new_elem = {current_index, current_value};
                partial_topk = arg_max(partial_topk, new_elem);
            }
        }

        cub_kvp total_topk = BlockReduceTopK(shared_temp_storage.topk_smem).Reduce(partial_topk, arg_max);
        __syncthreads();

        if (tid == 0)
        {
            // Store kv pairs in shared mem buffer
            int temp_offset = total_topk.key;
            int global_offset = reinterpret_cast<int*>(local_temp_storage)[temp_offset];
            total_topk.key = global_offset;
            buf_smem_kv[k] = total_topk;

            // Invalidate the maximum value within the chunk
            reinterpret_cast<int*>(local_temp_storage)[temp_offset] = nV - 1; // id in share memory
            local_temp_storage[temp_offset + PAD_2K] = -MAX_T_VAL;            // value in share memory
        }
        __syncthreads();
    }

    // Extract and reduce MD values across the chunks
    if (tid < nVPart)
    {
        partial_md.d = local_temp_storage[tid * PACKED_TOP_KMD_SIZE + 2 * PAD_2K];
        partial_md.m = local_temp_storage[tid * PACKED_TOP_KMD_SIZE + 2 * PAD_2K + 1];
    }
    __syncthreads();

    MD total_md = BlockReduceMD(shared_temp_storage.md_smem).Reduce(partial_md, reduce_md_func);

    if (tid == 0)
    {
        float d_total_log = logf(total_md.d);

        for (int i = 0; i < PAD_2K; ++i)
        {
            float val = (float) buf_smem_kv[i].value - total_md.m - d_total_log;
            if (i < 2 * nBM)
            {
                pTempId[bid * 2 * nBM + i] = buf_smem_kv[i].key;
                pTempVal[bid * 2 * nBM + i] = val + cum_log_probs[bid];
            }
        }
    }
}

#define BEAM_STAGE2_KERNEL(N_VOCAB_PART, IS_FAST_KERNEL)                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (IS_FAST_KERNEL && nShareMemory >= (48 << 10))                                                              \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(beamStage2Kernel<T, PAD_2K, N_VOCAB_PART, IS_FAST_KERNEL>,            \
                cudaFuncAttributeMaxDynamicSharedMemorySize, nShareMemory));                                           \
        }                                                                                                              \
        beamStage2Kernel<T, PAD_2K, N_VOCAB_PART, IS_FAST_KERNEL>                                                      \
            <<<nBS * nBM, N_VOCAB_PART, IS_FAST_KERNEL * nShareMemory, stream>>>(                                      \
                pTempId, pTempVal, pTemp, cum_log_probs, nBM, nV, nVPart);                                             \
    } while (0);                                                                                                       \
    return;

template <typename T, int PAD_2K>
__inline__ void beamStage2KernelLauncher(float* pTemp, float const* cum_log_probs, int* pTempId, T* pTempVal,
    int const nBS, int const nBM, int const nVPart, int const nV, int const max_smem_per_block, cudaStream_t stream)
{
    // TODO: rewrite kernel to remove dependence of constant block size to reduce compilation time
    size_t const nShareMemory = sizeof(float) * nVPart * (2 * PAD_2K + 2) + sizeof(cub::KeyValuePair<int, T>) * PAD_2K;
    if (nShareMemory < max_smem_per_block) // IS_FAST_KERNEL must be a compilation-time constant
    {
        if (nVPart <= 32)
        {
            BEAM_STAGE2_KERNEL(32, true)
        }
        if (nVPart <= 64)
        {
            BEAM_STAGE2_KERNEL(64, true)
        }
        BEAM_STAGE2_KERNEL(128, true)
        // No larger branch since nVPart <= nMaxVocabPartForStage1FastKernel
    }
    BEAM_STAGE2_KERNEL(128, false)
}

template <typename T, int PAD_K>
void topKSoftMaxKernelLauncher(T const* logits, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream)
{
    // Workflow of this function (reference: https://github.com/NVIDIA/online-softmax)
    // Using batch_size (BS) = 2, beam_width (BM) = 5, vocab_size (V) = vocan_size_padded (VP) = 32000 as an example:
    // nPaddedBeamWidth (PAD_K) = 2 ^ ceil(log(BM)) = 8, PAD_2K = 2 * PAD_K = 16
    // logits.shape = [BS, BM, V]
    // nBlockSize = 128, nVPart = 13, nVocabChunk = 2462 = ceil(32000/13)

    // The content of workspace (length aligned to 4):
    //             | allocated size                         | used size                | data type |
    // ┏━━━━━━━━━━┓ --------------------------------------------------------------------------------
    // ┃ pTempId  ┃ BS * PAD_K * PAD_K * 2                  |                          | int       |
    // ┣━━━━━━━━━━┫ ----------------------------------------- Change "PAD_K" into "BM" -------------
    // ┃ pTempVal ┃ BS * PAD_K * PAD_K * 2                  |                          | float     |
    // ┣━━━━━━━━━━┫ ----------------------------------------- in the left formulas     -------------
    // ┃ pTemp    ┃ BS * PAD_K * VP * (2 * (PAD_K * 2) + 2) |                          | float     |
    // ┗━━━━━━━━━━┛ --------------------------------------------------------------------------------

    // Stage1: gridDim(BS*BM,nVPart,1), blockDim(nBlockSize,1,1)
    // Each ThreadBlock takes `nVocabChunk` contiguous elements in logits to do TopK and reduce_md,
    //   then writes output into pTemp.
    // At end of this kernel, each ThreadBlock holds the indexes and values of the top 2*BM elements,
    //   as well as the m(x) and l(x) of those elements (see paper of Flash Attention, arXiv:2205.14135)
    // pTemp.shape = [BS*BM, nVPart, 2*PAD_2K+2]
    // The content of the last dimension of pTemp (updated by each ThreadBlock, we call it "Tile"):
    //                  ┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
    //                  ┃ topk_id ┃ topk_val ┃ md    ┃
    //                  ┗━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━┛
    // | allocated size | PAD_2K  | PAD_2K   | 2     |
    // | used size      | BM * 2  | BM * 2   | 2     |
    // | data type      | int     | float    | float |

    // Stage2: gridDim(BS*BM,1,1), blockDim(32/64/128,1,1)
    // Each TheadBlock takes `nVPart` contiguous Tiles in pTemp to do reduce_topk and reduce_md,
    //   writes output topk_id into in pTempId, writes topk_value + cum_log_probs into pTempVal.

    // beamStage3Kernel: gridDim(BS,1,1), blockDim(128,1,1)
    // Each TheadBlock is responsible for one batch, doing work below:
    //   + moves one beam into candidate-beam-array if it is finished (gemerated end_id in this step).
    //   + selects BM elements for the next generation step if not.
    //   + maintains related score array, min_normed_score / batchDones / finished, etc..

    int constexpr items_per_thread = 1;
    int constexpr nBlockSize = (PAD_K < 16) ? ((PAD_K < 8) ? nBlockSizeForSmallBeamWidth : 128) : 64;
    int const nBS{bh.nBatchSizeLocal};
    int const nBM{bh.nBeamWidth};
    int const nV{bh.nVocabSize};
    int const* endIds{bh.endIds};
    FinishedState const* finished{bh.finished};

    int const offset = roundUp(nBS * nBM * nBM * 2, 4);
    int* pTempId = reinterpret_cast<int*>(workspace);
    T* pTempVal = reinterpret_cast<T*>(pTempId + offset);
    float* pTemp = reinterpret_cast<float*>(pTempVal + offset);

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX

    // Upper limit count of ThreadBlock, gotten by using no share memory
    int max_active_blocks = -1;
    TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, beamStage1FastKernel<T, items_per_thread, 2 * PAD_K, nBlockSize>, nBlockSize, 0));

    // Find the max smem on the device and use that to determine the vocab parts in the best case.
    int max_smem_per_sm = -1;
    int max_smem_per_block = -1;
    int const device = tensorrt_llm::common::getDevice();
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attr;
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage1FastKernel<T, items_per_thread, 2 * PAD_K, nBlockSize>));

    // One ThreadBlock must at least have share memory of `sizeof(T) * nV / nMaxVocabPartForStage1FastKernel` bytes
    int const static_smem = attr.sharedSizeBytes;
    int const max_dyn_smem_per_block = max_smem_per_block - static_smem;
    TLLM_CHECK_WITH_INFO(sizeof(T) * nV <= max_dyn_smem_per_block * nMaxVocabPartForStage1FastKernel,
        "Vocab size is too large for split-k TopK beam search fast path.");

    // Find the maximum of ThreadBlock (maximum of nVPart, minimum of smem),
    // satisfying nVPart <= nMaxVocabPartForStage1FastKernel && dyn_smem_size * nVPart >= sizeof(T) * nV
    int const driver_smem_per_block = max_smem_per_sm - max_smem_per_block;
    int const extra_smem = driver_smem_per_block + static_smem;

    int nVPart = nMaxVocabPartForStage1FastKernel + 1;
    for (int n_block = max_active_blocks - 1; n_block > 0 && nVPart > nMaxVocabPartForStage1FastKernel; --n_block)
    {
        int dyn_smem_size = max_smem_per_sm / n_block - extra_smem;
        dyn_smem_size -= dyn_smem_size % sizeof(T);
        nVPart = ceilDiv(sizeof(T) * nV, dyn_smem_size);
    }

    if (nVPart <= nMaxVocabPartForStage1FastKernel)
    {
        // Use stage 1 fast kernel
        int const nVocabChunk = (nV + nVPart - 1) / nVPart;
        int const dyn_smem_size = sizeof(T) * nVocabChunk;
        if (dyn_smem_size >= (48 << 10))
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(beamStage1FastKernel<T, items_per_thread, 2 * PAD_K, nBlockSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem_size));
        }
        dim3 gridSize(nBS * nBM, nVPart);
        beamStage1FastKernel<T, items_per_thread, 2 * PAD_K, nBlockSize>
            <<<gridSize, nBlockSize, dyn_smem_size, stream>>>(
                logits, bias, pTemp, endIds, finished, nBM, nV, nVocabChunk);
    }
    else
    {
        // Use stage 1 base kernel, useless branch now
        int nVPart = 4;
        if (nBS * nBM < 256)
        {
            // TODO: add heuristics for base stage 1 kernel
            // Volta has 80 SMs, so we aim for three waves
            nVPart = (240 + nBS * nBM - 1) / (nBS * nBM);
            nVPart = std::min(128, nVPart); // we implement up to 128
        }
        cudaFuncSetAttribute(beamStage1BaseKernel<T, items_per_thread, 2 * PAD_K, nBlockSize>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
        dim3 gridSize(nBS * nBM, nVPart);
        beamStage1BaseKernel<T, items_per_thread, 2 * PAD_K, nBlockSize>
            <<<gridSize, nBlockSize, 0, stream>>>(logits, bias, pTemp, endIds, finished, nBM, nV);
    }
    sync_check_cuda_error();

    beamStage2KernelLauncher<T, 2 * PAD_K>(
        pTemp, bh.cumLogProbs, pTempId, pTempVal, nBS, nBM, nVPart, nV, max_smem_per_block, stream);

#else
    beamKernel<T, items_per_thread, PAD_K, nBlockSize>
        <<<nBS * nBM, nBlockSize, 0, stream>>>(logits, bias, pTempId, pTempVal, bh);
#endif

    sync_check_cuda_error();

    // Keep top 2K candidates in case of k candidates finishes in one iteration
    size_t const nShareMemory = sizeof(T) * nBM * nBM * 2;
    size_t constexpr nBlockSizeStage3 = (PAD_K + 31) / 32 * 32; // can not use `roundUp()`
    if (nShareMemory >= (48 << 10))
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(beamStage3Kernel<T, PAD_K * 2, nBlockSizeStage3>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, nShareMemory));
    }

    beamStage3Kernel<T, PAD_K * 2, nBlockSizeStage3>
        <<<nBS, nBlockSizeStage3, nShareMemory, stream>>>(pTempId, pTempVal, bh);
    sync_check_cuda_error();
}

#define INSTANTIATE_BEAMSEARCH_K(T, PAD_K)                                                                             \
    template void topKSoftMaxKernelLauncher<T, PAD_K>(                                                                 \
        T const* logits, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
