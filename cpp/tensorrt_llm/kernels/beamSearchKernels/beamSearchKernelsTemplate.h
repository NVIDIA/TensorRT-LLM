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

template <typename T, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batchBeamKernel(int const* __restrict topk_id_buffer, T const* __restrict topk_val_buffer, BeamHypotheses bh)
{
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const gbid{bh.ite * bh.local_batch_size + bid}; // global batch index
    int const K{bh.beam_width};
    int const V{bh.vocab_size};
    int const nCandidate{K * K * 2};
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;

    float const diversity_rate{bh.diversity_rates[gbid]};
    float const length_penalty{bh.length_penalties[gbid]};
    int const early_stopping{bh.early_stoppings[gbid]};

    __shared__ int nBeamForNextStep;
    __shared__ float smem_cum_log_probs[MAX_K2 / 2];

    if (tid == 0)
    {
        nBeamForNextStep = 0;
    }
    if (tid < K)
    {
        smem_cum_log_probs[tid] = bh.cum_log_probs[bid * K + tid];
    }
    __syncthreads();

    if (bh.num_beams != nullptr)
    {
        // Beam search is enabled
        if (bh.num_beams[gbid] == 0 && tid == 0)
        {
            // Initialize worst_score in the first time
            bh.min_normed_scores[gbid] = FLT_MAX;
        }
        else if (early_stopping == 1 && bh.num_beams[gbid] == K
            || early_stopping != 1 && bh.finished[bid * K].isFinished())
        {
            // New but false condition:
            // else if (early_stopping == 1 && bh.num_beams[gbid] == K || early_stopping != 1 && bh.is_done[bid])
            // Condition of early return:
            // 1. In EarlyStopping mode, and we have got enough beams
            // 2. In NonEarlyStopping mode, and this batch has been marked as done
            return;
        }
    }

    // Get top 2K tokens from candidates
    topk_id_buffer += bid * nCandidate;
    topk_val_buffer += bid * nCandidate;

    using cub_kvp = cub::KeyValuePair<int, T>;
    cub_kvp partial_topk{nCandidate - 1, -MAX_T_VAL};
    cub::ArgMax arg_max;
    extern __shared__ char smem[];
    T* smem_topk = reinterpret_cast<T*>(smem);

    for (int id = tid; id < nCandidate; id += THREADBLOCK_SIZE)
    {
        int const index = bh.num_beams == nullptr ? id % K : id / 2 / K;
        T val = topk_val_buffer[id] + static_cast<T>(diversity_rate * index);
        cub_kvp new_elem{id, val};
        partial_topk = arg_max(partial_topk, new_elem);
        smem_topk[id] = val;
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage reduce_buffer;
    __shared__ cub_kvp cta_topk[MAX_K2];
    __shared__ int thread_requiring_update;

    for (int i = 0; i < 2 * K; ++i)
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
        if (tid == thread_requiring_update && i < (2 * K - 1))
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
        // Adjust beams or select completed beams sequentially
        // Reference (might be changed along HF in the future):
        // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py#L272
        for (int i = 0; i < 2 * K; ++i)
        {
            int const current_key = cta_topk[i].key;
            T const current_value = cta_topk[i].value;
            bool const is_end_token = topk_id_buffer[current_key] % V == bh.end_ids[bid];
            if (i < K && bh.num_beams != nullptr && is_end_token)
            {
                // Condition of this branch
                // In Beam search mode, this token is end_token and belongs to top K range in Beam search mode
                int const seq_len = bh.seq_len[bid * K + i] + 1 - bh.input_lengths[gbid * K + i];
                float const normed_score = applyLengthPenalty(current_value, seq_len, length_penalty);
                int beam_idx = bh.num_beams[gbid];
                if (beam_idx == K)
                {
                    // There are already K beams
                    if (normed_score < bh.min_normed_scores[gbid])
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
                        for (int j = 0; j < K; j++)
                        {
                            if (bh.normed_scores_cba[gbid * (K * 2) + j] == bh.min_normed_scores[gbid])
                            {
                                beam_idx = j;
                                bh.num_beams[gbid]--;
                                bh.min_normed_scores[gbid] = FLT_MAX;
                                bh.normed_scores_cba[gbid * (K * 2) + j] = normed_score;
                                for (int l = 0; l < K; l++)
                                {
                                    bh.min_normed_scores[gbid]
                                        = min(bh.min_normed_scores[gbid], bh.normed_scores_cba[gbid * (K * 2) + l]);
                                }
                                break;
                            }
                        }
                    }
                }
                int prev_id = (topk_id_buffer[current_key] / V) % K;
                int const current_step = bh.seq_len[bid * K + prev_id];
                int const tgt_id_offset = ((bid + bh.ite * bh.local_batch_size) * (K * 2) + beam_idx) * bh.max_seq_len;
                bh.output_ids_cba[tgt_id_offset + current_step] = bh.end_ids[bid];
                if (bh.log_probs_cba != nullptr)
                {
                    bh.log_probs_cba[tgt_id_offset + current_step] = (float) topk_val_buffer[current_key]
                        - smem_cum_log_probs[(topk_id_buffer[current_key] / V) % K];
                }
                // Write finished beam from work tree to CBA
                for (int j = current_step - 1; j >= 0; j--)
                {
                    bh.output_ids_cba[tgt_id_offset + j] = bh.output_ids_ptr[bid][prev_id * bh.max_seq_len + j];
                    prev_id = bh.parent_ids_ptr[bid][prev_id * bh.max_seq_len + j];
                }
                if (bh.log_probs_cba != nullptr && bh.log_probs != nullptr)
                {
                    prev_id = (topk_id_buffer[current_key] / V) % K;
                    for (int j = current_step - 1; j >= 0; j--)
                    {
                        int const index = j * bh.batch_size * K + bh.ite * bh.local_batch_size * K + bid * K + prev_id;
                        bh.log_probs_cba[tgt_id_offset + j] = bh.log_probs[index];
                        prev_id = bh.parent_ids_ptr[bid][prev_id * bh.max_seq_len + j];
                    }
                }
                int const tgt_beam_idx = gbid * (K * 2) + beam_idx;
                bh.seq_len_cba[tgt_beam_idx] = current_step;
                bh.normed_scores_cba[tgt_beam_idx] = normed_score;
                bh.min_normed_scores[gbid] = min(bh.min_normed_scores[gbid], bh.normed_scores_cba[tgt_beam_idx]);
                bh.num_beams[gbid]++;
                bh.cum_log_probs_cba[tgt_beam_idx] = (float) topk_val_buffer[current_key];
            }
            else if (i < K || bh.num_beams != nullptr && !is_end_token)
            {
                // Condition of this branch
                // 1. bh.num_beams == nullptr && i <  K, i.e., beam search is disable
                // 2. bh.num_beams != nullptr && i <  K && is_end_token == false, i.e., add token at the end
                // 3. bh.num_beams != nullptr && i >= K && is_end_token == false, i.e., add token at the end
                int const current_step = bh.seq_len[bid * K + nBeamForNextStep];
                // Write the selected token to work tree
                bh.output_ids_ptr[bid][nBeamForNextStep * bh.max_seq_len + current_step] = topk_id_buffer[current_key];
                if (bh.log_probs != nullptr)
                {
                    bh.log_probs[current_step * bh.batch_size * K + bid * K + nBeamForNextStep]
                        = (float) topk_val_buffer[current_key]
                        - smem_cum_log_probs[(topk_id_buffer[current_key] / V) % K];
                }
                bh.cum_log_probs[bid * K + nBeamForNextStep] = (float) topk_val_buffer[current_key];
                nBeamForNextStep++;
            }
            else
            {
                // Condition of this branch, which we do nothing for it
                // 1. bh.num_beams == nullptr && i >= K, i.e., beam search is disable
                // 2. bh.num_beams != nullptr && i >= K && is_end_token == true, i.e., ignore the worse beams
            }

            // if (early_stopping == 1 && bh.num_beams[gbid] >= K || nBeamForNextStep >= K)
            if (nBeamForNextStep >= K)
            {
                // Condition of this branch:
                // 1. In EarlyStopping mode, and get enough candidate beams
                // 2. In EarlyStopping mode, and get enough tokens for the next generation step
                // 3. In NonEarlyStopping mode, and get enough tokens for the next generation step
                break;
            }
        }
    }

    // Update bh.is_done
    if (tid == 0 && bh.num_beams != nullptr)
    {
        if (bh.num_beams[bid] < K)
        {
            // no enough beams
            bh.is_done[bid] = false;
        }
        else if (early_stopping == 1)
        {
            // enough candidate beams in EarlyStopping mode
            bh.is_done[bid] = true;
        }
        else
        {
            // enough beams in NonEarlyStopping mode
            int seq_len = bh.seq_len[bid * K] + 1 - bh.input_lengths[gbid * K];
            float const best_sum_logprobs = cta_topk[0].value;
            // According to semantics of HF, cta_topk[0].value is used as best_sum_logprobs
            // But maybe bh.cum_log_probs[bid * K + i] is more suitable?
            // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py#L307
            if (early_stopping != 0 && length_penalty > 0.0f)
            {
                // Specialization for early_stopping == "never" and length_penalty > 0 in HF
                seq_len = bh.max_seq_len - bh.input_lengths[gbid * K];
            }
            float const highest_attainable_score = applyLengthPenalty(best_sum_logprobs, seq_len, length_penalty);
            bh.is_done[bid] = bh.min_normed_scores[gbid] >= highest_attainable_score;
        }
    }
    __syncthreads();

    // Update sequence_lengths, parent_ids, output_ids and finished
    __shared__ int s_sequence_lengths[MAX_K2 / 2];
    if (tid < K)
    {
        s_sequence_lengths[tid] = bh.seq_len[bid * K + tid];
    }
    __syncthreads();

    if (tid < K)
    {
        int const bb_index = bid * K + tid;
        int const current_step = s_sequence_lengths[tid];
        if (!bh.finished[bb_index].isFinished())
        {
            s_sequence_lengths[tid]++;
        }
        int const new_id = bh.output_ids_ptr[bid][tid * bh.max_seq_len + current_step];
        int const new_beam_id = (new_id / V) % K;
        int const new_word_id = new_id % V;
        bh.seq_len[bb_index] = s_sequence_lengths[new_beam_id];
        if (new_word_id == bh.end_ids[bid])
        {
            bh.finished[bb_index].setFinishedEOS();
        }
        bh.parent_ids_ptr[bid][tid * bh.max_seq_len + current_step] = new_beam_id;
        bh.output_ids_ptr[bid][tid * bh.max_seq_len + current_step] = new_word_id;
        if ((early_stopping == 1) && (bh.num_beams != nullptr && bh.num_beams[gbid] == K)
            || (early_stopping != 1) && bh.is_done[bid])
        {
            bh.is_done[bid] = true;
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

template <typename T, int MAX_K>
struct TopKMD
{
    MD md;
    TopK<T, MAX_K> topk;
};

template <typename T, int MAX_K>
__device__ __forceinline__ TopKMD<T, MAX_K> reduce_topk_md_op(TopKMD<T, MAX_K> const& a, TopKMD<T, MAX_K> const& b)
{
    TopKMD<T, MAX_K> res;
    res.md = reduce_md_op(a.md, b.md);
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

template <typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beamKernel(T const* __restrict logits, T const* __restrict bias,
    float const* __restrict cum_log_probs, FinishedState const* __restrict finished, int* __restrict topk_id_buffer,
    T* __restrict topk_val_buffer, int V, int K, int const* __restrict end_ids)
{
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;

    TopKMD<float, MAX_K> partial;
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;
    partial.topk.init();

    if (finished[bid].isFinished())
    {
        for (int id = tid; id < V; id += THREADBLOCK_SIZE)
        {
            float const val = id == end_ids[bid / K] ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(val, id);
        }
    }
    else
    {
        T const* local_logits = logits + bid * V;
        for (int id = tid; id < V; id += THREADBLOCK_SIZE)
        {
            float const val = local_logits[id] + bias[id];
            MD new_elem{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(val, id);
        }
    }

    typedef cub::BlockReduce<TopKMD<float, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_buffer;

    TopKMD<float, MAX_K> total = BlockReduce(reduce_buffer).Reduce(partial, reduce_topk_md_op<float, MAX_K>);

    if (tid == 0)
    {
        int* local_topk_id = topk_id_buffer + bid * K;
        T const* local_topk_val = topk_val_buffer + bid * K;
        float const total_m = total.md.m;
        float const total_d = logf(total.md.d);
        float local_cum_log_probs = cum_log_probs[bid];
        for (int i = 0; i < K; ++i)
        {
            local_topk_id[i] = total.topk.p[i] + bid * V;
            local_topk_val[i] = total.topk.u[i] - total_m - total_d + local_cum_log_probs;
        }
    }
}

template <typename T, int ITEMS_PER_THREAD, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__ void beamStage1BaseKernel(T const* __restrict logits,
    T const* __restrict bias, FinishedState const* __restrict finished, float* __restrict temp_buffer, int V, int K,
    int const* __restrict end_ids)
{
    // Compare to beamStage1FastKernel, here is no share memory for storage of logits,
    // and each ThreadBlock is responsible for `V / voc_parts` elements
    constexpr int PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const V_local = (V + gridDim.y - 1) / gridDim.y;
    int const section_start = V_local * blockIdx.y;
    int const section_end = std::min(section_start + V_local, V);
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;

    // Load element from logits to do reduce_md and arg_max meanwhile
#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, MAX_K2> partial;
#else
    TopKMD<T, MAX_K2> partial;
#endif
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;
    partial.topk.init();

    if (finished[bid].isFinished())
    {
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            float const val = (id == end_ids[bid / K]) ? MAX_T_VAL : -MAX_T_VAL;
            MD const new_elem_md{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem_md);
            partial.topk.insert(val, id);
        }
    }
    else
    {
        T const* local_logits = logits + bid * V;
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

    // Search the top 2K elements among `V` elements and write into smem_output
#if TOPK_FP16_STORAGE == 1
    typedef cub::BlockReduce<TopKMD<__half, MAX_K2>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_buffer;
    TopKMD<__half, MAX_K2> total = BlockReduce(reduce_buffer).Reduce(partial, reduce_topk_md_op<__half, MAX_K2>);
#else
    typedef cub::BlockReduce<TopKMD<T, MAX_K2>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_buffer;
    TopKMD<T, MAX_K2> total = BlockReduce(reduce_buffer).Reduce(partial, reduce_topk_md_op<T, MAX_K2>);
#endif
    __shared__ float smem_output[PACKED_TOP_KMD_SIZE];

    if (tid == 0)
    {
        for (int i = 0; i < 2 * K; i++)
        {
            int const index = bid * V + total.topk.p[i];
            reinterpret_cast<int*>(smem_output)[i] = index;
            smem_output[MAX_K2 + i] = total.topk.u[i];
        }
        smem_output[2 * MAX_K2] = total.md.d;
        smem_output[2 * MAX_K2 + 1] = total.md.m;
    }
    __syncthreads();

    // Write the smem_output into temp_buffer
    float* local_temp_buffer = temp_buffer + bid * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE;
#pragma unroll
    for (int id = tid; id < PACKED_TOP_KMD_SIZE; id += THREADBLOCK_SIZE)
    {
        local_temp_buffer[id] = smem_output[id];
    }
}

template <typename T, int ITEMS_PER_THREAD, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__ void beamStage1FastKernel(T const* __restrict logits,
    T const* __restrict bias, FinishedState const* __restrict finished, float* __restrict temp_buffer, int V, int K,
    int const* __restrict end_ids, int const V_local)
{
    constexpr int PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const section_start = V_local * blockIdx.y;
    int const section_end = std::min(section_start + V_local, V);
    int const valid_smem_length = section_end - section_start;
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;

    // Load element from logits to smem_logprobs, doing reduce_md and arg_max meanwhile
    // Each thread is responsible for `V_local / THREADBLOCK_SIZE` elements
    extern __shared__ char smem_[];
    T* smem_logprobs = reinterpret_cast<T*>(smem_);

    MD partial_md{-MAX_T_VAL, 0.0f};

#if TOPK_FP16_STORAGE == 1
    using cub_kvp = cub::KeyValuePair<int, __half>;
#else
    using cub_kvp = cub::KeyValuePair<int, T>;
#endif
    cub_kvp partial_topk{V - 1, -MAX_T_VAL};
    cub::ArgMax arg_max;

    if (finished[bid].isFinished())
    {
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            float const val = (id == end_ids[bid / K]) ? MAX_T_VAL : -MAX_T_VAL;
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
        T const* local_logits = logits + bid * V;
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

    // Search the top 2K elements among `V_local` elements of this ThreadBlock and write into smem_output
    __shared__ float smem_output[PACKED_TOP_KMD_SIZE];
    __shared__ int thread_requiring_update;

    using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;
    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;

    __shared__ union
    {
        typename BlockReduceMD::TempStorage md_smem;
        typename BlockReduceTopK::TempStorage topk_smem;
    } reduce_buffer;

    for (int i = 0; i < 2 * K; ++i)
    {
        // Pop the element with largest value to "smem_output" per iteration
        cub_kvp total_topk = BlockReduceTopK(reduce_buffer.topk_smem).Reduce(partial_topk, arg_max);
        if (tid == 0)
        {
            int const index = bid * V + section_start + total_topk.key;
            reinterpret_cast<int*>(smem_output)[i] = index;
            smem_output[MAX_K2 + i] = total_topk.value;
            smem_logprobs[total_topk.key] = -MAX_T_VAL; // pollute the value of the popped element
            thread_requiring_update = total_topk.key % THREADBLOCK_SIZE;
        }
        __syncthreads();

        if (tid == thread_requiring_update && i < 2 * K - 1)
        {
            // The thread popped the element need to update its partial_topk
            // No need to do this in the last iteration
            partial_topk.key = V - 1;
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
        smem_output[2 * MAX_K2] = total_md.d;
        smem_output[2 * MAX_K2 + 1] = total_md.m;
    }
    __syncthreads();

    // Write the smem_output into temp_buffer
    float* local_temp_buffer = temp_buffer + bid * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE;
#pragma unroll
    for (int id = tid; id < PACKED_TOP_KMD_SIZE; id += THREADBLOCK_SIZE)
    {
        local_temp_buffer[id] = smem_output[id];
    }
}

template <typename T, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beamStage2Kernel(float const* __restrict temp_buffer, float const* __restrict cum_log_probs,
        int* __restrict topk_id_buffer, T* __restrict topk_val_buffer, int const K, int const voc_parts, int const V)
{
    constexpr int PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;
    int const bid = blockIdx.x;
    int const tid = threadIdx.x;
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;

    using cub_kvp = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
    using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;

    extern __shared__ char smem[];
    float* smem_topk = reinterpret_cast<float*>(smem);
    __shared__ cub_kvp buf_smem_kv[MAX_K2];

    __shared__ union
    {
        typename BlockReduceTopK::TempStorage topk_smem;
        typename BlockReduceMD::TempStorage md_smem;

    } shared_temp_storage;

    cub::ArgMax arg_max;
    MD partial_md{-MAX_T_VAL, 0.0f};
    cub_kvp total_topk{V - 1, -MAX_T_VAL};

    auto reduce_md_func = [](const MD& a, const MD& b) { return reduce_md_op(a, b); };

    // Load and unpack into registers through smem
    float const* local_temp_storage = temp_buffer + PACKED_TOP_KMD_SIZE * bid * voc_parts;
    for (int idx = tid; idx < PACKED_TOP_KMD_SIZE * voc_parts; idx += THREADBLOCK_SIZE)
    {
        smem_topk[idx] = local_temp_storage[idx];
    }
    __syncthreads();

    // Find the argmax within each voc_parts
    // Find the topK across all voc_parts
    for (int k = 0; k < 2 * K; ++k)
    {
        cub_kvp partial_topk{V - 1, -MAX_T_VAL};
        // Only threads responsible for a chunk will do the computation
        if (tid < voc_parts)
        {
            for (int i = 0; i < 2 * K; ++i)
            {
                int const current_index = tid * PACKED_TOP_KMD_SIZE + i;
                T current_value = smem_topk[current_index + MAX_K2];
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
            int global_offset = reinterpret_cast<int*>(smem_topk)[temp_offset];
            total_topk.key = global_offset;
            buf_smem_kv[k] = total_topk;

            // Invalidate the maximum value within the chunk
            reinterpret_cast<int*>(smem_topk)[temp_offset] = V - 1; // id in share memory
            smem_topk[temp_offset + MAX_K2] = -MAX_T_VAL;           // value in share memory
        }
        __syncthreads();
    }

    // Extract and reduce MD values across the chunks
    if (tid < voc_parts)
    {
        partial_md.d = smem_topk[tid * PACKED_TOP_KMD_SIZE + 2 * MAX_K2];
        partial_md.m = smem_topk[tid * PACKED_TOP_KMD_SIZE + 2 * MAX_K2 + 1];
    }
    __syncthreads();

    MD total_md = BlockReduceMD(shared_temp_storage.md_smem).Reduce(partial_md, reduce_md_func);

    if (tid == 0)
    {
        float d_total_log = logf(total_md.d);

        for (int i = 0; i < MAX_K2; ++i)
        {
            float val = (float) buf_smem_kv[i].value - total_md.m - d_total_log;
            if (i < 2 * K)
            {
                topk_id_buffer[bid * 2 * K + i] = buf_smem_kv[i].key;
                topk_val_buffer[bid * 2 * K + i] = val + cum_log_probs[bid];
            }
        }
    }
}

template <typename T, int MAX_K2>
void beamStage2KernelLauncher(float const* temp_buffer, float const* cum_log_probs, int* topk_id_buffer,
    T* topk_val_buffer, int const batch_size, int const beam_width, int const voc_parts, int const V,
    cudaStream_t stream)
{
    // TODO: rewrite kernel to remove dependence of constant block size to reduce compilation time
    size_t const smem_size = sizeof(float) * voc_parts * (2 * MAX_K2 + 2);

    if (voc_parts <= 32)
    {
        beamStage2Kernel<T, MAX_K2, 32><<<batch_size * beam_width, 32, smem_size, stream>>>(
            temp_buffer, cum_log_probs, topk_id_buffer, topk_val_buffer, beam_width, voc_parts, V);
        return;
    }
    if (voc_parts <= 64)
    {
        beamStage2Kernel<T, MAX_K2, 64><<<batch_size * beam_width, 64, smem_size, stream>>>(
            temp_buffer, cum_log_probs, topk_id_buffer, topk_val_buffer, beam_width, voc_parts, V);
        return;
    }
    if (voc_parts <= 128)
    {
        beamStage2Kernel<T, MAX_K2, 128><<<batch_size * beam_width, 128, smem_size, stream>>>(
            temp_buffer, cum_log_probs, topk_id_buffer, topk_val_buffer, beam_width, voc_parts, V);
        return;
    }
    assert(0);
}

template <typename T, int MAX_K>
void topK_softMax_kernelLauncher(
    T const* logits, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream)
{
    // Workflow of this function (reference: https://github.com/NVIDIA/online-softmax)
    // Using batch_size (BS) = 2, beam_width (BM) = 5, vocab_size (V) = 32000 as an example:
    // nPaddedBeamWidth (pBM) = 8 = 2 ^ ceil(log(BM)), nSmallTopKMaxVocParts (nVP) = 128 (Constant)
    // MAX_K = 8 = pBM, MAX_K2 = 16 = 2 * pBM
    // logits.shape = [BS, BM, V]
    // blockSize = 128, voc_parts = 13, voc_size_chunk = 2462 = ceil(32000/13)

    // The content of workspace (length aligned to 4):
    //                    | allocated size                      | used size              | data type |
    // ┏━━━━━━━━━━━━━━━━━┓ ---------------------------------------------------------------------------
    // ┃ topk_id_buffer  ┃ BS * pBM * pBM * 2                   |                        | int       |
    // ┣━━━━━━━━━━━━━━━━━┫ -------------------------------------- Change "pBM" into "BM" -------------
    // ┃ topk_val_buffer ┃ BS * pBM * pBM * 2                   |                        | float     |
    // ┣━━━━━━━━━━━━━━━━━┫ -------------------------------------- in the left formulas   -------------
    // ┃ temp_buffer     ┃ BS * pBM * nVP * (2 * (pBM * 2) + 2) |                        | float     |
    // ┗━━━━━━━━━━━━━━━━━┛ ---------------------------------------------------------------------------

    // Stage1: gridDim(BS*BM,voc_parts,1), blockDim(blockSize,1,1)
    // Each ThreadBlock takes `voc_size_chunk` contiguous elements in logits to do TopK and reduce_md,
    //   then writes output into temp_buffer.
    // At end of this kernel, each ThreadBlock holds the indexes and values of the top 2*K elements,
    //   as well as the m(x) and l(x) of those elements (see paper of Flash Attention, arXiv:2205.14135)
    // temp_buffer.shape = [BS*BM, voc_parts, 2*MAX_K2+2]
    // The content of the last dimension of temp_buffer (updated by each ThreadBlock, we call it "Tile"):
    //                  ┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
    //                  ┃ topk_id ┃ topk_val ┃ md    ┃
    //                  ┗━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━┛
    // | allocated size | MAX_K2  | MAX_K2   | 2     |
    // | used size      | 2*BM    | 2*BM     | 2     |
    // | data type      | int     | float    | float |

    // Stage2: gridDim(BS*BM,1,1), blockDim(32/64/128,1,1)
    // Each TheadBlock takes `voc_parts` contiguous Tiles in temp_buffer to do reduce_topk and reduce_md,
    //   writes output topk_id into in topk_id_buffer, writes topk_value + cum_log_probs into topk_val_buffer.

    // batchBeamKernel: gridDim(BS,1,1), blockDim(128,1,1)
    // Each TheadBlock is responsible for one batch, doing work below:
    //   + moves one beam into candidate-beam-array if it is finished (gemerated end_id in this step).
    //   + selects BM elements for the next generation step if not.
    //   + maintains related score array, min_normed_score / is_done / finished, etc..

    constexpr int items_per_thread = 1;
    constexpr int blockSize = (MAX_K < 16) ? ((MAX_K < 8) ? nSmallTopKBlockSize : 128) : 64;
    int const batch_size{bh.local_batch_size};
    int const beam_width{bh.beam_width};
    int const V{bh.vocab_size};
    int const* end_ids{bh.end_ids};
    float* cum_log_probs{bh.cum_log_probs};
    FinishedState const* finished{bh.finished};

    int const offset = roundUp(batch_size * beam_width * beam_width * 2, 4);
    int* topk_id_buffer = reinterpret_cast<int*>(workspace);
    T* topk_val_buffer = reinterpret_cast<T*>(topk_id_buffer + offset);
    float* temp_buffer = reinterpret_cast<float*>(topk_val_buffer + offset);

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX

    // Upper limit count of ThreadBlock, gotten by using no share memory
    int max_active_blocks = -1;
    TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, beamStage1FastKernel<T, items_per_thread, 2 * MAX_K, blockSize>, blockSize, 0));

    // Find the max smem on the device and use that to determine the vocab parts in the best case.
    int max_smem_per_sm = -1;
    int max_smem_per_block = -1;
    int const device = tensorrt_llm::common::getDevice();
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attr;
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage1FastKernel<T, items_per_thread, 2 * MAX_K, blockSize>));

    // One ThreadBlock must at least have share memory of `sizeof(T) * V / nSmallTopKMaxVocParts` bytes
    int const static_smem = attr.sharedSizeBytes;
    int const max_dyn_smem_per_block = max_smem_per_block - static_smem;
    TLLM_CHECK_WITH_INFO(sizeof(T) * V <= max_dyn_smem_per_block * nSmallTopKMaxVocParts,
        "Vocab size is too large for split-k TopK beam search fast path.");

    // Find the maximum of ThreadBlock (maximum of voc_parts, minimum of smem),
    // satisfying voc_parts <= nSmallTopKMaxVocParts && dyn_smem_size * voc_parts >= sizeof(T) * V
    int const driver_smem_per_block = max_smem_per_sm - max_smem_per_block;
    int const extra_smem = driver_smem_per_block + static_smem;
    int voc_parts = nSmallTopKMaxVocParts + 1;
    for (int n_block = max_active_blocks - 1; n_block > 0 && voc_parts > nSmallTopKMaxVocParts; --n_block)
    {
        int smem_per_block = max_smem_per_sm / n_block;
        int dyn_smem_size = smem_per_block - extra_smem;
        dyn_smem_size -= dyn_smem_size % sizeof(T);
        voc_parts = (sizeof(T) * V + dyn_smem_size - 1) / dyn_smem_size;
    }

    if (voc_parts <= nSmallTopKMaxVocParts)
    {
        // Use stage 1 fast kernel
        int const voc_size_chunk = (V + voc_parts - 1) / voc_parts;
        int const dyn_smem_size = sizeof(T) * voc_size_chunk;
        if (dyn_smem_size >= (48 << 10))
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(beamStage1FastKernel<T, items_per_thread, 2 * MAX_K, blockSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem_size));
        }
        dim3 gridSize(batch_size * beam_width, voc_parts);
        beamStage1FastKernel<T, items_per_thread, 2 * MAX_K, blockSize><<<gridSize, blockSize, dyn_smem_size, stream>>>(
            logits, bias, finished, temp_buffer, V, beam_width, end_ids, voc_size_chunk);
    }
    else
    {
        // use stage 1 base kernel
        int voc_parts = 4;
        if (batch_size * beam_width < 256)
        {
            // TODO: add heuristics for base stage 1 kernel
            // Volta has 80 SMs, so we aim for three waves
            voc_parts = (240 + batch_size * beam_width - 1) / (batch_size * beam_width);
            voc_parts = std::min(128, voc_parts); // we implement up to 128
        }
        cudaFuncSetAttribute(beamStage1BaseKernel<T, items_per_thread, 2 * MAX_K, blockSize>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
        dim3 gridSize(batch_size * beam_width, voc_parts);
        beamStage1BaseKernel<T, items_per_thread, 2 * MAX_K, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(logits, bias, finished, temp_buffer, V, beam_width, end_ids);
    }
    sync_check_cuda_error();

    beamStage2KernelLauncher<T, 2 * MAX_K>(
        temp_buffer, cum_log_probs, topk_id_buffer, topk_val_buffer, batch_size, beam_width, voc_parts, V, stream);
#else
    beamKernel<T, items_per_thread, MAX_K, blockSize><<<batch_size * beam_width, blockSize, 0, stream>>>(
        logits, bias, cum_log_probs, finished, topk_id_buffer, topk_val_buffer, V, beam_width, end_ids);
#endif

    sync_check_cuda_error();

    // Keep 2 * beam_width candidates in case of k candidates finishes in one iteration
    size_t const smem_size = sizeof(T) * beam_width * beam_width * 2;

    if (smem_size >= (48 << 10))
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            batchBeamKernel<T, MAX_K * 2, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    batchBeamKernel<T, MAX_K * 2, 32><<<batch_size, 32, smem_size, stream>>>(topk_id_buffer, topk_val_buffer, bh);
    sync_check_cuda_error();
}

#define INSTANTIATE_BEAMSEARCH_K(T, MAX_K)                                                                             \
    template void topK_softMax_kernelLauncher<T, MAX_K>(                                                               \
        T const* logits, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
