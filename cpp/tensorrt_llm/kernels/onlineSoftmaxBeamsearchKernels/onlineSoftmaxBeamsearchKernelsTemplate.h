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
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/onlineSoftmaxBeamsearchKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

#define DO_SPLIT_SMALL_TOP_K_SOFTMAX
static int const SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE = 256;

#define TOPK_FP16_STORAGE 0

#pragma nv_diag_suppress static_var_with_dynamic_init

template <typename T, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void batch_topk_kernel(
    int const* __restrict topk_id, T const* __restrict topk_val, BeamHypotheses bh, int const candidate_size)
{
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const global_batch_idx{bh.ite * bh.local_batch_size + bid};
    int const K{bh.beam_width};
    int const vocab_size{bh.vocab_size};
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;

    float const diversity_rate{bh.diversity_rates[global_batch_idx]};
    float const length_penalty{bh.length_penalties[global_batch_idx]};
    int const early_stopping{bh.early_stoppings[global_batch_idx]};
    int const* input_lengths{bh.input_lengths};
    int* sequence_lengths{bh.sequence_lengths_src};

    using cub_kvp = cub::KeyValuePair<int, T>;
    using BlockReduce = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;

    extern __shared__ char buf_s_[];
    T* buf_s = reinterpret_cast<T*>(buf_s_);
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float old_cum_log_probs[MAX_K2 / 2];
    __shared__ cub_kvp cta_topk[MAX_K2];
    __shared__ int selected_beams;
    __shared__ int thread_requiring_update;

    // reposition topk_id, topk_val to data for the current vector
    topk_id += bid * candidate_size;
    topk_val += bid * candidate_size;

    if (tid == 0)
    {
        selected_beams = 0;
    }
    if (tid < K)
    {
        old_cum_log_probs[tid] = bh.cum_log_probs_src[bid * K + tid];
    }
    __syncthreads();

    if (bh.num_beams != nullptr)
    {
        // Beam search is enabled
        if (bh.num_beams[global_batch_idx] == 0 && tid == 0)
        {
            // Initialize worst_score in the first time
            bh.min_normed_scores[global_batch_idx] = FLT_MAX;
        }
        else if (early_stopping && bh.num_beams[global_batch_idx] == K
            || !early_stopping && bh.finished[bid * K].isFinished())
        {
            // We have got enough beams
            return;
        }
    }

    // Get top 2K tokens from cadidates
    cub::ArgMax arg_max;
    cub_kvp partial_topk{candidate_size - 1, -MAX_T_VAL};

    for (int id = tid; id < candidate_size; id += THREADBLOCK_SIZE)
    {
        int const index = bh.num_beams == nullptr ? id % K : id / 2 / K;
        T val = topk_val[id] + static_cast<T>(diversity_rate * index); // use token score for TopK
        cub_kvp new_elem{id, val};
        partial_topk = arg_max(partial_topk, new_elem);
        buf_s[id] = val;
    }
    __syncthreads();

    for (int i = 0; i < 2 * K; ++i)
    {
        cub_kvp total_topk = BlockReduce(temp_storage).Reduce(partial_topk, arg_max);
        if (tid == 0)
        {
            cta_topk[i] = total_topk;
            buf_s[total_topk.key] = -MAX_T_VAL;
            thread_requiring_update = total_topk.key % THREADBLOCK_SIZE;
        }
        __syncthreads();

        // Only one thread needs to update the old partial before the next block reduce.
        // No need to do this in the last iteration.
        if (tid == thread_requiring_update && i < (2 * K - 1))
        {
            partial_topk.key = candidate_size - 1;
            partial_topk.value = -MAX_T_VAL;
            for (int index = tid; index < candidate_size; index += THREADBLOCK_SIZE)
            {
                cub_kvp new_elem{index, buf_s[index]};
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
            bool const is_end_token = topk_id[current_key] % vocab_size == bh.end_ids[bid];
            if (i < K && bh.num_beams != nullptr && is_end_token)
            {
                // Condition of this branch
                // In Beam search mode, this token is end_token and belongs to top K range in Beam search mode
                int const seq_len = sequence_lengths[bid * K + i] - input_lengths[global_batch_idx];
                int const pad = static_cast<int>(!bh.finished[bid * K + i].isFinished());
                float const normed_score = apply_length_penalty(current_value, seq_len + pad, length_penalty);
                int beam_idx = bh.num_beams[global_batch_idx];
                if (beam_idx == K)
                {
                    // There are already K beams
                    if (normed_score < bh.min_normed_scores[global_batch_idx])
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
                            if (bh.normed_scores[global_batch_idx * (K * 2) + j]
                                == bh.min_normed_scores[global_batch_idx])
                            {
                                beam_idx = j;
                                bh.num_beams[global_batch_idx]--;
                                bh.min_normed_scores[global_batch_idx] = FLT_MAX;
                                bh.normed_scores[global_batch_idx * (K * 2) + j] = normed_score;
                                for (int l = 0; l < K; l++)
                                {
                                    bh.min_normed_scores[global_batch_idx] = min(bh.min_normed_scores[global_batch_idx],
                                        bh.normed_scores[global_batch_idx * (K * 2) + l]);
                                }
                                break;
                            }
                        }
                    }
                }
                int prev_id = (topk_id[current_key] / vocab_size) % K;
                int const current_step = sequence_lengths[bid * K + prev_id];
                int const tgt_id_offset = ((bid + bh.ite * bh.local_batch_size) * (K * 2) + beam_idx) * bh.max_seq_len;
                bh.output_ids_tgt[tgt_id_offset + current_step] = bh.end_ids[bid];
                if (bh.log_probs != nullptr)
                {
                    bh.log_probs[tgt_id_offset + current_step]
                        = (float) topk_val[current_key] - old_cum_log_probs[(topk_id[current_key] / vocab_size) % K];
                }
                // Copy finished beam from "%% self.output_ids" to "%% self.beam_hyps_output_ids_tgt"
                for (int j = current_step - 1; j >= 0; j--)
                {
                    bh.output_ids_tgt[tgt_id_offset + j] = bh.output_ids_tgt_ptr[bid][prev_id * bh.max_seq_len + j];
                    prev_id = bh.parent_ids_tgt_ptr[bid][prev_id * bh.max_seq_len + j];
                }
                if (bh.log_probs != nullptr && bh.log_probs_src != nullptr)
                {
                    prev_id = (topk_id[current_key] / vocab_size) % K;
                    for (int j = current_step - 1; j >= 0; j--)
                    {
                        int const index = j * bh.batch_size * K + bh.ite * bh.local_batch_size * K + bid * K + prev_id;
                        bh.log_probs[tgt_id_offset + j] = bh.log_probs_src[index];
                        prev_id = bh.parent_ids_tgt_ptr[bid][prev_id * bh.max_seq_len + j];
                    }
                }
                int const tgt_beam_idx = global_batch_idx * (K * 2) + beam_idx;
                bh.sequence_lengths_tgt[tgt_beam_idx] = current_step;
                bh.normed_scores[tgt_beam_idx] = normed_score;
                bh.min_normed_scores[global_batch_idx]
                    = min(bh.min_normed_scores[global_batch_idx], bh.normed_scores[tgt_beam_idx]);
                bh.num_beams[global_batch_idx]++;
                bh.cum_log_probs[tgt_beam_idx] = (float) topk_val[current_key];
            }
            else if (i < K || bh.num_beams != nullptr && !is_end_token)
            {
                // Condition of this branch
                // 1. bh.num_beams == nullptr && i <  K, i.e., beam search is disable
                // 2. bh.num_beams != nullptr && i <  K && is_end_token == false, i.e., add token at the end
                // 3. bh.num_beams != nullptr && i >= K && is_end_token == false, i.e., add token at the end
                int const current_step = sequence_lengths[bid * K + selected_beams];
                // Write the selected token to output.output_ids
                bh.output_ids_tgt_ptr[bid][selected_beams * bh.max_seq_len + current_step] = topk_id[current_key];
                if (bh.log_probs_src != nullptr)
                {
                    bh.log_probs_src[current_step * bh.batch_size * K + bid * K + selected_beams]
                        = (float) topk_val[current_key] - old_cum_log_probs[(topk_id[current_key] / vocab_size) % K];
                }
                bh.cum_log_probs_src[bid * K + selected_beams] = (float) topk_val[current_key];
                selected_beams++;
            }
            else
            {
                // Condition of this branch, which we do nothing for it
                // 1. bh.num_beams == nullptr && i >= K, i.e., beam search is disable
                // 2. bh.num_beams != nullptr && i >= K && is_end_token == true, i.e., ignore the worse beams
            }

            if (selected_beams >= K)
            {
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
            // enough beams with early_stopping
            bh.is_done[bid] = true;
        }
        else
        {
            // Condition of this branch
            // 1. enough beams with early_stopping == 0, i.e. non_early_stopping
            // 2. enough beams with early_stopping being other values, i.e. early_stopping == "never" in HF
            // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py#L990
            int seq_len = sequence_lengths[bid * K] + 1 - input_lengths[global_batch_idx];
            float const best_sum_logprobs = cta_topk[0].value;
            // According to semantics of HF, cta_topk[0].value is used as best_sum_logprobs
            // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py#L307
            // But maybe bh.cum_log_probs_src[bid * K + i] is more suitable?
            if (early_stopping != 0 && length_penalty > 0.0f)
            {
                // Specialize for early_stopping == "never" and length_penalty > 0
                seq_len = bh.max_seq_len - input_lengths[global_batch_idx];
            }
            float const highest_attainable_score = apply_length_penalty(best_sum_logprobs, seq_len, length_penalty);
            bh.is_done[bid] = bh.min_normed_scores[global_batch_idx] >= highest_attainable_score;
        }
    }
    __syncthreads();

    // Update sequence_lengths, parent_ids, output_ids and finished
    __shared__ int s_sequence_lengths[MAX_K2 / 2];
    if (tid < K)
    {
        s_sequence_lengths[tid] = sequence_lengths[bid * K + tid];
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
        int const new_id = bh.output_ids_tgt_ptr[bid][tid * bh.max_seq_len + current_step];
        int const new_beam_id = (new_id / vocab_size) % K;
        int const new_word_id = new_id % vocab_size;
        sequence_lengths[bb_index] = s_sequence_lengths[new_beam_id];
        if (new_word_id == bh.end_ids[bid])
        {
            bh.finished[bb_index].setFinishedEOS();
        }
        bh.parent_ids_tgt_ptr[bid][tid * bh.max_seq_len + current_step] = new_beam_id;
        bh.output_ids_tgt_ptr[bid][tid * bh.max_seq_len + current_step] = new_word_id;
        if (early_stopping && (bh.num_beams != nullptr && bh.num_beams[bh.ite * bh.local_batch_size + bid] == K)
            || !early_stopping && bh.is_done[bid]) // TODO: simplify this condition
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
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beam_online_softmax_topk_kernel(T const* __restrict log_probs,
    T const* __restrict bias, float const* __restrict cum_log_probs, FinishedState const* __restrict finished,
    int* __restrict topk_id, T* __restrict topk_val, int vocab_size, int K, int const* __restrict end_ids)
{
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;

    typedef cub::BlockReduce<TopKMD<float, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopKMD<float, MAX_K> partial;
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    if (finished[bid].isFinished())
    {
        for (int id = tid; id < vocab_size; id += THREADBLOCK_SIZE)
        {
            float val = (id == end_ids[bid / K]) ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(val, id);
        }
    }
    else
    {
        T const* local_log_probs = log_probs + bid * vocab_size;
        for (int id = tid; id < vocab_size; id += THREADBLOCK_SIZE)
        {
            float val = local_log_probs[id] + bias[id];
            MD new_elem{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(val, id);
        }
    }

    TopKMD<float, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<float, MAX_K>);

    if (tid == 0)
    {
        int* local_topk_id = topk_id + bid * K;
        T const* local_topk_val = topk_val + bid * K;
        float const d_total_log = logf(total.md.d);
        float local_cum_log_probs = cum_log_probs[bid];
        for (int i = 0; i < K; ++i)
        {
            local_topk_id[i] = total.topk.p[i] + bid * vocab_size;
            local_topk_val[i] = total.topk.u[i] - total.md.m - d_total_log + local_cum_log_probs;
        }
    }
}

template <typename T, int ITEMS_PER_THREAD, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__ void beam_online_softmax_topk_stage1_kernel_base(
    T const* __restrict log_probs, T const* __restrict bias, FinishedState const* __restrict finished,
    float* __restrict tmp_buffer, int vocab_size, int K, int const* __restrict end_ids)
{
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;
    int const PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;
    // one threadblock has multiple sections per vocab_size
    int const v_local = (vocab_size + gridDim.y - 1) / gridDim.y;
    int const section_start = v_local * blockIdx.y;
    int const section_end = std::min(section_start + v_local, vocab_size);

#if TOPK_FP16_STORAGE == 1
    typedef cub::BlockReduce<TopKMD<__half, MAX_K2>, THREADBLOCK_SIZE> BlockReduce;
    TopKMD<__half, MAX_K2> partial;
#else
    typedef cub::BlockReduce<TopKMD<T, MAX_K2>, THREADBLOCK_SIZE> BlockReduce;
    TopKMD<T, MAX_K2> partial;
#endif

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float buf_s[PACKED_TOP_KMD_SIZE];

    for (int i = 0; i < MAX_K2; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    if (finished[bid].isFinished())
    {
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            float val = (id == end_ids[bid / K]) ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem_md{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem_md);
            partial.topk.insert(val, id);
        }
    }
    else
    {
        T const* local_log_probs = log_probs + bid * vocab_size;
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            T b = bias == nullptr ? (T) 0.0f : bias[id];
            T val = local_log_probs[id] + b;
            MD new_elem_md{val, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem_md);
            partial.topk.insert(val, id);
        }
    }

#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, MAX_K2> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<__half, MAX_K2>);
#else
    TopKMD<T, MAX_K2> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K2>);
#endif

    if (tid == 0)
    {
        for (int i = 0; i < 2 * K; i++)
        {
            reinterpret_cast<int*>(buf_s)[i] = total.topk.p[i] + bid * vocab_size; // trtllm needs absolute id
            buf_s[MAX_K2 + i] = total.topk.u[i];
        }
        buf_s[2 * MAX_K2] = total.md.d;
        buf_s[2 * MAX_K2 + 1] = total.md.m;
    }
    __syncthreads();

    float* local_tmp_buffer = tmp_buffer + bid * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE;
    for (int id = tid; id < PACKED_TOP_KMD_SIZE; id += THREADBLOCK_SIZE)
    {
        local_tmp_buffer[id] = buf_s[id];
    }
}

template <typename T, int ITEMS_PER_THREAD, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__ void beam_online_softmax_topk_stage1_kernel_fast(
    T const* __restrict log_probs, T const* __restrict bias, FinishedState const* __restrict finished,
    float* __restrict t, int vocab_size, int K, int const* __restrict end_ids, int const v_local)
{
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;
    int const PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;
    // one threadblock has multiple sections per vocab_size
    int const section_start = v_local * blockIdx.y;
    int const section_end = std::min(section_start + v_local, vocab_size);
    int const valid_smem_length = section_end - section_start;

#if TOPK_FP16_STORAGE == 1
    using cub_kvp = cub::KeyValuePair<int, __half>;
#else
    using cub_kvp = cub::KeyValuePair<int, T>;
#endif

    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
    using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;
    auto reduce_md_func = [](const MD& a, const MD& b) { return reduce_md_op(a, b); };

    extern __shared__ char buf_smem_logprobs_[];
    T* buf_smem_logprobs = reinterpret_cast<T*>(buf_smem_logprobs_);
    __shared__ float buf_s[PACKED_TOP_KMD_SIZE];
    __shared__ int thread_requiring_update;

    __shared__ union
    {
        typename BlockReduceMD::TempStorage md_smem;
        typename BlockReduceTopK::TempStorage topk_smem;
    } temp_storage;

    cub::ArgMax arg_max;
    cub_kvp partial_topk{vocab_size - 1, -MAX_T_VAL};
    MD partial_md{-MAX_T_VAL, 0.0f};
    if (finished[bid].isFinished())
    {
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            float const val = (id == end_ids[bid / K]) ? MAX_T_VAL : -MAX_T_VAL;
            int const smem_index = id - section_start;
            buf_smem_logprobs[smem_index] = val;
            MD new_elem_md{val, 1.0F};
            partial_md = reduce_md_op(partial_md, new_elem_md);
            cub_kvp new_elem_topk{smem_index, val};
            partial_topk = arg_max(partial_topk, new_elem_topk);
        }
    }
    else
    {
        T const* local_log_probs = log_probs + bid * vocab_size;
#pragma unroll 1
        for (int id = section_start + tid; id < section_end; id += THREADBLOCK_SIZE)
        {
            T const b = bias == nullptr ? (T) 0.0f : bias[id];
            T const val = local_log_probs[id] + b;
            int const smem_index = id - section_start;
            buf_smem_logprobs[smem_index] = val;
            MD new_elem_md{val, 1.0F};
            partial_md = reduce_md_op(partial_md, new_elem_md);
            cub_kvp new_elem_topk{smem_index, val};
            partial_topk = arg_max(partial_topk, new_elem_topk);
        }
    }
    __syncthreads();

    for (int i = 0; i < 2 * K; ++i)
    {
        // Pop the best choice from "total_topk" to "buf_s" per iteration
        cub_kvp total_topk = BlockReduceTopK(temp_storage.topk_smem).Reduce(partial_topk, arg_max);

        if (tid == 0)
        {
            int const index = bid * vocab_size + section_start + total_topk.key;
            reinterpret_cast<int*>(buf_s)[i] = index;
            buf_s[MAX_K2 + i] = total_topk.value;
            buf_smem_logprobs[total_topk.key] = -MAX_T_VAL; // delete the value of the best choice
            thread_requiring_update = total_topk.key % THREADBLOCK_SIZE;
        }
        __syncthreads();

        if (tid == thread_requiring_update && i < 2 * K - 1)
        {
            // The thread with the biggest element updates its partial_topk
            // No need to do this in the last iteration
            partial_topk.key = vocab_size - 1;
            partial_topk.value = -MAX_T_VAL;
            for (int index = tid; index < valid_smem_length; index += THREADBLOCK_SIZE)
            {
                cub_kvp new_elem{index, buf_smem_logprobs[index]};
                partial_topk = arg_max(partial_topk, new_elem);
            }
        }
    }

    MD total_md = BlockReduceMD(temp_storage.md_smem).Reduce(partial_md, reduce_md_func);

    if (tid == 0)
    {
        buf_s[2 * MAX_K2] = total_md.d;
        buf_s[2 * MAX_K2 + 1] = total_md.m;
    }
    __syncthreads();

    float* local_t = t + bid * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE;
    for (int id = tid; id < PACKED_TOP_KMD_SIZE; id += THREADBLOCK_SIZE)
    {
        local_t[id] = buf_s[id];
    }
}

template <typename T, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beam_online_softmax_topk_stage2_kernel(
    float const* __restrict temp_storage, float const* __restrict cum_log_probs, int* __restrict ids,
    T* __restrict vals, int K, int parts_per_beam, int const vocab_size)
{
    int const bid = blockIdx.x;
    int const tid = threadIdx.x;
    T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;
    int const PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;

    using cub_kvp = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
    using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;

    extern __shared__ char buf_s_[];
    float* buf_s = reinterpret_cast<float*>(buf_s_);
    __shared__ cub_kvp buf_smem_kv[MAX_K2];

    __shared__ union
    {
        typename BlockReduceTopK::TempStorage topk_smem;
        typename BlockReduceMD::TempStorage md_smem;

    } shared_temp_storage;

    cub::ArgMax arg_max;
    MD partial_md{-MAX_T_VAL, 0.0f};
    cub_kvp total_topk{vocab_size - 1, -MAX_T_VAL};

    auto reduce_md_func = [](const MD& a, const MD& b) { return reduce_md_op(a, b); };

    // Load and unpack into registers through smem
    float const* local_temp_storage = temp_storage + bid * PACKED_TOP_KMD_SIZE * parts_per_beam;
    for (int idx = tid; idx < PACKED_TOP_KMD_SIZE * parts_per_beam; idx += THREADBLOCK_SIZE)
    {
        buf_s[idx] = local_temp_storage[idx];
    }
    __syncthreads();

    // Find the argmax within each parts_per_beam
    // Find the topK across all parts_per_beam
    for (int k = 0; k < 2 * K; ++k)
    {
        cub_kvp partial_topk{vocab_size - 1, -MAX_T_VAL};
        // Only threads responsible for a chunk will do the computation
        if (tid < parts_per_beam)
        {
            for (int i = 0; i < 2 * K; ++i)
            {
                int const current_index = tid * PACKED_TOP_KMD_SIZE + i;
                T current_value = buf_s[current_index + MAX_K2];
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
            int global_offset = reinterpret_cast<int*>(buf_s)[temp_offset];
            total_topk.key = global_offset;
            buf_smem_kv[k] = total_topk;

            // Invalidate the maximum value within the chunk
            reinterpret_cast<int*>(buf_s)[temp_offset] = vocab_size - 1; // id in share memory
            buf_s[temp_offset + MAX_K2] = -MAX_T_VAL;                    // value in share memory
        }
        __syncthreads();
    }

    // Extract and reduce MD values across the chunks
    if (tid < parts_per_beam)
    {
        partial_md.d = buf_s[tid * PACKED_TOP_KMD_SIZE + 2 * MAX_K2];
        partial_md.m = buf_s[tid * PACKED_TOP_KMD_SIZE + 2 * MAX_K2 + 1];
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
                ids[bid * 2 * K + i] = buf_smem_kv[i].key;
                vals[bid * 2 * K + i] = val + cum_log_probs[bid];
            }
        }
    }
}

template <typename T, int MAX_K2>
void beam_online_softmax_topk_stage2_kernelLauncher(float const* temp_storage, float const* cum_log_probs, int* ids,
    T* vals, int batch_size, int beam_width, int parts_per_beam, cudaStream_t stream, int const vocab_size)
{
    // TODO: rewrite kernel to remove dependence of constant block size to reduce compilation time
    int const smem_stage2_size = parts_per_beam * (2 * MAX_K2 + 2) * sizeof(float);

    if (parts_per_beam <= 32)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K2, 32>
            <<<batch_size * beam_width, 32, smem_stage2_size, stream>>>(
                temp_storage, cum_log_probs, ids, vals, beam_width, parts_per_beam, vocab_size);
        return;
    }
    if (parts_per_beam <= 64)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K2, 64>
            <<<batch_size * beam_width, 64, smem_stage2_size, stream>>>(
                temp_storage, cum_log_probs, ids, vals, beam_width, parts_per_beam, vocab_size);
        return;
    }
    if (parts_per_beam <= 128)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K2, 128>
            <<<batch_size * beam_width, 128, smem_stage2_size, stream>>>(
                temp_storage, cum_log_probs, ids, vals, beam_width, parts_per_beam, vocab_size);
        return;
    }
    assert(0);
}

template <typename T, int MAX_K>
void topK_softMax_kernelLauncher(T const* log_probs, T const* bias, void* temp_storage, int const temp_storage_size,
    BeamHypotheses& bh, cudaStream_t stream)
{
    int const batch_size{bh.local_batch_size};
    int const beam_width{bh.beam_width};
    int const vocab_size{bh.vocab_size};
    int const* end_ids{bh.end_ids};
    float* cum_log_probs{bh.cum_log_probs_src};
    FinishedState const* finished{bh.finished};

    int const items_per_thread = 1;
    int const block_sz = (MAX_K < 16) ? ((MAX_K < 8) ? SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE : 128) : 64;

    assert(temp_storage_size % 2 == 0);
    assert(temp_storage_size >= 2 * batch_size * beam_width * beam_width * 2);
    // Input and current sequence lengths are needed for computation of length penalty
    assert(bh.length_penalties == nullptr || bh.sequence_lengths_src != nullptr);

    int const topk_buf_offset = ceil(batch_size * beam_width * beam_width * 2 / 4.) * 4;
    int* topk_id = reinterpret_cast<int*>(temp_storage);
    T* topk_val = reinterpret_cast<T*>(topk_id + topk_buf_offset);
    float* tmp_buffer = reinterpret_cast<float*>(topk_val + topk_buf_offset);

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
    // First, we query the occupancy assuming we need no smem. The goal of this heuristic is to simply run
    // at max occupancy.
    int max_active_blocks = -1;
    TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
        beam_online_softmax_topk_stage1_kernel_fast<T, items_per_thread, 2 * MAX_K, block_sz>, block_sz, 0));

    // We now need to find the max smem on the device and use that to determine the vocab parts in the best case.
    int max_smem_per_sm = -1;
    int max_smem_per_block = -1;
    int device = tensorrt_llm::common::getDevice();
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attr;
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(
        &attr, beam_online_softmax_topk_stage1_kernel_fast<T, items_per_thread, 2 * MAX_K, block_sz>));

    int const constant_smem = attr.sharedSizeBytes;
    int const max_dyn_smem_per_block = max_smem_per_block - constant_smem;
    constexpr int max_parts = 128;
    TLLM_CHECK_WITH_INFO(vocab_size * sizeof(T) <= max_dyn_smem_per_block * max_parts,
        "Vocab size too large for split-k top-k beam search fast path.");

    int const driver_smem_per_block = max_smem_per_sm - max_smem_per_block;
    int const extra_smem = driver_smem_per_block + constant_smem;

    int smem_per_block = max_smem_per_sm / max_active_blocks;
    int dyn_smem_size = smem_per_block - extra_smem;
    dyn_smem_size = dyn_smem_size - (dyn_smem_size % sizeof(T));
    int voc_parts = (sizeof(T) * vocab_size + dyn_smem_size - 1) / dyn_smem_size;

    for (int occ = max_active_blocks - 1; occ > 0 && voc_parts > max_parts; occ--)
    {
        smem_per_block = max_smem_per_sm / occ;
        dyn_smem_size = smem_per_block - extra_smem;
        dyn_smem_size = dyn_smem_size - (dyn_smem_size % sizeof(T));
        voc_parts = (sizeof(T) * vocab_size + dyn_smem_size - 1) / dyn_smem_size;
    }

    // TLLM_CHECK_WITH_INFO(voc_parts <= max_parts, "Invalid value for voc parts");

    // Adjust to use the smallest possible value for dynamic smem to evenly distribute the vocab.
    // This is the smallest value satisfying:
    // voc_parts = ceil((vocab_size * sizeof(T)) / dyn_smem_size)
    // Simple proof:
    // voc_parts >= (vocab_size * sizeof(T)) / dyn_smem_size
    // dyn_smem_size >= (vocab_size * sizeof(T)) / voc_parts
    // For smallest int value, we need:
    // dyn_smem_size >= ceil((vocab_size * sizeof(T)) / voc_parts)

    if (voc_parts <= max_parts)
    {
        // use stage 1 fast kernel
        dyn_smem_size = sizeof(T) * (vocab_size + voc_parts - 1) / voc_parts;
        dim3 grid(batch_size * beam_width, voc_parts);
        // dynamically allocate shared memory
        int const voc_size_chunk = dyn_smem_size / sizeof(T);

        if (dyn_smem_size >= (48 << 10))
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(
                beam_online_softmax_topk_stage1_kernel_fast<T, items_per_thread, 2 * MAX_K, block_sz>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem_size));
        }

        beam_online_softmax_topk_stage1_kernel_fast<T, items_per_thread, 2 * MAX_K, block_sz>
            <<<grid, block_sz, dyn_smem_size, stream>>>(
                log_probs, bias, finished, tmp_buffer, vocab_size, beam_width, end_ids, voc_size_chunk);
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
        dim3 grid(batch_size * beam_width, voc_parts);

        cudaFuncSetAttribute(beam_online_softmax_topk_stage1_kernel_base<T, items_per_thread, 2 * MAX_K, block_sz>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
        beam_online_softmax_topk_stage1_kernel_base<T, items_per_thread, 2 * MAX_K, block_sz>
            <<<grid, block_sz, 0, stream>>>(log_probs, bias, finished, tmp_buffer, vocab_size, beam_width, end_ids);
    }

    sync_check_cuda_error();
#endif

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
    beam_online_softmax_topk_stage2_kernelLauncher<T, 2 * MAX_K>(
        tmp_buffer, cum_log_probs, topk_id, topk_val, batch_size, beam_width, voc_parts, stream, vocab_size);
    sync_check_cuda_error();
#else
    beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
        <<<batch_size * beam_width, block_sz, 0, stream>>>(
            log_probs, bias, cum_log_probs, finished, topk_id, topk_val, vocab_size, beam_width, end_ids);
#endif

    // Keep 2 * MAX_K candidates in case of k candidates finishes in one iteration
    int const candidates = beam_width * beam_width * 2;
    int const smem_size_batch_topk = sizeof(T) * candidates;
    if (smem_size_batch_topk >= (48 << 10))
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            batch_topk_kernel<T, MAX_K * 2, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_batch_topk));
    }

    batch_topk_kernel<T, MAX_K * 2, 32>
        <<<batch_size, 32, smem_size_batch_topk, stream>>>(topk_id, topk_val, bh, candidates);
    sync_check_cuda_error();
}

#define INSTANTIATE_BEAMSEARCH_K(T, MAX_K)                                                                             \
    template void topK_softMax_kernelLauncher<T, MAX_K>(T const* log_probs, T const* bias, void* temp_storage,         \
        int const temp_storage_size, BeamHypotheses& bh, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
