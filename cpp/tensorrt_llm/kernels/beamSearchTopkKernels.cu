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
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__device__ __forceinline__ T apply_length_penalty(T log_prob, int length, float length_penalty)
{
    // score = log(prob) / (length ^ length_penalty)
    if (length_penalty == 0.0f || length == 1)
    {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf((float) length, length_penalty));
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beam_topK_kernel(const T* log_probs, int* topk_tmp_id_buf, T* topk_tmp_val_buf, const bool* finished,
        const int* sequence_lengths, const int vocab_size, T diversity_rate, float length_penalty)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x; // batch beam index.
    TopK<T, MAX_K> partial;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

#pragma unroll
    for (int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
    {
        int index = elem_id + block_id * vocab_size;
        T score = length_penalty == 0.0f
            ? log_probs[index]
            : apply_length_penalty(log_probs[index],
                finished[block_id] ? sequence_lengths[block_id] : sequence_lengths[block_id] + 1, length_penalty);
        partial.insert(score, index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        int index = block_id * MAX_K;

#pragma unroll
        for (int i = 0; i < MAX_K; ++i)
        {
            topk_tmp_id_buf[index + i] = total.p[i];
            topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T) i;
        }
    }
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel(int* topk_tmp_id_buf, T* topk_tmp_val_buf, int* id_buf)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    TopK<T, MAX_K> partial;
    if (thread_id == 0)
    {
        for (int i = 0; i < MAX_K; ++i)
        {
            partial.p[i] = -1;
            partial.u[i] = -MAX_T_VAL;
        }

        int index = block_id * MAX_K * MAX_K;
        for (int i = 0; i < MAX_K * MAX_K; i++)
        {
            partial.insert((T) topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i]);
        }

        index = block_id * MAX_K;
        for (int i = 0; i < MAX_K; i++)
        {
            id_buf[index + i] = partial.p[i];
        }
    }
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel_v2(int* topk_tmp_id_buf, T* topk_tmp_val_buf, int* id_buf)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    TopK<T, MAX_K> partial;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

    int ite = MAX_K * MAX_K / THREADBLOCK_SIZE;
#pragma unroll
    for (int i = 0; i < ite; i++)
    {
        int index = bid * MAX_K * MAX_K + i * THREADBLOCK_SIZE + tid;
        partial.insert((T) topk_tmp_val_buf[index], topk_tmp_id_buf[index]);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (tid == 0)
    {
#pragma unroll
        for (int i = 0; i < MAX_K; i++)
        {
            id_buf[bid * MAX_K + i] = total.p[i];
        }
    }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1_opt3(const T* __restrict log_probs, T* tmp_log_probs, int* topk_tmp_id_buf,
    T* topk_tmp_val_buf, const bool* finished, const int* sequence_lengths, const int k, const int vocab_size,
    const float length_penalty, const int* end_ids)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int row_id = bid / BLOCKS_PER_BEAM_;     // row id for log_probs (batchbeam index)
    const int block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam
    const int tmp_log_buf_index = row_id * vocab_size;
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
    TopK_2<T> partial;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    if (finished != nullptr && finished[row_id] == true)
    {
        if (tid < k)
        {
            const int index = tmp_topk_buf_index + tid;
            if (block_lane == 0 && tid == 0)
            {
                const int end_id = end_ids[row_id / k];
                topk_tmp_id_buf[index] = tmp_log_buf_index + end_id;
                topk_tmp_val_buf[index] = log_probs[tmp_log_buf_index + end_id];
            }
            else
            {
                topk_tmp_id_buf[index] = -1;
                topk_tmp_val_buf[index] = -MAX_T_VAL;
            }
        }
        return;
    }

    for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index];
    }

    for (int ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
             elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -MAX_T_VAL;
        }
        __syncthreads();
    }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3(const int* __restrict topk_tmp_id_buf, T* topk_tmp_val_buf, int* ids,
    BeamHypotheses beam_hyps, const int* end_ids, const int vocab_size, const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM_;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    const float length_penalty{beam_hyps.length_penalties == nullptr ? 1.0f : beam_hyps.length_penalties[batch_id]};

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T* s_val = topk_tmp_val_buf + batch_id * size;
    int* s_id = (int*) (array);

    __shared__ int selected_beams;
    __shared__ bool is_stop;

    if (tid == 0)
    {
        selected_beams = 0;
        is_stop = false;
    }
    __syncthreads();
    if (beam_hyps.num_beams != nullptr)
    {
        const int global_batch_idx = beam_hyps.ite * beam_hyps.local_batch_size + batch_id;
        if (beam_hyps.num_beams[global_batch_idx] == 0 && tid == 0)
        {
            // initialize the buffer
            beam_hyps.min_normed_scores[global_batch_idx] = FLT_MAX;
        }
        else if (beam_hyps.num_beams[global_batch_idx] == k)
        {
            return;
        }
    }

    TopK_2<T> partial;

    // In some cases, we may encounter k finished sentences, but scores are bad.
    // So, the max iteration is 2*k here
    for (int ite = 0; ite < 2 * k; ite++)
    {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE_)
        {
            partial.insert(s_val[i], i);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            if (beam_hyps.num_beams != nullptr
                && topk_tmp_id_buf[batch_id * size + total.p] % vocab_size == end_ids[batch_id])
            {
                // if beam_token does not belong to top num_beams tokens, it should not
                // be added. Refer from
                // https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/generation_beam_search.py#L257
                if (ite >= k)
                {
                    s_val[total.p] = -MAX_T_VAL;
                }
                else
                {
                    const int global_batch_idx = beam_hyps.ite * beam_hyps.local_batch_size + batch_id;
                    const float normed_score = apply_length_penalty(s_val[total.p], beam_hyps.step, length_penalty);
                    const int num_beam = beam_hyps.num_beams[global_batch_idx];
                    int beam_idx = num_beam;
                    // If there are beam_width finished sentences, check that the score of
                    // selected candidatet is higher than min_normed_score or not. If
                    // current score is better, replace worst one and update the
                    // min_normed_score.
                    if (num_beam == k)
                    {
                        if (normed_score < beam_hyps.min_normed_scores[global_batch_idx])
                        {
                            // end the tracing and exist this for loop
                            selected_beams = k;
                            is_stop = true;
                            break;
                        }
                        else
                        {
                            // find the beam index which's score = min_normed_score, erase it.
                            for (int j = 0; j < k; j++)
                            {
                                if (beam_hyps.normed_scores[global_batch_idx * k + j]
                                    == beam_hyps.min_normed_scores[global_batch_idx])
                                {
                                    beam_idx = j;
                                    beam_hyps.num_beams[global_batch_idx]--;

                                    beam_hyps.min_normed_scores[global_batch_idx] = FLT_MAX;
                                    beam_hyps.normed_scores[global_batch_idx * k + j] = normed_score;
                                    for (int l = 0; l < k; l++)
                                    {
                                        beam_hyps.min_normed_scores[global_batch_idx]
                                            = min(beam_hyps.min_normed_scores[global_batch_idx],
                                                beam_hyps.normed_scores[global_batch_idx * k + l]);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    const int tgt_id_offset = ((batch_id + beam_hyps.ite * beam_hyps.local_batch_size) * k + beam_idx)
                        * beam_hyps.max_seq_len;
                    beam_hyps.output_ids_tgt[tgt_id_offset + beam_hyps.step] = end_ids[batch_id];

                    int prev_id = (topk_tmp_id_buf[batch_id * size + total.p] / vocab_size) % k;
                    for (int j = beam_hyps.step - 1; j >= 0; j--)
                    {
                        const int src_idx = j * beam_hyps.batch_size * k
                            + beam_hyps.ite * beam_hyps.local_batch_size * k + batch_id * k + prev_id;

                        beam_hyps.output_ids_tgt[tgt_id_offset + j] = beam_hyps.output_ids_src[src_idx];
                        prev_id = beam_hyps.parent_ids_src[src_idx];
                    }
                    const int tgt_beam_idx = global_batch_idx * k + beam_idx;
                    beam_hyps.sequence_lengths_tgt[tgt_beam_idx] = beam_hyps.step;
                    beam_hyps.normed_scores[tgt_beam_idx] = normed_score;
                    beam_hyps.min_normed_scores[global_batch_idx]
                        = min(beam_hyps.min_normed_scores[global_batch_idx], beam_hyps.normed_scores[tgt_beam_idx]);

                    s_val[total.p] = -MAX_T_VAL;

                    beam_hyps.num_beams[global_batch_idx]++;
                }
            }
            else
            {
                s_id[selected_beams] = total.p;
                s_val[total.p] = -MAX_T_VAL;
                selected_beams++;
            }
        }
        __syncthreads();
        if (selected_beams >= k)
        {
            break;
        }
    }
    if (tid < k && is_stop == false)
    {
        ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
    }
}

template <typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_1_opt2_general(const T* __restrict log_probs, T* tmp_log_probs, int* topk_tmp_id_buf,
    T* topk_tmp_val_buf, const bool* finished, const int* sequence_lengths, const int k, const int vocab_size,
    const float length_penalty)
{
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM;     // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM; // block id for a beam
    const int tmp_log_buf_index = row_id * vocab_size;
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM * k + block_lane * k;
    TopK_2<T> partial;

    for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index];
    }

    for (int ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -MAX_T_VAL;
        }
        __syncthreads();
    }
}

template <typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_2_opt2_general(const int* __restrict topk_tmp_id_buf, T* topk_tmp_val_buf, int* ids,
    BeamHypotheses beam_hyps, const int* end_ids, const int k, const int vocab_size)
{
    const int size = k * k * BLOCKS_PER_BEAM;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    const float length_penalty{beam_hyps.length_penalties == nullptr ? 1.0f : beam_hyps.length_penalties[batch_id]};

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T* s_val = topk_tmp_val_buf + batch_id * size;
    int* s_id = (int*) (array);

    __shared__ int selected_beams;
    __shared__ bool is_stop;

    if (tid == 0)
    {
        selected_beams = 0;
        is_stop = false;
    }
    __syncthreads();
    if (beam_hyps.num_beams != nullptr)
    {
        const int global_batch_idx = beam_hyps.ite * beam_hyps.local_batch_size + batch_id;
        if (beam_hyps.num_beams[global_batch_idx] == 0 && tid == 0)
        {
            beam_hyps.min_normed_scores[global_batch_idx] = FLT_MAX;
        }
        else if (beam_hyps.num_beams[global_batch_idx] == k)
        {
            return;
        }
    }

    TopK_2<T> partial;

    // In some cases, we may encounter k finished sentences, but scores are bad.
    // So, the max iteration is 2*k here
    for (int ite = 0; ite < 2 * k; ite++)
    {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE)
        {
            partial.insert(s_val[i], i);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            if (beam_hyps.num_beams != nullptr
                && topk_tmp_id_buf[batch_id * size + total.p] % vocab_size == end_ids[batch_id])
            {
                // if beam_token does not belong to top num_beams tokens, it should not
                // be added. Refer from
                // https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/generation_beam_search.py#L257
                if (ite >= k)
                {
                    s_val[total.p] = -MAX_T_VAL;
                }
                else
                {
                    const int global_batch_idx = beam_hyps.ite * beam_hyps.local_batch_size + batch_id;
                    const float normed_score = apply_length_penalty(s_val[total.p], beam_hyps.step, length_penalty);
                    const int num_beam = beam_hyps.num_beams[global_batch_idx];
                    int beam_idx = num_beam;
                    // If there are beam_width finished sentences, check that the score of
                    // selected candidatet is higher than min_normed_score or not. If
                    // current score is better, replace worst one and update the
                    // min_normed_score.
                    if (num_beam == k)
                    {
                        if (normed_score < beam_hyps.min_normed_scores[global_batch_idx])
                        {
                            // end the tracing and exist this for loop
                            selected_beams = k;
                            is_stop = true;
                            break;
                        }
                        else
                        {
                            // find the beam index which's score = min_normed_score, erase it.
                            for (int j = 0; j < k; j++)
                            {
                                if (beam_hyps.normed_scores[global_batch_idx * k + j]
                                    == beam_hyps.min_normed_scores[global_batch_idx])
                                {
                                    beam_idx = j;
                                    beam_hyps.num_beams[global_batch_idx]--;

                                    beam_hyps.min_normed_scores[global_batch_idx] = FLT_MAX;
                                    beam_hyps.normed_scores[global_batch_idx * k + j] = normed_score;
                                    for (int l = 0; l < k; l++)
                                    {
                                        beam_hyps.min_normed_scores[global_batch_idx]
                                            = min(beam_hyps.min_normed_scores[global_batch_idx],
                                                beam_hyps.normed_scores[global_batch_idx * k + l]);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    const int tgt_id_offset = ((batch_id + beam_hyps.ite * beam_hyps.local_batch_size) * k + beam_idx)
                        * beam_hyps.max_seq_len;
                    beam_hyps.output_ids_tgt[tgt_id_offset + beam_hyps.step] = end_ids[batch_id];

                    int prev_id = (topk_tmp_id_buf[batch_id * size + total.p] / vocab_size) % k;
                    for (int j = beam_hyps.step - 1; j >= 0; j--)
                    {
                        const int src_idx = j * beam_hyps.batch_size * k
                            + beam_hyps.ite * beam_hyps.local_batch_size * k + batch_id * k + prev_id;

                        beam_hyps.output_ids_tgt[tgt_id_offset + j] = beam_hyps.output_ids_src[src_idx];
                        prev_id = beam_hyps.parent_ids_src[src_idx];
                    }
                    const int tgt_beam_idx = global_batch_idx * k + beam_idx;
                    beam_hyps.sequence_lengths_tgt[tgt_beam_idx] = beam_hyps.step;
                    beam_hyps.normed_scores[tgt_beam_idx] = normed_score;
                    beam_hyps.min_normed_scores[global_batch_idx]
                        = min(beam_hyps.min_normed_scores[global_batch_idx], beam_hyps.normed_scores[tgt_beam_idx]);

                    s_val[total.p] = -MAX_T_VAL;

                    beam_hyps.num_beams[global_batch_idx]++;
                }
            }
            else
            {
                s_id[selected_beams] = total.p;
                s_val[total.p] = -MAX_T_VAL;
                selected_beams++;
            }
        }
        __syncthreads();
        if (selected_beams >= k)
        {
            break;
        }
    }
    if (tid < k && is_stop == false)
    {
        ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
    }
}

#define CASE_K_DIV(K, BLOCK_SIZE_1, BLOCK_SIZE_2)                                                                      \
    case K:                                                                                                            \
        beam_topK_kernel<T, K, BLOCK_SIZE_2><<<batch_size * beam_width, BLOCK_SIZE_2, 0, stream>>>(log_probs,          \
            topk_tmp_id_buf, topk_tmp_val_buf, finished, sequence_lengths, vocab_size, diversity_rate,                 \
            length_penalty);                                                                                           \
        if (K < 10)                                                                                                    \
            batch_topK_kernel<T, K, BLOCK_SIZE_1>                                                                      \
                <<<batch_size, BLOCK_SIZE_1, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);                     \
        else                                                                                                           \
            batch_topK_kernel_v2<T, K, 32><<<batch_size, 32, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);     \
        break;

#define CASE_K(K, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)                                                      \
    case K:                                                                                                            \
        topk_stage_1_opt3<float, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                                                      \
            <<<batch_size * K * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>(log_probs, temp_log_probs,               \
                topk_tmp_id_buf, topk_tmp_val_buf, finished, sequence_lengths, beam_width, vocab_size, length_penalty, \
                end_ids);                                                                                              \
        topk_stage_2_opt3<float, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                                      \
            <<<batch_size, BLOCK_SIZE_2_, K * sizeof(int), stream>>>(                                                  \
                topk_tmp_id_buf, topk_tmp_val_buf, ids, *beam_hyps, end_ids, vocab_size, beam_width);                  \
        sync_check_cuda_error();                                                                                       \
        break;

template <typename T>
void invokeTopkBeamSearch(void* workspace, size_t& workspace_size, T* log_probs, int* ids, BeamHypotheses* beam_hyps,
    const bool* finished, const int* sequence_lengths, const int batch_size, const int beam_width,
    const int vocab_size_padded_, const T diversity_rate, const float length_penalty, const int* end_ids,
    cudaStream_t stream)
{
    // log_probs: (batch, beam, vocab) cumulative log_probs of beams ending with a
    // token.
    const int vocab_size = vocab_size_padded_;
    // Beam size should be less than or equal to vocab size.
    assert(beam_width <= vocab_size);
    // Beam search needs the sequence lengths of beams to apply length penalty.
    assert(length_penalty == 0.0f || sequence_lengths != nullptr);
    const int max_block_per_beam = 8;
    int temp_log_probs_buf_size = batch_size * beam_width * vocab_size;                    // type float
    int topk_tmp_ids_buf_size = batch_size * beam_width * beam_width * max_block_per_beam; // type int
    int topk_tmp_val_buf_size = batch_size * beam_width * beam_width * max_block_per_beam; // type float

    // prevent memory misaligned address
    temp_log_probs_buf_size = (int) (ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size = (int) (ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int) (ceil(topk_tmp_val_buf_size / 4.)) * 4;

    if (workspace == nullptr)
    {
        workspace_size = sizeof(float) * temp_log_probs_buf_size + sizeof(int) * topk_tmp_ids_buf_size
            + sizeof(float) * topk_tmp_val_buf_size;
        return;
    }
    else
    {
        T* temp_log_probs = (T*) workspace;
        int* topk_tmp_id_buf = (int*) (temp_log_probs + temp_log_probs_buf_size);
        T* topk_tmp_val_buf = (T*) (topk_tmp_id_buf + topk_tmp_ids_buf_size);
        if (diversity_rate == 0.0f)
        {
            switch (beam_width)
            {
                CASE_K(1, 128, 128, 8);
                CASE_K(4, 128, 128, 8);
                CASE_K(10, 128, 128, 8);
                CASE_K(16, 128, 128, 5);
                CASE_K(32, 256, 128, 1);
                CASE_K(64, 256, 256, 1);
            default:
                topk_stage_1_opt2_general<T, 128, 1><<<batch_size * beam_width * 1, 128, 0, stream>>>(log_probs,
                    temp_log_probs, topk_tmp_id_buf, topk_tmp_val_buf, finished, sequence_lengths, beam_width,
                    vocab_size, length_penalty);
                topk_stage_2_opt2_general<T, 128, 1>
                    <<<batch_size, 128, beam_width * beam_width * 1 * sizeof(float) + beam_width * sizeof(int),
                        stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids, *beam_hyps, end_ids, beam_width, vocab_size);
                break;
            }
        }
        else
        {
            switch (beam_width)
            {
                CASE_K_DIV(1, 256, 256);
                CASE_K_DIV(4, 256, 256);
                CASE_K_DIV(16, 256, 64);
                CASE_K_DIV(32, 256, 64);
                CASE_K_DIV(64, 256, 64);
            default: TLLM_THROW("Topk kernel does not support beamwidth = %d \n", beam_width);
            }
        }
        return;
    }
}

#undef CASE_K
#undef CASE_K_DIV

template void invokeTopkBeamSearch(void* workspace, size_t& workspace_size, float* log_probs, int* ids,
    BeamHypotheses* beam_hyps, const bool* finished, const int* sequence_lengths, const int batch_size,
    const int beam_width, const int vocab_size_padded_, const float diversity_rate, const float length_penalty,
    const int* end_ids, cudaStream_t stream);

template <typename T>
__global__ void tileEncoderResults(T* tiled_output, int* tiled_sequence_length, const T* output,
    const int* sequence_length, const uint32_t batch_size, const uint32_t beam_width, const uint32_t d_model)
{
    if (blockIdx.x == 0)
    {
        for (uint32_t i = threadIdx.x; i < batch_size * beam_width; i += blockDim.x)
        {
            tiled_sequence_length[i] = sequence_length[i / beam_width];
        }
    }

    int tgt_offset
        = blockIdx.x * gridDim.y * gridDim.z * d_model + blockIdx.y * gridDim.z * d_model + blockIdx.z * d_model;
    int src_offset = blockIdx.x * gridDim.z * d_model + blockIdx.z * d_model;
    for (uint32_t i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        tiled_output[i + tgt_offset] = output[i + src_offset];
    }
}

template <typename T>
void invokeTileEncoderResults(T* tiled_output, int* tiled_sequence_length, const T* output, const int* sequence_length,
    const size_t batch_size, const size_t beam_width, const size_t mem_max_seq_len, const size_t d_model,
    cudaStream_t stream)
{
    // tiled_output: [batch_size, beam_width, mem_max_seq_len, d_model]
    // tiled_sequence_length: [batch_size, beam_width]

    // output: [batch_size, mem_max_seq_len, d_model]
    // sequence_length [batch_size]

    dim3 grid(batch_size, beam_width, mem_max_seq_len);
    bool is_half2 = (std::is_same<T, half>::value) && (d_model % 2 == 0);

    if (is_half2)
    {
        using T2 = typename TypeConverter<T>::Type; // fp16 to half2, bf16 to bf162
        dim3 block(min(512, (int) (d_model / 2)));
        tileEncoderResults<T2><<<grid, block, 0, stream>>>((T2*) tiled_output, tiled_sequence_length,
            (const T2*) output, sequence_length, batch_size, beam_width, d_model / 2);
    }
    else
    {
        dim3 block(min(512, (int) d_model));
        tileEncoderResults<T><<<grid, block, 0, stream>>>(
            tiled_output, tiled_sequence_length, output, sequence_length, batch_size, beam_width, d_model);
    }
}

template void invokeTileEncoderResults(float* tiled_output, int* tiled_sequence_length, const float* output,
    const int* sequence_length, const size_t batch_size, const size_t beam_width, const size_t mem_max_seq_len,
    const size_t d_model, cudaStream_t stream);

template void invokeTileEncoderResults(half* tiled_output, int* tiled_sequence_length, const half* output,
    const int* sequence_length, const size_t batch_size, const size_t beam_width, const size_t mem_max_seq_len,
    const size_t d_model, cudaStream_t stream);

template void invokeTileEncoderResults(half2* tiled_output, int* tiled_sequence_length, const half2* output,
    const int* sequence_length, const size_t batch_size, const size_t beam_width, const size_t mem_max_seq_len,
    const size_t d_model, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeTileEncoderResults(__nv_bfloat16* tiled_output, int* tiled_sequence_length,
    const __nv_bfloat16* output, const int* sequence_length, const size_t batch_size, const size_t beam_width,
    const size_t mem_max_seq_len, const size_t d_model, cudaStream_t stream);
#endif

__global__ void insertUnfinishedPath(BeamHypotheses beam_hyps, const FinishedState* finished,
    const float* cum_log_probs, const int batch_size, const int beam_width)
{
    const int bid = blockIdx.x;
    const int tgt_start_idx = beam_hyps.num_beams[bid];
    const int max_seq_len{beam_hyps.max_seq_len};
    const float length_penalty{beam_hyps.length_penalties == nullptr ? 1.0f : beam_hyps.length_penalties[bid]};
    if (beam_hyps.is_done[bid])
    {
        return;
    }
    for (int beam_idx = 0; beam_idx < beam_width; beam_idx++)
    {
        if (threadIdx.x == 0)
        {
            const int src_beam_idx = bid * beam_width + beam_idx;
            const int tgt_beam_idx = bid * beam_width * 2 + beam_idx + tgt_start_idx;

            const int last_token_idx = beam_hyps.sequence_lengths_src[src_beam_idx] - 1;

            beam_hyps.output_ids_tgt[tgt_beam_idx * max_seq_len + last_token_idx]
                = beam_hyps.output_ids_src[src_beam_idx * max_seq_len + last_token_idx];
            if (beam_hyps.log_probs != nullptr && beam_hyps.log_probs_src != nullptr)
            {
                beam_hyps.log_probs[tgt_beam_idx * max_seq_len + last_token_idx]
                    = beam_hyps.log_probs_src[last_token_idx * batch_size * beam_width + src_beam_idx];
            }
            int prev_id = beam_hyps.parent_ids_src[src_beam_idx * max_seq_len + last_token_idx];
            for (int token_idx = last_token_idx - 1; token_idx >= 0; token_idx--)
            {
                // output_ids_tgt need to use max_seq_len + 1 because its shape is
                // [bs, beam_width, max_seq_len + 1]
                beam_hyps.output_ids_tgt[tgt_beam_idx * max_seq_len + token_idx]
                    = beam_hyps.output_ids_src[bid * beam_width * max_seq_len + prev_id * max_seq_len + token_idx];
                if (beam_hyps.log_probs != nullptr && beam_hyps.log_probs_src != nullptr)
                {
                    beam_hyps.log_probs[tgt_beam_idx * max_seq_len + token_idx]
                        = beam_hyps.log_probs_src[token_idx * batch_size * beam_width + bid * beam_width + prev_id];
                }
                prev_id = beam_hyps.parent_ids_src[bid * beam_width * max_seq_len + prev_id * max_seq_len + token_idx];
            }
            beam_hyps.sequence_lengths_tgt[tgt_beam_idx] = last_token_idx + 1;

            // TODO huggingface uses total length to normalize the scores, instead of number of generated tokens.
            // Check that is it reasonable or not.
            beam_hyps.normed_scores[tgt_beam_idx] = apply_length_penalty(cum_log_probs[src_beam_idx],
                finished[src_beam_idx].isFinished() ? last_token_idx + 1 : last_token_idx, length_penalty);
            beam_hyps.cum_log_probs[tgt_beam_idx] = cum_log_probs[src_beam_idx];

            beam_hyps.num_beams[bid]++;
        }
    }
}

void invokeInsertUnfinishedPath(BeamHypotheses beam_hyps, const FinishedState* finished, const float* cum_log_probs,
    const int batch_size, const int beam_width, cudaStream_t stream)
{
    insertUnfinishedPath<<<batch_size, 256, 0, stream>>>(beam_hyps, finished, cum_log_probs, batch_size, beam_width);
}

__global__ void copyBatchMajorToGeneralPtr(
    void* output_ids_ptr, int* output_ids, int batch_size, int beam_width, int max_seq_len)
{
    // output_ids_ptr: batch_size int*, each int* has [beam_width, max_seq_len]
    // output_ids: [max_seq_len, batch, beam]
    int** output_ids_int_ptr = (int**) output_ids_ptr;
    for (int idx = threadIdx.x; idx < beam_width * max_seq_len; idx += blockDim.x)
    {
        auto const src_step = idx % max_seq_len;
        auto const src_beam_idx = idx / max_seq_len;
        output_ids_int_ptr[blockIdx.x][idx]
            = output_ids[src_step * batch_size * beam_width + blockIdx.x * beam_width + src_beam_idx];
    }
}

void invokeCopyBatchMajorToGeneralPtr(
    void* output_ids_ptr, int* output_ids, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream)
{
    copyBatchMajorToGeneralPtr<<<batch_size, 256, 0, stream>>>(
        output_ids_ptr, output_ids, batch_size, beam_width, max_seq_len);
}

__global__ void copyGeneralPtrToBatchMajor(
    int* output_ids, void* output_ids_ptr, int batch_size, int beam_width, int max_seq_len)
{
    // output_ids_ptr: batch_size int*, each int* has [beam_width, max_seq_len]
    // output_ids: [max_seq_len, batch, beam]
    int** output_ids_int_ptr = (int**) output_ids_ptr;
    for (int idx = threadIdx.x; idx < beam_width * max_seq_len; idx += blockDim.x)
    {
        auto const tgt_step = idx % max_seq_len;
        auto const tgt_beam_idx = idx / max_seq_len;
        output_ids[tgt_step * batch_size * beam_width + blockIdx.x * beam_width + tgt_beam_idx]
            = output_ids_int_ptr[blockIdx.x][idx];
    }
}

void invokeCopyGeneralPtrToBatchMajor(
    int* output_ids, void* output_ids_ptr, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream)
{
    copyGeneralPtrToBatchMajor<<<batch_size, 256, 0, stream>>>(
        output_ids, output_ids_ptr, batch_size, beam_width, max_seq_len);
}

__global__ void SeqlenMajorToBatchMajor(
    int* batchMajoredIds, int* seqlenMajorIds, int batch_size, int beam_width, int max_seq_len)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size * beam_width * max_seq_len;
         idx += gridDim.x * blockDim.x)
    {
        auto tmp_idx{idx};
        auto const beam_idx{tmp_idx % beam_width};
        tmp_idx = (tmp_idx - beam_idx) / beam_width;
        auto const batch_idx{tmp_idx % batch_size};
        tmp_idx = (tmp_idx - batch_idx) / batch_size;
        auto const seqlen_idx{tmp_idx % max_seq_len};

        batchMajoredIds[batch_idx * beam_width * max_seq_len + beam_idx * max_seq_len + seqlen_idx]
            = seqlenMajorIds[idx];
    }
}

void invokeSeqlenMajorToBatchMajor(
    int* batchMajoredIds, int* seqlenMajorIds, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream)
{
    SeqlenMajorToBatchMajor<<<batch_size, 256, 0, stream>>>(
        batchMajoredIds, seqlenMajorIds, batch_size, beam_width, max_seq_len);
}

} // namespace kernels
} // namespace tensorrt_llm
