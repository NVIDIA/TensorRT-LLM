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
static const int SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE = 256;

#define TOPK_FP16_STORAGE 0

template <typename T>
__device__ __forceinline__ T apply_length_penalty(T log_prob, int length, float length_penalty)
{
    // score = log(prob) / (length)^length_penalty.
    if (length_penalty == 0.0f || length == 1)
    {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf(length, length_penalty));
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel(int* topk_tmp_id_buf, T* topk_tmp_val_buf, int* id_buf)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;
    if (thread_id == 0)
    {
        for (int i = 0; i < MAX_K; ++i)
        {
            partial.p[i] = -1;
            partial.u[i] = -FLT_MAX;
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
__launch_bounds__(THREADBLOCK_SIZE) __global__ void batch_topK_kernel(const int* __restrict topk_tmp_id_buf,
    const T* __restrict topk_tmp_val_buf, int* __restrict id_buf, T* __restrict val_buf)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;
    if (thread_id == 0)
    {
        for (int i = 0; i < MAX_K; ++i)
        {
            partial.p[i] = -1;
            partial.u[i] = -FLT_MAX;
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
            val_buf[index + i] = partial.u[i];
        }
    }
}

template <typename T, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topk_kernel(const int* __restrict x, const T* __restrict y, int** output_ids_ptr, float* __restrict v,
        float* output_log_probs, const FinishedState* finished, const int* sequence_lengths, BeamHypotheses beam_hyps,
        const int V, const int K, const int vocab_size, const float* length_penalties, const float* diversity_rates)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;
    const int global_batch_idx{beam_hyps.ite * beam_hyps.local_batch_size + vector_id};
    const T diversity_rate{diversity_rates[global_batch_idx]};
    const float length_penalty{length_penalties[global_batch_idx]};

    // reposition x, y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    extern __shared__ char buf_s_[]; // intermediate result
    T* buf_s = reinterpret_cast<T*>(buf_s_);
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    using cub_kvp = cub::KeyValuePair<int, T>;
    using BlockReduce = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ int selected_beams;
    __shared__ float old_cum_log_probs[MAX_K2];
    __shared__ cub_kvp cta_topk[MAX_K2];

    if (thread_id == 0)
    {
        selected_beams = 0;
    }
    if (thread_id < K)
    {
        old_cum_log_probs[thread_id] = v[vector_id * K + thread_id];
    }
    __syncthreads();
    if (beam_hyps.num_beams != nullptr)
    {
        if (beam_hyps.num_beams[global_batch_idx] == 0 && thread_id == 0)
        {
            beam_hyps.min_normed_scores[global_batch_idx] = FLT_MAX;
        }
        else if (beam_hyps.num_beams[global_batch_idx] == K)
        {
            return;
        }
    }

    cub::ArgMax arg_max;
    cub_kvp partial_topk{V - 1, -MAX_T_VAL};

    for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        int i = beam_hyps.num_beams == nullptr ? elem_id % K : elem_id / 2 / K;
        T elem = length_penalty == 0.0f
            ? y[elem_id]
            : apply_length_penalty(y[elem_id],
                finished[vector_id * K + i].isFinished() ? sequence_lengths[vector_id * K + i]
                                                         : sequence_lengths[vector_id * K + i] + 1,
                length_penalty);
        elem += diversity_rate * (T) i;
        int elem_idx = elem_id; // x[elem_id];
        cub_kvp new_elem{elem_idx, elem};
        partial_topk = arg_max(partial_topk, new_elem);
        buf_s[elem_id] = elem;
    }
    __syncthreads();

    __shared__ int thread_requiring_update;

    for (int i = 0; i < 2 * K; ++i)
    {
        cub_kvp total_topk = BlockReduce(temp_storage).Reduce(partial_topk, arg_max);

        if (threadIdx.x == 0)
        {
            cta_topk[i] = total_topk;
            buf_s[total_topk.key] = -MAX_T_VAL;
            thread_requiring_update = total_topk.key % THREADBLOCK_SIZE;
        }
        __syncthreads();

        // Only 1 thread needs to update the old partial before the next block reduce. We don't need to do this update
        // on the last iteration.
        if (thread_id == thread_requiring_update && i < (2 * K - 1))
        {
            partial_topk.key = V - 1;
            partial_topk.value = -MAX_T_VAL;
            for (int tid = thread_id; tid < V; tid += THREADBLOCK_SIZE)
            {
                cub_kvp new_elem{tid, buf_s[tid]};
                partial_topk = arg_max(partial_topk, new_elem);
            }
        }
    }

    if (thread_id == 0)
    {
        v += vector_id * K;

        for (int i = 0; i < 2 * K; ++i)
        {
            const int current_key = cta_topk[i].key;
            const T current_value = cta_topk[i].value;
            if (i < K && beam_hyps.num_beams != nullptr && x[current_key] % vocab_size == beam_hyps.end_ids[vector_id])
            {
                // if beam_token does not belong to top num_beams tokens, it should not
                // be added. Refer from
                // https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/generation_beam_search.py#L257
                {
                    const float normed_score = (float) current_value;
                    const int num_beam = beam_hyps.num_beams[global_batch_idx];
                    int beam_idx = num_beam;
                    // If there are beam_width finished sentences, check that the score of
                    // selected candidatet is higher than min_normed_score or not. If
                    // current score is better, replace worst one and update the
                    // min_normed_score.
                    if (num_beam == K)
                    {
                        if (normed_score < beam_hyps.min_normed_scores[global_batch_idx])
                        {
                            // end the tracing and exist this for loop
                            selected_beams = K;
                            break;
                        }
                        else
                        {
                            // find the beam index which's score = min_normed_score, erase it.
                            for (int j = 0; j < K; j++)
                            {
                                if (beam_hyps.normed_scores[global_batch_idx * (K * 2) + j]
                                    == beam_hyps.min_normed_scores[global_batch_idx])
                                {
                                    beam_idx = j;
                                    beam_hyps.num_beams[global_batch_idx]--;

                                    beam_hyps.min_normed_scores[global_batch_idx] = FLT_MAX;
                                    beam_hyps.normed_scores[global_batch_idx * (K * 2) + j] = normed_score;
                                    for (int l = 0; l < K; l++)
                                    {
                                        beam_hyps.min_normed_scores[global_batch_idx]
                                            = min(beam_hyps.min_normed_scores[global_batch_idx],
                                                beam_hyps.normed_scores[global_batch_idx * (K * 2) + l]);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    const int tgt_id_offset
                        = ((vector_id + beam_hyps.ite * beam_hyps.local_batch_size) * (K * 2) + beam_idx)
                        * (beam_hyps.max_seq_len);

                    int prev_id = (x[current_key] / vocab_size) % K;
                    const int current_step{sequence_lengths[vector_id * K + prev_id]};
                    beam_hyps.output_ids_tgt[tgt_id_offset + current_step] = beam_hyps.end_ids[vector_id];

                    if (beam_hyps.log_probs != nullptr)
                    {
                        beam_hyps.log_probs[tgt_id_offset + current_step]
                            = (float) y[current_key] - old_cum_log_probs[(x[current_key] / vocab_size) % K];
                    }

                    for (int j = current_step - 1; j >= 0; j--)
                    {
                        const int src_idx = j * beam_hyps.batch_size * K
                            + beam_hyps.ite * beam_hyps.local_batch_size * K + vector_id * K + prev_id;

                        beam_hyps.output_ids_tgt[tgt_id_offset + j]
                            = beam_hyps.output_ids_src_ptr[vector_id][prev_id * beam_hyps.max_seq_len + j];
                        if (beam_hyps.log_probs != nullptr && beam_hyps.log_probs_src != nullptr)
                        {
                            beam_hyps.log_probs[tgt_id_offset + j] = beam_hyps.log_probs_src[src_idx];
                        }
                        prev_id = beam_hyps.parent_ids_src_ptr[vector_id][prev_id * beam_hyps.max_seq_len + j];
                    }
                    const int tgt_beam_idx = global_batch_idx * (K * 2) + beam_idx;
                    beam_hyps.sequence_lengths_tgt[tgt_beam_idx] = current_step;
                    beam_hyps.normed_scores[tgt_beam_idx] = normed_score;
                    beam_hyps.min_normed_scores[global_batch_idx]
                        = min(beam_hyps.min_normed_scores[global_batch_idx], beam_hyps.normed_scores[tgt_beam_idx]);

                    beam_hyps.num_beams[global_batch_idx]++;
                    beam_hyps.cum_log_probs[tgt_beam_idx] = (float) y[current_key];
                }
            }
            else if ((beam_hyps.num_beams != nullptr && i < 2 * K) || (beam_hyps.num_beams == nullptr && i < K))
            {
                const int current_step{sequence_lengths[vector_id * K + selected_beams]};
                output_ids_ptr[vector_id][selected_beams * beam_hyps.max_seq_len + current_step] = x[current_key];
                if (output_log_probs != nullptr)
                {
                    output_log_probs[current_step * beam_hyps.batch_size * K + vector_id * K + selected_beams]
                        = (float) y[current_key] - old_cum_log_probs[(x[current_key] / vocab_size) % K];
                }
                v[selected_beams] = (float) y[current_key];
                selected_beams++;
            }
            __syncthreads();
            if (selected_beams >= K)
            {
                break;
            }
        }
    }
    if (threadIdx.x == 0 && beam_hyps.num_beams != nullptr)
    {
        if (beam_hyps.num_beams[blockIdx.x] < K)
        {
            beam_hyps.is_done[blockIdx.x] = false;
        }
        else if (beam_hyps.early_stopping)
        {
            beam_hyps.is_done[blockIdx.x] = true;
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
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template <typename T, int MAX_K>
struct TopKMD
{
    MD md;
    TopK<T, MAX_K> topk;
};

template <typename T, int MAX_K>
__device__ __forceinline__ TopKMD<T, MAX_K> reduce_topk_md_op(const TopKMD<T, MAX_K>& a, const TopKMD<T, MAX_K>& b)
{
    TopKMD<T, MAX_K> res;
    res.md = reduce_md_op(a.md, b.md);
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

template <typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beam_online_softmax_topk_kernel(const T* __restrict x,
    const T* __restrict b, const float* __restrict c, const FinishedState* __restrict finished, int* __restrict z,
    T* __restrict v, int V, int K, const int* __restrict end_ids)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    // reposition y to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<TopKMD<float, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopKMD<float, MAX_K> partial;
    bool finish = finished[vector_id].isFinished();
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    if (finish)
    {
        for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        {
            float elem = (elem_id == end_ids[vector_id / K]) ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
            // if (elem_id > THREADBLOCK_SIZE * MAX_K && (elem_id == E)) break;
        }
    }
    else
    {
        for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        {
            float elem = x[elem_id] + b[elem_id];
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

    TopKMD<float, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<float, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * K;
        v += vector_id * K;
        c += vector_id;

        // float d_total_inverse = __fdividef(1.0F, total.md.d);
        float d_total_log = logf(total.md.d);
        for (int i = 0; i < MAX_K; ++i)
        {
            // float val = __expf(total.topk.u[i] - total.md.m) * d_total_inverse;
            float val = total.topk.u[i] - total.md.m - d_total_log;
            if (i < K)
            {
                z[i] = total.topk.p[i] + vector_id * V; // faster transformer needs absolute id
                v[i] = val + c[0];
            }
        }
    }
}

template <typename T, int ITEMS_PER_THREAD, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__
    void beam_online_softmax_topk_stage1_kernel_base(const T* __restrict x, const T* __restrict b,
        const FinishedState* __restrict finished, float* __restrict t, int V, int K, const int* __restrict end_ids)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x; // batch beam index.

    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    // one will have multiple sections per V
    const int v_local = (V + gridDim.y - 1) / gridDim.y;
    const int section_start = v_local * blockIdx.y;
    int section_end = section_start + v_local;
    section_end = (section_end > V) ? V : section_end;

    // reposition x to data for the current vector
    x += vector_id * V;
#if TOPK_FP16_STORAGE == 1
    typedef cub::BlockReduce<TopKMD<__half, MAX_K2>, THREADBLOCK_SIZE> BlockReduce;
#else
    typedef cub::BlockReduce<TopKMD<T, MAX_K2>, THREADBLOCK_SIZE> BlockReduce;
#endif
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float buf_s[PACKED_TOP_KMD_SIZE]; // save intermediate result

#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, MAX_K2> partial;
#else
    TopKMD<T, MAX_K2> partial;
#endif
    bool finish = finished[vector_id].isFinished();
    for (int i = 0; i < MAX_K2; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    if (finish)
    {
#pragma unroll 1
        for (int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE)
        {
            float elem = (elem_id == end_ids[vector_id / K]) ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }
    else
    {
#pragma unroll 1
        for (int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE)
        {
            T bias = b == nullptr ? (T) 0.0f : b[elem_id]; // gpt-2 does not use bias
            T elem = x[elem_id] + bias;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, MAX_K2> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<__half, MAX_K2>);
#else
    TopKMD<T, MAX_K2> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K2>);
#endif

    if (thread_id == 0)
    {
        for (int i = 0; i < 2 * K; i++)
        {
            reinterpret_cast<int*>(buf_s)[i] = total.topk.p[i] + vector_id * V; // faster transformer needs absolute id
            buf_s[MAX_K2 + i] = total.topk.u[i];
        }
        buf_s[2 * MAX_K2] = total.md.d;
        buf_s[2 * MAX_K2 + 1] = total.md.m;
    }
    __syncthreads();
    for (int elem_id = thread_id; elem_id < PACKED_TOP_KMD_SIZE; elem_id += THREADBLOCK_SIZE)
    {
        t[blockIdx.x * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE + elem_id] = buf_s[elem_id];
    }
}

template <typename T, int ITEMS_PER_THREAD, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__ void beam_online_softmax_topk_stage1_kernel_fast(
    const T* __restrict x, const T* __restrict b, const FinishedState* __restrict finished, float* __restrict t, int V,
    int K, const int* __restrict end_ids, const int v_local)
{
    extern __shared__ char buf_smem_logprobs_[];
    T* buf_smem_logprobs = reinterpret_cast<T*>(buf_smem_logprobs_);

    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x; // batch beam index.

    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    // reposition x to data for the current vector
    x += vector_id * V;

    // one will have multiple sections per V
    const int section_start = v_local * blockIdx.y;
    int section_end = section_start + v_local;
    section_end = (section_end > V) ? V : section_end;
    const int valid_smem_length = section_end - section_start;

    bool finish = finished[vector_id].isFinished();
    MD partial_md{-MAX_T_VAL, 0.0f};

#if TOPK_FP16_STORAGE == 1
    using cub_kvp = cub::KeyValuePair<int, __half>;
    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
#else
    using cub_kvp = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
#endif

    using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;

    cub::ArgMax arg_max;
    cub_kvp partial_topk{V - 1, -MAX_T_VAL};

    if (finish)
    {
#pragma unroll 1
        for (int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE)
        {
            float elem = (elem_id == end_ids[vector_id / K]) ? MAX_T_VAL : -MAX_T_VAL;
            buf_smem_logprobs[elem_id - section_start] = elem;
            MD new_elem{elem, 1.0F};
            partial_md = reduce_md_op(partial_md, new_elem);

            const int smem_index = elem_id - section_start;
            cub_kvp new_elem_topk{smem_index, elem};
            partial_topk = arg_max(partial_topk, new_elem_topk);
            buf_smem_logprobs[smem_index] = elem;
        }
    }
    else
    {
#pragma unroll 1
        for (int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE)
        {
            T bias = b == nullptr ? (T) 0.0f : b[elem_id]; // gpt-2 does not use bias
            T elem = x[elem_id] + bias;
            MD new_elem_md{elem, 1.0F};
            partial_md = reduce_md_op(partial_md, new_elem_md);

            const int smem_index = elem_id - section_start;
            cub_kvp new_elem_topk{smem_index, elem};
            partial_topk = arg_max(partial_topk, new_elem_topk);
            buf_smem_logprobs[smem_index] = elem;
        }
    }

    __syncthreads();

    __shared__ union
    {
        typename BlockReduceMD::TempStorage md_smem;
        typename BlockReduceTopK::TempStorage topk_smem;
    } temp_storage;

    __shared__ float buf_s[PACKED_TOP_KMD_SIZE]; // save intermediate result
    __shared__ int thread_requiring_update;

    for (int i = 0; i < 2 * K; ++i)
    {
        cub_kvp total_topk = BlockReduceTopK(temp_storage.topk_smem).Reduce(partial_topk, arg_max);

        if (threadIdx.x == 0)
        {
            reinterpret_cast<int*>(buf_s)[i]
                = section_start + total_topk.key + vector_id * V; // faster transformer needs absolute id
            buf_s[MAX_K2 + i] = total_topk.value;
            buf_smem_logprobs[total_topk.key] = -MAX_T_VAL;
            thread_requiring_update = total_topk.key % THREADBLOCK_SIZE;
        }
        __syncthreads();

        // Only 1 thread needs to update the old partial before the next block reduce. We don't need to do this update
        // on the last iteration.
        if (thread_id == thread_requiring_update && i < (2 * K - 1))
        {
            partial_topk.key = V - 1;
            partial_topk.value = -MAX_T_VAL;
            for (int tid = thread_id; tid < valid_smem_length; tid += THREADBLOCK_SIZE)
            {
                cub_kvp new_elem{tid, buf_smem_logprobs[tid]};
                partial_topk = arg_max(partial_topk, new_elem);
            }
        }
    }

    auto reduce_md_func = [](const MD& a, const MD& b) { return reduce_md_op(a, b); };
    MD total_md = BlockReduceMD(temp_storage.md_smem).Reduce(partial_md, reduce_md_func);

    if (threadIdx.x == 0)
    {
        buf_s[2 * MAX_K2] = total_md.d;
        buf_s[2 * MAX_K2 + 1] = total_md.m;
    }
    __syncthreads();

    for (int elem_id = thread_id; elem_id < PACKED_TOP_KMD_SIZE; elem_id += THREADBLOCK_SIZE)
    {
        t[blockIdx.x * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE + elem_id] = buf_s[elem_id];
    }
}

template <typename T, int MAX_K2, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beam_online_softmax_topk_stage2_kernel(const float* __restrict x,
    const float* __restrict c, int* __restrict z, T* __restrict v, int K, int parts_per_beam, const int V)
{
    const int vector_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K2 + 2;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    extern __shared__ char buf_s_[]; // intermediate result
    float* buf_s = reinterpret_cast<float*>(buf_s_);

    using cub_kvp = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<cub_kvp, THREADBLOCK_SIZE>;
    using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;

    __shared__ union
    {
        typename BlockReduceTopK::TempStorage topk_smem;
        typename BlockReduceMD::TempStorage md_smem;

    } temp_storage;

    cub::ArgMax arg_max;

    x += vector_id * PACKED_TOP_KMD_SIZE * parts_per_beam;

    MD partial_md{-MAX_T_VAL, 0.0f};
    cub_kvp total_topk{V - 1, -MAX_T_VAL};

    __shared__ cub_kvp buf_smem_kv[MAX_K2];

    // load and unpack into registers through smem
    for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * parts_per_beam; idx += THREADBLOCK_SIZE)
    {
        buf_s[idx] = x[idx];
    }
    __syncthreads();

    // find the argmax within each parts_per_beam,
    // find the topK across all parts_per_beam.

    for (int k = 0; k < 2 * K; ++k)
    {
        cub_kvp partial_topk{V - 1, -MAX_T_VAL};
        // Only threads responsible for a chunk will do the computation
        if (threadIdx.x < parts_per_beam)
        {
            float* b_s = buf_s + threadIdx.x * PACKED_TOP_KMD_SIZE;

            for (int i = 0; i < K; ++i)
            {
                int current_index = threadIdx.x * PACKED_TOP_KMD_SIZE + i;
                T current_value = b_s[MAX_K2 + i];
                cub_kvp new_elem = {current_index, current_value};
                partial_topk = arg_max(partial_topk, new_elem);
            }
        }

        cub_kvp total_topk = BlockReduceTopK(temp_storage.topk_smem).Reduce(partial_topk, arg_max);

        __syncthreads();

        if (threadIdx.x == 0)
        {
            // store kv pairs in shared mem buffer
            int temp_offset = total_topk.key;
            int global_offset = reinterpret_cast<int*>(buf_s)[temp_offset];
            total_topk.key = global_offset;
            buf_smem_kv[k] = total_topk;

            // Invalidate the maximum value within the chunk
            reinterpret_cast<int*>(buf_s)[temp_offset] = V - 1; // id in share memory
            buf_s[temp_offset + MAX_K2] = -MAX_T_VAL;           // value in share memory
        }
        __syncthreads();
    }

    // Extract and reduce MD values across the chunks
    if (threadIdx.x < parts_per_beam)
    {
        float* b_s = buf_s + threadIdx.x * PACKED_TOP_KMD_SIZE;
        partial_md.d = b_s[2 * MAX_K2];
        partial_md.m = b_s[2 * MAX_K2 + 1];
    }

    auto reduce_md_func = [](const MD& a, const MD& b) { return reduce_md_op(a, b); };
    MD total_md = BlockReduceMD(temp_storage.md_smem).Reduce(partial_md, reduce_md_func);

    __syncthreads();

    if (thread_id == 0)
    {
        z += vector_id * 2 * K;
        v += vector_id * 2 * K;
        c += vector_id;

        float d_total_log = logf(total_md.d);

        for (int i = 0; i < MAX_K2; ++i)
        {
            float val = (float) buf_smem_kv[i].value - total_md.m - d_total_log;
            if (i < 2 * K)
            {
                z[i] = buf_smem_kv[i].key;
                v[i] = (float) val + (float) c[0];
            }
        }
    }
}

template <typename T, int MAX_K2>
void beam_online_softmax_topk_stage2_kernelLauncher(const float* temp_storage, const float* cum_log_probs, int* ids,
    T* vals, int batch_size, int beam_width, int parts_per_beam, cudaStream_t stream, const int vocab_size)
{
    // might rewrite beam_online_softmax_topk_stage2_kernel no to depend on
    // constant block size in oreder to reduce compilation time
    int smem_stage2_size = parts_per_beam * (2 * MAX_K2 + 2) * sizeof(float);

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
void topK_softMax_kernelLauncher(const T* log_probs, const T* bias, const FinishedState* finished,
    const int* sequence_lengths, float* cum_log_probs, float* output_log_probs, int** output_ids_ptr,
    void* temp_storage, const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size,
    const int beam_width, const int vocab_size, const int* end_ids, const float* diversity_rates,
    const float* length_penalties, cudaStream_t stream)
{
    const int items_per_thread = 1;
    const int block_sz = (MAX_K < 16) ? (MAX_K < 8) ? SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE : 128 : 64;
    // const int block_sz = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

    assert(temp_storage_size % 2 == 0);
    assert(temp_storage_size >= 2 * batch_size * beam_width * beam_width * 2);
    // Beam search needs the sequence lengths of beams to apply length penalty.
    assert(length_penalties == nullptr || sequence_lengths != nullptr);

    const int topk_buf_offset = ceil(batch_size * beam_width * beam_width * 2 / 4.) * 4;
    int* topk_tmp_id_buf = reinterpret_cast<int*>(temp_storage);
    T* topk_tmp_val_buf = reinterpret_cast<T*>(topk_tmp_id_buf + topk_buf_offset);
    float* tmp_buffer = reinterpret_cast<float*>(topk_tmp_val_buf + topk_buf_offset);

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

    const int constant_smem = attr.sharedSizeBytes;
    const int max_dyn_smem_per_block = max_smem_per_block - constant_smem;
    constexpr int max_parts = 128;
    TLLM_CHECK_WITH_INFO(vocab_size * sizeof(T) <= max_dyn_smem_per_block * max_parts,
        "Vocab size too large for split-k top-k beam search fast path.");

    const int driver_smem_per_block = max_smem_per_sm - max_smem_per_block;
    const int extra_smem = driver_smem_per_block + constant_smem;

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
        const int voc_size_chunk = dyn_smem_size / sizeof(T);

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
        // use original stage 1 base kernel
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
    beam_online_softmax_topk_stage2_kernelLauncher<T, 2 * MAX_K>(tmp_buffer, cum_log_probs, topk_tmp_id_buf,
        topk_tmp_val_buf, batch_size, beam_width, voc_parts, stream, vocab_size);
    sync_check_cuda_error();
#else
    beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
        <<<batch_size * beam_width, block_sz, 0, stream>>>(log_probs, bias, cum_log_probs, finished, topk_tmp_id_buf,
            topk_tmp_val_buf, vocab_size, beam_width, end_ids);
#endif

    // We need 2*MAX_K candidates because at most k candidates are finished, and
    // we will not put them into next iteration

    const int candidates = beam_width * beam_width * 2;
    int smem_size_batch_topk = sizeof(T) * candidates;
    if (smem_size_batch_topk >= (48 << 10))
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            batch_topk_kernel<T, MAX_K * 2, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_batch_topk));
    }

    batch_topk_kernel<T, MAX_K * 2, 32><<<batch_size, 32, smem_size_batch_topk, stream>>>(topk_tmp_id_buf,
        topk_tmp_val_buf, output_ids_ptr, cum_log_probs, output_log_probs, finished, sequence_lengths, *beam_hyps,
        candidates, beam_width, vocab_size, length_penalties, diversity_rates);
    sync_check_cuda_error();
}

#define INSTANTIATE_BEAMSEARCH_K(T, MAX_K)                                                                             \
    template void topK_softMax_kernelLauncher<T, MAX_K>(const T* log_probs, const T* bias,                             \
        const FinishedState* finished, const int* sequence_lengths, float* cum_log_probs, float* output_log_probs,     \
        int** output_ids_ptr, void* temp_storage, const int temp_storage_size, BeamHypotheses* beam_hyps,              \
        const int batch_size, const int beam_width, const int vocab_size, const int* end_ids,                          \
        const float* diversity_rates, const float* length_penalties, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
