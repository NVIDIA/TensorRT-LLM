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

#include <algorithm> // all_of
#include <assert.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/beamSearchPenaltyKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void add_bias_temperature(T* logits, const T* bias, const int batch_size, const int beam_width,
    const int vocab_size, const int vocab_size_padded, const float* temperatures)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bbid = blockIdx.y;

    logits += bbid * vocab_size_padded;
    const float temperature{temperatures[bbid / beam_width]};

    const T MASK_VAL = (std::is_same<T, half>::value) ? -HALF_FLT_MAX : -FLT_MAX;
    const T inv_temp = static_cast<T>(1.0f / (temperature + 1e-6f));
    for (int i = tid + bid * blockDim.x; i < vocab_size_padded; i += blockDim.x * gridDim.x)
    {
        if (i < vocab_size)
        {
            T bias_val = bias == nullptr ? (T) (0.0f) : bias[i];
            logits[i] = (logits[i] + bias_val) * inv_temp;
        }
        else
        {
            logits[i] = MASK_VAL;
        }
    }
}

template <>
__global__ void add_bias_temperature(half2* logits, const half2* bias, const int batch_size, const int beam_width,
    const int vocab_size, const int vocab_size_padded, const float* temperatures)
{
    assert(vocab_size % 2 == 0);
    assert(vocab_size_padded % 2 == 0);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int bbid = blockIdx.y;
    const float temperature{temperatures[bbid / beam_width]};

    const half2 mask_val = __float2half2_rn(-HALF_FLT_MAX);
    const half2 inv_temp = __float2half2_rn(1.0f / (temperature + 1e-6f));

    const int half_vocab_size = vocab_size / 2;
    const int half_vocab_size_padded = vocab_size_padded / 2;

    logits += bbid * half_vocab_size_padded;
    for (int index = tid + bid * blockDim.x; index < half_vocab_size_padded; index += blockDim.x * gridDim.x)
    {
        int vocab_idx = index % half_vocab_size_padded;
        half2 logit = vocab_idx < half_vocab_size ? __ldg(&logits[index]) : mask_val;
        if (vocab_idx < half_vocab_size)
        {
            if (bias != nullptr)
            {
                logit = __hadd2(logit, bias[vocab_idx]);
            }
            logit = __hmul2(logit, inv_temp);
        }
        logits[index] = logit;
    }
}

template <typename T, bool IS_ADDITIVE>
__global__ void apply_repetition_penalty(T* logits, const int batch_size, const int beam_width, const int vocab_size,
    const int vocab_size_padded, const int** output_ids_ptr, const int** parent_ids_ptr, const int* input_lengths,
    const int* sequence_lengths, const float* repetition_penalties, int max_seq_len)
{
    const int tid = threadIdx.x;
    const int bbid = blockIdx.x;
    const int batch_id = bbid / beam_width;
    const int beam_idx{bbid % beam_width};
    const float repetition_penalty{repetition_penalties[batch_id]};

    logits += bbid * vocab_size_padded;
    extern __shared__ char sbuf[];
    T* penalty_logits = reinterpret_cast<T*>(sbuf);
    // prevent misaligment when sizeof(T) = 2
    int* penalty_indices = reinterpret_cast<int*>(sbuf + (sizeof(T) * max_seq_len + 31) / 32 * 32);
    const int current_step{sequence_lengths[bbid]};
    if (tid == 0)
    {
        T repet_penalty = static_cast<T>(repetition_penalty);
        int prev_id = output_ids_ptr[batch_id][beam_idx * max_seq_len + current_step - 1];
        T prev_logit = logits[prev_id];
        penalty_indices[current_step - 1] = prev_id;

        if (IS_ADDITIVE)
        {
            penalty_logits[current_step - 1] = prev_logit - repet_penalty;
        }
        else
        {
            penalty_logits[current_step - 1]
                = prev_logit > T(0) ? prev_logit / repet_penalty : prev_logit * repet_penalty;
        }
        if (current_step > 1)
        {
            int parent_beam = bbid % beam_width;
            for (int i = current_step - 2; i >= 0; --i)
            {
                parent_beam = parent_ids_ptr[batch_id][parent_beam * max_seq_len + i];
                prev_id = output_ids_ptr[batch_id][parent_beam * max_seq_len + i];
                prev_logit = logits[prev_id];
                penalty_indices[i] = prev_id;
                if (IS_ADDITIVE)
                {
                    penalty_logits[i] = prev_logit - repet_penalty;
                }
                else
                {
                    penalty_logits[i] = prev_logit > T(0) ? prev_logit / repet_penalty : prev_logit * repet_penalty;
                }
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < current_step; i += blockDim.x)
    {
        logits[penalty_indices[i]] = penalty_logits[i];
    }
}

template <typename T>
__global__ void apply_min_length_penalty(T* logits, const int* min_lengths, const int* end_ids,
    const int* sequence_lengths, const int* input_lengths, const int beam_width, const int vocab_size_padded)
{
    int bbid = threadIdx.x + blockIdx.x * blockDim.x; // batch-beam index
    int bid = bbid / beam_width;                      // batch index
    const int min_length{min_lengths[bid]};
    auto const input_length{input_lengths == nullptr ? 0 : input_lengths[bbid]};
    // We need +1 because sequence_lengths = num_gen_tokens + input_length - 1,
    // which is equal to the length of k/v caches.
    if (sequence_lengths[bbid] + 1 - input_length < min_length)
    {
        T mask_val = (std::is_same<T, half>::value) ? -HALF_FLT_MAX : -FLT_MAX;
        logits[bbid * vocab_size_padded + end_ids[bid]] = mask_val;
    }
}

template <typename T>
void invokeAddBiasApplyPenalties(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr,
    const int* input_lengths, const int* sequence_lengths, const T* bias, const int ite, const int local_batch_size,
    const int batch_size, const int beam_width, const int vocab_size, const int vocab_size_padded, const int* end_ids,
    const float* temperatures, const std::vector<float>& h_temperatures, const float* repetition_penalties,
    const std::vector<float>& h_repetition_penalties, const RepetitionPenaltyType repetition_penalty_type,
    const int* min_lengths, const int max_seq_len, cudaStream_t stream)
{

#define ALL_OF(p_, sz_, dt_, v_) (std::all_of(p_, p_ + sz_, [&](dt_ b) { return b == v_; }))

    if (bias != nullptr
        || (temperatures != nullptr
            && !ALL_OF(std::begin(h_temperatures) + ite * local_batch_size, local_batch_size, float, 1.0f))
        || vocab_size != vocab_size_padded)
    {
        dim3 block(512);
        if (std::is_same<T, half>::value && vocab_size % 2 == 0 && vocab_size_padded % 2 == 0)
        {
            dim3 grid((vocab_size_padded / 2 + block.x - 1) / block.x, beam_width * local_batch_size);
            add_bias_temperature<<<grid, block, 0, stream>>>(reinterpret_cast<half2*>(logits),
                reinterpret_cast<const half2*>(bias), batch_size, beam_width, vocab_size, vocab_size_padded,
                temperatures);
        }
        else
        {
            dim3 grid((vocab_size_padded + block.x - 1) / block.x, beam_width * local_batch_size);
            add_bias_temperature<<<grid, block, 0, stream>>>(
                logits, bias, batch_size, beam_width, vocab_size, vocab_size_padded, temperatures);
        }
    }

    if (repetition_penalty_type != RepetitionPenaltyType::None)
    {
        if (repetition_penalties != nullptr)
        {
            size_t smem_size = (sizeof(T) * max_seq_len + 31) / 32 * 32 + sizeof(int) * max_seq_len;
            dim3 block(256);
            dim3 grid(beam_width * local_batch_size);
            float default_value = getDefaultPenaltyValue(repetition_penalty_type);
            if (repetition_penalty_type == RepetitionPenaltyType::Multiplicative
                && !ALL_OF(std::begin(h_repetition_penalties) + ite * local_batch_size, local_batch_size, float,
                    default_value))
            {
                apply_repetition_penalty<T, false><<<grid, block, smem_size, stream>>>(logits, batch_size, beam_width,
                    vocab_size, vocab_size_padded, output_ids_ptr, parent_ids_ptr, input_lengths, sequence_lengths,
                    repetition_penalties, max_seq_len);
                sync_check_cuda_error();
            }
            else if (repetition_penalty_type == RepetitionPenaltyType::Additive
                && !ALL_OF(std::begin(h_repetition_penalties) + ite * local_batch_size, local_batch_size, float,
                    default_value))
            {
                apply_repetition_penalty<T, true><<<grid, block, smem_size, stream>>>(logits, batch_size, beam_width,
                    vocab_size, vocab_size_padded, output_ids_ptr, parent_ids_ptr, input_lengths, sequence_lengths,
                    repetition_penalties, max_seq_len);
                sync_check_cuda_error();
            }
        }
    }

    TLLM_CHECK_WITH_INFO(sequence_lengths != nullptr, "Need sequence_lengths to apply min length penlaty");
    TLLM_CHECK_WITH_INFO(end_ids != nullptr, "Need end_id to apply min length penlaty");

    const int block_size = min(local_batch_size * beam_width, 1024);
    const int grid_size = (local_batch_size * beam_width + block_size - 1) / block_size;
    apply_min_length_penalty<<<grid_size, block_size, 0, stream>>>(
        logits, min_lengths, end_ids, sequence_lengths, input_lengths, beam_width, vocab_size_padded);
    sync_check_cuda_error();

#undef ALL_OF
}

template void invokeAddBiasApplyPenalties(float* logits, const int** output_ids_ptr, const int** parent_ids_ptr,
    const int* input_lengths, const int* sequence_lengths, const float* bias, const int ite, const int local_batch_size,
    const int batch_size, const int beam_width, const int vocab_size, const int vocab_size_padded, const int* end_ids,
    const float* temperatures, const std::vector<float>& h_temperatures, const float* repetition_penalties,
    const std::vector<float>& h_repetition_penalties, const RepetitionPenaltyType repetition_penalty_type,
    const int* min_lengths, int max_seq_len, cudaStream_t stream);

template void invokeAddBiasApplyPenalties(half* logits, const int** output_ids_ptr, const int** parent_ids_ptr,
    const int* input_lengths, const int* sequence_lengths, const half* bias, const int ite, const int local_batch_size,
    const int batch_size, const int beam_width, const int vocab_size, const int vocab_size_padded, const int* end_ids,
    const float* temperatures, const std::vector<float>& h_temperatures, const float* repetition_penalties,
    const std::vector<float>& h_repetition_penalties, const RepetitionPenaltyType repetition_penalty_type,
    const int* min_lengths, int max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
