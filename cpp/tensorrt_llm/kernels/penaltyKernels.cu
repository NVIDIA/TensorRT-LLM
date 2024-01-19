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

#include <assert.h>
#include <float.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/penaltyKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void batchApplyPenalty(T* logits, const T* biases, int* penalty_workspace, const int* penalty_workspace_prev,
    const float* temperatures, const float* repetition_penalties, const float* presence_penalties,
    const float* frequency_penalties, const bool accumulate_vocab, const int max_seq_len, const int vocab_size,
    const int vocab_size_padded, const int** output_ids_ptr, const int** parent_ids_ptr, const int* input_lengths,
    const int* sequence_lengths, const int* min_lengths, const int* end_ids)
{
    const int beam_width = gridDim.y;
    const int batch_id = blockIdx.x;
    const int beam_id = blockIdx.y;
    const int bbid = batch_id * beam_width + beam_id;
    const int input_len = input_lengths == nullptr ? 0 : input_lengths[bbid];
    const int current_step = sequence_lengths == nullptr ? 0 : sequence_lengths[bbid];
    // Initialize or update the number of occurrences of tokens
    if (accumulate_vocab)
    {
        penalty_workspace += bbid * vocab_size;
        if (current_step <= input_len)
        { // Context phase
            for (int index = threadIdx.x; index < vocab_size; index += blockDim.x)
            {
                penalty_workspace[index] = 0;
            }
            __syncthreads();
            for (int step = threadIdx.x; step < input_len; step += blockDim.x)
            {
                // All beams in the context phase are identical
                int penalty_index = output_ids_ptr[batch_id][beam_id * max_seq_len + step];
                if (penalty_index < vocab_size)
                {
                    atomicAdd(&penalty_workspace[penalty_index], 1);
                }
            }
        }
        else
        { // Generation phase
            if (beam_width > 1)
            {
                int parent_beam = parent_ids_ptr[batch_id][beam_id * max_seq_len + current_step - 2];
                penalty_workspace_prev += (batch_id * beam_width + parent_beam) * vocab_size;
                for (int index = threadIdx.x; index < vocab_size; index += blockDim.x)
                {
                    penalty_workspace[index] = penalty_workspace_prev[index];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                int penalty_index = output_ids_ptr[batch_id][beam_id * max_seq_len + current_step - 1];
                if (penalty_index < vocab_size)
                {
                    penalty_workspace[penalty_index] += 1;
                }
            }
        }
        __syncthreads();
    }
    // Apply bias and penalties
    logits += bbid * vocab_size_padded;
    const T MASK_VAL = (std::is_same<T, half>::value) ? -HALF_FLT_MAX : -FLT_MAX;
    float inv_temperature, repetition_penalty, presence_penalty, frequency_penalty;
    if (temperatures != nullptr)
    {
        inv_temperature = 1.0f / (temperatures[batch_id] + 1e-6f);
    }
    if (repetition_penalties != nullptr)
    {
        repetition_penalty = repetition_penalties[batch_id];
    }
    if (presence_penalties != nullptr)
    {
        presence_penalty = presence_penalties[batch_id];
    }
    if (frequency_penalties != nullptr)
    {
        frequency_penalty = frequency_penalties[batch_id];
    }
    for (int index = threadIdx.x; index < vocab_size_padded; index += blockDim.x)
    {
        if (index < vocab_size)
        {
            float logit = (float) logits[index];
            // Bias
            if (biases != nullptr)
            {
                logit += (float) biases[index];
            }
            // Temperature
            if (temperatures != nullptr)
            {
                logit *= inv_temperature;
            }
            int num_occurences = penalty_workspace[index];
            if (num_occurences > 0)
            {
                // Repetition
                if (repetition_penalties != nullptr)
                {
                    logit = logit < 0.0f ? logit * repetition_penalty : logit / repetition_penalty;
                }
                // Presence
                if (presence_penalties != nullptr)
                {
                    logit -= presence_penalty;
                }
                // Frequency
                if (frequency_penalties != nullptr)
                {
                    logit -= frequency_penalty * num_occurences;
                }
            }
            logits[index] = logit;
        }
        else
        {
            logits[index] = MASK_VAL;
        }
    }
    if (min_lengths != nullptr)
    {
        __syncthreads();
        // Min length
        if ((threadIdx.x == 0) && (current_step - input_len < min_lengths[batch_id]))
        {
            logits[end_ids[batch_id]] = MASK_VAL;
        }
    }
}

template <typename T>
void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<T>& params)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    dim3 block(256);
    dim3 grid(params.batch_size, params.beam_width);
    batchApplyPenalty<T><<<grid, block, 0, params.stream>>>(params.logits, params.biases, params.penalty_workspace,
        params.penalty_workspace_prev, params.temperatures, params.repetition_penalties, params.presence_penalties,
        params.frequency_penalties, params.accumulate_vocab, params.max_seq_len, params.vocab_size,
        params.vocab_size_padded, params.output_ids_ptr, params.parent_ids_ptr, params.input_lengths,
        params.sequence_lengths, params.min_lengths, params.end_ids);
}

template void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<float>& params);

template void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<half>& params);

} // namespace kernels
} // namespace tensorrt_llm
