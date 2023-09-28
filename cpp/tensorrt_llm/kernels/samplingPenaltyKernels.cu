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

#include "tensorrt_llm/kernels/samplingPenaltyKernels.h"

namespace tensorrt_llm
{
namespace kernels
{

// TODO Add half2 implementation
template <typename T>
__global__ void applyTemperaturePenalty(T* logits, const T* bias, const float temperature_inverse, const int m,
    const int vocab_size, const int vocab_size_padd)
{
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < m * vocab_size_padd;
         index += blockDim.x * gridDim.x)
    {
        T bias_val = bias == nullptr ? (T) (0.0f) : bias[index % vocab_size_padd];
        if (index % vocab_size_padd < vocab_size)
        {
            logits[index] = (logits[index] + bias_val) * (T) temperature_inverse;
        }
        else
        {
            logits[index] = -MAX_T_VAL;
        }
    }
}

template <>
__global__ void applyTemperaturePenalty(half2* logits, const half2* bias, const float temperature_inverse,
    const int batch_size, const int vocab_size, const int vocab_size_padded)
{
    assert(vocab_size % 2 == 0);
    assert(vocab_size_padded % 2 == 0);
    const half2 mask_val = __float2half2_rn(-65504.0f);
    const half2 temp_inv = __float2half2_rn(temperature_inverse);

    const int half_vocab_size = vocab_size / 2;
    const int half_vocab_size_padded = vocab_size_padded / 2;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * half_vocab_size_padded;
         index += blockDim.x * gridDim.x)
    {
        int vocab_idx = index % half_vocab_size_padded;
        half2 logit = vocab_idx < half_vocab_size ? __ldg(&logits[index]) : mask_val;
        if (vocab_idx < half_vocab_size)
        {
            if (bias != nullptr)
            {
                logit = __hadd2(logit, bias[vocab_idx]);
            }
            logits[index] = __hmul2(logit, temp_inv);
        }
    }
}

template <typename T>
void invokeApplyTemperaturePenalty(T* logits, const T* bias, const float temperature, const int batch_size,
    const int vocab_size, const int vocab_size_padd, cudaStream_t stream)
{
    dim3 block(min(vocab_size_padd, 1024));
    dim3 grid(min(batch_size * vocab_size_padd / block.x, 65536));
    const T temperature_inverse = (T) (1.f / (temperature + 1e-6f));
    if (std::is_same<T, half>::value && vocab_size % 2 == 0 && vocab_size_padd % 2 == 0)
    {
        applyTemperaturePenalty<<<grid, block, 0, stream>>>(reinterpret_cast<half2*>(logits),
            reinterpret_cast<const half2*>(bias), temperature_inverse, batch_size, vocab_size, vocab_size_padd);
    }
    else
    {
        applyTemperaturePenalty<T>
            <<<grid, block, 0, stream>>>(logits, bias, temperature_inverse, batch_size, vocab_size, vocab_size_padd);
    }
}

template void invokeApplyTemperaturePenalty(float* logits, const float* bias, const float temperature,
    const int batch_size, const int vocab_size, const int vocab_size_padd, cudaStream_t stream);

template void invokeApplyTemperaturePenalty(half* logits, const half* bias, const float temperature,
    const int batch_size, const int vocab_size, const int vocab_size_padd, cudaStream_t stream);

template <typename T>
__global__ void batchApplyTemperaturePenalty(T* logits, const T* bias, const float* temperatures, const int batch_size,
    const int vocab_size, const int vocab_size_padd)
{
    // TODO: Add macro or device function to get MAX_T_VAL.
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;
    extern __shared__ float inv_temperatures[];
    if (threadIdx.x < batch_size)
    {
        inv_temperatures[threadIdx.x] = 1.0f / (temperatures[threadIdx.x] + 1e-6f);
    }
    __syncthreads();

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * vocab_size_padd;
         index += blockDim.x * gridDim.x)
    {
        int batch_idx = index / vocab_size_padd;
        int vocab_idx = index % vocab_size_padd;
        T logit = (vocab_idx < vocab_size) ? logits[index] : -MAX_T_VAL;
        if (vocab_idx < vocab_size)
        {
            if (bias != nullptr)
            {
                logit += bias[vocab_idx];
            }
            logit *= inv_temperatures[batch_idx];
        }
        logits[index] = logit;
    }
}

__global__ void batchApplyTemperaturePenalty_h2(half2* logits, const half2* bias, const float* temperatures,
    const int batch_size, const int vocab_size, const int vocab_size_padded)
{
    assert(vocab_size % 2 == 0);
    assert(vocab_size_padded % 2 == 0);
    extern __shared__ half2 h2_inv_temperatures[];
    if (threadIdx.x < batch_size)
    {
        h2_inv_temperatures[threadIdx.x] = __float2half2_rn(1.f / (temperatures[threadIdx.x] + 1e-6f));
    }
    __syncthreads();

    const half2 mask_val = __float2half2_rn(-65504.0f);
    const int half_vocab_size = vocab_size / 2;
    const int half_vocab_size_padded = vocab_size_padded / 2;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * half_vocab_size_padded;
         index += blockDim.x * gridDim.x)
    {
        int batch_idx = index / half_vocab_size_padded;
        int vocab_idx = index % half_vocab_size_padded;
        half2 logit = vocab_idx < half_vocab_size ? __ldg(&logits[index]) : mask_val;
        if (vocab_idx < half_vocab_size)
        {
            if (bias != nullptr)
            {
                logit = __hadd2(logit, bias[vocab_idx]);
            }
            logits[index] = __hmul2(logit, h2_inv_temperatures[batch_idx]);
        }
    }
}

template <typename T>
void invokeBatchApplyTemperaturePenalty(T* logits, const T* bias, const float* temperatures, const int batch_size,
    const int vocab_size, const int vocab_size_padd, cudaStream_t stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    dim3 block(min(vocab_size_padd, 1024));
    dim3 grid(min(batch_size * vocab_size_padd / block.x, 65536));
    if (std::is_same<T, half>::value && vocab_size % 2 == 0 && vocab_size_padd % 2 == 0)
    {
        size_t smem_size = sizeof(half2) * batch_size;
        batchApplyTemperaturePenalty_h2<<<grid, block, smem_size, stream>>>(reinterpret_cast<half2*>(logits),
            reinterpret_cast<const half2*>(bias), temperatures, batch_size, vocab_size, vocab_size_padd);
    }
    else
    {
        size_t smem_size = sizeof(float) * batch_size;
        batchApplyTemperaturePenalty<T>
            <<<grid, block, smem_size, stream>>>(logits, bias, temperatures, batch_size, vocab_size, vocab_size_padd);
    }
}

template void invokeBatchApplyTemperaturePenalty(float* logits, const float* bias, const float* temperatures,
    const int batch_size, const int vocab_size, const int vocab_size_padd, cudaStream_t stream);

template void invokeBatchApplyTemperaturePenalty(half* logits, const half* bias, const float* temperatures,
    const int batch_size, const int vocab_size, const int vocab_size_padd, cudaStream_t stream);

template <typename T, RepetitionPenaltyType penalty_type>
__global__ void applyRepetitionPenalty(T* logits, const float penalty, const int* start_ids, int* output_ids,
    const int batch_size, const int local_batch_size, const int vocab_size, const int vocab_size_padd,
    const int* input_lengths, const int step)
{
    extern __shared__ float penalty_logits[];
    int* penalty_indices = (int*) (penalty_logits + step);

    logits = logits + blockIdx.x * vocab_size_padd;
    const int input_length = input_lengths != nullptr ? input_lengths[blockIdx.x] : 0;
    for (int index = threadIdx.x; index < step; index += blockDim.x)
    {
        // output_ids shape: (batch_size, input_len + output_len)
        int penalty_index = output_ids[index * batch_size + blockIdx.x];
        if (penalty_index >= vocab_size)
        {
            continue;
        }
        penalty_indices[index] = penalty_index;
        float logit = (float) logits[penalty_index];
        if (penalty_type == RepetitionPenaltyType::Additive)
        {
            penalty_logits[index] = logit - penalty;
        }
        else if (penalty_type == RepetitionPenaltyType::Multiplicative)
        {
            penalty_logits[index] = logit < 0.0f ? logit * penalty : logit / penalty;
        }
        else if (penalty_type == RepetitionPenaltyType::None)
        {
            penalty_logits[index] = logit;
        }
        else
        {
            // Unsupported type
            assert(false);
        }
    }

    if (blockDim.x > 32)
    {
        __syncthreads();
    }

    for (int index = threadIdx.x; index < step; index += blockDim.x)
    {
        // output_ids shape: (batch_size, input_len + output_len)
        if (penalty_indices[index] >= vocab_size)
        {
            continue;
        }
        logits[penalty_indices[index]] = penalty_logits[index];
    }
}

template <typename T, RepetitionPenaltyType penalty_type>
__global__ void batchApplyRepetitionPenalty(T* logits, const float* penalties, const int** output_ids,
    const int* sequence_lengths, const int batch_size, const int vocab_size, const int* input_lengths,
    const int max_seq_len)
{
    extern __shared__ float penalty_logits[];
    int* penalty_indices = (int*) (penalty_logits + max_seq_len);
    const int batch_idx = blockIdx.x;
    const float penalty = penalties[batch_idx];
    const int current_step = sequence_lengths[batch_idx];

    logits += batch_idx * vocab_size;

    // Phase 1. Find indices to penalize and keep the penalized values.
    // A vocab id can appear multiple times but should be penalized once.
    for (int index = threadIdx.x; index < current_step; index += blockDim.x)
    {
        // output_ids shape: (batch_size, input_len + output_len)
        int penalty_index = output_ids[batch_idx][blockIdx.y * max_seq_len + index];
        assert(penalty_index < vocab_size);
        penalty_indices[index] = penalty_index;
        float logit = (float) logits[penalty_index];
        if (penalty_type == RepetitionPenaltyType::Additive)
        {
            penalty_logits[index] = logit - penalty;
        }
        else if (penalty_type == RepetitionPenaltyType::Multiplicative)
        {
            penalty_logits[index] = logit < 0.0f ? logit * penalty : logit / penalty;
        }
        else if (penalty_type == RepetitionPenaltyType::None)
        {
            penalty_logits[index] = logit;
        }
        else
        {
            // Unsupported type
            assert(false);
        }
    }

    if (blockDim.x > 32)
    {
        __syncthreads();
    }

    // Phase 2. Replace a logit value by the penalized one.
    for (int index = threadIdx.x; index < current_step; index += blockDim.x)
    {
        logits[penalty_indices[index]] = penalty_logits[index];
    }
}

template <typename T>
void invokeBatchApplyRepetitionPenalty(T* logits, const float* penalties, const int** output_ids,
    const int* sequence_lengths, const int batch_size, const int local_batch_size, const int vocab_size,
    const int* input_lengths, RepetitionPenaltyType penalty_type, int max_seq_len, cudaStream_t stream)
{
    // Inputs
    //   logits [local_batch_size, vocab_size] : logit values.
    //   penalties [local_batch_size] : repetition penalty factors.
    //   output_ids int**, [bs] array, each array has [1, max_seq_len]
    //   sequence_lengths int*, [bs]
    //   input_lengths [local_batch_size], input lengths

    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    dim3 block(min(max_seq_len, 1024));
    dim3 grid(batch_size);
    size_t smem_size = max_seq_len * (sizeof(float) + sizeof(int));
    if (penalty_type == RepetitionPenaltyType::Additive)
    {
        batchApplyRepetitionPenalty<T, RepetitionPenaltyType::Additive><<<grid, block, smem_size, stream>>>(
            logits, penalties, output_ids, sequence_lengths, batch_size, vocab_size, input_lengths, max_seq_len);
    }
    else if (penalty_type == RepetitionPenaltyType::Multiplicative)
    {
        batchApplyRepetitionPenalty<T, RepetitionPenaltyType::Multiplicative><<<grid, block, smem_size, stream>>>(
            logits, penalties, output_ids, sequence_lengths, batch_size, vocab_size, input_lengths, max_seq_len);
    }
    else if (penalty_type == RepetitionPenaltyType::None)
    {
        // do nothing
    }
}

template void invokeBatchApplyRepetitionPenalty(float* logits, const float* penalties, const int** output_ids,
    const int* sequence_lengths, const int batch_size, const int local_batch_size, const int vocab_size,
    const int* input_lengths, RepetitionPenaltyType penalty_type, int max_seq_len, cudaStream_t stream);

template void invokeBatchApplyRepetitionPenalty(half* logits, const float* penalties, const int** output_ids,
    const int* sequence_lengths, const int batch_size, const int local_batch_size, const int vocab_size,
    const int* input_lengths, RepetitionPenaltyType penalty_type, int max_seq_len, cudaStream_t stream);

template <typename T>
__global__ void batchApplyMinLengthPenalty(T* logits, const int* min_lengths, const int* end_ids,
    const int* sequence_lengths, const int* input_lengths, const int vocab_size_padded)
{
    int bid = threadIdx.x + blockIdx.x * blockDim.x; // batch index
    auto const input_length{input_lengths == nullptr ? 0 : input_lengths[bid]};
    // We need +1 because sequence_lengths = num_gen_tokens + input_length - 1, which is equal to the length of k/v
    // caches.
    if (sequence_lengths[bid] + 1 - input_length < min_lengths[bid])
    {
        T mask_val = (std::is_same<T, half>::value) ? -65504.0f : -FLT_MAX;
        logits[bid * vocab_size_padded + end_ids[bid]] = mask_val;
    }
}

template <typename T>
void invokeMinLengthPenalty(T* logits, const int* min_lengths, const int* end_ids, const int* sequnece_lengths,
    const int* input_lengths, const int batch_size, const int vocab_size_padded, cudaStream_t stream)

{
    const int block_size = min(batch_size, 1024);
    const int grid_size = (batch_size + block_size - 1) / block_size;
    batchApplyMinLengthPenalty<<<grid_size, block_size, 0, stream>>>(
        logits, min_lengths, end_ids, sequnece_lengths, input_lengths, vocab_size_padded);
}

template void invokeMinLengthPenalty(float* logits, const int* min_lengths, const int* end_ids,
    const int* sequnece_lengths, const int* input_lengths, const int batch_size, const int vocab_size_padded,
    cudaStream_t stream);

template void invokeMinLengthPenalty(half* logits, const int* min_lengths, const int* end_ids,
    const int* sequnece_lengths, const int* input_lengths, const int batch_size, const int vocab_size_padded,
    cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
