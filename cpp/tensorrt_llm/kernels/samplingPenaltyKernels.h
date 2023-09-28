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
#pragma once

#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeBatchApplyRepetitionPenalty(T* logits, const float* penalties, const int** output_ids,
    const int* sequence_lengths, const int batch_size, const int local_batch_size, const int vocab_size,
    const int* input_lengths, const RepetitionPenaltyType penalty_type, int max_seq_len, cudaStream_t stream);

template <typename T>
void invokeApplyTemperaturePenalty(T* logits, const T* bias, const float temperature, const int batch_size,
    const int vocab_size, const int vocab_size_padd, cudaStream_t stream);

template <typename T>
void invokeBatchApplyTemperaturePenalty(T* logits, const T* bias, const float* temperatures, const int batch_size,
    const int vocab_size, const int vocab_size_padd, cudaStream_t stream);

template <typename T>
void invokeMinLengthPenalty(T* logits, const int* min_lengths, const int* end_ids, const int* sequnece_lengths,
    const int* input_lengths, const int batch_size, const int vocab_size_padded, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
