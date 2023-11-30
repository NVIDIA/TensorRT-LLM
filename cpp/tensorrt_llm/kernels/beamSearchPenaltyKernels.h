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

#include "tensorrt_llm/kernels/penaltyTypes.h"

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeAddBiasApplyPenalties(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr,
    const int* input_lengths, const int* sequence_lengths, const T* bias, const int ite, const int local_batch_size,
    const int batch_size, const int beam_width, const int vocab_size, const int vocab_size_padded, const int* end_ids,
    const float* temperatures, const std::vector<float>& h_temperatures, const float* repetition_penalties,
    const std::vector<float>& h_repetition_penalties, const RepetitionPenaltyType repetition_penalty_type,
    const int* min_lengths, int max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
