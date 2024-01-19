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
struct InvokeBatchApplyPenaltyParams
{
    T* logits;
    const T* biases;
    int* penalty_workspace;
    const int* penalty_workspace_prev;
    const float* temperatures;
    const float* repetition_penalties;
    const float* presence_penalties;
    const float* frequency_penalties;
    const bool accumulate_vocab;
    const size_t batch_size;
    const int beam_width;
    const int max_seq_len;
    const size_t vocab_size;
    const size_t vocab_size_padded;
    const int** output_ids_ptr;
    const int** parent_ids_ptr;
    const int* input_lengths;
    const int* sequence_lengths;
    const int* min_lengths;
    const int* end_ids;
    cudaStream_t stream;
};

template <typename T>
void invokeBatchApplyPenalty(const InvokeBatchApplyPenaltyParams<T>& params);

} // namespace kernels
} // namespace tensorrt_llm
