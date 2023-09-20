/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

void invokeStopWordsCriterion(const int** output_ids, const int** parent_ids, const int* stop_words, bool* finished,
    const int* sequence_lengths, size_t id_offset, size_t stop_words_len, int batch_size, int beam_width,
    int max_seq_len, cudaStream_t stream);

void invokeLengthCriterion(bool* finished, int* finished_sum, const uint32_t* sequence_limit_length,
    const int* sequence_lengths, int batch_size, int beam_width, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
