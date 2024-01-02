/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeBanBadWords(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr, int batch_size,
    int local_batch_size, int beam_width, const int* bad_words, bool share_words, size_t bad_words_len,
    int vocab_size_padded, const int* sequence_lengths, int max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
