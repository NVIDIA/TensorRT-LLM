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
void invokeBanBadWords(T* logits, int32_t const** output_ids_ptr, int32_t const** parent_ids_ptr,
    int32_t const* batch_slot, int32_t batch_size, int32_t beam_width, int32_t const** bad_words,
    int32_t const* bad_words_len, int32_t max_bad_words_len, int32_t vocab_size_padded, int32_t const* sequence_lengths,
    int32_t max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
