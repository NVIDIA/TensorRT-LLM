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

#include "tensorrt_llm/runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeBanBadWords(T* logits, runtime::TokenIdType const** output_ids_ptr,
    runtime::SizeType32 const** parent_ids_ptr, runtime::SizeType32 const* batch_slot, runtime::SizeType32 batch_size,
    runtime::SizeType32 beam_width, runtime::TokenIdType const* const* bad_words,
    runtime::SizeType32 const* bad_words_len, runtime::SizeType32 max_bad_words_len,
    runtime::SizeType32 vocab_size_padded, runtime::SizeType32 const* sequence_lengths, runtime::SizeType32 max_seq_len,
    cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
