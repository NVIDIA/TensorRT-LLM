/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/decodingCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeBanRepeatNgram(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
    const int* parent_ids_buf, int batch_size, int local_batch_size, int beam_width,
    const int* no_repeat_ngram_size_buf, int id_offset, int vocab_size_padded, size_t step, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
