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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
{
size_t invokeScanSpecDecodingGenerationLenghtsTempStorageBytes(int batch_size, cudaStream_t stream);
size_t invokeReduceMaxSpecDecodingGenerationLengthsTempStorageBytes(int batch_size, cudaStream_t stream);

// inclusive prefix sum spec_decoding_generation_lengths
void invokeScanSpecDecodingGenerationLenghths(int batch_size, int* __restrict__ const spec_decoding_generation_lengths,
    void* __restrict__ scan_temp_storage, size_t scan_temp_storage_bytes,
    int* __restrict__ scaned_spec_decoding_generation_lengths, void* __restrict__ reduce_max_temp_storage,
    size_t reduce_max_temp_storage_bytes, int* max_spec_decoding_generation_lengths, cudaStream_t stream);

void invokeConvertSpecDecodingMaskToPackedMask(int batch_size,
    int const* __restrict__ spec_decoding_cum_generation_lengths, bool const* __restrict__ spec_decoding_mask,
    int max_draft_tokens, int max_generation_length, int* __restrict__ spec_decoding_packed_mask, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
