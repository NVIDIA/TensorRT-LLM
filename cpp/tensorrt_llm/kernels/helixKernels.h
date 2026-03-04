/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
template <typename T>
struct HelixPostProcParams
{
    T* output;
    T const* gathered_o;
    float2 const* gathered_stats;
    int cp_size;
    int num_tokens;
    int num_heads;
    int kv_lora_rank;
};

template <typename T>
void helixPostProcess(HelixPostProcParams<T> const& params, cudaStream_t stream);

// Version 1: cp_dim=2 layout.
// gathered_o: [num_tokens, num_heads, cp_size, kv_lora_rank].
// gathered_stats: [num_tokens, num_heads, cp_size, 2].
template <typename T>
void helixPostProcessNativeV1(HelixPostProcParams<T> const& params, cudaStream_t stream);

// Version 2: cp_dim=1 layout.
// gathered_o: [num_tokens, cp_size, num_heads, kv_lora_rank].
// gathered_stats: [num_tokens, cp_size, num_heads, 2].
template <typename T>
void helixPostProcessNativeV2(HelixPostProcParams<T> const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
