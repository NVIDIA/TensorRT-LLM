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

#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
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

template <typename T>
void helixPostProcessNative(HelixPostProcParams<T> const& params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
