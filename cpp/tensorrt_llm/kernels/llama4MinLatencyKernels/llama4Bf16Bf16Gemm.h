/*
 * Copyright (c) 2025-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm::kernels::llama4_min_latency::llama4_bf16_bf16_gemm
{

void llama4_bf16_bf16_gemm_op(int num_tokens, void const* A, void const* B, void* C, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::llama4_min_latency::llama4_bf16_bf16_gemm
