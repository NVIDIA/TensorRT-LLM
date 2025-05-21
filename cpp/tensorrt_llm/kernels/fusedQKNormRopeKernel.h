/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/assert.h"
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

// Perform fused QK Normalization and RoPE in a single CUDA kernel
// This function efficiently applies RMS normalization and RoPE embeddings to query and key tensors
void launchFusedQKNormRope(
    void* qkv,                      // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int const* position_ids,        // Position IDs for RoPE [num_tokens]
    int const num_tokens,           // Number of tokens
    int const num_heads_q,          // Number of query heads
    int const num_heads_k,          // Number of key heads
    int const num_heads_v,          // Number of value heads
    int const head_dim,             // Dimension per head
    bool const interleave,          // Whether RoPE is applied in interleave mode (non-Neox style)
    float const eps = 1e-5f,        // Epsilon for RMS normalization
    float const base = 10000.0f,    // Base for RoPE computation
    cudaStream_t stream = nullptr); // CUDA stream

} // namespace kernels
} // namespace tensorrt_llm
