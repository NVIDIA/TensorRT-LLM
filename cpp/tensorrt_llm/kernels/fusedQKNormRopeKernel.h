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

#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

// Perform fused QK Normalization and RoPE in a single CUDA kernel
// This function efficiently applies RMS normalization and RoPE embeddings to query and key tensors
void launchFusedQKNormRope(
    void* qkv,               // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int const num_tokens,    // Number of tokens
    int const num_heads_q,   // Number of query heads
    int const num_heads_k,   // Number of key heads
    int const num_heads_v,   // Number of value heads
    int const head_dim,      // Dimension per head
    float const eps,         // Epsilon for RMS normalization
    void const* q_weight,    // RMSNorm weights for query [head_dim]
    void const* k_weight,    // RMSNorm weights for key [head_dim]
    float const base,        // Base for RoPE computation
    bool const interleave,   // Whether RoPE is applied in interleave mode (non-Neox style)
    int const* position_ids, // Position IDs for RoPE [num_tokens]
    float factor, // factor in rope_scaling in config.json. When it is not 1.0, it means the model is using yarn.
    float low,    // threshold for high frequency
    float high,   // threshold for low frequency
    float attention_factor, // attention_factor applied on cos and sin
    cudaStream_t stream);   // CUDA stream

} // namespace kernels
} // namespace tensorrt_llm
