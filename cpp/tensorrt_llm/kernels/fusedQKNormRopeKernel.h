/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

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
    int const rotary_dim,    // Dimension for RoPE
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
    cudaStream_t stream,    // CUDA stream
    bool is_qk_norm,        // Whether to apply QK norm
    bool use_gemma,         // Whether QK norm uses Gemma-style RMSNorm (scale by (1 + weight))
    bool use_mrope,         // Whether to use interleaved mRoPE position selection
    int mrope_section1,     // mrope_section[1] (height)
    int mrope_section2);    // mrope_section[2] (width)

// Out-of-place FP8 variant of launchFusedQKNormRope, folding the FP8
// activation-quant into the norm+RoPE epilogue. Q and K get RMSNorm + RoPE; V is
// copy-cast only.
void launchFusedQKNormRopeToFp8(void const* qkv_in, // BF16 input [num_tokens, total_heads*head_dim]
    void* qkv_out,                                  // FP8 E4M3 output buffer, same layout as input
    int const num_tokens, int const num_heads_q, int const num_heads_k, int const num_heads_v, int const head_dim,
    int const rotary_dim, float const eps, void const* q_weight, void const* k_weight, float const base,
    bool const interleave, int const* position_ids, float factor, float low, float high, float attention_factor,
    cudaStream_t stream, bool is_qk_norm, bool use_gemma, bool use_mrope, int mrope_section1, int mrope_section2);

// MiniMax-M3-specific main-branch producer. It returns contiguous
// FP8 Q and inserts normalized/RoPE'd FP8 K plus copy-cast FP8 V directly into
// a paged HND pool [num_pages, 2, num_heads, page_size, head_dim].
void launchMinimaxM3Fp8QKNormRopeKVInsert(void const* qkv_input, void* q_output, void* kv_cache,
    int const* out_cache_loc, int64_t page_stride, int64_t plane_stride, int64_t head_stride, int64_t token_stride,
    int64_t num_pages, int page_size, int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim,
    int rotary_dim, float eps, void const* q_weight, void const* k_weight, float base, int const* position_ids,
    cudaStream_t stream);

// MiniMax-M3 sparse producer for the packed [Q|K|V|index-Q|index-K]
// projection. It uses a precomputed FP32 RoPE table, emits compact FP8 Q and
// index-Q, and inserts main K/V plus index-K into their paged FP8 HND caches.
void launchMinimaxM3Fp8QKVIndexerNormRopeKVInsert(void const* packed_input, void* q_output, void* index_q_output,
    void* kv_cache, void* index_k_cache, int const* out_cache_loc, int64_t kv_page_stride, int64_t kv_plane_stride,
    int64_t kv_head_stride, int64_t kv_token_stride, int64_t index_page_stride, int64_t index_token_stride,
    int64_t num_pages, int page_size, int num_tokens, int num_heads_q, int num_heads_kv, int num_heads_index,
    int head_dim, int rotary_dim, float eps, void const* q_weight, void const* k_weight, void const* index_q_weight,
    void const* index_k_weight, float const* rotary_cos_sin, int const* position_ids, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
