/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TRTLLM_FUSEDDITQKNORMROPEKERNEL_H
#define TRTLLM_FUSEDDITQKNORMROPEKERNEL_H

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Fused QK Normalization + RoPE for Diffusion Transformers (DiT).
//
// Unlike the LLM variant (launchFusedQKNormRope) which computes cos/sin from
// position_ids + theta, this kernel takes precomputed cos/sin embeddings and
// supports dual-stream attention (separate norm weights for text vs image tokens).
//
// Operates in-place on the packed QKV tensor. Only Q and K are modified;
// V is left untouched.
void launchFusedDiTQKNormRope(void* qkv, // [num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim], in-place
    int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v,
    int head_dim,                        // Must be 64, 128, or 256
    float eps,                           // Epsilon for RMS normalization
    void const* q_weight,                // RMSNorm weight for query  [head_dim]
    void const* k_weight,                // RMSNorm weight for key    [head_dim]
    void const* q_add_weight,            // RMSNorm weight for text-stream query  [head_dim], or nullptr
    void const* k_add_weight,            // RMSNorm weight for text-stream key    [head_dim], or nullptr
    float const* cos_emb,                // Precomputed cos embeddings [num_tokens, head_dim], float32
    float const* sin_emb,                // Precomputed sin embeddings [num_tokens, head_dim], float32
    int num_txt_tokens,                  // Text token boundary; tokens [0, num_txt_tokens) use add_weights.
                                         // Set to -1 to disable dual-stream (all tokens use primary weights).
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_FUSEDDITQKNORMROPEKERNEL_H
