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

// Fused per-head QK Normalization + RoPE for Diffusion Transformers (DiT).
//
// Per-head norm: one warp per head, warp-level shuffle reduction.
// For FLUX, Cosmos3, UniVideo.
//
// Features:
//   - Precomputed cos/sin embeddings
//   - Dual-stream attention: separate norm weights for text vs image (FLUX)
//   - Interleaved or rotate_half RoPE modes
//
// Operates in-place on the packed QKV tensor. Only Q and K are modified;
// V is left untouched.

void launchFusedDiTQKNormRope(void* qkv, // [num_tokens, (Hq+Hk+Hv)*head_dim], in-place
    int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v,
    int head_dim,                        // Must be 64, 128, or 256
    float eps,
    void const* q_weight,                // [head_dim]
    void const* k_weight,                // [head_dim]
    void const* q_add_weight,            // [head_dim] or nullptr (dual-stream text norm)
    void const* k_add_weight,            // [head_dim] or nullptr
    float const* cos_emb,                // [num_tokens, head_dim], float32
    float const* sin_emb,                // [num_tokens, head_dim], float32
    int num_txt_tokens,                  // Text token boundary; -1 = no dual-stream
    bool interleave,                     // true = interleaved pairs, false = rotate_half
    int tokens_per_batch,                // seq_len per batch element for dual-stream; 0 = flat
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_FUSEDDITQKNORMROPEKERNEL_H
