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
    float const* cos_emb,                // [cos_rows, head_dim], float32  (cos_rows = num_tokens or num_tokens/B)
    float const* sin_emb,                // [cos_rows, head_dim], float32
    int num_txt_tokens,                  // Text token boundary; -1 = no dual-stream
    bool interleave,                     // true = interleaved pairs, false = rotate_half
    int tokens_per_batch,                // seq_len per batch element for dual-stream; 0 = flat
    int cos_seq_per_batch,               // cos rows per batch for broadcast; 0 = no broadcast
    cudaStream_t stream);

// Out-of-place static-E4M3 variant for FLUX CUTEDSL attention. Reads packed
// BF16 QKV, applies the same per-head Q/K norm + RoPE math as
// launchFusedDiTQKNormRope, and writes three dense FP8 tensors. V is scaled
// and converted directly from the packed BF16 input without norm or RoPE.
void launchFusedDiTQKNormRopeQuantFp8(void const* qkv, // [num_tokens, (Hq+Hk+Hv)*head_dim]
    void* q_out,                                       // [num_tokens, Hq*head_dim], E4M3
    void* k_out,                                       // [num_tokens, Hk*head_dim], E4M3
    void* v_out,                                       // [num_tokens, Hv*head_dim], E4M3
    int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v,
    int head_dim,                                      // Must be 64, 128, or 256
    float eps,
    void const* q_weight,                              // [head_dim], BF16
    void const* k_weight,                              // [head_dim], BF16
    void const* q_add_weight,                          // [head_dim] or nullptr
    void const* k_add_weight,                          // [head_dim] or nullptr
    float const* q_dequant_scale,                      // [1], FP32
    float const* k_dequant_scale,                      // [1], FP32
    float const* v_dequant_scale,                      // [1], FP32
    float const* cos_emb,                              // [cos_rows, head_dim], FP32
    float const* sin_emb,                              // [cos_rows, head_dim], FP32
    int num_txt_tokens, bool interleave, int tokens_per_batch, int cos_seq_per_batch, cudaStream_t stream);

// Full-dim variant for LTX-2 / WAN: RMSNorm range = num_heads_per_side * head_dim.
// Requires num_heads_q == num_heads_k. No dual-stream support.
// per_head_cos=false: cos/sin shape [num_tokens, head_dim] (head broadcast).
// per_head_cos=true:  cos/sin shape [num_tokens, num_heads*head_dim]
//                     (LTX-2 INTERLEAVED 3D RoPE — different freqs per head).
void launchFusedDiTQKNormRopeFullDim(void* qkv, // [num_tokens, (Hq+Hk+Hv)*head_dim], in-place
    int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v,
    int head_dim,                               // Must be 64 or 128
    float eps,
    void const* q_weight,                       // [num_heads_q * head_dim]
    void const* k_weight,                       // [num_heads_k * head_dim]
    void const* cos_emb,                        // float32 or bfloat16 (selected by cos_is_bf16)
    void const* sin_emb,                        // same dtype as cos_emb
    bool interleave, bool per_head_cos,
    bool cos_is_bf16,                           // true → cos/sin are bf16
    int cos_seq_per_batch,                      // 0 = flat cos [num_tokens, …]; >0 = cos broadcast over B
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_FUSEDDITQKNORMROPEKERNEL_H
