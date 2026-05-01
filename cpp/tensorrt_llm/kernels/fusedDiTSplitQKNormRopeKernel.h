/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TRTLLM_FUSEDDITSPLITQKNORMROPEKERNEL_H
#define TRTLLM_FUSEDDITSPLITQKNORMROPEKERNEL_H

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Fused RMSNorm + RoPE for a SINGLE Q or K tensor (DiT models, e.g. LTX-2).
//
// Designed for SEPARATE_QKV layout: input is a contiguous 2D tensor
// [num_tokens, num_heads * head_dim] (e.g. output of self.to_q / self.to_k).
// One block per token, one warp per head (block_size = num_heads * 32, max 1024).
//
// For FUSE_QKV layout (packed QKV buffer), use fusedDiTQKNormRopeKernel
// (per-head or full-dim variant) instead — it processes Q+K in one launch.
//
// Constraints (Phase 1):
//   - full_dim_norm = true (LTX-2 only mode)
//   - do_norm = true (norm + RoPE; do_norm=false will be added in Phase 2 for K-skip-norm)
//   - num_heads ≤ 32 (so block_size ≤ 1024)
//   - head_dim ∈ {64, 128}

void launchFusedDiTSplitNormRope(void* tensor, // [num_tokens, num_heads * head_dim], bf16, contiguous, in-place
    int num_tokens, int num_heads,
    int head_dim,                              // Must be 64 or 128
    float eps,
    void const* weight,                        // bf16, [num_heads * head_dim] when full_dim_norm=true
    float const* cos_emb,                      // float32
    float const* sin_emb,                      // float32
    bool full_dim_norm, bool do_norm,
    bool interleave,                           // true = pair (2i, 2i+1); false = rotate_half
    bool per_head_cos,                         // false: cos shape [N, head_dim]; true: [N, num_heads*head_dim]
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_FUSEDDITSPLITQKNORMROPEKERNEL_H
