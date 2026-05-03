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

// Fused full-dim RMSNorm + RoPE for a SINGLE Q or K tensor (DiT models, e.g. LTX-2).
//
// Designed for SEPARATE_QKV layout: input is a contiguous 2D tensor
// [num_tokens, num_heads * head_dim] (e.g. output of self.to_q / self.to_k).
// Block=256 with chunked reduce; cross-warp shared-memory sum² reduction over
// the full inner dim (num_heads * head_dim).
//
// For FUSE_QKV layout (packed QKV buffer), use fusedDiTQKNormFullDimRopeKernel
// (full-dim variant in fusedDiTQKNormRopeKernel.h) -- it processes Q+K together
// in one launch.
//
// Constraints:
//   - num_heads ≤ 32
//   - head_dim ∈ {64, 128}
//   - num_heads * head_dim ≤ 4096 (chunk count cap)

void launchFusedDiTSplitNormFullDimRope(void* tensor, // [num_tokens, num_heads * head_dim], bf16, contiguous, in-place
    int num_tokens, int num_heads,
    int head_dim,                                     // 64 or 128
    float eps,
    void const* weight,                               // bf16, [num_heads * head_dim] (full-dim norm weight)
    float const* cos_emb,                             // float32
    float const* sin_emb,                             // float32
    bool interleave,                                  // true = pair (2i, 2i+1); false = rotate_half
    bool per_head_cos,                                // false: cos shape [N, head_dim]; true: [N, num_heads*head_dim]
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_FUSEDDITSPLITQKNORMROPEKERNEL_H
