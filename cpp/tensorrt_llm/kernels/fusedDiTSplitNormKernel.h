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

#ifndef TRTLLM_FUSEDDITSPLITNORMKERNEL_H
#define TRTLLM_FUSEDDITSPLITNORMKERNEL_H

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Fused full-dim RMSNorm (no RoPE) for a SINGLE Q or K tensor.
//
// Mirror of fusedDiTSplitNormFullDimRopeKernel but without the RoPE step --
// used by LTX-2 cross-attn paths where Q/K need norm but no RoPE (e.g. text
// cross-attention where positional info is already baked into the text
// encoder output).
//
// Layout: input is a contiguous 2D tensor [num_tokens, num_heads * head_dim]
// (e.g. output of self.to_q / self.to_k). Block=256 with chunked reduce; cross
// -warp shared-memory sum^2 reduction over the full inner dim
// (num_heads * head_dim).
//
// Constraints:
//   - num_heads <= 32
//   - head_dim ∈ {64, 128}
//   - num_heads * head_dim <= 4096 (chunk count cap)

void launchFusedDiTSplitNormFullDim(void* tensor, // [num_tokens, num_heads * head_dim], bf16, contiguous, in-place
    int num_tokens, int num_heads,
    int head_dim,                                 // 64 or 128
    float eps,
    void const* weight,                           // bf16, [num_heads * head_dim] (full-dim norm weight)
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_FUSEDDITSPLITNORMKERNEL_H
