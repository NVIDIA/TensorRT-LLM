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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// MiniMax-M3-specific index-branch producer. It applies Gemma RMSNorm and
// NeoX partial RoPE to a packed BF16 [index-Q | index-K] projection, writes
// index-Q as unscaled E4M3, and inserts index-K directly into the paged E4M3
// HND cache. The direct cache store removes the standalone cast/scatter launch
// from the decode graph.
void launchMinimaxM3Fp8IndexerQKNormRope(void const* qk, void* q_out, void* k_cache, int const* out_cache_loc,
    int64_t page_stride, int64_t token_stride, int page_size, int num_tokens, int num_heads_q, int head_dim,
    int rotary_dim, float eps, void const* q_weight, void const* k_weight, float base, int const* position_ids,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
