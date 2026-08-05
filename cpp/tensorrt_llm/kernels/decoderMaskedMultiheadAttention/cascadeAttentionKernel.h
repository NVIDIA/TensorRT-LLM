/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace mmha
{
namespace cascade
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Cascade Attention Kernel
//
// Implements the divide-and-conquer attention from
// "Cascade Inference: Memory Bandwidth Efficient Shared Prefix Batch Decoding"
// (https://flashinfer.ai/2024/02/02/cascade-inference.html), specialized to
// beam-search decode and fused down to two kernels (the original Phase 0
// KV-write and Phase 3 merge are absorbed into the suffix-decode kernel).
//
// Phase 1 (cascade_prefix_mqa_kernel):
//     For each request, all `beam_width` queries attend to the *shared* prompt
//     KV (token range [0, L_p)). KV is loaded once per (head, request, token tile)
//     into shared memory and reused by every beam, eliminating the O(beam) HBM
//     traffic that the baseline `masked_multihead_attention_kernel` pays.
//
// Phase 2 (cascade_suffix_decode_kernel):
//     Each (head, beam) pair processes its own suffix KV [L_p, T) following
//     `cache_indir`. This is a regular single-query decode with no sharing.
//     The numerically stable online-softmax merge with the prefix partial
//     state is fused at the end of this kernel (in-register), so no separate
//     merge kernel launch is needed.
//
// This first version intentionally targets a *narrow* feature subset and
// falls back to MMHA otherwise:
//   - DO_CROSS_ATTENTION = false
//   - POS_SHIFT          = false
//   - BLOCK_SPARSE_ATTN  = false
//   - IMPLICIT_REL_ATTN  = false
//   - ATTN_LOGIT_SOFTCAP = off
//   - PositionEmbedding  in { LEARNED_ABSOLUTE, ROPE_GPT_NEOX } with full-head
//                         rotation and RotaryScalingType::kNONE (covers Qwen,
//                         Llama3, Mistral default configs).
//   - T_cache            = T (no INT8 / FP8 KV cache yet)
//   - Dh                 = 128
//   - T                  in { half, __nv_bfloat16 }
//
// Activation is gated by the env var `TRTLLM_ENABLE_CASCADE_MMHA` (default off).
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns true if the cascade pipeline launched and produced the final output.
// Returns false if the caller should fall back to the standard MMHA path
// (either because gating is off or the params are unsupported).
template <typename T, typename T_cache, typename KVCacheBuffer, int Dh>
bool launch_cascade_attention(
    Multihead_attention_params<T, false> const& params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t stream);

// Convenience predicate that does *not* allocate any workspace and only
// inspects host-side params. Used by the dispatcher to short-circuit cleanly.
template <typename KernelParamsType>
bool cascade_eligible(KernelParamsType const& params);

struct CascadeWorkspaceSizes
{
    size_t out{};
    size_t mMax{};
    size_t lSum{};
};

CascadeWorkspaceSizes getCascadeWorkspaceSizes(int batch_beam, int num_heads, int head_size) noexcept;

} // namespace cascade
} // namespace mmha
} // namespace kernels

TRTLLM_NAMESPACE_END
