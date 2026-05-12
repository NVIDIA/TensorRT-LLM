/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
 // Cascade Attention Kernel (experimental v0)
 //
 // Implements the three-stage divide-and-conquer attention from
 // "Cascade Inference: Memory Bandwidth Efficient Shared Prefix Batch Decoding"
 // (https://flashinfer.ai/2024/02/02/cascade-inference.html).
 //
 // Phase 1 (cascade_prefix_mqa_kernel):
 //     For each request, all `beam_width` queries attend to the *shared* prompt
 //     KV (token range [0, L_p)). KV is loaded once per (head, request, token tile)
 //     into shared memory and re-used by every beam, eliminating the O(beam) HBM
 //     traffic that the baseline `masked_multihead_attention_kernel` pays.
 //
 // Phase 2 (cascade_suffix_decode_kernel):
 //     Each (head, beam) pair processes its own suffix KV [L_p, T) following
 //     `cache_indir`. This is a regular single-query decode with no sharing.
 //
 // Phase 3 (cascade_merge_kernel):
 //     Combine the prefix and suffix attention states using the numerically
 //     stable online-softmax merge operator.
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
 //   - Dh                 in { 64, 128 }
 //   - T                  in { half, __nv_bfloat16 }
 //
 // Activation is gated by the env var `TRTLLM_ENABLE_CASCADE_MMHA` (default off).
 //
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
 // Returns true if the cascade pipeline launched and produced the final output.
 // Returns false if the caller should fall back to the standard MMHA path
 // (either because gating is off, the params are unsupported, or no shared
 // prefix is available).
 template <typename T, typename T_cache, typename KVCacheBuffer, int Dh>
 bool launch_cascade_attention(Multihead_attention_params<T, false> const& params,
     KVCacheBuffer const& kv_cache_buffer, cudaStream_t stream);
 
 // Convenience predicate that does *not* allocate any workspace and only
 // inspects host-side params. Used by the dispatcher to short-circuit cleanly.
 template <typename T, typename KernelParamsType>
 bool cascade_eligible(KernelParamsType const& params);
 
 } // namespace cascade
 } // namespace mmha
 } // namespace kernels
 
 TRTLLM_NAMESPACE_END
 