/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

// Index-K output of the MiniMax-M3 horizontal producers (their index_k_mode).
// Fp8Sink: E4M3 rows into a paged [num_pages, 1, page_size, head_dim] cache
// (the FP8 index-K cache, or the FP8 sink an NVFP4 index-K layer packs from).
// Nvfp4Pages: the index-K rounded to BF16, quantized straight into the NVFP4
// index-K pages (9216 bytes per page, byte page stride; the m31idxk writer's
// recipe). Bf16Tap (tests): those BF16-rounded rows as a paged BF16 cache.
constexpr int kMinimaxM3IndexKFp8Sink = 0;
constexpr int kMinimaxM3IndexKNvfp4Pages = 1;
constexpr int kMinimaxM3IndexKBf16Tap = 2;

// MiniMax-M3 sparse horizontal producer for an FP8 main KV cache: Q, index-Q and main K/V as E4M3, index-K as
// index_k_mode says (mode 0: the same outputs as trtllm::minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert).
void launchMinimaxM3Fp8QKVIndexerNormRopeKVInsert(void const* packed_input, void* q_output, void* index_q_output,
    void* kv_cache, void* index_k_cache, int const* out_cache_loc, int64_t kv_page_stride, int64_t kv_plane_stride,
    int64_t kv_head_stride, int64_t kv_token_stride, int64_t index_page_stride, int64_t index_token_stride,
    int64_t num_pages, int page_size, int num_tokens, int num_heads_q, int num_heads_kv, int num_heads_index,
    int head_dim, int rotary_dim, float eps, void const* q_weight, void const* k_weight, void const* index_q_weight,
    void const* index_k_weight, float const* rotary_cos_sin, int const* position_ids, int index_k_mode,
    cudaStream_t stream);

// MiniMax-M3 sparse horizontal producer for an NVFP4 main KV cache.  Q and
// index-Q remain FP8; index-K is written as index_k_mode says.  Main K/V are
// quantized directly from the normalized/RoPE registers into packed E2M1 data
// plus E4M3 SF16 scales.  early_trigger (IDX-PDL) fires the PDL launch_dependents
// trigger right after the kernel's own grid-dependency wait, so a PDL-launched
// dependent may launch while the producer runs; that dependent must read the
// producer's outputs only after its own grid-dependency wait.
void launchMinimaxM3Nvfp4QKVIndexerNormRopeKVInsert(void const* packed_input, void* q_output, void* index_q_output,
    void* kv_data_cache, void* kv_scale_cache, void* index_k_cache, int const* out_cache_loc,
    float const* kv_quant_scale, int64_t data_page_stride, int64_t data_plane_stride, int64_t data_head_stride,
    int64_t data_token_stride, int64_t scale_page_stride, int64_t scale_plane_stride, int64_t scale_head_stride,
    int64_t index_page_stride, int64_t index_token_stride, int64_t num_pages, int page_size, int num_tokens,
    int num_heads_q, int num_heads_kv, int num_heads_index, int head_dim, int rotary_dim, float eps,
    void const* q_weight, void const* k_weight, void const* index_q_weight, void const* index_k_weight,
    float const* rotary_cos_sin, int const* position_ids, int index_k_mode, cudaStream_t stream,
    bool early_trigger = false);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
