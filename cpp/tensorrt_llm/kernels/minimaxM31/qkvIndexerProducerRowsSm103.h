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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! Tokens per warp of the rows producer (#20629's kNvfp4ProducerRowsTokens).
constexpr int kQkvIndexerProducerRowsTokens = 4;

//! The MiniMax-M3 NVFP4 QKV + indexer norm / RoPE / KV-insert producer for long chunks (#20629 L2, cf8a77e816): one
//! warp per (head slot, 4 consecutive tokens) instead of per (token, head slot), bitwise the outputs of the per-token
//! launchMinimaxM3Nvfp4QKVIndexerNormRopeKVInsert. Same arguments; the caller decides when to take it (long chunks).
void launchQkvIndexerProducerRows(void const* packed_input, void* q_output, void* index_q_output, void* kv_data_cache,
    void* kv_scale_cache, void* index_k_cache, int const* out_cache_loc, float const* kv_quant_scale,
    int64_t data_page_stride, int64_t data_plane_stride, int64_t data_head_stride, int64_t data_token_stride,
    int64_t scale_page_stride, int64_t scale_plane_stride, int64_t scale_head_stride, int64_t index_page_stride,
    int64_t index_token_stride, int64_t num_pages, int page_size, int num_tokens, int num_heads_q, int num_heads_kv,
    int num_heads_index, int head_dim, int rotary_dim, float eps, void const* q_weight, void const* k_weight,
    void const* index_q_weight, void const* index_k_weight, float const* rotary_cos_sin, int const* position_ids,
    cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
