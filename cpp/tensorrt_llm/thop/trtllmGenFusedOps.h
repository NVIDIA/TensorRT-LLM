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

#include <optional>
#include <torch/extension.h>
#include <tuple>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t>
trtllmGenContextPreprocess(torch::Tensor qkv_input, torch::Tensor workspace, torch::Tensor sequence_lengths,
    torch::Tensor context_lengths, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> attention_output_orig_quant, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, std::optional<torch::Tensor> mrope_rotary_cos_sin, int64_t layer_idx,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_size, int64_t tokens_per_block, int64_t mask_type,
    int64_t kv_cache_quant_mode, int64_t max_attention_window_size, int64_t cyclic_attention_window_size,
    int64_t num_tokens, int64_t batch_size, int64_t input_seq_length, int64_t max_past_kv_length,
    int64_t rotary_embedding_dim, double rotary_embedding_base, int64_t rotary_embedding_scale_type,
    double rotary_embedding_scale, int64_t rotary_embedding_max_positions, int64_t position_embedding_type,
    double bmm1_scale, double bmm2_scale, int64_t attention_chunk_size, bool fp8_context_fmha, bool paged_context_fmha,
    bool is_mla_enable, int64_t multi_processor_count, int64_t total_num_blocks, int64_t kv_factor,
    bool need_build_kv_cache_metadata, std::optional<torch::Tensor> cross_kv = std::nullopt,
    bool cross_attention = false);

void trtllmGenContextPostprocess(torch::Tensor qkv_input, torch::Tensor workspace, torch::Tensor sequence_lengths,
    torch::Tensor context_lengths, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> attention_output_orig_quant, std::optional<torch::Tensor> rotary_cos_sin,
    std::optional<torch::Tensor> mrope_rotary_cos_sin, int64_t layer_idx, int64_t num_heads, int64_t num_kv_heads,
    int64_t head_size, int64_t tokens_per_block, int64_t mask_type, int64_t kv_cache_quant_mode,
    int64_t max_attention_window_size, int64_t cyclic_attention_window_size, int64_t num_tokens, int64_t batch_size,
    int64_t input_seq_length, int64_t max_past_kv_length, int64_t rotary_embedding_dim, double rotary_embedding_base,
    int64_t rotary_embedding_scale_type, double rotary_embedding_scale, int64_t rotary_embedding_max_positions,
    int64_t position_embedding_type, double bmm1_scale, bool fp8_context_fmha, bool paged_context_fmha,
    bool is_mla_enable, int64_t attention_chunk_size, int64_t multi_processor_count);

std::tuple<at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>, at::Tensor,
    at::Tensor, at::Tensor, std::optional<at::Tensor>, int64_t, int64_t, int64_t, bool>
trtllmGenGenerationPreprocess(torch::Tensor qkv_input, torch::Tensor workspace, torch::Tensor sequence_lengths,
    std::optional<torch::Tensor> spec_decoding_generation_lengths,
    std::optional<torch::Tensor> spec_decoding_position_offsets, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> attention_output_orig_quant, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, std::optional<torch::Tensor> mrope_position_deltas, int64_t layer_idx,
    int64_t seq_offset, int64_t num_heads, int64_t num_kv_heads, int64_t head_size, int64_t tokens_per_block,
    int64_t kv_cache_quant_mode, int64_t max_attention_window_size, int64_t cyclic_attention_window_size,
    int64_t num_tokens, int64_t batch_beam, int64_t input_seq_length, int64_t max_past_kv_length,
    int64_t rotary_embedding_dim, double rotary_embedding_base, int64_t rotary_embedding_scale_type,
    double rotary_embedding_scale, int64_t rotary_embedding_max_positions, int64_t position_embedding_type,
    double bmm1_scale, double bmm2_scale, bool fp8_context_fmha, int64_t predicted_tokens_per_seq,
    int64_t attention_chunk_size, int64_t multi_processor_count, int64_t total_num_blocks, int64_t kv_factor,
    bool need_build_kv_cache_metadata, bool cross_attention = false);

} // namespace torch_ext

TRTLLM_NAMESPACE_END
