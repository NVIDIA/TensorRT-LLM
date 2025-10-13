/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

namespace torch_ext
{

/**
 * @brief Attention operation for TensorRT-LLM
 *
 * This function performs multi-head attention computation in-place, supporting both
 * context and generation phases with various optimization features including:
 * - Fused QKV processing
 * - KV cache management
 * - Multiple position embedding types (RoPE, ALiBi, etc.)
 * - Quantization support (FP8, FP4, etc.)
 * - Multi-layer attention (MLA)
 * - Speculative decoding
 */
void attention(torch::Tensor q, std::optional<torch::Tensor> k, std::optional<torch::Tensor> v, torch::Tensor& output,
    std::optional<torch::Tensor> output_sf, std::optional<torch::ScalarType> out_dtype,
    std::optional<torch::Tensor> workspace_, torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths,
    torch::Tensor host_total_kv_lens, torch::Tensor context_lengths, torch::Tensor host_context_lengths,
    torch::Tensor host_request_types, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_block_offsets, std::optional<torch::Tensor> host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> host_kv_cache_pool_mapping, std::optional<torch::Tensor> cache_indirection,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> out_scale, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, std::optional<torch::Tensor> latent_cache,
    std::optional<torch::Tensor> q_pe, std::optional<torch::Tensor> block_ids_per_seq,
    std::optional<torch::Tensor> attention_sinks, bool const is_fused_qkv, bool const update_kv_cache,
    int64_t const predicted_tokens_per_seq, int64_t const layer_idx, int64_t const num_heads,
    int64_t const num_kv_heads, int64_t const head_size, std::optional<int64_t> const tokens_per_block,
    int64_t const max_num_requests, int64_t const max_context_length, int64_t const attention_window_size,
    int64_t const sink_token_length, int64_t const beam_width, int64_t const mask_type, int64_t const quant_mode,
    double const q_scaling, int64_t const position_embedding_type, int64_t const rotary_embedding_dim,
    double const rotary_embedding_base, int64_t const rotary_embedding_scale_type,
    std::vector<double> rotary_embedding_scales, std::vector<int64_t> rotary_embedding_max_position_info,
    bool const use_paged_context_fmha, std::optional<int64_t> attention_input_type, bool is_mla_enable,
    std::optional<int64_t> chunked_prefill_buffer_batch_size, std::optional<int64_t> q_lora_rank,
    std::optional<int64_t> kv_lora_rank, std::optional<int64_t> qk_nope_head_dim,
    std::optional<int64_t> qk_rope_head_dim, std::optional<int64_t> v_head_dim,
    std::optional<torch::Tensor> mrope_rotary_cos_sin, std::optional<torch::Tensor> mrope_position_deltas,
    std::vector<std::optional<torch::Tensor>> mla_tensor_params, std::optional<int64_t> attention_chunk_size,
    std::optional<torch::Tensor> softmax_stats_tensor, std::vector<bool> spec_decoding_bool_params,
    std::vector<std::optional<torch::Tensor>> spec_decoding_tensor_params,
    std::vector<std::optional<torch::Tensor>> sparse_attention_tensor_params,
    std::vector<int64_t> sparse_attention_params);

} // namespace torch_ext
