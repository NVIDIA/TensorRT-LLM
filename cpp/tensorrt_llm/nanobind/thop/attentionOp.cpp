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

#include <nanobind/nanobind.h>
#include <nanobind/stl.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <torch/extension.h>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::thop::attentionOp
{

void initBindings(nanobind::module_& m)
{
    m.def(
        "attention_inplace",
        [](torch::Tensor q, torch::optional<torch::Tensor> k, torch::optional<torch::Tensor> v, torch::Tensor& output,
            torch::optional<torch::Tensor> output_sf, std::optional<torch::ScalarType> out_dtype,
            torch::optional<torch::Tensor> workspace_, torch::Tensor sequence_length,
            torch::Tensor host_past_key_value_lengths, torch::Tensor context_lengths,
            torch::Tensor host_context_lengths, torch::Tensor host_request_types,
            torch::optional<torch::Tensor> kv_cache_block_offsets,
            torch::optional<torch::Tensor> host_kv_cache_block_offsets,
            torch::optional<torch::Tensor> host_kv_cache_pool_pointers,
            torch::optional<torch::Tensor> host_kv_cache_pool_mapping, torch::optional<torch::Tensor> cache_indirection,
            torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
            torch::optional<torch::Tensor> out_scale, torch::optional<torch::Tensor> rotary_inv_freq,
            torch::optional<torch::Tensor> rotary_cos_sin, torch::optional<torch::Tensor> latent_cache,
            torch::optional<torch::Tensor> q_pe, torch::optional<torch::Tensor> block_ids_per_seq,
            torch::optional<torch::Tensor> attention_sinks, bool is_fused_qkv, bool update_kv_cache,
            int64_t predicted_tokens_per_seq, int64_t layer_idx, int64_t num_heads, int64_t num_kv_heads,
            int64_t head_size, std::optional<int64_t> tokens_per_block, int64_t max_num_requests,
            int64_t max_context_length, int64_t attention_window_size, int64_t sink_token_length, int64_t beam_width,
            int64_t mask_type, int64_t quant_mode, double q_scaling, int64_t position_embedding_type,
            int64_t rotary_embedding_dim, double rotary_embedding_base, int64_t rotary_embedding_scale_type,
            std::vector<double> rotary_embedding_scales, std::vector<int64_t> rotary_embedding_max_position_info,
            bool use_paged_context_fmha, std::optional<int64_t> attention_input_type, bool is_mla_enable,
            std::optional<int64_t> q_lora_rank, std::optional<int64_t> kv_lora_rank,
            std::optional<int64_t> qk_nope_head_dim, std::optional<int64_t> qk_rope_head_dim,
            std::optional<int64_t> v_head_dim, torch::optional<torch::Tensor> mrope_rotary_cos_sin,
            torch::optional<torch::Tensor> mrope_position_deltas, std::optional<torch::Tensor> mla_context_paged_kv,
            std::optional<torch::Tensor> mla_context_kv_cache_block_offsets,
            std::optional<int64_t> attention_chunk_size, std::optional<torch::Tensor> softmax_stats_tensor,
            std::vector<bool> spec_decoding_bool_params,
            std::vector<std::optional<torch::Tensor>> spec_decoding_tensor_params)
        {
            // Transform vector<bool> to c10::List<bool>
            c10::List<bool> cpp_spec_decoding_bool_list;
            for (bool val : spec_decoding_bool_params)
            {
                cpp_spec_decoding_bool_list.push_back(val);
            }

            // Call the original attention_inplace function
            torch_ext::attention_inplace(q, k, v, output, output_sf, out_dtype, workspace_, sequence_length,
                host_past_key_value_lengths, context_lengths, host_context_lengths, host_request_types,
                kv_cache_block_offsets, host_kv_cache_block_offsets, host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping, cache_indirection, kv_scale_orig_quant, kv_scale_quant_orig, out_scale,
                rotary_inv_freq, rotary_cos_sin, latent_cache, q_pe, block_ids_per_seq, attention_sinks, is_fused_qkv,
                update_kv_cache, predicted_tokens_per_seq, layer_idx, num_heads, num_kv_heads, head_size,
                tokens_per_block, max_num_requests, max_context_length, attention_window_size, sink_token_length,
                beam_width, mask_type, quant_mode, q_scaling, position_embedding_type, rotary_embedding_dim,
                rotary_embedding_base, rotary_embedding_scale_type, c10::ArrayRef<double>(rotary_embedding_scales),
                c10::ArrayRef<int64_t>(rotary_embedding_max_position_info), use_paged_context_fmha,
                attention_input_type, is_mla_enable, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
                v_head_dim, mrope_rotary_cos_sin, mrope_position_deltas, mla_context_paged_kv,
                mla_context_kv_cache_block_offsets, attention_chunk_size, softmax_stats_tensor,
                cpp_spec_decoding_bool_list, c10::ArrayRef<std::optional<torch::Tensor>>(spec_decoding_tensor_params));
        },
        // Parameters with default values using std::nullopt for optional arguments
        nb::arg("q"), nb::arg("k") = torch::nullopt, nb::arg("v") = torch::nullopt, nb::arg("output"),
        nb::arg("output_sf") = torch::nullopt, nb::arg("out_dtype") = std::nullopt,
        nb::arg("workspace_") = torch::nullopt, nb::arg("sequence_length"), nb::arg("host_past_key_value_lengths"),
        nb::arg("context_lengths"), nb::arg("host_context_lengths"), nb::arg("host_request_types"),
        nb::arg("kv_cache_block_offsets") = torch::nullopt, nb::arg("host_kv_cache_block_offsets") = torch::nullopt,
        nb::arg("host_kv_cache_pool_pointers") = torch::nullopt, nb::arg("host_kv_cache_pool_mapping") = torch::nullopt,
        nb::arg("cache_indirection") = torch::nullopt, nb::arg("kv_scale_orig_quant") = torch::nullopt,
        nb::arg("kv_scale_quant_orig") = torch::nullopt, nb::arg("out_scale") = torch::nullopt,
        nb::arg("rotary_inv_freq") = torch::nullopt, nb::arg("rotary_cos_sin") = torch::nullopt,
        nb::arg("latent_cache") = torch::nullopt, nb::arg("q_pe") = torch::nullopt,
        nb::arg("block_ids_per_seq") = torch::nullopt, nb::arg("attention_sinks") = torch::nullopt,
        nb::arg("is_fused_qkv"), nb::arg("update_kv_cache"), nb::arg("predicted_tokens_per_seq"), nb::arg("layer_idx"),
        nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"), nb::arg("tokens_per_block") = std::nullopt,
        nb::arg("max_num_requests"), nb::arg("max_context_length"), nb::arg("attention_window_size"),
        nb::arg("sink_token_length"), nb::arg("beam_width"), nb::arg("mask_type"), nb::arg("quant_mode"),
        nb::arg("q_scaling"), nb::arg("position_embedding_type"), nb::arg("rotary_embedding_dim"),
        nb::arg("rotary_embedding_base"), nb::arg("rotary_embedding_scale_type"), nb::arg("rotary_embedding_scales"),
        nb::arg("rotary_embedding_max_position_info"), nb::arg("use_paged_context_fmha"),
        nb::arg("attention_input_type") = std::nullopt, nb::arg("is_mla_enable"), nb::arg("q_lora_rank") = std::nullopt,
        nb::arg("kv_lora_rank") = std::nullopt, nb::arg("qk_nope_head_dim") = std::nullopt,
        nb::arg("qk_rope_head_dim") = std::nullopt, nb::arg("v_head_dim") = std::nullopt,
        nb::arg("mrope_rotary_cos_sin") = torch::nullopt, nb::arg("mrope_position_deltas") = torch::nullopt,
        nb::arg("mla_context_paged_kv") = std::nullopt, nb::arg("mla_context_kv_cache_block_offsets") = std::nullopt,
        nb::arg("attention_chunk_size") = std::nullopt, nb::arg("softmax_stats_tensor") = std::nullopt,
        nb::arg("spec_decoding_bool_params"), nb::arg("spec_decoding_tensor_params"),
        "In-place multi-head attention operation");

    m.def("attention_supports_nvfp4_output", &torch_ext::attention_supports_nvfp4_output, nb::arg("num_heads"),
        nb::arg("num_kv_heads"), nb::arg("head_size"), nb::arg("tokens_per_block") = std::nullopt, nb::arg("mask_type"),
        nb::arg("quant_mode"), nb::arg("use_paged_context_fmha"), nb::arg("is_mla_enable"),
        "Check if attention operation supports NVFP4 output format");
}
} // namespace tensorrt_llm::nanobind::thop::attentionOp
