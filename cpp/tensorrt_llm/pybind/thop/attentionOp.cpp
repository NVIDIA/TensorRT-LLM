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

#include "attentionOp.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace tensorrt_llm::pybind::thop::attentionOp
{

void initBindings(pybind11::module_& m)
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
        py::arg("q"), py::arg("k") = torch::nullopt, py::arg("v") = torch::nullopt, py::arg("output"),
        py::arg("output_sf") = torch::nullopt, py::arg("out_dtype") = std::nullopt,
        py::arg("workspace_") = torch::nullopt, py::arg("sequence_length"), py::arg("host_past_key_value_lengths"),
        py::arg("context_lengths"), py::arg("host_context_lengths"), py::arg("host_request_types"),
        py::arg("kv_cache_block_offsets") = torch::nullopt, py::arg("host_kv_cache_block_offsets") = torch::nullopt,
        py::arg("host_kv_cache_pool_pointers") = torch::nullopt, py::arg("host_kv_cache_pool_mapping") = torch::nullopt,
        py::arg("cache_indirection") = torch::nullopt, py::arg("kv_scale_orig_quant") = torch::nullopt,
        py::arg("kv_scale_quant_orig") = torch::nullopt, py::arg("out_scale") = torch::nullopt,
        py::arg("rotary_inv_freq") = torch::nullopt, py::arg("rotary_cos_sin") = torch::nullopt,
        py::arg("latent_cache") = torch::nullopt, py::arg("q_pe") = torch::nullopt,
        py::arg("block_ids_per_seq") = torch::nullopt, py::arg("attention_sinks") = torch::nullopt,
        py::arg("is_fused_qkv"), py::arg("update_kv_cache"), py::arg("predicted_tokens_per_seq"), py::arg("layer_idx"),
        py::arg("num_heads"), py::arg("num_kv_heads"), py::arg("head_size"), py::arg("tokens_per_block") = std::nullopt,
        py::arg("max_num_requests"), py::arg("max_context_length"), py::arg("attention_window_size"),
        py::arg("sink_token_length"), py::arg("beam_width"), py::arg("mask_type"), py::arg("quant_mode"),
        py::arg("q_scaling"), py::arg("position_embedding_type"), py::arg("rotary_embedding_dim"),
        py::arg("rotary_embedding_base"), py::arg("rotary_embedding_scale_type"), py::arg("rotary_embedding_scales"),
        py::arg("rotary_embedding_max_position_info"), py::arg("use_paged_context_fmha"),
        py::arg("attention_input_type") = std::nullopt, py::arg("is_mla_enable"), py::arg("q_lora_rank") = std::nullopt,
        py::arg("kv_lora_rank") = std::nullopt, py::arg("qk_nope_head_dim") = std::nullopt,
        py::arg("qk_rope_head_dim") = std::nullopt, py::arg("v_head_dim") = std::nullopt,
        py::arg("mrope_rotary_cos_sin") = torch::nullopt, py::arg("mrope_position_deltas") = torch::nullopt,
        py::arg("mla_context_paged_kv") = std::nullopt, py::arg("mla_context_kv_cache_block_offsets") = std::nullopt,
        py::arg("attention_chunk_size") = std::nullopt, py::arg("softmax_stats_tensor") = std::nullopt,
        py::arg("spec_decoding_bool_params"), py::arg("spec_decoding_tensor_params"),
        "In-place multi-head attention operation");

    m.def("attention_supports_nvfp4_output", &torch_ext::attention_supports_nvfp4_output, py::arg("num_heads"),
        py::arg("num_kv_heads"), py::arg("head_size"), py::arg("tokens_per_block") = std::nullopt, py::arg("mask_type"),
        py::arg("quant_mode"), py::arg("use_paged_context_fmha"), py::arg("is_mla_enable"),
        "Check if attention operation supports NVFP4 output format");
}
} // namespace tensorrt_llm::pybind::thop::attentionOp
