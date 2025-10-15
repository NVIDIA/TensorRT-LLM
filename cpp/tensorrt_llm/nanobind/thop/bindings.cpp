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

#include "bindings.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <torch/extension.h>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::thop
{

void initBindings(nb::module_& m)
{
    m.def("attention", &torch_ext::attention,
        // Parameters with default values using std::nullopt for optional arguments
        nb::arg("q"), nb::arg("k") = std::nullopt, nb::arg("v") = std::nullopt, nb::arg("output"),
        nb::arg("output_sf") = std::nullopt, nb::arg("out_dtype") = std::nullopt, nb::arg("workspace_") = std::nullopt,
        nb::arg("sequence_length"), nb::arg("host_past_key_value_lengths"), nb::arg("host_total_kv_lens"),
        nb::arg("context_lengths"), nb::arg("host_context_lengths"), nb::arg("host_request_types"),
        nb::arg("kv_cache_block_offsets") = std::nullopt, nb::arg("host_kv_cache_block_offsets") = std::nullopt,
        nb::arg("host_kv_cache_pool_pointers") = std::nullopt, nb::arg("host_kv_cache_pool_mapping") = std::nullopt,
        nb::arg("cache_indirection") = std::nullopt, nb::arg("kv_scale_orig_quant") = std::nullopt,
        nb::arg("kv_scale_quant_orig") = std::nullopt, nb::arg("out_scale") = std::nullopt,
        nb::arg("rotary_inv_freq") = std::nullopt, nb::arg("rotary_cos_sin") = std::nullopt,
        nb::arg("latent_cache") = std::nullopt, nb::arg("q_pe") = std::nullopt,
        nb::arg("block_ids_per_seq") = std::nullopt, nb::arg("attention_sinks") = std::nullopt, nb::arg("is_fused_qkv"),
        nb::arg("update_kv_cache"), nb::arg("predicted_tokens_per_seq"), nb::arg("layer_idx"), nb::arg("num_heads"),
        nb::arg("num_kv_heads"), nb::arg("head_size"), nb::arg("tokens_per_block") = std::nullopt,
        nb::arg("max_num_requests"), nb::arg("max_context_length"), nb::arg("attention_window_size"),
        nb::arg("sink_token_length"), nb::arg("beam_width"), nb::arg("mask_type"), nb::arg("quant_mode"),
        nb::arg("q_scaling"), nb::arg("position_embedding_type"), nb::arg("rotary_embedding_dim"),
        nb::arg("rotary_embedding_base"), nb::arg("rotary_embedding_scale_type"), nb::arg("rotary_embedding_scales"),
        nb::arg("rotary_embedding_max_position_info"), nb::arg("use_paged_context_fmha"),
        nb::arg("attention_input_type") = std::nullopt, nb::arg("is_mla_enable"),
        nb::arg("chunked_prefill_buffer_batch_size") = std::nullopt, nb::arg("q_lora_rank") = std::nullopt,
        nb::arg("kv_lora_rank") = std::nullopt, nb::arg("qk_nope_head_dim") = std::nullopt,
        nb::arg("qk_rope_head_dim") = std::nullopt, nb::arg("v_head_dim") = std::nullopt,
        nb::arg("mrope_rotary_cos_sin") = std::nullopt, nb::arg("mrope_position_deltas") = std::nullopt,
        nb::arg("mla_tensor_params"), nb::arg("attention_chunk_size") = std::nullopt,
        nb::arg("softmax_stats_tensor") = std::nullopt, nb::arg("spec_decoding_bool_params"),
        nb::arg("spec_decoding_tensor_params"), nb::arg("sparse_attention_params"), "Multi-head attention operation",
        nb::call_guard<nb::gil_scoped_release>());
}
} // namespace tensorrt_llm::nanobind::thop
