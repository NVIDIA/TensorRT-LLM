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
#include <tensorrt_llm/kernels/helixAllToAll.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <tensorrt_llm/thop/moeAlltoAllMeta.h>
#include <torch/extension.h>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::thop
{

void initBindings(nb::module_& m)
{
    // Export MoE A2A constants
    for (auto const& kv : torch_ext::moe_comm::getMoeA2AMetaInfoIndexPairs())
    {
        m.attr(kv.first) = kv.second;
    }

    m.def("attention", &torch_ext::attention,
        // Parameters with default values using std::nullopt for optional arguments
        nb::arg("q"), nb::arg("k").none(), nb::arg("v").none(), nb::arg("output"), nb::arg("output_sf").none(),
        nb::arg("workspace_").none(), nb::arg("sequence_length"), nb::arg("host_past_key_value_lengths"),
        nb::arg("host_total_kv_lens"), nb::arg("context_lengths"), nb::arg("host_context_lengths"),
        nb::arg("host_request_types"), nb::arg("kv_cache_block_offsets").none(),
        nb::arg("host_kv_cache_pool_pointers").none(), nb::arg("host_kv_cache_pool_mapping").none(),
        nb::arg("cache_indirection").none(), nb::arg("kv_scale_orig_quant").none(),
        nb::arg("kv_scale_quant_orig").none(), nb::arg("out_scale").none(), nb::arg("rotary_inv_freq").none(),
        nb::arg("rotary_cos_sin").none(), nb::arg("latent_cache").none(), nb::arg("q_pe").none(),
        nb::arg("block_ids_per_seq").none(), nb::arg("attention_sinks").none(), nb::arg("is_fused_qkv"),
        nb::arg("update_kv_cache"), nb::arg("predicted_tokens_per_seq"), nb::arg("layer_idx"), nb::arg("num_heads"),
        nb::arg("num_kv_heads"), nb::arg("head_size"), nb::arg("tokens_per_block").none(), nb::arg("max_num_requests"),
        nb::arg("max_context_length"), nb::arg("attention_window_size"), nb::arg("sink_token_length"),
        nb::arg("beam_width"), nb::arg("mask_type"), nb::arg("quant_mode"), nb::arg("q_scaling"),
        nb::arg("position_embedding_type"), nb::arg("rotary_embedding_dim"), nb::arg("rotary_embedding_base"),
        nb::arg("rotary_embedding_scale_type"), nb::arg("rotary_embedding_scales"),
        nb::arg("rotary_embedding_max_position_info"), nb::arg("use_paged_context_fmha"),
        nb::arg("attention_input_type").none(), nb::arg("is_mla_enable"),
        nb::arg("chunked_prefill_buffer_batch_size").none(), nb::arg("q_lora_rank").none(),
        nb::arg("kv_lora_rank").none(), nb::arg("qk_nope_head_dim").none(), nb::arg("qk_rope_head_dim").none(),
        nb::arg("v_head_dim").none(), nb::arg("mrope_rotary_cos_sin").none(), nb::arg("mrope_position_deltas").none(),
        nb::arg("mla_tensor_params"), nb::arg("attention_chunk_size").none(), nb::arg("softmax_stats_tensor").none(),
        nb::arg("spec_decoding_bool_params"), nb::arg("spec_decoding_tensor_params"),
        nb::arg("sparse_kv_indices").none(), nb::arg("sparse_kv_offsets").none(), nb::arg("sparse_attn_indices").none(),
        nb::arg("sparse_attn_offsets").none(), nb::arg("sparse_attn_indices_block_size"),
        nb::arg("sparse_mla_topk") = std::nullopt,
        nb::arg("skip_softmax_threshold_scale_factor_prefill") = std::nullopt,
        nb::arg("skip_softmax_threshold_scale_factor_decode") = std::nullopt,
        nb::arg("skip_softmax_stat") = std::nullopt, nb::arg("cu_q_seqlens") = std::nullopt,
        nb::arg("cu_kv_seqlens") = std::nullopt, nb::arg("fmha_scheduler_counter") = std::nullopt,
        nb::arg("mla_bmm1_scale") = std::nullopt, nb::arg("mla_bmm2_scale") = std::nullopt,
        nb::arg("quant_q_buffer") = std::nullopt, "Multi-head attention operation",
        nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "get_helix_workspace_size_per_rank",
        [](int cp_size) { return tensorrt_llm::kernels::computeHelixWorkspaceSizePerRank(cp_size); },
        nb::arg("cp_size"), "Get helix all-to-all workspace size per rank in bytes");
}
} // namespace tensorrt_llm::nanobind::thop
