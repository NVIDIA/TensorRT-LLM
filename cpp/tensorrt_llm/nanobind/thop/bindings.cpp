/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <tensorrt_llm/kernels/helixAllToAll.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <tensorrt_llm/thop/moeAlltoAllMeta.h>
#include <tensorrt_llm/thop/outputTensor.h>
#include <tensorrt_llm/thop/trtllmGenFusedOps.h>
#include <torch/extension.h>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::thop
{

namespace
{

template <typename T>
nb::object optionalToObject(std::optional<T> const& value)
{
    if (value.has_value())
    {
        return nb::cast(*value);
    }
    return nb::none();
}

nb::tuple trtllmGenContextPreprocessBinding(torch::Tensor qkv_input, torch::Tensor workspace,
    torch::Tensor sequence_lengths, torch::Tensor context_lengths, std::optional<torch::Tensor> kv_cache_block_offsets,
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
    bool need_build_kv_cache_metadata, std::optional<torch::Tensor> cross_kv, bool cross_attention)
{
    auto result = [&]()
    {
        nb::gil_scoped_release release;
        return torch_ext::trtllmGenContextPreprocess(qkv_input, workspace, sequence_lengths, context_lengths,
            kv_cache_block_offsets, host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_scale_orig_quant,
            kv_scale_quant_orig, attention_output_orig_quant, rotary_inv_freq, rotary_cos_sin, mrope_rotary_cos_sin,
            layer_idx, num_heads, num_kv_heads, head_size, tokens_per_block, mask_type, kv_cache_quant_mode,
            max_attention_window_size, cyclic_attention_window_size, num_tokens, batch_size, input_seq_length,
            max_past_kv_length, rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, bmm1_scale, bmm2_scale,
            attention_chunk_size, fp8_context_fmha, paged_context_fmha, is_mla_enable, multi_processor_count,
            total_num_blocks, kv_factor, need_build_kv_cache_metadata, cross_kv, cross_attention);
    }();

    return nb::make_tuple(std::get<0>(result), optionalToObject(std::get<1>(result)),
        optionalToObject(std::get<2>(result)), optionalToObject(std::get<3>(result)),
        optionalToObject(std::get<4>(result)), optionalToObject(std::get<5>(result)), std::get<6>(result),
        std::get<7>(result), std::get<8>(result), std::get<9>(result), std::get<10>(result), std::get<11>(result));
}

nb::tuple trtllmGenGenerationPreprocessBinding(torch::Tensor qkv_input, torch::Tensor workspace,
    torch::Tensor sequence_lengths, std::optional<torch::Tensor> spec_decoding_generation_lengths,
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
    bool need_build_kv_cache_metadata, bool cross_attention)
{
    auto result = [&]()
    {
        nb::gil_scoped_release release;
        return torch_ext::trtllmGenGenerationPreprocess(qkv_input, workspace, sequence_lengths,
            spec_decoding_generation_lengths, spec_decoding_position_offsets, kv_cache_block_offsets,
            host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_scale_orig_quant, kv_scale_quant_orig,
            attention_output_orig_quant, rotary_inv_freq, rotary_cos_sin, mrope_position_deltas, layer_idx, seq_offset,
            num_heads, num_kv_heads, head_size, tokens_per_block, kv_cache_quant_mode, max_attention_window_size,
            cyclic_attention_window_size, num_tokens, batch_beam, input_seq_length, max_past_kv_length,
            rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale_type, rotary_embedding_scale,
            rotary_embedding_max_positions, position_embedding_type, bmm1_scale, bmm2_scale, fp8_context_fmha,
            predicted_tokens_per_seq, attention_chunk_size, multi_processor_count, total_num_blocks, kv_factor,
            need_build_kv_cache_metadata, cross_attention);
    }();

    return nb::make_tuple(std::get<0>(result), optionalToObject(std::get<1>(result)),
        optionalToObject(std::get<2>(result)), optionalToObject(std::get<3>(result)), std::get<4>(result),
        std::get<5>(result), std::get<6>(result), optionalToObject(std::get<7>(result)), std::get<8>(result),
        std::get<9>(result), std::get<10>(result), std::get<11>(result));
}

} // namespace

void initBindings(nb::module_& m)
{
    // Sync with torch_ext::BufferKind in tensorrt_llm/thop/outputTensor.h
    nb::enum_<torch_ext::BufferKind>(m, "BufferKind", nb::is_arithmetic())
        .value("DEFAULT", torch_ext::BufferKind::Default)
        .value("USERBUFFERS", torch_ext::BufferKind::Userbuffers)
        .value("NCCL_WINDOW", torch_ext::BufferKind::NcclWindow);

    // Export MoE A2A constants
    for (auto const& kv : torch_ext::moe_comm::getMoeA2AMetaInfoIndexPairs())
    {
        m.attr(kv.first) = kv.second;
    }

    m.def("attention", &torch_ext::attention,
        // Parameters with default values: using std::nullopt for trailing optional arguments (omittable)
        // and .none() for optional arguments followed by parameters without default values (not omittable).
        nb::arg("q"), nb::arg("k").none(), nb::arg("v").none(), nb::arg("output"), nb::arg("output_sf").none(),
        nb::arg("workspace_").none(), nb::arg("sequence_length"), nb::arg("host_past_key_value_lengths"),
        nb::arg("host_total_kv_lens"), nb::arg("context_lengths"), nb::arg("host_context_lengths"),
        nb::arg("host_request_types"), nb::arg("max_context_q_len_override").none(),
        nb::arg("kv_cache_block_offsets").none(), nb::arg("host_kv_cache_pool_pointers").none(),
        nb::arg("host_kv_cache_pool_mapping").none(), nb::arg("cache_indirection").none(),
        nb::arg("kv_scale_orig_quant").none(), nb::arg("kv_scale_quant_orig").none(), nb::arg("out_scale").none(),
        nb::arg("rotary_inv_freq").none(), nb::arg("rotary_cos_sin").none(), nb::arg("latent_cache").none(),
        nb::arg("q_pe").none(), nb::arg("block_ids_per_seq").none(), nb::arg("attention_sinks").none(),
        nb::arg("is_fused_qkv"), nb::arg("update_kv_cache"), nb::arg("predicted_tokens_per_seq"),
        nb::arg("local_layer_idx"), nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"),
        nb::arg("tokens_per_block").none(), nb::arg("max_num_requests"), nb::arg("max_context_length"),
        nb::arg("max_seq_len"), nb::arg("attention_window_size"), nb::arg("beam_width"), nb::arg("mask_type"),
        nb::arg("quant_mode"), nb::arg("q_scaling"), nb::arg("position_embedding_type"), nb::arg("rope_dim"),
        nb::arg("rope_base"), nb::arg("rope_scale_type"), nb::arg("rope_scale"), nb::arg("rope_short_m_scale"),
        nb::arg("rope_long_m_scale"), nb::arg("rope_max_positions"), nb::arg("rope_original_max_positions"),
        nb::arg("use_paged_context_fmha"), nb::arg("attention_input_type").none(), nb::arg("is_mla_enable"),
        nb::arg("chunked_prefill_buffer_batch_size").none(), nb::arg("q_lora_rank").none(),
        nb::arg("kv_lora_rank").none(), nb::arg("qk_nope_head_dim").none(), nb::arg("qk_rope_head_dim").none(),
        nb::arg("v_head_dim").none(), nb::arg("rope_append").none(), nb::arg("mrope_rotary_cos_sin").none(),
        nb::arg("mrope_position_deltas").none(), nb::arg("helix_position_offsets").none(),
        nb::arg("helix_is_inactive_rank").none(), nb::arg("attention_chunk_size").none(),
        nb::arg("softmax_stats_tensor").none(), nb::arg("is_spec_decoding_enabled"), nb::arg("use_spec_decoding"),
        nb::arg("is_spec_dec_tree"), nb::arg("spec_decoding_generation_lengths").none(),
        nb::arg("spec_decoding_position_offsets_for_cpp").none(), nb::arg("spec_decoding_packed_mask").none(),
        nb::arg("spec_decoding_bl_tree_mask_offset").none(), nb::arg("spec_decoding_bl_tree_mask").none(),
        nb::arg("spec_bl_tree_first_sparse_mask_offset_kv").none(), nb::arg("sparse_kv_indices").none(),
        nb::arg("sparse_kv_offsets").none(), nb::arg("sparse_attn_indices").none(),
        nb::arg("sparse_attn_offsets").none(), nb::arg("sparse_attn_indices_block_size"),
        nb::arg("num_sparse_topk") = std::nullopt, nb::arg("sparse_mla_topk_lens") = std::nullopt,
        nb::arg("skip_softmax_threshold_scale_factor_prefill") = std::nullopt,
        nb::arg("skip_softmax_threshold_scale_factor_decode") = std::nullopt,
        nb::arg("skip_softmax_stat") = std::nullopt, nb::arg("cu_q_seqlens") = std::nullopt,
        nb::arg("cu_kv_seqlens") = std::nullopt, nb::arg("fmha_scheduler_counter") = std::nullopt,
        nb::arg("mla_bmm1_scale") = std::nullopt, nb::arg("mla_bmm2_scale") = std::nullopt,
        nb::arg("quant_q_buffer") = std::nullopt, nb::arg("flash_mla_tile_scheduler_metadata") = std::nullopt,
        nb::arg("flash_mla_num_splits") = std::nullopt, nb::arg("sage_attn_num_elts_per_blk_q") = 0,
        nb::arg("sage_attn_num_elts_per_blk_k") = 0, nb::arg("sage_attn_num_elts_per_blk_v") = 0,
        nb::arg("sage_attn_qk_int8") = false, nb::arg("num_contexts") = 0, nb::arg("num_ctx_tokens") = 0,
        nb::arg("trtllm_gen_jit_warmup") = false, nb::arg("compressed_kv_cache_pool_ptr") = std::nullopt,
        nb::arg("is_cross") = false, nb::arg("cross_kv") = std::nullopt,
        nb::arg("relative_attention_bias") = std::nullopt, nb::arg("relative_attention_max_distance") = 0,
        nb::arg("spec_decoding_target_max_draft_tokens") = std::nullopt, nb::arg("quant_scale_qkv") = std::nullopt,
        nb::arg("dsv4_inv_rope_cos_sin_cache") = std::nullopt, nb::arg("enable_dsv4_epilogue_fusion") = false,
        "Multi-head attention operation", nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "get_helix_workspace_size_per_rank",
        [](int cp_size) { return tensorrt_llm::kernels::computeHelixWorkspaceSizePerRank(cp_size); },
        nb::arg("cp_size"), "Get helix all-to-all workspace size per rank in bytes");

    m.def("compute_flash_mla_metadata", &tensorrt_llm::computeFlashMlaMetadata, nb::arg("seqlens_k"),
        nb::arg("tile_scheduler_metadata"), nb::arg("num_splits"), nb::arg("batch_size"), nb::arg("s_q"),
        nb::arg("num_q_heads"), nb::arg("num_kv_heads"), nb::arg("head_size_v"),
        "Compute FlashMLA tile-scheduler metadata in-place. Call once per forward pass before attention layers.",
        nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "get_trtllm_gen_context_workspace_layout",
        [](at::ScalarType dtype, int64_t batch_size, int64_t num_tokens, int64_t num_heads, int64_t head_size,
            int64_t rotary_embedding_dim, bool separate_q_kv_input, bool fp8_context_fmha)
        {
            auto const layout = torch_ext::TrtllmAttentionWorkspaceManager::buildContextLayout(dtype, batch_size,
                num_tokens, num_heads, head_size, rotary_embedding_dim, separate_q_kv_input, fp8_context_fmha);
            nb::dict result;
            result["trtllm_gen_workspace_offset"] = layout.trtllmGenWorkspaceOffset;
            result["cu_q_seqlens_offset"] = layout.cuQSeqlensOffset;
            result["cu_kv_seqlens_offset"] = layout.cuKvSeqlensOffset;
            result["cu_mask_rows_offset"] = layout.cuMaskRowsOffset;
            result["rotary_inv_freq_offset"] = layout.rotaryInvFreqOffset;
            result["q_buf_offset"] = layout.qBufOffset;
            result["tokens_info_offset"] = layout.tokensInfoOffset;
            result["fmha_tile_counter_offset"] = layout.fmhaTileCounterOffset;
            result["fmha_bmm1_scale_offset"] = layout.fmhaBmm1ScaleOffset;
            result["fmha_bmm2_scale_offset"] = layout.fmhaBmm2ScaleOffset;
            result["trtllm_gen_workspace_size"] = layout.trtllmGenWorkspaceSize;
            result["cu_seqlens_size"] = layout.cuSeqlensSize;
            result["rotary_inv_freq_size"] = layout.rotaryInvFreqSize;
            result["q_buf_size"] = layout.qBufSize;
            result["tokens_info_size"] = layout.tokensInfoSize;
            result["fmha_scheduler_counter_size"] = layout.fmhaTileCounterSize;
            result["fmha_bmm1_scale_size"] = layout.fmhaBmm1ScaleSize;
            result["fmha_bmm2_scale_size"] = layout.fmhaBmm2ScaleSize;
            result["total_size"] = layout.totalSize;
            return result;
        },
        nb::arg("dtype"), nb::arg("batch_size"), nb::arg("num_tokens"), nb::arg("num_heads"), nb::arg("head_size"),
        nb::arg("rotary_embedding_dim"), nb::arg("separate_q_kv_input"), nb::arg("fp8_context_fmha"),
        "Return the C++ trtllm-gen context workspace layout.");

    m.def(
        "get_trtllm_gen_generation_workspace_layout",
        [](at::ScalarType dtype, int64_t batch_beam, int64_t num_tokens, int64_t num_heads, int64_t head_size,
            int64_t rotary_embedding_dim, int64_t num_kv_heads, int64_t max_blocks_per_sequence,
            bool use_sparse_attention)
        {
            auto const layout = torch_ext::TrtllmAttentionWorkspaceManager::buildGenerationLayout(dtype, batch_beam,
                num_tokens, num_heads, head_size, rotary_embedding_dim, num_kv_heads, max_blocks_per_sequence,
                use_sparse_attention);
            nb::dict result;
            result["trtllm_gen_workspace_offset"] = layout.trtllmGenWorkspaceOffset;
            result["cu_seqlens_offset"] = layout.cuSeqlensOffset;
            result["cu_kv_seqlens_offset"] = layout.cuKvSeqlensOffset;
            result["rotary_inv_freq_offset"] = layout.rotaryInvFreqOffset;
            result["tokens_info_offset"] = layout.tokensInfoOffset;
            result["q_buf_offset"] = layout.qBufOffset;
            result["bmm1_scale_offset"] = layout.bmm1ScaleOffset;
            result["bmm2_scale_offset"] = layout.bmm2ScaleOffset;
            result["sparse_attn_cache_offset"] = layout.sparseAttnCacheOffset;
            result["trtllm_gen_workspace_size"] = layout.trtllmGenWorkspaceSize;
            result["cu_seqlens_size"] = layout.cuSeqlensSize;
            result["cu_kv_seqlens_size"] = layout.cuKvSeqlensSize;
            result["rotary_inv_freq_size"] = layout.rotaryInvFreqSize;
            result["tokens_info_size"] = layout.tokensInfoSize;
            result["q_buf_size"] = layout.qBufSize;
            result["bmm1_scale_size"] = layout.bmm1ScaleSize;
            result["bmm2_scale_size"] = layout.bmm2ScaleSize;
            result["sparse_attn_cache_size"] = layout.sparseAttnCacheSize;
            result["total_size"] = layout.totalSize;
            return result;
        },
        nb::arg("dtype"), nb::arg("batch_beam"), nb::arg("num_tokens"), nb::arg("num_heads"), nb::arg("head_size"),
        nb::arg("rotary_embedding_dim"), nb::arg("num_kv_heads"), nb::arg("max_blocks_per_sequence") = 0,
        nb::arg("use_sparse_attention") = false, "Return the C++ trtllm-gen generation workspace layout.");

    m.def("trtllm_gen_context_preprocess", &trtllmGenContextPreprocessBinding, nb::arg("qkv_input"),
        nb::arg("workspace"), nb::arg("sequence_lengths"), nb::arg("context_lengths"),
        nb::arg("kv_cache_block_offsets").none(), nb::arg("host_kv_cache_pool_pointers").none(),
        nb::arg("host_kv_cache_pool_mapping").none(), nb::arg("kv_scale_orig_quant").none(),
        nb::arg("kv_scale_quant_orig").none(), nb::arg("attention_output_orig_quant").none(),
        nb::arg("rotary_inv_freq").none(), nb::arg("rotary_cos_sin").none(), nb::arg("mrope_rotary_cos_sin").none(),
        nb::arg("layer_idx"), nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"),
        nb::arg("tokens_per_block"), nb::arg("mask_type"), nb::arg("kv_cache_quant_mode"),
        nb::arg("max_attention_window_size"), nb::arg("cyclic_attention_window_size"), nb::arg("num_tokens"),
        nb::arg("batch_size"), nb::arg("input_seq_length"), nb::arg("max_past_kv_length"),
        nb::arg("rotary_embedding_dim"), nb::arg("rotary_embedding_base"), nb::arg("rotary_embedding_scale_type"),
        nb::arg("rotary_embedding_scale"), nb::arg("rotary_embedding_max_positions"),
        nb::arg("position_embedding_type"), nb::arg("bmm1_scale"), nb::arg("bmm2_scale"),
        nb::arg("attention_chunk_size"), nb::arg("fp8_context_fmha"), nb::arg("paged_context_fmha"),
        nb::arg("is_mla_enable"), nb::arg("multi_processor_count"), nb::arg("total_num_blocks"), nb::arg("kv_factor"),
        nb::arg("need_build_kv_cache_metadata") = true, nb::arg("cross_kv").none() = nb::none(),
        nb::arg("cross_attention") = false, "Fused nanobind context preprocess for trtllm-gen attention.");

    m.def("trtllm_gen_context_postprocess", &torch_ext::trtllmGenContextPostprocess, nb::arg("qkv_input"),
        nb::arg("workspace"), nb::arg("sequence_lengths"), nb::arg("context_lengths"),
        nb::arg("kv_cache_block_offsets").none(), nb::arg("host_kv_cache_pool_pointers").none(),
        nb::arg("host_kv_cache_pool_mapping").none(), nb::arg("kv_scale_orig_quant").none(),
        nb::arg("kv_scale_quant_orig").none(), nb::arg("attention_output_orig_quant").none(),
        nb::arg("rotary_cos_sin").none(), nb::arg("mrope_rotary_cos_sin").none(), nb::arg("layer_idx"),
        nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"), nb::arg("tokens_per_block"),
        nb::arg("mask_type"), nb::arg("kv_cache_quant_mode"), nb::arg("max_attention_window_size"),
        nb::arg("cyclic_attention_window_size"), nb::arg("num_tokens"), nb::arg("batch_size"),
        nb::arg("input_seq_length"), nb::arg("max_past_kv_length"), nb::arg("rotary_embedding_dim"),
        nb::arg("rotary_embedding_base"), nb::arg("rotary_embedding_scale_type"), nb::arg("rotary_embedding_scale"),
        nb::arg("rotary_embedding_max_positions"), nb::arg("position_embedding_type"), nb::arg("bmm1_scale"),
        nb::arg("fp8_context_fmha"), nb::arg("paged_context_fmha"), nb::arg("is_mla_enable"),
        nb::arg("attention_chunk_size"), nb::arg("multi_processor_count"),
        "Fused nanobind context postprocess for trtllm-gen attention.", nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "build_trtllm_gen_kv_cache_metadata",
        [](torch::Tensor host_kv_cache_pool_pointers, torch::Tensor host_kv_cache_pool_mapping,
            torch::Tensor kv_cache_block_offsets, int64_t layer_idx, int64_t num_kv_heads, int64_t tokens_per_block,
            int64_t head_dim, int64_t kv_factor, int64_t total_num_blocks, int64_t kv_cache_quant_mode,
            int64_t batch_start, int64_t batch_size, at::ScalarType dtype) -> nb::tuple
        {
            at::Tensor kvPool;
            std::optional<at::Tensor> kvScalePool;
            at::Tensor blockTables;
            {
                nb::gil_scoped_release release;
                std::tie(kvPool, kvScalePool) = torch_ext::buildFlashinferTrtllmGenPagedKvCacheBuffers(
                    host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, layer_idx, num_kv_heads, tokens_per_block,
                    head_dim, kv_factor, total_num_blocks, kv_cache_quant_mode, dtype);
                auto const mapping = torch_ext::readKvCachePoolMapping(host_kv_cache_pool_mapping, layer_idx);
                blockTables = kv_cache_block_offsets.select(0, mapping.poolIndex).narrow(0, batch_start, batch_size);
            }
            return nb::make_tuple(nb::cast(kvPool), nb::cast(blockTables), optionalToObject(kvScalePool));
        },
        nb::arg("host_kv_cache_pool_pointers"), nb::arg("host_kv_cache_pool_mapping"),
        nb::arg("kv_cache_block_offsets"), nb::arg("layer_idx"), nb::arg("num_kv_heads"), nb::arg("tokens_per_block"),
        nb::arg("head_dim"), nb::arg("kv_factor"), nb::arg("total_num_blocks"), nb::arg("kv_cache_quant_mode"),
        nb::arg("batch_start"), nb::arg("batch_size"), nb::arg("dtype"),
        "Build flashinfer-style KV cache pool view and slice block tables for a given layer.");

    m.def("trtllm_gen_generation_preprocess", &trtllmGenGenerationPreprocessBinding, nb::arg("qkv_input"),
        nb::arg("workspace"), nb::arg("sequence_lengths"), nb::arg("spec_decoding_generation_lengths").none(),
        nb::arg("spec_decoding_position_offsets").none(), nb::arg("kv_cache_block_offsets").none(),
        nb::arg("host_kv_cache_pool_pointers").none(), nb::arg("host_kv_cache_pool_mapping").none(),
        nb::arg("kv_scale_orig_quant").none(), nb::arg("kv_scale_quant_orig").none(),
        nb::arg("attention_output_orig_quant").none(), nb::arg("rotary_inv_freq").none(),
        nb::arg("rotary_cos_sin").none(), nb::arg("mrope_position_deltas").none(), nb::arg("layer_idx"),
        nb::arg("seq_offset"), nb::arg("num_heads"), nb::arg("num_kv_heads"), nb::arg("head_size"),
        nb::arg("tokens_per_block"), nb::arg("kv_cache_quant_mode"), nb::arg("max_attention_window_size"),
        nb::arg("cyclic_attention_window_size"), nb::arg("num_tokens"), nb::arg("batch_beam"),
        nb::arg("input_seq_length"), nb::arg("max_past_kv_length"), nb::arg("rotary_embedding_dim"),
        nb::arg("rotary_embedding_base"), nb::arg("rotary_embedding_scale_type"), nb::arg("rotary_embedding_scale"),
        nb::arg("rotary_embedding_max_positions"), nb::arg("position_embedding_type"), nb::arg("bmm1_scale"),
        nb::arg("bmm2_scale"), nb::arg("fp8_context_fmha"), nb::arg("predicted_tokens_per_seq"),
        nb::arg("attention_chunk_size"), nb::arg("multi_processor_count"), nb::arg("total_num_blocks"),
        nb::arg("kv_factor"), nb::arg("need_build_kv_cache_metadata") = true, nb::arg("cross_attention") = false,
        "Fused nanobind generation preprocess for trtllm-gen attention.");
}
} // namespace tensorrt_llm::nanobind::thop
