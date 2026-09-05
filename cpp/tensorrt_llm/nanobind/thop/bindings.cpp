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
#include <nanobind/stl/set.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <tensorrt_llm/kernels/helixAllToAll.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <tensorrt_llm/thop/moeAlltoAllMeta.h>
#include <tensorrt_llm/thop/outputTensor.h>
#include <tensorrt_llm/thop/trtllmGenFusedOps.h>
#include <torch/extension.h>
#include <type_traits>
#include <utility>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::thop
{

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

    // ---- Phased attention API ----
    // Bind the nested structs before FmhaParams so its `fwd` member has a registered
    // type. def_rw hands back a reference into the parent, which is what lets Python
    // fill a nested struct in place.
    auto sparseBackendForwardArgs
        = nb::class_<torch_ext::SparseBackendForwardArgs>(m, "SparseBackendForwardArgs").def(nb::init<>());
#define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type)                                                                        \
    sparseBackendForwardArgs.def_rw(#name, &torch_ext::SparseBackendForwardArgs::name);
#include "tensorrt_llm/thop/sparse_backend_forward_args_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD

    auto sparseRuntimeParams = nb::class_<torch_ext::SparseRuntimeParams>(m, "SparseRuntimeParams").def(nb::init<>());
#define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type)                                                                        \
    sparseRuntimeParams.def_rw(#name, &torch_ext::SparseRuntimeParams::name);
#include "tensorrt_llm/thop/sparse_runtime_params_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD

    auto attentionForwardArgs
        = nb::class_<torch_ext::AttentionForwardArgs>(m, "AttentionForwardArgs").def(nb::init<>());
#define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type)                                                                        \
    attentionForwardArgs.def_rw(#name, &torch_ext::AttentionForwardArgs::name);
#include "tensorrt_llm/thop/attention_forward_args_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD

    auto staticAttentionConfig
        = nb::class_<torch_ext::StaticAttentionConfig>(m, "StaticAttentionConfig").def(nb::init<>());
#define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type)                                                                        \
    staticAttentionConfig.def_rw(#name, &torch_ext::StaticAttentionConfig::name);
#include "tensorrt_llm/thop/static_attention_config_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD

    auto fmhaParams = nb::class_<torch_ext::FmhaParams>(m, "FmhaParams").def(nb::init<>());
#define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type) fmhaParams.def_rw(#name, &torch_ext::FmhaParams::name);
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD

    nb::class_<torch_ext::AttentionOp>(m, "AttentionOp",
        "One attention layer's native op. Reuse the same instance for every call from that layer so its "
        "kernel runners are built once; drop it when the layer's quantization config changes and on "
        "teardown, since it owns a cuBLAS handle, the runners, and the context-parallel communicator.")
        .def(nb::init<torch_ext::StaticAttentionConfig const&>(), nb::arg("config"))
        .def("run_context", &torch_ext::AttentionOp::runContext, nb::arg("params"), "Phased attention context pass.",
            nb::call_guard<nb::gil_scoped_release>())
        .def("run_generation", &torch_ext::AttentionOp::runGeneration, nb::arg("params"),
            "Phased attention generation pass.", nb::call_guard<nb::gil_scoped_release>())
        .def("run_mla_generation", &torch_ext::AttentionOp::runMlaGeneration, nb::arg("params"),
            "Phased attention MLA generation pass.", nb::call_guard<nb::gil_scoped_release>())
        .def("get_attention_workspace_size", &torch_ext::AttentionOp::getAttentionWorkspaceSize, nb::arg("params"),
            nb::arg("num_tokens"), nb::arg("max_attention_window_size"), nb::arg("num_gen_tokens"),
            nb::arg("max_blocks_per_sequence"), nb::arg("ctx_total_kv_len") = 0,
            "Max of the context/generation workspace byte requirements for sizing FmhaParams.workspace.",
            nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "get_helix_workspace_size_per_rank",
        [](int cp_size) { return tensorrt_llm::kernels::computeHelixWorkspaceSizePerRank(cp_size); },
        nb::arg("cp_size"), "Get helix all-to-all workspace size per rank in bytes");

    m.def("get_context_mla_workspace_bytes_per_token",
        &tensorrt_llm::torch_ext::AttentionOp::contextMlaWorkspaceBytesPerToken, nb::arg("num_attn_heads"),
        nb::arg("qk_rope_head_dim"), nb::arg("qk_nope_head_dim"), nb::arg("v_head_dim"), nb::arg("fp8_context_mla"),
        nb::arg("separate_q_and_kv_input"), nb::arg("sparse_mla"),
        "Per-token byte cost of the context-MLA K/V dequant staging buffers (scales with summed attended KV "
        "length). Returns 0 outside the fp8 context-MLA separate-Q/KV path. Used by the KV-cache estimator to "
        "reserve workspace headroom before sizing the KV pool.");

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

    m.def("trtllm_gen_context_preprocess", &torch_ext::trtllmGenContextPreprocess, nb::arg("qkv_input"),
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
        nb::arg("cross_attention") = false, "Fused nanobind context preprocess for trtllm-gen attention.",
        nb::call_guard<nb::gil_scoped_release>());

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
            int64_t batch_start, int64_t batch_size,
            at::ScalarType dtype) -> std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
        {
            auto [kvPool, kvScalePool] = torch_ext::buildFlashinferTrtllmGenPagedKvCacheBuffers(
                host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, layer_idx, num_kv_heads, tokens_per_block,
                head_dim, kv_factor, total_num_blocks, kv_cache_quant_mode, dtype);
            auto const mapping = torch_ext::readKvCachePoolMapping(host_kv_cache_pool_mapping, layer_idx);
            auto blockTables = kv_cache_block_offsets.select(0, mapping.poolIndex).narrow(0, batch_start, batch_size);
            return {std::move(kvPool), std::move(blockTables), std::move(kvScalePool)};
        },
        nb::arg("host_kv_cache_pool_pointers"), nb::arg("host_kv_cache_pool_mapping"),
        nb::arg("kv_cache_block_offsets"), nb::arg("layer_idx"), nb::arg("num_kv_heads"), nb::arg("tokens_per_block"),
        nb::arg("head_dim"), nb::arg("kv_factor"), nb::arg("total_num_blocks"), nb::arg("kv_cache_quant_mode"),
        nb::arg("batch_start"), nb::arg("batch_size"), nb::arg("dtype"),
        "Build flashinfer-style KV cache pool view and slice block tables for a given layer.",
        nb::call_guard<nb::gil_scoped_release>());

    m.def("trtllm_gen_generation_preprocess", &torch_ext::trtllmGenGenerationPreprocess, nb::arg("qkv_input"),
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
        "Fused nanobind generation preprocess for trtllm-gen attention.", nb::call_guard<nb::gil_scoped_release>());
}
} // namespace tensorrt_llm::nanobind::thop
