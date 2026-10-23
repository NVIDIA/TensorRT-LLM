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
#include <c10/cuda/CUDAGuard.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <tensorrt_llm/common/cudaUtils.h>
#include <tensorrt_llm/common/tllmDataType.h>
#include <tensorrt_llm/kernels/helixAllToAll.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <tensorrt_llm/thop/moe/communication/moeAlltoAllMeta.h>
#include <tensorrt_llm/thop/outputTensor.h>
#include <tensorrt_llm/thop/phasedFmha.h>
#include <tensorrt_llm/thop/trtllmGenFusedOps.h>
#include <torch/extension.h>
#include <utility>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::thop
{

namespace
{

bool fusedContextFmhaKernelExists(
    int headSize, tensorrt_llm::DataType kvCacheDtype, int tokensPerBlock, tensorrt_llm::DataType outputDtype)
{
    using tensorrt_llm::kernels::Data_type;

    // A fused kernel exists only for a power-of-two page size; the kernel
    // lookup asserts on other values, so reject them here rather than let that
    // assertion escape this diagnostic query. tokensPerBlock > 0 is guaranteed
    // before the bit test.
    if (headSize <= 0 || tokensPerBlock <= 0 || (tokensPerBlock & (tokensPerBlock - 1)) != 0)
    {
        return false;
    }

    // Probe ordinary dense, causal paged context with Q at the KV precision,
    // except for an NVFP4 KV cache, which is read by an FP8 Q kernel. thop
    // enables FP8 context computation for FP8 paged KV and attentionOp requires
    // it for FP4 KV. Output precision is independent and must be supplied
    // explicitly by the caller.
    Data_type dataType{};
    Data_type dataTypeKv{};
    Data_type dataTypeOut{};
    switch (kvCacheDtype)
    {
    case tensorrt_llm::DataType::kFP8: dataType = dataTypeKv = Data_type::DATA_TYPE_E4M3; break;
    case tensorrt_llm::DataType::kBF16: dataType = dataTypeKv = Data_type::DATA_TYPE_BF16; break;
    case tensorrt_llm::DataType::kHALF: dataType = dataTypeKv = Data_type::DATA_TYPE_FP16; break;
    case tensorrt_llm::DataType::kFP4:
        // Only the trtllm-gen kernel set has E2M1-KV context kernels. The
        // FMHA-v2 runner used everywhere else asserts that Q and KV precision
        // match, so answer "no" wherever the attention op would pick FMHA-v2
        // instead of tripping that assertion inside a diagnostic query. This
        // mirrors the op's trtllm-gen selection: SM100 family, and
        // head size 72 excluded (trtllm-gen has no kernel for it).
        if (!tensorrt_llm::common::isSM100Family() || headSize == 72)
        {
            return false;
        }
        dataType = Data_type::DATA_TYPE_E4M3;
        dataTypeKv = Data_type::DATA_TYPE_E2M1;
        break;
    default: return false;
    }

    switch (outputDtype)
    {
    case tensorrt_llm::DataType::kBF16: dataTypeOut = Data_type::DATA_TYPE_BF16; break;
    case tensorrt_llm::DataType::kHALF: dataTypeOut = Data_type::DATA_TYPE_FP16; break;
    case tensorrt_llm::DataType::kFP8: dataTypeOut = Data_type::DATA_TYPE_E4M3; break;
    case tensorrt_llm::DataType::kFP4: dataTypeOut = Data_type::DATA_TYPE_E2M1; break;
    default: return false;
    }

    tensorrt_llm::kernels::MHARunnerFixedParams params{};
    params.dataType = dataType;
    params.dataTypeKv = dataTypeKv;
    params.dataTypeOut = dataTypeOut;
    params.forceFp32Acc = false;
    params.attentionMaskType = tensorrt_llm::kernels::ContextAttentionMaskType::CAUSAL;
    // The paged-context FMHA path: Q plus a paged KV cache, i.e. the layout
    // attentionOp selects when mPagedKVCache && mPagedContextFMHA.
    params.attentionInputLayout = tensorrt_llm::kernels::AttentionInputLayout::Q_PAGED_KV;
    params.isSPadded = false;
    params.numQHeads = 1;
    params.numKvHeads = 1;
    params.numTokensPerBlock = tokensPerBlock;
    params.headSize = headSize;
    params.headSizeV = headSize;
    params.qScaling = 1.f;
    params.attnLogitSoftcappingScale = 0.f;
    params.hasAlibi = false;
    params.scaleAlibi = false;
    params.saveSoftmax = false;

    // The library the attention op picks: trtllm-gen on the SM100 family except head size 72, fmha_v2 otherwise.
    if (tensorrt_llm::common::isSM100Family() && headSize != 72)
    {
        return torch_ext::TrtllmGenFmha<torch_ext::MHA>::isContextSupported(params);
    }
    return torch_ext::FmhaV2Fmha<torch_ext::MHA>::isContextSupported(params);
}

// Readers for the cached native parameters, as the measured prototype spelled them.
inline nb::object attr(nb::handle o, char const* name)
{
    return o.attr(name);
}

template <typename T>
T get(nb::handle o, char const* name)
{
    return nb::cast<T>(o.attr(name));
}

template <typename T>
std::optional<T> opt(nb::handle o, char const* name)
{
    nb::object const v = o.attr(name);
    return v.is_none() ? std::nullopt : std::optional<T>(nb::cast<T>(v));
}

#include "tensorrt_llm/thop/fmha_params_update.inc"

/// Fill the native parameters from the Python FmhaParams: the generated field copy, then
/// the two reads the schema cannot express as one.
void prepare(torch_ext::FmhaParams& params, nb::handle source)
{
    update(params, source);
    if (!params.context_lengths.has_value())
    {
        // Direct phase callers may omit the prompt-length view.
        params.context_lengths
            = get<torch::Tensor>(attr(source, "meta"), "prompt_lens_cuda_runtime").slice(0, params.seq_offset);
    }
    // Encoder KV is request-scoped; each decoder beam reads it independently.
    params.num_requests = params.meta.is_cross ? params.batch_size : params.num_requests;
}

// Each phase entry builds the native parameters for this call on the stack, so every
// tensor reference is released when the call returns, on success or on a throw.
void runContext(nb::handle source)
{
    torch_ext::FmhaParams params{};
    prepare(params, source);
    auto const key = torch_ext::AttentionBase::getCacheKey(params);

    nb::gil_scoped_release const gil;
    c10::cuda::CUDAGuard const device(params.getQueryInput().device());
    torch_ext::AttentionBase::getCached(key)->runContext(params);
}

void runGeneration(nb::handle source)
{
    torch_ext::FmhaParams params{};
    prepare(params, source);
    auto const key = torch_ext::AttentionBase::getCacheKey(params);

    nb::gil_scoped_release const gil;
    c10::cuda::CUDAGuard const device(params.getQueryInput().device());
    torch_ext::AttentionBase::getCached(key)->runGeneration(params);
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

    m.def("run_context", &runContext, nb::arg("params"),
        "Phased attention context pass. Reads the Python FmhaParams into native parameters, resolves the op "
        "for the layer's configuration from a per-thread cache (building it outside CUDA graph capture on "
        "first use), and launches.");
    m.def("run_generation", &runGeneration, nb::arg("params"), "Phased attention generation pass, MLA or not.");

    m.def(
        "get_helix_workspace_size_per_rank",
        [](int cp_size) { return tensorrt_llm::kernels::computeHelixWorkspaceSizePerRank(cp_size); },
        nb::arg("cp_size"), "Get helix all-to-all workspace size per rank in bytes");

    m.def("get_context_mla_workspace_bytes_per_token", &tensorrt_llm::torch_ext::contextMlaWorkspaceBytesPerToken,
        nb::arg("num_attn_heads"), nb::arg("qk_rope_head_dim"), nb::arg("qk_nope_head_dim"), nb::arg("v_head_dim"),
        nb::arg("fp8_context_mla"), nb::arg("separate_q_and_kv_input"), nb::arg("sparse_mla"),
        "Per-token byte cost of the context-MLA K/V dequant staging buffers (scales with summed attended KV "
        "length). Returns 0 outside the fp8 context-MLA separate-Q/KV path. Used by the KV-cache estimator to "
        "reserve workspace headroom before sizing the KV pool.");

    m.def("fused_context_fmha_kernel_exists", &fusedContextFmhaKernelExists, nb::arg("head_size"),
        nb::arg("kv_cache_dtype"), nb::arg("tokens_per_block"), nb::arg("output_dtype"),
        "Whether this build has a fused paged-context FMHA kernel for (this device's SM, head_size, KV cache dtype, "
        "tokens per block, output dtype), with Q at the KV precision (FP8 Q for an NVFP4 KV cache, which only the "
        "SM100-family kernel set carries), causal masking, and equal Q/KV heads. "
        "This is a diagnostic for a fixed probe configuration and does not describe what a given model will run; "
        "The attention op checks its own exact initialized configuration before allowing paged-context attention. "
        "A build whose --cuda_architectures omits this device's SM carries no kernels for it and reports false.",
        nb::call_guard<nb::gil_scoped_release>());

    m.def("compute_flash_mla_metadata", &tensorrt_llm::computeFlashMlaMetadata, nb::arg("seqlens_k"),
        nb::arg("tile_scheduler_metadata"), nb::arg("num_splits"), nb::arg("batch_size"), nb::arg("s_q"),
        nb::arg("num_q_heads"), nb::arg("num_kv_heads"), nb::arg("head_size_v"),
        "Compute FlashMLA tile-scheduler metadata in-place. Call once per forward pass before attention layers.",
        nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "get_trtllm_gen_context_workspace_layout",
        [](at::ScalarType dtype, int64_t batch_size, int64_t num_tokens, int64_t num_heads, int64_t head_size,
            int64_t rotary_embedding_dim, bool separate_q_kv_input, bool fp8_context_fmha, bool skip_fmha_workspace)
        {
            auto const layout = torch_ext::TrtllmAttentionWorkspaceManager::buildContextLayout(dtype, batch_size,
                num_tokens, num_heads, head_size, rotary_embedding_dim, separate_q_kv_input, fp8_context_fmha,
                skip_fmha_workspace);
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
        nb::arg("skip_fmha_workspace") = false, "Return the C++ trtllm-gen context workspace layout.");

    m.def(
        "get_trtllm_gen_generation_workspace_layout",
        [](at::ScalarType dtype, int64_t batch_beam, int64_t num_tokens, int64_t num_heads, int64_t head_size,
            int64_t rotary_embedding_dim, int64_t num_kv_heads, int64_t max_blocks_per_sequence,
            bool use_sparse_attention, bool skip_fmha_workspace)
        {
            auto const layout = torch_ext::TrtllmAttentionWorkspaceManager::buildGenerationLayout(dtype, batch_beam,
                num_tokens, num_heads, head_size, rotary_embedding_dim, num_kv_heads, max_blocks_per_sequence,
                use_sparse_attention, skip_fmha_workspace);
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
        nb::arg("use_sparse_attention") = false, nb::arg("skip_fmha_workspace") = false,
        "Return the C++ trtllm-gen generation workspace layout.");

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
        nb::arg("cross_attention") = false, nb::arg("skip_fmha_workspace") = false,
        "Fused nanobind context preprocess for trtllm-gen attention.", nb::call_guard<nb::gil_scoped_release>());

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
        nb::arg("attention_chunk_size"), nb::arg("multi_processor_count"), nb::arg("skip_fmha_workspace") = false,
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
        nb::arg("skip_fmha_workspace") = false, "Fused nanobind generation preprocess for trtllm-gen attention.",
        nb::call_guard<nb::gil_scoped_release>());
}
} // namespace tensorrt_llm::nanobind::thop
