/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/attentionOp.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

using tensorrt_llm::common::op::AttentionOp;
using tensorrt_llm::kernels::AttentionMaskType;
using tensorrt_llm::kernels::BlockSparseParams;
using tensorrt_llm::kernels::BuildDecoderInfoParams;
using tensorrt_llm::kernels::KVBlockArray;
using tensorrt_llm::kernels::KvCacheDataType;
using tensorrt_llm::kernels::PositionEmbeddingType;
using tensorrt_llm::kernels::QKVPreprocessingParams;
using tensorrt_llm::kernels::RotaryScalingType;
using tensorrt_llm::kernels::MlaMetaParams;
using tensorrt_llm::kernels::MlaParams;
using tensorrt_llm::kernels::cacheTypeFromQuantMode;
using tensorrt_llm::runtime::TorchUtils;

namespace
{

template <typename T, typename OptTensorT>
T* optPtr(OptTensorT&& t, std::enable_if_t<!std::is_const_v<OptTensorT>>* = nullptr)
{
    return t.has_value() ? static_cast<T*>(t->data_ptr()) : nullptr;
}

template <typename T, typename OptTensorT>
T const* optPtr(OptTensorT&& t, std::enable_if_t<std::is_const_v<OptTensorT>>* = nullptr)
{
    return t.has_value() ? static_cast<T const*>(t->data_ptr()) : nullptr;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// build_decoder_info
// ---------------------------------------------------------------------------

template <typename AttentionMaskDataType>
bool invokeBuildDecoderInfoTyped(BuildDecoderInfoParams<AttentionMaskDataType>& params, cudaStream_t stream)
{
    if (!params.isBuildDecoderInfoKernelNeeded())
    {
        return false;
    }
    tensorrt_llm::kernels::invokeBuildDecoderInfo(params, stream);
    sync_check_cuda_error(stream);
    return true;
}

bool build_decoder_info(
    // Tensors
    std::optional<torch::Tensor> seq_q_offsets, std::optional<torch::Tensor> seq_kv_offsets,
    std::optional<torch::Tensor> padding_offsets, std::optional<torch::Tensor> tokens_info,
    std::optional<torch::Tensor> encoder_padding_offsets, std::optional<torch::Tensor> packed_mask_row_offsets,
    std::optional<torch::Tensor> seq_cp_partial_offsets, std::optional<torch::Tensor> attention_mask,
    std::optional<torch::Tensor> seq_q_lengths, std::optional<torch::Tensor> seq_kv_lengths,
    std::optional<torch::Tensor> fmha_tile_counter, std::optional<torch::Tensor> dequant_scale_qkv,
    std::optional<torch::Tensor> quant_scale_o, std::optional<torch::Tensor> fmha_bmm1_scale,
    std::optional<torch::Tensor> fmha_bmm2_scale, std::optional<torch::Tensor> rotary_embedding_inv_freq,
    std::optional<torch::Tensor> rotary_embedding_inv_freq_cache,
    // Scalars
    int64_t cp_size, bool separate_qkv_scales, double fmha_host_bmm1_scale, int64_t batch_size,
    int64_t max_q_seq_length, int64_t max_encoder_q_seq_length, int64_t attention_window_size,
    int64_t sink_token_length, int64_t num_tokens, bool remove_padding, int64_t attention_mask_type,
    double rotary_embedding_scale, double rotary_embedding_base, int64_t rotary_embedding_dim,
    int64_t rotary_scaling_type, int64_t rotary_embedding_max_positions)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    BuildDecoderInfoParams<void> p{};
    p.seqQOffsets = optPtr<int>(seq_q_offsets);
    p.seqKVOffsets = optPtr<int>(seq_kv_offsets);
    p.paddingOffsets = optPtr<int>(padding_offsets);
    p.tokensInfo = optPtr<int2>(tokens_info);
    p.encoderPaddingOffsets = optPtr<int>(encoder_padding_offsets);
    p.packedMaskRowOffsets = optPtr<int>(packed_mask_row_offsets);
    p.seqCpPartialOffsets = optPtr<int>(seq_cp_partial_offsets);
    p.seqQLengths = optPtr<int>(seq_q_lengths);
    p.seqKVLengths = optPtr<int>(seq_kv_lengths);
    p.cpSize = static_cast<int>(cp_size);
    p.fmhaTileCounter = optPtr<uint32_t>(fmha_tile_counter);
    p.dequantScaleQkv = optPtr<float>(dequant_scale_qkv);
    p.separateQkvScales = separate_qkv_scales;
    p.quantScaleO = optPtr<float>(quant_scale_o);
    p.fmhaHostBmm1Scale = static_cast<float>(fmha_host_bmm1_scale);
    p.fmhaBmm1Scale = optPtr<float>(fmha_bmm1_scale);
    p.fmhaBmm2Scale = optPtr<float>(fmha_bmm2_scale);
    p.batchSize = static_cast<int>(batch_size);
    p.maxQSeqLength = static_cast<int>(max_q_seq_length);
    p.maxEncoderQSeqLength = static_cast<int>(max_encoder_q_seq_length);
    p.attentionWindowSize = static_cast<int>(attention_window_size);
    p.sinkTokenLength = static_cast<int>(sink_token_length);
    p.numTokens = static_cast<int>(num_tokens);
    p.removePadding = remove_padding;
    p.attentionMaskType = static_cast<AttentionMaskType>(attention_mask_type);
    p.blockSparseParams = BlockSparseParams{};
    p.rotaryEmbeddingScale = static_cast<float>(rotary_embedding_scale);
    p.rotaryEmbeddingBase = static_cast<float>(rotary_embedding_base);
    p.rotaryEmbeddingDim = static_cast<int>(rotary_embedding_dim);
    p.rotaryScalingType = static_cast<RotaryScalingType>(rotary_scaling_type);
    p.rotaryEmbeddingInvFreq = optPtr<float>(rotary_embedding_inv_freq);
    p.rotaryEmbeddingInvFreqCache = optPtr<float>(rotary_embedding_inv_freq_cache);
    p.rotaryEmbeddingCoeffCache = nullptr;
    p.rotaryEmbeddingMaxPositions = static_cast<int>(rotary_embedding_max_positions);
    p.attentionMask = attention_mask.has_value() ? attention_mask->data_ptr() : nullptr;

    if (attention_mask.has_value())
    {
        auto dtype = TorchUtils::dataType(attention_mask->scalar_type());
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
            return invokeBuildDecoderInfoTyped(reinterpret_cast<BuildDecoderInfoParams<float>&>(p), stream);
        case nvinfer1::DataType::kHALF:
            return invokeBuildDecoderInfoTyped(reinterpret_cast<BuildDecoderInfoParams<half>&>(p), stream);
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16:
            return invokeBuildDecoderInfoTyped(reinterpret_cast<BuildDecoderInfoParams<__nv_bfloat16>&>(p), stream);
#endif
#ifdef ENABLE_FP8
        case nvinfer1::DataType::kFP8:
            return invokeBuildDecoderInfoTyped(reinterpret_cast<BuildDecoderInfoParams<__nv_fp8_e4m3>&>(p), stream);
#endif
        default: TORCH_CHECK(false, "Unsupported attention mask dtype");
        }
    }
    else
    {
        return invokeBuildDecoderInfoTyped(reinterpret_cast<BuildDecoderInfoParams<float>&>(p), stream);
    }
}

// ---------------------------------------------------------------------------
// qkv_preprocessing / kv_cache_postprocessing (shared implementation)
// ---------------------------------------------------------------------------

template <typename T, bool isPreprocessing>
void dispatchQkvProcessing(QKVPreprocessingParams<T, KVBlockArray> params, cudaStream_t stream)
{
    if constexpr (isPreprocessing)
    {
        tensorrt_llm::kernels::invokeQKVPreprocessing(params, stream);
    }
    else
    {
        tensorrt_llm::kernels::invokeKvCachePostprocessing(params, stream);
    }
    sync_check_cuda_error(stream);
}

template <bool isPreprocessing>
void qkv_processing(
    // Tensors
    std::optional<torch::Tensor> qkv_input, std::optional<torch::Tensor> cross_kv_input,
    std::optional<torch::Tensor> quantized_qkv_output, std::optional<torch::Tensor> q_output,
    std::optional<torch::Tensor> kv_cache_block_offsets, std::optional<torch::Tensor> host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> host_kv_cache_pool_mapping, std::optional<torch::Tensor> qkv_bias,
    std::optional<torch::Tensor> qkv_scale_quant_orig, std::optional<torch::Tensor> qkv_scale_orig_quant,
    std::optional<torch::Tensor> o_scale_orig_quant, std::optional<torch::Tensor> fmha_bmm1_scale,
    std::optional<torch::Tensor> fmha_bmm2_scale, std::optional<torch::Tensor> fmha_tile_counter,
    std::optional<torch::Tensor> logn_scaling, std::optional<torch::Tensor> tokens_info,
    std::optional<torch::Tensor> seq_lens, std::optional<torch::Tensor> cache_seq_lens,
    std::optional<torch::Tensor> encoder_seq_lens, std::optional<torch::Tensor> cu_seq_lens,
    std::optional<torch::Tensor> cu_kv_seq_lens, std::optional<torch::Tensor> sparse_kv_offsets,
    std::optional<torch::Tensor> sparse_kv_indices, std::optional<torch::Tensor> rotary_embedding_inv_freq,
    std::optional<torch::Tensor> rotary_coef_cache_buffer, std::optional<torch::Tensor> spec_decoding_position_offsets,
    std::optional<torch::Tensor> mrope_rotary_cos_sin, std::optional<torch::Tensor> mrope_position_deltas,
    // Scalars
    int64_t batch_size, int64_t max_input_seq_len, int64_t max_kv_seq_len, int64_t cyclic_kv_cache_len,
    int64_t sink_token_len, int64_t token_num, bool remove_padding, bool is_last_chunk, bool cross_attention,
    int64_t head_num, int64_t kv_head_num, int64_t qheads_per_kv_head, int64_t size_per_head,
    double fmha_host_bmm1_scale, int64_t rotary_embedding_dim, double rotary_embedding_base,
    int64_t rotary_scaling_type, double rotary_embedding_scale, int64_t rotary_embedding_max_positions,
    int64_t position_embedding_type, bool position_shift_enabled, bool separate_q_kv_output, bool quantized_fp8_output,
    bool generation_phase, int64_t rotary_vision_start, int64_t rotary_vision_length,
    // Extra args (for KV cache computation)
    int64_t layer_idx, int64_t tokens_per_block, int64_t max_attention_window_size, int64_t kv_cache_quant_mode,
    int64_t cyclic_attention_window_size, int64_t beam_width, int64_t sink_token_length, int64_t seq_offset,
    bool is_mla_enable)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(qkv_input.has_value(), "qkv_input is required");
    auto qkvDtype = TorchUtils::dataType(qkv_input->scalar_type());

    auto quantMode = tensorrt_llm::common::QuantMode(static_cast<uint32_t>(kv_cache_quant_mode));
    auto kvArrays = buildPagedKvCacheBuffers(kv_cache_block_offsets, host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping, quantMode, layer_idx, batch_size, tokens_per_block, kv_head_num, size_per_head,
        cyclic_attention_window_size, max_attention_window_size, sink_token_length, beam_width, seq_offset,
        is_mla_enable, static_cast<size_t>(qkv_input->element_size()));

    QKVPreprocessingParams<void, KVBlockArray> p{};
    p.qkv_input = qkv_input.has_value() ? qkv_input->data_ptr() : nullptr;
    p.cross_kv_input = cross_kv_input.has_value() ? cross_kv_input->data_ptr() : nullptr;
    p.quantized_qkv_output = optPtr<void>(quantized_qkv_output);
    p.q_output = q_output.has_value() ? q_output->data_ptr() : nullptr;
    p.kv_cache_buffer = kvArrays.kvCacheBuffer;
    p.kv_cache_block_scales_buffer = kvArrays.kvScaleCacheBuffer;
    p.qkv_bias = qkv_bias.has_value() ? static_cast<void const*>(qkv_bias->data_ptr()) : nullptr;
    p.qkv_scale_quant_orig = optPtr<float>(qkv_scale_quant_orig);
    p.qkv_scale_orig_quant = optPtr<float>(qkv_scale_orig_quant);
    p.o_scale_orig_quant = optPtr<float>(o_scale_orig_quant);
    p.fmha_bmm1_scale = optPtr<float>(fmha_bmm1_scale);
    p.fmha_bmm2_scale = optPtr<float>(fmha_bmm2_scale);
    p.fmha_tile_counter = optPtr<float>(fmha_tile_counter);
    p.logn_scaling = optPtr<float>(logn_scaling);
    p.tokens_info = optPtr<int2>(tokens_info);
    p.seq_lens = optPtr<int>(seq_lens);
    p.cache_seq_lens = optPtr<int>(cache_seq_lens);
    p.encoder_seq_lens = optPtr<int>(encoder_seq_lens);
    p.cu_seq_lens = optPtr<int>(cu_seq_lens);
    p.cu_kv_seq_lens = optPtr<int>(cu_kv_seq_lens);
    p.sparse_kv_offsets = optPtr<int>(sparse_kv_offsets);
    p.sparse_kv_indices = optPtr<int>(sparse_kv_indices);
    p.rotary_embedding_inv_freq = optPtr<float>(rotary_embedding_inv_freq);
    p.rotary_coef_cache_buffer = optPtr<float2>(rotary_coef_cache_buffer);
    p.spec_decoding_position_offsets = optPtr<int>(spec_decoding_position_offsets);
    p.mrope_rotary_cos_sin = optPtr<float2>(mrope_rotary_cos_sin);
    p.mrope_position_deltas = optPtr<int32_t>(mrope_position_deltas);
    p.batch_size = static_cast<int>(batch_size);
    p.max_input_seq_len = static_cast<int>(max_input_seq_len);
    p.max_kv_seq_len = static_cast<int>(max_kv_seq_len);
    p.cyclic_kv_cache_len = static_cast<int>(cyclic_kv_cache_len);
    p.sink_token_len = static_cast<int>(sink_token_len);
    p.token_num = static_cast<int>(token_num);
    p.remove_padding = remove_padding;
    p.is_last_chunk = is_last_chunk;
    p.cross_attention = cross_attention;
    p.head_num = static_cast<int>(head_num);
    p.kv_head_num = static_cast<int>(kv_head_num);
    p.qheads_per_kv_head = static_cast<int>(qheads_per_kv_head);
    p.size_per_head = static_cast<int>(size_per_head);
    p.fmha_host_bmm1_scale = static_cast<float>(fmha_host_bmm1_scale);
    p.rotary_embedding_dim = static_cast<int>(rotary_embedding_dim);
    p.rotary_embedding_base = static_cast<float>(rotary_embedding_base);
    p.rotary_scale_type = static_cast<RotaryScalingType>(rotary_scaling_type);
    p.rotary_embedding_scale = static_cast<float>(rotary_embedding_scale);
    p.rotary_embedding_max_positions = static_cast<int>(rotary_embedding_max_positions);
    p.position_embedding_type = static_cast<PositionEmbeddingType>(position_embedding_type);
    p.position_shift_enabled = position_shift_enabled;
    p.cache_type = cacheTypeFromQuantMode(quantMode);
    p.separate_q_kv_output = separate_q_kv_output;
    p.quantized_fp8_output = quantized_fp8_output;
    p.generation_phase = generation_phase;
    p.multi_processor_count = tensorrt_llm::common::getMultiProcessorCount();
    p.rotary_vision_start = static_cast<int>(rotary_vision_start);
    p.rotary_vision_length = static_cast<int>(rotary_vision_length);

    switch (qkvDtype)
    {
    case nvinfer1::DataType::kFLOAT:
        dispatchQkvProcessing<float, isPreprocessing>(
            reinterpret_cast<QKVPreprocessingParams<float, KVBlockArray>&>(p), stream);
        break;
    case nvinfer1::DataType::kHALF:
        dispatchQkvProcessing<half, isPreprocessing>(
            reinterpret_cast<QKVPreprocessingParams<half, KVBlockArray>&>(p), stream);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        dispatchQkvProcessing<__nv_bfloat16, isPreprocessing>(
            reinterpret_cast<QKVPreprocessingParams<__nv_bfloat16, KVBlockArray>&>(p), stream);
        break;
#endif
    default: TORCH_CHECK(false, "Unsupported data type for QKV processing.");
    }
}

// ---------------------------------------------------------------------------
// MLA context operations (mla_rope_context / mla_context_fp8_quantize)
// ---------------------------------------------------------------------------

void mla_rope_context(
    // MlaParams tensor fields
    std::optional<torch::Tensor> latent_cache, std::optional<torch::Tensor> q_buf, std::optional<torch::Tensor> k_buf,
    std::optional<torch::Tensor> v_buf, std::optional<torch::Tensor> quant_q_buf,
    std::optional<torch::Tensor> quant_k_buf, std::optional<torch::Tensor> quant_v_buf,
    std::optional<torch::Tensor> context_buf, std::optional<torch::Tensor> q_pe,
    std::optional<torch::Tensor> cos_sin_cache, std::optional<torch::Tensor> workspace,
    std::optional<torch::Tensor> cache_seq_lens, std::optional<torch::Tensor> seq_q_offset,
    std::optional<torch::Tensor> fmha_tile_counter, std::optional<torch::Tensor> cu_q_seqlens,
    std::optional<torch::Tensor> cu_kv_seqlens, std::optional<torch::Tensor> block_ids_per_seq,
    std::optional<torch::Tensor> bmm1_scale, std::optional<torch::Tensor> bmm2_scale,
    std::optional<torch::Tensor> quant_scale_o, std::optional<torch::Tensor> quant_scale_q,
    std::optional<torch::Tensor> quant_scale_kv, std::optional<torch::Tensor> dequant_scale_q,
    std::optional<torch::Tensor> dequant_scale_kv, std::optional<torch::Tensor> quant_scale_qkv,
    std::optional<torch::Tensor> helix_position_offsets, std::optional<torch::Tensor> helix_is_inactive_rank,
    // MlaParams scalar fields
    int64_t batch_size, int64_t acc_q_len, int64_t head_num, int64_t max_input_seq_len, int64_t q_pe_ld,
    int64_t q_pe_stride, int64_t q_lora_rank, int64_t kv_lora_rank, int64_t qk_nope_head_dim, int64_t qk_rope_head_dim,
    int64_t v_head_dim, int64_t predicted_tokens_per_seq, int64_t num_layers, double host_bmm1_scale,
    bool absorption_mode,
    // KV cache args
    std::optional<torch::Tensor> kv_cache_block_offsets, std::optional<torch::Tensor> host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> host_kv_cache_pool_mapping, int64_t layer_idx, int64_t tokens_per_block,
    int64_t kv_head_num, int64_t size_per_head, int64_t kv_cache_quant_mode, int64_t cyclic_attention_window_size,
    int64_t max_attention_window_size, int64_t sink_token_length, int64_t beam_width, int64_t seq_offset)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(q_buf.has_value(), "q_buf is required for mla_rope_context");
    auto dtype = TorchUtils::dataType(q_buf->scalar_type());

    auto quantMode = tensorrt_llm::common::QuantMode(static_cast<uint32_t>(kv_cache_quant_mode));
    auto kvArrays = buildPagedKvCacheBuffers(kv_cache_block_offsets, host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping, quantMode, layer_idx, batch_size, tokens_per_block, kv_head_num, size_per_head,
        cyclic_attention_window_size, max_attention_window_size, sink_token_length, beam_width, seq_offset,
        /*is_mla_enable=*/true, static_cast<size_t>(q_buf->element_size()));

    MlaParams<void> p{};
    p.latent_cache = latent_cache.has_value() ? latent_cache->data_ptr() : nullptr;
    p.q_buf = q_buf.has_value() ? q_buf->data_ptr() : nullptr;
    p.k_buf = k_buf.has_value() ? k_buf->data_ptr() : nullptr;
    p.v_buf = v_buf.has_value() ? v_buf->data_ptr() : nullptr;
    p.quant_q_buf = quant_q_buf.has_value() ? quant_q_buf->data_ptr() : nullptr;
    p.quant_k_buf = quant_k_buf.has_value() ? quant_k_buf->data_ptr() : nullptr;
    p.quant_v_buf = quant_v_buf.has_value() ? quant_v_buf->data_ptr() : nullptr;
    p.context_buf = context_buf.has_value() ? context_buf->data_ptr() : nullptr;
    p.q_pe = q_pe.has_value() ? q_pe->data_ptr() : nullptr;
    p.cos_sin_cache = optPtr<float2>(cos_sin_cache);
    p.batch_size = static_cast<int32_t>(batch_size);
    p.acc_q_len = static_cast<int32_t>(acc_q_len);
    p.head_num = static_cast<int32_t>(head_num);
    p.workspace = workspace.has_value() ? workspace->data_ptr() : nullptr;
    p.cache_seq_lens = optPtr<int32_t>(cache_seq_lens);
    p.seqQOffset = optPtr<int>(seq_q_offset);
    p.fmha_tile_counter = optPtr<uint32_t>(fmha_tile_counter);
    p.max_input_seq_len = static_cast<int32_t>(max_input_seq_len);
    p.cu_q_seqlens = optPtr<int>(cu_q_seqlens);
    p.cu_kv_seqlens = optPtr<int>(cu_kv_seqlens);
    p.q_pe_ld = static_cast<int32_t>(q_pe_ld);
    p.q_pe_stride = static_cast<int32_t>(q_pe_stride);
    p.meta.q_lora_rank = static_cast<int32_t>(q_lora_rank);
    p.meta.kv_lora_rank = static_cast<int32_t>(kv_lora_rank);
    p.meta.qk_nope_head_dim = static_cast<int32_t>(qk_nope_head_dim);
    p.meta.qk_rope_head_dim = static_cast<int32_t>(qk_rope_head_dim);
    p.meta.v_head_dim = static_cast<int32_t>(v_head_dim);
    p.meta.predicted_tokens_per_seq = static_cast<int32_t>(predicted_tokens_per_seq);
    p.meta.num_layers = static_cast<int32_t>(num_layers);
    p.block_ids_per_seq = optPtr<int>(block_ids_per_seq);
    p.cache_type = cacheTypeFromQuantMode(quantMode);
    p.bmm1_scale = optPtr<float>(bmm1_scale);
    p.bmm2_scale = optPtr<float>(bmm2_scale);
    p.quant_scale_o = optPtr<float>(quant_scale_o);
    p.quant_scale_q = optPtr<float>(quant_scale_q);
    p.quant_scale_kv = optPtr<float>(quant_scale_kv);
    p.dequant_scale_q = optPtr<float>(dequant_scale_q);
    p.dequant_scale_kv = optPtr<float>(dequant_scale_kv);
    p.host_bmm1_scale = static_cast<float>(host_bmm1_scale);
    p.absorption_mode = absorption_mode;
    p.quant_scale_qkv = optPtr<float>(quant_scale_qkv);
    p.helix_position_offsets = optPtr<int32_t>(helix_position_offsets);
    p.helix_is_inactive_rank
        = helix_is_inactive_rank.has_value() ? static_cast<bool const*>(helix_is_inactive_rank->data_ptr()) : nullptr;

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        tensorrt_llm::kernels::invokeMLARopeContext(
            reinterpret_cast<MlaParams<float>&>(p), kvArrays.kvCacheBuffer, stream);
        break;
    case nvinfer1::DataType::kHALF:
        tensorrt_llm::kernels::invokeMLARopeContext(
            reinterpret_cast<MlaParams<half>&>(p), kvArrays.kvCacheBuffer, stream);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        tensorrt_llm::kernels::invokeMLARopeContext(
            reinterpret_cast<MlaParams<__nv_bfloat16>&>(p), kvArrays.kvCacheBuffer, stream);
        break;
#endif
    default: TORCH_CHECK(false, "Unsupported data type for MLA RoPE context.");
    }
    sync_check_cuda_error(stream);
}

void mla_context_fp8_quantize(
    // MlaParams tensor fields
    std::optional<torch::Tensor> latent_cache, std::optional<torch::Tensor> q_buf, std::optional<torch::Tensor> k_buf,
    std::optional<torch::Tensor> v_buf, std::optional<torch::Tensor> quant_q_buf,
    std::optional<torch::Tensor> quant_k_buf, std::optional<torch::Tensor> quant_v_buf,
    std::optional<torch::Tensor> context_buf, std::optional<torch::Tensor> q_pe,
    std::optional<torch::Tensor> cos_sin_cache, std::optional<torch::Tensor> workspace,
    std::optional<torch::Tensor> cache_seq_lens, std::optional<torch::Tensor> seq_q_offset,
    std::optional<torch::Tensor> fmha_tile_counter, std::optional<torch::Tensor> cu_q_seqlens,
    std::optional<torch::Tensor> cu_kv_seqlens, std::optional<torch::Tensor> block_ids_per_seq,
    std::optional<torch::Tensor> bmm1_scale, std::optional<torch::Tensor> bmm2_scale,
    std::optional<torch::Tensor> quant_scale_o, std::optional<torch::Tensor> quant_scale_q,
    std::optional<torch::Tensor> quant_scale_kv, std::optional<torch::Tensor> dequant_scale_q,
    std::optional<torch::Tensor> dequant_scale_kv, std::optional<torch::Tensor> quant_scale_qkv,
    std::optional<torch::Tensor> helix_position_offsets, std::optional<torch::Tensor> helix_is_inactive_rank,
    // MlaParams scalar fields
    int64_t batch_size, int64_t acc_q_len, int64_t head_num, int64_t max_input_seq_len, int64_t q_pe_ld,
    int64_t q_pe_stride, int64_t q_lora_rank, int64_t kv_lora_rank, int64_t qk_nope_head_dim, int64_t qk_rope_head_dim,
    int64_t v_head_dim, int64_t predicted_tokens_per_seq, int64_t num_layers, double host_bmm1_scale,
    bool absorption_mode,
    // Extra args
    int64_t total_kv_len)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(q_buf.has_value(), "q_buf is required for mla_context_fp8_quantize");
    auto dtype = TorchUtils::dataType(q_buf->scalar_type());

    MlaParams<void> p{};
    p.latent_cache = latent_cache.has_value() ? latent_cache->data_ptr() : nullptr;
    p.q_buf = q_buf.has_value() ? q_buf->data_ptr() : nullptr;
    p.k_buf = k_buf.has_value() ? k_buf->data_ptr() : nullptr;
    p.v_buf = v_buf.has_value() ? v_buf->data_ptr() : nullptr;
    p.quant_q_buf = quant_q_buf.has_value() ? quant_q_buf->data_ptr() : nullptr;
    p.quant_k_buf = quant_k_buf.has_value() ? quant_k_buf->data_ptr() : nullptr;
    p.quant_v_buf = quant_v_buf.has_value() ? quant_v_buf->data_ptr() : nullptr;
    p.context_buf = context_buf.has_value() ? context_buf->data_ptr() : nullptr;
    p.q_pe = q_pe.has_value() ? q_pe->data_ptr() : nullptr;
    p.cos_sin_cache = optPtr<float2>(cos_sin_cache);
    p.batch_size = static_cast<int32_t>(batch_size);
    p.acc_q_len = static_cast<int32_t>(acc_q_len);
    p.head_num = static_cast<int32_t>(head_num);
    p.workspace = workspace.has_value() ? workspace->data_ptr() : nullptr;
    p.cache_seq_lens = optPtr<int32_t>(cache_seq_lens);
    p.seqQOffset = optPtr<int>(seq_q_offset);
    p.fmha_tile_counter = optPtr<uint32_t>(fmha_tile_counter);
    p.max_input_seq_len = static_cast<int32_t>(max_input_seq_len);
    p.cu_q_seqlens = optPtr<int>(cu_q_seqlens);
    p.cu_kv_seqlens = optPtr<int>(cu_kv_seqlens);
    p.q_pe_ld = static_cast<int32_t>(q_pe_ld);
    p.q_pe_stride = static_cast<int32_t>(q_pe_stride);
    p.meta.q_lora_rank = static_cast<int32_t>(q_lora_rank);
    p.meta.kv_lora_rank = static_cast<int32_t>(kv_lora_rank);
    p.meta.qk_nope_head_dim = static_cast<int32_t>(qk_nope_head_dim);
    p.meta.qk_rope_head_dim = static_cast<int32_t>(qk_rope_head_dim);
    p.meta.v_head_dim = static_cast<int32_t>(v_head_dim);
    p.meta.predicted_tokens_per_seq = static_cast<int32_t>(predicted_tokens_per_seq);
    p.meta.num_layers = static_cast<int32_t>(num_layers);
    p.block_ids_per_seq = optPtr<int>(block_ids_per_seq);
    p.cache_type = KvCacheDataType::FP8;
    p.bmm1_scale = optPtr<float>(bmm1_scale);
    p.bmm2_scale = optPtr<float>(bmm2_scale);
    p.quant_scale_o = optPtr<float>(quant_scale_o);
    p.quant_scale_q = optPtr<float>(quant_scale_q);
    p.quant_scale_kv = optPtr<float>(quant_scale_kv);
    p.dequant_scale_q = optPtr<float>(dequant_scale_q);
    p.dequant_scale_kv = optPtr<float>(dequant_scale_kv);
    p.host_bmm1_scale = static_cast<float>(host_bmm1_scale);
    p.absorption_mode = absorption_mode;
    p.quant_scale_qkv = optPtr<float>(quant_scale_qkv);
    p.helix_position_offsets = optPtr<int32_t>(helix_position_offsets);
    p.helix_is_inactive_rank
        = helix_is_inactive_rank.has_value() ? static_cast<bool const*>(helix_is_inactive_rank->data_ptr()) : nullptr;

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        tensorrt_llm::kernels::invokeMLAContextFp8Quantize(
            reinterpret_cast<MlaParams<float>&>(p), static_cast<int>(total_kv_len), stream);
        break;
    case nvinfer1::DataType::kHALF:
        tensorrt_llm::kernels::invokeMLAContextFp8Quantize(
            reinterpret_cast<MlaParams<half>&>(p), static_cast<int>(total_kv_len), stream);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        tensorrt_llm::kernels::invokeMLAContextFp8Quantize(
            reinterpret_cast<MlaParams<__nv_bfloat16>&>(p), static_cast<int>(total_kv_len), stream);
        break;
#endif
    default: TORCH_CHECK(false, "Unsupported data type for MLA context FP8 quantize.");
    }
    sync_check_cuda_error(stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// Schema string shared by qkv_preprocessing and kv_cache_postprocessing.
// clang-format off
#define QKV_PROCESSING_SCHEMA                          \
    "Tensor(a!)? qkv_input, "                          \
    "Tensor(b!)? cross_kv_input, "                     \
    "Tensor(c!)? quantized_qkv_output, "               \
    "Tensor(d!)? q_output, "                           \
    "Tensor? kv_cache_block_offsets, "                 \
    "Tensor? host_kv_cache_pool_pointers, "            \
    "Tensor? host_kv_cache_pool_mapping, "             \
    "Tensor? qkv_bias, "                               \
    "Tensor? qkv_scale_quant_orig, "                   \
    "Tensor? qkv_scale_orig_quant, "                   \
    "Tensor? o_scale_orig_quant, "                     \
    "Tensor(e!)? fmha_bmm1_scale, "                    \
    "Tensor(f!)? fmha_bmm2_scale, "                    \
    "Tensor(g!)? fmha_tile_counter, "                  \
    "Tensor? logn_scaling, "                           \
    "Tensor? tokens_info, "                            \
    "Tensor? seq_lens, "                               \
    "Tensor? cache_seq_lens, "                         \
    "Tensor? encoder_seq_lens, "                       \
    "Tensor? cu_seq_lens, "                            \
    "Tensor? cu_kv_seq_lens, "                         \
    "Tensor? sparse_kv_offsets, "                      \
    "Tensor? sparse_kv_indices, "                      \
    "Tensor? rotary_embedding_inv_freq, "              \
    "Tensor? rotary_coef_cache_buffer, "               \
    "Tensor? spec_decoding_position_offsets, "         \
    "Tensor? mrope_rotary_cos_sin, "                   \
    "Tensor? mrope_position_deltas, "                  \
    "int batch_size, "                                 \
    "int max_input_seq_len, "                          \
    "int max_kv_seq_len, "                             \
    "int cyclic_kv_cache_len, "                        \
    "int sink_token_len, "                             \
    "int token_num, "                                  \
    "bool remove_padding, "                            \
    "bool is_last_chunk, "                             \
    "bool cross_attention, "                           \
    "int head_num, "                                   \
    "int kv_head_num, "                                \
    "int qheads_per_kv_head, "                         \
    "int size_per_head, "                              \
    "float fmha_host_bmm1_scale, "                     \
    "int rotary_embedding_dim, "                       \
    "float rotary_embedding_base, "                    \
    "int rotary_scaling_type, "                        \
    "float rotary_embedding_scale, "                   \
    "int rotary_embedding_max_positions, "             \
    "int position_embedding_type, "                    \
    "bool position_shift_enabled, "                    \
    "bool separate_q_kv_output, "                      \
    "bool quantized_fp8_output, "                      \
    "bool generation_phase, "                          \
    "int rotary_vision_start, "                        \
    "int rotary_vision_length, "                       \
    "int layer_idx, "                                  \
    "int tokens_per_block, "                           \
    "int max_attention_window_size, "                  \
    "int kv_cache_quant_mode, "                        \
    "int cyclic_attention_window_size, "               \
    "int beam_width, "                                 \
    "int sink_token_length, "                          \
    "int seq_offset=0, "                               \
    "bool is_mla_enable=False"

// Schema string shared by mla_rope_context and mla_context_fp8_quantize.
// clang-format off
#define MLA_PARAMS_SCHEMA                              \
    "Tensor? latent_cache, "                           \
    "Tensor(a!)? q_buf, "                              \
    "Tensor(b!)? k_buf, "                              \
    "Tensor? v_buf, "                                  \
    "Tensor(c!)? quant_q_buf, "                        \
    "Tensor(d!)? quant_k_buf, "                        \
    "Tensor(e!)? quant_v_buf, "                        \
    "Tensor? context_buf, "                            \
    "Tensor(f!)? q_pe, "                               \
    "Tensor? cos_sin_cache, "                          \
    "Tensor? workspace, "                              \
    "Tensor? cache_seq_lens, "                         \
    "Tensor? seq_q_offset, "                           \
    "Tensor(g!)? fmha_tile_counter, "                  \
    "Tensor? cu_q_seqlens, "                           \
    "Tensor? cu_kv_seqlens, "                          \
    "Tensor? block_ids_per_seq, "                      \
    "Tensor(h!)? bmm1_scale, "                         \
    "Tensor(i!)? bmm2_scale, "                         \
    "Tensor? quant_scale_o, "                          \
    "Tensor? quant_scale_q, "                          \
    "Tensor? quant_scale_kv, "                         \
    "Tensor? dequant_scale_q, "                        \
    "Tensor? dequant_scale_kv, "                       \
    "Tensor? quant_scale_qkv, "                        \
    "Tensor? helix_position_offsets, "                 \
    "Tensor? helix_is_inactive_rank, "                 \
    "int batch_size, "                                 \
    "int acc_q_len, "                                  \
    "int head_num, "                                   \
    "int max_input_seq_len, "                          \
    "int q_pe_ld, "                                    \
    "int q_pe_stride, "                                \
    "int q_lora_rank, "                                \
    "int kv_lora_rank, "                               \
    "int qk_nope_head_dim, "                           \
    "int qk_rope_head_dim, "                           \
    "int v_head_dim, "                                 \
    "int predicted_tokens_per_seq, "                   \
    "int num_layers, "                                 \
    "float host_bmm1_scale, "                          \
    "bool absorption_mode"

// clang-format on

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // clang-format off
    m.def(
        "build_decoder_info("
        "Tensor(a!)? seq_q_offsets, "
        "Tensor(b!)? seq_kv_offsets, "
        "Tensor(c!)? padding_offsets, "
        "Tensor(d!)? tokens_info, "
        "Tensor(e!)? encoder_padding_offsets, "
        "Tensor(f!)? packed_mask_row_offsets, "
        "Tensor(g!)? seq_cp_partial_offsets, "
        "Tensor(h!)? attention_mask, "
        "Tensor? seq_q_lengths, "
        "Tensor? seq_kv_lengths, "
        "Tensor(i!)? fmha_tile_counter, "
        "Tensor? dequant_scale_qkv, "
        "Tensor? quant_scale_o, "
        "Tensor(j!)? fmha_bmm1_scale, "
        "Tensor(k!)? fmha_bmm2_scale, "
        "Tensor(l!)? rotary_embedding_inv_freq, "
        "Tensor? rotary_embedding_inv_freq_cache, "
        "int cp_size, "
        "bool separate_qkv_scales, "
        "float fmha_host_bmm1_scale, "
        "int batch_size, "
        "int max_q_seq_length, "
        "int max_encoder_q_seq_length, "
        "int attention_window_size, "
        "int sink_token_length, "
        "int num_tokens, "
        "bool remove_padding, "
        "int attention_mask_type, "
        "float rotary_embedding_scale, "
        "float rotary_embedding_base, "
        "int rotary_embedding_dim, "
        "int rotary_scaling_type, "
        "int rotary_embedding_max_positions"
        ") -> bool");
    // clang-format on

    m.def("qkv_preprocessing(" QKV_PROCESSING_SCHEMA ") -> ()");
    m.def("kv_cache_postprocessing(" QKV_PROCESSING_SCHEMA ") -> ()");

    // clang-format off
    m.def("mla_rope_context(" MLA_PARAMS_SCHEMA ", "
        "Tensor? kv_cache_block_offsets, "
        "Tensor? host_kv_cache_pool_pointers, "
        "Tensor? host_kv_cache_pool_mapping, "
        "int layer_idx, "
        "int tokens_per_block, "
        "int kv_head_num, "
        "int size_per_head, "
        "int kv_cache_quant_mode, "
        "int cyclic_attention_window_size, "
        "int max_attention_window_size, "
        "int sink_token_length, "
        "int beam_width, "
        "int seq_offset=0"
        ") -> ()");
    // clang-format on

    m.def("mla_context_fp8_quantize(" MLA_PARAMS_SCHEMA
          ", "
          "int total_kv_len"
          ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("build_decoder_info", &tensorrt_llm::torch_ext::build_decoder_info);
    m.impl("qkv_preprocessing", &tensorrt_llm::torch_ext::qkv_processing<true>);
    m.impl("kv_cache_postprocessing", &tensorrt_llm::torch_ext::qkv_processing<false>);
    m.impl("mla_rope_context", &tensorrt_llm::torch_ext::mla_rope_context);
    m.impl("mla_context_fp8_quantize", &tensorrt_llm::torch_ext::mla_context_fp8_quantize);
}
