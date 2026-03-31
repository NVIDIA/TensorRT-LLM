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
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("build_decoder_info", &tensorrt_llm::torch_ext::build_decoder_info);
    m.impl("qkv_preprocessing", &tensorrt_llm::torch_ext::qkv_processing<true>);
    m.impl("kv_cache_postprocessing", &tensorrt_llm::torch_ext::qkv_processing<false>);
}
