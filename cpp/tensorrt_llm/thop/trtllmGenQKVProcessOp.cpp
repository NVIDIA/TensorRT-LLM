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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <climits>
#include <cmath>
#include <torch/extension.h>
#include <type_traits>

using namespace tensorrt_llm::kernels;
using tensorrt_llm::runtime::TorchUtils;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

template <typename T>
T* getDataPtrOrNull(torch::optional<torch::Tensor> const& tensor)
{
    return tensor.has_value() ? static_cast<T*>(tensor->data_ptr()) : nullptr;
}

template <typename T>
int getKvCacheElemSizeInBits(tensorrt_llm::common::QuantMode const& kvCacheQuantMode)
{
    if (kvCacheQuantMode.hasInt8KvCache() || kvCacheQuantMode.hasFp8KvCache())
    {
        return CHAR_BIT;
    }
    else if (kvCacheQuantMode.hasFp4KvCache())
    {
        return 4;
    }
    return sizeof(T) * CHAR_BIT;
}

} // namespace

void buildDecoderInfo(torch::optional<torch::Tensor> seq_q_offsets, torch::optional<torch::Tensor> seq_kv_offsets,
    torch::optional<torch::Tensor> padding_offsets, torch::optional<torch::Tensor> tokens_info,
    torch::optional<torch::Tensor> encoder_padding_offsets, torch::optional<torch::Tensor> packed_mask_row_offsets,
    torch::optional<torch::Tensor> seq_cp_partial_offsets, torch::optional<torch::Tensor> attention_mask,
    torch::optional<torch::Tensor> seq_q_lengths, torch::optional<torch::Tensor> seq_kv_lengths, int64_t cp_size,
    torch::optional<torch::Tensor> fmha_tile_counter, torch::optional<torch::Tensor> dequant_scale_qkv,
    bool separate_qkv_scales, torch::optional<torch::Tensor> quant_scale_o, double fmha_host_bmm1_scale,
    torch::optional<torch::Tensor> fmha_bmm1_scale, torch::optional<torch::Tensor> fmha_bmm2_scale, int64_t batch_size,
    int64_t max_q_seq_length, int64_t max_encoder_q_seq_length, int64_t attention_window_size,
    int64_t sink_token_length, int64_t num_tokens, bool remove_padding, int64_t attention_mask_type,
    double rotary_embedding_scale, double rotary_embedding_base, int64_t rotary_embedding_dim,
    int64_t rotary_scaling_type, torch::optional<torch::Tensor> rotary_embedding_inv_freq,
    torch::optional<torch::Tensor> rotary_embedding_inv_freq_cache, int64_t rotary_embedding_max_positions)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto mask_dtype = nvinfer1::DataType::kFLOAT;
    if (attention_mask.has_value())
    {
        mask_dtype = TorchUtils::dataType(attention_mask.value().scalar_type());
    }

    auto buildDecoderInfoImpl = [&](auto* dummy_mask_ptr)
    {
        using MaskT = std::remove_pointer_t<decltype(dummy_mask_ptr)>;

        BuildDecoderInfoParams<MaskT> params{};

        params.seqQOffsets = getDataPtrOrNull<int>(seq_q_offsets);
        params.seqKVOffsets = getDataPtrOrNull<int>(seq_kv_offsets);
        params.seqCpPartialOffsets = getDataPtrOrNull<int>(seq_cp_partial_offsets);
        params.cpSize = cp_size;
        params.packedMaskRowOffsets = getDataPtrOrNull<int>(packed_mask_row_offsets);
        params.paddingOffsets = getDataPtrOrNull<int>(padding_offsets);
        params.tokensInfo = getDataPtrOrNull<int2>(tokens_info);
        params.encoderPaddingOffsets = getDataPtrOrNull<int>(encoder_padding_offsets);
        params.attentionMask = getDataPtrOrNull<MaskT>(attention_mask);
        params.seqQLengths = getDataPtrOrNull<int>(seq_q_lengths);
        params.seqKVLengths = getDataPtrOrNull<int>(seq_kv_lengths);
        params.batchSize = batch_size;
        params.maxQSeqLength = max_q_seq_length;
        params.maxEncoderQSeqLength = max_encoder_q_seq_length;
        params.attentionWindowSize = attention_window_size;
        params.sinkTokenLength = sink_token_length;
        params.numTokens = num_tokens;
        params.removePadding = remove_padding;
        params.attentionMaskType = static_cast<AttentionMaskType>(attention_mask_type);
        params.blockSparseParams = BlockSparseParams{};
        params.fmhaTileCounter = getDataPtrOrNull<uint32_t>(fmha_tile_counter);
        params.quantScaleO = getDataPtrOrNull<float>(quant_scale_o);
        params.dequantScaleQkv = getDataPtrOrNull<float>(dequant_scale_qkv);
        params.separateQkvScales = separate_qkv_scales;
        params.fmhaHostBmm1Scale = fmha_host_bmm1_scale;
        params.fmhaBmm1Scale = getDataPtrOrNull<float>(fmha_bmm1_scale);
        params.fmhaBmm2Scale = getDataPtrOrNull<float>(fmha_bmm2_scale);
        params.rotaryEmbeddingScale = rotary_embedding_scale;
        params.rotaryEmbeddingBase = rotary_embedding_base;
        params.rotaryEmbeddingDim = rotary_embedding_dim;
        params.rotaryScalingType = static_cast<RotaryScalingType>(rotary_scaling_type);
        params.rotaryEmbeddingInvFreq = getDataPtrOrNull<float>(rotary_embedding_inv_freq);
        params.rotaryEmbeddingInvFreqCache = getDataPtrOrNull<float>(rotary_embedding_inv_freq_cache);
        params.rotaryEmbeddingMaxPositions = rotary_embedding_max_positions;

        invokeBuildDecoderInfo(params, stream);
        sync_check_cuda_error(stream);
    };

    switch (mask_dtype)
    {
    case nvinfer1::DataType::kFLOAT: buildDecoderInfoImpl(static_cast<float*>(nullptr)); break;
    case nvinfer1::DataType::kHALF: buildDecoderInfoImpl(static_cast<half*>(nullptr)); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: buildDecoderInfoImpl(static_cast<__nv_bfloat16*>(nullptr)); break;
#endif
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8: buildDecoderInfoImpl(static_cast<__nv_fp8_e4m3*>(nullptr)); break;
#endif
    default: TORCH_CHECK(false, "Unsupported attention mask dtype");
    }
}

template <bool isPreprocessing>
void qkvProcessing(
    // Input/Output buffers
    torch::Tensor qkv_input, torch::optional<torch::Tensor> cross_kv_input,
    torch::optional<torch::Tensor> quantized_qkv_output, torch::optional<torch::Tensor> q_output,
    // KV cache pool pointers and mapping (aligned with thop/attentionOp.cpp)
    torch::optional<torch::Tensor> kv_cache_block_offsets, torch::optional<torch::Tensor> host_kv_cache_pool_pointers,
    torch::optional<torch::Tensor> host_kv_cache_pool_mapping,
    // Bias and quantization scales
    torch::optional<torch::Tensor> qkv_bias, torch::optional<torch::Tensor> qkv_scale_quant_orig,
    torch::optional<torch::Tensor> qkv_scale_orig_quant, torch::optional<torch::Tensor> o_scale_orig_quant,
    torch::optional<torch::Tensor> fmha_bmm1_scale, torch::optional<torch::Tensor> fmha_bmm2_scale,
    torch::optional<torch::Tensor> fmha_tile_counter,
    // Additional buffers
    torch::optional<torch::Tensor> logn_scaling, torch::optional<torch::Tensor> tokens_info, torch::Tensor seq_lens,
    torch::optional<torch::Tensor> cache_seq_lens, torch::optional<torch::Tensor> encoder_seq_lens,
    torch::Tensor cu_seq_lens, torch::optional<torch::Tensor> cu_kv_seq_lens,
    torch::optional<torch::Tensor> sparse_kv_offsets, torch::optional<torch::Tensor> sparse_kv_indices,
    torch::optional<torch::Tensor> rotary_embedding_inv_freq, torch::optional<torch::Tensor> rotary_coef_cache_buffer,
    torch::optional<torch::Tensor> spec_decoding_position_offsets, torch::optional<torch::Tensor> mrope_rotary_cos_sin,
    torch::optional<torch::Tensor> mrope_position_deltas,
    // Scalar parameters
    int64_t batch_size, int64_t max_input_seq_len, int64_t max_kv_seq_len, int64_t cyclic_kv_cache_len,
    int64_t sink_token_len, int64_t token_num, bool remove_padding, bool is_last_chunk, bool cross_attention,
    int64_t head_num, int64_t kv_head_num, int64_t qheads_per_kv_head, int64_t size_per_head,
    double fmha_host_bmm1_scale, int64_t rotary_embedding_dim, double rotary_embedding_base, int64_t rotary_scale_type,
    double rotary_embedding_scale, int64_t rotary_embedding_max_positions, int64_t position_embedding_type,
    bool position_shift_enabled, int64_t cache_type, bool separate_q_kv_output, bool quantized_fp8_output,
    bool generation_phase, int64_t rotary_vision_start, int64_t rotary_vision_length,
    // KV cache parameters
    int64_t layer_idx, int64_t tokens_per_block, int64_t max_attention_window_size, int64_t kv_cache_quant_mode)
{
    auto const kvQuantMode = tensorrt_llm::common::QuantMode(static_cast<uint32_t>(kv_cache_quant_mode));
    bool const use_kv_cache = host_kv_cache_pool_pointers.has_value() && host_kv_cache_pool_mapping.has_value()
        && kv_cache_block_offsets.has_value();
    bool const use_paged_kv_cache = use_kv_cache; // pool-pointer API implies paged KV cache
    bool const use_nvfp4_kv_cache = use_kv_cache && kvQuantMode.hasFp4KvCache();

    auto qkvProcessingImpl = [&](auto* dummy_qkv_input_ptr, auto* dummy_kv_cache_buffer_ptr)
    {
        using T = std::remove_pointer_t<decltype(dummy_qkv_input_ptr)>;
        using KVCacheBuffer = std::remove_pointer_t<decltype(dummy_kv_cache_buffer_ptr)>;

        auto const kvElemBits = getKvCacheElemSizeInBits<T>(kvQuantMode);
        auto const sizePerToken = static_cast<int32_t>(kv_head_num * size_per_head * kvElemBits / 8);

        QKVPreprocessingParams<T, KVCacheBuffer> params{};

        params.qkv_input = static_cast<T*>(qkv_input.data_ptr());
        params.cross_kv_input = getDataPtrOrNull<T>(cross_kv_input);
        params.quantized_qkv_output = getDataPtrOrNull<void>(quantized_qkv_output);
        params.q_output = getDataPtrOrNull<T>(q_output);

        if (use_kv_cache)
        {
            int32_t const pool_index = host_kv_cache_pool_mapping.value().index({layer_idx, 0}).item<int32_t>();
            int32_t const layer_idx_in_cache_pool
                = host_kv_cache_pool_mapping.value().index({layer_idx, 1}).item<int32_t>();

            int max_blocks_per_sequence = kv_cache_block_offsets.value().size(-1);
            KVBlockArray::DataType* block_offsets
                = static_cast<KVBlockArray::DataType*>(kv_cache_block_offsets.value().index({pool_index}).data_ptr());

            int32_t const kv_factor = 2;
            auto const block_size = tokens_per_block * kv_head_num * size_per_head;
            auto const bytes_per_block = block_size * kvElemBits / 8 /*bits*/;
            auto const intra_pool_offset = layer_idx_in_cache_pool * kv_factor * bytes_per_block;

            // Prepare block pool pointers for NVFP4 KV cache.
            void* host_primary_pool_pointer{nullptr};
            void* host_secondary_pool_pointer{nullptr};
            void* host_primary_block_scale_pool_pointer{nullptr};
            void* host_secondary_block_scale_pool_pointer{nullptr};

            if (use_nvfp4_kv_cache)
            {
                // For NVFP4 KV cache, extra block scales are stored in separate pools.
                // The layout of host_kv_cache_pool_pointers is [num_pools, 2 (primary and secondary), 2 (data and
                // scale)].
                TORCH_CHECK(host_kv_cache_pool_pointers.value().dim() == 3);
                host_primary_pool_pointer = reinterpret_cast<void*>(
                    reinterpret_cast<char*>(
                        host_kv_cache_pool_pointers.value().index({pool_index, 0, 0}).item<int64_t>())
                    + intra_pool_offset);
                host_secondary_pool_pointer = reinterpret_cast<void*>(
                    reinterpret_cast<char*>(
                        host_kv_cache_pool_pointers.value().index({pool_index, 1, 0}).item<int64_t>())
                    + intra_pool_offset);
                // Calculate the intra-pool offset for scaling factors.
                // Note that NVFP4 block scaling use a fixed vector size of 16.
                auto constexpr vector_size = 16;
                auto const bytes_per_block_sf = block_size / vector_size * 1 /*bytes per E4M3 sf*/;
                auto const intra_pool_offset_sf = layer_idx_in_cache_pool * kv_factor * bytes_per_block_sf;
                host_primary_block_scale_pool_pointer = reinterpret_cast<void*>(
                    reinterpret_cast<char*>(
                        host_kv_cache_pool_pointers.value().index({pool_index, 0, 1}).item<int64_t>())
                    + intra_pool_offset_sf);
                host_secondary_block_scale_pool_pointer = reinterpret_cast<void*>(
                    reinterpret_cast<char*>(
                        host_kv_cache_pool_pointers.value().index({pool_index, 1, 1}).item<int64_t>())
                    + intra_pool_offset_sf);
            }
            else
            {
                TORCH_CHECK(host_kv_cache_pool_pointers.value().dim() == 2);
                host_primary_pool_pointer = reinterpret_cast<void*>(
                    reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 0}).item<int64_t>())
                    + intra_pool_offset);
                host_secondary_pool_pointer = reinterpret_cast<void*>(
                    reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 1}).item<int64_t>())
                    + intra_pool_offset);
            }

            if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
            {
                params.kv_cache_buffer = KVBlockArray(batch_size, max_blocks_per_sequence, tokens_per_block,
                    sizePerToken, max_attention_window_size, max_attention_window_size, sink_token_len,
                    /*canUseOneMoreBlock=*/false, host_primary_pool_pointer, host_secondary_pool_pointer,
                    block_offsets);
                if (kvQuantMode.hasFp4KvCache())
                {
                    params.kv_cache_block_scales_buffer = KVBlockArray(batch_size, max_blocks_per_sequence,
                        tokens_per_block, sizePerToken / 8, max_attention_window_size, max_attention_window_size,
                        sink_token_len, /*canUseOneMoreBlock=*/false, host_primary_block_scale_pool_pointer,
                        host_secondary_block_scale_pool_pointer, block_offsets);
                }
            }
            else if constexpr (std::is_same_v<KVCacheBuffer, KVLinearBuffer>)
            {
                using BufferDataType = typename KVCacheBuffer::DataType;
                params.kv_cache_buffer
                    = KVLinearBuffer(batch_size, cross_attention ? max_kv_seq_len : max_attention_window_size,
                        sizePerToken, cyclic_kv_cache_len > 0 ? cyclic_kv_cache_len : max_attention_window_size,
                        sink_token_len, false, reinterpret_cast<BufferDataType*>(host_primary_pool_pointer));
                TLLM_CHECK_WITH_INFO(!(kvQuantMode.hasFp4KvCache()), "FP4 KV cache only supports paged KV.");
            }
        }

        params.qkv_bias = getDataPtrOrNull<T const>(qkv_bias);

        params.qkv_scale_quant_orig = getDataPtrOrNull<float>(qkv_scale_quant_orig);
        params.qkv_scale_orig_quant = getDataPtrOrNull<float>(qkv_scale_orig_quant);
        params.o_scale_orig_quant = getDataPtrOrNull<float>(o_scale_orig_quant);
        params.fmha_bmm1_scale = getDataPtrOrNull<float>(fmha_bmm1_scale);
        params.fmha_bmm2_scale = getDataPtrOrNull<float>(fmha_bmm2_scale);

        params.logn_scaling = getDataPtrOrNull<float>(logn_scaling);
        params.tokens_info = getDataPtrOrNull<int2>(tokens_info);
        params.seq_lens = seq_lens.data_ptr<int>();
        params.cache_seq_lens = getDataPtrOrNull<int>(cache_seq_lens);
        params.encoder_seq_lens = getDataPtrOrNull<int>(encoder_seq_lens);
        params.cu_seq_lens = cu_seq_lens.data_ptr<int>();
        params.cu_kv_seq_lens = getDataPtrOrNull<int>(cu_kv_seq_lens);
        params.sparse_kv_offsets = getDataPtrOrNull<int>(sparse_kv_offsets);
        params.sparse_kv_indices = getDataPtrOrNull<int>(sparse_kv_indices);
        params.rotary_embedding_inv_freq = getDataPtrOrNull<float>(rotary_embedding_inv_freq);
        params.rotary_coef_cache_buffer = getDataPtrOrNull<float2 const>(rotary_coef_cache_buffer);
        params.spec_decoding_position_offsets = getDataPtrOrNull<int>(spec_decoding_position_offsets);
        params.mrope_rotary_cos_sin = getDataPtrOrNull<float2 const>(mrope_rotary_cos_sin);
        params.mrope_position_deltas = getDataPtrOrNull<int32_t>(mrope_position_deltas);

        params.batch_size = batch_size;
        params.max_input_seq_len = max_input_seq_len;
        params.max_kv_seq_len = max_kv_seq_len;
        params.cyclic_kv_cache_len = cyclic_kv_cache_len;
        params.sink_token_len = sink_token_len;
        params.token_num = token_num;
        params.remove_padding = remove_padding;
        params.is_last_chunk = is_last_chunk;
        params.cross_attention = cross_attention;
        params.head_num = head_num;
        params.kv_head_num = kv_head_num;
        params.qheads_per_kv_head = qheads_per_kv_head;
        params.size_per_head = size_per_head;
        params.fmha_host_bmm1_scale = fmha_host_bmm1_scale;
        params.rotary_embedding_dim = rotary_embedding_dim;
        params.rotary_embedding_base = rotary_embedding_base;
        params.rotary_scale_type = static_cast<RotaryScalingType>(rotary_scale_type);
        params.rotary_embedding_scale = rotary_embedding_scale;
        params.rotary_embedding_max_positions = rotary_embedding_max_positions;
        params.position_embedding_type = static_cast<PositionEmbeddingType>(position_embedding_type);
        params.position_shift_enabled = position_shift_enabled;
        params.cache_type = static_cast<KvCacheDataType>(cache_type);
        params.separate_q_kv_output = separate_q_kv_output;
        params.quantized_fp8_output = quantized_fp8_output;
        params.generation_phase = generation_phase;
        params.multi_processor_count = tensorrt_llm::common::getMultiProcessorCount();
        params.rotary_vision_start = rotary_vision_start;
        params.rotary_vision_length = rotary_vision_length;

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if constexpr (isPreprocessing)
        {
            invokeQKVPreprocessing(params, stream);
        }
        else
        {
            invokeKvCachePostprocessing(params, stream);
        }
        sync_check_cuda_error(stream);
    };

    auto dispatch = [&](auto* dummy_ptr)
    {
        using T = std::remove_pointer_t<decltype(dummy_ptr)>;
        if (use_paged_kv_cache)
        {
            qkvProcessingImpl(static_cast<T*>(nullptr), static_cast<KVBlockArray*>(nullptr));
        }
        else
        {
            qkvProcessingImpl(static_cast<T*>(nullptr), static_cast<KVLinearBuffer*>(nullptr));
        }
    };

    switch (TorchUtils::dataType(qkv_input.scalar_type()))
    {
    case nvinfer1::DataType::kFLOAT: dispatch(static_cast<float*>(nullptr)); break;
    case nvinfer1::DataType::kHALF: dispatch(static_cast<half*>(nullptr)); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: dispatch(static_cast<__nv_bfloat16*>(nullptr)); break;
#endif
    default: TORCH_CHECK(false, "Unsupported data type for QKV preprocessing. FP8 is not yet supported.");
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "build_decoder_info("
        "Tensor? seq_q_offsets"
        ", Tensor? seq_kv_offsets"
        ", Tensor? padding_offsets"
        ", Tensor? tokens_info"
        ", Tensor? encoder_padding_offsets"
        ", Tensor? packed_mask_row_offsets"
        ", Tensor? seq_cp_partial_offsets"
        ", Tensor? attention_mask"
        ", Tensor? seq_q_lengths"
        ", Tensor? seq_kv_lengths"
        ", int cp_size"
        ", Tensor? fmha_tile_counter"
        ", Tensor? dequant_scale_qkv"
        ", bool separate_qkv_scales"
        ", Tensor? quant_scale_o"
        ", float fmha_host_bmm1_scale"
        ", Tensor? fmha_bmm1_scale"
        ", Tensor? fmha_bmm2_scale"
        ", int batch_size"
        ", int max_q_seq_length"
        ", int max_encoder_q_seq_length"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int num_tokens"
        ", bool remove_padding"
        ", int attention_mask_type"
        ", float rotary_embedding_scale"
        ", float rotary_embedding_base"
        ", int rotary_embedding_dim"
        ", int rotary_scaling_type"
        ", Tensor? rotary_embedding_inv_freq"
        ", Tensor? rotary_embedding_inv_freq_cache"
        ", int rotary_embedding_max_positions"
        ") -> ()");

    m.def(
        "qkv_preprocessing("
        "Tensor qkv_input"
        ", Tensor? cross_kv_input"
        ", Tensor? quantized_qkv_output"
        ", Tensor? q_output"
        ", Tensor? kv_cache_block_offsets"
        ", Tensor? host_kv_cache_pool_pointers"
        ", Tensor? host_kv_cache_pool_mapping"
        ", Tensor? qkv_bias"
        ", Tensor? qkv_scale_quant_orig"
        ", Tensor? qkv_scale_orig_quant"
        ", Tensor? o_scale_orig_quant"
        ", Tensor? fmha_bmm1_scale"
        ", Tensor? fmha_bmm2_scale"
        ", Tensor? fmha_tile_counter"
        ", Tensor? logn_scaling"
        ", Tensor? tokens_info"
        ", Tensor seq_lens"
        ", Tensor? cache_seq_lens"
        ", Tensor? encoder_seq_lens"
        ", Tensor cu_seq_lens"
        ", Tensor? cu_kv_seq_lens"
        ", Tensor? sparse_kv_offsets"
        ", Tensor? sparse_kv_indices"
        ", Tensor? rotary_embedding_inv_freq"
        ", Tensor? rotary_coef_cache_buffer"
        ", Tensor? spec_decoding_position_offsets"
        ", Tensor? mrope_rotary_cos_sin"
        ", Tensor? mrope_position_deltas"
        ", int batch_size"
        ", int max_input_seq_len"
        ", int max_kv_seq_len"
        ", int cyclic_kv_cache_len"
        ", int sink_token_len"
        ", int token_num"
        ", bool remove_padding"
        ", bool is_last_chunk"
        ", bool cross_attention"
        ", int head_num"
        ", int kv_head_num"
        ", int qheads_per_kv_head"
        ", int size_per_head"
        ", float fmha_host_bmm1_scale"
        ", int rotary_embedding_dim"
        ", float rotary_embedding_base"
        ", int rotary_scale_type"
        ", float rotary_embedding_scale"
        ", int rotary_embedding_max_positions"
        ", int position_embedding_type"
        ", bool position_shift_enabled"
        ", int cache_type"
        ", bool separate_q_kv_output"
        ", bool quantized_fp8_output"
        ", bool generation_phase"
        ", int rotary_vision_start"
        ", int rotary_vision_length"
        ", int layer_idx"
        ", int tokens_per_block"
        ", int max_attention_window_size"
        ", int kv_cache_quant_mode"
        ") -> ()");

    m.def(
        "kv_cache_postprocessing("
        "Tensor qkv_input"
        ", Tensor? cross_kv_input"
        ", Tensor? quantized_qkv_output"
        ", Tensor? q_output"
        ", Tensor? kv_cache_block_offsets"
        ", Tensor? host_kv_cache_pool_pointers"
        ", Tensor? host_kv_cache_pool_mapping"
        ", Tensor? qkv_bias"
        ", Tensor? qkv_scale_quant_orig"
        ", Tensor? qkv_scale_orig_quant"
        ", Tensor? o_scale_orig_quant"
        ", Tensor? fmha_bmm1_scale"
        ", Tensor? fmha_bmm2_scale"
        ", Tensor? fmha_tile_counter"
        ", Tensor? logn_scaling"
        ", Tensor? tokens_info"
        ", Tensor seq_lens"
        ", Tensor? cache_seq_lens"
        ", Tensor? encoder_seq_lens"
        ", Tensor cu_seq_lens"
        ", Tensor? cu_kv_seq_lens"
        ", Tensor? sparse_kv_offsets"
        ", Tensor? sparse_kv_indices"
        ", Tensor? rotary_embedding_inv_freq"
        ", Tensor? rotary_coef_cache_buffer"
        ", Tensor? spec_decoding_position_offsets"
        ", Tensor? mrope_rotary_cos_sin"
        ", Tensor? mrope_position_deltas"
        ", int batch_size"
        ", int max_input_seq_len"
        ", int max_kv_seq_len"
        ", int cyclic_kv_cache_len"
        ", int sink_token_len"
        ", int token_num"
        ", bool remove_padding"
        ", bool is_last_chunk"
        ", bool cross_attention"
        ", int head_num"
        ", int kv_head_num"
        ", int qheads_per_kv_head"
        ", int size_per_head"
        ", float fmha_host_bmm1_scale"
        ", int rotary_embedding_dim"
        ", float rotary_embedding_base"
        ", int rotary_scale_type"
        ", float rotary_embedding_scale"
        ", int rotary_embedding_max_positions"
        ", int position_embedding_type"
        ", bool position_shift_enabled"
        ", int cache_type"
        ", bool separate_q_kv_output"
        ", bool quantized_fp8_output"
        ", bool generation_phase"
        ", int rotary_vision_start"
        ", int rotary_vision_length"
        ", int layer_idx"
        ", int tokens_per_block"
        ", int max_attention_window_size"
        ", int kv_cache_quant_mode"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("build_decoder_info", &tensorrt_llm::torch_ext::buildDecoderInfo);
    m.impl("qkv_preprocessing", &tensorrt_llm::torch_ext::qkvProcessing</*isPreprocessing=*/true>);
    m.impl("kv_cache_postprocessing", &tensorrt_llm::torch_ext::qkvProcessing</*isPreprocessing=*/false>);
}
