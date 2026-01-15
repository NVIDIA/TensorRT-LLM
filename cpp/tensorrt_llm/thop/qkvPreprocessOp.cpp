/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION &
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

/**
 * @file qkvPreprocessOp.cpp
 * @brief PyTorch bindings for QKV preprocessing operations.
 *
 * This file exports QKV preprocessing functionality to Python, enabling:
 * - RoPE (Rotary Position Embedding) application
 * - KV cache updates for both context and generation phases
 * - Support for paged KV cache
 *
 * These operations are required by the TrtllmGenAttention backend to
 * preprocess QKV tensors before calling the TRTLLM-Gen FMHA kernels.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

using tk::KVBlockArray;
using tk::KVLinearBuffer;
using tk::KvCacheDataType;
using tk::PositionEmbeddingType;
using tk::QKVPreprocessingParams;
using tk::RotaryScalingType;

namespace
{

/**
 * @brief Create a KVBlockArray for paged KV cache management.
 */
KVBlockArray createKVBlockArray(int batch_size, int max_blocks_per_sequence, int tokens_per_block, int size_per_token,
    int cyclic_attention_window_size, int max_cyclic_attention_window_size, int sink_token_length,
    bool can_use_one_more_block, void* host_primary_pool_pointer, void* host_secondary_pool_pointer,
    KVBlockArray::DataType* block_offsets)
{
    return KVBlockArray(batch_size, max_blocks_per_sequence, tokens_per_block, size_per_token,
        cyclic_attention_window_size, max_cyclic_attention_window_size, sink_token_length, can_use_one_more_block,
        host_primary_pool_pointer, host_secondary_pool_pointer, block_offsets);
}

/**
 * @brief Convert torch dtype to KvCacheDataType.
 */
KvCacheDataType getKvCacheDataType(tc::QuantMode const& quant_mode, at::ScalarType dtype)
{
    if (quant_mode.hasInt8KvCache())
    {
        return KvCacheDataType::INT8;
    }
    else if (quant_mode.hasFp8KvCache())
    {
        return KvCacheDataType::FP8;
    }
    else if (quant_mode.hasFp4KvCache())
    {
        return KvCacheDataType::NVFP4;
    }
    return KvCacheDataType::BASE;
}

/**
 * @brief Convert integer to PositionEmbeddingType.
 */
PositionEmbeddingType intToPositionEmbeddingType(int64_t type)
{
    return static_cast<PositionEmbeddingType>(static_cast<int8_t>(type));
}

/**
 * @brief Convert integer to RotaryScalingType.
 */
RotaryScalingType intToRotaryScalingType(int64_t type)
{
    return static_cast<RotaryScalingType>(static_cast<int8_t>(type));
}

template <typename T, typename KVCacheBuffer>
void qkvPreprocessingHelper(QKVPreprocessingParams<T, KVCacheBuffer>& params, cudaStream_t stream)
{
    tk::invokeQKVPreprocessing<T, KVCacheBuffer>(params, stream);
}

template <typename T, typename KVCacheBuffer>
void kvCachePostprocessingHelper(QKVPreprocessingParams<T, KVCacheBuffer>& params, cudaStream_t stream)
{
    tk::invokeKvCachePostprocessing<T, KVCacheBuffer>(params, stream);
}

/**
 * @brief Build QKVPreprocessingParams from direct arguments.
 */
template <typename T>
QKVPreprocessingParams<T, KVBlockArray> buildQkvPreprocessingParams(
    // Tensors
    torch::Tensor const& qkv_input, std::optional<torch::Tensor> const& q_output,
    std::optional<torch::Tensor> const& qkv_bias, std::optional<torch::Tensor> const& seq_lens,
    std::optional<torch::Tensor> const& cache_seq_lens, std::optional<torch::Tensor> const& cu_seq_lens,
    std::optional<torch::Tensor> const& rotary_embedding_inv_freq,
    std::optional<torch::Tensor> const& rotary_coef_cache_buffer,
    std::optional<torch::Tensor> const& qkv_scale_orig_quant, std::optional<torch::Tensor> const& qkv_scale_quant_orig,
    std::optional<torch::Tensor> const& spec_decoding_position_offsets,
    std::optional<torch::Tensor> const& mrope_rotary_cos_sin, std::optional<torch::Tensor> const& mrope_position_deltas,
    // Scalars
    int64_t batch_size, int64_t max_input_seq_len, int64_t max_kv_seq_len, int64_t cyclic_kv_cache_len,
    int64_t sink_token_len, int64_t token_num, bool remove_padding, bool is_last_chunk, bool cross_attention,
    int64_t head_num, int64_t kv_head_num, int64_t size_per_head, int64_t rotary_embedding_dim,
    double rotary_embedding_base, int64_t rotary_scale_type, double rotary_embedding_scale,
    int64_t rotary_embedding_max_positions, int64_t position_embedding_type, bool position_shift_enabled,
    bool separate_q_kv_output, bool quantized_fp8_output, bool generation_phase, int64_t multi_processor_count,
    int64_t rotary_vision_start, int64_t rotary_vision_length, int64_t quant_mode,
    // KV cache buffer
    KVBlockArray const& kv_cache_buffer)
{
    QKVPreprocessingParams<T, KVBlockArray> params;
    memset(&params, 0, sizeof(params));

    // Set tensor pointers
    params.qkv_input = static_cast<T*>(qkv_input.data_ptr());
    if (q_output.has_value())
    {
        params.q_output = static_cast<T*>(q_output->data_ptr());
    }
    if (qkv_bias.has_value())
    {
        params.qkv_bias = static_cast<T const*>(qkv_bias->data_ptr());
    }
    if (seq_lens.has_value())
    {
        params.seq_lens = seq_lens->data_ptr<int>();
    }
    if (cache_seq_lens.has_value())
    {
        params.cache_seq_lens = cache_seq_lens->data_ptr<int>();
    }
    if (cu_seq_lens.has_value())
    {
        params.cu_seq_lens = cu_seq_lens->data_ptr<int>();
    }
    if (rotary_embedding_inv_freq.has_value())
    {
        params.rotary_embedding_inv_freq = rotary_embedding_inv_freq->data_ptr<float>();
    }
    if (rotary_coef_cache_buffer.has_value())
    {
        params.rotary_coef_cache_buffer = static_cast<float2 const*>(rotary_coef_cache_buffer->data_ptr());
    }
    if (qkv_scale_orig_quant.has_value())
    {
        params.qkv_scale_orig_quant = qkv_scale_orig_quant->data_ptr<float>();
    }
    if (qkv_scale_quant_orig.has_value())
    {
        params.qkv_scale_quant_orig = qkv_scale_quant_orig->data_ptr<float>();
    }
    if (spec_decoding_position_offsets.has_value())
    {
        params.spec_decoding_position_offsets = spec_decoding_position_offsets->data_ptr<int>();
    }
    if (mrope_rotary_cos_sin.has_value())
    {
        params.mrope_rotary_cos_sin = static_cast<float2 const*>(mrope_rotary_cos_sin->data_ptr());
    }
    if (mrope_position_deltas.has_value())
    {
        params.mrope_position_deltas = mrope_position_deltas->data_ptr<int32_t>();
    }

    // Set KV cache buffer
    params.kv_cache_buffer = kv_cache_buffer;

    // Set scalars
    params.batch_size = static_cast<int>(batch_size);
    params.max_input_seq_len = static_cast<int>(max_input_seq_len);
    params.max_kv_seq_len = static_cast<int>(max_kv_seq_len);
    params.cyclic_kv_cache_len = static_cast<int>(cyclic_kv_cache_len);
    params.sink_token_len = static_cast<int>(sink_token_len);
    params.token_num = static_cast<int>(token_num);
    params.remove_padding = remove_padding;
    params.is_last_chunk = is_last_chunk;
    params.cross_attention = cross_attention;
    params.head_num = static_cast<int>(head_num);
    params.kv_head_num = static_cast<int>(kv_head_num);
    params.qheads_per_kv_head = kv_head_num > 0 ? static_cast<int>(head_num / kv_head_num) : static_cast<int>(head_num);
    params.size_per_head = static_cast<int>(size_per_head);
    params.rotary_embedding_dim = static_cast<int>(rotary_embedding_dim);
    params.rotary_embedding_base = static_cast<float>(rotary_embedding_base);
    params.rotary_scale_type = intToRotaryScalingType(rotary_scale_type);
    params.rotary_embedding_scale = static_cast<float>(rotary_embedding_scale);
    params.rotary_embedding_max_positions = static_cast<int>(rotary_embedding_max_positions);
    params.position_embedding_type = intToPositionEmbeddingType(position_embedding_type);
    params.position_shift_enabled = position_shift_enabled;
    params.cache_type = getKvCacheDataType(tc::QuantMode(static_cast<uint32_t>(quant_mode)), qkv_input.scalar_type());
    params.separate_q_kv_output = separate_q_kv_output;
    params.quantized_fp8_output = quantized_fp8_output;
    params.generation_phase = generation_phase;
    params.multi_processor_count = static_cast<int>(multi_processor_count);
    params.rotary_vision_start = static_cast<int>(rotary_vision_start);
    params.rotary_vision_length = static_cast<int>(rotary_vision_length);

    // Compute derived parameters
    params.setCommonParameters();

    return params;
}

} // anonymous namespace

/**
 * @brief Run QKV preprocessing.
 *
 * This function applies RoPE and updates the KV cache for both context
 * and generation phases. All parameters are passed directly.
 *
 * @param qkv_input Input QKV tensor
 * @param q_output Optional output Q tensor (for separate Q/KV output)
 * @param qkv_bias Optional QKV bias tensor
 * @param seq_lens Sequence lengths tensor
 * @param cache_seq_lens Cache sequence lengths tensor
 * @param cu_seq_lens Cumulative sequence lengths tensor
 * @param rotary_embedding_inv_freq Rotary embedding inverse frequencies
 * @param rotary_coef_cache_buffer Rotary coefficient cache buffer
 * @param qkv_scale_orig_quant QKV scale for original to quantized conversion
 * @param qkv_scale_quant_orig QKV scale for quantized to original conversion
 * @param spec_decoding_position_offsets Speculative decoding position offsets
 * @param mrope_rotary_cos_sin Multi-rope rotary cos/sin values
 * @param mrope_position_deltas Multi-rope position deltas
 * @param block_offsets KV cache block offsets tensor
 * @param batch_size Batch size
 * @param max_input_seq_len Maximum input sequence length
 * @param max_kv_seq_len Maximum KV sequence length
 * @param cyclic_kv_cache_len Cyclic KV cache length
 * @param sink_token_len Sink token length
 * @param token_num Total number of tokens
 * @param remove_padding Whether to remove padding
 * @param is_last_chunk Whether this is the last chunk
 * @param cross_attention Whether this is cross attention
 * @param head_num Number of attention heads
 * @param kv_head_num Number of KV heads
 * @param size_per_head Size per attention head
 * @param rotary_embedding_dim Rotary embedding dimension
 * @param rotary_embedding_base Rotary embedding base
 * @param rotary_scale_type Rotary scaling type
 * @param rotary_embedding_scale Rotary embedding scale
 * @param rotary_embedding_max_positions Rotary embedding max positions
 * @param position_embedding_type Position embedding type
 * @param position_shift_enabled Whether position shift is enabled
 * @param separate_q_kv_output Whether to separate Q and KV output
 * @param quantized_fp8_output Whether to output quantized FP8
 * @param generation_phase Whether this is generation phase
 * @param multi_processor_count Number of multiprocessors
 * @param rotary_vision_start Rotary vision start position
 * @param rotary_vision_length Rotary vision length
 * @param quant_mode Quantization mode
 * @param tokens_per_block Tokens per KV cache block
 * @param max_blocks_per_sequence Maximum blocks per sequence
 * @param attention_window_size Attention window size
 * @param size_per_token Size per token in bytes
 * @param sink_token_length Sink token length for KV cache
 * @param max_cyclic_attention_window_size Maximum cyclic attention window size
 * @param can_use_one_more_block Whether can use one more block (beam width > 1)
 * @param host_primary_pool_pointer Host primary pool pointer (as int64)
 * @param host_secondary_pool_pointer Host secondary pool pointer (as int64)
 */
void runQkvPreprocessing(
    // Tensors
    torch::Tensor const& qkv_input, std::optional<torch::Tensor> const& q_output,
    std::optional<torch::Tensor> const& qkv_bias, std::optional<torch::Tensor> const& seq_lens,
    std::optional<torch::Tensor> const& cache_seq_lens, std::optional<torch::Tensor> const& cu_seq_lens,
    std::optional<torch::Tensor> const& rotary_embedding_inv_freq,
    std::optional<torch::Tensor> const& rotary_coef_cache_buffer,
    std::optional<torch::Tensor> const& qkv_scale_orig_quant, std::optional<torch::Tensor> const& qkv_scale_quant_orig,
    std::optional<torch::Tensor> const& spec_decoding_position_offsets,
    std::optional<torch::Tensor> const& mrope_rotary_cos_sin, std::optional<torch::Tensor> const& mrope_position_deltas,
    torch::Tensor const& block_offsets,
    // Scalars
    int64_t batch_size, int64_t max_input_seq_len, int64_t max_kv_seq_len, int64_t cyclic_kv_cache_len,
    int64_t sink_token_len, int64_t token_num, bool remove_padding, bool is_last_chunk, bool cross_attention,
    int64_t head_num, int64_t kv_head_num, int64_t size_per_head, int64_t rotary_embedding_dim,
    double rotary_embedding_base, int64_t rotary_scale_type, double rotary_embedding_scale,
    int64_t rotary_embedding_max_positions, int64_t position_embedding_type, bool position_shift_enabled,
    bool separate_q_kv_output, bool quantized_fp8_output, bool generation_phase, int64_t multi_processor_count,
    int64_t rotary_vision_start, int64_t rotary_vision_length, int64_t quant_mode,
    // KV cache buffer parameters
    int64_t tokens_per_block, int64_t max_blocks_per_sequence, int64_t attention_window_size, int64_t size_per_token,
    int64_t sink_token_length, int64_t max_cyclic_attention_window_size, bool can_use_one_more_block,
    int64_t host_primary_pool_pointer, int64_t host_secondary_pool_pointer)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const dtype = qkv_input.scalar_type();

    // Create KV cache buffer from parameters
    void* primaryPoolPtr = reinterpret_cast<void*>(host_primary_pool_pointer);
    void* secondaryPoolPtr = reinterpret_cast<void*>(host_secondary_pool_pointer);
    auto* blockOffsetsPtr = static_cast<KVBlockArray::DataType*>(block_offsets.data_ptr());

    KVBlockArray kvCacheBuffer = createKVBlockArray(static_cast<int>(batch_size),
        static_cast<int>(max_blocks_per_sequence), static_cast<int>(tokens_per_block), static_cast<int>(size_per_token),
        static_cast<int>(attention_window_size), static_cast<int>(max_cyclic_attention_window_size),
        static_cast<int>(sink_token_length), can_use_one_more_block, primaryPoolPtr, secondaryPoolPtr, blockOffsetsPtr);

    // Run preprocessing based on dtype
    if (dtype == at::ScalarType::Half)
    {
        auto params = buildQkvPreprocessingParams<half>(qkv_input, q_output, qkv_bias, seq_lens, cache_seq_lens,
            cu_seq_lens, rotary_embedding_inv_freq, rotary_coef_cache_buffer, qkv_scale_orig_quant,
            qkv_scale_quant_orig, spec_decoding_position_offsets, mrope_rotary_cos_sin, mrope_position_deltas,
            batch_size, max_input_seq_len, max_kv_seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
            remove_padding, is_last_chunk, cross_attention, head_num, kv_head_num, size_per_head, rotary_embedding_dim,
            rotary_embedding_base, rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions,
            position_embedding_type, position_shift_enabled, separate_q_kv_output, quantized_fp8_output,
            generation_phase, multi_processor_count, rotary_vision_start, rotary_vision_length, quant_mode,
            kvCacheBuffer);
        qkvPreprocessingHelper<half, KVBlockArray>(params, stream);
    }
#ifdef ENABLE_BF16
    else if (dtype == at::ScalarType::BFloat16)
    {
        auto params = buildQkvPreprocessingParams<__nv_bfloat16>(qkv_input, q_output, qkv_bias, seq_lens,
            cache_seq_lens, cu_seq_lens, rotary_embedding_inv_freq, rotary_coef_cache_buffer, qkv_scale_orig_quant,
            qkv_scale_quant_orig, spec_decoding_position_offsets, mrope_rotary_cos_sin, mrope_position_deltas,
            batch_size, max_input_seq_len, max_kv_seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
            remove_padding, is_last_chunk, cross_attention, head_num, kv_head_num, size_per_head, rotary_embedding_dim,
            rotary_embedding_base, rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions,
            position_embedding_type, position_shift_enabled, separate_q_kv_output, quantized_fp8_output,
            generation_phase, multi_processor_count, rotary_vision_start, rotary_vision_length, quant_mode,
            kvCacheBuffer);
        qkvPreprocessingHelper<__nv_bfloat16, KVBlockArray>(params, stream);
    }
#endif
    else if (dtype == at::ScalarType::Float)
    {
        auto params = buildQkvPreprocessingParams<float>(qkv_input, q_output, qkv_bias, seq_lens, cache_seq_lens,
            cu_seq_lens, rotary_embedding_inv_freq, rotary_coef_cache_buffer, qkv_scale_orig_quant,
            qkv_scale_quant_orig, spec_decoding_position_offsets, mrope_rotary_cos_sin, mrope_position_deltas,
            batch_size, max_input_seq_len, max_kv_seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
            remove_padding, is_last_chunk, cross_attention, head_num, kv_head_num, size_per_head, rotary_embedding_dim,
            rotary_embedding_base, rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions,
            position_embedding_type, position_shift_enabled, separate_q_kv_output, quantized_fp8_output,
            generation_phase, multi_processor_count, rotary_vision_start, rotary_vision_length, quant_mode,
            kvCacheBuffer);
        qkvPreprocessingHelper<float, KVBlockArray>(params, stream);
    }
    else
    {
        TLLM_THROW("Unsupported dtype for QKV preprocessing");
    }
}

/**
 * @brief Run KV cache postprocessing.
 *
 * This function handles sparse KV cache updates after FMHA.
 * It should be called after the context FMHA computation for non-MLA attention.
 *
 * @param qkv_input Input QKV tensor (used for dtype detection)
 * @param sparse_kv_indices Sparse KV indices tensor (optional)
 * @param sparse_kv_offsets Sparse KV offsets tensor (optional)
 * @param block_offsets KV cache block offsets tensor
 * @param batch_size Batch size
 * @param is_last_chunk Whether this is the last chunk
 * @param head_num Number of attention heads
 * @param kv_head_num Number of KV heads
 * @param size_per_head Size per attention head
 * @param quant_mode Quantization mode
 * @param tokens_per_block Tokens per KV cache block
 * @param max_blocks_per_sequence Maximum blocks per sequence
 * @param attention_window_size Attention window size
 * @param size_per_token Size per token in bytes
 * @param sink_token_length Sink token length for KV cache
 * @param max_cyclic_attention_window_size Maximum cyclic attention window size
 * @param can_use_one_more_block Whether can use one more block (beam width > 1)
 * @param host_primary_pool_pointer Host primary pool pointer (as int64)
 * @param host_secondary_pool_pointer Host secondary pool pointer (as int64)
 */
void runKvCachePostprocessing(
    // Tensors
    torch::Tensor const& qkv_input, std::optional<torch::Tensor> const& sparse_kv_indices,
    std::optional<torch::Tensor> const& sparse_kv_offsets, torch::Tensor const& block_offsets,
    // Scalars
    int64_t batch_size, bool is_last_chunk, int64_t head_num, int64_t kv_head_num, int64_t size_per_head,
    int64_t quant_mode,
    // KV cache buffer parameters
    int64_t tokens_per_block, int64_t max_blocks_per_sequence, int64_t attention_window_size, int64_t size_per_token,
    int64_t sink_token_length, int64_t max_cyclic_attention_window_size, bool can_use_one_more_block,
    int64_t host_primary_pool_pointer, int64_t host_secondary_pool_pointer)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const dtype = qkv_input.scalar_type();

    // Create KV cache buffer from parameters
    void* primaryPoolPtr = reinterpret_cast<void*>(host_primary_pool_pointer);
    void* secondaryPoolPtr = reinterpret_cast<void*>(host_secondary_pool_pointer);
    auto* blockOffsetsPtr = static_cast<KVBlockArray::DataType*>(block_offsets.data_ptr());

    KVBlockArray kvCacheBuffer = createKVBlockArray(static_cast<int>(batch_size),
        static_cast<int>(max_blocks_per_sequence), static_cast<int>(tokens_per_block), static_cast<int>(size_per_token),
        static_cast<int>(attention_window_size), static_cast<int>(max_cyclic_attention_window_size),
        static_cast<int>(sink_token_length), can_use_one_more_block, primaryPoolPtr, secondaryPoolPtr, blockOffsetsPtr);

    // Build minimal params for postprocessing
    auto buildPostprocessParams = [&](auto* dummy)
    {
        using T = std::remove_pointer_t<decltype(dummy)>;
        QKVPreprocessingParams<T, KVBlockArray> params;
        memset(&params, 0, sizeof(params));

        // Set sparse KV cache pointers if provided
        if (sparse_kv_indices.has_value())
        {
            params.sparse_kv_indices = sparse_kv_indices->data_ptr<int>();
        }
        if (sparse_kv_offsets.has_value())
        {
            params.sparse_kv_offsets = sparse_kv_offsets->data_ptr<int>();
        }

        // Set KV cache buffer
        params.kv_cache_buffer = kvCacheBuffer;

        // Set required scalars
        params.batch_size = static_cast<int>(batch_size);
        params.is_last_chunk = is_last_chunk;
        params.head_num = static_cast<int>(head_num);
        params.kv_head_num = static_cast<int>(kv_head_num);
        params.qheads_per_kv_head
            = kv_head_num > 0 ? static_cast<int>(head_num / kv_head_num) : static_cast<int>(head_num);
        params.size_per_head = static_cast<int>(size_per_head);
        params.cache_type = getKvCacheDataType(tc::QuantMode(static_cast<uint32_t>(quant_mode)), dtype);

        return params;
    };

    // Run postprocessing based on dtype
    if (dtype == at::ScalarType::Half)
    {
        auto params = buildPostprocessParams(static_cast<half*>(nullptr));
        kvCachePostprocessingHelper<half, KVBlockArray>(params, stream);
    }
#ifdef ENABLE_BF16
    else if (dtype == at::ScalarType::BFloat16)
    {
        auto params = buildPostprocessParams(static_cast<__nv_bfloat16*>(nullptr));
        kvCachePostprocessingHelper<__nv_bfloat16, KVBlockArray>(params, stream);
    }
#endif
    else if (dtype == at::ScalarType::Float)
    {
        auto params = buildPostprocessParams(static_cast<float*>(nullptr));
        kvCachePostprocessingHelper<float, KVBlockArray>(params, stream);
    }
    else
    {
        TLLM_THROW("Unsupported dtype for KV cache postprocessing");
    }
}

/**
 * @brief Build decoder info (cu_seqlens and rotary_inv_freq).
 *
 * This function builds the cumulative sequence lengths and optionally computes
 * rotary inverse frequencies for the attention mechanism.
 */
std::vector<torch::Tensor> buildDecoderInfo(torch::Tensor const& seqLens, int64_t const batchSize,
    int64_t const maxInputLength, bool const removePadding, int64_t const attentionMaskType,
    int64_t const positionEmbeddingType, int64_t const rotaryEmbeddingDim, double const rotaryEmbeddingBase,
    int64_t const rotaryScaleType, double const rotaryEmbeddingScale, int64_t const rotaryEmbeddingMaxPositions,
    int64_t const rotaryEmbeddingOriginalMaxPositions, bool const specDecodingGenerationLengths,
    std::optional<torch::Tensor> specDecodingPositionOffsets)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Allocate output tensors
    auto cuSeqLens = torch::empty({batchSize + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto rotaryInvFreq = torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // If RoPE is enabled, allocate rotary_inv_freq
    auto posType = static_cast<tk::PositionEmbeddingType>(static_cast<int8_t>(positionEmbeddingType));
    if (posType == tk::PositionEmbeddingType::kROPE_GPTJ || posType == tk::PositionEmbeddingType::kROPE_GPT_NEOX
        || posType == tk::PositionEmbeddingType::kLONG_ROPE)
    {
        rotaryInvFreq
            = torch::empty({batchSize, rotaryEmbeddingDim / 2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    }

    // Build decoder info parameters
    tk::BuildDecoderInfoParams<float> params;
    memset(&params, 0, sizeof(params));
    params.seqQLengths = seqLens.data_ptr<int>();
    params.seqKVLengths = seqLens.data_ptr<int>();
    params.batchSize = batchSize;
    params.maxQSeqLength = maxInputLength;
    params.removePadding = removePadding;
    params.attentionMaskType = static_cast<tk::AttentionMaskType>(static_cast<int32_t>(attentionMaskType));
    params.seqQOffsets = cuSeqLens.data_ptr<int>();
    params.seqKVOffsets = cuSeqLens.data_ptr<int>();
    params.rotaryEmbeddingDim = rotaryEmbeddingDim;
    params.rotaryEmbeddingBase = static_cast<float>(rotaryEmbeddingBase);
    params.rotaryScalingType = static_cast<tk::RotaryScalingType>(static_cast<int8_t>(rotaryScaleType));
    params.rotaryEmbeddingScale = static_cast<float>(rotaryEmbeddingScale);
    params.rotaryEmbeddingInvFreq = rotaryInvFreq.numel() > 0 ? rotaryInvFreq.data_ptr<float>() : nullptr;
    params.rotaryEmbeddingMaxPositions = rotaryEmbeddingMaxPositions;

    // Note: specDecodingPositionOffsets is not currently supported in BuildDecoderInfoParams
    (void) specDecodingPositionOffsets;
    (void) specDecodingGenerationLengths;

    tk::invokeBuildDecoderInfo(params, stream);

    return {cuSeqLens, rotaryInvFreq};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// PyTorch bindings
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // Register the QKV preprocessing function with all parameters directly
    m.def(
        "run_qkv_preprocessing("
        // Tensors
        "Tensor qkv_input"
        ", Tensor? q_output"
        ", Tensor? qkv_bias"
        ", Tensor? seq_lens"
        ", Tensor? cache_seq_lens"
        ", Tensor? cu_seq_lens"
        ", Tensor? rotary_embedding_inv_freq"
        ", Tensor? rotary_coef_cache_buffer"
        ", Tensor? qkv_scale_orig_quant"
        ", Tensor? qkv_scale_quant_orig"
        ", Tensor? spec_decoding_position_offsets"
        ", Tensor? mrope_rotary_cos_sin"
        ", Tensor? mrope_position_deltas"
        ", Tensor block_offsets"
        // Scalars
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
        ", int size_per_head"
        ", int rotary_embedding_dim"
        ", float rotary_embedding_base"
        ", int rotary_scale_type"
        ", float rotary_embedding_scale"
        ", int rotary_embedding_max_positions"
        ", int position_embedding_type"
        ", bool position_shift_enabled"
        ", bool separate_q_kv_output"
        ", bool quantized_fp8_output"
        ", bool generation_phase"
        ", int multi_processor_count"
        ", int rotary_vision_start"
        ", int rotary_vision_length"
        ", int quant_mode"
        // KV cache buffer parameters
        ", int tokens_per_block"
        ", int max_blocks_per_sequence"
        ", int attention_window_size"
        ", int size_per_token"
        ", int sink_token_length"
        ", int max_cyclic_attention_window_size"
        ", bool can_use_one_more_block"
        ", int host_primary_pool_pointer"
        ", int host_secondary_pool_pointer"
        ") -> ()",
        &tensorrt_llm::torch_ext::runQkvPreprocessing);

    // Register the build decoder info function
    m.def(
        "build_decoder_info("
        "Tensor seq_lens"
        ", int batch_size"
        ", int max_input_length"
        ", bool remove_padding"
        ", int attention_mask_type"
        ", int position_embedding_type"
        ", int rotary_embedding_dim"
        ", float rotary_embedding_base"
        ", int rotary_scale_type"
        ", float rotary_embedding_scale"
        ", int rotary_embedding_max_positions"
        ", int rotary_embedding_original_max_positions"
        ", bool spec_decoding_generation_lengths"
        ", Tensor? spec_decoding_position_offsets"
        ") -> Tensor[]",
        &tensorrt_llm::torch_ext::buildDecoderInfo);

    // Register the KV cache postprocessing function
    m.def(
        "run_kv_cache_postprocessing("
        // Tensors
        "Tensor qkv_input"
        ", Tensor? sparse_kv_indices"
        ", Tensor? sparse_kv_offsets"
        ", Tensor block_offsets"
        // Scalars
        ", int batch_size"
        ", bool is_last_chunk"
        ", int head_num"
        ", int kv_head_num"
        ", int size_per_head"
        ", int quant_mode"
        // KV cache buffer parameters
        ", int tokens_per_block"
        ", int max_blocks_per_sequence"
        ", int attention_window_size"
        ", int size_per_token"
        ", int sink_token_length"
        ", int max_cyclic_attention_window_size"
        ", bool can_use_one_more_block"
        ", int host_primary_pool_pointer"
        ", int host_secondary_pool_pointer"
        ") -> ()",
        &tensorrt_llm::torch_ext::runKvCachePostprocessing);
}
