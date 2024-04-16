/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include <cuda_runtime_api.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeAddQKVBiasIA3Transpose(T* q_buf, T* k_buf, T* v_buf, T* Q, T const* bias_Q, T* K, T const* bias_K, T* V,
    T const* bias_V, int const batch_size, int const seq_len, int const head_num, int const size_per_head,
    int const* ia3_tasks, T const* ia3_key_weights, T const* ia3_value_weights, cudaStream_t stream);

template <typename T, typename T_IN>
struct MaskedSoftmaxParam
{
    // Common parameters.
    T* attention_score = nullptr;      // (batch_size, head_num, q_length, k_length)
    const T_IN* qk = nullptr;          // (batch_size, head_num, q_length, k_length)
    T const* attention_mask = nullptr; // (batch_size, q_length, k_length)
    int batch_size = 0;
    int q_length = 0;
    int k_length = 0;
    int num_heads = 0;
    T qk_scale = T(0.0f);

    // Optional parameters that depend on the type of attention.
    // The slopes of the linear position bias of ALiBi.
    T const* linear_bias_slopes = nullptr; // (head_num,), optional
};

enum class KvCacheDataType
{
    BASE = 0,
    INT8,
    FP8
};

enum class RotaryPositionEmbeddingType
{
    NONE = 0,
    GPTJ,
    GPT_NEOX,
};

template <typename T, typename KVCacheBuffer>
struct QKVPreprocessingParams
{
    // Buffers.
    T* QKV;
    // Only used by fp8 quantized output currently.
    void* QuantizedQKV;
    T* Q;
    KVCacheBuffer const kv_cache_buffer;
    T const* qkv_bias;
    int const* seq_lens;
    int const* cache_seq_lens;
    int const* cu_seq_lens;
    float const* rotary_embedding_inv_freq;
    float2 const* rotary_coef_cache_buffer;
    float const* kvScaleOrigQuant;
    int const* medusa_position_offsets;

    // Scalars.
    int const batch_size;
    int const max_input_seq_len;
    int const max_kv_seq_len;
    int const cyclic_kv_cache_len;
    int const sink_token_len;
    int const token_num;
    int const head_num;
    int const kv_head_num;
    int const qheads_per_kv_head;
    int const size_per_head;
    int const rotary_embedding_dim;
    float const rotary_embedding_base;
    RotaryScalingType const rotary_scale_type;
    float rotary_embedding_scale;
    int const rotary_embedding_max_positions;
    PositionEmbeddingType const position_embedding_type;
    bool const position_shift_enabled;
    const KvCacheDataType cache_type;
    bool const enable_paged_kv_fmha;
    bool const quantized_fp8_output;
    int const multi_processor_count;

    // Pre-compute on host.
    int half_rotary_dim;
    int q_hidden_size;
    int kv_hidden_size;
    int hidden_size;

    void setCommonParameters()
    {
        half_rotary_dim = rotary_embedding_dim / 2;
        q_hidden_size = head_num * size_per_head;
        kv_hidden_size = kv_head_num * size_per_head;
        hidden_size = q_hidden_size + 2 * kv_hidden_size;
    }
};

template <typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream);

template <typename T>
void invokeTransposeQKV(T* dst, T* src, int const batch_size, int const seq_len, int const head_num,
    int const size_per_head, float const* scale, int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddQKVBiasIA3RebuildPadding(T* Q, T const* bias_Q, T* K, T const* bias_K, T* V, T const* bias_V, T* q_buf,
    T* k_buf, T* v_buf, int const batch_size, int const seq_len, int const head_num, int const size_per_head,
    int const valid_word_num, int const* mask_offset, int const* ia3_tasks, T const* ia3_key_weights,
    T const* ia3_value_weights, cudaStream_t stream);

template <typename T>
void invokeTransposeAttentionOutRemovePadding(T* src, T* dst, int const valid_word_num, int const batch_size,
    int const seq_len, int const head_num, int const size_per_head, int const* mask_offset, float const* scale,
    int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, T const* qkv_bias, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float rotary_embedding_scale, int const rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, float const* scale, int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, T const* qkv_bias, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf, k_buf, v_buf, QKV, qkv_bias, seq_lens, padding_offset, batch_size, seq_len,
        token_num, head_num, kv_head_num, size_per_head, 0, false, (float*) nullptr, 0, stream);
}

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float rotary_embedding_scale, int const rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, float const* scale, int const int8_mode, cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf, k_buf, v_buf, QKV, (T const*) nullptr, seq_lens, padding_offset, batch_size,
        seq_len, token_num, head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base,
        rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, scale,
        int8_mode, stream);
}

template <typename T, typename KVCacheBuffer>
void invokeTranspose4dBatchMajor(T const* k_src, T const* v_src, KVCacheBuffer& kvTable, int const local_batch_size,
    int const seq_len, int const max_attention_window_size, int const size_per_head, int const local_head_num,
    const KvCacheDataType cache_type, float const* kvScaleOrigQuant, int const* sequence_lengths, cudaStream_t stream);

template <typename T, typename T_cache, typename KVCacheBuffer>
void invokeApplyBiasRopeUpdateKVCacheDispatch(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);

// NOTE: this kernel is in-place, QKV will be modified, if other kernels need that, may need copy or use before it.
template <typename T, typename KVCacheBuffer>
void invokeQKVPreprocessing(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    params.setCommonParameters();
    if (params.cache_type == KvCacheDataType::INT8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer>(params, stream);
    }
#ifdef ENABLE_FP8
    else if (params.cache_type == KvCacheDataType::FP8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, __nv_fp8_e4m3, KVCacheBuffer>(params, stream);
    }
#endif // ENABLE_FP8
    else
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, T, KVCacheBuffer>(params, stream);
    }
}

template <typename T, typename BT>
void invokeAddRelativeAttentionBiasUnaligned(T* qk_buf, const BT* relative_attention_bias, int const batch_size,
    int const head_num, int const seq_len, int const max_seq_len, cudaStream_t stream, bool implicit = false,
    int num_buckets = 0, int max_distance = 0, bool bidirectional = true);

template <typename T, typename KVCacheBuffer>
void invokeShiftKCache(KVCacheBuffer const& kvCacheBuffer, KVLinearBuffer const& shiftKCacheBuffer,
    const KvCacheDataType cache_type, int const sizePerHead, int const timestep, int const batch_beam,
    int const kv_head_num, int const beam_width, int const maxKCacheLen, int const sinkTokenLen,
    float const* kScaleQuantOrig, int const* sequence_lengths, int const* input_lengths, int const rotary_embedding_dim,
    float rotary_embedding_base, RotaryScalingType const rotary_scale_type, float rotary_embedding_scale,
    int const rotary_embedding_max_positions, PositionEmbeddingType const position_embedding_type, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
