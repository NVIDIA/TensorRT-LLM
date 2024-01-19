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
void invokeAddQKVBiasIA3Transpose(T* q_buf, T* k_buf, T* v_buf, T* Q, const T* bias_Q, T* K, const T* bias_K, T* V,
    const T* bias_V, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
    const int* ia3_tasks, const T* ia3_key_weights, const T* ia3_value_weights, cudaStream_t stream);

template <typename T, typename T_IN>
struct MaskedSoftmaxParam
{
    // Common parameters.
    T* attention_score = nullptr;      // (batch_size, head_num, q_length, k_length)
    const T_IN* qk = nullptr;          // (batch_size, head_num, q_length, k_length)
    const T* attention_mask = nullptr; // (batch_size, q_length, k_length)
    int batch_size = 0;
    int q_length = 0;
    int k_length = 0;
    int num_heads = 0;
    T qk_scale = T(0.0f);

    // Optional parameters that depend on the type of attention.
    // The slopes of the linear position bias of ALiBi.
    const T* linear_bias_slopes = nullptr; // (head_num,), optional
};

enum class KvCacheDataType
{
    BASE = 0,
    INT8,
    FP8
};

template <typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream);

template <typename T>
void invokeTransposeQKV(T* dst, T* src, const int batch_size, const int seq_len, const int head_num,
    const int size_per_head, const float* scale, const int int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddQKVBiasIA3RebuildPadding(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf,
    T* k_buf, T* v_buf, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
    const int valid_word_num, const int* mask_offset, const int* ia3_tasks, const T* ia3_key_weights,
    const T* ia3_value_weights, cudaStream_t stream);

template <typename T>
void invokeTransposeAttentionOutRemovePadding(T* src, T* dst, const int valid_word_num, const int batch_size,
    const int seq_len, const int head_num, const int size_per_head, const int* mask_offset, const float* scale,
    const int int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, const T* qkv_bias, const int* seq_lens,
    const int* padding_offset, const int batch_size, const int seq_len, const int token_num, const int head_num,
    const int kv_head_num, const int size_per_head, const int rotary_embedding_dim, float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float rotary_embedding_scale, const int rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, const float* scale, const int int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, const T* qkv_bias, const int* seq_lens,
    const int* padding_offset, const int batch_size, const int seq_len, const int token_num, const int head_num,
    const int kv_head_num, const int size_per_head, cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf, k_buf, v_buf, QKV, qkv_bias, seq_lens, padding_offset, batch_size, seq_len,
        token_num, head_num, kv_head_num, size_per_head, 0, false, (float*) nullptr, 0, stream);
}

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, const int* seq_lens,
    const int* padding_offset, const int batch_size, const int seq_len, const int token_num, const int head_num,
    const int kv_head_num, const int size_per_head, const int rotary_embedding_dim, float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float rotary_embedding_scale, const int rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, const float* scale, const int int8_mode, cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf, k_buf, v_buf, QKV, (const T*) nullptr, seq_lens, padding_offset, batch_size,
        seq_len, token_num, head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base,
        rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, scale,
        int8_mode, stream);
}

template <typename T, typename KVCacheBuffer>
void invokeTranspose4dBatchMajor(const T* k_src, const T* v_src, KVCacheBuffer& kvTable, const int local_batch_size,
    const int seq_len, const int max_attention_window_size, const int size_per_head, const int local_head_num,
    const KvCacheDataType cache_type, const float* kvScaleOrigQuant, const int* sequence_lengths, cudaStream_t stream);

// NOTE: this kernel is in-place, QKV will be modified, if other kernels need that, may need copy or use before it.
template <typename T, typename KVCacheBuffer, bool IsGenerate = false>
void invokeApplyBiasRopeUpdateKVCache(T* QKV, T* Q, KVCacheBuffer& kvTable, const T* qkv_bias, const int* seq_lens,
    const int* kv_seq_lens, const int* padding_offset, const int batch_size, const int seq_len,
    const int cyclic_kv_cache_len, const int sink_token_len, const int token_num, const int head_num,
    const int kv_head_num, const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,
    const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
    const int* medusa_position_offsets, const bool position_shift_enabled, const float* scale, const int int8_mode,
    const KvCacheDataType cache_type, const float* kvScaleOrigQuant, const bool enable_paged_kv_fmha,
    const int beam_width, int2& grid_block_cache, cudaStream_t stream);

template <typename T, typename BT>
void invokeAddRelativeAttentionBiasUnaligned(T* qk_buf, const BT* relative_attention_bias, const int batch_size,
    const int head_num, const int seq_len, const int max_seq_len, cudaStream_t stream, bool implicit = false,
    int num_buckets = 0, int max_distance = 0, bool bidirectional = true);

template <typename T, typename KVCacheBuffer>
void invokeShiftKCache(KVCacheBuffer kvCacheBuffer, KVLinearBuffer shiftKCacheBuffer, const KvCacheDataType cache_type,
    const int sizePerHead, const int timestep, const int batch_beam, const int kv_head_num, const int beam_width,
    const int maxKCacheLen, const int sinkTokenLen, const float* kScaleQuantOrig, const int* sequence_lengths,
    const int* input_lengths, const int rotary_embedding_dim, float rotary_embedding_base,
    RotaryScalingType const rotary_scale_type, float rotary_embedding_scale, const int rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
