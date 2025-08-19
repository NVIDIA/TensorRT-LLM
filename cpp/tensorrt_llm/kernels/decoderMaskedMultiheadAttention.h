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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Round up to next higher power of 2 (return x if it's already a power
/// of 2).
inline int pow2roundup(int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template <typename T>
struct Multihead_attention_params_base
{

    // The output buffer. Dimensions B x D.
    void* out = nullptr;

    // The input Qs and the associated bias. Dimensions B x D and D, resp.
    T const *q = nullptr, *q_bias = nullptr;
    // The input Ks and the associated bias. Dimensions B x D and D, resp.
    T const *k = nullptr, *k_bias = nullptr;
    // The input Vs and the associated bias. Dimensions B x D and D, resp.
    T const *v = nullptr, *v_bias = nullptr;

    // The indirections to use for cache when beam sampling.
    int const* cache_indir = nullptr;

    // scales
    float const* query_weight_output_scale = nullptr;
    float const* attention_qk_scale = nullptr;
    float const* attention_output_weight_input_scale_inv = nullptr;

    // Stride to handle the case when KQV is a single buffer
    int stride = 0;

    // The batch size.
    int batch_size = 0;
    // The beam width
    int beam_width = 0;
    // The chunked attention size.
    int chunked_attention_size = INT_MAX;
    // The chunked attention size in log2.
    int chunked_attention_size_log2 = 0;
    // By default, max_attention_window_size == cyclic_attention_window_size
    // unless each layer has different cyclic kv cache length.
    // Max cache capacity (used to allocate KV cache)
    int max_attention_window_size = 0;
    // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
    int cyclic_attention_window_size = 0;
    // Length of the sink token in KV cache
    int sink_token_length = 0;
    // The number of heads (H).
    int num_heads = 0;
    // Controls MHA/MQA/GQA
    int num_kv_heads = 0;
    // The hidden dimension per head (Dh).
    int hidden_size_per_head = 0;
    // Rotary position embedding type
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::kLEARNED_ABSOLUTE;
    // The per-head latent space reserved for rotary embeddings.
    int rotary_embedding_dim = 0;
    float rotary_embedding_base = 0.0f;
    RotaryScalingType rotary_embedding_scale_type = RotaryScalingType::kNONE;
    float rotary_embedding_scale = 1.0f;
    // The pre-computed rotary inv freq when building the engines (as constant weights).
    float const* rotary_embedding_inv_freq_cache = nullptr;
    // The pre-computed cos/sin cache.
    float2 const* rotary_embedding_cos_sin_cache = nullptr;
    float rotary_embedding_short_m_scale = 1.0f;
    float rotary_embedding_long_m_scale = 1.0f;
    int rotary_embedding_max_positions = 0;
    int rotary_embedding_original_max_positions = 0;
    int rotary_cogvlm_vision_start = -1;
    int rotary_cogvlm_vision_length = -1;
    // Position shift for streamingllm
    bool position_shift_enabled = false;
    // The current timestep. TODO Check that do we only this param in cross attention?
    int timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)

    // The 1.f / sqrt(Dh). Computed on the host.
    float inv_sqrt_dh = 0.0f;

    // The attention logit softcapping scale.
    float attn_logit_softcapping_scale = 0.0f;
    float attn_logit_softcapping_inverse_scale = 0.0f;

    // The attention mask [batch_size, attention_mask_stride (i.e. max_kv_seqlen)]
    bool const* attention_mask = nullptr;
    int attention_mask_stride = 0;

    // The attention sinks [num_heads_q].
    float const* attention_sinks = nullptr;

    // If relative position embedding is used
    T const* relative_attention_bias = nullptr;
    int relative_attention_bias_stride = 0;
    int max_distance = 0;

    // If logn scaling is used
    float const* logn_scaling_ptr = nullptr;

    // block sparse config
    bool block_sparse_attention = false;
    BlockSparseParams block_sparse_params{64, false, 16, 8};

    // The slope per head of linear position bias to attention score (H).
    T const* linear_bias_slopes = nullptr;

    T const* ia3_key_weights = nullptr;
    T const* ia3_value_weights = nullptr;
    int const* ia3_tasks = nullptr;

    float const* qkv_scale_quant_orig = nullptr;
    float const* attention_out_scale_orig_quant = nullptr;

    // 8 bits kv cache scales.
    float const* kv_scale_orig_quant = nullptr;
    float const* kv_scale_quant_orig = nullptr;

    bool int8_kv_cache = false;
    bool fp8_kv_cache = false;

    // Multi-block setups
    mutable bool multi_block_mode = true;

    // Number of streaming processors on the device.
    // Tune block size to maximum occupancy.
    int multi_processor_count = 1;

    mutable int timesteps_per_block = 1;
    mutable int seq_len_tile = 1;

    mutable int min_seq_len_tile = 1;
    mutable int max_seq_len_tile = 1;
    // The partial output buffer. Dimensions max_seq_len_tile x B x D. (for each timestep only seq_len_tile x B x D is
    // needed)
    T* partial_out = nullptr;
    // ThreadBlock sum. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float* partial_sum = nullptr;
    // ThreadBlock max. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float* partial_max = nullptr;
    // threadblock counter to identify the complete of partial attention computations
    int* block_counter = nullptr;

    // sparse indices and offsets for attention calculation
    int32_t const* sparse_attn_indices = nullptr;
    int32_t const* sparse_attn_offsets = nullptr;
    int32_t num_sparse_attn_indices = 0;

    int const* memory_length_per_sample = nullptr;
    int32_t const* mrope_position_deltas = nullptr;
};

template <typename T, bool USE_CROSS_ATTENTION = false>
struct Multihead_attention_params;

// self-attention params
template <typename T>
struct Multihead_attention_params<T, false> : public Multihead_attention_params_base<T>
{
    static constexpr bool DO_CROSS_ATTENTION = false;

    int max_decoder_seq_len = 0;

    // allows to exit attention early
    bool* finished = nullptr;

    // required in case of masked attention with different length
    int const* length_per_sample = nullptr;

    // input lengths to identify the paddings (i.e. input seq < padding < new generated seq).
    int const* input_lengths = nullptr;
};
template <class T>
using Masked_multihead_attention_params = Multihead_attention_params<T, false>;

// cross-attention params
template <typename T>
struct Multihead_attention_params<T, true> : public Multihead_attention_params_base<T>
{
    static constexpr bool DO_CROSS_ATTENTION = true;

    int max_decoder_seq_len = 0;

    // allows to exit attention early
    bool* finished = nullptr;

    // required in case of masked attention with different length
    int const* length_per_sample = nullptr;

    // input lengths to identify the paddings (i.e. input seq < padding < new generated seq).
    int const* input_lengths = nullptr;
};
template <class T>
using Cross_multihead_attention_params = Multihead_attention_params<T, true>;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Pre-check whether head size is supported when building engines.
bool mmha_supported(int head_size);

#define DECLARE_MMHA_NORMAL_AND_PAGED(T)                                                                               \
    void masked_multihead_attention(const Masked_multihead_attention_params<T>& params,                                \
        const KVBlockArray& block_array, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);             \
    void masked_multihead_attention(const Masked_multihead_attention_params<T>& params,                                \
        const KVLinearBuffer& kv_cache_buffer, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);       \
    void masked_multihead_attention(const Cross_multihead_attention_params<T>& params,                                 \
        const KVBlockArray& block_array, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);             \
    void masked_multihead_attention(const Cross_multihead_attention_params<T>& params,                                 \
        const KVLinearBuffer& kv_cache_buffer, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);
DECLARE_MMHA_NORMAL_AND_PAGED(float);
DECLARE_MMHA_NORMAL_AND_PAGED(uint16_t);
#ifdef ENABLE_BF16
DECLARE_MMHA_NORMAL_AND_PAGED(__nv_bfloat16);
#endif
#undef DECLARE_MMHA_NORMAL_AND_PAGED

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int estimate_min_multi_block_count(int max_timesteps, int max_dynamic_shmem_per_block, int num_bytes_per_elt)
{
    auto const qk_elts = static_cast<int>((max_timesteps + 1 + 4 - 1) / 4);
    int size_per_elts = 16;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (num_bytes_per_elt != 4)
    {
        size_per_elts += 4 * num_bytes_per_elt;
    }
#endif
    int elts_per_block = max_dynamic_shmem_per_block / size_per_elts;
    int min_block_count = (qk_elts + elts_per_block - 1) / elts_per_block;
    return std::max(1, min_block_count);
}

} // namespace kernels
} // namespace tensorrt_llm
