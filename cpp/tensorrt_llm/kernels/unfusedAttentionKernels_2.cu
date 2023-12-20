/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

// Separate from unfusedAttentionKernel to accelerate compiling.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
struct Vec_t
{
    static constexpr int size = 0;
};

template <>
struct Vec_t<float>
{
    using Type = float2;
    static constexpr int size = 2;
};

template <>
struct Vec_t<half>
{
    using Type = uint32_t;
    static constexpr int size = 2;
};

#ifdef ENABLE_BF16
template <>
struct Vec_t<__nv_bfloat16>
{
    using Type = __nv_bfloat162;
    static constexpr int size = 2;
};
#endif

template <typename T, typename T_cache, bool ADD_BIAS, bool STORE_QKV, typename KVCacheBuffer, bool IsGenerate>
__global__ void applyBiasRopeUpdateKVCache(T* QKV, T* Q, KVCacheBuffer kvCacheBuffer, const T* __restrict qkv_bias,
    const int* seq_lens, const int* kv_seq_lens, const int* padding_offset, const float* kvScaleOrigQuant,
    const int batch_size, const int seq_len, const int cyclic_kv_cache_len, const int head_num, const int kv_head_num,
    const int size_per_head, const int rotary_embedding_dim, float rotary_embedding_base,
    RotaryScalingType const rotary_scale_type, float rotary_embedding_scale, const int rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, int beam_width)
{
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // NOTE:
    // head_num == kv_head_num
    //   QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //                  ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                           m                        n
    //   QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    // head_num != kv_head_num
    //   QKV src shape: (batch_size, seq_len, head_num * size_per_head + 2 * kv_head_num * size_per_head)
    //                   ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                             m                               n
    //   Q dst shape: (batch_size, head_num, seq_len, size_per_head)
    //   KV dst shape: (batch_size, kv_head_num, seq_len, size_per_head)

    // FMHA with paged kv cache input:
    // Need separate contiguous Q buffer.

    extern __shared__ __align__(sizeof(float2)) char smem_[]; // align on largest vector type

    constexpr int vec_size = Vec_t<T>::size;
    using Vec_t = typename Vec_t<T>::Type;
    const int token_idx = blockIdx.x;
    const bool has_padding = padding_offset == nullptr;

    constexpr bool ENABLE_8BITS_CACHE = sizeof(T_cache) == 1;
    constexpr int X_ELEMS = vec_size;
    const int sizePerHeadDivX = size_per_head / X_ELEMS;
    using T_dst = T_cache;

    // The index of the token in the batch. It includes "virtual" padding (even if the input is not padded)
    // such that the sequence index and the position in the sequence can be obtained using the max.
    // sequence length as:
    const int token_padding_offset = (has_padding || IsGenerate) ? 0 : padding_offset[token_idx];
    const int global_token_idx = (!IsGenerate) ? token_idx + token_padding_offset : token_idx;
    const int batch_beam_idx = global_token_idx / seq_len;
    const int batch_idx = (!IsGenerate) ? batch_beam_idx : batch_beam_idx / beam_width;
    const int final_kv_seq_len = (!IsGenerate) ? kv_seq_lens[batch_idx] : 0;
    const int actual_seq_len = seq_lens[batch_idx];
    const int token_idx_in_seq
        = (!IsGenerate) ? (final_kv_seq_len - actual_seq_len) + global_token_idx % seq_len : actual_seq_len - 1;
    const bool valid_seq = IsGenerate || (token_idx_in_seq < actual_seq_len || !has_padding);

    const int head_idx = blockIdx.y;
    const int tidx = threadIdx.x;

    const bool is_seq_masked = !valid_seq;
    const bool is_head_size_masked = tidx * vec_size >= size_per_head;
    const bool is_masked = is_head_size_masked || is_seq_masked;

    const int hidden_size = head_num * size_per_head;
    const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
    const int qheads_per_kv_head = head_num / kv_head_num;
    const int kv_head_idx = head_idx / qheads_per_kv_head;
    const int hidden_idx_kv = kv_head_idx * size_per_head + tidx * vec_size;
    const int n = (head_num + 2 * kv_head_num) * size_per_head;

    const int dst_kv_seq_idx = token_idx_in_seq;
    const int src_k_offset = hidden_size;
    const int src_v_offset = hidden_size + kv_head_num * size_per_head;

    // NOTE: q has seq len excluding prefix prompt
    // head_num == kv_head_num:
    //   src QKV: [batch, time, 3, head_num, size_per_head]
    // head_num != kv_head_num:
    //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
    const int src_q_idx = token_idx * n + hidden_idx;
    const int src_k_idx = token_idx * n + src_k_offset + hidden_idx_kv;
    const int src_v_idx = token_idx * n + src_v_offset + hidden_idx_kv;

    Vec_t q, k, v, zero;
    Vec_t q_bias, k_bias, v_bias;
    if (valid_seq)
    {
        mmha::update_rotary_base_n_scale(rotary_embedding_base, rotary_embedding_scale, rotary_scale_type,
            rotary_embedding_dim, rotary_embedding_max_positions, actual_seq_len);
    }

#pragma unroll
    for (int i = 0; i < sizeof(Vec_t) / sizeof(uint32_t); i++)
    {
        reinterpret_cast<uint32_t*>(&zero)[i] = 0u;
    }

    // load q,k,v and add bias
    if (!is_masked)
    {
        q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (ADD_BIAS)
        {
            q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx_kv + src_k_offset]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx_kv + src_v_offset]);

            q = mmha::add(q, q_bias);
            k = mmha::add(k, k_bias);
            v = mmha::add(v, v_bias);
        }
    }

    switch (position_embedding_type)
    {
    case PositionEmbeddingType::kROPE_GPTJ:
    {
        mmha::apply_rotary_embedding(
            q, k, tidx, rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale, dst_kv_seq_idx);
        break;
    }
    case PositionEmbeddingType::kROPE_GPT_NEOX:
    {
        const bool do_rotary = !is_masked && vec_size * tidx < rotary_embedding_dim;

        T* q_smem = reinterpret_cast<T*>(smem_);
        T* k_smem = q_smem + rotary_embedding_dim;

        const int half_rotary_dim = rotary_embedding_dim / 2;
        const int half_idx = (tidx * vec_size) / half_rotary_dim;
        const int intra_half_idx = (tidx * vec_size) % half_rotary_dim;
        const int smem_pitch = half_rotary_dim; // TODO: adjust for bank conflicts?

        if (do_rotary)
        {
            *reinterpret_cast<Vec_t*>(q_smem + half_idx * smem_pitch + intra_half_idx) = q;
            *reinterpret_cast<Vec_t*>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
        }

        __syncthreads();

        const int transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor = vec_size / 2;
        if (do_rotary)
        {
            mmha::vec_from_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
            mmha::vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

            mmha::apply_rotary_embedding(q, k, transpose_idx / tidx_factor, rotary_embedding_dim, rotary_embedding_base,
                rotary_embedding_scale, dst_kv_seq_idx);

            mmha::write_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
            mmha::write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary)
        {
            q = *reinterpret_cast<Vec_t*>(q_smem + half_idx * smem_pitch + intra_half_idx);
            k = *reinterpret_cast<Vec_t*>(k_smem + half_idx * smem_pitch + intra_half_idx);
        }
        break;
    }
    }

    const int channelIdx{tidx};
    const bool valid_kv_cache_pos = kvCacheBuffer.data != nullptr // In KV-cache-less mode. No need to store KV values
        && token_idx_in_seq >= (actual_seq_len - cyclic_kv_cache_len);
    const int token_idx_in_kv_cache = token_idx_in_seq % cyclic_kv_cache_len;
    auto kDst = reinterpret_cast<T_dst*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, token_idx_in_kv_cache));
    auto vDst = reinterpret_cast<T_dst*>(kvCacheBuffer.getVBlockPtr(batch_beam_idx, token_idx_in_kv_cache));
    int inBlockIdx = kvCacheBuffer.getKVLocalIdx(token_idx_in_kv_cache, kv_head_idx, sizePerHeadDivX, channelIdx);
    if (!is_masked)
    {
        if constexpr (STORE_QKV)
        {
            *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
        }
        else
        {
            *reinterpret_cast<Vec_t*>(&Q[token_idx * head_num * size_per_head + hidden_idx]) = q;
        }
        if ((head_num == kv_head_num) || (head_idx == (kv_head_idx * qheads_per_kv_head)))
        {
            if constexpr (STORE_QKV)
            {
                *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
                *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
            }

            if (valid_kv_cache_pos)
            {
                if (ENABLE_8BITS_CACHE)
                {
                    inBlockIdx = inBlockIdx * vec_size;
                    // Cast float scale to dst data type.
                    using T_scale = typename mmha::kv_cache_scale_type_t<T, T_cache>::Type;
                    T_scale scaleOrigQuant;
                    mmha::convert_from_float(&scaleOrigQuant, kvScaleOrigQuant[0]);
                    // Store 8bits kv cache.
                    mmha::store_8bits_kv_cache_vec(kDst, k, inBlockIdx, scaleOrigQuant);
                    mmha::store_8bits_kv_cache_vec(vDst, v, inBlockIdx, scaleOrigQuant);
                }
                else
                {
                    reinterpret_cast<Vec_t*>(kDst)[inBlockIdx] = k;
                    reinterpret_cast<Vec_t*>(vDst)[inBlockIdx] = v;
                }
            }
        }
    }
}

#define APPLY_BIAS_ROPE_UPDATE_KV_CACHE(ADD_BIAS, STORE_QKV)                                                           \
    applyBiasRopeUpdateKVCache<T, T_cache, ADD_BIAS, STORE_QKV, KVCacheBuffer, IsGenerate>                             \
        <<<grid, block, smem_size, stream>>>(QKV, Q, kvTable, qkv_bias, seq_lens, kv_seq_lens, padding_offset,         \
            kvScaleOrigQuant, batch_size, seq_len, cyclic_kv_cache_len, head_num, kv_head_num, size_per_head,          \
            rotary_embedding_dim, rotary_embedding_base, rotary_scale_type, rotary_embedding_scale,                    \
            rotary_embedding_max_positions, position_embedding_type, beam_width);

template <typename T, typename T_cache, typename KVCacheBuffer, bool IsGenerate>
void invokeApplyBiasRopeUpdateKVCacheDispatch(T* QKV, T* Q, KVCacheBuffer& kvTable, const T* qkv_bias,
    const int* seq_lens, const int* kv_seq_lens, const int* padding_offset, const int batch_size, const int seq_len,
    const int cyclic_kv_cache_len, const int token_num, const int head_num, const int kv_head_num,
    const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,
    const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type, const float* scale,
    const float* kvScaleOrigQuant, const int int8_mode, const bool enable_paged_kv_fmha, cudaStream_t stream,
    int beam_width)
{
    TLLM_CHECK_WITH_INFO(int8_mode != 2, "w8a8 not yet implemented with RoPE"); // TODO
    if constexpr (!IsGenerate)
    {
        TLLM_CHECK_WITH_INFO(beam_width == 1, "beam_width should be default 1 for context phase.");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(seq_len == 1, "Generation phase should have seq_len of 1.");
        TLLM_CHECK_WITH_INFO(padding_offset == nullptr, "Generation phase should not use padding_offset");
        TLLM_CHECK_WITH_INFO(
            token_num == batch_size * beam_width, "token_num should be batch_size * beam_width for generation phase.");
    }
    // To implement rotary embeddings, each thread processes two QKV elems:
    dim3 block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3 grid(token_num, head_num);
    size_t smem_size
        = (position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX ? 2 * rotary_embedding_dim * sizeof(T) : 0);

    // Launch template parameters.
    const bool add_bias = qkv_bias != nullptr;
    const bool store_qkv = !enable_paged_kv_fmha;

    // NOTE: add offset for rotary embedding
    if (add_bias)
    {
        if (store_qkv)
        {
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(true, true);
        }
        else
        {
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(true, false);
        }
    }
    else
    {
        if (store_qkv)
        {
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(false, true);
        }
        else
        {
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(false, false);
        }
    }
}

template <typename T, typename KVCacheBuffer, bool IsGenerate>
void invokeApplyBiasRopeUpdateKVCache(T* QKV, T* Q, KVCacheBuffer& kvTable, const T* qkv_bias, const int* seq_lens,
    const int* kv_seq_lens, const int* padding_offset, const int batch_size, const int seq_len,
    const int cyclic_kv_cache_len, const int token_num, const int head_num, const int kv_head_num,
    const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,
    const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type, const float* scale,
    const int int8_mode, const KvCacheDataType cache_type, const float* kvScaleOrigQuant,
    const bool enable_paged_kv_fmha, cudaStream_t stream, int beam_width)
{
    // Block handles both K and V tile.
    constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    TLLM_CHECK_WITH_INFO(size_per_head % x == 0, "Size per head is not a multiple of X");

    if (cache_type == KvCacheDataType::INT8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer, IsGenerate>(QKV, Q, kvTable, qkv_bias,
            seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, token_num, head_num,
            kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, scale, kvScaleOrigQuant,
            int8_mode, enable_paged_kv_fmha, stream, beam_width);
    }
#ifdef ENABLE_FP8
    else if (cache_type == KvCacheDataType::FP8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, __nv_fp8_e4m3, KVCacheBuffer, IsGenerate>(QKV, Q, kvTable, qkv_bias,
            seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, token_num, head_num,
            kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, scale, kvScaleOrigQuant,
            int8_mode, enable_paged_kv_fmha, stream, beam_width);
    }
#endif // ENABLE_FP8
    else
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, T, KVCacheBuffer, IsGenerate>(QKV, Q, kvTable, qkv_bias, seq_lens,
            kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, token_num, head_num, kv_head_num,
            size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type, rotary_embedding_scale,
            rotary_embedding_max_positions, position_embedding_type, scale, kvScaleOrigQuant, int8_mode,
            enable_paged_kv_fmha, stream, beam_width);
    }
}

#define INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(T, KVCacheBuffer, IsGenerate)                                            \
    template void invokeApplyBiasRopeUpdateKVCache<T, KVCacheBuffer, IsGenerate>(T * QKV, T * Q,                       \
        KVCacheBuffer & kvTable, const T* qkv_bias, const int* seq_lens, const int* kv_seq_lens,                       \
        const int* padding_offset, const int batch_size, const int seq_len, const int cyclic_kv_cache_len,             \
        const int token_num, const int head_num, const int kv_head_num, const int size_per_head,                       \
        const int rotary_embedding_dim, const float rotary_embedding_base, const RotaryScalingType rotary_scale_type,  \
        const float rotary_embedding_scale, const int rotary_embedding_max_positions,                                  \
        const PositionEmbeddingType position_embedding_type, const float* scale, const int int8_mode,                  \
        const KvCacheDataType cache_type, const float* kvScaleOrigQuant, const bool enable_paged_kv_fmha,              \
        cudaStream_t stream, int beam_width)

INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(float, KVBlockArray, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(float, KVLinearBuffer, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVBlockArray, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVLinearBuffer, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(float, KVBlockArray, true);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(float, KVLinearBuffer, true);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVBlockArray, true);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVLinearBuffer, true);
#ifdef ENABLE_BF16
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16, KVBlockArray, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16, KVLinearBuffer, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16, KVBlockArray, true);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16, KVLinearBuffer, true);
#endif
#undef INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE

} // namespace kernels
} // namespace tensorrt_llm
