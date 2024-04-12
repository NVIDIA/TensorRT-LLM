/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#define WARP_SIZE 32
#define HALF_WARP_SIZE 16

////////////////////////////////////////////////////////////////////////////////////////////////////

// One warp of threads handle one head in terms of rotary embedding.
// Balance the work across threads in one warp for different head size.
// The minimum head size should be 32.
// Assume head size <= 256.
template <typename T, int Dh_MAX>
struct Rotary_vec_t
{
    using Type = T;
    // Quantized output type only supports fp8 currently.
    using Packed_type = __nv_fp8_e4m3;
    using Quantized_type = void;
    static constexpr int size = 1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Rotary_vec_t<float, 32>
{
    using Type = float;
    using Quantized_type = __nv_fp8_e4m3;
    using Packed_type = float;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<float, 64>
{
    using Type = float2;
    using Quantized_type = mmha::fp8_2_t;
    using Packed_type = float2;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<float, 128>
{
    using Type = float4;
    using Quantized_type = mmha::fp8_4_t;
    using Packed_type = float2;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<float, 256>
{
    using Type = mmha::Float8_;
    using Quantized_type = mmha::fp8_8_t;
    using Packed_type = float2;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Rotary_vec_t<half, 32>
{
    using Type = uint16_t;
    using Quantized_type = __nv_fp8_e4m3;
    using Packed_type = uint16_t;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<half, 64>
{
    using Type = uint32_t;
    using Quantized_type = mmha::fp8_2_t;
    using Packed_type = uint32_t;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<half, 128>
{
    using Type = uint2;
    using Quantized_type = mmha::fp8_4_t;
    using Packed_type = uint32_t;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<half, 256>
{
    using Type = uint4;
    using Quantized_type = mmha::fp8_8_t;
    using Packed_type = uint32_t;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16

template <>
struct Rotary_vec_t<__nv_bfloat16, 32>
{
    using Type = __nv_bfloat16;
    using Quantized_type = __nv_fp8_e4m3;
    using Packed_type = __nv_bfloat16;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 64>
{
    using Type = __nv_bfloat162;
    using Quantized_type = mmha::fp8_2_t;
    using Packed_type = __nv_bfloat162;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 128>
{
    using Type = mmha::bf16_4_t;
    using Quantized_type = mmha::fp8_4_t;
    using Packed_type = __nv_bfloat162;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 256>
{
    using Type = mmha::bf16_8_t;
    using Quantized_type = mmha::fp8_8_t;
    using Packed_type = __nv_bfloat162;
    static constexpr int size = 8;
};

#endif

template <typename T, typename T_cache, int Dh_MAX, bool ADD_BIAS, bool STORE_QKV, bool POS_SHIFT,
    typename KVCacheBuffer, bool IS_GENERATE>
__global__ void applyBiasRopeUpdateKVCache(T* QKV, void* O, T* Q, KVCacheBuffer kvCacheBuffer,
    T const* __restrict qkv_bias, int const* seq_lens, int const* kv_seq_lens, int const* padding_offset,
    float const* kvScaleOrigQuant, int const num_tokens, int const batch_size, int const seq_len,
    int const cyclic_kv_cache_len, int const sink_token_len, int const head_num, int const kv_head_num,
    int const qheads_per_kv_head, int const size_per_head, int const rotary_embedding_dim, float rotary_embedding_base,
    RotaryScalingType const rotary_scale_type, float rotary_embedding_scale, int const rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, int const* medusa_position_offsets,
    bool const quantized_fp8_output, int const beam_width)
{
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head]
    // Extract the Q input when using paged KV FMHA.
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

    // There are two kinds of output:
    //  1. Contiguous QKV output.
    //  2. Contiguous Q output + Paged KV output (needed by Paged KV FMHA kernels).

    // VEC_SIZE is power of 2.
    constexpr int VEC_SIZE = Rotary_vec_t<T, Dh_MAX>::size;
    using Vec_type = typename Rotary_vec_t<T, Dh_MAX>::Type;
    // Quantized output only supports fp8 currently.
    using Quantized_elt_type = __nv_fp8_e4m3;
    using Quantized_type = typename Rotary_vec_t<T, Dh_MAX>::Quantized_type;
    using Packed_type = typename Rotary_vec_t<T, Dh_MAX>::Packed_type;
    bool const has_padding = padding_offset == nullptr;

    constexpr bool ENABLE_8BITS_CACHE = sizeof(T_cache) == 1;
    int const sizePerHeadDivX = size_per_head / VEC_SIZE;
    using T_dst = T_cache;

    int const head_idx = blockIdx.y;
    // Block size is always 32 in the x dimension.
    int tidx = threadIdx.x;
    // The half head dimension for remapping.
    // 32 threads in one warp
    // (first rotary threads + first no rotary threads) = first 16 threads
    // (second rotary threads + second no rotary threads) = second 16 threads
    int const half_within_bound_dim = size_per_head / 2;
    int const half_rotary_embedding_dim = rotary_embedding_dim / 2;
    int const half_rotary_embedding_threads = rotary_embedding_dim / (2 * VEC_SIZE);
    int const half_non_rotary_embedding_threads = (size_per_head - rotary_embedding_dim) / (2 * VEC_SIZE);
    // Remap to the correct half head size when head size is not power of 2.
    // This is mianly designed for the gptneox_style_rotary_embedding (which rotates the half embedding.)
    // The first 16 threads will handle the first half head size.
    bool const first_half = tidx < HALF_WARP_SIZE;
    int const second_half = !first_half;

    int rotary_local_tidx = (tidx - second_half * HALF_WARP_SIZE);

    // Three partitions for each half threads.
    //  apply rotary (tidx * VEC_SIZE < half_rotary_embdding)
    //  don't apply rotary= (half_rotary_embedding <= tidx * VEC_SIZE < half_size_per_head)
    //  out of the bound (tidx * VEC_SIZE >= half_size_per_head)
    tidx = rotary_local_tidx * VEC_SIZE >= half_within_bound_dim
        ? -1
        : (rotary_local_tidx * VEC_SIZE < half_rotary_embedding_dim
                ? (rotary_local_tidx + second_half * half_rotary_embedding_threads)
                : (rotary_local_tidx + half_rotary_embedding_threads
                    + second_half * half_non_rotary_embedding_threads));

    int const hidden_size = head_num * size_per_head;
    int const hidden_idx = head_idx * size_per_head + tidx * VEC_SIZE;
    int const kv_head_idx = head_idx / qheads_per_kv_head;
    int const hidden_idx_kv = kv_head_idx * size_per_head + tidx * VEC_SIZE;
    int const n = (head_num + 2 * kv_head_num) * size_per_head;
    int const src_k_offset = hidden_size;
    int const src_v_offset = hidden_size + kv_head_num * size_per_head;

    // Dynamic scaling of rotary embedding.
    bool const dynamic_scale = rotary_scale_type == RotaryScalingType::kDYNAMIC;

    for (int token_idx = blockIdx.x * blockDim.y + threadIdx.y; token_idx < num_tokens;
         token_idx += gridDim.x * blockDim.y)
    {
        // The index of the token in the batch. It includes "virtual" padding (even if the input is not padded)
        // such that the sequence index and the position in the sequence can be obtained using the max.
        // sequence length as:
        int const token_padding_offset = (has_padding || IS_GENERATE) ? 0 : padding_offset[token_idx];
        int const global_token_idx = token_idx + token_padding_offset;
        int const batch_beam_idx = global_token_idx / seq_len;
        // TODO: optimize this for generation by using anther dimension of grid.
        int const seq_idx = global_token_idx % seq_len;
        int const final_kv_seq_len = (!IS_GENERATE) ? kv_seq_lens[batch_beam_idx] : 0;
        int const actual_seq_len = seq_lens[batch_beam_idx];
        // Chunked attention: takes past_kv_sequence_length into consideration.
        int const token_idx_in_seq
            = (!IS_GENERATE) ? (final_kv_seq_len - actual_seq_len) + seq_idx : (actual_seq_len - seq_len + seq_idx);
        bool const valid_seq = IS_GENERATE || (token_idx_in_seq < actual_seq_len || !has_padding);
        // NOTE: only Medusa needs the position offsets.
        // In the generation phase, we assume all sequences should have the same input length.
        int const rotary_position = medusa_position_offsets != nullptr && IS_GENERATE
            ? (medusa_position_offsets[seq_idx + batch_beam_idx * seq_len] + actual_seq_len - seq_len)
            : token_idx_in_seq;

        // only update the base and/or scale if needed based on scale_type
        // we have already updated the scale in host if it is linear scale.
        float2 updated_base_scale = mmha::update_dynamic_scaling_rotary(rotary_embedding_base, rotary_embedding_scale,
            actual_seq_len, rotary_embedding_max_positions, rotary_embedding_dim, dynamic_scale);
        float const updated_base = updated_base_scale.x;
        float const updated_scale = updated_base_scale.y;

        bool const is_masked = !valid_seq || tidx < 0;

        // head_num == kv_head_num:
        //   src QKV: [batch, time, 3, head_num, size_per_head]
        // head_num != kv_head_num:
        //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
        auto const src_q_idx = static_cast<size_t>(token_idx) * n + hidden_idx;
        auto const src_k_idx = static_cast<size_t>(token_idx) * n + src_k_offset + hidden_idx_kv;
        auto const src_v_idx = static_cast<size_t>(token_idx) * n + src_v_offset + hidden_idx_kv;

        Vec_type q, k, v;
        Vec_type q_bias, k_bias, v_bias;
        // key without position embedding
        Vec_type k_wo_pos;

        // load q,k,v and add bias
        if (!is_masked)
        {
            q = *reinterpret_cast<Vec_type const*>(&QKV[src_q_idx]);
            k = *reinterpret_cast<Vec_type const*>(&QKV[src_k_idx]);
            v = *reinterpret_cast<Vec_type const*>(&QKV[src_v_idx]);

            if constexpr (ADD_BIAS)
            {
                q_bias = *reinterpret_cast<Vec_type const*>(&qkv_bias[hidden_idx]);
                k_bias = *reinterpret_cast<Vec_type const*>(&qkv_bias[hidden_idx_kv + src_k_offset]);
                v_bias = *reinterpret_cast<Vec_type const*>(&qkv_bias[hidden_idx_kv + src_v_offset]);

                q = mmha::add(q, q_bias);
                k = mmha::add(k, k_bias);
                v = mmha::add(v, v_bias);
            }
            k_wo_pos = k;
        }

        // Rotary Emedding.
        switch (position_embedding_type)
        {
        // Rotate every two elements (need at two elements per thead).
        // e.g.  0  1  2  3  4  5  6  7 (head size 8)
        //      -1  0 -3  2 -5  4 -7  6
        case PositionEmbeddingType::kROPE_GPTJ:
        {
            mmha::apply_rotary_embedding(
                q, k, tidx, rotary_embedding_dim, updated_base, updated_scale, rotary_position);
            break;
        }
        // Rotate by half rotary embedding.
        // e.g.  0  1  2  3  4  5  6  7 (head size 8)
        //      -4 -5 -6 -7  0  1  2  3
        case PositionEmbeddingType::kROPE_GPT_NEOX:
        {
            // One warp of threads handle one head.
            // where the first 16 threads process the first half of rotary embedding,
            //  and second 16 threads process the second half.
            // Note that the half rotary embedding may not be power of 2.
            // e.g. 80 head size (next power of 2 is 128, so each thread will process 4 elements),
            //  which means only thread 0 ~ 10 (exclusive), and 16 ~ 26 (exclusive) have work to do.
            mmha::apply_rotary_embedding_gptneox<Vec_type, Packed_type, T>(
                q, k, tidx, rotary_embedding_dim, updated_base, updated_scale, rotary_position, first_half);
            break;
        }
        }

        int const channelIdx{tidx};
        int const tokenIdxLowerBound = max(actual_seq_len - cyclic_kv_cache_len + sink_token_len, sink_token_len);
        bool const valid_kv_cache_pos
            = kvCacheBuffer.data != nullptr // In KV-cache-less mode. No need to store KV values
            && (token_idx_in_seq >= tokenIdxLowerBound || token_idx_in_seq < sink_token_len);
        int const token_kv_idx = kvCacheBuffer.getKVTokenIdx(token_idx_in_seq);

        if (!is_masked)
        {
            auto kDst = reinterpret_cast<T_dst*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, token_kv_idx));
            auto vDst = reinterpret_cast<T_dst*>(kvCacheBuffer.getVBlockPtr(batch_beam_idx, token_kv_idx));
            int inBlockIdx = kvCacheBuffer.getKVLocalIdx(token_kv_idx, kv_head_idx, sizePerHeadDivX, channelIdx);
            Vec_type k_to_cache = (POS_SHIFT) ? k_wo_pos : k;

            if constexpr (STORE_QKV)
            {
                if (quantized_fp8_output)
                {
                    // use 1.0f scale currently for qkv input of FP8 FMHA.
                    mmha::convert_to_fp8(
                        reinterpret_cast<Quantized_type*>(reinterpret_cast<Quantized_elt_type*>(O) + src_q_idx), q);
                }
                else
                {
                    *reinterpret_cast<Vec_type*>(&QKV[src_q_idx]) = q;
                }
            }
            else
            {
                *reinterpret_cast<Vec_type*>(&Q[token_idx * head_num * size_per_head + hidden_idx]) = q;
            }
            if ((head_num == kv_head_num) || (head_idx == (kv_head_idx * qheads_per_kv_head)))
            {
                if constexpr (STORE_QKV)
                {
                    if (quantized_fp8_output)
                    {
                        // use 1.0f scale currently for qkv input of FP8 FMHA.
                        mmha::convert_to_fp8(
                            reinterpret_cast<Quantized_type*>(reinterpret_cast<Quantized_elt_type*>(O) + src_k_idx), k);
                        mmha::convert_to_fp8(
                            reinterpret_cast<Quantized_type*>(reinterpret_cast<Quantized_elt_type*>(O) + src_v_idx), v);
                    }
                    else
                    {
                        *reinterpret_cast<Vec_type*>(&QKV[src_k_idx]) = k;
                        if constexpr (ADD_BIAS)
                        {
                            *reinterpret_cast<Vec_type*>(&QKV[src_v_idx]) = v;
                        }
                    }
                }

                if (valid_kv_cache_pos)
                {
                    if constexpr (ENABLE_8BITS_CACHE)
                    {
                        inBlockIdx = inBlockIdx * VEC_SIZE;
                        // Cast float scale to dst data type.
                        using T_scale = typename mmha::kv_cache_scale_type_t<T, T_cache>::Type;
                        T_scale scaleOrigQuant;
                        mmha::convert_from_float(&scaleOrigQuant, kvScaleOrigQuant[0]);
                        // Store 8bits kv cache.
                        mmha::store_8bits_kv_cache_vec(kDst, k_to_cache, inBlockIdx, scaleOrigQuant);
                        mmha::store_8bits_kv_cache_vec(vDst, v, inBlockIdx, scaleOrigQuant);
                    }
                    else
                    {
                        reinterpret_cast<Vec_type*>(kDst)[inBlockIdx] = k_to_cache;
                        reinterpret_cast<Vec_type*>(vDst)[inBlockIdx] = v;
                    }
                }
            }
        }
    }
}

// Grid_block_cache (grid dim, block dim).
// This caches the block_size, grid_size calculated by cudaOccupancyMaxPotentialBlockSize.
#define APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, STORE_QKV, POS_SHIFT)                                        \
    if (grid_block_cache.x == 0 || grid_block_cache.y == 0)                                                            \
    {                                                                                                                  \
        TLLM_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&grid_block_cache.y, &grid_block_cache.x,                   \
            applyBiasRopeUpdateKVCache<T, T_cache, Dh_MAX, ADD_BIAS, STORE_QKV, POS_SHIFT, KVCacheBuffer,              \
                IS_GENERATE>));                                                                                        \
    }                                                                                                                  \
    int block_size = grid_block_cache.x, grid_size = grid_block_cache.y;                                               \
    int tokens_per_block = (block_size + WARP_SIZE - 1) / WARP_SIZE;                                                   \
    dim3 block(WARP_SIZE, tokens_per_block);                                                                           \
    int blocks_per_sequence                                                                                            \
        = std::min((grid_size + head_num - 1) / head_num, (token_num + tokens_per_block - 1) / tokens_per_block);      \
    dim3 grid(blocks_per_sequence, head_num);                                                                          \
    applyBiasRopeUpdateKVCache<T, T_cache, Dh_MAX, ADD_BIAS, STORE_QKV, POS_SHIFT, KVCacheBuffer, IS_GENERATE>         \
        <<<grid, block, 0, stream>>>(QKV, O, Q, kvTable, qkv_bias, seq_lens, kv_seq_lens, padding_offset,              \
            kvScaleOrigQuant, token_num, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, head_num,           \
            kv_head_num, head_num / kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base,           \
            rotary_scale_type, updated_rotary_embedding_scale, rotary_embedding_max_positions,                         \
            position_embedding_type, medusa_position_offsets, quantized_fp8_output, beam_width);

template <int Dh_MAX, typename T, typename T_cache, typename KVCacheBuffer, bool IS_GENERATE>
void kernelDispatchHeadSize(T* QKV, void* O, T* Q, KVCacheBuffer& kvTable, T const* qkv_bias, int const* seq_lens,
    int const* kv_seq_lens, int const* padding_offset, int const batch_size, int const seq_len,
    int const cyclic_kv_cache_len, int const sink_token_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float const rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float const rotary_embedding_scale,
    int const rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
    int const* medusa_position_offsets, bool const position_shift_enabled, float const* scale,
    float const* kvScaleOrigQuant, int const int8_mode, bool const enable_paged_kv_fmha,
    bool const quantized_fp8_output, int const beam_width, int2& grid_block_cache, cudaStream_t stream)
{
    bool const add_bias = qkv_bias != nullptr;
    bool const store_contiguous_qkv = !enable_paged_kv_fmha;

    // Update scale if scale_type == RotaryScalingType::kLINEAR.
    float const updated_rotary_embedding_scale
        = rotary_scale_type == RotaryScalingType::kLINEAR ? 1.0f / rotary_embedding_scale : rotary_embedding_scale;

    if (add_bias)
    {
        if (store_contiguous_qkv)
        {
            if (position_shift_enabled)
            {
                APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, true, true, true);
            }
            else
            {
                APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, true, true, false);
            }
        }
        else
        {
            if (position_shift_enabled)
            {
                APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, true, false, true);
            }
            else
            {
                APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, true, false, false);
            }
        }
    }
    else
    {
        if (store_contiguous_qkv)
        {
            if (position_shift_enabled)
            {
                APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, false, true, true);
            }
            else
            {
                APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, false, true, false);
            }
        }
        else
        {
            if (position_shift_enabled)
            {
                APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, false, false, true);
            }
            else
            {
                APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, false, false, false);
            }
        }
    }
}

template <typename T, typename T_cache, typename KVCacheBuffer, bool IS_GENERATE>
void invokeApplyBiasRopeUpdateKVCacheDispatch(T* QKV, void* O, T* Q, KVCacheBuffer& kvTable, T const* qkv_bias,
    int const* seq_lens, int const* kv_seq_lens, int const* padding_offset, int const batch_size, int const seq_len,
    int const cyclic_kv_cache_len, int const sink_token_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float const rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float const rotary_embedding_scale,
    int const rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
    int const* medusa_position_offsets, bool const position_shift_enabled, float const* scale,
    float const* kvScaleOrigQuant, int const int8_mode, bool const enable_paged_kv_fmha,
    bool const quantized_fp8_output, int const beam_width, int2& grid_block_cache, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(int8_mode != 2, "w8a8 not yet implemented with RoPE"); // TODO
    if constexpr (!IS_GENERATE)
    {
        TLLM_CHECK_WITH_INFO(beam_width == 1, "beam_width should be default 1 for context phase.");
    }
    else
    {
        // NOTE: generation phase may have input sequence length > 1 under the medusa mode.
        TLLM_CHECK_WITH_INFO(padding_offset == nullptr, "Generation phase should not use padding_offset");
    }

    // Use specialized kernels for different heads (better balance of work).
    TLLM_CHECK_WITH_INFO(size_per_head % 8 == 0, "Head size needs to be multiple of 8!");
    TLLM_CHECK_WITH_INFO(rotary_embedding_dim % 8 == 0, "Rotary embedding dimension needs to be multiple of 8!");
    // GPTJ Rotary embedding needs at least two elements per thread.
    if (size_per_head <= 64)
    {
        kernelDispatchHeadSize<64, T, T_cache, KVCacheBuffer, IS_GENERATE>(QKV, O, Q, kvTable, qkv_bias, seq_lens,
            kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num, head_num,
            kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
            position_shift_enabled, scale, kvScaleOrigQuant, int8_mode, enable_paged_kv_fmha, quantized_fp8_output,
            beam_width, grid_block_cache, stream);
    }
    else if (size_per_head <= 128)
    {
        kernelDispatchHeadSize<128, T, T_cache, KVCacheBuffer, IS_GENERATE>(QKV, O, Q, kvTable, qkv_bias, seq_lens,
            kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num, head_num,
            kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
            position_shift_enabled, scale, kvScaleOrigQuant, int8_mode, enable_paged_kv_fmha, quantized_fp8_output,
            beam_width, grid_block_cache, stream);
    }
    else if (size_per_head <= 256)
    {
        kernelDispatchHeadSize<256, T, T_cache, KVCacheBuffer, IS_GENERATE>(QKV, O, Q, kvTable, qkv_bias, seq_lens,
            kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num, head_num,
            kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
            position_shift_enabled, scale, kvScaleOrigQuant, int8_mode, enable_paged_kv_fmha, quantized_fp8_output,
            beam_width, grid_block_cache, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "applyBiasRopeUpdateKVCache kernel doesn't support head size = %d", size_per_head);
    }
}

template <typename T, typename KVCacheBuffer, bool IS_GENERATE>
void invokeApplyBiasRopeUpdateKVCache(T* QKV, void* O, T* Q, KVCacheBuffer& kvTable, T const* qkv_bias,
    int const* seq_lens, int const* kv_seq_lens, int const* padding_offset, int const batch_size, int const seq_len,
    int const cyclic_kv_cache_len, int const sink_token_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float const rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float const rotary_embedding_scale,
    int const rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
    int const* medusa_position_offsets, bool const position_shift_enabled, float const* scale, int const int8_mode,
    const KvCacheDataType cache_type, float const* kvScaleOrigQuant, bool const enable_paged_kv_fmha,
    bool const quantized_fp8_output, int const beam_width, int2& grid_block_cache, cudaStream_t stream)
{
    // Block handles both K and V tile.
    constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    TLLM_CHECK_WITH_INFO(size_per_head % x == 0, "Size per head is not a multiple of X");

    if (cache_type == KvCacheDataType::INT8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer, IS_GENERATE>(QKV, O, Q, kvTable, qkv_bias,
            seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
            head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
            position_shift_enabled, scale, kvScaleOrigQuant, int8_mode, enable_paged_kv_fmha, quantized_fp8_output,
            beam_width, grid_block_cache, stream);
    }
#ifdef ENABLE_FP8
    else if (cache_type == KvCacheDataType::FP8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, __nv_fp8_e4m3, KVCacheBuffer, IS_GENERATE>(QKV, O, Q, kvTable,
            qkv_bias, seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len,
            token_num, head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base,
            rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type,
            medusa_position_offsets, position_shift_enabled, scale, kvScaleOrigQuant, int8_mode, enable_paged_kv_fmha,
            quantized_fp8_output, beam_width, grid_block_cache, stream);
    }
#endif // ENABLE_FP8
    else
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, T, KVCacheBuffer, IS_GENERATE>(QKV, O, Q, kvTable, qkv_bias,
            seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
            head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
            rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
            position_shift_enabled, scale, kvScaleOrigQuant, int8_mode, enable_paged_kv_fmha, quantized_fp8_output,
            beam_width, grid_block_cache, stream);
    }
}

#define INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(T, KVCacheBuffer, IS_GENERATE)                                           \
    template void invokeApplyBiasRopeUpdateKVCache<T, KVCacheBuffer, IS_GENERATE>(T * QKV, void* O, T* Q,              \
        KVCacheBuffer& kvTable, const T* qkv_bias, const int* seq_lens, const int* kv_seq_lens,                        \
        const int* padding_offset, const int batch_size, const int seq_len, const int cyclic_kv_cache_len,             \
        const int sink_token_len, const int token_num, const int head_num, const int kv_head_num,                      \
        const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,                    \
        const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,                                 \
        const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,                 \
        const int* medusa_position_offsets, const bool position_shift_enabled, const float* scale,                     \
        const int int8_mode, const KvCacheDataType cache_type, const float* kvScaleOrigQuant,                          \
        const bool enable_paged_kv_fmha, bool const quantized_fp8_output, const int beam_width,                        \
        int2& grid_block_cache, cudaStream_t stream)

} // namespace kernels
} // namespace tensorrt_llm
