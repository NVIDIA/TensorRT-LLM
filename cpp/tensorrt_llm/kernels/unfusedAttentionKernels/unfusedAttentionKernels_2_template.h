/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

#define WARP_SIZE 32
#define HALF_WARP_SIZE 16
#define WARPS_PER_SM 32
#define MIN_SEQUENCES_PER_WARP 4
#define WARPS_PER_BLOCK 32

////////////////////////////////////////////////////////////////////////////////////////////////////

// One warp of threads handle one head in terms of rotary embedding.
// Balance the work across threads in one warp for different head size.
// The minimum head size should be 32.
// Assume head size <= 256.
template <typename T, int Dh_MAX>
struct Rotary_vec_t
{
    using Type = T;
    using BaseType = T;
    // Quantized output type only supports fp8 currently.
    using QuantizedType = __nv_fp8_e4m3;
    static constexpr int size = 1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Rotary_vec_t<float, 32>
{
    using Type = float;
    using BaseType = float;
    using QuantizedType = __nv_fp8_e4m3;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<float, 64>
{
    using Type = float2;
    using BaseType = float;
    using QuantizedType = mmha::fp8_2_t;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<float, 128>
{
    using Type = float4;
    using BaseType = float;
    using QuantizedType = mmha::fp8_4_t;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<float, 256>
{
    using Type = mmha::Float8_;
    using BaseType = float;
    using QuantizedType = mmha::fp8_8_t;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Rotary_vec_t<half, 32>
{
    using Type = uint16_t;
    using BaseType = uint16_t;
    using QuantizedType = __nv_fp8_e4m3;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<half, 64>
{
    using Type = uint32_t;
    using BaseType = uint16_t;
    using QuantizedType = mmha::fp8_2_t;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<half, 128>
{
    using Type = uint2;
    using BaseType = uint16_t;
    using QuantizedType = mmha::fp8_4_t;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<half, 256>
{
    using Type = uint4;
    using BaseType = uint16_t;
    using QuantizedType = mmha::fp8_8_t;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16

template <>
struct Rotary_vec_t<__nv_bfloat16, 32>
{
    using Type = __nv_bfloat16;
    using BaseType = __nv_bfloat16;
    using QuantizedType = __nv_fp8_e4m3;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 64>
{
    using Type = __nv_bfloat162;
    using BaseType = __nv_bfloat16;
    using QuantizedType = mmha::fp8_2_t;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 128>
{
    using Type = mmha::bf16_4_t;
    using BaseType = __nv_bfloat16;
    using QuantizedType = mmha::fp8_4_t;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 256>
{
    using Type = mmha::bf16_8_t;
    using BaseType = __nv_bfloat16;
    using QuantizedType = mmha::fp8_8_t;
    static constexpr int size = 8;
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Multiple calls of reinterpret_cast.
template <typename type_in, typename type_out>
inline __device__ type_out* reinterpret_ptr(void* ptr, size_t offset)
{
    return reinterpret_cast<type_out*>(reinterpret_cast<type_in*>(ptr) + offset);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, typename TVec, bool GEN_PHASE, typename KVCacheBuffer>
inline __device__ void zeroPagedVCacheTail(QKVPreprocessingParams<T, KVCacheBuffer> const& params, int batchIdx,
    int tokenIdxInKvCache, int cacheSeqLen, int kvHeadIdx, int vecsPerHead, int channelIdx, bool isUniqueTokenWriter)
{
    // Context owns the last physical page exclusively. Clear unused V rows because paged FMHA loads full pages and
    // masked zero probabilities can still propagate NaNs through BMM2.
    if constexpr (!GEN_PHASE && std::is_same_v<KVCacheBuffer, KVBlockArray> && std::is_same_v<T, TCache>)
    {
        if (params.generation_phase || !isUniqueTokenWriter || cacheSeqLen <= 0 || tokenIdxInKvCache != cacheSeqLen - 1)
        {
            return;
        }

        int const tokensPerBlock = params.kv_cache_buffer.mTokensPerBlock;
        int const validTokensInLastBlock = params.kv_cache_buffer.getLocalIdx(cacheSeqLen - 1) + 1;
        if (validTokensInLastBlock >= tokensPerBlock)
        {
            return;
        }

        auto* vBlock = reinterpret_cast<TCache*>(params.kv_cache_buffer.getVBlockPtr(batchIdx, cacheSeqLen - 1));
        auto* vBlockVec = reinterpret_cast<TVec*>(vBlock);
        TVec const zero{};
        for (int tokenInBlock = validTokensInLastBlock; tokenInBlock < tokensPerBlock; ++tokenInBlock)
        {
            int const vIdx = kvHeadIdx * tokensPerBlock * vecsPerHead + tokenInBlock * vecsPerHead + channelIdx;
            vBlockVec[vIdx] = zero;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Make sure each thread at least processes two elements (gptj rotary embedding).

template <typename T, RotaryPositionEmbeddingType ROTARY_TYPE>
struct Rotary_base_t
{
    using RotaryBaseType = T;
};

template <>
struct Rotary_base_t<uint16_t, RotaryPositionEmbeddingType::GPTJ>
{
    using RotaryBaseType = uint32_t;
};

#ifdef ENABLE_BF16
template <>
struct Rotary_base_t<__nv_bfloat16, RotaryPositionEmbeddingType::GPTJ>
{
    using RotaryBaseType = __nv_bfloat162;
};
#endif

template <>
struct Rotary_base_t<float, RotaryPositionEmbeddingType::GPTJ>
{
    using RotaryBaseType = float2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VecType, typename T, int VEC_SIZE, bool RECOMPUTE>
inline __device__ void apply_rotary_embedding_gptneox(VecType& q, VecType& q_pair, VecType& k, VecType& k_pair,
    bool first_half, float2 (&rotary_coef_cache)[VEC_SIZE], float const* rotary_inv_freq_buffer,
    int const rotary_dim_idx, int const half_rotary_dim, int const rotary_position, int const vision_start = -1,
    int const vision_length = -1)
{
    // Each thread holds NUM_ELTS elements.
    // Currently we apply the rotary embedding in float data type for accuracy reasons.
    using RotaryBaseType = typename Rotary_base_t<T, RotaryPositionEmbeddingType::GPT_NEOX>::RotaryBaseType;
#pragma unroll
    for (int elt_id = 0; elt_id < VEC_SIZE; elt_id++)
    {
        RotaryBaseType& q_ = reinterpret_cast<RotaryBaseType*>(&q)[elt_id];
        RotaryBaseType q_pair_ = reinterpret_cast<RotaryBaseType*>(&q_pair)[elt_id];
        RotaryBaseType& k_ = reinterpret_cast<RotaryBaseType*>(&k)[elt_id];
        RotaryBaseType k_pair_ = reinterpret_cast<RotaryBaseType*>(&k_pair)[elt_id];

        bool const valid_rotary_pos = rotary_dim_idx + elt_id < half_rotary_dim;

        if (RECOMPUTE)
        {
            int real_rotary_position = rotary_position;
            if (vision_start != -1 && vision_length != -1)
            {
                int t_step_int = rotary_position;
                if (t_step_int <= vision_start)
                {
                    real_rotary_position = t_step_int;
                }
                else if (t_step_int > vision_start && t_step_int <= (vision_length + vision_start))
                {
                    real_rotary_position = vision_start + 1;
                }
                else
                {
                    real_rotary_position = t_step_int - (vision_length - 1);
                }
            }
            float const rotary_inv_freq = float(real_rotary_position)
                * rotary_inv_freq_buffer[min(rotary_dim_idx + elt_id, half_rotary_dim - 1)];
            rotary_coef_cache[elt_id] = make_float2(cosf(rotary_inv_freq), sinf(rotary_inv_freq));
        }

        // Mask non-rotary dim.
        float2 rotary_coef = valid_rotary_pos ? rotary_coef_cache[elt_id] : make_float2(1.0f, 0.0f);
        // Pre-process different half of rotary dimension.
        rotary_coef.y = first_half ? -rotary_coef.y : rotary_coef.y;

        mmha::apply_rotary_embedding_gptneox(q_, q_pair_, k_, k_pair_, rotary_coef);
    }
}

template <typename VecType, typename T, int VEC_SIZE, bool RECOMPUTE>
inline __device__ void apply_rotary_embedding_gptj(VecType& q, VecType& k, float2 (&rotary_coef_cache)[VEC_SIZE],
    float const* rotary_inv_freq_buffer, int const rotary_dim_idx, int const half_rotary_dim, int const rotary_position)
{
    // Each thread holds NUM_ELTS elements.
    // Currently we apply the rotary embedding in float data type for accuracy reasons.
    using RotaryBaseType = typename Rotary_base_t<T, RotaryPositionEmbeddingType::GPTJ>::RotaryBaseType;
#pragma unroll
    for (int elt_id = 0; elt_id < VEC_SIZE; elt_id++)
    {
        RotaryBaseType q_ = reinterpret_cast<RotaryBaseType*>(&q)[elt_id];
        RotaryBaseType k_ = reinterpret_cast<RotaryBaseType*>(&k)[elt_id];

        bool const valid_rotary_pos = rotary_dim_idx + elt_id < half_rotary_dim;

        if (RECOMPUTE)
        {
            float const rotary_inv_freq
                = float(rotary_position) * rotary_inv_freq_buffer[min(rotary_dim_idx + elt_id, half_rotary_dim - 1)];
            rotary_coef_cache[elt_id] = make_float2(cosf(rotary_inv_freq), sinf(rotary_inv_freq));
        }

        mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache[elt_id]);

        if (valid_rotary_pos)
        {
            reinterpret_cast<RotaryBaseType*>(&q)[elt_id] = q_;
            reinterpret_cast<RotaryBaseType*>(&k)[elt_id] = k_;
        }
    }
}

template <typename T, int VECS_PER_HEAD>
inline __device__ void quantizeAndWriteFP4KVCache(uint8_t* kBlockScales, uint8_t* vBlockScales, uint32_t* kDst,
    uint32_t* vDst, float kSecondLevelSF, float vSecondLevelSF, int inBlockIdx, PackedVec<T>& kPacked,
    PackedVec<T>& vPacked)
{
    uint8_t* kSfOut = nullptr;
    uint8_t* vSfOut = nullptr;
    // WARNING: 8 elements per thread is assumed.
    // Two threads are involved in the reduction for block scales inside
    // cvt_warp_fp16_to_fp4, but only one thread needs to write out the
    // final answer.
    constexpr int NUM_SFS_PER_HEAD = VECS_PER_HEAD / 2;
    if (inBlockIdx % 2 == 0)
    {
        auto blockScaleIdxDst = inBlockIdx / 2;
        kSfOut = kBlockScales + blockScaleIdxDst;
        // A interleaved layout (num_tokens / 4, num_sfs_per_head, 4) is used for nvfp4 kv cache in order to achieve
        // better performance. This is only used by trtllm-gen kernels.
        auto tokenIdxV = blockScaleIdxDst / NUM_SFS_PER_HEAD;
        auto headDimIdxV = blockScaleIdxDst % NUM_SFS_PER_HEAD;
        auto blockScaleIdxDstV = (tokenIdxV / 4) * 4 * NUM_SFS_PER_HEAD + headDimIdxV * 4 + (tokenIdxV % 4);
        vSfOut = vBlockScales + blockScaleIdxDstV;
    }

    // Despite the name of cvt_warp_fp16_to_fp4, it is used by
    // the quantize op for BF16 as well.
    constexpr int SF_VEC_SIZE = 16;
    kDst[inBlockIdx] = cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, false>(kPacked, kSecondLevelSF, kSfOut);
    vDst[inBlockIdx] = cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, false>(vPacked, vSecondLevelSF, vSfOut);
}

template <typename T, typename TCache, int Dh_MAX, bool ADD_BIAS, bool STORE_QKV, typename KVCacheBuffer,
    RotaryPositionEmbeddingType ROTARY_TYPE, bool DYNAMIC_ROTARY_SCALING, bool FP8_OUTPUT, bool GEN_PHASE>
__global__ void applyBiasRopeUpdateKVCache(QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head]
    // Extract the Q input when using paged KV FMHA.
    // For q and k, also apply the rotary embedding.

    // NOTE:
    // In the case of in-place modifications:
    //      head_num == kv_head_num
    //      QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //                      ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                              m                        n
    //      head_num != kv_head_num
    //      QKV src shape: (batch_size, seq_len, head_num * size_per_head + 2 * kv_head_num * size_per_head)
    //                      ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                                 m                               n
    // Additionally, if there is no padding:
    //      QKV src shape (num_tokens, 3, head_num, size_per_head)
    //
    //  In these above cases, output shape stays the same
    //
    //   Q dst shape: (batch_size, head_num, seq_len, size_per_head)
    //   KV dst shape: (batch_size, kv_head_num, seq_len, size_per_head)
    // See QKVPreprocessingParams for further details

    // There are two kinds of output:
    //  1. Contiguous QKV output.
    //  2. Contiguous Q output + Paged KV output (needed by Paged KV FMHA kernels).

    // VEC_SIZE is power of 2.
    constexpr int VEC_SIZE = Rotary_vec_t<T, Dh_MAX>::size;
    using VecType = typename Rotary_vec_t<T, Dh_MAX>::Type;
    // The base type will share the rotary coefficient.
    using BaseType = typename Rotary_vec_t<T, Dh_MAX>::BaseType;
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Rotary_vec_t<T, Dh_MAX>::QuantizedType;
    // GPTJ rotary embedding: two elements share the same rotary coefficient.
    constexpr int ROTARY_COEF_VEC_SIZE = ROTARY_TYPE == RotaryPositionEmbeddingType::GPTJ ? VEC_SIZE / 2 : VEC_SIZE;

#ifdef ENABLE_FP4
    constexpr bool ENABLE_4BITS_CACHE = std::is_same_v<TCache, __nv_fp4_e2m1> &&
        // TODO: enable for FP32. Requires adding new
        // quantization functions in kernels/quantization.cuh.
        sizeof(T) == 2;
#else
    constexpr bool ENABLE_4BITS_CACHE = false;
#endif
    constexpr bool ENABLE_8BITS_CACHE = sizeof(TCache) == 1 && !ENABLE_4BITS_CACHE;
    int const sizePerHeadDivX = params.size_per_head / VEC_SIZE;
    // This is only used by nvfp4 kv cache where Dh_MAX is same as head size (others are not supported yet).
    constexpr int VECS_PER_HEAD = Dh_MAX / VEC_SIZE;
    using TDst = TCache;

    // Variable sequence length.
    bool const variable_sequence_length = !GEN_PHASE && (params.cu_seq_lens != nullptr) && (params.seq_lens != nullptr);

    int const head_idx = blockIdx.y;
    // Block size is always 32 in the x dimension (handles one head size).
    int const tidx = threadIdx.x;
    int const head_dim_idx = tidx * VEC_SIZE;
    bool const first_half = head_dim_idx < params.half_rotary_dim;
    int const rotated_head_dim_offset = first_half ? params.half_rotary_dim : -params.half_rotary_dim;
    int const gptneox_rotary_dim_idx = first_half ? head_dim_idx : (head_dim_idx - params.half_rotary_dim);
    int const gptj_rotary_dim_idx = head_dim_idx / 2;

    int const hidden_idx = head_idx * params.size_per_head + head_dim_idx;
    int const kv_head_idx = head_idx / params.qheads_per_kv_head;
    int const hidden_idx_kv = kv_head_idx * params.size_per_head + head_dim_idx;
    int const input_hidden_size = params.q_only_input ? params.q_hidden_size : params.hidden_size;
    int const src_k_offset = params.q_hidden_size;
    int const src_v_offset = src_k_offset + params.kv_head_num * params.size_per_head;

    // Reuse the rotary coefficients for the same rotary position.
    float2 rotary_coef_cache[ROTARY_COEF_VEC_SIZE];

    int local_token_idx = blockIdx.x * blockDim.y + threadIdx.y;
    {
        int cached_rotary_position = -1;
        for (int batch_idx = blockIdx.z; batch_idx < params.batch_size; batch_idx += gridDim.z)
        {
            // The index of the token in the batch.
            int const global_token_idx = local_token_idx
                + ((variable_sequence_length && params.remove_padding) ? params.cu_seq_lens[batch_idx]
                                                                       : batch_idx * params.max_input_seq_len);
            int const cache_seq_len = params.cache_seq_lens[batch_idx];
            int const actual_seq_len = variable_sequence_length ? params.seq_lens[batch_idx] : params.max_input_seq_len;
            // Chunked attention: takes past_kv_sequence_length into consideration.
            int const past_seq_len = (cache_seq_len - actual_seq_len);
            int const token_idx_in_seq = past_seq_len + local_token_idx;
            bool const valid_token = token_idx_in_seq < cache_seq_len;

            if (!valid_token)
            {
                continue;
            }

            // NOTE: only spec decoding needs the position offsets.
            // In the generation phase, we assume all sequences should have the same input length.
            // Helix parallelism: use helix_position_offsets if available (absolute position).
            int const rotary_position
                = (params.helix_position_offsets != nullptr ? params.helix_position_offsets[global_token_idx]
                          : params.spec_decoding_position_offsets != nullptr
                          ? (params.spec_decoding_position_offsets[local_token_idx
                                 + batch_idx * params.max_input_seq_len]
                              + past_seq_len)
                          : token_idx_in_seq)
                + (params.mrope_position_deltas != nullptr ? params.mrope_position_deltas[batch_idx] : 0);

            // Helix parallelism: determine if this rank is inactive for this request.
            bool const helix_inactive
                = params.helix_is_inactive_rank != nullptr && params.helix_is_inactive_rank[batch_idx];

            // Is the token and head dim maksed.
            bool const valid_head_dim_idx = head_dim_idx < params.size_per_head;

            // head_num == kv_head_num:
            //   src QKV: [batch, time, 3, head_num, size_per_head]
            // head_num != kv_head_num:
            //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
            auto const src_q_idx = static_cast<size_t>(global_token_idx) * input_hidden_size + hidden_idx;
            auto const src_k_idx
                = static_cast<size_t>(global_token_idx) * input_hidden_size + src_k_offset + hidden_idx_kv;
            auto const src_v_idx
                = static_cast<size_t>(global_token_idx) * input_hidden_size + src_v_offset + hidden_idx_kv;

            VecType q, q_pair;
            VecType k, v, k_pair;
            // key without position embedding
            VecType k_wo_pos;

            // load q,k,v and add bias
            if (valid_head_dim_idx)
            {
                q = *reinterpret_cast<VecType const*>(&params.qkv_input[src_q_idx]);
                q_pair = *reinterpret_cast<VecType const*>(&params.qkv_input[src_q_idx + rotated_head_dim_offset]);
                if (!params.q_only_input)
                {
                    k = *reinterpret_cast<VecType const*>(&params.qkv_input[src_k_idx]);
                    v = *reinterpret_cast<VecType const*>(&params.qkv_input[src_v_idx]);
                    k_pair = *reinterpret_cast<VecType const*>(&params.qkv_input[src_k_idx + rotated_head_dim_offset]);
                }
                else
                {
                    // The rotary helpers update Q and K together. Reuse Q as the ignored K input.
                    k = q;
                    k_pair = q_pair;
                }

                if constexpr (ADD_BIAS)
                {
                    auto const q_bias = *reinterpret_cast<VecType const*>(&params.qkv_bias[hidden_idx]);
                    auto const q_pair_bias
                        = *reinterpret_cast<VecType const*>(&params.qkv_bias[hidden_idx + rotated_head_dim_offset]);

                    q = mmha::add(q, q_bias);
                    q_pair = mmha::add(q_pair, q_pair_bias);
                    if (!params.q_only_input)
                    {
                        auto const k_bias
                            = *reinterpret_cast<VecType const*>(&params.qkv_bias[hidden_idx_kv + src_k_offset]);
                        auto const v_bias
                            = *reinterpret_cast<VecType const*>(&params.qkv_bias[hidden_idx_kv + src_v_offset]);
                        auto const k_pair_bias = *reinterpret_cast<VecType const*>(
                            &params.qkv_bias[hidden_idx_kv + src_k_offset + rotated_head_dim_offset]);
                        k = mmha::add(k, k_bias);
                        v = mmha::add(v, v_bias);
                        k_pair = mmha::add(k_pair, k_pair_bias);
                    }
                }
                k_wo_pos = k;
            }

            // The offset of rotary_embedding_inv_freq.
            // Dynamic rotary scaling might have different inv_freq values for different sequences.
            size_t const inv_freq_buffer_offset = DYNAMIC_ROTARY_SCALING ? batch_idx * params.half_rotary_dim : 0;
            switch (ROTARY_TYPE)
            {
            // Rotate every two elements (need at two elements per thead).
            // e.g.  0  1  2  3  4  5  6  7 (head size 8)
            //      -1  0 -3  2 -5  4 -7  6
            case RotaryPositionEmbeddingType::GPTJ:
            {
                if (DYNAMIC_ROTARY_SCALING || rotary_position != cached_rotary_position)
                {
                    apply_rotary_embedding_gptj<VecType, BaseType, ROTARY_COEF_VEC_SIZE, true>(q, k, rotary_coef_cache,
                        params.rotary_embedding_inv_freq + inv_freq_buffer_offset, gptj_rotary_dim_idx,
                        params.half_rotary_dim, rotary_position);
                    cached_rotary_position = rotary_position;
                }
                else
                {
                    apply_rotary_embedding_gptj<VecType, BaseType, ROTARY_COEF_VEC_SIZE, false>(q, k, rotary_coef_cache,
                        params.rotary_embedding_inv_freq + inv_freq_buffer_offset, gptj_rotary_dim_idx,
                        params.half_rotary_dim, rotary_position);
                }
                break;
            }
            // Rotate by half rotary embedding.
            // e.g.  0  1  2  3  4  5  6  7 (head size 8)
            //      -4 -5 -6 -7  0  1  2  3
            case RotaryPositionEmbeddingType::GPT_NEOX:
            {
                if (DYNAMIC_ROTARY_SCALING || rotary_position != cached_rotary_position)
                {
                    apply_rotary_embedding_gptneox<VecType, BaseType, ROTARY_COEF_VEC_SIZE, true>(q, q_pair, k, k_pair,
                        first_half, rotary_coef_cache, params.rotary_embedding_inv_freq + inv_freq_buffer_offset,
                        gptneox_rotary_dim_idx, params.half_rotary_dim, rotary_position, params.rotary_vision_start,
                        params.rotary_vision_length);
                    cached_rotary_position = rotary_position;
                }
                else
                {
                    apply_rotary_embedding_gptneox<VecType, BaseType, ROTARY_COEF_VEC_SIZE, false>(q, q_pair, k, k_pair,
                        first_half, rotary_coef_cache, params.rotary_embedding_inv_freq + inv_freq_buffer_offset,
                        gptneox_rotary_dim_idx, params.half_rotary_dim, rotary_position, params.rotary_vision_start,
                        params.rotary_vision_length);
                }
                break;
            }
            }

            if (params.logn_scaling != nullptr)
            {
                float logn_scale = params.logn_scaling[token_idx_in_seq];
                q = mmha::mul<VecType, float, VecType>(logn_scale, q);
            }
            auto const channelIdx{tidx};

            bool const useKVCache = !params.q_only_input && params.kv_cache_buffer.data != nullptr;
            auto token_idx_in_kv_cache = token_idx_in_seq;
            bool valid_kv_cache_pos = useKVCache;

            // Make sure pairs of q or v vecs have been read before write.
            // One block will handle single head.
            __syncthreads();

            if (valid_head_dim_idx)
            {
                auto kDst
                    = reinterpret_cast<TDst*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto vDst
                    = reinterpret_cast<TDst*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, token_idx_in_kv_cache));
                int inBlockIdx = params.kv_cache_buffer.getKVLocalIdx(
                    token_idx_in_kv_cache, kv_head_idx, sizePerHeadDivX, channelIdx);
                VecType k_to_cache = params.position_shift_enabled ? k_wo_pos : k;

                auto const dst_q_idx = static_cast<size_t>(global_token_idx) * params.q_hidden_size + hidden_idx;
                VecType* q_ptr = STORE_QKV ? reinterpret_ptr<T, VecType>(params.qkv_input, src_q_idx)
                                           : reinterpret_ptr<T, VecType>(params.q_output, dst_q_idx);

                // Cast float scale to dst data type.
                using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
                [[maybe_unused]] TScale scaleOrigQuant;
                if constexpr (FP8_OUTPUT || ENABLE_8BITS_CACHE)
                {
                    mmha::convert_from_float(
                        &scaleOrigQuant, params.qkv_scale_orig_quant ? params.qkv_scale_orig_quant[0] : 1.0f);
                }

                if constexpr (FP8_OUTPUT)
                {
                    // Quant the vec to fp8 vec with the scale.
                    QuantizedEltType* quantized_q_ptr = STORE_QKV
                        ? reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output) + src_q_idx
                        : reinterpret_cast<QuantizedEltType*>(params.q_output) + dst_q_idx;
                    mmha::store_8bits_vec(quantized_q_ptr, q, 0, scaleOrigQuant);
                }
                else
                {
                    *q_ptr = q;
                }
                if (!params.q_only_input
                    && ((params.head_num == params.kv_head_num)
                        || (head_idx == (kv_head_idx * params.qheads_per_kv_head))))
                {
                    if constexpr (STORE_QKV)
                    {
                        if constexpr (FP8_OUTPUT)
                        {
                            // Quant the vec to fp8 vec with the scale.
                            mmha::store_8bits_vec(reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output), k,
                                src_k_idx, scaleOrigQuant);
                            mmha::store_8bits_vec(reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output), v,
                                src_v_idx, scaleOrigQuant);
                        }
                        else
                        {
                            *reinterpret_cast<VecType*>(&params.qkv_input[src_k_idx]) = k;
                            if constexpr (ADD_BIAS)
                            {
                                *reinterpret_cast<VecType*>(&params.qkv_input[src_v_idx]) = v;
                            }
                        }
                    }

                    if (valid_kv_cache_pos && !helix_inactive)
                    {
                        if constexpr (ENABLE_8BITS_CACHE)
                        {
                            inBlockIdx = inBlockIdx * VEC_SIZE;
                            // Store 8bits kv cache.
                            mmha::store_8bits_vec(kDst, k_to_cache, inBlockIdx, scaleOrigQuant);
                            mmha::store_8bits_vec(vDst, v, inBlockIdx, scaleOrigQuant);
                        }
                        else if constexpr (ENABLE_4BITS_CACHE)
                        {
                            auto* kBlockScales = reinterpret_cast<uint8_t*>(
                                params.kv_cache_block_scales_buffer.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                            auto* vBlockScales = reinterpret_cast<uint8_t*>(
                                params.kv_cache_block_scales_buffer.getVBlockPtr(batch_idx, token_idx_in_kv_cache));
                            float kSecondLevelSF = params.qkv_scale_orig_quant[1];
                            float vSecondLevelSF = params.qkv_scale_orig_quant[2];
                            auto& kPacked = reinterpret_cast<PackedVec<T>&>(k_to_cache);
                            auto& vPacked = reinterpret_cast<PackedVec<T>&>(v);
                            quantizeAndWriteFP4KVCache<T, VECS_PER_HEAD>(kBlockScales, vBlockScales,
                                reinterpret_cast<uint32_t*>(kDst), reinterpret_cast<uint32_t*>(vDst), kSecondLevelSF,
                                vSecondLevelSF, inBlockIdx, kPacked, vPacked);
                        }
                        else
                        {
                            reinterpret_cast<VecType*>(kDst)[inBlockIdx] = k_to_cache;
                            reinterpret_cast<VecType*>(vDst)[inBlockIdx] = v;
                        }
                        zeroPagedVCacheTail<T, TCache, VecType, GEN_PHASE>(params, batch_idx, token_idx_in_kv_cache,
                            cache_seq_len, kv_head_idx, sizePerHeadDivX, channelIdx, /*isUniqueTokenWriter=*/true);
                    }
                }
            }
        }
    }

    // Prepare values for fmha.
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
        // Reset fmha tile counter to 0 before launching fmha kernels.
        if (params.fmha_tile_counter)
        {
            params.fmha_tile_counter[0] = 0u;
        }
        // Take the quantization scales into consideration.
        float q_scale_quant_orig, k_scale_quant_orig, v_scale_quant_orig;
        if constexpr (ENABLE_4BITS_CACHE)
        {
            q_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
            k_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[1] : 1.f;
            v_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[2] : 1.f;
        }
        else
        {
            q_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
            k_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
            v_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
        }
        float o_scale_orig_quant = params.o_scale_orig_quant ? params.o_scale_orig_quant[0] : 1.f;
        if (params.fmha_bmm1_scale)
        {
            // The scale after fmha bmm1.
            params.fmha_bmm1_scale[0] = q_scale_quant_orig * k_scale_quant_orig * params.fmha_host_bmm1_scale;
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            params.fmha_bmm1_scale[1] = params.fmha_bmm1_scale[0] * kLog2e;
        }
        if (params.fmha_bmm2_scale)
        {
            // The scale after fmha bmm2.
            params.fmha_bmm2_scale[0] = o_scale_orig_quant * v_scale_quant_orig;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Bandwidth-bound kernel by reading cos/sin coefficients from global memory (pre-computed and saved as weights).

template <typename T>
struct VecType
{
    using Type = T;
    using QuantizedType = mmha::fp8_8_t;
    using GPTNeoXEltType = T;
    using GPTJEltType = T;
};

template <>
struct VecType<float>
{
    using Type = float4;
    using QuantizedType = mmha::fp8_4_t;
    using GPTNeoXEltType = float;
    using GPTJEltType = float2;
};

template <>
struct VecType<half>
{
    using Type = uint4;
    using QuantizedType = mmha::fp8_8_t;
    using GPTNeoXEltType = uint16_t;
    using GPTJEltType = uint32_t;
};

template <>
struct VecType<__nv_bfloat16>
{
    using Type = mmha::bf16_8_t;
    using QuantizedType = mmha::fp8_8_t;
    using GPTNeoXEltType = __nv_bfloat16;
    using GPTJEltType = __nv_bfloat162;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, int BLOCK_SIZE, int Dh, bool ADD_BIAS, bool STORE_QKV, bool FP8_OUTPUT,
    bool GEN_PHASE, typename KVCacheBuffer, RotaryPositionEmbeddingType ROTARY_TYPE>
__global__ void applyBiasRopeUpdateKVCacheV2(QKVPreprocessingParams<T, KVCacheBuffer> params)
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
    // Additionally, if there is no padding:
    //   QKV src shape (num_tokens, 3, head_num, size_per_head)
    //
    //   Q dst shape: (batch_size, head_num, seq_len, size_per_head)
    //   KV dst shape: (batch_size, kv_head_num, seq_len, size_per_head)
    // See QKVPreprocessingParams for further details

    // There are two kinds of output:
    //  1. Contiguous QKV output.
    //  2. Contiguous Q output + Paged KV output (needed by Paged KV FMHA kernels).

    // Constants.
    using VecT = typename VecType<T>::Type;
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename VecType<T>::QuantizedType;
    using GPTNeoXEltT = typename VecType<T>::GPTNeoXEltType;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = Dh;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
#ifdef ENABLE_FP4
    constexpr bool ENABLE_4BITS_CACHE = std::is_same_v<TCache, __nv_fp4_e2m1> &&
        // TODO: enable for FP32. Requires adding new
        // quantization functions in kernels/quantization.cuh.
        sizeof(T) == 2;
#else
    constexpr bool ENABLE_4BITS_CACHE = false;
#endif
    // int8 / fp8 kv cache.
    constexpr bool ENABLE_8BITS_CACHE = sizeof(TCache) == 1 && !ENABLE_4BITS_CACHE;

    // Head idx.
    int const head_idx = blockIdx.y;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // Variable sequence length.
    bool const variable_sequence_length = params.tokens_info != nullptr && params.seq_lens != nullptr;
    int const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
    int const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
    bool const first_half = head_dim_idx < params.half_rotary_dim;
    [[maybe_unused]] int const gptneox_rotary_dim_idx
        = first_half ? head_dim_idx : (head_dim_idx - params.half_rotary_dim);
    [[maybe_unused]] int const gptj_rotary_dim_idx = head_dim_idx / 2;
    // Assume that either all vector elements are valid rotary idx or not.
    [[maybe_unused]] int const valid_rotary_dim_idx = head_dim_idx < params.rotary_embedding_dim;
    float2 const masked_rotary_cos_sin = make_float2(1.0f, 0.0f);
    int const hidden_idx = head_idx * params.size_per_head + head_dim_idx;
    int const kv_head_idx = head_idx / params.qheads_per_kv_head;
    int const hidden_idx_kv = kv_head_idx * params.size_per_head + head_dim_idx;
    int const src_k_offset = params.q_hidden_size;
    int const src_v_offset = src_k_offset + params.kv_hidden_size;
    int const input_hidden_size = params.q_only_input ? params.q_hidden_size : params.hidden_size;

    int const rotated_head_dim_offset = first_half ? params.half_rotary_dim : -params.half_rotary_dim;
    // Make sure there are multiple of tokens_per_block otherwise syncthreads will lead to deadlocks.
    int const tokens_loop_end = int((params.token_num + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;

    // Mainloop.
    for (int global_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
         global_token_idx < tokens_loop_end; global_token_idx += TOKENS_PER_BLOCK * gridDim.x)
    {
        // Global token idx bounded by num of tokens.
        int bounded_global_token_idx = std::min(global_token_idx, params.token_num - 1);
        // The batch_idx and token idx in the sequence.
        int batch_idx, token_idx_in_seq;
        if constexpr (GEN_PHASE)
        {
            batch_idx = bounded_global_token_idx;
            token_idx_in_seq = 0;
        }
        else if (variable_sequence_length)
        {
            auto token_info = params.tokens_info[bounded_global_token_idx];
            batch_idx = token_info.x;
            token_idx_in_seq = token_info.y;
        }
        else
        {
            batch_idx = bounded_global_token_idx / params.max_input_seq_len;
            token_idx_in_seq = bounded_global_token_idx % params.max_input_seq_len;
        }
        // The cache sequence length that includes the input sequence length.
        int const cache_seq_len = params.cache_seq_lens[batch_idx];
        int const actual_seq_len = variable_sequence_length ? params.seq_lens[batch_idx] : params.max_input_seq_len;
        // Chunked attention: takes past_kv_sequence_length into consideration.
        int const past_seq_len = (cache_seq_len - actual_seq_len);
        // Is it a valid token to be stored ?
        bool valid_token = GEN_PHASE || (token_idx_in_seq < actual_seq_len);
        // Make sure token_idx_in_seq is within the bound of actual_seq_len.
        token_idx_in_seq = std::min(actual_seq_len - 1, token_idx_in_seq);
        int token_idx_in_kv_cache = past_seq_len + token_idx_in_seq;
        // The same as token_idx_in_seq < actual_seq_len.
        valid_token = valid_token && (token_idx_in_kv_cache < cache_seq_len);
        // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
        token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);

        // NOTE: only spec decoding needs the position offsets.
        // In the generation phase, we assume all sequences should have the same input length.
        // Helix parallelism: use helix_position_offsets if available (absolute position).
        int const rotary_position = params.helix_position_offsets != nullptr
            ? params.helix_position_offsets[bounded_global_token_idx]
            : params.spec_decoding_position_offsets != nullptr
            ? (params.spec_decoding_position_offsets[token_idx_in_seq + batch_idx * params.max_input_seq_len]
                + cache_seq_len - actual_seq_len)
            : token_idx_in_kv_cache;

        // Helix parallelism: determine if this rank is inactive for this request.
        bool const helix_inactive
            = params.helix_is_inactive_rank != nullptr && params.helix_is_inactive_rank[batch_idx];

        // head_num == kv_head_num:
        //   src QKV: [batch, time, 3, head_num, size_per_head]
        // head_num != kv_head_num:
        //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
        auto const src_q_idx = static_cast<size_t>(bounded_global_token_idx) * input_hidden_size + hidden_idx;
        auto const src_k_idx
            = static_cast<size_t>(bounded_global_token_idx) * input_hidden_size + src_k_offset + hidden_idx_kv;
        auto const src_v_idx
            = static_cast<size_t>(bounded_global_token_idx) * input_hidden_size + src_v_offset + hidden_idx_kv;

        auto q = *reinterpret_cast<VecT const*>(&params.qkv_input[src_q_idx]);
        VecT k;
        VecT v;
        [[maybe_unused]] auto q_pair
            = *reinterpret_cast<VecT const*>(&params.qkv_input[src_q_idx + rotated_head_dim_offset]);
        [[maybe_unused]] VecT k_pair;
        if (!params.q_only_input)
        {
            k = *reinterpret_cast<VecT const*>(&params.qkv_input[src_k_idx]);
            v = *reinterpret_cast<VecT const*>(&params.qkv_input[src_v_idx]);
            k_pair = *reinterpret_cast<VecT const*>(&params.qkv_input[src_k_idx + rotated_head_dim_offset]);
        }
        else
        {
            // The rotary helpers update Q and K together. Reuse Q as the ignored K input.
            k = q;
            k_pair = q_pair;
        }

        // Bias should have been fused with QKV projection, but we keep the logic here for unit tests.
        if constexpr (ADD_BIAS)
        {
            auto const q_bias = *reinterpret_cast<VecT const*>(&params.qkv_bias[hidden_idx]);
            auto const q_pair_bias
                = *reinterpret_cast<VecT const*>(&params.qkv_bias[hidden_idx + rotated_head_dim_offset]);

            q = mmha::add(q, q_bias);
            q_pair = mmha::add(q_pair, q_pair_bias);
            if (!params.q_only_input)
            {
                auto const k_bias = *reinterpret_cast<VecT const*>(&params.qkv_bias[hidden_idx_kv + src_k_offset]);
                auto const v_bias = *reinterpret_cast<VecT const*>(&params.qkv_bias[hidden_idx_kv + src_v_offset]);
                auto const k_pair_bias = *reinterpret_cast<VecT const*>(
                    &params.qkv_bias[hidden_idx_kv + src_k_offset + rotated_head_dim_offset]);
                k = mmha::add(k, k_bias);
                v = mmha::add(v, v_bias);
                k_pair = mmha::add(k_pair, k_pair_bias);
            }
        }

        // Cos/sin cache.
        // For mrope, index by `bounded_global_token_idx` (batch-flat
        // per-token entry) rather than `rotary_position` (request-internal
        // KV-cache position). Mrope cos/sin is a per-token, per-axis quantity
        // -- different requests at the same `rotary_position` have different
        // (T, H, W) coordinates, so sharing a buffer slot across requests
        // (which request-internal indexing forces once `batch_idx` is
        // dropped) silently corrupts attention for multi-context-request
        // batches. Batch-flat indexing also lets Python materialize cos/sin
        // for only the current iteration's tokens (no chunk_end_pos padding).
        [[maybe_unused]] float2 const* rotary_coef_cache_buffer = nullptr;
        if (params.mrope_rotary_cos_sin != nullptr)
        {
            rotary_coef_cache_buffer
                = params.mrope_rotary_cos_sin + static_cast<size_t>(bounded_global_token_idx) * params.half_rotary_dim;
        }
        else
        {
            rotary_coef_cache_buffer
                = params.rotary_coef_cache_buffer + static_cast<size_t>(rotary_position) * params.half_rotary_dim;
        }

        if constexpr (ROTARY_TYPE == RotaryPositionEmbeddingType::GPT_NEOX)
        {
            rotary_coef_cache_buffer += gptneox_rotary_dim_idx;
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
            {
                GPTNeoXEltT& q_ = reinterpret_cast<GPTNeoXEltT*>(&q)[elt_id];
                GPTNeoXEltT q_pair_ = reinterpret_cast<GPTNeoXEltT*>(&q_pair)[elt_id];
                GPTNeoXEltT& k_ = reinterpret_cast<GPTNeoXEltT*>(&k)[elt_id];
                GPTNeoXEltT k_pair_ = reinterpret_cast<GPTNeoXEltT*>(&k_pair)[elt_id];

                // Load cos/sin from cache.
                float2 rotary_coef_cache
                    = valid_rotary_dim_idx ? rotary_coef_cache_buffer[elt_id] : masked_rotary_cos_sin;

                // Preprocess sin for second half rotary dim.
                rotary_coef_cache.y = first_half ? -rotary_coef_cache.y : rotary_coef_cache.y;
                mmha::apply_rotary_embedding_gptneox(q_, q_pair_, k_, k_pair_, rotary_coef_cache);
            }
        }
        else if constexpr (ROTARY_TYPE == RotaryPositionEmbeddingType::GPTJ)
        {
            rotary_coef_cache_buffer += gptj_rotary_dim_idx;
// Pack two elements into one for gptj rotary embedding.
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
            {
                GPTJEltT& q_ = reinterpret_cast<GPTJEltT*>(&q)[elt_id];
                GPTJEltT& k_ = reinterpret_cast<GPTJEltT*>(&k)[elt_id];

                // Load cos/sin from cache.
                float2 rotary_coef_cache
                    = valid_rotary_dim_idx ? rotary_coef_cache_buffer[elt_id] : masked_rotary_cos_sin;
                mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache);
            }
        }

        if (params.logn_scaling != nullptr)
        {
            float logn_scale = params.logn_scaling[token_idx_in_kv_cache];
            q = mmha::mul<VecT, float, VecT>(logn_scale, q);
        }

        auto const channelIdx = head_dim_vec_idx;
        bool const useKVCache = !params.q_only_input && (GEN_PHASE || params.kv_cache_buffer.data != nullptr);
        bool valid_kv_cache_pos = useKVCache;

        auto kDst = useKVCache
            ? reinterpret_cast<TCache*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, token_idx_in_kv_cache))
            : (TCache*) (nullptr);
        auto vDst = useKVCache
            ? reinterpret_cast<TCache*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, token_idx_in_kv_cache))
            : (TCache*) (nullptr);
        auto inBlockIdx = useKVCache
            ? params.kv_cache_buffer.getKVLocalIdx(token_idx_in_kv_cache, kv_head_idx, VECS_PER_HEAD, channelIdx)
            : int32_t(0);

        // Make sure pairs of q or v vecs have been read before write.
        __syncthreads();

        // Only update valid tokens.
        if (valid_token)
        {
            auto const dst_q_idx = static_cast<size_t>(bounded_global_token_idx) * params.q_hidden_size + hidden_idx;
            VecT* q_ptr = STORE_QKV ? reinterpret_ptr<T, VecT>(params.qkv_input, src_q_idx)
                                    : reinterpret_ptr<T, VecT>(params.q_output, dst_q_idx);

            // Cast float scale to dst data type.
            using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
            [[maybe_unused]] TScale scaleOrigQuant;
            if constexpr (FP8_OUTPUT || ENABLE_8BITS_CACHE)
            {
                mmha::convert_from_float(
                    &scaleOrigQuant, params.qkv_scale_orig_quant ? params.qkv_scale_orig_quant[0] : 1.0f);
            }

            if constexpr (FP8_OUTPUT)
            {
                // Quant the vec to fp8 vec with the scale.
                QuantizedEltType* quantized_q_ptr = STORE_QKV
                    ? reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output) + src_q_idx
                    : reinterpret_cast<QuantizedEltType*>(params.q_output) + dst_q_idx;
                mmha::store_8bits_vec(quantized_q_ptr, q, 0, scaleOrigQuant);
            }
            else
            {
                *q_ptr = q;
            }
            if (!params.q_only_input
                && ((params.head_num == params.kv_head_num) || (head_idx == (kv_head_idx * params.qheads_per_kv_head))))
            {
                if constexpr (STORE_QKV)
                {
                    if constexpr (FP8_OUTPUT)
                    {
                        // Quant the vec to fp8 vec with the scale.
                        mmha::store_8bits_vec(reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output), k,
                            src_k_idx, scaleOrigQuant);
                        mmha::store_8bits_vec(reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output), v,
                            src_v_idx, scaleOrigQuant);
                    }
                    else
                    {
                        *reinterpret_cast<VecT*>(&params.qkv_input[src_k_idx]) = k;
                        if constexpr (ADD_BIAS)
                        {
                            *reinterpret_cast<VecT*>(&params.qkv_input[src_v_idx]) = v;
                        }
                    }
                }

                if (valid_kv_cache_pos && !helix_inactive)
                {
                    if constexpr (ENABLE_8BITS_CACHE)
                    {
                        inBlockIdx = inBlockIdx * ELTS_PER_VEC;
                        // Cast float scale to dst data type. Default to 1.0
                        // when the scale pointer is nullptr; the surrounding
                        // peer reads in this file already follow this pattern
                        // (e.g. lines 579, 989).
                        using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
                        TScale scaleOrigQuant;
                        mmha::convert_from_float(&scaleOrigQuant,
                            params.qkv_scale_orig_quant != nullptr ? params.qkv_scale_orig_quant[0] : 1.0f);
                        // Store 8bits kv cache.
                        mmha::store_8bits_vec(kDst, k, inBlockIdx, scaleOrigQuant);
                        mmha::store_8bits_vec(vDst, v, inBlockIdx, scaleOrigQuant);
                    }
                    else if constexpr (ENABLE_4BITS_CACHE)
                    {
                        auto* kBlockScales = reinterpret_cast<uint8_t*>(
                            params.kv_cache_block_scales_buffer.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                        auto* vBlockScales = reinterpret_cast<uint8_t*>(
                            params.kv_cache_block_scales_buffer.getVBlockPtr(batch_idx, token_idx_in_kv_cache));
                        float kSecondLevelSF = params.qkv_scale_orig_quant[1];
                        float vSecondLevelSF = params.qkv_scale_orig_quant[2];
                        auto& kPacked = reinterpret_cast<PackedVec<T>&>(k);
                        auto& vPacked = reinterpret_cast<PackedVec<T>&>(v);
                        quantizeAndWriteFP4KVCache<T, VECS_PER_HEAD>(kBlockScales, vBlockScales,
                            reinterpret_cast<uint32_t*>(kDst), reinterpret_cast<uint32_t*>(vDst), kSecondLevelSF,
                            vSecondLevelSF, inBlockIdx, kPacked, vPacked);
                    }
                    else
                    {
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = k;
                        reinterpret_cast<VecT*>(vDst)[inBlockIdx] = v;
                    }
                    zeroPagedVCacheTail<T, TCache, VecT, GEN_PHASE>(params, batch_idx, token_idx_in_kv_cache,
                        cache_seq_len, kv_head_idx, VECS_PER_HEAD, channelIdx,
                        /*isUniqueTokenWriter=*/global_token_idx == bounded_global_token_idx);
                }
            }
        }
    }

    // Prepare values for fmha.
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
        // Reset fmha tile counter to 0 before launching fmha kernels.
        if (params.fmha_tile_counter)
        {
            params.fmha_tile_counter[0] = 0u;
        }
        // Take the quantization scales into consideration.
        float q_scale_quant_orig, k_scale_quant_orig, v_scale_quant_orig;
        if constexpr (ENABLE_4BITS_CACHE)
        {
            q_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
            k_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[1] : 1.f;
            v_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[2] : 1.f;
        }
        else
        {
            q_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
            k_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
            v_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
        }
        float o_scale_orig_quant = params.o_scale_orig_quant ? params.o_scale_orig_quant[0] : 1.f;
        if (params.fmha_bmm1_scale)
        {
            // The scale after fmha bmm1.
            params.fmha_bmm1_scale[0] = q_scale_quant_orig * k_scale_quant_orig * params.fmha_host_bmm1_scale;
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            params.fmha_bmm1_scale[1] = params.fmha_bmm1_scale[0] * kLog2e;
        }
        if (params.fmha_bmm2_scale)
        {
            // The scale after fmha bmm2.
            params.fmha_bmm2_scale[0] = o_scale_orig_quant * v_scale_quant_orig;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR=1 (default unset = off): token-major context-phase QKV preprocessing.
//
// applyBiasRopeUpdateKVCacheV2 gives every Q head its own grid row (blockIdx.y). For each token, each of the head_num
// Q-head rows reloads the token's metadata (tokens_info, sequence and cache lengths, paged-KV block pointers) and its
// cos/sin, and loads and rotates the K vector and loads the V vector of its KV head, although only the first Q head of
// each KV head stores them: with 64 Q heads on 8 KV heads, every K and V vector is loaded and every K vector rotated
// 8 times per token. ropeKvTokenMajorKernel gives each warp whole tokens. A lane keeps one 16-byte channel of the head
// dimension, so its cos/sin, the token's metadata and the block pointers are read once per lane and token, and the
// warp walks the token's [Q heads | K heads | V heads] row in 512-byte steps: each Q, K and V vector is loaded once
// (Q and K with their rotary pair) and each K vector is rotated once.
// Per element it runs the V2 kernel's helpers with the V2 kernel's operands (mmha::apply_rotary_embedding_gptneox
// with the same cos/sin and sign, mmha::store_8bits_vec with the same scale, the KVBlockArray addressing,
// zeroPagedVCacheTail, the FMHA scale block and PDL), so the FP8 Q, the paged K/V bytes and the FMHA scales it writes
// are the V2 kernel's.
// It takes context-phase launches only, with a cos/sin cache and GPT-NeoX-style rotation (RoPE, long RoPE, YaRN),
// BF16 QKV without bias, a separate FP8 Q output, an FP8 paged KV cache, head size 64 and Q and KV head counts that
// are multiples of 4 (GPT-OSS: 64 and 8 Q and KV heads per rank); everything else, and every generation-phase launch,
// keeps the V2 kernel (ropeKvTokenMajorUnsupported). With the flag set, one line goes to stderr at the first launch
// the kernel takes, and one at the first context-phase launch it leaves to the V2 kernel, with the reason.
// The flag's value picks one of three occupancy/ILP variants of the same kernel (identical per-element work and
// output; kRopeKvTokenMajorVariants): 1 = 4 Q steps of loads in flight at up to 64 registers (4 blocks of 256 threads
// per SM), 2 = 2 steps at up to 51 registers (5 blocks), 3 = 4 steps at up to 85 registers (3 blocks). Any other
// value is off. tllm_gptoss_ropekv_token_major_override() (unfusedAttentionKernels_2_bf16_fp8.cu) sets the mode per
// process for in-process A/B tests.

#if defined(ENABLE_BF16) && defined(ENABLE_FP8)
template <typename T, typename TCache, typename KVCacheBuffer, int Dh>
inline constexpr bool kRopeKvTokenMajorInstantiated = std::is_same_v<T, __nv_bfloat16>
    && std::is_same_v<TCache, __nv_fp8_e4m3> && std::is_same_v<KVCacheBuffer, KVBlockArray> && Dh == 64;
#else
template <typename T, typename TCache, typename KVCacheBuffer, int Dh>
inline constexpr bool kRopeKvTokenMajorInstantiated = false;
#endif

constexpr int kRopeKvTokenMajorThreads = 256;
constexpr int kRopeKvTokenMajorVariants = 3;

// Process-wide mode: -1 follows TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR, 0 keeps the V2 kernel, 1..3 takes the launches the
// token-major kernel covers with that variant.
inline std::atomic<int>& ropeKvTokenMajorOverride()
{
    static std::atomic<int> mode{-1};
    return mode;
}

// 0 = off, else the variant (1..kRopeKvTokenMajorVariants).
inline int ropeKvTokenMajorVariant()
{
    static int const fromEnv = []
    {
        char const* v = std::getenv("TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR");
        if (v == nullptr || v[0] < '1' || v[0] > '0' + kRopeKvTokenMajorVariants || v[1] != '\0')
        {
            return 0;
        }
        return v[0] - '0';
    }();
    int const mode = ropeKvTokenMajorOverride().load(std::memory_order_relaxed);
    return mode < 0 ? fromEnv : (mode <= kRopeKvTokenMajorVariants ? mode : 0);
}

// nullptr if ropeKvTokenMajorKernel covers this context-phase launch, else why not.
template <int Dh, typename T, typename KVCacheBuffer>
char const* ropeKvTokenMajorUnsupported(QKVPreprocessingParams<T, KVCacheBuffer> const& p)
{
    constexpr int kHeadsPerStep = 32 / (Dh * int(sizeof(T)) / 16);
    if (p.generation_phase || p.cross_attention)
    {
        return "not a self-attention context launch";
    }
    if (p.q_only_input)
    {
        return "Q-only input";
    }
    if (!p.separate_q_kv_output || p.q_output == nullptr || p.q_output == p.qkv_input)
    {
        return "no separate Q output";
    }
    if (!p.quantized_fp8_output)
    {
        return "Q output is not FP8";
    }
    if (p.qkv_bias != nullptr)
    {
        return "QKV bias";
    }
    if (p.logn_scaling != nullptr)
    {
        return "logn scaling";
    }
    if (p.position_embedding_type != PositionEmbeddingType::kROPE_GPT_NEOX
        && p.position_embedding_type != PositionEmbeddingType::kLONG_ROPE
        && p.position_embedding_type != PositionEmbeddingType::kYARN)
    {
        return "position embedding type";
    }
    if (p.mrope_rotary_cos_sin != nullptr || p.rotary_coef_cache_buffer == nullptr)
    {
        return "cos/sin source";
    }
    if (p.rotary_embedding_dim <= 0 || p.rotary_embedding_dim > p.size_per_head)
    {
        return "rotary dimension";
    }
    if (p.spec_decoding_position_offsets != nullptr || p.helix_position_offsets != nullptr
        || p.helix_is_inactive_rank != nullptr)
    {
        return "position offsets";
    }
    if (p.size_per_head != Dh || p.kv_head_num <= 0 || p.head_num % kHeadsPerStep != 0
        || p.kv_head_num % kHeadsPerStep != 0 || p.head_num % p.kv_head_num != 0
        || p.qheads_per_kv_head != p.head_num / p.kv_head_num)
    {
        return "head size or head counts";
    }
    if (p.token_num <= 0 || p.multi_processor_count <= 0
        || ((p.tokens_info == nullptr || p.seq_lens == nullptr) && p.max_input_seq_len <= 0))
    {
        return "empty launch";
    }
    return nullptr;
}

template <typename VecT, typename T>
inline __device__ VecT ropeKvTokenMajorLoad(T const* ptr)
{
    static_assert(sizeof(VecT) == sizeof(uint4), "16-byte vectors only");
    // The input is not written by this kernel (separate Q output, paged KV), so the read-only path is safe.
    uint4 const raw = __ldg(reinterpret_cast<uint4 const*>(ptr));
    return *reinterpret_cast<VecT const*>(&raw);
}

// Q_BATCH: Q steps (HEADS_PER_STEP Q heads each) whose loads are issued before the first of them is rotated and
// stored. MIN_BLOCKS: resident blocks per SM the register allocation must allow.
template <typename T, typename TCache, int Dh, typename KVCacheBuffer, int Q_BATCH, int MIN_BLOCKS>
__global__ void __launch_bounds__(kRopeKvTokenMajorThreads, MIN_BLOCKS) ropeKvTokenMajorKernel(
    QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    using VecT = typename VecType<T>::Type;
    using QuantizedEltType = __nv_fp8_e4m3;
    using GPTNeoXEltT = typename VecType<T>::GPTNeoXEltType;
    constexpr int BYTES_PER_LOAD = 16;
    constexpr int ELTS_PER_VEC = BYTES_PER_LOAD / sizeof(T);
    constexpr int VECS_PER_HEAD = Dh * sizeof(T) / BYTES_PER_LOAD;
    constexpr int HEADS_PER_STEP = WARP_SIZE / VECS_PER_HEAD;
    static_assert(sizeof(VecT) == BYTES_PER_LOAD && WARP_SIZE % VECS_PER_HEAD == 0, "one head per lane group");
    static_assert(sizeof(TCache) == 1, "8-bit KV cache only");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // The lane's channel of the head dimension: the same in every Q, K and V head it handles.
    int const lane = threadIdx.x % WARP_SIZE;
    int const channelIdx = lane % VECS_PER_HEAD;
    int const headInStep = lane / VECS_PER_HEAD;
    int const head_dim_idx = channelIdx * ELTS_PER_VEC;
    bool const first_half = head_dim_idx < params.half_rotary_dim;
    int const gptneox_rotary_dim_idx = first_half ? head_dim_idx : (head_dim_idx - params.half_rotary_dim);
    bool const valid_rotary_dim_idx = head_dim_idx < params.rotary_embedding_dim;
    float2 const masked_rotary_cos_sin = make_float2(1.0f, 0.0f);
    int const rotated_head_dim_offset = first_half ? params.half_rotary_dim : -params.half_rotary_dim;
    bool const variable_sequence_length = params.tokens_info != nullptr && params.seq_lens != nullptr;
    bool const useKVCache = params.kv_cache_buffer.data != nullptr;
    int const qSteps = params.head_num / HEADS_PER_STEP;
    int const kvSteps = params.kv_head_num / HEADS_PER_STEP;

    using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
    TScale scaleOrigQuant;
    mmha::convert_from_float(&scaleOrigQuant, params.qkv_scale_orig_quant ? params.qkv_scale_orig_quant[0] : 1.0f);

    int const warpsPerBlock = blockDim.x / WARP_SIZE;
    for (int token = blockIdx.x * warpsPerBlock + threadIdx.x / WARP_SIZE; token < params.token_num;
         token += gridDim.x * warpsPerBlock)
    {
        // Metadata, as the V2 kernel derives it for a context-phase token.
        int batch_idx, token_idx_in_seq;
        if (variable_sequence_length)
        {
            auto const token_info = params.tokens_info[token];
            batch_idx = token_info.x;
            token_idx_in_seq = token_info.y;
        }
        else
        {
            batch_idx = token / params.max_input_seq_len;
            token_idx_in_seq = token % params.max_input_seq_len;
        }
        int const cache_seq_len = params.cache_seq_lens[batch_idx];
        int const actual_seq_len = variable_sequence_length ? params.seq_lens[batch_idx] : params.max_input_seq_len;
        int const past_seq_len = (cache_seq_len - actual_seq_len);
        bool valid_token = token_idx_in_seq < actual_seq_len;
        token_idx_in_seq = std::min(actual_seq_len - 1, token_idx_in_seq);
        int token_idx_in_kv_cache = past_seq_len + token_idx_in_seq;
        valid_token = valid_token && (token_idx_in_kv_cache < cache_seq_len);
        token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
        // The V2 kernel writes nothing for an invalid token (warp-uniform: one token per warp).
        if (!valid_token)
        {
            continue;
        }
        int const rotary_position = token_idx_in_kv_cache;

        // The lane's cos/sin, with the sign the V2 kernel gives the second half of the rotary dimension.
        float2 rotary_coef[ELTS_PER_VEC];
        {
            float2 const* rotary_coef_cache_buffer = params.rotary_coef_cache_buffer
                + static_cast<size_t>(rotary_position) * params.half_rotary_dim + gptneox_rotary_dim_idx;
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
            {
                float2 c = valid_rotary_dim_idx ? rotary_coef_cache_buffer[elt_id] : masked_rotary_cos_sin;
                c.y = first_half ? -c.y : c.y;
                rotary_coef[elt_id] = c;
            }
        }

        T const* src = params.qkv_input + static_cast<size_t>(token) * params.hidden_size + head_dim_idx;
        QuantizedEltType* qDst = reinterpret_cast<QuantizedEltType*>(params.q_output)
            + static_cast<size_t>(token) * params.q_hidden_size + head_dim_idx;

        // Q: rotate and quantize to the FP8 Q output.
        for (int s0 = 0; s0 < qSteps; s0 += Q_BATCH)
        {
            VecT q[Q_BATCH];
            VecT q_pair[Q_BATCH];
#pragma unroll
            for (int b = 0; b < Q_BATCH; ++b)
            {
                if (s0 + b < qSteps)
                {
                    int const off = ((s0 + b) * HEADS_PER_STEP + headInStep) * Dh;
                    q[b] = ropeKvTokenMajorLoad<VecT>(src + off);
                    q_pair[b] = ropeKvTokenMajorLoad<VecT>(src + off + rotated_head_dim_offset);
                }
            }
#pragma unroll
            for (int b = 0; b < Q_BATCH; ++b)
            {
                if (s0 + b < qSteps)
                {
#pragma unroll
                    for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
                    {
                        GPTNeoXEltT& q_ = reinterpret_cast<GPTNeoXEltT*>(&q[b])[elt_id];
                        GPTNeoXEltT q_pair_ = reinterpret_cast<GPTNeoXEltT*>(&q_pair[b])[elt_id];
                        // The helper rotates a Q and a K element together; this lane's K slot is unused.
                        GPTNeoXEltT k_unused = q_;
                        mmha::apply_rotary_embedding_gptneox(q_, q_pair_, k_unused, q_pair_, rotary_coef[elt_id]);
                    }
                    int const off = ((s0 + b) * HEADS_PER_STEP + headInStep) * Dh;
                    mmha::store_8bits_vec(qDst + off, q[b], 0, scaleOrigQuant);
                }
            }
        }

        if (!useKVCache)
        {
            continue;
        }
        // K and V: rotate K, quantize both into the paged KV cache.
        auto* kDst
            = reinterpret_cast<TCache*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
        auto* vDst
            = reinterpret_cast<TCache*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, token_idx_in_kv_cache));
        T const* srcK = src + params.q_hidden_size;
        T const* srcV = srcK + params.kv_hidden_size;
        for (int s = 0; s < kvSteps; ++s)
        {
            int const kv_head_idx = s * HEADS_PER_STEP + headInStep;
            int const off = kv_head_idx * Dh;
            VecT k = ropeKvTokenMajorLoad<VecT>(srcK + off);
            VecT k_pair = ropeKvTokenMajorLoad<VecT>(srcK + off + rotated_head_dim_offset);
            VecT const v = ropeKvTokenMajorLoad<VecT>(srcV + off);
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
            {
                GPTNeoXEltT& k_ = reinterpret_cast<GPTNeoXEltT*>(&k)[elt_id];
                GPTNeoXEltT k_pair_ = reinterpret_cast<GPTNeoXEltT*>(&k_pair)[elt_id];
                // The helper rotates a Q and a K element together; this lane's Q slot is unused.
                GPTNeoXEltT q_unused = k_;
                mmha::apply_rotary_embedding_gptneox(q_unused, k_pair_, k_, k_pair_, rotary_coef[elt_id]);
            }
            int const inBlockIdx = params.kv_cache_buffer.getKVLocalIdx(
                                       token_idx_in_kv_cache, kv_head_idx, VECS_PER_HEAD, channelIdx)
                * ELTS_PER_VEC;
            mmha::store_8bits_vec(kDst, k, inBlockIdx, scaleOrigQuant);
            mmha::store_8bits_vec(vDst, v, inBlockIdx, scaleOrigQuant);
            zeroPagedVCacheTail<T, TCache, VecT, /*GEN_PHASE=*/false>(params, batch_idx, token_idx_in_kv_cache,
                cache_seq_len, kv_head_idx, VECS_PER_HEAD, channelIdx, /*isUniqueTokenWriter=*/true);
        }
    }

    // Prepare values for fmha (the V2 kernel's block, run by one thread of the grid).
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Reset fmha tile counter to 0 before launching fmha kernels.
        if (params.fmha_tile_counter)
        {
            params.fmha_tile_counter[0] = 0u;
        }
        // Take the quantization scales into consideration (8-bit KV cache: one scale for Q, K and V).
        float const q_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
        float const k_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
        float const v_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
        float const o_scale_orig_quant = params.o_scale_orig_quant ? params.o_scale_orig_quant[0] : 1.f;
        if (params.fmha_bmm1_scale)
        {
            // The scale after fmha bmm1.
            params.fmha_bmm1_scale[0] = q_scale_quant_orig * k_scale_quant_orig * params.fmha_host_bmm1_scale;
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            params.fmha_bmm1_scale[1] = params.fmha_bmm1_scale[0] * kLog2e;
        }
        if (params.fmha_bmm2_scale)
        {
            // The scale after fmha bmm2.
            params.fmha_bmm2_scale[0] = o_scale_orig_quant * v_scale_quant_orig;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Launches ropeKvTokenMajorKernel (variant 1..kRopeKvTokenMajorVariants) and returns true if it covers this launch,
// else returns false (the caller runs the V2 kernel).
template <typename T, typename TCache, int Dh, typename KVCacheBuffer>
bool launchRopeKvTokenMajor(QKVPreprocessingParams<T, KVCacheBuffer> const& params, cudaStream_t stream, int variant)
{
    char const* const reason = ropeKvTokenMajorUnsupported<Dh>(params);
    if (reason != nullptr)
    {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true))
        {
            std::fprintf(stderr,
                "[TRT-LLM] [INFO] TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR=%d: a context-phase launch keeps "
                "applyBiasRopeUpdateKVCacheV2 (%s; heads %d/%d, head size %d, tokens %d); printed once\n",
                variant, reason, params.head_num, params.kv_head_num, params.size_per_head, params.token_num);
        }
        return false;
    }
    using Params = QKVPreprocessingParams<T, KVCacheBuffer>;
    void (*kernel)(Params) = variant == 2 ? ropeKvTokenMajorKernel<T, TCache, Dh, KVCacheBuffer, 2, 5>
        : variant == 3                    ? ropeKvTokenMajorKernel<T, TCache, Dh, KVCacheBuffer, 4, 3>
                                          : ropeKvTokenMajorKernel<T, TCache, Dh, KVCacheBuffer, 4, 4>;
    constexpr int kWarpsPerBlock = kRopeKvTokenMajorThreads / WARP_SIZE;
    // Grid-stride over tokens: at most one full occupancy of the SMs (2048 threads each), one token per warp.
    int const numBlocks = std::min(int(divUp(params.token_num, kWarpsPerBlock)),
        params.multi_processor_count * (2048 / kRopeKvTokenMajorThreads));
    cudaLaunchConfig_t config;
    config.gridDim = dim3(numBlocks);
    config.blockDim = dim3(kRopeKvTokenMajorThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, params);
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true))
    {
        std::fprintf(stderr,
            "[TRT-LLM] [INFO] TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR=%d active: context-phase RoPE + FP8 Q + paged KV write "
            "run in ropeKvTokenMajorKernel (heads %d/%d, head size %d, rotary dim %d; first launch %d tokens, %d "
            "blocks of %d threads)\n",
            variant, params.head_num, params.kv_head_num, params.size_per_head, params.rotary_embedding_dim,
            params.token_num, numBlocks, kRopeKvTokenMajorThreads);
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TLLM_GPTOSS_ROPEKV_GEN=1|2|3 (default unset = off): split-token generation-phase QKV preprocessing.
//
// A decode step runs one QKV preprocessing launch per attention layer on a few thousand tokens (requests x query
// tokens per request). applyBiasRopeUpdateKVCacheV2 gives every Q head its own grid row (blockIdx.y), so for each token
// each of the head_num Q-head rows reloads the token's metadata, its cos/sin and the paged-KV block pointers, and loads
// and rotates the K vector and loads the V vector of its KV head, although only the first Q head of each KV head
// stores them. At a few thousand tokens the launch moves little data; its instruction count and the latency of its
// short dependent chains set its time.
// ropeKvGenKernel gives each token WARPS_PER_TOKEN warps ("slices"): slice s owns the KV heads
// [s * kvPerSlice, (s + 1) * kvPerSlice) and the Q heads that share them. A lane keeps one 16-byte channel of the head
// dimension, so its cos/sin, the token's metadata and the block pointers are read once per lane and slice. The slice's
// [Q heads | K heads | V heads] are walked in 512-byte steps (4 heads of 8 lanes); the loads of STEP_BATCH steps are
// issued before the first of them is rotated, quantized and stored. Each K vector is loaded and rotated once and each V
// vector loaded once.
// Per element it runs the V2 kernel's helpers with the V2 kernel's operands, as ropeKvTokenMajorKernel does
// (mmha::apply_rotary_embedding_gptneox with the same cos/sin and sign, mmha::store_8bits_vec with the same scale, the
// KVBlockArray addressing, zeroPagedVCacheTail, the FMHA scale block and PDL), and it derives batch, position and
// validity with the V2 kernel's own expressions (including speculative-decoding position offsets and the GEN_PHASE
// specialisation the dispatch picks), so the FP8 Q, the paged K/V bytes and the FMHA scales it writes are the V2
// kernel's.
// It takes generation-phase launches only (params.generation_phase, as XqaDispatcher and
// trtllm_gen_generation_preprocess set it), with a cos/sin cache and GPT-NeoX-style rotation (RoPE, long RoPE, YaRN),
// BF16 QKV without bias, a separate FP8 Q output, an FP8 paged KV cache, head size 64, and KV heads divisible by the
// variant's warps per token; everything else keeps the V2 kernel (ropeKvGenUnsupported). With the flag set, one line
// goes to stderr at the first launch the kernel takes, and one at the first generation-phase launch it leaves to the V2
// kernel, with the reason. The flag's value picks a variant (kRopeKvGenVariants; identical per-element work and
// output): 1 = 4 warps per token, all of a slice's loads in flight (5 steps at 64/8 heads), up to 85 registers;
// 2 = 8 warps per token, 3 steps, up to 64 registers; 3 = 2 warps per token, two rounds of 5 steps, up to 85
// registers. Any other value is off. tllm_gptoss_ropekv_gen_override() (unfusedAttentionKernels_2_bf16_fp8.cu) sets
// the mode per process for in-process A/B tests. Context-phase launches are TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR's.

constexpr int kRopeKvGenThreads = 128;
constexpr int kRopeKvGenVariants = 3;

// Process-wide mode: -1 follows TLLM_GPTOSS_ROPEKV_GEN, 0 keeps the V2 kernel, 1..3 takes the generation-phase
// launches the split-token kernel covers with that variant.
inline std::atomic<int>& ropeKvGenOverride()
{
    static std::atomic<int> mode{-1};
    return mode;
}

// 0 = off, else the variant (1..kRopeKvGenVariants).
inline int ropeKvGenVariant()
{
    static int const fromEnv = []
    {
        char const* v = std::getenv("TLLM_GPTOSS_ROPEKV_GEN");
        if (v == nullptr || v[0] < '1' || v[0] > '0' + kRopeKvGenVariants || v[1] != '\0')
        {
            return 0;
        }
        return v[0] - '0';
    }();
    int const mode = ropeKvGenOverride().load(std::memory_order_relaxed);
    return mode < 0 ? fromEnv : (mode <= kRopeKvGenVariants ? mode : 0);
}

// Warps per token of a variant.
inline constexpr int ropeKvGenWarpsPerToken(int variant)
{
    return variant == 2 ? 8 : (variant == 3 ? 2 : 4);
}

// nullptr if ropeKvGenKernel covers this generation-phase launch with warpsPerToken warps per token, else why not.
template <int Dh, typename T, typename KVCacheBuffer>
char const* ropeKvGenUnsupported(QKVPreprocessingParams<T, KVCacheBuffer> const& p, int warpsPerToken)
{
    if (!p.generation_phase || p.cross_attention)
    {
        return "not a self-attention generation launch";
    }
    if (p.q_only_input)
    {
        return "Q-only input";
    }
    if (!p.separate_q_kv_output || p.q_output == nullptr || p.q_output == p.qkv_input)
    {
        return "no separate Q output";
    }
    if (!p.quantized_fp8_output)
    {
        return "Q output is not FP8";
    }
    if (p.qkv_bias != nullptr)
    {
        return "QKV bias";
    }
    if (p.logn_scaling != nullptr)
    {
        return "logn scaling";
    }
    if (p.position_embedding_type != PositionEmbeddingType::kROPE_GPT_NEOX
        && p.position_embedding_type != PositionEmbeddingType::kLONG_ROPE
        && p.position_embedding_type != PositionEmbeddingType::kYARN)
    {
        return "position embedding type";
    }
    if (p.mrope_rotary_cos_sin != nullptr || p.rotary_coef_cache_buffer == nullptr)
    {
        return "cos/sin source";
    }
    if (p.rotary_embedding_dim <= 0 || p.rotary_embedding_dim > p.size_per_head)
    {
        return "rotary dimension";
    }
    if (p.helix_position_offsets != nullptr || p.helix_is_inactive_rank != nullptr)
    {
        return "helix";
    }
    if (p.kv_cache_buffer.data == nullptr)
    {
        return "no KV cache";
    }
    if (p.size_per_head != Dh || p.kv_head_num <= 0 || p.head_num % p.kv_head_num != 0
        || p.qheads_per_kv_head != p.head_num / p.kv_head_num || p.kv_head_num % warpsPerToken != 0)
    {
        return "head size or head counts";
    }
    if (p.token_num <= 0 || p.multi_processor_count <= 0 || p.max_input_seq_len <= 0)
    {
        return "empty launch";
    }
    return nullptr;
}

// GEN_PHASE: the V2 kernel's template flag for this launch (kernelV2DispatchHeadSize: a KV cache, one query token per
// request and params.generation_phase). STEP_BATCH: steps (4 heads each) whose loads are issued before the first of
// them is used. MIN_BLOCKS: resident blocks of kRopeKvGenThreads per SM the register allocation must allow.
template <typename T, typename TCache, int Dh, typename KVCacheBuffer, bool GEN_PHASE, int WARPS_PER_TOKEN,
    int STEP_BATCH, int MIN_BLOCKS>
__global__ void __launch_bounds__(kRopeKvGenThreads, MIN_BLOCKS)
    ropeKvGenKernel(QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    using VecT = typename VecType<T>::Type;
    using QuantizedEltType = __nv_fp8_e4m3;
    using GPTNeoXEltT = typename VecType<T>::GPTNeoXEltType;
    constexpr int BYTES_PER_LOAD = 16;
    constexpr int ELTS_PER_VEC = BYTES_PER_LOAD / sizeof(T);
    constexpr int VECS_PER_HEAD = Dh * sizeof(T) / BYTES_PER_LOAD;
    constexpr int HEADS_PER_STEP = WARP_SIZE / VECS_PER_HEAD;
    static_assert(sizeof(VecT) == BYTES_PER_LOAD && WARP_SIZE % VECS_PER_HEAD == 0, "one head per lane group");
    static_assert(sizeof(TCache) == 1, "8-bit KV cache only");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // The lane's channel of the head dimension: the same in every Q, K and V head it handles.
    int const lane = threadIdx.x % WARP_SIZE;
    int const channelIdx = lane % VECS_PER_HEAD;
    int const headInStep = lane / VECS_PER_HEAD;
    int const head_dim_idx = channelIdx * ELTS_PER_VEC;
    bool const first_half = head_dim_idx < params.half_rotary_dim;
    int const gptneox_rotary_dim_idx = first_half ? head_dim_idx : (head_dim_idx - params.half_rotary_dim);
    bool const valid_rotary_dim_idx = head_dim_idx < params.rotary_embedding_dim;
    float2 const masked_rotary_cos_sin = make_float2(1.0f, 0.0f);
    int const rotated_head_dim_offset = first_half ? params.half_rotary_dim : -params.half_rotary_dim;
    bool const variable_sequence_length = params.tokens_info != nullptr && params.seq_lens != nullptr;
    // Head slots of a slice: [qPerSlice Q heads | kvPerSlice K heads | kvPerSlice V heads].
    int const kvPerSlice = params.kv_head_num / WARPS_PER_TOKEN;
    int const qPerSlice = kvPerSlice * params.qheads_per_kv_head;
    int const kEnd = qPerSlice + kvPerSlice;
    int const slots = kEnd + kvPerSlice;
    int const steps = (slots + HEADS_PER_STEP - 1) / HEADS_PER_STEP;

    using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
    TScale scaleOrigQuant;
    mmha::convert_from_float(&scaleOrigQuant, params.qkv_scale_orig_quant ? params.qkv_scale_orig_quant[0] : 1.0f);

    int const warpsPerBlock = blockDim.x / WARP_SIZE;
    int const numItems = params.token_num * WARPS_PER_TOKEN;
    for (int item = blockIdx.x * warpsPerBlock + threadIdx.x / WARP_SIZE; item < numItems;
         item += gridDim.x * warpsPerBlock)
    {
        int const token = item / WARPS_PER_TOKEN;
        int const slice = item % WARPS_PER_TOKEN;
        int const qHead0 = slice * qPerSlice;
        int const kvHead0 = slice * kvPerSlice;
        T const* src = params.qkv_input + static_cast<size_t>(token) * params.hidden_size + head_dim_idx;
        // Loads the vectors of steps [s0, s0 + STEP_BATCH): the slot's Q, K or V head, at its offset in the token's
        // QKV row, and the rotary pair of a Q or K head.
        VecT x[STEP_BATCH];
        VecT x_pair[STEP_BATCH];
        auto const loadSteps = [&](int s0)
        {
#pragma unroll
            for (int b = 0; b < STEP_BATCH; ++b)
            {
                int const slot = (s0 + b) * HEADS_PER_STEP + headInStep;
                if (s0 + b < steps && slot < slots)
                {
                    int const off = slot < qPerSlice ? (qHead0 + slot) * Dh
                        : slot < kEnd                ? params.q_hidden_size + (kvHead0 + slot - qPerSlice) * Dh
                                      : params.q_hidden_size + params.kv_hidden_size + (kvHead0 + slot - kEnd) * Dh;
                    x[b] = ropeKvTokenMajorLoad<VecT>(src + off);
                    if (slot < kEnd)
                    {
                        x_pair[b] = ropeKvTokenMajorLoad<VecT>(src + off + rotated_head_dim_offset);
                    }
                }
            }
        };
        // The first steps' loads go out before the metadata chain below (cache length, then cos/sin and block
        // pointers), so they overlap it. Every token < token_num has a QKV row; the V2 kernel also loads before its
        // validity check.
        loadSteps(0);
        // Metadata, with the V2 kernel's expressions for this launch (GEN_PHASE as the dispatch picked it).
        int batch_idx, token_idx_in_seq;
        if constexpr (GEN_PHASE)
        {
            batch_idx = token;
            token_idx_in_seq = 0;
        }
        else if (variable_sequence_length)
        {
            auto const token_info = params.tokens_info[token];
            batch_idx = token_info.x;
            token_idx_in_seq = token_info.y;
        }
        else
        {
            batch_idx = token / params.max_input_seq_len;
            token_idx_in_seq = token % params.max_input_seq_len;
        }
        int const cache_seq_len = params.cache_seq_lens[batch_idx];
        int const actual_seq_len = variable_sequence_length ? params.seq_lens[batch_idx] : params.max_input_seq_len;
        int const past_seq_len = (cache_seq_len - actual_seq_len);
        bool valid_token = GEN_PHASE || (token_idx_in_seq < actual_seq_len);
        token_idx_in_seq = std::min(actual_seq_len - 1, token_idx_in_seq);
        int token_idx_in_kv_cache = past_seq_len + token_idx_in_seq;
        valid_token = valid_token && (token_idx_in_kv_cache < cache_seq_len);
        token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
        // The V2 kernel writes nothing for an invalid token (warp-uniform: a slice is one token).
        if (!valid_token)
        {
            continue;
        }
        int const rotary_position = params.spec_decoding_position_offsets != nullptr
            ? (params.spec_decoding_position_offsets[token_idx_in_seq + batch_idx * params.max_input_seq_len]
                + cache_seq_len - actual_seq_len)
            : token_idx_in_kv_cache;

        // The lane's cos/sin, with the sign the V2 kernel gives the second half of the rotary dimension.
        float2 rotary_coef[ELTS_PER_VEC];
        {
            float2 const* rotary_coef_cache_buffer = params.rotary_coef_cache_buffer
                + static_cast<size_t>(rotary_position) * params.half_rotary_dim + gptneox_rotary_dim_idx;
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
            {
                float2 c = valid_rotary_dim_idx ? rotary_coef_cache_buffer[elt_id] : masked_rotary_cos_sin;
                c.y = first_half ? -c.y : c.y;
                rotary_coef[elt_id] = c;
            }
        }
        auto* kDst = reinterpret_cast<TCache*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
        auto* vDst = reinterpret_cast<TCache*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, token_idx_in_kv_cache));

        QuantizedEltType* qDst = reinterpret_cast<QuantizedEltType*>(params.q_output)
            + static_cast<size_t>(token) * params.q_hidden_size + head_dim_idx;

        for (int s0 = 0; s0 < steps; s0 += STEP_BATCH)
        {
            if (s0 > 0)
            {
                loadSteps(s0);
            }
#pragma unroll
            for (int b = 0; b < STEP_BATCH; ++b)
            {
                int const slot = (s0 + b) * HEADS_PER_STEP + headInStep;
                if (s0 + b < steps && slot < slots)
                {
                    if (slot < qPerSlice)
                    {
                        // Q: rotate and quantize to the FP8 Q output.
#pragma unroll
                        for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
                        {
                            GPTNeoXEltT& q_ = reinterpret_cast<GPTNeoXEltT*>(&x[b])[elt_id];
                            GPTNeoXEltT q_pair_ = reinterpret_cast<GPTNeoXEltT*>(&x_pair[b])[elt_id];
                            // The helper rotates a Q and a K element together; this lane's K slot is unused.
                            GPTNeoXEltT k_unused = q_;
                            mmha::apply_rotary_embedding_gptneox(q_, q_pair_, k_unused, q_pair_, rotary_coef[elt_id]);
                        }
                        mmha::store_8bits_vec(qDst + (qHead0 + slot) * Dh, x[b], 0, scaleOrigQuant);
                    }
                    else if (slot < kEnd)
                    {
                        // K: rotate and quantize into the paged KV cache.
                        int const kv_head_idx = kvHead0 + slot - qPerSlice;
#pragma unroll
                        for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
                        {
                            GPTNeoXEltT& k_ = reinterpret_cast<GPTNeoXEltT*>(&x[b])[elt_id];
                            GPTNeoXEltT k_pair_ = reinterpret_cast<GPTNeoXEltT*>(&x_pair[b])[elt_id];
                            // The helper rotates a Q and a K element together; this lane's Q slot is unused.
                            GPTNeoXEltT q_unused = k_;
                            mmha::apply_rotary_embedding_gptneox(q_unused, k_pair_, k_, k_pair_, rotary_coef[elt_id]);
                        }
                        int const inBlockIdx = params.kv_cache_buffer.getKVLocalIdx(
                                                   token_idx_in_kv_cache, kv_head_idx, VECS_PER_HEAD, channelIdx)
                            * ELTS_PER_VEC;
                        mmha::store_8bits_vec(kDst, x[b], inBlockIdx, scaleOrigQuant);
                    }
                    else
                    {
                        // V: quantize into the paged KV cache.
                        int const kv_head_idx = kvHead0 + slot - kEnd;
                        int const inBlockIdx = params.kv_cache_buffer.getKVLocalIdx(
                                                   token_idx_in_kv_cache, kv_head_idx, VECS_PER_HEAD, channelIdx)
                            * ELTS_PER_VEC;
                        mmha::store_8bits_vec(vDst, x[b], inBlockIdx, scaleOrigQuant);
                        zeroPagedVCacheTail<T, TCache, VecT, GEN_PHASE>(params, batch_idx, token_idx_in_kv_cache,
                            cache_seq_len, kv_head_idx, VECS_PER_HEAD, channelIdx, /*isUniqueTokenWriter=*/true);
                    }
                }
            }
        }
    }

    // Prepare values for fmha (the V2 kernel's block, run by one thread of the grid).
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Reset fmha tile counter to 0 before launching fmha kernels.
        if (params.fmha_tile_counter)
        {
            params.fmha_tile_counter[0] = 0u;
        }
        // Take the quantization scales into consideration (8-bit KV cache: one scale for Q, K and V).
        float const q_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
        float const k_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
        float const v_scale_quant_orig = params.qkv_scale_quant_orig ? params.qkv_scale_quant_orig[0] : 1.f;
        float const o_scale_orig_quant = params.o_scale_orig_quant ? params.o_scale_orig_quant[0] : 1.f;
        if (params.fmha_bmm1_scale)
        {
            // The scale after fmha bmm1.
            params.fmha_bmm1_scale[0] = q_scale_quant_orig * k_scale_quant_orig * params.fmha_host_bmm1_scale;
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            params.fmha_bmm1_scale[1] = params.fmha_bmm1_scale[0] * kLog2e;
        }
        if (params.fmha_bmm2_scale)
        {
            // The scale after fmha bmm2.
            params.fmha_bmm2_scale[0] = o_scale_orig_quant * v_scale_quant_orig;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, typename TCache, int Dh, typename KVCacheBuffer, bool GEN_PHASE>
auto ropeKvGenKernelFor(int variant) -> void (*)(QKVPreprocessingParams<T, KVCacheBuffer>)
{
    return variant == 2 ? ropeKvGenKernel<T, TCache, Dh, KVCacheBuffer, GEN_PHASE, ropeKvGenWarpsPerToken(2), 3, 8>
        : variant == 3  ? ropeKvGenKernel<T, TCache, Dh, KVCacheBuffer, GEN_PHASE, ropeKvGenWarpsPerToken(3), 5, 6>
                        : ropeKvGenKernel<T, TCache, Dh, KVCacheBuffer, GEN_PHASE, ropeKvGenWarpsPerToken(1), 5, 6>;
}

// Launches ropeKvGenKernel (variant 1..kRopeKvGenVariants) and returns true if it covers this generation-phase launch,
// else returns false (the caller runs the V2 kernel). genPhaseKernel: the V2 kernel's GEN_PHASE for this launch.
template <typename T, typename TCache, int Dh, typename KVCacheBuffer>
bool launchRopeKvGen(
    QKVPreprocessingParams<T, KVCacheBuffer> const& params, cudaStream_t stream, int variant, bool genPhaseKernel)
{
    int const warpsPerToken = ropeKvGenWarpsPerToken(variant);
    char const* const reason = ropeKvGenUnsupported<Dh>(params, warpsPerToken);
    if (reason != nullptr)
    {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true))
        {
            std::fprintf(stderr,
                "[TRT-LLM] [INFO] TLLM_GPTOSS_ROPEKV_GEN=%d: a generation-phase launch keeps "
                "applyBiasRopeUpdateKVCacheV2 (%s; heads %d/%d, head size %d, tokens %d); printed once\n",
                variant, reason, params.head_num, params.kv_head_num, params.size_per_head, params.token_num);
        }
        return false;
    }
    using Params = QKVPreprocessingParams<T, KVCacheBuffer>;
    void (*kernel)(Params) = genPhaseKernel ? ropeKvGenKernelFor<T, TCache, Dh, KVCacheBuffer, true>(variant)
                                            : ropeKvGenKernelFor<T, TCache, Dh, KVCacheBuffer, false>(variant);
    constexpr int kWarpsPerBlock = kRopeKvGenThreads / WARP_SIZE;
    // One warp per (token, slice); grid-stride beyond 16 blocks per SM.
    int64_t const numWarps = static_cast<int64_t>(params.token_num) * warpsPerToken;
    int const numBlocks = static_cast<int>(
        std::min<int64_t>((numWarps + kWarpsPerBlock - 1) / kWarpsPerBlock, params.multi_processor_count * 16));
    cudaLaunchConfig_t config;
    config.gridDim = dim3(numBlocks);
    config.blockDim = dim3(kRopeKvGenThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, params);
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true))
    {
        std::fprintf(stderr,
            "[TRT-LLM] [INFO] TLLM_GPTOSS_ROPEKV_GEN=%d active: generation-phase RoPE + FP8 Q + paged KV write run "
            "in ropeKvGenKernel (heads %d/%d, head size %d, rotary dim %d, %d warps per token; first launch %d "
            "tokens, %d per request, spec offsets %d, %d blocks of %d threads)\n",
            variant, params.head_num, params.kv_head_num, params.size_per_head, params.rotary_embedding_dim,
            warpsPerToken, params.token_num, params.max_input_seq_len,
            params.spec_decoding_position_offsets != nullptr ? 1 : 0, numBlocks, kRopeKvGenThreads);
    }
    return true;
}

// Use more blocks for the batch dimension in the generation phase.
#define APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, GEN_PHASE, STORE_QKV, DYNAMIC_ROTARY_SCALING, FP8_OUTPUT)    \
    dim3 block(WARP_SIZE, 1);                                                                                          \
    dim3 grid(params.max_input_seq_len, params.head_num);                                                              \
    grid.z = std::min(int(divUp(params.multi_processor_count * WARPS_PER_SM, grid.x * grid.y)),                        \
        int(divUp(params.batch_size, MIN_SEQUENCES_PER_WARP)));                                                        \
    if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX                                        \
        || params.position_embedding_type == PositionEmbeddingType::kLONG_ROPE                                         \
        || params.position_embedding_type == PositionEmbeddingType::kROPE_M                                            \
        || params.position_embedding_type == PositionEmbeddingType::kYARN)                                             \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCache<T, TCache, Dh_MAX, ADD_BIAS, STORE_QKV, KVCacheBuffer,                              \
            RotaryPositionEmbeddingType::GPT_NEOX, DYNAMIC_ROTARY_SCALING, FP8_OUTPUT, GEN_PHASE>                      \
            <<<grid, block, 0, stream>>>(params);                                                                      \
    }                                                                                                                  \
    else if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPTJ)                                      \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCache<T, TCache, Dh_MAX, ADD_BIAS, STORE_QKV, KVCacheBuffer,                              \
            RotaryPositionEmbeddingType::GPTJ, DYNAMIC_ROTARY_SCALING, FP8_OUTPUT, GEN_PHASE>                          \
            <<<grid, block, 0, stream>>>(params);                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCache<T, TCache, Dh_MAX, ADD_BIAS, STORE_QKV, KVCacheBuffer,                              \
            RotaryPositionEmbeddingType::NONE, DYNAMIC_ROTARY_SCALING, FP8_OUTPUT, GEN_PHASE>                          \
            <<<grid, block, 0, stream>>>(params);                                                                      \
    }

#define DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(ADD_BIAS, GEN_PHASE, STORE_QKV)                                 \
    if (dynamic_rotary_scaling)                                                                                        \
    {                                                                                                                  \
        if (params.quantized_fp8_output)                                                                               \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, GEN_PHASE, STORE_QKV, true, true);                       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, GEN_PHASE, STORE_QKV, true, false);                      \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        if (params.quantized_fp8_output)                                                                               \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, GEN_PHASE, STORE_QKV, false, true);                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, GEN_PHASE, STORE_QKV, false, false);                     \
        }                                                                                                              \
    }

template <int Dh_MAX, typename T, typename TCache, typename KVCacheBuffer>
void kernelDispatchHeadSize(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    bool const add_bias = params.qkv_bias != nullptr;
    bool const store_packed_qkv = !params.separate_q_kv_output;
    bool const generation_phase
        = (params.kv_cache_buffer.data != nullptr) && (params.max_input_seq_len == 1) && params.generation_phase;
    bool const dynamic_rotary_scaling = params.rotary_scale_type == RotaryScalingType::kDYNAMIC
        && params.max_input_seq_len > params.rotary_embedding_max_positions;

    constexpr int VEC_SIZE = Rotary_vec_t<T, Dh_MAX>::size;
    // Make sure we have multiple of paired vectors so that the access is aligned.
    TLLM_CHECK_WITH_INFO((params.position_embedding_type != PositionEmbeddingType::kROPE_GPT_NEOX
                             && params.position_embedding_type != PositionEmbeddingType::kLONG_ROPE
                             && params.position_embedding_type == PositionEmbeddingType::kROPE_M)
            || params.half_rotary_dim % VEC_SIZE == 0,
        "Rotary dim size is not supported.");

    if (add_bias)
    {
        if (generation_phase)
        {
            if (store_packed_qkv)
            {
                DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(true, true, true);
            }
            else
            {
                DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(true, true, false);
            }
        }
        else
        {
            if (store_packed_qkv)
            {
                DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(true, false, true);
            }
            else
            {
                DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(true, false, false);
            }
        }
    }
    else
    {
        if (generation_phase)
        {
            if (store_packed_qkv)
            {
                DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(false, true, true);
            }
            else
            {
                DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(false, true, false);
            }
        }
        else
        {
            if (store_packed_qkv)
            {
                DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(false, false, true);
            }
            else
            {
                DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(false, false, false);
            }
        }
    }
}

template <typename T, typename TCache, typename KVCacheBuffer>
void kernelV1Dispatch(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
#ifdef ENABLE_FP4
    constexpr bool isFP4 = std::is_same_v<TCache, __nv_fp4_e2m1>;
#else
    constexpr bool isFP4 = false;
#endif

    // Fall back to v1 kernel.
    // GPTJ Rotary embedding needs at least two elements per thread.
    // NOTE: the FP4 quantize helpers are assuming that each thread processes
    // 8 elements. Hence we set Dh_max == 256 for FP4, even when the head dim
    // is smaller than 128. If this causes performance issues, we need to
    // update the quantization logic.
    if (!isFP4 && params.size_per_head <= 64)
    {
        kernelDispatchHeadSize<64, T, TCache, KVCacheBuffer>(params, stream);
    }
    else if (!isFP4 && params.size_per_head <= 128)
    {
        kernelDispatchHeadSize<128, T, TCache, KVCacheBuffer>(params, stream);
    }
    else if (params.size_per_head <= 256)
    {
        kernelDispatchHeadSize<256, T, TCache, KVCacheBuffer>(params, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "applyBiasRopeUpdateKVCache kernel doesn't support head size = %d", params.size_per_head);
    }
}

#define APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, GEN_PHASE, STORE_QKV, FP8_OUTPUT)                                 \
    dim3 block(BLOCK_SIZE);                                                                                            \
    dim3 grid(1, params.head_num);                                                                                     \
    int num_blocks_for_tokens = int(divUp(params.token_num, tokens_per_cuda_block));                                   \
    calGridSizeWithBestEfficiency(block, grid, num_blocks_for_tokens, params.multi_processor_count, 1024);             \
    cudaLaunchConfig_t config;                                                                                         \
    config.gridDim = grid;                                                                                             \
    config.blockDim = block;                                                                                           \
    config.dynamicSmemBytes = 0;                                                                                       \
    config.stream = stream;                                                                                            \
    cudaLaunchAttribute attrs[1];                                                                                      \
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                                  \
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();                     \
    config.numAttrs = 1;                                                                                               \
    config.attrs = attrs;                                                                                              \
    if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX                                        \
        || params.position_embedding_type == PositionEmbeddingType::kLONG_ROPE                                         \
        || params.position_embedding_type == PositionEmbeddingType::kROPE_M                                            \
        || params.position_embedding_type == PositionEmbeddingType::kYARN)                                             \
    {                                                                                                                  \
        cudaLaunchKernelEx(&config,                                                                                    \
            applyBiasRopeUpdateKVCacheV2<T, TCache, BLOCK_SIZE, Dh, ADD_BIAS, STORE_QKV, FP8_OUTPUT, GEN_PHASE,        \
                KVCacheBuffer, RotaryPositionEmbeddingType::GPT_NEOX>,                                                 \
            params);                                                                                                   \
    }                                                                                                                  \
    else if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPTJ)                                      \
    {                                                                                                                  \
        cudaLaunchKernelEx(&config,                                                                                    \
            applyBiasRopeUpdateKVCacheV2<T, TCache, BLOCK_SIZE, Dh, ADD_BIAS, STORE_QKV, FP8_OUTPUT, GEN_PHASE,        \
                KVCacheBuffer, RotaryPositionEmbeddingType::GPTJ>,                                                     \
            params);                                                                                                   \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        cudaLaunchKernelEx(&config,                                                                                    \
            applyBiasRopeUpdateKVCacheV2<T, TCache, BLOCK_SIZE, Dh, ADD_BIAS, STORE_QKV, FP8_OUTPUT, GEN_PHASE,        \
                KVCacheBuffer, RotaryPositionEmbeddingType::NONE>,                                                     \
            params);                                                                                                   \
    }

#define STORE_QKV_AND_FP8_OUTPUT_DISPATCH(ADD_BIAS, GEN_PHASE)                                                         \
    if (store_packed_qkv)                                                                                              \
    {                                                                                                                  \
        if (params.quantized_fp8_output)                                                                               \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, GEN_PHASE, true, true);                                       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, GEN_PHASE, true, false);                                      \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        if (params.quantized_fp8_output)                                                                               \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, GEN_PHASE, false, true);                                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, GEN_PHASE, false, false);                                     \
        }                                                                                                              \
    }

template <int BLOCK_SIZE, int Dh, typename T, typename TCache, typename KVCacheBuffer>
void kernelV2DispatchHeadSize(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    bool const add_bias = params.qkv_bias != nullptr;
    bool const store_packed_qkv = !params.separate_q_kv_output;
    int const vecs_per_head = (params.size_per_head * sizeof(T) / 16);
    TLLM_CHECK_WITH_INFO(BLOCK_SIZE % vecs_per_head == 0, "Kernel block should be able to handle entire heads.");
    int const tokens_per_cuda_block = BLOCK_SIZE / vecs_per_head;
    bool generation_phase
        = (params.kv_cache_buffer.data != nullptr) && (params.max_input_seq_len == 1) && params.generation_phase;

    if constexpr (kRopeKvTokenMajorInstantiated<T, TCache, KVCacheBuffer, Dh>)
    {
        // TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR: the context-phase launches the token-major kernel covers.
        if (!params.generation_phase)
        {
            int const variant = ropeKvTokenMajorVariant();
            if (variant > 0 && launchRopeKvTokenMajor<T, TCache, Dh>(params, stream, variant))
            {
                return;
            }
        }
        // TLLM_GPTOSS_ROPEKV_GEN: the generation-phase launches the split-token kernel covers.
        else
        {
            int const variant = ropeKvGenVariant();
            if (variant > 0 && launchRopeKvGen<T, TCache, Dh>(params, stream, variant, generation_phase))
            {
                return;
            }
        }
    }

    if (add_bias)
    {
        if (generation_phase)
        {
            STORE_QKV_AND_FP8_OUTPUT_DISPATCH(true, true);
        }
        else
        {
            STORE_QKV_AND_FP8_OUTPUT_DISPATCH(true, false);
        }
    }
    else
    {
        if (generation_phase)
        {
            STORE_QKV_AND_FP8_OUTPUT_DISPATCH(false, true);
        }
        else
        {
            STORE_QKV_AND_FP8_OUTPUT_DISPATCH(false, false);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, typename KVCacheBuffer, int BLOCK_SIZE, int Dh, bool FP8_OUTPUT>
__global__ void updateKVCacheForCrossAttention(QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    // For cross-attention,
    // 1. Load Q from params.qkv_input, and store it to params.q_output.
    // 2. Load K,V from params.cross_kv_input, and store it to params.kv_cache_buffer.

    // NOTE:
    // head_num == kv_head_num
    //   QKV src shape (num_tokens, 3, head_num, size_per_head)
    //                  ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                           m                        n
    // head_num != kv_head_num
    //   QKV src shape: (num_tokens, head_num * size_per_head + 2 * kv_head_num, size_per_head)
    //                   ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                             m                               n
    // Q dst shape: (num_tokens, head_num, size_per_head)
    // KV dst shape: refer to the kvCacheBuffer.

    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr int ELTS_PER_VEC = /* 16bytes */ 16 / sizeof(T);
    constexpr int VECS_PER_HEAD = Dh / ELTS_PER_VEC;
    // One thread process one vector per step.
    constexpr int TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    // Manually tune the block size to make sure all threads have work to do.
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");

    // The head idx.
    int const head_idx = blockIdx.y;
    // The batch idx.
    int const batch_idx = blockIdx.z;

    // The decoder sequence length.
    // Spec decoding not supported for cross-attention at the moment so we can set 1 and batch_idx here
    int const decoder_seq_len = params.generation_phase ? 1 : params.seq_lens[batch_idx];
    // The decoder sequence offset.
    int const decoder_seq_offset = params.generation_phase ? batch_idx : params.cu_seq_lens[batch_idx];
    // The decoder cache sequence length (includes the current input).
    int const decoder_cache_seq_len = params.cache_seq_lens[batch_idx];
    // The encoder sequence length.
    int const encoder_seq_len = params.encoder_seq_lens[batch_idx];
    // The encoder sequence offset.
    // Not needed in Gen phase
    int const encoder_seq_offset = params.generation_phase ? -1 : params.cu_kv_seq_lens[batch_idx];
    // THe maximum sequence length of encoder and decoder.
    int const max_seq_len = max(decoder_seq_len, encoder_seq_len);

    // Only the first chunk needs to store encoder kv input to the kv cache.
    bool const store_encoder_kv_cache = params.cross_kv_input != nullptr && (decoder_seq_len == decoder_cache_seq_len);

    // Offsets and strides.
    int const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
    int const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
    int const hidden_idx = head_idx * params.size_per_head + head_dim_idx;
    int const kv_head_idx = head_idx / params.qheads_per_kv_head;
    int const hidden_idx_kv = kv_head_idx * params.size_per_head + head_dim_idx;
    int const src_v_offset = params.kv_hidden_size;

    // Cast float scale to dst data type.
    using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
    [[maybe_unused]] TScale scale_orig_quant;
    if constexpr (sizeof(TCache) == 1 || FP8_OUTPUT)
    {
        mmha::convert_from_float(
            &scale_orig_quant, params.qkv_scale_orig_quant ? params.qkv_scale_orig_quant[0] : 1.0f);
    }

    // For loop in the sequence length dimension.
    // There might be multiple blocks (blockIdx.x) that process the same sequence in order to fully utilize
    for (int token_idx = blockIdx.x * TOKENS_PER_BLOCK + (threadIdx.x / VECS_PER_HEAD); token_idx < max_seq_len;
         token_idx += (gridDim.x * TOKENS_PER_BLOCK))
    {
        // Decoder tokens (i.e. Q tokens).
        if (token_idx < decoder_seq_len)
        {
            // The global token idx in all sequences.
            int global_token_idx = token_idx + decoder_seq_offset;

            // The memory offset.
            auto const src_q_idx = static_cast<size_t>(global_token_idx) * params.hidden_size + hidden_idx;
            auto const dst_q_idx = static_cast<size_t>(global_token_idx) * params.q_hidden_size + hidden_idx;

            // Only load Q tokens from decoder qkv input.
            auto q = *reinterpret_cast<VecT const*>(params.qkv_input + src_q_idx);

            // Quantize the output to fp8.
            if constexpr (FP8_OUTPUT)
            {
                using OutputType = __nv_fp8_e4m3;
                OutputType* quantized_q_ptr = reinterpret_cast<OutputType*>(params.q_output) + dst_q_idx;
                mmha::store_8bits_vec(quantized_q_ptr, q, 0, scale_orig_quant);
            }
            else
            {
                // Store it to a separate q output.
                *reinterpret_cast<VecT*>(params.q_output + dst_q_idx) = q;
            }
        }

        if (!params.generation_phase)
        {
            // Encoder tokens (i.e. KV tokens).
            if (head_idx == (kv_head_idx * params.qheads_per_kv_head) && token_idx < encoder_seq_len
                && store_encoder_kv_cache && params.kv_cache_buffer.data != nullptr)
            {
                // The global token idx in all sequences.
                int global_token_idx = token_idx + encoder_seq_offset;

                // The memory offset.
                auto const src_k_idx
                    = static_cast<size_t>(global_token_idx) * params.kv_hidden_size * 2 + hidden_idx_kv;
                auto const src_v_idx
                    = static_cast<size_t>(global_token_idx) * params.kv_hidden_size * 2 + src_v_offset + hidden_idx_kv;

                // Only load K,V tokens from encoder qkv input.
                auto k = *reinterpret_cast<VecT const*>(&params.cross_kv_input[src_k_idx]);
                auto v = *reinterpret_cast<VecT const*>(&params.cross_kv_input[src_v_idx]);

                // The kv cache pointers.
                auto k_cache_block_ptr
                    = reinterpret_cast<TCache*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, token_idx));
                auto v_cache_block_ptr
                    = reinterpret_cast<TCache*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, token_idx));
                // The vector idx in the cache block.
                auto block_vec_idx
                    = params.kv_cache_buffer.getKVLocalIdx(token_idx, kv_head_idx, VECS_PER_HEAD, head_dim_vec_idx);

                // Store K and V to the cache.
                // INT8/FP8 kv cache.
                if constexpr (sizeof(TCache) == 1)
                {
                    // The element index inside the block.
                    auto block_elt_idx = block_vec_idx * ELTS_PER_VEC;
                    // Store 8bits kv cache.
                    mmha::store_8bits_vec(k_cache_block_ptr, k, block_elt_idx, scale_orig_quant);
                    mmha::store_8bits_vec(v_cache_block_ptr, v, block_elt_idx, scale_orig_quant);
                }
                else
                {
                    reinterpret_cast<VecT*>(k_cache_block_ptr)[block_vec_idx] = k;
                    reinterpret_cast<VecT*>(v_cache_block_ptr)[block_vec_idx] = v;
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE, int Dh, typename T, typename TCache, typename KVCacheBuffer>
void invokeUpdateKvCacheForCrossAttention(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    constexpr int VECS_PER_HEAD = (Dh * sizeof(T) / 16);
    constexpr int TOKENS_PER_CUDA_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    TLLM_CHECK_WITH_INFO(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");

    // The grid, block size for kernel launch.
    dim3 block(BLOCK_SIZE);

    // The maximum sequence length of encoder and decoder inputs.
    int const max_seq_len = std::max(params.max_input_seq_len, params.max_kv_seq_len);
    // Assume each SM can hold 2048 threads.
    int const num_blocks_per_sm = 2048 / BLOCK_SIZE;
    // Use more blocks for the token dimension if possible.
    int num_seq_blocks
        = int(divUp(params.multi_processor_count * num_blocks_per_sm, params.head_num * params.batch_size));
    // Make sure we don't launch too many blocks which have no work to do.
    num_seq_blocks = std::min(num_seq_blocks, int(divUp(max_seq_len, TOKENS_PER_CUDA_BLOCK)));
    // The final grid dimension.
    dim3 grid(num_seq_blocks, params.head_num, params.batch_size);

    // Launch the kernel.
    if (params.quantized_fp8_output)
    {
        updateKVCacheForCrossAttention<T, TCache, KVCacheBuffer, BLOCK_SIZE, Dh, true>
            <<<grid, block, 0, stream>>>(params);
    }
    else
    {
        updateKVCacheForCrossAttention<T, TCache, KVCacheBuffer, BLOCK_SIZE, Dh, false>
            <<<grid, block, 0, stream>>>(params);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, typename KVCacheBuffer>
void invokeApplyBiasRopeUpdateKVCacheDispatch(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    // Use specialized kernels for different heads (better balance of work).
    TLLM_CHECK_WITH_INFO(params.size_per_head % 8 == 0, "Head size needs to be multiple of 8!");
    TLLM_CHECK_WITH_INFO(params.rotary_embedding_dim % 8 == 0, "Rotary embedding dimension needs to be multiple of 8!");

// NVFP4 kv cache requires head size to be power of 2.
#ifdef ENABLE_FP4
    if (std::is_same_v<TCache, __nv_fp4_e2m1>)
    {
        TLLM_CHECK_WITH_INFO((params.size_per_head & (params.size_per_head - 1)) == 0,
            "Head size needs to be power of 2 for nvfp4 kv cache.");
    }
#endif

    // TODO: this should be extended to support quantized FP4 outputs as well.
    // For now, we will assume that the attention kernel reads directly from the KV cache
    // and FP16 inputs.
    TLLM_CHECK_WITH_INFO(
        !(params.quantized_fp8_output && !params.separate_q_kv_output && params.quantized_qkv_output == nullptr)
            && !(params.quantized_fp8_output && params.separate_q_kv_output && params.q_output == nullptr),
        "Separate quantized buffer is not provided!");
    bool const absolute_position_embedding
        = (params.position_embedding_type == PositionEmbeddingType::kLEARNED_ABSOLUTE);

    // Launch kernels for cross-attention.
    if (params.cross_attention)
    {
        TLLM_CHECK_WITH_INFO((absolute_position_embedding && params.remove_padding && params.qkv_bias == nullptr),
            "Assume cross attention has learned_absolute position embedding, remove_padding is enabled and no bias");
#ifdef ENABLE_FP4
        // TODO: update the kernel and remove this check.
        TLLM_CHECK_WITH_INFO(
            (!std::is_same_v<TCache, __nv_fp4_e2m1>), "FP4 KV cache update for cross attention not supported yet.");
#endif
        switch (params.size_per_head)
        {
        case 32: invokeUpdateKvCacheForCrossAttention<1024, 32, T, TCache, KVCacheBuffer>(params, stream); break;
        case 64: invokeUpdateKvCacheForCrossAttention<1024, 64, T, TCache, KVCacheBuffer>(params, stream); break;
        case 72: invokeUpdateKvCacheForCrossAttention<1008, 72, T, TCache, KVCacheBuffer>(params, stream); break;
        case 128: invokeUpdateKvCacheForCrossAttention<1024, 128, T, TCache, KVCacheBuffer>(params, stream); break;
        case 256: invokeUpdateKvCacheForCrossAttention<1024, 256, T, TCache, KVCacheBuffer>(params, stream); break;
        default: TLLM_CHECK_WITH_INFO(false, "Not supported."); break;
        }
        return;
    }

    // Long-sequence-length that exceeds the max_position_size needs to compute the cos/sin on-the-fly.
    bool const long_seq_rotary_support = params.rotary_scale_type == RotaryScalingType::kDYNAMIC
        || params.max_kv_seq_len > params.rotary_embedding_max_positions;
    bool const has_rotary_cos_sin_cache = params.rotary_coef_cache_buffer != nullptr;
    bool const has_sink_tokens = params.sink_token_len > 0;
    bool const use_v1_for_mrope
        = params.position_embedding_type == PositionEmbeddingType::kROPE_M && params.mrope_rotary_cos_sin == nullptr;
    // V2 implementation requires multiple of paired 16 bytes for gpt-neox rotation.
    bool const support_rotary_for_v2 = (params.position_embedding_type != PositionEmbeddingType::kROPE_GPT_NEOX
                                           && params.position_embedding_type != PositionEmbeddingType::kLONG_ROPE)
        || params.rotary_embedding_dim % 16 == 0;

    // Use v2 kernel for absolute_position_embedding.
    if (!absolute_position_embedding
        && (long_seq_rotary_support || !has_rotary_cos_sin_cache || has_sink_tokens || !support_rotary_for_v2
            || use_v1_for_mrope))
    {
        kernelV1Dispatch<T, TCache, KVCacheBuffer>(params, stream);
        return;
    }

    // Attempt to adopt optimized memory-bound kernel first, otherwise fall back to the v1 kernel.
    switch (params.size_per_head)
    {
    case 32: kernelV2DispatchHeadSize<256, 32, T, TCache, KVCacheBuffer>(params, stream); break;
    case 48: kernelV2DispatchHeadSize<192, 48, T, TCache, KVCacheBuffer>(params, stream); break;
    case 64: kernelV2DispatchHeadSize<256, 64, T, TCache, KVCacheBuffer>(params, stream); break;
    case 72: kernelV2DispatchHeadSize<288, 72, T, TCache, KVCacheBuffer>(params, stream); break;
    case 80: kernelV2DispatchHeadSize<160, 80, T, TCache, KVCacheBuffer>(params, stream); break;
    case 96: kernelV2DispatchHeadSize<192, 96, T, TCache, KVCacheBuffer>(params, stream); break;
    case 104: kernelV2DispatchHeadSize<416, 104, T, TCache, KVCacheBuffer>(params, stream); break;
    case 112: kernelV2DispatchHeadSize<224, 112, T, TCache, KVCacheBuffer>(params, stream); break;
    case 128: kernelV2DispatchHeadSize<256, 128, T, TCache, KVCacheBuffer>(params, stream); break;
    case 144: kernelV2DispatchHeadSize<288, 144, T, TCache, KVCacheBuffer>(params, stream); break;
    case 160: kernelV2DispatchHeadSize<160, 160, T, TCache, KVCacheBuffer>(params, stream); break;
    case 192: kernelV2DispatchHeadSize<192, 192, T, TCache, KVCacheBuffer>(params, stream); break;
    case 224: kernelV2DispatchHeadSize<224, 224, T, TCache, KVCacheBuffer>(params, stream); break;
    case 256: kernelV2DispatchHeadSize<256, 256, T, TCache, KVCacheBuffer>(params, stream); break;
    case 512: kernelV2DispatchHeadSize<512, 512, T, TCache, KVCacheBuffer>(params, stream); break;
    case 576: kernelV2DispatchHeadSize<576, 576, T, TCache, KVCacheBuffer>(params, stream); break;
    default:
        // Fall back to v1 kernel.
        // GPTJ Rotary embedding needs at least two elements per thread.
        kernelV1Dispatch<T, TCache, KVCacheBuffer>(params, stream);
        break;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, typename KVCacheBuffer>
__global__ __launch_bounds__(1024) void updateCyclicKvCacheAfterFmha(QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    // The batch idx.
    int batch_idx = blockIdx.z;
    // The kv head idx.
    int kv_head_idx = blockIdx.y;

    // The number of 16B vectors per head size in the kv cache.
    int num_vecs_per_head = (params.size_per_head * sizeof(TCache)) / 16;

    // The current sequence length.
    int seq_length = params.seq_lens[batch_idx];
    // The cache sequence length.
    int cache_seq_length = params.cache_seq_lens[batch_idx];
    // The past cache sequence length.
    int past_cache_seq_length = cache_seq_length - seq_length;
    // Do we need to move the kv from the temporary kv cache to the cyclic kv cache (i.e. overwriting) ?
    bool const write_to_cyclic_kv_cache = cache_seq_length > params.cyclic_kv_cache_len;
    // The number of tokens in the temporary kv cache.
    int num_tmp_kv_tokens = past_cache_seq_length < params.cyclic_kv_cache_len
        ? (cache_seq_length - params.cyclic_kv_cache_len)
        : seq_length;
    // The kv sequence offset for the temporary kv tokens.
    int tmp_kv_seq_offset = (cache_seq_length - num_tmp_kv_tokens);
    // The first tmp kv cache idx that needs to be stored to the kv cache.
    // Only the last params.cyclic_kv_cache_len tokens needs to be stored.
    int tmp_kv_token_start_idx = max(num_tmp_kv_tokens - params.cyclic_kv_cache_len, 0);

    // Early step if this sequence doesn't need to update cyclic kv cache.
    if (!write_to_cyclic_kv_cache)
    {
        return;
    }

    // The kv cache buffer has the shape of [batch_size, 2, max_num_blocks_per_seq, [num_kv_heads, tokens_per_block,
    // head_size]] All threads in block.y and grid.x will iterate over all the tokens.
    int thread_token_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int num_tokens_per_loop = gridDim.x * blockDim.y;
    // Iterate over new tokens' kv blocks.
    for (int token_idx = tmp_kv_token_start_idx + thread_token_idx; token_idx < num_tmp_kv_tokens;
         token_idx += num_tokens_per_loop)
    {

        // The token idx in the kv cache for loading and storing.
        int load_token_idx_in_kv_cache = params.cyclic_kv_cache_len + token_idx;
        int store_token_idx_in_kv_cache = (token_idx + tmp_kv_seq_offset) % params.cyclic_kv_cache_len;
        // The block pointer.
        auto load_k_block_ptr
            = reinterpret_cast<uint4*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, load_token_idx_in_kv_cache));
        auto load_v_block_ptr
            = reinterpret_cast<uint4*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, load_token_idx_in_kv_cache));
        auto store_k_block_ptr
            = reinterpret_cast<uint4*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, store_token_idx_in_kv_cache));
        auto store_v_block_ptr
            = reinterpret_cast<uint4*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, store_token_idx_in_kv_cache));

        // Iterate over new tokens' kv hidden states in one cache block.
        int head_vec_idx = threadIdx.x;
        if (head_vec_idx < num_vecs_per_head)
        {
            // The vector index inside the block.
            auto load_vec_idx = params.kv_cache_buffer.getKVLocalIdx(
                load_token_idx_in_kv_cache, kv_head_idx, num_vecs_per_head, head_vec_idx);
            auto store_vec_idx = params.kv_cache_buffer.getKVLocalIdx(
                store_token_idx_in_kv_cache, kv_head_idx, num_vecs_per_head, head_vec_idx);
            // Load from the temporary cache and write it to the cyclic cache.
            store_k_block_ptr[store_vec_idx] = load_k_block_ptr[load_vec_idx];
            store_v_block_ptr[store_vec_idx] = load_v_block_ptr[load_vec_idx];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, typename KVCacheBuffer>
void invokeUpdateCyclicKvCacheAfterFmha(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    //
    // This function processes at most params.cyclic_kv_cache_len kv cache as others will be overwritten anyway.
    //

    // Launch kernels for updating cyclic kv cache. It is only needed when sliding window attention and chunked context
    // (i.e. paged kv context fmha) are used together.
    dim3 block(32, 32);
    dim3 grid(std::min(64, int(divUp(params.cyclic_kv_cache_len, block.y))), params.kv_head_num, params.batch_size);
    // separate_q_kv_output = true means that paged kv context fmha might be used.
    if (params.max_kv_seq_len > params.cyclic_kv_cache_len && params.separate_q_kv_output)
    {
        // Assume the bytes of head size is multiple of 16.
        TLLM_CHECK_WITH_INFO(
            (params.size_per_head * sizeof(TCache)) % 16 == 0 && (params.size_per_head * sizeof(TCache)) / 16 <= 32,
            "Head size is not supported.");
        updateCyclicKvCacheAfterFmha<T, TCache, KVCacheBuffer><<<grid, block, 0, stream>>>(params);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16

// Pipelined bf16 compaction kernels: double-buffered cp.async copies that
// compact paged KV pools in place through the V2 block-offset table.

namespace compact_detail
{
// Vendored cp.async wrappers (xqa's ldgsts.cuh cannot be included here).
template <uint32_t size>
__device__ __forceinline__ void copyAsync(void* dst, void const* src, uint32_t srcSize = size)
{
    static_assert(size == 16, "only the 16B cp.async variant is vendored");
    // srcSize == 0 zero-fills shared memory instead of reading global.
    if (srcSize == 0)
    {
        src = nullptr;
    }
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"l"(__cvta_generic_to_shared(dst)), "l"(src), "r"(srcSize));
}

__device__ __forceinline__ void commitGroup()
{
    asm volatile("cp.async.commit_group;\n");
}

// Wait until at most InFlightGroups cp.async groups remain in flight.
template <uint32_t InFlightGroups>
__device__ __forceinline__ void waitGroup()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(InFlightGroups));
}
} // namespace compact_detail

// One pipeline stage moves a 32-token tile regardless of the page geometry.
constexpr int32_t kSparseKvCompactTokensPerTile = 32;

//! Launch parameters for the pipelined bf16 fast-path compaction kernels.
struct SparseKvCacheCompactBf16Params
{
    int64_t const* poolPointers;
    int32_t const* pageTable;
    int32_t const* sourceIndices;
    int32_t const* sourceOffsets;
    int32_t const* sourceLayerIndices;
    int32_t const* destinationBases;
    int64_t sourceLayerStride;
    int64_t sourceHeadStride;
    int64_t pageTableRequestStride;
    int32_t numLayers;
    int32_t batchSize;
    int32_t numKvHeads;
    size_t bytesPerKvHalf;
    size_t bytesPerPage;
};

//! Double-buffered cp.async pipeline: the next tile streams in while the
//! current tile drains. One CTA per (layer, KV head, request).
template <typename T, int32_t HeadDim, int32_t TokensPerBlock>
__global__ __launch_bounds__(HeadDim * sizeof(T) / sizeof(uint4)
    * kSparseKvCompactTokensPerTile) void sparseKvCacheCompactV2Bf16PipelineKernel(SparseKvCacheCompactBf16Params
        params)
{
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    static_assert(HeadDim == 64 || HeadDim == 128);
    // 128-token pages are the geometry the kernel was written for; 32-token
    // pages cover the supported production configuration (one tile == one page).
    static_assert(TokensPerBlock == 32 || TokensPerBlock == 128);
    constexpr int32_t kVectorsPerHead = HeadDim * sizeof(T) / sizeof(uint4);
    constexpr int32_t kTokensPerTile = kSparseKvCompactTokensPerTile;
    constexpr int32_t kVectorsPerTile = kTokensPerTile * kVectorsPerHead;
    // A buffer holds one K tile plus one V tile; two buffers ping-pong.
    constexpr int32_t kVectorsPerBuffer = 2 * kVectorsPerTile;

    int32_t const layerIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const kvHeadIdx = static_cast<int32_t>(blockIdx.y);
    int32_t const batchIdx = static_cast<int32_t>(blockIdx.z);
    int32_t const moveBegin = params.sourceOffsets[batchIdx];
    int32_t const moveEnd = params.sourceOffsets[batchIdx + 1];
    int32_t const moveCount = moveEnd - moveBegin;
    if (moveCount <= 0)
    {
        return;
    }

    // Without an explicit layer map, launch layer i reads source plane i.
    int32_t const sourceLayer = params.sourceLayerIndices == nullptr ? layerIdx : params.sourceLayerIndices[layerIdx];
    int64_t const sourceMoveBase = static_cast<int64_t>(sourceLayer) * params.sourceLayerStride
        + static_cast<int64_t>(kvHeadIdx) * params.sourceHeadStride + moveBegin;
    int32_t const destinationBase = params.destinationBases[batchIdx];
    auto* const pool = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(params.poolPointers[layerIdx]));
    // Block-offset entries decode to a page with >> 1.
    int32_t const* const pageTable = params.pageTable + static_cast<int64_t>(batchIdx) * params.pageTableRequestStride;

    extern __shared__ uint4 sharedVectors[];
    int32_t const sharedVector
        = static_cast<int32_t>(threadIdx.y) * kVectorsPerHead + static_cast<int32_t>(threadIdx.x);
    int32_t currentRequestMove = static_cast<int32_t>(threadIdx.y);
    bool currentValid = currentRequestMove < moveCount;
    int32_t currentSourceToken = currentValid ? params.sourceIndices[sourceMoveBase + currentRequestMove] : -1;
    uint4* currentSharedK = sharedVectors;
    uint4* currentSharedV = currentSharedK + kVectorsPerTile;

    // Prologue: explicitly wait for tile 0 and synchronize the CTA before any thread stores it.
    uint4 const* currentSourceKVector = nullptr;
    uint4 const* currentSourceVVector = nullptr;
    if (currentValid)
    {
        int32_t const sourcePage = pageTable[currentSourceToken / TokensPerBlock] >> 1;
        auto* const sourcePageBase = pool + static_cast<size_t>(sourcePage) * params.bytesPerPage;
        auto const* const sourceK = reinterpret_cast<uint4 const*>(sourcePageBase);
        auto const* const sourceV = reinterpret_cast<uint4 const*>(sourcePageBase + params.bytesPerKvHalf);
        int32_t const localVector = (kvHeadIdx * TokensPerBlock + currentSourceToken % TokensPerBlock) * kVectorsPerHead
            + static_cast<int32_t>(threadIdx.x);
        currentSourceKVector = &sourceK[localVector];
        currentSourceVVector = &sourceV[localVector];
    }
    uint32_t const currentSourceBytes = currentValid ? sizeof(uint4) : 0U;
    compact_detail::copyAsync<sizeof(uint4)>(&currentSharedK[sharedVector], currentSourceKVector, currentSourceBytes);
    compact_detail::copyAsync<sizeof(uint4)>(&currentSharedV[sharedVector], currentSourceVVector, currentSourceBytes);
    compact_detail::commitGroup();
    compact_detail::waitGroup<0>();
    __syncthreads();

    for (int32_t nextTileBegin = kTokensPerTile; nextTileBegin < moveCount; nextTileBegin += kTokensPerTile)
    {
        int32_t const nextRequestMove = nextTileBegin + static_cast<int32_t>(threadIdx.y);
        bool const nextValid = nextRequestMove < moveCount;
        int32_t const nextSourceToken = nextValid ? params.sourceIndices[sourceMoveBase + nextRequestMove] : -1;
        int32_t const nextBuffer = (nextTileBegin / kTokensPerTile) & 1;
        uint4* const nextSharedK = sharedVectors + nextBuffer * kVectorsPerBuffer;
        uint4* const nextSharedV = nextSharedK + kVectorsPerTile;

        uint4 const* nextSourceKVector = nullptr;
        uint4 const* nextSourceVVector = nullptr;
        if (nextValid)
        {
            int32_t const sourcePage = pageTable[nextSourceToken / TokensPerBlock] >> 1;
            auto* const sourcePageBase = pool + static_cast<size_t>(sourcePage) * params.bytesPerPage;
            auto const* const sourceK = reinterpret_cast<uint4 const*>(sourcePageBase);
            auto const* const sourceV = reinterpret_cast<uint4 const*>(sourcePageBase + params.bytesPerKvHalf);
            int32_t const localVector
                = (kvHeadIdx * TokensPerBlock + nextSourceToken % TokensPerBlock) * kVectorsPerHead
                + static_cast<int32_t>(threadIdx.x);
            nextSourceKVector = &sourceK[localVector];
            nextSourceVVector = &sourceV[localVector];
        }
        uint32_t const nextSourceBytes = nextValid ? sizeof(uint4) : 0U;
        compact_detail::copyAsync<sizeof(uint4)>(&nextSharedK[sharedVector], nextSourceKVector, nextSourceBytes);
        compact_detail::copyAsync<sizeof(uint4)>(&nextSharedV[sharedVector], nextSourceVVector, nextSourceBytes);
        compact_detail::commitGroup();

        // In-place safety: sources are strictly increasing with dst(i) <= src(i),
        // so current stores never alias future prefetch sources.
        int32_t const destinationToken = destinationBase + currentRequestMove;
        if (currentValid && currentSourceToken != destinationToken)
        {
            int32_t const destinationPage = pageTable[destinationToken / TokensPerBlock] >> 1;
            auto* const destinationPageBase = pool + static_cast<size_t>(destinationPage) * params.bytesPerPage;
            auto* const destinationK = reinterpret_cast<uint4*>(destinationPageBase);
            auto* const destinationV = reinterpret_cast<uint4*>(destinationPageBase + params.bytesPerKvHalf);
            int32_t const localVector
                = (kvHeadIdx * TokensPerBlock + destinationToken % TokensPerBlock) * kVectorsPerHead
                + static_cast<int32_t>(threadIdx.x);
            destinationK[localVector] = currentSharedK[sharedVector];
            destinationV[localVector] = currentSharedV[sharedVector];
        }

        // waitGroup<0> completes the next tile's copies before the buffer swap.
        compact_detail::waitGroup<0>();
        __syncthreads();
        currentRequestMove = nextRequestMove;
        currentValid = nextValid;
        currentSourceToken = nextSourceToken;
        currentSharedK = nextSharedK;
        currentSharedV = nextSharedV;
    }

    // Epilogue: the final tile already completed its async wait and CTA barrier.
    int32_t const destinationToken = destinationBase + currentRequestMove;
    if (currentValid && currentSourceToken != destinationToken)
    {
        int32_t const destinationPage = pageTable[destinationToken / TokensPerBlock] >> 1;
        auto* const destinationPageBase = pool + static_cast<size_t>(destinationPage) * params.bytesPerPage;
        auto* const destinationK = reinterpret_cast<uint4*>(destinationPageBase);
        auto* const destinationV = reinterpret_cast<uint4*>(destinationPageBase + params.bytesPerKvHalf);
        int32_t const localVector = (kvHeadIdx * TokensPerBlock + destinationToken % TokensPerBlock) * kVectorsPerHead
            + static_cast<int32_t>(threadIdx.x);
        destinationK[localVector] = currentSharedK[sharedVector];
        destinationV[localVector] = currentSharedV[sharedVector];
    }
}

template <typename T, int32_t HeadDim, int32_t TokensPerBlock>
void launchSparseKvCacheCompactV2Bf16Pipeline(SparseKvCacheCompactBf16Params const& params, cudaStream_t stream)
{
    constexpr int32_t kVectorsPerHead = HeadDim * sizeof(T) / sizeof(uint4);
    dim3 const block(kVectorsPerHead, kSparseKvCompactTokensPerTile);
    dim3 const grid(params.numLayers, params.numKvHeads, params.batchSize);
    // Two ping-pong buffers of (K + V) tiles; both geometries fit the 48 KiB
    // per-CTA dynamic shared memory default.
    size_t const sharedBytes = 4 * kSparseKvCompactTokensPerTile * kVectorsPerHead * sizeof(uint4);
    sparseKvCacheCompactV2Bf16PipelineKernel<T, HeadDim, TokensPerBlock><<<grid, block, sharedBytes, stream>>>(params);
}

#endif // ENABLE_BF16

template <typename T, typename TCache, int BLOCK_SIZE, int Dh, typename KVCacheBuffer>
__global__ __launch_bounds__(BLOCK_SIZE) void updateSparseKvCacheAfterFmha(
    QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    // The number of 16B vectors per head size in the kv cache.
    constexpr int VECS_PER_HEAD = Dh * sizeof(TCache) / 16;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");

    int const batch_idx = blockIdx.z;
    int const kv_head_idx = blockIdx.y;

    int const total_num_sparse_kv_tokens = params.sparse_kv_offsets[params.batch_size];

    int const sparse_start_idx = params.sparse_kv_offsets[batch_idx];
    int const sparse_end_idx = params.sparse_kv_offsets[batch_idx + 1];
    int const num_sparse_tokens = sparse_end_idx - sparse_start_idx;

    int const tokens_per_block = blockDim.y;
    int const vecs_per_block = blockDim.x;

    extern __shared__ uint4 smem[];
    uint4* k_smem = smem;
    uint4* v_smem = k_smem + tokens_per_block * VECS_PER_HEAD;

    for (int token_block_offset = 0; token_block_offset < num_sparse_tokens; token_block_offset += tokens_per_block)
    {
        int const sparse_token_offset = token_block_offset + threadIdx.y;

        if (sparse_token_offset < num_sparse_tokens)
        {
            int const global_sparse_idx = sparse_start_idx + sparse_token_offset;
            int const sparse_idx_offset = kv_head_idx * total_num_sparse_kv_tokens + global_sparse_idx;

            int const src_token_idx = params.sparse_kv_indices[sparse_idx_offset];

            void* src_k_ptr = params.kv_cache_buffer.getKBlockPtr(batch_idx, src_token_idx);
            void* src_v_ptr = params.kv_cache_buffer.getVBlockPtr(batch_idx, src_token_idx);
            auto const src_k_block_ptr = reinterpret_cast<uint4*>(src_k_ptr);
            auto const src_v_block_ptr = reinterpret_cast<uint4*>(src_v_ptr);

            for (int head_vec_idx = threadIdx.x; head_vec_idx < VECS_PER_HEAD; head_vec_idx += vecs_per_block)
            {
                auto const src_k_vec_idx
                    = params.kv_cache_buffer.getKVLocalIdx(src_token_idx, kv_head_idx, VECS_PER_HEAD, head_vec_idx);
                auto const src_v_vec_idx
                    = params.kv_cache_buffer.getKVLocalIdx(src_token_idx, kv_head_idx, VECS_PER_HEAD, head_vec_idx);

                k_smem[threadIdx.y * VECS_PER_HEAD + head_vec_idx] = src_k_block_ptr[src_k_vec_idx];
                v_smem[threadIdx.y * VECS_PER_HEAD + head_vec_idx] = src_v_block_ptr[src_v_vec_idx];
            }
        }
        __syncthreads();

        if (sparse_token_offset < num_sparse_tokens)
        {
            int const global_sparse_idx = sparse_start_idx + sparse_token_offset;
            int const sparse_idx_offset = kv_head_idx * total_num_sparse_kv_tokens + global_sparse_idx;

            int const src_token_idx = params.sparse_kv_indices[sparse_idx_offset];
            int const dst_token_idx = sparse_token_offset;

            if (src_token_idx != dst_token_idx)
            {
                void* dst_k_ptr = params.kv_cache_buffer.getKBlockPtr(batch_idx, dst_token_idx);
                void* dst_v_ptr = params.kv_cache_buffer.getVBlockPtr(batch_idx, dst_token_idx);
                auto const dst_k_block_ptr = reinterpret_cast<uint4*>(dst_k_ptr);
                auto const dst_v_block_ptr = reinterpret_cast<uint4*>(dst_v_ptr);

                for (int head_vec_idx = threadIdx.x; head_vec_idx < VECS_PER_HEAD; head_vec_idx += vecs_per_block)
                {
                    auto const dst_k_vec_idx
                        = params.kv_cache_buffer.getKVLocalIdx(dst_token_idx, kv_head_idx, VECS_PER_HEAD, head_vec_idx);
                    auto const dst_v_vec_idx
                        = params.kv_cache_buffer.getKVLocalIdx(dst_token_idx, kv_head_idx, VECS_PER_HEAD, head_vec_idx);
                    dst_k_block_ptr[dst_k_vec_idx] = k_smem[threadIdx.y * VECS_PER_HEAD + head_vec_idx];
                    dst_v_block_ptr[dst_v_vec_idx] = v_smem[threadIdx.y * VECS_PER_HEAD + head_vec_idx];
                }
            }
        }
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Dh, typename T, typename TCache, typename KVCacheBuffer>
void kernelSparseDispatchHeadSize(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    constexpr int VECS_PER_HEAD = Dh * sizeof(TCache) / 16;
    constexpr int BLOCK_SIZE = 1024;
    dim3 block(32, 32); // x: head vectors, y: tokens

    int smem_size = 2 * block.y * VECS_PER_HEAD * sizeof(uint4);

    // grid.x is always 1 to avoid data races
    dim3 grid(1, params.kv_head_num, params.batch_size);

    updateSparseKvCacheAfterFmha<T, TCache, BLOCK_SIZE, Dh, KVCacheBuffer><<<grid, block, smem_size, stream>>>(params);
}

template <typename T, typename TCache, typename KVCacheBuffer>
void invokeUpdateSparseKvCacheAfterFmha(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    if (params.sparse_kv_indices == nullptr)
    {
        return;
    }

    switch (params.size_per_head)
    {
    case 16: kernelSparseDispatchHeadSize<16, T, TCache, KVCacheBuffer>(params, stream); break;
    case 32: kernelSparseDispatchHeadSize<32, T, TCache, KVCacheBuffer>(params, stream); break;
    case 64: kernelSparseDispatchHeadSize<64, T, TCache, KVCacheBuffer>(params, stream); break;
    case 128: kernelSparseDispatchHeadSize<128, T, TCache, KVCacheBuffer>(params, stream); break;
    case 256: kernelSparseDispatchHeadSize<256, T, TCache, KVCacheBuffer>(params, stream); break;
    default:
        TLLM_CHECK_WITH_INFO(
            false, "updateSparseKvCacheAfterFmha kernel doesn't support head size = %d", params.size_per_head);
        break;
    }
}

template <typename T>
void invokeSparseKvCacheCompactLayers(int64_t const* poolPointers, int32_t const* pageTable, int32_t numLayers,
    int64_t pageTableRequestStride, int32_t const* sparseKvIndices, int32_t const* sourceLayerIndices,
    int64_t sourceLayerStride, int64_t sourceHeadStride, int32_t const* sparseKvOffsets,
    int32_t const* destinationBases, int32_t batchSize, int32_t numKvHeads, int32_t tokensPerBlock, int32_t headDim,
    cudaStream_t stream)
{
#ifdef ENABLE_BF16
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        if ((headDim == 64 || headDim == 128) && (tokensPerBlock == 32 || tokensPerBlock == 128))
        {
            SparseKvCacheCompactBf16Params fastParams{};
            fastParams.poolPointers = poolPointers;
            fastParams.pageTable = pageTable;
            fastParams.sourceIndices = sparseKvIndices;
            fastParams.sourceOffsets = sparseKvOffsets;
            fastParams.sourceLayerIndices = sourceLayerIndices;
            fastParams.destinationBases = destinationBases;
            fastParams.sourceLayerStride = sourceLayerStride;
            fastParams.sourceHeadStride = sourceHeadStride;
            fastParams.pageTableRequestStride = pageTableRequestStride;
            fastParams.numLayers = numLayers;
            fastParams.batchSize = batchSize;
            fastParams.numKvHeads = numKvHeads;
            fastParams.bytesPerKvHalf = static_cast<size_t>(numKvHeads) * tokensPerBlock * headDim * sizeof(T);
            fastParams.bytesPerPage = 2 * fastParams.bytesPerKvHalf;
            if (headDim == 64 && tokensPerBlock == 32)
            {
                launchSparseKvCacheCompactV2Bf16Pipeline<T, 64, 32>(fastParams, stream);
            }
            else if (headDim == 64 && tokensPerBlock == 128)
            {
                launchSparseKvCacheCompactV2Bf16Pipeline<T, 64, 128>(fastParams, stream);
            }
            else if (headDim == 128 && tokensPerBlock == 32)
            {
                launchSparseKvCacheCompactV2Bf16Pipeline<T, 128, 32>(fastParams, stream);
            }
            else
            {
                launchSparseKvCacheCompactV2Bf16Pipeline<T, 128, 128>(fastParams, stream);
            }
            return;
        }
    }
#endif // ENABLE_BF16

    TLLM_CHECK_WITH_INFO(false,
        "Sparse KV compaction ships only the pipelined bf16 kernels (head size 64/128, page size 32/128 "
        "tokens); got element size %zu, head size %d, %d tokens per page",
        sizeof(T), headDim, tokensPerBlock);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_ATTENTION_INPUT_PROCESSING(T, TCache, KVCacheBuffer)                                               \
    template void invokeApplyBiasRopeUpdateKVCacheDispatch<T, TCache, KVCacheBuffer>(                                  \
        QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);

#define INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(T, TCache, KVCacheBuffer)                                        \
    template void invokeApplyBiasRopeUpdateKVCacheDispatch<T, TCache, KVCacheBuffer>(                                  \
        QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);                                         \
    template void invokeUpdateCyclicKvCacheAfterFmha<T, TCache, KVCacheBuffer>(                                        \
        QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);                                         \
    template void invokeUpdateSparseKvCacheAfterFmha<T, TCache, KVCacheBuffer>(                                        \
        QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);                                         \
    ////////////////////////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_SPARSE_KV_CACHE_COMPACT_LAYERS(T)                                                                  \
    template void invokeSparseKvCacheCompactLayers<T>(int64_t const*, int32_t const*, int32_t, int64_t,                \
        int32_t const*, int32_t const*, int64_t, int64_t, int32_t const*, int32_t const*, int32_t, int32_t, int32_t,   \
        int32_t, cudaStream_t);

} // namespace kernels

TRTLLM_NAMESPACE_END
