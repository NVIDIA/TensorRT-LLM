/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

// A stateful callback functor that maintains the running sum between consecutive scans.
struct BlockPrefixCallbackOp
{
    // Running prefix
    int mRunningTotal;

    // Constructor
    __device__ BlockPrefixCallbackOp(int runningTotal)
        : mRunningTotal(runningTotal)
    {
    }

    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int blockAggregate)
    {
        int oldPrefix = mRunningTotal;
        mRunningTotal += blockAggregate;
        return oldPrefix;
    }
};

template <typename T>
struct VecType
{
    using Type = T;
    using GPTJEltType = T;
};

template <>
struct VecType<float>
{
    using Type = float4;
    using GPTJEltType = float2;
};

template <>
struct VecType<half>
{
    using Type = uint4;
    using GPTJEltType = uint32_t;
};

template <>
struct VecType<__nv_bfloat16>
{
    using Type = mmha::bf16_8_t;
    using GPTJEltType = __nv_bfloat162;
};

struct __align__(16) fp8_16_t
{
    __nv_fp8x4_e4m3 x;
    __nv_fp8x4_e4m3 y;
    __nv_fp8x4_e4m3 z;
    __nv_fp8x4_e4m3 w;
};

template <>
struct VecType<__nv_fp8_e4m3>
{
    using Type = fp8_16_t;
    using GPTJEltType = __nv_fp8x2_e4m3;
};

template <typename T>
struct loadPagedKVKernelTraits
{
    static constexpr int kLoraSize = 512;
    static constexpr int kRopeSize = 64;
    static constexpr int kHeadSize = kLoraSize + kRopeSize;
    using VecT = typename VecType<T>::Type;
    static constexpr int kBytesPerElem = sizeof(T);
    static constexpr int kBytesPerLoad = 16;
    static constexpr int kElemPerLoad = kBytesPerLoad / kBytesPerElem;
    static_assert((kHeadSize * kBytesPerElem) % kBytesPerLoad == 0,
        "kHeadSize * kBytesPerElem must be multiple of kBytesPerLoad (16Bytes)");
    static constexpr int kVecPerHead = (kHeadSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kThreadPerHead = kVecPerHead; // for each head, we use kThreadPerHead threads to fetch all the
                                                       // kv cache data, each thread read kv cache only once.
    static constexpr int kTokenPerBlock
        = std::is_same_v<T, float> ? 4 : 8; // for each block, we fetch 4 tokens for fp32, 8 tokens for other types.
    static constexpr int kBlockSize = kThreadPerHead * kTokenPerBlock;
    static constexpr int kKVThreadPerHead = (kLoraSize * kBytesPerElem) / kBytesPerLoad;
};

template <typename SrcType, int NUM>
inline __device__ void quantCopy(
    __nv_fp8_e4m3* dst_global_ptr, SrcType const* src_fragment_ptr, float const scale_val = 1.f)
{
    using DstVecType = typename std::conditional<sizeof(SrcType) == 2, float2, float>::type;
    using SrcType2 =
        typename std::conditional<sizeof(SrcType) == 2, typename TypeConverter<SrcType>::Type, float2>::type;
    static constexpr int COPY_SIZE = sizeof(DstVecType);
    static constexpr int TOTAL_COPY_SIZE = NUM * sizeof(__nv_fp8_e4m3);
    static constexpr int LOOP_NUM = TOTAL_COPY_SIZE / COPY_SIZE;
    static_assert(TOTAL_COPY_SIZE % COPY_SIZE == 0);
    static constexpr int CVT_NUM = COPY_SIZE / sizeof(__nv_fp8_e4m3) / 2;
    static_assert(COPY_SIZE % (sizeof(__nv_fp8_e4m3) * 2) == 0);
    DstVecType fragment;
    int offset = 0;
#pragma unroll
    for (int i = 0; i < LOOP_NUM; ++i)
    {
#pragma unroll
        for (int j = 0; j < CVT_NUM; ++j)
        {
            float2 val2 = cuda_cast<float2>(reinterpret_cast<SrcType2 const*>(src_fragment_ptr)[j + offset]);
            val2.x *= scale_val;
            val2.y *= scale_val;
            reinterpret_cast<__nv_fp8x2_e4m3*>(&fragment)[j] = __nv_fp8x2_e4m3(val2);
        }
        reinterpret_cast<DstVecType*>(dst_global_ptr)[i] = fragment;
        offset += CVT_NUM;
    }
}

template <typename DstType, int NUM>
inline __device__ void dequantCopy(
    DstType* dst_global_ptr, __nv_fp8_e4m3 const* src_fragment_ptr, float const scale_val = 1.f)
{
    using DstVecType = typename VecType<DstType>::Type;
    using DstType2 =
        typename std::conditional<sizeof(DstType) == 2, typename TypeConverter<DstType>::Type, float2>::type;
    static constexpr int COPY_SIZE = sizeof(DstVecType);
    static constexpr int TOTAL_COPY_SIZE = NUM * sizeof(DstType);
    static constexpr int LOOP_NUM = TOTAL_COPY_SIZE / COPY_SIZE;
    static_assert(TOTAL_COPY_SIZE % COPY_SIZE == 0);
    static constexpr int CVT_NUM = COPY_SIZE / sizeof(DstType) / 2;
    static_assert(COPY_SIZE % (sizeof(DstType) * 2) == 0);
    DstVecType fragment;
    int offset = 0;
#pragma unroll
    for (int i = 0; i < LOOP_NUM; ++i)
    {
#pragma unroll
        for (int j = 0; j < CVT_NUM; ++j)
        {
            float2 val2 = cuda_cast<float2>(reinterpret_cast<__nv_fp8x2_e4m3 const*>(src_fragment_ptr)[j + offset]);
            val2.x *= scale_val;
            val2.y *= scale_val;
            reinterpret_cast<DstType2*>(&fragment)[j] = cuda_cast<DstType2>(val2);
        }
        reinterpret_cast<DstVecType*>(dst_global_ptr)[i] = fragment;
        offset += CVT_NUM;
    }
}

template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void applyMLARopeAndAssignQKVKernelOptContext(T* q_ptr, T* k_ptr, T const* fuse_buf, KVCacheBuffer kv_cache,
    float2 const* cos_sin_cache, size_t head_num, int head_size, int c_k, int* cu_q_seqlens,
    int32_t const* kv_cache_lengths, uint32_t max_input_seq_len, KvCacheDataType cache_type,
    float const* quant_scale_kv)
{

    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VECS_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const batch_idx = blockIdx.y;
    size_t const head_idx = blockIdx.z;

    if (head_idx < head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {

            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];
            int token_idx_in_kv_cache = local_token_idx;
            bool const valid_token = token_idx_in_kv_cache < cache_seq_len;
            // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
            token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
            local_token_idx = std::min(local_token_idx, cache_seq_len - 1);
            int const global_token_idx = local_token_idx + global_token_offset;

            auto const position_id = local_token_idx;
            float2 const* rotary_coef_cache_buffer
                = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

            VecT q, k;
            auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM) + c_k;
            auto const src_q_global_offset = static_cast<size_t>(global_token_idx) * head_num * (head_size + ROPE_DIM)
                + (head_size + ROPE_DIM) * head_idx + head_size;

            q = *reinterpret_cast<VecT const*>(&q_ptr[src_q_global_offset + head_dim_idx]);
            k = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);

            // Pack two elements into one for gptj rotary embedding.
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
            {
                GPTJEltT& q_ = reinterpret_cast<GPTJEltT*>(&q)[elt_id];
                GPTJEltT& k_ = reinterpret_cast<GPTJEltT*>(&k)[elt_id];

                float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache);
            }
            // do sync
            __syncwarp();
            if (valid_token)
            {
                if (head_idx == 0)
                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(
                        token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                    if (cache_type == KvCacheDataType::FP8)
                    {

                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                            reinterpret_cast<T const*>(&k), quant_scale_kv_val);
                    }
                    else
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = k;
                }
                auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (head_size + ROPE_DIM)
                    + head_idx * (head_size + ROPE_DIM) + head_size + head_dim_idx;
                auto const dst_k_idx = static_cast<size_t>(global_token_idx) * head_num * (head_size + ROPE_DIM)
                    + head_idx * (head_size + ROPE_DIM) + head_size + head_dim_idx;
                reinterpret_cast<VecT*>(q_ptr)[dst_q_idx / ELTS_PER_VEC] = q;
                reinterpret_cast<VecT*>(k_ptr)[dst_k_idx / ELTS_PER_VEC] = k;
            }
        }
    }
    else
    {
        int block_dim = gridDim.z - head_num;
        int block_id = head_idx - head_num;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {

            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];
            int token_idx_in_kv_cache = local_token_idx;
            bool const valid_token = token_idx_in_kv_cache < cache_seq_len;
            // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
            token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
            local_token_idx = std::min(local_token_idx, cache_seq_len - 1);
            int const global_token_idx = local_token_idx + global_token_offset;

            if (valid_token)
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                if (cache_type == KvCacheDataType::FP8)
                {

                    quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                        fuse_buf + src_k_global_offset + head_dim_idx, quant_scale_kv_val);
                }
                else
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                        = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
            }
        }
    }
}

template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void applyMLARopeAndAssignQKVKernelGeneration(T* qkv_output, T* q_pe, T const* fuse_buf, void* quant_q,
    KVCacheBuffer kv_cache, float2 const* cos_sin_cache, size_t head_num, int c_k, int total_s_len, int seq_len,
    int* seqQOffset, uint32_t* fmha_tile_counter, int32_t const* kv_cache_lengths, int* seqKVOffsets, int q_pe_ld,
    int q_pe_stride, KvCacheDataType cache_type, float* bmm1_scale, float* bmm2_scale, float const* quant_scale_o,
    float const* quant_scale_q, float const* quant_scale_kv, float const* dequant_scale_q,
    float const* dequant_scale_kv, float host_bmm1_scale, int32_t const* helix_position_offsets)
{

    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VEC_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const head_idx = blockIdx.y;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
        fmha_tile_counter[0] = 0;
        seqQOffset[0] = 0;

        // Calculate bmm scale for FP8 MLA
        if (cache_type == KvCacheDataType::FP8)
        {
            float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
            float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
            float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
            if (bmm1_scale)
            {
                // The scale prepared for log2 optimization.
                constexpr float kLog2e = 1.4426950408889634074f;
                // The scale after fmha bmm1.
                float bmm1_scale_val = dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
                bmm1_scale[0] = bmm1_scale_val;
                bmm1_scale[1] = bmm1_scale_val * kLog2e;
            }
            if (bmm2_scale)
            {
                // The scale after fmha bmm2.
                bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
            }
        }
    }

    if (head_idx <= head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        int const seq_len_loop_end = size_t((total_s_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float const quant_scale_q_val = quant_scale_q ? quant_scale_q[0] : 1.0f;
        float const quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.0f;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx / seq_len;
            auto local_token_idx = global_token_idx % seq_len;
            bool const valid_token = global_token_idx < total_s_len;
            VecT data;

            if (valid_token)
            {

                auto const position_id
                    = (helix_position_offsets != nullptr ? helix_position_offsets[global_token_idx]
                                                         : kv_cache_lengths[batch_idx] - seq_len + local_token_idx);
                float2 const* rotary_coef_cache_buffer
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

                if (head_idx == head_num)
                {
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM) + c_k;

                    data = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
                }
                else
                {
                    auto const src_q_global_offset
                        = static_cast<size_t>(global_token_idx) * q_pe_stride + q_pe_ld * head_idx;

                    data = *reinterpret_cast<VecT const*>(&q_pe[src_q_global_offset + head_dim_idx]);
                }

                // Pack two elements into one for gptj rotary embedding.
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
                {
                    GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
                }
            }

            __syncwarp();

            if (valid_token)
            {
                if (head_idx == head_num)
                {
                    auto const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;

                    {
                        auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                        auto inBlockIdx = kv_cache.getKVLocalIdx(
                            token_kv_idx, 0, TOTAL_VEC_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                        if (cache_type == KvCacheDataType::FP8)
                        {

                            quantCopy<T, ELTS_PER_VEC>(
                                reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                                reinterpret_cast<T const*>(&data), quant_scale_kv_val);
                        }
                        else
                            reinterpret_cast<VecT*>(kDst)[inBlockIdx] = data;
                    }
                }
                else
                {
                    auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (c_k + ROPE_DIM)
                        + head_idx * (c_k + ROPE_DIM) + c_k + head_dim_idx;
                    if (cache_type == KvCacheDataType::FP8)
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(quant_q) + dst_q_idx,
                            reinterpret_cast<T const*>(&data), quant_scale_q_val);
                    }
                    else
                        reinterpret_cast<VecT*>(qkv_output)[dst_q_idx / ELTS_PER_VEC] = data;
                }
            }
        }
    }
    else if (head_idx <= head_num + 8)
    {
        int block_dim = gridDim.y - head_num - 1;
        int block_id = head_idx - head_num - 1;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.0f;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx / seq_len;
            auto local_token_idx = global_token_idx % seq_len;
            bool valid_token = global_token_idx < total_s_len;

            if (valid_token)
            {
                if (head_dim_vec_idx == 0)
                {
                    seqQOffset[batch_idx + 1] = head_num * seq_len * (batch_idx + 1);
                }

                auto const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;
                auto const src_kv_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(token_kv_idx, 0, TOTAL_VEC_PER_HEAD, head_dim_vec_idx);

                    if (cache_type == KvCacheDataType::FP8)
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                            fuse_buf + src_kv_global_offset + head_dim_idx, quant_scale_kv_val);
                    }
                    else
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                            = *reinterpret_cast<VecT const*>(&fuse_buf[src_kv_global_offset + head_dim_idx]);
                }
            }
        }
    }
    else
    {
        if (cache_type == KvCacheDataType::FP8)
        {
            int block_dim = gridDim.y - head_num - 1 - 8;
            int block_id = head_idx - head_num - 1 - 8;
            size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
            size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
            size_t const head_num_idx = (block_id % head_num) * (K_HEAD_SIZE + HEAD_SIZE);

            size_t const seq_len_loop_end
                = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
            float quant_scale_q_val = quant_scale_q ? quant_scale_q[0] : 1.0f;

            // Mainloop.
            for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD)
                     + (block_id / head_num) * gridDim.x * K_TOKENS_PER_BLOCK + blockIdx.x * K_TOKENS_PER_BLOCK;
                 global_token_idx < seq_len_loop_end;
                 global_token_idx += (block_dim / head_num) * gridDim.x * K_TOKENS_PER_BLOCK)
            {
                if (global_token_idx < total_s_len)
                {
                    size_t const load_idx
                        = global_token_idx * head_num * (K_HEAD_SIZE + HEAD_SIZE) + head_num_idx + head_dim_idx;
                    quantCopy<T, ELTS_PER_VEC>(
                        reinterpret_cast<__nv_fp8_e4m3*>(quant_q) + load_idx, qkv_output + load_idx, quant_scale_q_val);
                }
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempKVStorage;
    BlockPrefixCallbackOp prefixKVOp(0);

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        int const batchSizeBound = total_s_len / seq_len;
        for (int batchOffset = 0; batchOffset <= batchSizeBound; batchOffset += BLOCK_SIZE)
        {
            // The index of the batch.
            int batchIdx = batchOffset + threadIdx.x;
            int seqKVLength = 0;
            if (batchIdx < batchSizeBound)
            {
                seqKVLength = kv_cache_lengths[batchIdx];
            }
            int seqKVOffset;
            BlockScan(tempKVStorage).ExclusiveSum(seqKVLength, seqKVOffset, prefixKVOp);
            if (batchIdx <= batchSizeBound)
            {
                seqKVOffsets[batchIdx] = seqKVOffset;
            }
        }
    }
}

template <typename T, typename TCache>
__global__ void loadPagedKVCacheForMLAKernel(T* compressed_kv_ptr, T* k_pe_ptr,
    tensorrt_llm::kernels::KVBlockArray const kv_cache, int64_t const* cu_ctx_cached_kv_lens, int max_input_seq_len,
    float const* kv_scale_quant_orig_ptr)
{
    static_assert(std::is_same_v<T, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as T or __nv_fp8_e4m3");
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<TCache>;

    int const batch_idx = static_cast<int>(blockIdx.y);
    float const kv_scale_quant_orig = kv_scale_quant_orig_ptr ? kv_scale_quant_orig_ptr[0] : 1.0f;

    size_t const head_dim_vec_idx = (threadIdx.x % KT::kVecPerHead);
    size_t const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;
    bool const is_valid_kv = head_dim_vec_idx < KT::kKVThreadPerHead;

    size_t const seq_len_loop_end
        = (max_input_seq_len + KT::kTokenPerBlock - 1) / KT::kTokenPerBlock * KT::kTokenPerBlock;

    int64_t const global_token_offset = cu_ctx_cached_kv_lens[batch_idx];
    int64_t const cache_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];

    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kTokenPerBlock;
         local_token_idx < seq_len_loop_end; local_token_idx += KT::kTokenPerBlock * gridDim.x)
    {
        int token_idx_in_kv_cache = local_token_idx;
        bool const valid_token = token_idx_in_kv_cache < cache_kv_len;

        if (valid_token)
        {
            auto* kvSrc = reinterpret_cast<TCache*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
            // head_idx === 0
            auto kvBlockIdx
                = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, KT::kVecPerHead, static_cast<int>(head_dim_vec_idx));

            auto src_data = reinterpret_cast<typename KT::VecT*>(kvSrc)[kvBlockIdx];

            int const global_token_idx = local_token_idx + global_token_offset;

            if (is_valid_kv)
            {
                // compressed_kv {total_token, lora_size}
                int const dstIdx = global_token_idx * KT::kLoraSize + head_dim_idx;

                // copy back to compressed_kv
                if constexpr (std::is_same_v<TCache, T>)
                {
                    *reinterpret_cast<typename KT::VecT*>(compressed_kv_ptr + dstIdx) = src_data;
                }
                else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    dequantCopy<T, KT::kElemPerLoad>(compressed_kv_ptr + dstIdx,
                        reinterpret_cast<__nv_fp8_e4m3 const*>(&src_data), kv_scale_quant_orig);
                }
            }
            else
            {
                // k_pe {total_token, rope_size}
                int const dstIdx = global_token_idx * KT::kRopeSize + (head_dim_idx - KT::kLoraSize);

                // copy back to k_pe
                if constexpr (std::is_same_v<TCache, T>)
                {
                    *reinterpret_cast<typename KT::VecT*>(k_pe_ptr + dstIdx) = src_data;
                }
                else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    dequantCopy<T, KT::kElemPerLoad>(
                        k_pe_ptr + dstIdx, reinterpret_cast<__nv_fp8_e4m3 const*>(&src_data), kv_scale_quant_orig);
                }
            }
        }
    }
}

// q {total_uncached_tokens, h, d_nope + d_rope}
// latent_cache {total_uncached_tokens, d_k + d_rope}
template <typename T, typename TCache, int BLOCK_SIZE, int K_DIM, int ROPE_DIM>
__global__ void applyMLARopeAppendPagedKVAssignQKernel(KVBlockArray kv_cache, T* q_ptr, T* latent_cache_ptr,
    int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_uncached_seq_len,
    float2 const* cos_sin_cache, size_t head_num, int nope_size, float const* kv_scale_orig_quant_ptr)
{
    static_assert(std::is_same_v<T, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as T or __nv_fp8_e4m3");
    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VECS_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const batch_idx = blockIdx.y;
    size_t const head_idx = blockIdx.z;

    int64_t const global_token_offset = cu_seq_lens[batch_idx] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const cached_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const uncached_kv_len = cu_seq_lens[batch_idx + 1] - cu_seq_lens[batch_idx] - cached_kv_len;

    if (head_idx <= head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_uncached_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float quant_scale_kv_val = kv_scale_orig_quant_ptr ? kv_scale_orig_quant_ptr[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {

            int token_idx_in_kv_cache = local_token_idx + cached_kv_len;
            bool valid_token = local_token_idx < uncached_kv_len;
            int const global_token_idx = local_token_idx + global_token_offset;
            VecT data;

            if (valid_token)
            {
                auto const position_id = token_idx_in_kv_cache;
                float2 const* rotary_coef_cache_buffer
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

                if (head_idx == head_num)
                {
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM) + K_DIM;
                    data = *reinterpret_cast<VecT const*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]);
                }
                else
                {
                    auto const src_q_global_offset
                        = static_cast<size_t>(global_token_idx) * head_num * (nope_size + ROPE_DIM)
                        + (nope_size + ROPE_DIM) * head_idx + nope_size;
                    data = *reinterpret_cast<VecT const*>(&q_ptr[src_q_global_offset + head_dim_idx]);
                }

                // Pack two elements into one for gptj rotary embedding.
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
                {
                    GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
                }
            }
            // do sync
            __syncwarp();
            if (valid_token)
            {
                if (head_idx == head_num)
                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(
                        token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                    if constexpr (std::is_same_v<TCache, T>)
                    {
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = data;
                    }
                    else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                            reinterpret_cast<T const*>(&data), quant_scale_kv_val);
                    }
                    // copy to latent_cache (for chunked prefill, it will not load kv cache for uncached k_pe)
                    // we only need to copy original value.
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM) + K_DIM;
                    *reinterpret_cast<VecT*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]) = data;
                }
                else
                {
                    auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (nope_size + ROPE_DIM)
                        + head_idx * (nope_size + ROPE_DIM) + nope_size + head_dim_idx;
                    reinterpret_cast<VecT*>(q_ptr)[dst_q_idx / ELTS_PER_VEC] = data;
                }
            }
        }
    }
    else
    {
        int block_dim = gridDim.z - head_num - 1;
        int block_id = head_idx - head_num - 1;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_uncached_seq_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = kv_scale_orig_quant_ptr ? kv_scale_orig_quant_ptr[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {

            int token_idx_in_kv_cache = local_token_idx + cached_kv_len;
            bool valid_token = local_token_idx < uncached_kv_len;
            int const global_token_idx = local_token_idx + global_token_offset;

            if (valid_token)
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM);

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                if constexpr (std::is_same_v<TCache, T>)
                {
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                        = *reinterpret_cast<VecT const*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]);
                }
                else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                        latent_cache_ptr + src_k_global_offset + head_dim_idx, quant_scale_kv_val);
                }
            }
        }
    }
}

template <typename T, int BLOCK_SIZE, int QK_NOPE_HEAD_DIM, int QK_ROPE_HEAD_DIM, int V_HEAD_DIM>
__global__ void quantizeCopyInputToFp8Kernel(T const* q_buf, __nv_fp8_e4m3* quant_q_buf, T const* k_buf,
    __nv_fp8_e4m3* quant_k_buf, T const* v_buf, __nv_fp8_e4m3* quant_v_buf, int total_q_len, int total_kv_len,
    float const* quant_scale_qkv_ptr, float* bmm1_scale, float* bmm2_scale, float const* quant_scale_o,
    float const* dequant_scale_q, float const* dequant_scale_kv, float host_bmm1_scale)
{
    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    constexpr auto QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
    static_assert(
        (QK_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "QK head size needs to be multiple of 16 bytes.");
    static_assert((V_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "V head size needs to be multiple of 16 bytes.");
    constexpr auto QK_VECS_PER_HEAD = QK_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto V_VECS_PER_HEAD = V_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % QK_VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    static_assert(BLOCK_SIZE % V_VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto QK_TOKENS_PER_BLOCK = BLOCK_SIZE / QK_VECS_PER_HEAD;
    constexpr auto V_TOKENS_PER_BLOCK = BLOCK_SIZE / V_VECS_PER_HEAD;

    size_t const head_idx = blockIdx.z;
    size_t const head_num = gridDim.z;

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        // Calculate bmm scale for FP8 MLA
        float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
        float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
        float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
        if (bmm1_scale)
        {
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            // The scale after fmha bmm1.
            float bmm1_scale_val = dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
            bmm1_scale[0] = bmm1_scale_val;
            bmm1_scale[1] = bmm1_scale_val * kLog2e;
        }
        if (bmm2_scale)
        {
            // The scale after fmha bmm2.
            bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
        }
    }

    size_t const qk_head_dim_vec_idx = (threadIdx.x % QK_VECS_PER_HEAD);
    size_t const v_head_dim_vec_idx = (threadIdx.x % V_VECS_PER_HEAD);
    size_t const qk_head_dim_idx = qk_head_dim_vec_idx * ELTS_PER_VEC;
    size_t const v_head_dim_idx = v_head_dim_vec_idx * ELTS_PER_VEC;

    size_t const q_len_loop_end
        = size_t((total_q_len + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK) * QK_TOKENS_PER_BLOCK;
    size_t const k_len_loop_end
        = size_t((total_kv_len + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK) * QK_TOKENS_PER_BLOCK;
    size_t const v_len_loop_end
        = size_t((total_kv_len + V_TOKENS_PER_BLOCK - 1) / V_TOKENS_PER_BLOCK) * V_TOKENS_PER_BLOCK;
    float quant_scale_qkv_val = quant_scale_qkv_ptr ? quant_scale_qkv_ptr[0] : 1.f;

    // Quantize Q, both src and dst are contiguous
    for (int q_token_idx = (threadIdx.x / QK_VECS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
         q_token_idx < q_len_loop_end; q_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x)
    {
        if (q_token_idx < total_q_len)
        {
            auto const src_q_idx
                = static_cast<size_t>(q_token_idx) * QK_HEAD_DIM * head_num + head_idx * QK_HEAD_DIM + qk_head_dim_idx;
            auto const dst_q_idx = src_q_idx;
            quantCopy<T, ELTS_PER_VEC>(quant_q_buf + dst_q_idx, &q_buf[src_q_idx], quant_scale_qkv_val);
        }
    }

    // Quantize K, both src and dst are contiguous
    for (int k_token_idx = (threadIdx.x / QK_VECS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
         k_token_idx < k_len_loop_end; k_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x)
    {
        if (k_token_idx < total_kv_len)
        {
            auto const src_k_idx
                = static_cast<size_t>(k_token_idx) * QK_HEAD_DIM * head_num + head_idx * QK_HEAD_DIM + qk_head_dim_idx;
            auto const dst_k_idx = src_k_idx;
            quantCopy<T, ELTS_PER_VEC>(quant_k_buf + dst_k_idx, &k_buf[src_k_idx], quant_scale_qkv_val);
        }
    }

    // Quantize V, dst V is contiguous, but src V is not contiguous, so we need to calculate the stride
    size_t const src_v_token_stride = (QK_NOPE_HEAD_DIM + V_HEAD_DIM) * head_num;
    for (int v_token_idx = (threadIdx.x / V_VECS_PER_HEAD) + blockIdx.x * V_TOKENS_PER_BLOCK;
         v_token_idx < v_len_loop_end; v_token_idx += V_TOKENS_PER_BLOCK * gridDim.x)
    {
        if (v_token_idx < total_kv_len)
        {
            auto const src_v_idx
                = static_cast<size_t>(v_token_idx) * src_v_token_stride + head_idx * V_HEAD_DIM + v_head_dim_idx;
            auto const dst_v_idx
                = static_cast<size_t>(v_token_idx) * V_HEAD_DIM * head_num + head_idx * V_HEAD_DIM + v_head_dim_idx;
            quantCopy<T, ELTS_PER_VEC>(quant_v_buf + dst_v_idx, &v_buf[src_v_idx], quant_scale_qkv_val);
        }
    }
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.max_input_seq_len, 32)), params.batch_size, params.head_num + 8);
    auto head_size = params.meta.qk_nope_head_dim;
    applyMLARopeAndAssignQKVKernelOptContext<T, 256, 512, 64, KVCacheBuffer><<<grid, 256, 0, stream>>>(params.q_buf,
        params.k_buf, params.latent_cache, kv_cache_buffer, params.cos_sin_cache, params.head_num, head_size,
        params.meta.kv_lora_rank, params.cu_q_seqlens, params.cache_seq_lens, params.max_input_seq_len,
        params.cache_type, params.quant_scale_kv);
}

template <typename T>
void invokeMLAContextFp8Quantize(MlaParams<T>& params, int total_kv_len, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.cache_type == KvCacheDataType::FP8, "MLA Context: cache_type must be FP8");
    TLLM_CHECK_WITH_INFO(params.q_buf != nullptr, "MLA Context: q_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.k_buf != nullptr, "MLA Context: k_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.v_buf != nullptr, "MLA Context: v_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.quant_q_buf != nullptr, "MLA Context: quant_q_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.quant_k_buf != nullptr, "MLA Context: quant_k_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.quant_v_buf != nullptr, "MLA Context: quant_v_buf must be non-null");

    TLLM_LOG_DEBUG("MLA RoPE Context: Quantizing separate qkv to FP8");

    if (params.acc_q_len > 0)
    {
        constexpr int threads_per_block = 384;
        dim3 grid(int(tensorrt_llm::common::divUp(total_kv_len, 48)), 1, params.head_num);

        TLLM_LOG_DEBUG(
            "Launching quantizeCopyInputToFp8Kernel with grid_size: (%d, %d, %d), threads_per_block: %d, "
            "total_kv_len: %d, acc_q_len: %d",
            grid.x, grid.y, grid.z, threads_per_block, total_kv_len, params.acc_q_len);

        quantizeCopyInputToFp8Kernel<T, threads_per_block, 128, 64, 128>
            <<<grid, threads_per_block, 0, stream>>>(params.q_buf, static_cast<__nv_fp8_e4m3*>(params.quant_q_buf),
                params.k_buf, static_cast<__nv_fp8_e4m3*>(params.quant_k_buf), params.v_buf,
                static_cast<__nv_fp8_e4m3*>(params.quant_v_buf), params.acc_q_len, total_kv_len, params.quant_scale_qkv,
                params.bmm1_scale, params.bmm2_scale, params.quant_scale_o, params.dequant_scale_q,
                params.dequant_scale_kv, params.host_bmm1_scale);
    }
    else
    {
        TLLM_LOG_WARNING("MLA RoPE Context: acc_q_len is 0, skipping quantization.");
    }
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.acc_q_len, 32)), params.head_num + 1 + 8);
    if (params.cache_type == KvCacheDataType::FP8)
        grid.y += params.head_num * 8;
    TLLM_CHECK_WITH_INFO(params.acc_q_len % params.batch_size == 0,
        "MLA can only support input sequences with the same sequence length.");
    auto seq_len = params.acc_q_len / params.batch_size;

    auto* kernel_instance = &applyMLARopeAndAssignQKVKernelGeneration<T, 256, 512, 64, KVCacheBuffer>;
    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = 256;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel_instance, params.q_buf, params.q_pe, params.latent_cache, params.quant_q_buf,
        kv_cache_buffer, params.cos_sin_cache, params.head_num, params.meta.kv_lora_rank, params.acc_q_len, seq_len,
        params.seqQOffset, params.fmha_tile_counter, params.cache_seq_lens, params.cu_kv_seqlens, params.q_pe_ld,
        params.q_pe_stride, params.cache_type, params.bmm1_scale, params.bmm2_scale, params.quant_scale_o,
        params.quant_scale_q, params.quant_scale_kv, params.dequant_scale_q, params.dequant_scale_kv,
        params.host_bmm1_scale, params.helix_position_offsets);
}

template <typename T, typename TCache>
void invokeMLALoadPagedKV(T* compressed_kv_ptr, T* k_pe_ptr, KVBlockArray& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len, int const lora_size, int const rope_size,
    float const* kv_scale_quant_orig_ptr, cudaStream_t stream)
{
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<TCache>;
    // {seq_len / token_per_block, batch_size, head_num}
    TLLM_CHECK_WITH_INFO(lora_size == KT::kLoraSize, "lora_size should be equal to %d", KT::kLoraSize);
    TLLM_CHECK_WITH_INFO(rope_size == KT::kRopeSize, "rope_size should be equal to %d", KT::kRopeSize);
    TLLM_CHECK_WITH_INFO(lora_size + rope_size == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    dim3 grid(static_cast<int>(tensorrt_llm::common::divUp(max_input_seq_len, KT::kTokenPerBlock)), num_contexts, 1);
    loadPagedKVCacheForMLAKernel<T, TCache><<<grid, KT::kBlockSize, 0, stream>>>(
        compressed_kv_ptr, k_pe_ptr, kv_cache, cu_ctx_cached_kv_lens, max_input_seq_len, kv_scale_quant_orig_ptr);
}

template <typename T, typename TCache>
void invokeMLARopeAppendPagedKVAssignQ(KVBlockArray& kv_cache, T* q_ptr, T* latent_cache_ptr, int const num_requests,
    int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_uncached_seq_len,
    float2 const* cos_sin_cache, size_t head_num, int nope_size, int rope_size, int lora_size,
    float const* kv_scale_orig_quant_ptr, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(max_input_uncached_seq_len, 32)), num_requests, head_num + 1 + 8);
    TLLM_CHECK_WITH_INFO(lora_size == 512, "lora_size should be equal to %d", 512);
    TLLM_CHECK_WITH_INFO(rope_size == 64, "rope_size should be equal to %d", 64);
    applyMLARopeAppendPagedKVAssignQKernel<T, TCache, 256, 512, 64><<<grid, 256, 0, stream>>>(kv_cache, q_ptr,
        latent_cache_ptr, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
        nope_size, kv_scale_orig_quant_ptr);
}

#define INSTANTIATE_MLA_ROPE(T, KVCacheBuffer)                                                                         \
    template void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);      \
    template void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

INSTANTIATE_MLA_ROPE(float, KVBlockArray);
INSTANTIATE_MLA_ROPE(half, KVBlockArray);
INSTANTIATE_MLA_ROPE(float, KVLinearBuffer);
INSTANTIATE_MLA_ROPE(half, KVLinearBuffer);
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVBlockArray);
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVLinearBuffer);

#define INSTANTIATE_MLA_QUANTIZE(T)                                                                                    \
    template void invokeMLAContextFp8Quantize<T>(MlaParams<T> & params, int total_kv_len, cudaStream_t stream);

INSTANTIATE_MLA_QUANTIZE(float);
INSTANTIATE_MLA_QUANTIZE(half);
INSTANTIATE_MLA_QUANTIZE(__nv_bfloat16);

#define INSTANTIATE_RW_KVCACHE_MLA(T, TCache)                                                                          \
    template void invokeMLALoadPagedKV<T, TCache>(T * compressed_kv_ptr, T * k_pe_ptr, KVBlockArray & kv_cache,        \
        int const num_contexts, int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len,                     \
        int const lora_size, int const rope_size, float const* kv_scale_quant_orig_ptr, cudaStream_t stream);          \
    template void invokeMLARopeAppendPagedKVAssignQ<T, TCache>(KVBlockArray & kv_cache, T * q_ptr,                     \
        T * latent_cache_ptr, int const num_requests, int64_t const* cu_ctx_cached_kv_lens,                            \
        int64_t const* cu_seq_lens, int const max_input_uncached_seq_len, float2 const* cos_sin_cache,                 \
        size_t head_num, int nope_size, int rope_size, int lora_size, float const* kv_scale_orig_quant_ptr,            \
        cudaStream_t stream);

INSTANTIATE_RW_KVCACHE_MLA(float, float);
INSTANTIATE_RW_KVCACHE_MLA(float, __nv_fp8_e4m3);
INSTANTIATE_RW_KVCACHE_MLA(half, half);
INSTANTIATE_RW_KVCACHE_MLA(half, __nv_fp8_e4m3);
INSTANTIATE_RW_KVCACHE_MLA(__nv_bfloat16, __nv_bfloat16);
INSTANTIATE_RW_KVCACHE_MLA(__nv_bfloat16, __nv_fp8_e4m3);

} // namespace kernels

} // namespace tensorrt_llm
