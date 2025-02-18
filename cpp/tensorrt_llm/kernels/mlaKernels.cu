/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
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
};

template <>
struct VecType<float>
{
    using Type = float4;
};

template <>
struct VecType<half>
{
    using Type = uint4;
};

template <>
struct VecType<__nv_bfloat16>
{
    using Type = mmha::bf16_8_t;
};

namespace mla
{

template <typename T>
inline __device__ void apply_rotary_embedding_mla(
    T& q, T q_pair_left, T q_pair_right, T& k, T k_pair_left, T k_pair_right, float2 const& coef)
{
    T cos = cuda_cast<T>(coef.x);
    T sin = cuda_cast<T>(coef.y);

    q = cuda_cast<T>(cuda_cast<float>(cos * q_pair_left)) + cuda_cast<T>(cuda_cast<float>(sin * q_pair_right));
    k = cuda_cast<T>(cuda_cast<float>(cos * k_pair_left)) + cuda_cast<T>(cuda_cast<float>(sin * k_pair_right));
}

template <typename T>
inline __device__ void apply_rotary_embedding_mla(T& q, T q_left, T q_right, float2 const& coef)
{
    T cos = cuda_cast<T>(coef.x);
    T sin = cuda_cast<T>(coef.y);

    q = cuda_cast<T>(cuda_cast<float>(cos * q_left)) + cuda_cast<T>(cuda_cast<float>(sin * q_right));
}

} // namespace mla

template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void applyMLARopeAndAssignQKVKernelOptContext(T* qkv_output, T const* fuse_buf, KVCacheBuffer kv_cache,
    float2 const* cos_sin_cache, size_t head_num, int head_size, int c_q, int c_k, int* cu_q_seqlens,
    int32_t const* kv_cache_lengths, uint32_t max_input_seq_len)
{

    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto HALF_ROTATARY_DIM = ROPE_DIM / 2;
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
        bool const first_half = head_dim_idx < HALF_ROTATARY_DIM;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;

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
            auto const src_bias = first_half ? head_dim_idx * 2 : (head_dim_idx - HALF_ROTATARY_DIM) * 2;
            float2 const* rotary_coef_cache_buffer
                = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx);

            VecT q, k;
            VecT q_ref[2], k_ref[2];
            auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_q + c_k + ROPE_DIM) + c_q + c_k;
            auto const src_q_global_offset
                = static_cast<size_t>(global_token_idx) * head_num * ((head_size + ROPE_DIM) * 2 + head_size)
                + (head_size + ROPE_DIM) * head_idx + head_size;

            for (int i = 0; i < 2; ++i)
            {
                q_ref[i]
                    = *reinterpret_cast<VecT const*>(&qkv_output[src_q_global_offset + src_bias + i * ELTS_PER_VEC]);
                k_ref[i] = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + src_bias + i * ELTS_PER_VEC]);
            }

            for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
            {
                float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                rotary_coef_cache.y = first_half ? -rotary_coef_cache.y : rotary_coef_cache.y;
                auto& q_ = reinterpret_cast<T*>(&q)[elt_id];
                auto& k_ = reinterpret_cast<T*>(&k)[elt_id];
                auto q_left = first_half ? reinterpret_cast<T*>(&q_ref)[elt_id * 2]
                                         : reinterpret_cast<T*>(&q_ref)[elt_id * 2 + 1];
                auto q_right = first_half ? reinterpret_cast<T*>(&q_ref)[elt_id * 2 + 1]
                                          : reinterpret_cast<T*>(&q_ref)[elt_id * 2];
                auto k_left = first_half ? reinterpret_cast<T*>(&k_ref)[elt_id * 2]
                                         : reinterpret_cast<T*>(&k_ref)[elt_id * 2 + 1];
                auto k_right = first_half ? reinterpret_cast<T*>(&k_ref)[elt_id * 2 + 1]
                                          : reinterpret_cast<T*>(&k_ref)[elt_id * 2];
                // float2 rotary_coef_cache;
                // T q_left, q_right, k_left, k_right;
                mla::apply_rotary_embedding_mla(q_, q_left, q_right, k_, k_left, k_right, rotary_coef_cache);
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
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx] = k;
                }
                auto const dst_q_idx
                    = static_cast<size_t>(global_token_idx) * head_num * ((head_size + ROPE_DIM) * 2 + head_size)
                    + head_idx * (head_size + ROPE_DIM) + head_size + head_dim_idx;
                auto const dst_k_idx
                    = static_cast<size_t>(global_token_idx) * head_num * ((head_size + ROPE_DIM) * 2 + head_size)
                    + head_num * (head_size + ROPE_DIM) + head_idx * (head_size + ROPE_DIM) + head_size + head_dim_idx;
                reinterpret_cast<VecT*>(qkv_output)[dst_q_idx / ELTS_PER_VEC] = q;
                reinterpret_cast<VecT*>(qkv_output)[dst_k_idx / ELTS_PER_VEC] = k;
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
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_q + c_k + ROPE_DIM) + c_q;

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                    = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
            }
        }
    }
}

template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void applyMLARopeAndAssignQKVKernelGeneration(T* qkv_output, T* q_buf, T const* fuse_buf,
    KVCacheBuffer kv_cache, float2 const* cos_sin_cache, size_t head_num, int head_size, int c_q, int c_k,
    int total_s_len, int* seqQOffset, uint32_t* fmha_tile_counter, int32_t const* kv_cache_lengths, int* seqKVOffsets)
{

    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto HALF_ROTATARY_DIM = ROPE_DIM / 2;
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

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
        fmha_tile_counter[0] = 0;
        seqQOffset[0] = 0;
    }

    if (head_idx <= head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
        bool const first_half = head_dim_idx < HALF_ROTATARY_DIM;

        int const seq_len_loop_end = size_t((total_s_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx;
            bool const valid_token = batch_idx < total_s_len;
            VecT data;

            if (valid_token)
            {
                VecT ref[2];

                auto const position_id = kv_cache_lengths[batch_idx] - 1;
                auto const src_bias = first_half ? head_dim_idx * 2 : (head_dim_idx - HALF_ROTATARY_DIM) * 2;
                float2 const* rotary_coef_cache_buffer
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx);

                if (head_idx == head_num)
                {
                    auto const src_k_global_offset
                        = static_cast<size_t>(global_token_idx) * (c_q + c_k + ROPE_DIM) + c_q + c_k;

                    for (int i = 0; i < 2; ++i)
                    {
                        ref[i] = *reinterpret_cast<VecT const*>(
                            &fuse_buf[src_k_global_offset + src_bias + i * ELTS_PER_VEC]);
                    }
                }
                else
                {
                    auto const src_q_global_offset
                        = static_cast<size_t>(global_token_idx) * head_num * (head_size + ROPE_DIM)
                        + (head_size + ROPE_DIM) * head_idx + head_size;

                    for (int i = 0; i < 2; ++i)
                    {
                        ref[i]
                            = *reinterpret_cast<VecT const*>(&q_buf[src_q_global_offset + src_bias + i * ELTS_PER_VEC]);
                    }
                }

                for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
                {
                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    rotary_coef_cache.y = first_half ? -rotary_coef_cache.y : rotary_coef_cache.y;
                    auto& data_ = reinterpret_cast<T*>(&data)[elt_id];
                    auto data_left = first_half ? reinterpret_cast<T*>(&ref)[elt_id * 2]
                                                : reinterpret_cast<T*>(&ref)[elt_id * 2 + 1];
                    auto data_right = first_half ? reinterpret_cast<T*>(&ref)[elt_id * 2 + 1]
                                                 : reinterpret_cast<T*>(&ref)[elt_id * 2];
                    mla::apply_rotary_embedding_mla(data_, data_left, data_right, rotary_coef_cache);
                }
            }

            __syncwarp();

            if (valid_token)
            {
                if (head_idx == head_num)
                {
                    auto const batch_idx = global_token_idx;
                    auto const token_kv_idx = kv_cache_lengths[batch_idx] - 1;

                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(
                        token_kv_idx, 0, TOTAL_VEC_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx] = data;
                }
                else
                {
                    auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (c_k + ROPE_DIM)
                        + head_idx * (c_k + ROPE_DIM) + c_k + head_dim_idx;
                    reinterpret_cast<VecT*>(qkv_output)[dst_q_idx / ELTS_PER_VEC] = data;
                }
            }
        }
    }
    else
    {
        int block_dim = gridDim.y - head_num - 1;
        int block_id = head_idx - head_num - 1;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {
            bool valid_token = global_token_idx < total_s_len;
            auto const batch_idx = std::min(global_token_idx, total_s_len - 1);

            if (valid_token)
            {
                if (head_dim_vec_idx == 0)
                {
                    seqQOffset[batch_idx + 1] = head_num * (batch_idx + 1);
                }

                auto const token_kv_idx = kv_cache_lengths[batch_idx] - 1;
                auto const src_kv_global_offset = static_cast<size_t>(global_token_idx) * (c_q + c_k + ROPE_DIM) + c_q;

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                auto inBlockIdx = kv_cache.getKVLocalIdx(token_kv_idx, 0, TOTAL_VEC_PER_HEAD, head_dim_vec_idx);

                reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                    = *reinterpret_cast<VecT const*>(&fuse_buf[src_kv_global_offset + head_dim_idx]);
            }
        }
    }

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempKVStorage;
    BlockPrefixCallbackOp prefixKVOp(0);

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        int const batchSizeBound = total_s_len;
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

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.max_input_seq_len, 32)), params.batch_size, params.head_num + 8);
    auto head_size = params.meta.qk_nope_head_dim;
    applyMLARopeAndAssignQKVKernelOptContext<T, 256, 512, 64, KVCacheBuffer>
        <<<grid, 256, 0, stream>>>(params.attention_input_buf, params.fused_a_input, kv_cache_buffer,
            params.cos_sin_cache, params.head_num, head_size, params.meta.q_lora_rank, params.meta.kv_lora_rank,
            params.cu_q_seqlens, params.cache_seq_lens, params.max_input_seq_len);
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.acc_q_len, 32)), params.head_num + 1 + 8);
    auto head_size = params.meta.qk_nope_head_dim;
    applyMLARopeAndAssignQKVKernelGeneration<T, 256, 512, 64, KVCacheBuffer>
        <<<grid, 256, 0, stream>>>(params.attention_input_buf, params.q_buf, params.fused_a_input, kv_cache_buffer,
            params.cos_sin_cache, params.head_num, head_size, params.meta.q_lora_rank, params.meta.kv_lora_rank,
            params.acc_q_len, params.seqQOffset, params.fmha_tile_counter, params.cache_seq_lens, params.cu_kv_seqlens);
}

#define INSTANTIATE_MLA_ROPE(T, KVCacheBuffer)                                                                         \
    template void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);      \
    template void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

INSTANTIATE_MLA_ROPE(float, KVBlockArray);
INSTANTIATE_MLA_ROPE(half, KVBlockArray);
INSTANTIATE_MLA_ROPE(float, KVLinearBuffer);
INSTANTIATE_MLA_ROPE(half, KVLinearBuffer);

#ifdef ENABLE_BF16
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVBlockArray);
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVLinearBuffer);
#endif

} // namespace kernels

} // namespace tensorrt_llm
