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

#include "decoderMaskedMultiheadAttentionTemplate.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <cuda_runtime_api.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#include <type_traits>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

namespace mmha
{

template <typename T, int Dh, bool DO_MULTI_BLOCK, bool DO_CROSS_ATTENTION>
inline size_t smem_size_in_bytes(const Multihead_attention_params<T, DO_CROSS_ATTENTION>& params, int threads_per_block)
{
    using Tk = typename kernel_type_t<T>::Type;
    // The amount of shared memory needed to store the Q*K^T values in float.
    const int max_timesteps = DO_CROSS_ATTENTION
        ? params.memory_max_len
        : min((DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep), params.memory_max_len);
    const auto qk_elts = static_cast<std::size_t>(divUp(max_timesteps + 1, 4)); // explicit cast because of the sign
    const auto qk_sz = qk_elts * 16;

    // The extra memory needed if we are not using floats for the final logits.
    size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(Tk) != 4)
    {
        // TDOD
        logits_sz = qk_elts * 4 * sizeof(Tk);
    }
#endif

    // The total size needed during softmax.
    size_t softmax_sz = qk_sz + logits_sz;

    auto constexpr threads_per_value = mmha::threads_per_value<T>(mmha::dh_max(Dh));

    // The number of partial rows to reduce in the final reduction.
    int rows_per_red = threads_per_block / threads_per_value;
    // The amount of storage needed to finalize the outputs.
    size_t red_sz = rows_per_red * params.hidden_size_per_head * sizeof(Tk) / 2;

    size_t transpose_rotary_size = 0;
    if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX)
    {
        assert(params.rotary_embedding_dim > 0);
        transpose_rotary_size = 2 * params.rotary_embedding_dim * sizeof(Tk);
    }

    size_t out_oi_sz = 0;
    if (params.multi_block_mode)
    {
        // The size for partial output reduction computation.
        out_oi_sz = params.max_seq_len_tile * params.hidden_size_per_head * sizeof(T);
    }

    // The max.
    return max(max(max(softmax_sz, red_sz), transpose_rotary_size), out_oi_sz);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int Dh, bool DO_CROSS_ATTENTION>
inline size_t multi_block_grid_setup(const Multihead_attention_params<T, DO_CROSS_ATTENTION>& params,
    int threads_per_block, int tlength, bool do_multi_block)
{
    if (!do_multi_block)
    {
        return 1;
    }

    auto constexpr threads_per_value = mmha::threads_per_value<T>(mmha::dh_max(Dh));

    // Make sure: seq_len_tile * threads_per_value <= threads_per_block (for multi_block_mode)
    params.seq_len_tile = std::floor(threads_per_block / threads_per_value);

    assert(params.seq_len_tile <= params.max_seq_len_tile);

    params.timesteps_per_block = mmha::divUp(tlength, params.seq_len_tile);

#ifndef ENABLE_MULTI_BLOCK_OPTION
    do_multi_block = false;
#endif

    // Return the sequence length tile if using multi block modes.
    return params.seq_len_tile;
}

#define MMHA_LAUNCH_CHECK(DYNAMIC_THDS_PER_BLOCK)                                                                      \
    std::size_t const dynamic_smem_sz{                                                                                 \
        mmha::smem_size_in_bytes<T, Dh, DO_MULTI_BLOCK>(params, DYNAMIC_THDS_PER_BLOCK)};                              \
    /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */              \
    if (dynamic_smem_sz >= 46 * 1024)                                                                                  \
    {                                                                                                                  \
        cudaError_t res = cudaFuncSetAttribute(mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh,  \
                                                   DYNAMIC_THDS_PER_BLOCK, HAS_BEAMS, DO_MULTI_BLOCK>,                 \
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_sz);                                             \
        TLLM_CHECK_WITH_INFO(                                                                                          \
            res == cudaSuccess, "Sequence Length is too long for the MMHA kernel (not enough shared memory).");        \
    }                                                                                                                  \
    TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&available_blocks,                                   \
        mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh, DYNAMIC_THDS_PER_BLOCK, HAS_BEAMS,      \
            DO_MULTI_BLOCK>,                                                                                           \
        DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz));

#define MMHA_KERNEL(DYNAMIC_THDS_PER_BLOCK)                                                                            \
    std::size_t const dynamic_smem_sz{                                                                                 \
        mmha::smem_size_in_bytes<T, Dh, DO_MULTI_BLOCK>(params, DYNAMIC_THDS_PER_BLOCK)};                              \
    /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */              \
    if (dynamic_smem_sz >= 46 * 1024)                                                                                  \
    {                                                                                                                  \
        cudaError_t res = cudaFuncSetAttribute(                                                                        \
            mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh, DYNAMIC_THDS_PER_BLOCK,             \
                KernelParamsType::DO_CROSS_ATTENTION, HAS_BEAMS, DO_MULTI_BLOCK>,                                      \
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_sz);                                             \
        TLLM_CHECK_WITH_INFO(                                                                                          \
            res == cudaSuccess, "Sequence Length is too long for the MMHA kernel (not enough shared memory).");        \
    }                                                                                                                  \
    mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh, DYNAMIC_THDS_PER_BLOCK,                     \
        KernelParamsType::DO_CROSS_ATTENTION, HAS_BEAMS, DO_MULTI_BLOCK>                                               \
        <<<grid, DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz, stream>>>(params, kv_cache_buffer);

// if resources are not enough to launch 512 threads per block, we will fallback to 256.
#define MMHA_LAUNCH_512_BLOCKSIZE()                                                                                    \
    int available_blocks = -1;                                                                                         \
    MMHA_LAUNCH_CHECK(512);                                                                                            \
    if (available_blocks <= 0)                                                                                         \
    {                                                                                                                  \
        MMHA_KERNEL(256);                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        MMHA_KERNEL(512);                                                                                              \
    }

// if resources are not enough to launch 1024 threads per block, we will fallback to 512.
#define MMHA_LAUNCH_1024_BLOCKSIZE()                                                                                   \
    int available_blocks = -1;                                                                                         \
    MMHA_LAUNCH_CHECK(1024);                                                                                           \
    if (available_blocks <= 0)                                                                                         \
    {                                                                                                                  \
        MMHA_LAUNCH_512_BLOCKSIZE();                                                                                   \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        MMHA_KERNEL(1024);                                                                                             \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename T_cache, typename KVCacheBuffer, typename KernelParamsType, int Dh, int THDS_PER_BLOCK,
    bool HAS_BEAMS, bool DO_MULTI_BLOCK>
void mmha_launch_kernel_ex(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream, int tlength)
{
    std::size_t const seq_len_tile{mmha::multi_block_grid_setup<T, Dh, KernelParamsType::DO_CROSS_ATTENTION>(
        params, THDS_PER_BLOCK, tlength, DO_MULTI_BLOCK)};
    dim3 grid{static_cast<unsigned>(params.num_heads), static_cast<unsigned>(params.batch_size),
        static_cast<unsigned>(seq_len_tile)};

    if (DO_MULTI_BLOCK)
    {
        MMHA_KERNEL(THDS_PER_BLOCK);
    }
    else
    {
        const int kernel_total_blocks = params.batch_size * params.num_heads;
        // Don't tune the block size if batchxhead is large enough.
        // The max number of warps we can launch per SM is 32 limited by registers.
        if (kernel_total_blocks >= params.multi_processor_count * 4)
        {
            MMHA_KERNEL(THDS_PER_BLOCK);
            return;
        }

        // Tune block size based on batchxhead to increase occupancy.
        int num_blocks_per_sm = -1;
        // Set 0 dynamic shared memory size as we need the number of available blocks limited by registers.
        // Dynamic shared memory is fixed for different block size.
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
            mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh, THDS_PER_BLOCK, HAS_BEAMS,
                DO_MULTI_BLOCK>,
            THDS_PER_BLOCK, 0));

        int block_size_factor = min(
            mmha::divUp(params.multi_processor_count * num_blocks_per_sm, kernel_total_blocks), num_blocks_per_sm);
        // Max block size is 1024.
        const int dynamic_block_size = min(THDS_PER_BLOCK * block_size_factor, 1024);

        // Make sure number of threads per block is power of 2.
        if (dynamic_block_size <= 256)
        {
            MMHA_KERNEL(256);
        }
        else if (dynamic_block_size <= 512)
        {
            // Check if the kernel with new block size can be launched in terms of resources.
            MMHA_LAUNCH_512_BLOCKSIZE();
        }
        else if (dynamic_block_size <= 1024)
        {
            // Check if the kernel with new block size can be launched in terms of resources.
            MMHA_LAUNCH_1024_BLOCKSIZE();
        }
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh, int THDS_PER_BLOCK, bool HAS_BEAMS,
    bool DO_MULTI_BLOCK>
void mmha_launch_kernel_dispatch_8bits_kv_cache(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream, int tlength)
{
    if (params.int8_kv_cache)
    {
        mmha_launch_kernel_ex<T, int8_t, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK, HAS_BEAMS,
            DO_MULTI_BLOCK>(params, kv_cache_buffer, stream, tlength);
    }
#ifdef ENABLE_FP8
    else if (params.fp8_kv_cache)
    {
        mmha_launch_kernel_ex<T, __nv_fp8_e4m3, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK, HAS_BEAMS,
            DO_MULTI_BLOCK>(params, kv_cache_buffer, stream, tlength);
    }
#endif // ENABLE_FP8
    else
    {
        mmha_launch_kernel_ex<T, T, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK, HAS_BEAMS, DO_MULTI_BLOCK>(
            params, kv_cache_buffer, stream, tlength);
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh, bool HAS_BEAMS>
void mmha_launch_kernel_dispatch(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    int const tlength = params.timestep;
    if (tlength < 1024)
    {
        mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, HAS_BEAMS, false>(
            params, kv_cache_buffer, stream, tlength);
    }
    else
    {
        if (params.multi_block_mode)
        {
            mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, HAS_BEAMS, true>(
                params, kv_cache_buffer, stream, tlength);
        }
        else
        {
            mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, HAS_BEAMS, false>(
                params, kv_cache_buffer, stream, tlength);
        }
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh>
void mmha_launch_kernel(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    assert((params.rotary_embedding_dim != 0)
        == (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX
            || params.position_embedding_type == PositionEmbeddingType::kROPE_GPTJ));
    if (params.beam_width == 1)
    {
        mmha_launch_kernel_dispatch<T, KVCacheBuffer, KernelParamsType, Dh, false>(params, kv_cache_buffer, stream);
    }
    else
    {
        mmha_launch_kernel_dispatch<T, KVCacheBuffer, KernelParamsType, Dh, true>(params, kv_cache_buffer, stream);
    }
}

} // namespace mmha

#define INSTANTIATE_MMHA_LAUNCHERS(T, Dh)                                                                              \
    template void mmha_launch_kernel<T, KVLinearBuffer, Masked_multihead_attention_params<T>, Dh>(                     \
        const Masked_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                     \
        const cudaStream_t& stream);                                                                                   \
    template void mmha_launch_kernel<T, KVBlockArray, Masked_multihead_attention_params<T>, Dh>(                       \
        const Masked_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                       \
        const cudaStream_t& stream);                                                                                   \
    template void mmha_launch_kernel<T, KVLinearBuffer, Cross_multihead_attention_params<T>, Dh>(                      \
        const Cross_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                      \
        const cudaStream_t& stream);                                                                                   \
    template void mmha_launch_kernel<T, KVBlockArray, Cross_multihead_attention_params<T>, Dh>(                        \
        const Cross_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                        \
        const cudaStream_t& stream);

} // namespace kernels
} // namespace tensorrt_llm
