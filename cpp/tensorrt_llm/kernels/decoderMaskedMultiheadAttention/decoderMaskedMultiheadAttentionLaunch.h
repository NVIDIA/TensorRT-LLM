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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <algorithm>
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
inline size_t smem_size_in_bytes(Multihead_attention_params<T, DO_CROSS_ATTENTION> const& params, int threads_per_block)
{
    using Tk = typename kernel_type_t<T>::Type;
    // The amount of shared memory needed to store the Q*K^T values in float.
    auto const attention_window_size_bound = divUp(
        std::min(params.cyclic_attention_window_size, params.chunked_attention_size), std::max(params.seq_len_tile, 1));
    auto const max_timesteps = DO_CROSS_ATTENTION
        ? attention_window_size_bound
        : min((DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep), attention_window_size_bound);
    auto const qk_elts = static_cast<std::size_t>(divUp(max_timesteps + 1, 4)); // explicit cast because of the sign
    auto const qk_sz = qk_elts * 16;

    // The extra memory needed if we are not using floats for the final logits.
    size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACCUM_FOR_LOGITS
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
    if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX
        || params.position_embedding_type == PositionEmbeddingType::kLONG_ROPE
        || params.position_embedding_type == PositionEmbeddingType::kROPE_M)
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
inline void multi_block_grid_setup(dim3& grid, Multihead_attention_params<T, DO_CROSS_ATTENTION> const& params,
    int blocks_per_sm, int block_size, int tlength)
{
    if (!params.multi_block_mode)
    {
        return;
    }

    int balanced_seq_len_tile
        = mmha::divUp(params.multi_processor_count * blocks_per_sm, params.batch_size * params.num_heads);

    int const threads_per_value = mmha::threads_per_value<T>(mmha::dh_max(Dh));
    // Make sure that each block at least processes one loop of kv (unroll size is default at 8).
    int const seq_len_per_kv_loop = mmha::divUp(block_size, threads_per_value) * 8;
    int max_seq_len_tile = params.max_seq_len_tile;

    bool const multi_block_debug_flag = getEnvMmhaMultiblockDebug();

    // User defined number of blocks.
    if (multi_block_debug_flag)
    {
        int const env_seq_len_tile = getEnvMmhaBlocksPerSequence();
        balanced_seq_len_tile = env_seq_len_tile > 0 ? env_seq_len_tile : balanced_seq_len_tile;
    }
    else
    {
        max_seq_len_tile = std::min(mmha::divUp(tlength + 1, seq_len_per_kv_loop), max_seq_len_tile);
    }

    params.seq_len_tile = std::clamp(balanced_seq_len_tile, params.min_seq_len_tile, max_seq_len_tile);

    TLLM_CHECK_WITH_INFO(
        params.seq_len_tile <= block_size, "The number of blocks per sequence may not exceed the thread block size.");

    // We should consider the new timestep.
    params.timesteps_per_block
        = mmha::divUp(std::min({tlength, params.cyclic_attention_window_size, params.chunked_attention_size}) + 1,
            params.seq_len_tile);

    params.multi_block_mode = (params.seq_len_tile > 1);

    static bool debug_flag_printed_once = false;
    if (multi_block_debug_flag && !debug_flag_printed_once)
    {
        TLLM_LOG_INFO("MMHA kernel info: threads per block(%d), launched_blocks_per_sequence(%d), sequence_length(%d).",
            block_size, params.seq_len_tile, tlength + 1);
        debug_flag_printed_once = true;
    }

    grid.z = params.seq_len_tile;
}

#define MMHA_LAUNCH_CHECK(DYNAMIC_THDS_PER_BLOCK, DO_MULTI_BLOCK)                                                      \
    std::size_t const dynamic_smem_sz{                                                                                 \
        mmha::smem_size_in_bytes<T, Dh, DO_MULTI_BLOCK>(params, DYNAMIC_THDS_PER_BLOCK)};                              \
    /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */              \
    if (dynamic_smem_sz >= 46 * 1024)                                                                                  \
    {                                                                                                                  \
        cudaError_t res = cudaFuncSetAttribute(                                                                        \
            mmha::masked_multihead_attention_kernel<T, T_cache, TKcache, KVCacheBuffer, KCacheBuffer, Dh,              \
                DYNAMIC_THDS_PER_BLOCK, KernelParamsType::DO_CROSS_ATTENTION, HAS_BEAMS, DO_MULTI_BLOCK, POS_SHIFT,    \
                BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>,                                    \
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_sz);                                             \
        TLLM_CHECK_WITH_INFO(                                                                                          \
            res == cudaSuccess, "Sequence Length is too long for the MMHA kernel (not enough shared memory).");        \
    }                                                                                                                  \
    TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&available_blocks,                                   \
        mmha::masked_multihead_attention_kernel<T, T_cache, TKcache, KVCacheBuffer, KCacheBuffer, Dh,                  \
            DYNAMIC_THDS_PER_BLOCK, KernelParamsType::DO_CROSS_ATTENTION, HAS_BEAMS, DO_MULTI_BLOCK, POS_SHIFT,        \
            BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>,                                        \
        DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz));

#define MMHA_KERNEL(DYNAMIC_THDS_PER_BLOCK, ENABLE_MULTI_BLOCK)                                                        \
    std::size_t const dynamic_smem_sz{                                                                                 \
        mmha::smem_size_in_bytes<T, Dh, ENABLE_MULTI_BLOCK>(params, DYNAMIC_THDS_PER_BLOCK)};                          \
    /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */              \
    if (dynamic_smem_sz >= 46 * 1024)                                                                                  \
    {                                                                                                                  \
        cudaError_t res = cudaFuncSetAttribute(                                                                        \
            mmha::masked_multihead_attention_kernel<T, T_cache, TKcache, KVCacheBuffer, KCacheBuffer, Dh,              \
                DYNAMIC_THDS_PER_BLOCK, KernelParamsType::DO_CROSS_ATTENTION, HAS_BEAMS, ENABLE_MULTI_BLOCK,           \
                POS_SHIFT, BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>,                         \
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_sz);                                             \
        TLLM_CHECK_WITH_INFO(                                                                                          \
            res == cudaSuccess, "Sequence Length is too long for the MMHA kernel (not enough shared memory).");        \
    }                                                                                                                  \
    const auto mmhaFunc = mmha::masked_multihead_attention_kernel<T, T_cache, TKcache, KVCacheBuffer, KCacheBuffer,    \
        Dh, DYNAMIC_THDS_PER_BLOCK, KernelParamsType::DO_CROSS_ATTENTION, HAS_BEAMS, ENABLE_MULTI_BLOCK, POS_SHIFT,    \
        BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>;                                            \
    if (tensorrt_llm::common::getEnvEnablePDL())                                                                       \
    {                                                                                                                  \
        TLLM_LOG_DEBUG("Enable PDL in MMHA");                                                                          \
        cudaLaunchConfig_t kernelConfig = {0};                                                                         \
        kernelConfig.gridDim = grid;                                                                                   \
        kernelConfig.blockDim = DYNAMIC_THDS_PER_BLOCK;                                                                \
        kernelConfig.dynamicSmemBytes = dynamic_smem_sz;                                                               \
        kernelConfig.stream = stream;                                                                                  \
                                                                                                                       \
        cudaLaunchAttribute attribute[1];                                                                              \
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                          \
        attribute[0].val.programmaticStreamSerializationAllowed = 1;                                                   \
        kernelConfig.attrs = attribute;                                                                                \
        kernelConfig.numAttrs = 1;                                                                                     \
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&kernelConfig, mmhaFunc, params, kv_cache_buffer, k_cache_buffer));         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        mmhaFunc<<<grid, DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz, stream>>>(params, kv_cache_buffer, k_cache_buffer);  \
    }

// if resources are not enough to launch 512 threads per block, we will fallback to 256.
#define MMHA_512_BLOCKSIZE_CHECK(DO_MULTI_BLOCK)                                                                       \
    MMHA_LAUNCH_CHECK(512, DO_MULTI_BLOCK);                                                                            \
    if (available_blocks <= 0)                                                                                         \
    {                                                                                                                  \
        MMHA_LAUNCH_CHECK(256, DO_MULTI_BLOCK);                                                                        \
        dynamic_block_size = 256;                                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        dynamic_block_size = 512;                                                                                      \
    }

// if resources are not enough to launch 1024 threads per block, we will fallback to 512.
#define MMHA_1024_BLOCKSIZE_CHECK(DO_MULTI_BLOCK)                                                                      \
    MMHA_LAUNCH_CHECK(1024, DO_MULTI_BLOCK);                                                                           \
    if (available_blocks > 0)                                                                                          \
    {                                                                                                                  \
        dynamic_block_size = 1024;                                                                                     \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        MMHA_512_BLOCKSIZE_CHECK(DO_MULTI_BLOCK);                                                                      \
    }

// The previous dynamic_block_size might be calculated based on muli-block-mode enabled,
// while the final launch set it disabled, so we need to fallback to smaller kernel block size
// if there are not enough resources.
#define MMHA_DYNAMIC_LAUNCH(DO_MULTI_BLOCK)                                                                            \
    if (dynamic_block_size == 256)                                                                                     \
    {                                                                                                                  \
        MMHA_KERNEL(256, DO_MULTI_BLOCK);                                                                              \
    }                                                                                                                  \
    else if (dynamic_block_size == 512)                                                                                \
    {                                                                                                                  \
        MMHA_KERNEL(512, DO_MULTI_BLOCK);                                                                              \
    }                                                                                                                  \
    else if (dynamic_block_size == 1024)                                                                               \
    {                                                                                                                  \
        MMHA_KERNEL(1024, DO_MULTI_BLOCK);                                                                             \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename T_cache, typename TKcache, typename KVCacheBuffer, typename KCacheBuffer,
    typename KernelParamsType, int Dh, int THDS_PER_BLOCK, bool HAS_BEAMS, bool DO_MULTI_BLOCK, bool POS_SHIFT,
    bool BLOCK_SPARSE_ATTN, bool IMPLICIT_REL_ATTN_BIAS, bool ATTN_LOGIT_SOFTCAPPING>
void mmha_launch_kernel_ex(KernelParamsType const& params, KVCacheBuffer const& kv_cache_buffer,
    KCacheBuffer const& k_cache_buffer, cudaStream_t const& stream, int tlength)
{
    dim3 grid{static_cast<unsigned>(params.num_heads), static_cast<unsigned>(params.batch_size), 1};

    int const kernel_total_blocks = params.batch_size * params.num_heads;
    // Don't tune the block size if batchxhead is large enough.
    // The max number of warps we can launch per SM is 32 limited by registers.
    if (!DO_MULTI_BLOCK && kernel_total_blocks >= params.multi_processor_count * 4)
    {
        MMHA_KERNEL(THDS_PER_BLOCK, false);
        return;
    }

    // Tune block size based on batchxhead to increase occupancy.
    int num_blocks_per_sm = -1;
    // Set 0 dynamic shared memory size as we need the number of available blocks limited by registers.
    // Dynamic shared memory is fixed for different block size.
    TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
        mmha::masked_multihead_attention_kernel<T, T_cache, TKcache, KVCacheBuffer, KCacheBuffer, Dh, THDS_PER_BLOCK,
            KernelParamsType::DO_CROSS_ATTENTION, HAS_BEAMS, DO_MULTI_BLOCK, POS_SHIFT, BLOCK_SPARSE_ATTN,
            IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>,
        THDS_PER_BLOCK, 0));

    int block_size_factor
        = min(mmha::divUp(params.multi_processor_count * num_blocks_per_sm, kernel_total_blocks), num_blocks_per_sm);

    // Max block size is 1024.
    int dynamic_block_size = min(THDS_PER_BLOCK * block_size_factor, 1024);

    // Check if resources are enough for launch.
    int available_blocks = -1;
    if (dynamic_block_size < 512)
    {
        MMHA_LAUNCH_CHECK(256, DO_MULTI_BLOCK);
        dynamic_block_size = 256;
    }
    else if (dynamic_block_size < 1024)
    {
        MMHA_512_BLOCKSIZE_CHECK(DO_MULTI_BLOCK);
    }
    else if (dynamic_block_size == 1024)
    {
        MMHA_1024_BLOCKSIZE_CHECK(DO_MULTI_BLOCK);
    }

    // Block size can be finetuned by TRTLLM_MMHA_KERNEL_BLOCK_SIZE.
    dynamic_block_size = getEnvMmhaKernelBlockSize() > 0 ? getEnvMmhaKernelBlockSize() : dynamic_block_size;

    // If blocks with larger block size already fill all SMs, then disable the multi blocks mode.
    mmha::multi_block_grid_setup<T, Dh>(grid, params, available_blocks, dynamic_block_size, tlength);

    // Launch kernels based on the valid block size.
    switch (dynamic_block_size)
    {
    case 256:
        if (params.multi_block_mode)
        {
            MMHA_KERNEL(256, true);
        }
        else
        {
            MMHA_KERNEL(256, false);
        }
        break;
    case 512:
        if (params.multi_block_mode)
        {
            MMHA_KERNEL(512, true);
        }
        else
        {
            MMHA_512_BLOCKSIZE_CHECK(false);
            MMHA_DYNAMIC_LAUNCH(false);
        }
        break;
    case 1024:
        if (params.multi_block_mode)
        {
            MMHA_KERNEL(1024, true);
        }
        else
        {
            MMHA_1024_BLOCKSIZE_CHECK(false);
            MMHA_DYNAMIC_LAUNCH(false);
        }
        break;
    default: TLLM_CHECK_WITH_INFO(false, "Wrong kernel block size for launching the MMHA kernel.");
    }
}

template <typename T, typename T_cache, typename KVCacheBuffer, typename KernelParamsType, int Dh, int THDS_PER_BLOCK,
    bool HAS_BEAMS, bool DO_MULTI_BLOCK, bool BLOCK_SPARSE_ATTN, bool IMPLICIT_REL_ATTN_BIAS,
    bool ATTN_LOGIT_SOFTCAPPING>
void mmha_launch_kernel_dispatch_pos_shift(KernelParamsType const& params, KVCacheBuffer const& kv_cache_buffer,
    KVLinearBuffer const& shift_k_cache, cudaStream_t const& stream, int tlength)
{
    if (params.position_shift_enabled && !KernelParamsType::DO_CROSS_ATTENTION)
    {
        mmha_launch_kernel_ex<T, T_cache, T, KVCacheBuffer, KVLinearBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
            HAS_BEAMS, DO_MULTI_BLOCK, true, BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(
            params, kv_cache_buffer, shift_k_cache, stream, tlength);
    }
    else
    {
        mmha_launch_kernel_ex<T, T_cache, T_cache, KVCacheBuffer, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
            HAS_BEAMS, DO_MULTI_BLOCK, false, BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(
            params, kv_cache_buffer, kv_cache_buffer, stream, tlength);
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh, int THDS_PER_BLOCK, bool HAS_BEAMS,
    bool DO_MULTI_BLOCK, bool BLOCK_SPARSE_ATTN, bool IMPLICIT_REL_ATTN_BIAS, bool ATTN_LOGIT_SOFTCAPPING>
void mmha_launch_kernel_dispatch_8bits_kv_cache(KernelParamsType const& params, KVCacheBuffer const& kv_cache_buffer,
    KVLinearBuffer const& shift_k_cache, cudaStream_t const& stream, int tlength)
{
    if (params.int8_kv_cache)
    {
        mmha_launch_kernel_dispatch_pos_shift<T, int8_t, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK, HAS_BEAMS,
            DO_MULTI_BLOCK, BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(
            params, kv_cache_buffer, shift_k_cache, stream, tlength);
    }
#ifdef ENABLE_FP8
    else if (params.fp8_kv_cache)
    {
        mmha_launch_kernel_dispatch_pos_shift<T, __nv_fp8_e4m3, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
            HAS_BEAMS, DO_MULTI_BLOCK, BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(
            params, kv_cache_buffer, shift_k_cache, stream, tlength);
    }
#endif // ENABLE_FP8
    else
    {
        mmha_launch_kernel_dispatch_pos_shift<T, T, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK, HAS_BEAMS,
            DO_MULTI_BLOCK, BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(
            params, kv_cache_buffer, shift_k_cache, stream, tlength);
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh, bool HAS_BEAMS, bool BLOCK_SPARSE_ATTN,
    bool IMPLICIT_REL_ATTN_BIAS, bool ATTN_LOGIT_SOFTCAPPING>
void mmha_launch_kernel_dispatch(KernelParamsType const& params, KVCacheBuffer const& kv_cache_buffer,
    KVLinearBuffer const& shift_k_cache, cudaStream_t const& stream)
{
    int const tlength = params.timestep;
    if (params.multi_block_mode)
    {
        mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, HAS_BEAMS, true,
            BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(
            params, kv_cache_buffer, shift_k_cache, stream, tlength);
    }
    else
    {
        mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, HAS_BEAMS, false,
            BLOCK_SPARSE_ATTN, IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(
            params, kv_cache_buffer, shift_k_cache, stream, tlength);
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh, bool BLOCK_SPARSE_ATTN,
    bool IMPLICIT_REL_ATTN_BIAS, bool ATTN_LOGIT_SOFTCAPPING>
void mmha_launch_kernel(KernelParamsType const& params, KVCacheBuffer const& kv_cache_buffer,
    KVLinearBuffer const& shift_k_cache, cudaStream_t const& stream)
{
    assert((params.rotary_embedding_dim != 0)
        == (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX
            || params.position_embedding_type == PositionEmbeddingType::kROPE_GPTJ
            || params.position_embedding_type == PositionEmbeddingType::kLONG_ROPE
            || params.position_embedding_type == PositionEmbeddingType::kROPE_M));
    if (params.beam_width == 1)
    {
        mmha_launch_kernel_dispatch<T, KVCacheBuffer, KernelParamsType, Dh, false, BLOCK_SPARSE_ATTN,
            IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(params, kv_cache_buffer, shift_k_cache, stream);
    }
    else
    {
        mmha_launch_kernel_dispatch<T, KVCacheBuffer, KernelParamsType, Dh, true, BLOCK_SPARSE_ATTN,
            IMPLICIT_REL_ATTN_BIAS, ATTN_LOGIT_SOFTCAPPING>(params, kv_cache_buffer, shift_k_cache, stream);
    }
}

} // namespace mmha

#define INSTANTIATE_MMHA_LAUNCHERS(T, Dh)                                                                              \
    template void mmha_launch_kernel<T, KVLinearBuffer, Masked_multihead_attention_params<T>, Dh, false, false,        \
        false>(const Masked_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,              \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVBlockArray, Masked_multihead_attention_params<T>, Dh, false, false, false>(  \
        const Masked_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                       \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVLinearBuffer, Cross_multihead_attention_params<T>, Dh, false, false, false>( \
        const Cross_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                      \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVBlockArray, Cross_multihead_attention_params<T>, Dh, false, false, false>(   \
        const Cross_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                        \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);

#define INSTANTIATE_MMHA_LAUNCHERS_WITH_IMPLICIT_REL_ATTN_BIAS(T, Dh)                                                  \
    template void mmha_launch_kernel<T, KVLinearBuffer, Masked_multihead_attention_params<T>, Dh, false, true, false>( \
        const Masked_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                     \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVBlockArray, Masked_multihead_attention_params<T>, Dh, false, true, false>(   \
        const Masked_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                       \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVLinearBuffer, Cross_multihead_attention_params<T>, Dh, false, true, false>(  \
        const Cross_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                      \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVBlockArray, Cross_multihead_attention_params<T>, Dh, false, true, false>(    \
        const Cross_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                        \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);

#define INSTANTIATE_MMHA_LAUNCHERS_WITH_ATTN_LOGIT_SOFTCAPPING_SCALE(T, Dh)                                            \
    template void mmha_launch_kernel<T, KVLinearBuffer, Masked_multihead_attention_params<T>, Dh, false, false, true>( \
        const Masked_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                     \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVBlockArray, Masked_multihead_attention_params<T>, Dh, false, false, true>(   \
        const Masked_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                       \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVLinearBuffer, Cross_multihead_attention_params<T>, Dh, false, false, true>(  \
        const Cross_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                      \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVBlockArray, Cross_multihead_attention_params<T>, Dh, false, false, true>(    \
        const Cross_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                        \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);

#define INSTANTIATE_MMHA_LAUNCHERS_WITH_BLOCK_SPARSE_ATTN(T, Dh)                                                       \
    template void mmha_launch_kernel<T, KVLinearBuffer, Masked_multihead_attention_params<T>, Dh, true, false, false>( \
        const Masked_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                     \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVBlockArray, Masked_multihead_attention_params<T>, Dh, true, false, false>(   \
        const Masked_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                       \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVLinearBuffer, Cross_multihead_attention_params<T>, Dh, true, false, false>(  \
        const Cross_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,                      \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);                                              \
    template void mmha_launch_kernel<T, KVBlockArray, Cross_multihead_attention_params<T>, Dh, true, false, false>(    \
        const Cross_multihead_attention_params<T>& params, const KVBlockArray& kv_cache_buffer,                        \
        const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);

} // namespace kernels
} // namespace tensorrt_llm
