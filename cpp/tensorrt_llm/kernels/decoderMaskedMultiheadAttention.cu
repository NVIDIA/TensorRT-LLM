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

#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

namespace tensorrt_llm
{
namespace kernels
{
namespace mmha
{
////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declaration of the kernel launcher to avoid including decoderMaskedMultiheadAttentionLaunch.h
template <typename T, typename KVCacheBuffer, typename T_PARAMS, int Dh, bool BLOCK_SPARSE_ATTN,
    bool IMPLICIT_REL_ATTN_BIAS, bool ATTN_LOGIT_SOFTCAPPING>
void mmha_launch_kernel(const T_PARAMS& params, KVCacheBuffer const& kv_cache_buffer,
    KVLinearBuffer const& shift_k_cache, cudaStream_t const& stream);

} // namespace mmha

namespace
{

#define MMHA_LAUNCH_KERNEL(Dh)                                                                                         \
    mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, false, false, false>(                           \
        params, kv_cache_buffer, shift_k_cache, stream);                                                               \
    break;

#define MMHA_LAUNCH_KERNE_EX1(Dh)                                                                                      \
    if (has_implicit_rel_attn_bias)                                                                                    \
    {                                                                                                                  \
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, false, true, false>(                        \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, false, false, false>(                       \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    break;

#define MMHA_LAUNCH_KERNE_EX2(Dh)                                                                                      \
    if (has_implicit_rel_attn_bias)                                                                                    \
    {                                                                                                                  \
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, false, true, false>(                        \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    else if (enable_attn_logit_softcapping)                                                                            \
    {                                                                                                                  \
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, false, false, true>(                        \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    else if (has_block_sparse_attn)                                                                                    \
    {                                                                                                                  \
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, true, false, false>(                        \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, false, false, false>(                       \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    break;

#define MMHA_LAUNCH_KERNE_EX3(Dh)                                                                                      \
    if (enable_attn_logit_softcapping)                                                                                 \
    {                                                                                                                  \
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, false, false, true>(                        \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh, false, false, false>(                       \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    break;

template <typename T, typename KVCacheBuffer, typename KERNEL_PARAMS_TYPE>
void multihead_attention_(const KERNEL_PARAMS_TYPE& params, KVCacheBuffer const& kv_cache_buffer,
    KVLinearBuffer const& shift_k_cache, cudaStream_t const& stream)
{
    bool const has_implicit_rel_attn_bias = params.max_distance > 0 && params.relative_attention_bias != nullptr;
    bool const enable_attn_logit_softcapping = params.attn_logit_softcapping_scale > 0.f;
    int const head_size = params.hidden_size_per_head;
    TLLM_CHECK_WITH_INFO(!has_implicit_rel_attn_bias || head_size == 32 || head_size == 64 || head_size == 128,
        "MMHA kernels haven't instantiate implicit_relative_attention_bias paths for head size %d.", head_size);
    TLLM_CHECK_WITH_INFO(!enable_attn_logit_softcapping || head_size == 128 || head_size == 256,
        "MMHA kernels haven't instantiate attn_logit_softcapping_scale paths for head size %d.", head_size);
    TLLM_CHECK_WITH_INFO(!(enable_attn_logit_softcapping && has_implicit_rel_attn_bias),
        "MMHA kernels haven't instantiate implicit_relative_attention_bias + attn_logit_softcapping_scale paths for "
        "head size %d.",
        head_size);

    bool const has_block_sparse_attn = params.block_sparse_attention;
    TLLM_CHECK_WITH_INFO(!has_block_sparse_attn || head_size == 128,
        "MMHA kernels were not instantiated for block_sparse_attention for head size %d.", head_size);
    TLLM_CHECK_WITH_INFO(!(has_implicit_rel_attn_bias && has_block_sparse_attn),
        "MMHA kernels do not support combining implicit_relative_attention_bias and block_sparse_attention");

    switch (params.hidden_size_per_head)
    {
    case 32: MMHA_LAUNCH_KERNE_EX1(32);
    case 64: MMHA_LAUNCH_KERNE_EX1(64);
    case 128: MMHA_LAUNCH_KERNE_EX2(128);
    case 256: MMHA_LAUNCH_KERNE_EX3(256);
#ifndef FAST_BUILD // skip mmha 48, 80, 96, 104, 112, 144, 160, 192 and 224 for fast build
    case 48: MMHA_LAUNCH_KERNEL(48);
    case 80: MMHA_LAUNCH_KERNEL(80);
    case 96: MMHA_LAUNCH_KERNEL(96);
    case 104: MMHA_LAUNCH_KERNEL(104);
    case 112: MMHA_LAUNCH_KERNEL(112);
    case 144: MMHA_LAUNCH_KERNEL(144);
    case 160: MMHA_LAUNCH_KERNEL(160);
    case 192: MMHA_LAUNCH_KERNEL(192);
    case 224: MMHA_LAUNCH_KERNEL(224);
#endif // FAST_BUILD
    default: TLLM_CHECK_WITH_INFO(false, "unsupported head_size %d", params.hidden_size_per_head);
    }
}

#undef MMHA_LAUNCH_KERNEL

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr std::array<int, 13> MMHA_SUPPORTED_HEAD_SIZES{
    32, 48, 64, 80, 96, 104, 112, 128, 144, 160, 192, 224, 256};

bool mmha_supported(int head_size)
{
    return std::find(MMHA_SUPPORTED_HEAD_SIZES.begin(), MMHA_SUPPORTED_HEAD_SIZES.end(), head_size)
        != MMHA_SUPPORTED_HEAD_SIZES.end();
}

#define INSTANTIATE_MMHA_NORMAL_AND_PAGED(T, CROSS_ATTENTION)                                                          \
    void masked_multihead_attention(const Multihead_attention_params<T, CROSS_ATTENTION>& params,                      \
        const KVBlockArray& kv_cache_buffer, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream)          \
    {                                                                                                                  \
        multihead_attention_<T, KVBlockArray, Multihead_attention_params<T, CROSS_ATTENTION>>(                         \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }                                                                                                                  \
    void masked_multihead_attention(const Multihead_attention_params<T, CROSS_ATTENTION>& params,                      \
        const KVLinearBuffer& kv_cache_buffer, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream)        \
    {                                                                                                                  \
        multihead_attention_<T, KVLinearBuffer, Multihead_attention_params<T, CROSS_ATTENTION>>(                       \
            params, kv_cache_buffer, shift_k_cache, stream);                                                           \
    }
INSTANTIATE_MMHA_NORMAL_AND_PAGED(float, true)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(float, false)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(uint16_t, true)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(uint16_t, false)
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_NORMAL_AND_PAGED(__nv_bfloat16, true)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(__nv_bfloat16, false)
#endif
#undef INSTANTIATE_MMHA_NORMAL_AND_PAGED

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels
} // namespace tensorrt_llm
