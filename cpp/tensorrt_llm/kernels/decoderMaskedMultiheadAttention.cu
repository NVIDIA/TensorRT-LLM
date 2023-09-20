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

// Forward declaration of the kernel launcher to avoid including decoder_masked_multihead_attention_template.hpp
template <typename T, typename KVCacheBuffer, typename T_PARAMS, int Dh>
void mmha_launch_kernel(const T_PARAMS& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream);

} // namespace mmha

namespace
{

template <typename T, typename KVCacheBuffer, typename KERNEL_PARAMS_TYPE>
void multihead_attention_(
    const KERNEL_PARAMS_TYPE& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    switch (params.hidden_size_per_head)
    {
    case 32: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 32>(params, kv_cache_buffer, stream); break;
    case 48: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 48>(params, kv_cache_buffer, stream); break;
    case 64: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 64>(params, kv_cache_buffer, stream); break;
    case 80: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 80>(params, kv_cache_buffer, stream); break;
    case 96: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 96>(params, kv_cache_buffer, stream); break;
    case 128:
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 128>(params, kv_cache_buffer, stream);
        break;
    case 144:
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 144>(params, kv_cache_buffer, stream);
        break;
    case 160:
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 160>(params, kv_cache_buffer, stream);
        break;
    case 192:
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 192>(params, kv_cache_buffer, stream);
        break;
    case 224:
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 224>(params, kv_cache_buffer, stream);
        break;
    case 256:
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 256>(params, kv_cache_buffer, stream);
        break;
    default: TLLM_THROW("unsupported head_size");
    }
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params,
    const KVBlockArray& kv_cache_buffer, const cudaStream_t& stream)
{
    multihead_attention_<float, KVBlockArray, Masked_multihead_attention_params<float>>(
        params, kv_cache_buffer, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params,
    const KVBlockArray& kv_cache_buffer, const cudaStream_t& stream)
{
    multihead_attention_<uint16_t, KVBlockArray, Masked_multihead_attention_params<uint16_t>>(
        params, kv_cache_buffer, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
    const KVBlockArray& kv_cache_buffer, const cudaStream_t& stream)
{
    multihead_attention_<__nv_bfloat16, KVBlockArray, Masked_multihead_attention_params<__nv_bfloat16>>(
        params, kv_cache_buffer, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params,
    const KVLinearBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    multihead_attention_<float, KVLinearBuffer, Masked_multihead_attention_params<float>>(
        params, kv_cache_buffer, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params,
    const KVLinearBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    multihead_attention_<uint16_t, KVLinearBuffer, Masked_multihead_attention_params<uint16_t>>(
        params, kv_cache_buffer, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
    const KVLinearBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    multihead_attention_<__nv_bfloat16, KVLinearBuffer, Masked_multihead_attention_params<__nv_bfloat16>>(
        params, kv_cache_buffer, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels
} // namespace tensorrt_llm
