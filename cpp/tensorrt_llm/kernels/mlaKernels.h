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

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include <assert.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

enum class KvCacheDataType;

struct MlaMetaParams
{
    int32_t q_lora_rank = 0;
    int32_t kv_lora_rank = 0;
    int32_t qk_nope_head_dim = 0;
    int32_t qk_rope_head_dim = 0;
    int32_t v_head_dim = 0;
    int32_t predicted_tokens_per_seq = 1;
    int32_t num_layers = 0;

    auto data() const
    {
        return std::make_tuple(q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            predicted_tokens_per_seq, num_layers);
    }
};

template <typename T>
struct MlaParams
{
    T const* latent_cache;  // cKV + k_pe
    T* attention_input_buf; // [b, s, 3, h, d_h + r]
    void* quant_attention_input_buf;
    T* context_buf;
    T* q_pe;                     // [b, h, d_r], strided

    float2 const* cos_sin_cache; // [s, rope]
    int32_t batch_size;
    int32_t acc_q_len;
    int32_t head_num; // h
    void* workspace;
    int32_t const* cache_seq_lens;
    int* seqQOffset;
    uint32_t* fmha_tile_counter;
    int32_t max_input_seq_len;
    int* cu_q_seqlens;
    int* cu_kv_seqlens;
    int32_t q_pe_ld;
    int32_t q_pe_stride;
    MlaMetaParams meta;
    int const* block_ids_per_seq;
    KvCacheDataType cache_type;
    // Scales for mla quantization
    float* bmm1_scale;
    float* bmm2_scale;
    float const* quant_scale_o;
    float const* quant_scale_q;
    float const* quant_scale_kv;
    float const* dequant_scale_q;
    float const* dequant_scale_kv;
    float host_bmm1_scale;

    // for kv cache reuse/chunked context
    void* context_paged_kv_ptr = nullptr;
    void* context_kv_cache_block_offsets_ptr = nullptr;
    int32_t context_paged_kv_max_blocks_per_seq = 0;
    // for FP8 context qkv quantization
    float const* quant_scale_qkv = nullptr;
};

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

template <typename T, typename TCache>
void invokeMLALoadPagedKV(T* compressed_kv_ptr, T* k_pe_ptr, KVBlockArray& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len, int const lora_size, int const rope_size,
    float const* kv_scale_quant_orig_ptr, cudaStream_t stream);

template <typename T>
void invokeMLASetPagedKV(T* output, T const* k_ptr, T const* v_ptr, T const* k_pe_ptr, int const num_requests,
    int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads, int kv_dim, int rope_dim,
    int kv_cache_tokens_per_block, int64_t kv_token_stride, cudaStream_t stream);

template <typename T, typename TCache>
void invokeMLARopeAppendPagedKVAssignQ(KVBlockArray& kv_cache, T* q_ptr, T* latent_cache_ptr, int const num_requests,
    int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_uncached_seq_len,
    float2 const* cos_sin_cache, size_t head_num, int nope_size, int rope_size, int lora_size,
    float const* kv_scale_orig_quant_ptr, cudaStream_t stream);

template <typename T_IN>
__global__ void QuantizeCopyInputToFp8Kernel(
    T_IN const* input_buffer, __nv_fp8_e4m3* output_fp8_buffer, int num_total_elements, float const* device_scale_ptr);

} // namespace kernels
} // namespace tensorrt_llm
