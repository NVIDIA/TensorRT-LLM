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
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

struct MlaMetaParams
{
    int32_t q_lora_rank = 0;
    int32_t kv_lora_rank = 0;
    int32_t qk_nope_head_dim = 0;
    int32_t qk_rope_head_dim = 0;
    int32_t v_head_dim = 0;

    auto data() const
    {
        return std::make_tuple(q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    }
};

template <typename T>
struct MlaParams
{
    T const* fused_a_input;  // [b, s, c_q + c_k + r]
    T* attention_input_buf;  // [b, s, 3, h, d_h + r]
    T* context_buf;
    T* q_buf;                // [b, h, d_h + r]
    T const* q_b_proj;       // [(d_h + r) * h, c_q]
    T const* kv_b_proj;      // [h * d_h * 2, c_k]
    T const* k_b_proj_trans; // [h * c_k, d_h]
    float const* q_b_scale;
    float const* kv_b_scale;
    float const* k_b_trans_scale;
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
    MlaMetaParams meta;
};

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
