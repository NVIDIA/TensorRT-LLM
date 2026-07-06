/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

struct SparseAttentionParams
{
    int32_t* sparse_kv_indices{nullptr};   // [num_kv_heads, num_sparse_kv_indices]
    int32_t* sparse_attn_indices{nullptr}; // [num_kv_heads, num_sparse_attn_indices]
    int32_t* sparse_kv_offsets{nullptr};   // [num_contexts + 1]
    int32_t* sparse_attn_offsets{nullptr}; // [num_generations + 1]
    int32_t num_sparse_topk{0};            // topK for token sparse attention
    // Primary KV pool for sparse MQA/GQA. For dynamic sparse MLA this is the optional compressed
    // KV pool: nullptr for ratio == 1, compressed KV for ratio > 1.
    void* sparse_kv_cache_pool{nullptr};
    // SWA KV pool for dynamic sparse MLA. This is the host KV cache pool pointer for V4 and is
    // wired to trtllm-gen's sliding-window KV pool TMA descriptor.
    void* sliding_window_kv_cache_pool{nullptr};
    int32_t* sparse_mla_topk_lens{nullptr}; // [num_tokens]

    int32_t sparse_attn_indices_block_size{1};
    int32_t sparse_attn_indices_stride{0};

    std::string toString() const
    {
        std::stringstream ss;
        ss << "sparse_kv_indices: " << this->sparse_kv_indices << std::endl
           << "sparse_attn_indices: " << this->sparse_attn_indices << std::endl
           << "sparse_kv_offsets: " << this->sparse_kv_offsets << std::endl
           << "sparse_attn_offsets: " << this->sparse_attn_offsets << std::endl
           << "num_sparse_topk: " << this->num_sparse_topk << std::endl
           << "sparse_kv_cache_pool: " << this->sparse_kv_cache_pool << std::endl
           << "sliding_window_kv_cache_pool: " << this->sliding_window_kv_cache_pool << std::endl
           << "sparse_mla_topk_lens: " << this->sparse_mla_topk_lens << std::endl
           << "sparse_attn_indices_block_size: " << this->sparse_attn_indices_block_size << std::endl
           << "sparse_attn_indices_stride: " << this->sparse_attn_indices_stride << std::endl;
        return ss.str();
    }
};

struct CompactPseudoKvParams
{
    void const* key{nullptr};
    void const* value{nullptr};
    int32_t const* positions{nullptr};
    bool const* causal_mask{nullptr};
    int32_t compact_token_count{0};
    int32_t source_sequence_length{0};
    int32_t num_heads{0};
    int32_t head_size{0};
    int64_t key_stride_token_in_bytes{0};
    int64_t key_stride_head_in_bytes{0};
    int64_t value_stride_token_in_bytes{0};
    int64_t value_stride_head_in_bytes{0};

    bool isEnabled() const
    {
        return key != nullptr && value != nullptr && compact_token_count > 0;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "key: " << this->key << std::endl
           << "value: " << this->value << std::endl
           << "positions: " << this->positions << std::endl
           << "causal_mask: " << this->causal_mask << std::endl
           << "compact_token_count: " << this->compact_token_count << std::endl
           << "source_sequence_length: " << this->source_sequence_length << std::endl
           << "num_heads: " << this->num_heads << std::endl
           << "head_size: " << this->head_size << std::endl
           << "key_stride_token_in_bytes: " << this->key_stride_token_in_bytes << std::endl
           << "key_stride_head_in_bytes: " << this->key_stride_head_in_bytes << std::endl
           << "value_stride_token_in_bytes: " << this->value_stride_token_in_bytes << std::endl
           << "value_stride_head_in_bytes: " << this->value_stride_head_in_bytes << std::endl;
        return ss.str();
    }
};

struct CompactPseudoKvAttentionLaunchParams
{
    void const* query{nullptr};
    void* output{nullptr};
    CompactPseudoKvParams compact_pseudokv_params{};
    int32_t query_token_count{0};
    int64_t query_stride_token_in_bytes{0};
    int64_t query_stride_head_in_bytes{0};
    int64_t output_stride_token_in_bytes{0};
    int64_t output_stride_head_in_bytes{0};
    nvinfer1::DataType data_type{nvinfer1::DataType::kFLOAT};

    bool isEnabled() const
    {
        return query != nullptr && output != nullptr && query_token_count > 0 && compact_pseudokv_params.isEnabled();
    }
};

struct Pair
{
    int32_t max_val;
    int32_t sum_val;
};

struct PairReduceOp
{
#if defined(__CUDACC__)
    inline __device__
#endif
        Pair
        operator()(Pair const& a, Pair const& b) const
    {
        Pair result;
        result.max_val = a.max_val > b.max_val ? a.max_val : b.max_val;
        result.sum_val = a.sum_val + b.sum_val;
        return result;
    }
};

void invokeGatherKvPageOffsets(int32_t* output_kv_page_offsets, // [num_head_kv, batch_size, 2, max_num_pages_per_seq]
    int32_t* output_seq_lengths,                                // [num_head_kv, batch_size]
    int32_t const* kv_page_offsets,                             // [batch_size, 2, max_num_pages_per_seq]
    int32_t const* seq_lengths,                                 // [batch_size]
    SparseAttentionParams const sparse_params, int32_t const batch_size, int32_t const num_head_kv,
    int32_t const tokens_per_page, int32_t const max_num_pages_per_seq, cudaStream_t stream);

void invokeCompactPseudoKvAttention(CompactPseudoKvAttentionLaunchParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
