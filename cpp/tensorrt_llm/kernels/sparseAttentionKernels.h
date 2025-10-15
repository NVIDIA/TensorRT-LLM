/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <tuple>

namespace tensorrt_llm
{
namespace kernels
{

struct SparseAttentionParams
{
    int32_t* sparse_kv_indices{nullptr};   // [num_kv_heads, num_sparse_kv_indices]
    int32_t* sparse_attn_indices{nullptr}; // [num_kv_heads, num_sparse_attn_indices]
    int32_t* sparse_kv_offsets{nullptr};   // [num_contexts + 1]
    int32_t* sparse_attn_offsets{nullptr}; // [num_generations + 1]

    std::string toString() const
    {
        std::stringstream ss;
        ss << "sparse_kv_indices: " << this->sparse_kv_indices << std::endl
           << "sparse_attn_indices: " << this->sparse_attn_indices << std::endl
           << "sparse_kv_offsets: " << this->sparse_kv_offsets << std::endl
           << "sparse_attn_offsets: " << this->sparse_attn_offsets << std::endl;
        return ss.str();
    }

    auto data() const
    {
        return std::make_tuple(sparse_kv_indices, sparse_attn_indices, sparse_kv_offsets, sparse_attn_offsets);
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

} // namespace kernels
} // namespace tensorrt_llm
