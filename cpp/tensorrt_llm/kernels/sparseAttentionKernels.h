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

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Parameters for sparse attention, controlling which KV tokens/pages each head attends to.
///
/// There are two independent sparse index paths:
///   1. Context phase (sparse_kv_indices/offsets): selects which KV tokens to compact into the
///      KV cache during prefill, reducing memory footprint.
///   2. Generation phase (sparse_attn_indices/offsets): selects which KV pages/blocks each head
///      attends to during token generation, reducing compute.
///
/// For sparse MLA (DeepSeek-style), sparse_mla_topk and sparse_mla_kv_cache_pool provide
/// direct access to the KV cache pool for the dedicated sparse MLA kernel.
struct SparseAttentionParams
{
    /// KV token indices for context-phase KV cache compaction.
    /// Shape: [total_sparse_tokens] (ragged, indexed by sparse_kv_offsets). The same indices
    /// are shared across all KV heads; context compaction does not use per-head index layout.
    int32_t* sparse_kv_indices{nullptr};

    /// Per-head KV page/block indices for generation-phase sparse computation.
    /// Shape: [num_kv_heads, total_sparse_blocks] (ragged, indexed by sparse_attn_offsets).
    /// Fed to trtllm-gen FMHA as per-head sparse page tables.
    int32_t* sparse_attn_indices{nullptr};

    /// Batch offsets into sparse_kv_indices. Shape: [num_contexts + 1].
    /// sparse_kv_indices for request i spans [sparse_kv_offsets[i], sparse_kv_offsets[i+1]).
    int32_t* sparse_kv_offsets{nullptr};

    /// Batch offsets into sparse_attn_indices. Shape: [num_generations + 1].
    /// sparse_attn_indices for request i spans [sparse_attn_offsets[i], sparse_attn_offsets[i+1]).
    int32_t* sparse_attn_offsets{nullptr};

    /// Number of KV tokens each query attends to in sparse MLA (DeepSeek-style).
    /// When > 0, enables the sparse MLA kernel path.
    int32_t sparse_mla_topk{0};

    /// Raw pointer to the KV cache memory pool for sparse MLA direct access.
    /// The sparse MLA kernel bypasses the normal paged KV cache indirection and reads
    /// KV data directly from this pool using sparse_attn_indices.
    void* sparse_mla_kv_cache_pool{nullptr};

    /// Granularity of sparse_attn_indices entries: 1 = token-level indices,
    /// N > 1 = block-level indices where each entry covers N tokens.
    int32_t sparse_attn_indices_block_size{1};

    /// Stride (in elements) between consecutive heads in the sparse_attn_indices tensor.
    /// When 0, heads are stored contiguously (stride = number of indices per head).
    int32_t sparse_attn_indices_stride{0};

    std::string toString() const
    {
        std::stringstream ss;
        ss << "sparse_kv_indices: " << this->sparse_kv_indices << std::endl
           << "sparse_attn_indices: " << this->sparse_attn_indices << std::endl
           << "sparse_kv_offsets: " << this->sparse_kv_offsets << std::endl
           << "sparse_attn_offsets: " << this->sparse_attn_offsets << std::endl
           << "sparse_mla_topk: " << this->sparse_mla_topk << std::endl
           << "sparse_mla_kv_cache_pool: " << this->sparse_mla_kv_cache_pool << std::endl
           << "sparse_attn_indices_block_size: " << this->sparse_attn_indices_block_size << std::endl
           << "sparse_attn_indices_stride: " << this->sparse_attn_indices_stride << std::endl;
        return ss.str();
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

TRTLLM_NAMESPACE_END
