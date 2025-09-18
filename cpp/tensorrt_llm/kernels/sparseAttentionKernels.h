#pragma once

#include <cuda_runtime.h>
#include <sstream>

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

    // Scalars
    int32_t batch_size{0};
    int32_t num_head_kv{0};
    int32_t tokens_per_page{0};
    int32_t max_num_pages_per_seq{0};
    int32_t num_sparse_kv_tokens{0};

    std::string toString() const
    {
        std::stringstream ss;
        ss << "sparse_kv_indices: " << this->sparse_kv_indices << std::endl
           << "sparse_attn_indices: " << this->sparse_attn_indices << std::endl
           << "sparse_kv_offsets: " << this->sparse_kv_offsets << std::endl
           << "sparse_attn_offsets: " << this->sparse_attn_offsets << std::endl
           << "batch_size: " << this->batch_size << std::endl
           << "num_head_kv: " << this->num_head_kv << std::endl
           << "tokens_per_page: " << this->tokens_per_page << std::endl
           << "max_num_pages_per_seq: " << this->max_num_pages_per_seq << std::endl
           << "num_sparse_kv_tokens: " << this->num_sparse_kv_tokens << std::endl;
        return ss.str();
    }

    auto data() const
    {
        return std::make_tuple(sparse_kv_indices, sparse_attn_indices, sparse_kv_offsets, sparse_attn_offsets,
            batch_size, num_head_kv, tokens_per_page, max_num_pages_per_seq, num_sparse_kv_tokens);
    }
};

struct Pair
{
    int32_t max_val;
    int32_t sum_val;
};

struct PairReduceOp
{
    inline __device__ Pair operator()(Pair const& a, Pair const& b) const
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
    SparseAttentionParams sparse_params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
