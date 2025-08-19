#pragma once

namespace tensorrt_llm
{
namespace kernels
{

struct SparseAttentionParams
{
    int32_t* sparse_kv_indices{nullptr};   // [num_sparse_kv_indices, num_kv_heads]
    int32_t* sparse_attn_indices{nullptr}; // [num_sparse_attn_indices, num_kv_heads]
    int32_t* sparse_kv_offsets{nullptr};   // [num_contexts + 1]
    int32_t* sparse_attn_offsets{nullptr}; // [num_generations + 1]

    int32_t num_sparse_kv_indices{0};
    int32_t num_sparse_attn_indices{0};

    std::string toString() const
    {
        std::stringstream ss;
        ss << "num_sparse_kv_indices: " << this->num_sparse_kv_indices << std::endl;
        ss << "num_sparse_attn_indices: " << this->num_sparse_attn_indices << std::endl;
        ss << "sparse_kv_indices: " << this->sparse_kv_indices << std::endl;
        ss << "sparse_attn_indices: " << this->sparse_attn_indices << std::endl;
        ss << "sparse_kv_offsets: " << this->sparse_kv_offsets << std::endl;
        ss << "sparse_attn_offsets: " << this->sparse_attn_offsets << std::endl;
        return ss.str();
    }

    auto data() const
    {
        return std::make_tuple(sparse_kv_indices, sparse_attn_indices, sparse_kv_offsets, sparse_attn_offsets,
            num_sparse_kv_indices, num_sparse_attn_indices);
    }
};

} // namespace kernels
} // namespace tensorrt_llm
