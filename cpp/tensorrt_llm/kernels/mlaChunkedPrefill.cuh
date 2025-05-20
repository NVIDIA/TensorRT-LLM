#include "tensorrt_llm/kernels/kvCacheUtils.h"

namespace tensorrt_llm
{
namespace kernels
{
// merged_softmax_sum {2, B, S=chunked_token_size, H} (float), the first part is the softmax sum, the second part is the
// max value for each row of P = QK^T
template <typename T>
void invokeMergeAttnWithSoftmax(T* merged_attn, float* merged_softmax_sum, T* const pre_attn,
    float* const pre_softmax_sum, T* const curr_attn, float* const curr_softmax_sum, int const batch_size,
    int const chunked_token_size, int const num_heads, int const head_size, cudaStream_t stream);

// load single chunk kv from kv_cache for each request
template <typename T>
void invokeMLALoadChunkedKV(T* kv_output, KVBlockArray const& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int head_dim, int chunked_unit_size, int chunked_idx, cudaStream_t stream);

// output_kv {B, 2, H, S=chunked_token_size, D=uncompressed_h+rope_h}, padding with zero
// k, v {total_token, H, uncompressed_h=128}, k_pe {total_token, h=1, rope_h}
// input kv and k_pe can be cached tokens or uncached tokens
template <typename T>
void invokeMLASetChunkedKV(T* output_kv, T* const k, T* const v, T* const k_pe, int const batch_size,
    int const chunked_unit_size, int const num_heads, int uncompressed_head_size, int rope_size,
    int64_t* const cu_seq_lens, int const kv_cache_tokens_per_block, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
