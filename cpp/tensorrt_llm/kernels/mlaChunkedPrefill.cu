#include "mlaChunkedPrefill.cuh"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/mathUtils.h"
#include <cutlass/array.h>
#include <cutlass/half.h>

namespace
{

template <typename T>
struct MergeSoftmaxTraits
{
    static constexpr int kNumThreads = 128;
    static constexpr int kElemPerThread = 16 / sizeof(T);
    static constexpr int kElemPerBlock = kNumThreads * kElemPerThread;

    union VecReader
    {
        cutlass::Array<T, kElemPerThread> data;
        uint4 reader;
        static_assert(
            sizeof(uint4) == sizeof(cutlass::Array<T, kElemPerThread>), "Size mismatch for MergeSoftmaxTraits");
    };
};

template <typename T>
struct loadChunkedKVKernelTraits
{
    static constexpr int kLoraSize = 512;
    static constexpr int kRopeSize = 64;
    static constexpr int kHeadSize = kLoraSize + kRopeSize;
    using VecT = uint4;
    static constexpr int kBytesPerElem = sizeof(T);
    static constexpr int kBytesPerLoad = 16;
    static constexpr int kElemPerLoad = kBytesPerLoad / kBytesPerElem;
    static_assert((kHeadSize * kBytesPerElem) % kBytesPerLoad == 0,
        "kHeadSize * kBytesPerElem must be multiple of kBytesPerLoad (16Bytes)");
    static constexpr int kVecPerHead = (kHeadSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kThreadPerHead = kVecPerHead; // for each head, we use kThreadPerHead threads to fetch all the
                                                       // kv cache data, each thread read kv cache only once.
    static constexpr int kTokenPerBlock
        = std::is_same_v<T, float> ? 4 : 8;            // for each block, we fetch 8 token for fp16, 4 tokens for fp32.
    static constexpr int kBlockSize = kThreadPerHead * kTokenPerBlock;
    static constexpr int kKVThreadPerHead = (kLoraSize * kBytesPerElem) / kBytesPerLoad;
};

template <typename T>
struct setChunkedKVKernelTraits
{
    using VecT = uint4;
    static constexpr int kQKNopeSize = 128;
    static constexpr int kVHeadSize = 128;
    static_assert(kQKNopeSize == kVHeadSize);
    static constexpr int kRopeSize = 64;
    static constexpr int kHeadSize = kQKNopeSize + kRopeSize;
    static constexpr int kBytesPerElem = sizeof(T);
    static constexpr int kBytesPerLoad = 16;
    static constexpr int kElemPerLoad = kBytesPerLoad / kBytesPerElem;
    static_assert((kHeadSize * kBytesPerElem) % kBytesPerLoad == 0,
        "kHeadSize * kBytesPerElem must be multiple of kBytesPerLoad (16Bytes)");
    static constexpr int kNumHeads = 128;
    static constexpr int kThreadPerHead = (kHeadSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kKVThreadPerHead = (kQKNopeSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kCpTokenPerBlock = 16;
    static constexpr int kBlockSize = kThreadPerHead * kCpTokenPerBlock;
};

template <typename T>
__global__ void mergeAttnWithSoftmaxKernel(T* merged_attn, float* merged_softmax_sum, T* const pre_attn,
    float* const pre_softmax_sum, T* const curr_attn, float* const curr_softmax_sum, int const batch_size,
    int const chunked_token_size, int const num_heads, int const head_size)
{
    using KT = MergeSoftmaxTraits<T>;
    int const batch_idx = blockIdx.x;
    int const token_idx = blockIdx.y;
    int const CurrentElementIdx = (blockIdx.z * KT::kElemPerBlock) + threadIdx.x * KT::kElemPerThread;

    int const head_idx = CurrentElementIdx / head_size;
    int const dim_idx = CurrentElementIdx % head_size;

    int const global_attn_offset = batch_idx * chunked_token_size * num_heads * head_size
        + token_idx * num_heads * head_size + head_idx * head_size;
    int const global_softmax_sum_offset
        = (((batch_idx * chunked_token_size * num_heads) + (token_idx * num_heads) + head_idx) * 2) + 1;
    int const global_softmax_max_offset = global_softmax_sum_offset - 1;
    auto* merged_attn_ptr = merged_attn + global_attn_offset;
    auto* merged_softmax_sum_ptr = merged_softmax_sum + global_softmax_sum_offset;
    auto* merged_softmax_max_ptr = merged_softmax_sum + global_softmax_max_offset;
    auto* pre_attn_ptr = pre_attn + global_attn_offset;
    auto* pre_softmax_sum_ptr = pre_softmax_sum + global_softmax_sum_offset;
    auto* pre_softmax_max_ptr = pre_softmax_sum + global_softmax_max_offset;
    auto* curr_attn_ptr = curr_attn + global_attn_offset;
    auto* curr_softmax_sum_ptr = curr_softmax_sum + global_softmax_sum_offset;
    auto* curr_softmax_max_ptr = curr_softmax_sum + global_softmax_max_offset;

    float pre_softmax_sum_val = *pre_softmax_sum_ptr;
    float curr_softmax_sum_val = *curr_softmax_sum_ptr;
    float pre_softmax_max_val = *pre_softmax_max_ptr;
    float curr_softmax_max_val = *curr_softmax_max_ptr;
    // merge softmax sum
    float merged_softmax_max_val = fmaxf(pre_softmax_max_val, curr_softmax_max_val);
    float pre_shift = std::exp(pre_softmax_max_val - merged_softmax_max_val);
    float curr_shift = std::exp(curr_softmax_max_val - merged_softmax_max_val);
    float merged_softmax_sum_val = (pre_softmax_sum_val * pre_shift) + (curr_softmax_sum_val * curr_shift);

    // merge softmax
    typename KT::VecReader pre_attn_reader{};
    typename KT::VecReader curr_attn_reader{};
    typename KT::VecReader merged_attn_reader{};
    if (head_idx >= num_heads || dim_idx >= head_size)
    {
        return;
    }
    pre_attn_reader.reader = *reinterpret_cast<decltype(pre_attn_reader.reader)*>(pre_attn_ptr + dim_idx);
    curr_attn_reader.reader = *reinterpret_cast<decltype(curr_attn_reader.reader)*>(curr_attn_ptr + dim_idx);

    for (int i = 0; i < KT::kElemPerThread; ++i)
    {
        merged_attn_reader.data[i]
            = (static_cast<float>(pre_attn_reader.data[i]) * pre_softmax_sum_val * pre_shift
                  + static_cast<float>(curr_attn_reader.data[i]) * curr_softmax_sum_val * curr_shift)
            / merged_softmax_sum_val;
    }
    // write merged attn back to global memory
    *reinterpret_cast<decltype(merged_attn_reader.reader)*>(merged_attn_ptr + dim_idx) = merged_attn_reader.reader;

    // write merged softmax sum and max back to global memory
    if (dim_idx == 0)
    {
        *merged_softmax_sum_ptr = merged_softmax_sum_val;
        *merged_softmax_max_ptr = merged_softmax_max_val;
    }
}

// kv_output {b, chunked_unit_size, h=1, d_lora}
// k_pe_output {b, chunked_unit_size, h=1, d_rope}
template <typename T>
__global__ void loadChunkedKVCacheForMLAKernel(T* output_kv_ptr, T* output_k_pe_ptr,
    const tensorrt_llm::kernels::KVBlockArray kv_cache, int64_t const* cu_ctx_cached_kv_lens, int chunked_unit_size,
    int chunked_idx)
{
    using KT = loadChunkedKVKernelTraits<T>;
    int const batch_idx = static_cast<int>(blockIdx.y);
    int const head_idx = static_cast<int>(blockIdx.z); // default 0

    size_t const head_dim_vec_idx = (threadIdx.x % KT::kVecPerHead);
    size_t const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;

    int64_t const cache_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];
    bool const is_valid_kv = head_dim_vec_idx < KT::kKVThreadPerHead;
    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kTokenPerBlock;
         local_token_idx < chunked_unit_size; local_token_idx += gridDim.x * KT::kTokenPerBlock)
    {
        int token_idx_in_kv_cache = (chunked_idx * chunked_unit_size) + local_token_idx;
        bool const valid_token = (local_token_idx < chunked_unit_size) && (token_idx_in_kv_cache < cache_kv_len);
        if (valid_token)
        {
            auto* kvSrc = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
            // head_idx === 0
            auto kvBlockIdx
                = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, KT::kVecPerHead, static_cast<int>(head_dim_vec_idx));
            auto ld_data = (reinterpret_cast<typename KT::VecT*>(kvSrc))[kvBlockIdx];
            if (is_valid_kv)
            {
                // kv_output {b, chunked_unit_size, h=1, d}
                int const global_st_idx
                    = batch_idx * chunked_unit_size * KT::kLoraSize + local_token_idx * KT::kLoraSize + head_dim_idx;
                *reinterpret_cast<typename KT::VecT*>(output_kv_ptr + global_st_idx) = ld_data;
            }
            else
            {
                // k_pe_output {b, chunked_unit_size, h=1, d_rope}
                int const global_st_idx = batch_idx * chunked_unit_size * KT::kRopeSize
                    + local_token_idx * KT::kRopeSize + (head_dim_idx - KT::kLoraSize);
                *reinterpret_cast<typename KT::VecT*>(output_k_pe_ptr + global_st_idx) = ld_data;
            }
        }
    }
}

// in the most of cases, chunk_size =  max_seq_len
// output_kv {B, 2, ceil(max_seq_len / kv_cache_tokens_per_block), h, kv_cache_tokens_per_block, d}, padding with
// zero
// kv {token_size = B*chunked_unit_size, 2, H=128, uncompressed_h=128}, k_pe {token_size = B*chunked_unit_size, h=1,
// rope_h}
// cu_seq_lens {batch + 1}, fake cu_seq_len, for chunked prefill is {0, chunk_size, chunk_size * 2 ....}
template <typename T>
__global__ void setChunkedKVCacheForMLAKernel(T* output_kv, T* const kv, T* const k_pe, int const max_seq_len,
    int const num_heads, int uncompressed_head_size, int rope_size, int64_t* const cu_seq_lens,
    int kv_cache_tokens_per_block)
{
    using KT = setChunkedKVKernelTraits<T>;
    int const batch_idx = blockIdx.y;
    int const head_idx = blockIdx.z;
    int const head_dim_vec_idx = (threadIdx.x % KT::kThreadPerHead);
    int const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;
    bool const is_valid_kv = head_dim_idx < KT::kQKNopeSize;

    size_t const seq_len_loop_end
        = (max_seq_len + KT::kCpTokenPerBlock - 1) / KT::kCpTokenPerBlock * KT::kCpTokenPerBlock;
    int64_t const global_token_offset = cu_seq_lens[batch_idx];
    int64_t const cache_kv_len = cu_seq_lens[batch_idx + 1] - cu_seq_lens[batch_idx];
    int const kv_cache_block_num = (max_seq_len + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block;
    int const kv_cache_block_size = num_heads * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size);
    int const offset_for_kv_in_mem_pool = kv_cache_block_num * kv_cache_block_num;
    int64_t const kv_offset = num_heads * uncompressed_head_size;
    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kCpTokenPerBlock;
         local_token_idx < seq_len_loop_end; local_token_idx += gridDim.x * KT::kCpTokenPerBlock)
    {
        if (local_token_idx >= cache_kv_len)
        {
            break;
        }
        if (is_valid_kv)
        {

            int64_t ld_kv_global_offset
                = (global_token_offset + local_token_idx) * 2 * num_heads * uncompressed_head_size
                + head_idx * uncompressed_head_size;
            int64_t ld_kv_local_offset = head_dim_vec_idx;
            auto k_data = (reinterpret_cast<typename KT::VecT*>(kv + ld_kv_global_offset))[ld_kv_local_offset];
            auto v_data
                = (reinterpret_cast<typename KT::VecT*>(kv + kv_offset + ld_kv_global_offset))[ld_kv_local_offset];

            int64_t st_k_global_offset = batch_idx * 2 * offset_for_kv_in_mem_pool
                + (local_token_idx + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block * kv_cache_block_size
                + head_idx * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size)
                + (local_token_idx % kv_cache_tokens_per_block) * (uncompressed_head_size + rope_size);
            int64_t st_v_global_offset = st_k_global_offset + offset_for_kv_in_mem_pool;
            int64_t st_k_local_offset = head_dim_vec_idx;
            int64_t st_v_local_offset = head_dim_vec_idx;
            (reinterpret_cast<typename KT::VecT*>(output_kv + st_k_global_offset))[st_k_local_offset] = k_data;
            (reinterpret_cast<typename KT::VecT*>(output_kv + st_v_global_offset))[st_v_local_offset] = v_data;
        }
        else
        {
            // rope h = 1
            int64_t ld_rope_global_offset = (global_token_offset + local_token_idx) * rope_size;
            int64_t ld_rope_local_offset = head_dim_vec_idx - KT::kKVThreadPerHead;
            auto rope_data = (reinterpret_cast<typename KT::VecT*>(k_pe + ld_rope_global_offset))[ld_rope_local_offset];
            int64_t st_rope_global_offset = batch_idx * 2 * offset_for_kv_in_mem_pool
                + (local_token_idx + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block * kv_cache_block_size
                + head_idx * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size)
                + (local_token_idx % kv_cache_tokens_per_block) * (uncompressed_head_size + rope_size);
            int64_t st_rope_local_offset = head_dim_vec_idx;
            (reinterpret_cast<typename KT::VecT*>(output_kv + st_rope_global_offset))[st_rope_local_offset] = rope_data;
        }
    }
}

} // namespace

namespace tensorrt_llm
{
namespace kernels
{

// merged_attn [B, S, H=128, D=128] (T)
// merged_softmax_sum [B, S, H, 2] (float), the first part is the softmax sum, the second part is the max value for each
// row of P = QK^T we just ignore the second part.
template <typename T>
void invokeMergeAttnWithSoftmax(T* merged_attn, float* merged_softmax_sum, T* const pre_attn,
    float* const pre_softmax_sum, T* const curr_attn, float* const curr_softmax_sum, int const batch_size,
    int const chunked_token_size, int const num_heads, int const head_size, cudaStream_t stream)
{

    using KT = MergeSoftmaxTraits<T>;
    static int const kNumHeadIterNeeded = (num_heads * head_size + KT::kElemPerBlock - 1) / KT::kElemPerThread;

    dim3 grid(batch_size, chunked_token_size, kNumHeadIterNeeded);
    dim3 block(KT::kNumThreads);

    mergeAttnWithSoftmaxKernel<T><<<grid, block, 0, stream>>>(merged_attn, merged_softmax_sum, pre_attn,
        pre_softmax_sum, curr_attn, curr_softmax_sum, batch_size, chunked_token_size, num_heads, head_size);
}

// load single chunk kv from kv_cache for each request
template <typename T>
void invokeMLALoadChunkedKV(T* output_kv_ptr, T* output_k_pe_ptr, KVBlockArray const& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int lora_size, int rope_size, int chunked_unit_size, int chunked_idx,
    cudaStream_t stream)
{
    using KT = loadChunkedKVKernelTraits<T>;
    TLLM_CHECK_WITH_INFO(lora_size + rope_size == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    TLLM_CHECK_WITH_INFO(lora_size == KT::kLoraSize, "lora dim should be equal to %d", KT::kLoraSize);
    TLLM_CHECK_WITH_INFO(rope_size == KT::kRopeSize, "rope dim should be equal to %d", KT::kRopeSize);
    // {chunked_unit_size / token_per_block, batch_size, head_num}
    dim3 grid(static_cast<int>(tensorrt_llm::common::divUp(chunked_unit_size, KT::kTokenPerBlock)), num_contexts, 1);
    loadChunkedKVCacheForMLAKernel<T><<<grid, KT::kBlockSize, 0, stream>>>(
        output_kv_ptr, output_k_pe_ptr, kv_cache, cu_ctx_cached_kv_lens, chunked_unit_size, chunked_idx);
}

// output_kv {B, 2, ceil(chunked_unit_size / kv_cache_tokens_per_block), h, kv_cache_tokens_per_block, d}, padding with
// zero
// kv {total_token, 2, H, uncompressed_h=128} 0 for k and 1 for v, k_pe {total_token, h=1, rope_h}
// input kv and k_pe can be cached tokens or uncached tokens
template <typename T>
void invokeMLASetChunkedKV(T* output_kv, T* const kv, T* const k_pe, int const batch_size, int const max_seq_len,
    int const num_heads, int uncompressed_head_size, int rope_size, int64_t* const cu_seq_lens,
    int const kv_cache_tokens_per_block, cudaStream_t stream)
{
    using KT = setChunkedKVKernelTraits<T>;
    TLLM_CHECK_WITH_INFO(
        uncompressed_head_size + rope_size == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    TLLM_CHECK_WITH_INFO(kv_cache_tokens_per_block % KT::kCpTokenPerBlock == 0,
        "kv_cache_tokens_per_block should be multiple of %d", KT::kCpTokenPerBlock);

    dim3 grid(tensorrt_llm::common::divUp(max_seq_len, KT::kCpTokenPerBlock), batch_size, num_heads);
    setChunkedKVCacheForMLAKernel<T><<<grid, KT::kBlockSize, 0, stream>>>(output_kv, kv, k_pe, max_seq_len, num_heads,
        uncompressed_head_size, rope_size, cu_seq_lens, kv_cache_tokens_per_block);
}

#define INSTANTIATE_MLA_CHUNKED_PREFILL_KERNEL(T)                                                                      \
    template void invokeMergeAttnWithSoftmax<T>(T * merged_attn, float* merged_softmax_sum, T* const pre_attn,         \
        float* const pre_softmax_sum, T* const curr_attn, float* const curr_softmax_sum, int const batch_size,         \
        int const chunked_token_size, int const num_heads, int const head_size, cudaStream_t stream);                  \
    template void invokeMLALoadChunkedKV<T>(T * output_kv_ptr, T * output_k_pe_ptr, KVBlockArray const& kv_cache,      \
        int const num_contexts, int64_t const* cu_ctx_cached_kv_lens, int lora_size, int rope_size,                    \
        int chunked_unit_size, int chunked_idx, cudaStream_t stream);                                                  \
    template void invokeMLASetChunkedKV<T>(T * output_kv, T* const kv, T* const k_pe, int const batch_size,            \
        int const max_seq_len, int const num_heads, int uncompressed_head_size, int rope_size,                         \
        int64_t* const cu_seq_lens, int const kv_cache_tokens_per_block, cudaStream_t stream);

INSTANTIATE_MLA_CHUNKED_PREFILL_KERNEL(half);
INSTANTIATE_MLA_CHUNKED_PREFILL_KERNEL(float);
INSTANTIATE_MLA_CHUNKED_PREFILL_KERNEL(__nv_bfloat16);
} // namespace kernels
} // namespace tensorrt_llm
