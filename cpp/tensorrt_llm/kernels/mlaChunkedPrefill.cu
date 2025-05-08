#include "mlaChunkedPrefill.cuh"
#include "tensorrt_llm/common/assert.h"
#include <__clang_cuda_builtin_vars.h>
#include <cutlass/array.h>
#include <cutlass/half.h>

namespace
{

template <typename T>
struct MergeSoftmaxTraits
{
    static constexpr int kNumThreads = 128;
    static constexpr int kElemPerThread = 16 / sizeof(T);

    union VecReader
    {
        cutlass::Array<T, kElemPerThread> data;
        uint4 reader;
        static_assert(
            sizeof(uint4) == sizeof(cutlass::Array<T, kElemPerThread>), "Size mismatch for MergeSoftmaxTraits");
    };
};

template <typename ABC>
struct PrepareMLAContigousKVTraits
{
    using T = half;
    using VecT = uint4;
    static constexpr int kQKNopeSize = 128;
    static constexpr int kRopeSize = 64;
    static constexpr int kHeadSize = kQKNopeSize + kRopeSize;
    static constexpr int kBytesPerElem = sizeof(T);
    static constexpr int kBytesPerLoad = sizeof(VecT);
    static constexpr int kElemPerLoad = kBytesPerLoad / kBytesPerElem;
    static_assert((kHeadSize * kBytesPerElem) % kBytesPerLoad == 0,
        "kHeadSize * kBytesPerElem must be multiple of kBytesPerLoad (16Bytes)");
    static constexpr int kNumHeads = 128;
    static constexpr int kThreadPerHead = (kHeadSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kKVThreadPerHead = (kQKNopeSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kCpTokenPerBlock = 16;
    static constexpr int kBlockSize = kThreadPerHead * kCpTokenPerBlock;
};

template <typename ABC>
__global__ void mergeAttnWithSoftmaxKernel(ABC* merged_attn, float* merged_softmax_sum, ABC* const pre_attn,
    float* const pre_softmax_sum, ABC* const curr_attn, float* const curr_softmax_sum, int const batch_size,
    int const chunked_token_size, int const num_heads, int const head_size)
{
    using T = half;
    using KT = MergeSoftmaxTraits<T>;

    int const batch_idx = blockIdx.x;
    int const token_idx = blockIdx.y;
    static int const kNumHeadsPerBlock = KT::kElemPerThread * KT::kNumThreads / head_size;
    static int const kNumThreadsPerHead = head_size / KT::kElemPerThread;
    int const head_idx = (blockIdx.z * kNumHeadsPerBlock) + (threadIdx.x / kNumThreadsPerHead);
    int const dim_idx = (threadIdx.x % kNumThreadsPerHead) * KT::kElemPerThread;

    int const global_attn_offset = batch_idx * chunked_token_size * num_heads * head_size
        + token_idx * num_heads * head_size + head_idx * head_size;
    int const global_softmax_sum_offset = batch_idx * chunked_token_size * num_heads + token_idx * num_heads + head_idx;
    int const global_softmax_max_offset = global_softmax_sum_offset + batch_size * chunked_token_size * num_heads;
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
    KT::VecReader pre_attn_reader{};
    KT::VecReader curr_attn_reader{};
    KT::VecReader merged_attn_reader{};

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
    if (threadIdx.x % kNumThreadsPerHead == 0)
    {
        *merged_softmax_sum_ptr = merged_softmax_sum_val;
        *merged_softmax_max_ptr = merged_softmax_max_val;
    }
}

// output_kv {B, 2, H, S=chunked_token_size, D=uncompressed_h+rope_h}, padding with zero
// k, v {total_token, H=128, uncompressed_h=128}, k_pe {total_token, h=1, rope_h}
template <typename T>
__global__ void PrepareMLAContigousKV(T* output_kv, T* const k, T* const v, T* const k_pe, int const chunked_token_size,
    int const num_heads, int uncompressed_head_size, int rope_size, int64_t* const cu_seq_lens)
{
    using KT = PrepareMLAContigousKVTraits<T>;
    int const batch_idx = blockIdx.y;
    int const head_idx = blockIdx.z;
    int const head_dim_vec_idx = (threadIdx.x % KT::kThreadPerHead);
    int const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;
    bool const is_valid_kv = head_dim_idx < KT::kQKNopeSize;

    int64_t const global_token_offset = cu_seq_lens[batch_idx];
    // we assume current_valid_token_len is less than chunked_token_size
    int64_t const current_valid_token_len = cu_seq_lens[batch_idx + 1] - cu_seq_lens[batch_idx];
    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kCpTokenPerBlock;
         local_token_idx < current_valid_token_len; local_token_idx += gridDim.x * KT::kCpTokenPerBlock)
    {
        if (is_valid_kv)
        {
            int ld_kv_global_offset = (global_token_offset + local_token_idx) * num_heads * uncompressed_head_size
                + head_idx * uncompressed_head_size;
            int ld_kv_local_offset = head_dim_vec_idx;
            auto k_data = (reinterpret_cast<typename KT::VecT*>(k + ld_kv_global_offset))[ld_kv_local_offset];
            auto v_data = (reinterpret_cast<typename KT::VecT*>(v + ld_kv_global_offset))[ld_kv_local_offset];

            int st_k_global_offset
                = batch_idx * 2 * num_heads * chunked_token_size * (uncompressed_head_size + rope_size)
                + head_idx * chunked_token_size * (uncompressed_head_size + rope_size)
                + local_token_idx * (uncompressed_head_size + rope_size);
            int st_v_global_offset
                = st_k_global_offset + num_heads * chunked_token_size * (uncompressed_head_size + rope_size);
            int st_k_local_offset = head_dim_vec_idx;
            int st_v_local_offset = head_dim_vec_idx;
            (reinterpret_cast<typename KT::VecT*>(output_kv + st_k_global_offset))[st_k_local_offset] = k_data;
            (reinterpret_cast<typename KT::VecT*>(output_kv + st_v_global_offset))[st_v_local_offset] = v_data;
        }
        else
        {
            // rope h = 1
            int ld_rope_global_offset = (global_token_offset + local_token_idx) * rope_size;
            int ld_rope_local_offset = head_dim_vec_idx - KT::kKVThreadPerHead;
            auto rope_data = (reinterpret_cast<typename KT::VecT*>(k_pe + ld_rope_global_offset))[ld_rope_local_offset];
            int st_rope_global_offset
                = batch_idx * 2 * num_heads * chunked_token_size * (uncompressed_head_size + rope_size)
                + head_idx * chunked_token_size * (uncompressed_head_size + rope_size)
                + local_token_idx * (uncompressed_head_size + rope_size);
            int st_rope_local_offset = head_dim_vec_idx;
            (reinterpret_cast<typename KT::VecT*>(output_kv + st_rope_global_offset))[st_rope_local_offset] = rope_data;
        }
    }
}

} // namespace

// merged_attn [B, S, H=128, D=128] (T)
// merged_softmax_sum [2, B, S, H] (float), the first part is the softmax sum, the second part is the max value for each
// row of P = QK^T we just ignore the second part.
template <typename T>
void invokeMergeAttnWithSoftmax(T* merged_attn, float* merged_softmax_sum, T* const pre_attn,
    float* const pre_softmax_sum, T* const curr_attn, float* const curr_softmax_sum, int const batch_size,
    int const chunked_token_size, int const num_heads, int const head_size, cudaStream_t stream)
{

    using KT = MergeSoftmaxTraits<T>;
    // static const int kTHreadsPerHead = head_size / kElemPerThread;
    static int const kNumHeadsPerBlock = KT::kElemPerThread * KT::kNumThreads / head_size;

    dim3 grid(batch_size, chunked_token_size, (num_heads + kNumHeadsPerBlock - 1) / kNumHeadsPerBlock);
    dim3 block(KT::kNumThreads);

    mergeAttnWithSoftmaxKernel<T><<<grid, block, 0, stream>>>(merged_attn, merged_softmax_sum, pre_attn,
        pre_softmax_sum, curr_attn, curr_softmax_sum, batch_size, chunked_token_size, num_heads, head_size);
}

// output_kv {B, 2, H, S=chunked_token_size, D=uncompressed_h+rope_h}, padding with zero
// k, v {total_token, H=128, uncompressed_h=128}, k_pe {total_token, h=1, rope_h}
// input kv and k_pe can be cached tokens or uncached tokens
template <typename T>
void invokePrepareMLAContigousKV(T* output_kv, T* const k, T* const v, T* const k_pe, int const batch_size,
    int const chunked_token_size, int const num_heads, int uncompressed_head_size, int rope_size,
    int64_t* const cu_seq_lens, cudaStream_t stream)
{
    using KT = PrepareMLAContigousKVTraits<T>;
    TLLM_CHECK_WITH_INFO(
        uncompressed_head_size + rope_size == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    TLLM_CHECK_WITH_INFO(chunked_token_size % KT::kCpTokenPerBlock == 0, "chunked_token_size should be multiple of %d",
        KT::kCpTokenPerBlock);

    dim3 grid(chunked_token_size / KT::kCpTokenPerBlock, batch_size, num_heads);
    PrepareMLAContigousKV<T><<<grid, KT::kBlockSize, 0, stream>>>(
        output_kv, k, v, k_pe, chunked_token_size, num_heads, uncompressed_head_size, rope_size, cu_seq_lens);
}
