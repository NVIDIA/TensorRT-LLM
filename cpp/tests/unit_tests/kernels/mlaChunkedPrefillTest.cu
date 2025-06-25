#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include "tensorrt_llm/kernels/mlaChunkedPrefill.cuh"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

namespace
{
// kv_output {total_tokens, h=1, lora_size}
// k_pe_output {total_tokens, h=1, rope_size}
template <typename T>
void loadChunkedKVKernelRef(T* kv_output, T* k_pe_output, tensorrt_llm::kernels::KVBlockArray const& kv_cache,
    int num_contexts, int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_ctx_chunked_len, int const lora_size,
    int const rope_size, int const chunk_size, int const chunk_idx)
{
    int const head_size = lora_size + rope_size;
    for (int b = 0; b < num_contexts; b++)
    {
        int const kv_len = cu_ctx_cached_kv_lens[b + 1] - cu_ctx_cached_kv_lens[b];
        int const chunked_len = cu_ctx_chunked_len[b + 1] - cu_ctx_chunked_len[b];
        for (int s = 0; s < chunked_len; s++)
        {
            int const local_token_idx = chunk_idx * chunk_size + s;
            int const ld_token_offset = (cu_ctx_chunked_len[b] + s);

            auto const* kv_src = reinterpret_cast<T const*>(kv_cache.getKBlockPtr(b, local_token_idx));
            for (int d = 0; d < head_size; d++)
            {
                auto kv_block_idx = kv_cache.getKVLocalIdx(local_token_idx, 0, head_size, d);
                auto src_data = kv_src[kv_block_idx];

                if (d < lora_size)
                {
                    kv_output[ld_token_offset * lora_size + d] = src_data;
                }
                else
                {
                    k_pe_output[ld_token_offset * rope_size + (d - lora_size)] = src_data;
                }
            }
        }
    }
}

// kv {total_tokens, 2, h, nope_size}
// k_pe {total_tokens, h=1, rope_size}
// output {b, 2, ceil(max_seq / cache_tokens_per_block), h, cache_tokens_per_block, (nope_size + rope_size)}
// max_seq <= chunk_size
template <typename T>
void setChunkedKVCacheForMLAKernelRef(T* output, T* kv_ptr, T* k_pe_ptr, int num_contexts, int64_t const* cu_seq_len,
    int const max_input_seq_len, int num_heads, int nope_size, int rope_size, int cache_tokens_per_block)
{
    int head_size = nope_size + rope_size;
    int const kv_cache_size_per_block = num_heads * cache_tokens_per_block * head_size;
    int const kv_cache_block_num_per_seq = (max_input_seq_len + cache_tokens_per_block - 1) / cache_tokens_per_block;
    for (int b = 0; b < num_contexts; b++)
    {
        int const global_token_offset = cu_seq_len[b];
        int const current_seq_len = cu_seq_len[b + 1] - cu_seq_len[b];
        for (int s = 0; s < current_seq_len; s++)
        {
            int const global_token_idx = global_token_offset + s;
            int const kv_cache_block_offset_for_k
                = (b * 2 * kv_cache_block_num_per_seq + s / cache_tokens_per_block) * kv_cache_size_per_block;
            int const kv_cache_block_offset_for_v
                = kv_cache_block_offset_for_k + (kv_cache_block_num_per_seq * kv_cache_size_per_block);
            for (int h = 0; h < num_heads; h++)
            {
                int const ld_k_head_offset = (global_token_offset * 2 * num_heads * nope_size) + h * nope_size;
                int const ld_v_head_offset = ld_k_head_offset + num_heads * nope_size;
                int const ld_k_pe_head_offset = global_token_offset * rope_size;
                // copy kv
                for (int d = 0; d < nope_size; d++)
                {
                    int const ld_k_idx = ld_k_head_offset + d;
                    int const ld_v_idx = ld_v_head_offset + d;
                    int const st_k_idx = kv_cache_block_offset_for_k + h * cache_tokens_per_block * head_size
                        + s % cache_tokens_per_block * head_size + d;
                    int const st_v_idx = kv_cache_block_offset_for_v + h * cache_tokens_per_block * head_size
                        + s % cache_tokens_per_block * head_size + d;
                    output[st_k_idx] = kv_ptr[ld_k_idx];
                    output[st_v_idx] = kv_ptr[ld_v_idx];
                }

                // copy k_pe
                for (int d = 0; d < rope_size; d++)
                {
                    int const ld_k_pe_idx = ld_k_pe_head_offset + d;
                    int const st_k_pe_idx = kv_cache_block_offset_for_k + num_heads * cache_tokens_per_block * head_size
                        + h * cache_tokens_per_block * rope_size + s % cache_tokens_per_block * rope_size
                        + (rope_size + d);
                    output[st_k_pe_idx] = k_pe_ptr[ld_k_pe_idx];
                }
            }
        }
    }
}

// Q {total_q, H, D}
// KV {total_kv, 2, H, D}
// softmax_sum {total_q, H, 2} // {max/sum}
// output {total_q, H, D}
// total_q <= total_kv
template <typename T>
void selfAttentionRef(T* output, T* const Q, T* const KV, int batch_size, int num_heads, int64_t* const cu_seq_q_len,
    int64_t* const cu_seq_kv_len, int head_size, bool return_softmax, float* softmax_sum, bool causal_mask)
{
    int total_q_len = cu_seq_q_len[batch_size];
    int total_kv_len = cu_seq_kv_len[batch_size];

    for (int b = 0; b < batch_size; b++)
    {
        int curr_q_len = cu_seq_q_len[b + 1] - cu_seq_q_len[b];
        int curr_kv_len = cu_seq_kv_len[b + 1] - cu_seq_kv_len[b];
        int global_q_offset = cu_seq_q_len[b] * num_heads * head_size;
        int global_kv_offset = cu_seq_kv_len[b] * 2 * num_heads * head_size;
        int global_softmax_offset = cu_seq_q_len[b] * num_heads * 2;
        if (curr_q_len == 0 || curr_kv_len == 0)
        {
            continue; // skip empty sequences
        }
        std::vector<float> P(curr_q_len * curr_kv_len);
        for (int h = 0; h < num_heads; h++)
        {
            // BMM1
            std::fill(P.begin(), P.end(), std::numeric_limits<double>::lowest());
            T* const q_ptr = Q + global_q_offset + h * head_size;
            T* const k_ptr = KV + global_kv_offset + h * head_size;
            T* const v_ptr = k_ptr + num_heads * head_size;
            T* output_ptr = output + global_q_offset + h * head_size;
            for (int s_q = 0; s_q < curr_q_len; s_q++)
            {
                float softmax_max = std::numeric_limits<double>::lowest();
                for (int s_kv = 0; s_kv < curr_kv_len; s_kv++)
                {
                    // lower right mask
                    if (causal_mask && s_kv > curr_kv_len - curr_q_len + s_q)
                    {
                        break;
                    }
                    P[s_q * curr_kv_len + s_kv] = 0;
                    for (int d = 0; d < head_size; d++)
                    {
                        P[s_q * curr_kv_len + s_kv] += static_cast<float>(
                            q_ptr[s_q * num_heads * head_size + d] * k_ptr[s_kv * 2 * num_heads * head_size + d]);
                    }
                    if (softmax_max < P[s_q * curr_kv_len + s_kv])
                    {
                        softmax_max = P[s_q * curr_kv_len + s_kv];
                    }
                }
                for (int s_kv = 0; s_kv < curr_kv_len; s_kv++)
                {
                    // lower right mask
                    if (causal_mask && s_kv > curr_kv_len - curr_q_len + s_q)
                    {
                        break;
                    }
                    P[s_q * curr_kv_len + s_kv] -= softmax_max;
                }
                if (return_softmax)
                {
                    softmax_sum[global_softmax_offset + s_q * num_heads * 2 + h * 2] = softmax_max;
                }
            }
            // softmax
            for (int s_q = 0; s_q < curr_q_len; s_q++)
            {
                float sum = 0;
                for (int s_kv = 0; s_kv < curr_kv_len; s_kv++)
                {
                    P[s_q * curr_kv_len + s_kv] = std::exp(P[s_q * curr_kv_len + s_kv]);
                    sum += P[s_q * curr_kv_len + s_kv];
                }
                for (int s_kv = 0; s_kv < curr_kv_len; s_kv++)
                {
                    P[s_q * curr_kv_len + s_kv] /= sum;
                }
                if (return_softmax)
                {
                    softmax_sum[global_softmax_offset + s_q * num_heads * 2 + h * 2 + 1] = sum;
                }
            }
            // BMM2
            for (int s_q = 0; s_q < curr_q_len; s_q++)
            {
                for (int d = 0; d < head_size; d++)
                {
                    output_ptr[s_q * num_heads * head_size + d] = 0;
                    for (int s_kv = 0; s_kv < curr_kv_len; s_kv++)
                    {
                        output_ptr[s_q * num_heads * head_size + d] += static_cast<T>(P[s_q * curr_kv_len + s_kv]
                            * static_cast<float>(v_ptr[s_kv * 2 * num_heads * head_size + d]));
                    }
                }
            }
        }
    }
}

// chunked_KV {total_chunk_token, 2, H, D}
// KV {total_kv_token, 2, H, D}
template <typename T>
void copyRelatedChunkedKV(T* chunked_kv, T* const kv, int chunk_idx, int chunk_size, int batch_size, int num_heads,
    int64_t* const cu_kv_seq_len, int64_t* const cu_chunked_seq_len, int head_size)
{
    for (int b = 0; b < batch_size; b++)
    {
        int src_global_offset = (cu_kv_seq_len[b] + chunk_idx * chunk_size) * 2 * num_heads * head_size;
        int dst_global_offset = cu_chunked_seq_len[b] * 2 * num_heads * head_size;
        int copy_length = cu_chunked_seq_len[b + 1] - cu_chunked_seq_len[b];
        if (copy_length <= 0)
        {
            continue; // skip empty sequences
        }

        std::memcpy(chunked_kv + dst_global_offset, kv + src_global_offset,
            copy_length * 2 * num_heads * head_size * sizeof(T));
    }
}

// chunked_KV {total_chunk_token, 2, H, D}
// KV {total_kv_token, 2, H, D}
// It will copy the last chunk of KV cache to chunked_KV cache and calculate the cu_chunked_seq_len
template <typename T>
void copyFinalChunkedKV(T* chunked_kv, T* const kv, int chunk_size, int batch_size, int num_heads,
    int64_t* const cu_kv_seq_len, int64_t* cu_chunked_seq_len, int head_size, int64_t* merge_op)
{
    cu_chunked_seq_len[0] = 0;
    for (int b = 0; b < batch_size; b++)
    {
        int curr_kv_len = cu_kv_seq_len[b + 1] - cu_kv_seq_len[b];
        int last_chunk_size = curr_kv_len % chunk_size;
        if (last_chunk_size == 0)
        {
            last_chunk_size = chunk_size; // ensure at least one chunk
        }
        if (last_chunk_size == curr_kv_len)
        {
            merge_op[b] = 2; // no need to merge, just copy
        }
        else
        {
            merge_op[b] = 1;
        }
        cu_chunked_seq_len[b + 1] = cu_chunked_seq_len[b] + last_chunk_size;
        int global_token_offset = cu_kv_seq_len[b] + curr_kv_len - last_chunk_size;
        int copy_length = last_chunk_size;
        if (copy_length <= 0)
        {
            printf("copy_length is zero for batch %d, skipping...\n", b);
            continue; // skip empty sequences
        }
        int src_global_offset = global_token_offset * 2 * num_heads * head_size;
        int dst_global_offset = cu_chunked_seq_len[b] * 2 * num_heads * head_size;
        std::memcpy(chunked_kv + dst_global_offset, kv + src_global_offset,
            copy_length * 2 * num_heads * head_size * sizeof(T));
    }
}

template <typename WeightType>
float getTolerance(float scale = 1.f)
{
    float tol = 0.0;
    if constexpr (std::is_same_v<WeightType, uint8_t>)
    {
        tol = 0.1;
    }
    else if constexpr (std::is_same_v<WeightType, float>)
    {
        tol = 0.001;
    }
    else if constexpr (std::is_same_v<WeightType, half>)
    {
        tol = 0.005;
    }
    else if constexpr (std::is_same_v<WeightType, __nv_bfloat16>)
    {
        tol = 0.05;
    }
    // Keep the scale in a sane range
    return std::max(tol, scale * tol);
}
}; // namespace

template <typename _DataType>
class MlaChunkedPrefillTest : public ::testing::Test
{
protected:
    using DataType = _DataType;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    tensorrt_llm::runtime::BufferManager::ITensorPtr h_kv_cache_tensor{nullptr}, h_kv_cache_tensor_ref{nullptr},
        d_kv_cache_tensor{nullptr}, h_compressed_kv_cache_tensor{nullptr}, d_compressed_kv_cache_tensor{nullptr},
        h_compressed_offset_tensor{nullptr}, d_compressed_offset_tensor{nullptr}, h_cu_kv_seq_lens{nullptr},
        d_cu_kv_seq_lens{nullptr}, h_cu_chunk_lens{nullptr}, d_cu_chunk_lens{nullptr}, h_cu_q_seq_lens{nullptr},
        d_cu_q_seq_lens{nullptr},

        // for kernel 1
        h_compressed_kv_output{nullptr}, d_compressed_kv_output{nullptr}, h_k_pe_output{nullptr},
        d_k_pe_output{nullptr}, h_compressed_kv_output_ref{nullptr}, h_k_pe_output_ref{nullptr},

        // for kernel 2
        h_kv_tensor{nullptr}, d_kv_tensor{nullptr}, h_k_pe_tensor{nullptr}, d_k_pe_tensor{nullptr},

        // for merge attn {kv_full_tensor  = kv + k_pe}
        m_h_q_tensor{nullptr}, m_h_kv_full_tensor{nullptr}, m_h_chunked_kv_tensor{nullptr}, m_h_output_tensor{nullptr},
        m_h_softmax_sum_tensor{nullptr}, m_h_softmax_sum_accum_tensor{nullptr}, m_h_output_tensor_ref{nullptr},
        m_h_output_tensor_accum_ref{nullptr}, m_d_q_tensor{nullptr}, m_d_kv_full_tensor{nullptr},
        m_d_chunked_kv_tensor{nullptr}, m_d_output_tensor{nullptr}, m_d_softmax_sum_tensor{nullptr},
        m_d_softmax_sum_accum_tensor{nullptr}, m_d_output_tensor_ref{nullptr}, m_d_output_tensor_accum_ref{nullptr},
        m_h_merge_op{nullptr}, m_d_merge_op{nullptr};

    int mBatchSize{};
    int mMaxSeqLen{};
    int mMaxQSeqLen{};
    int mTotalQLen{};
    int mTotalKVLen{};
    int mChunkSize{};
    int mNumHeads{};
    int mLoraSize{};
    int mRopeSize{};
    int mNopeSize{};
    int mMaxGenLength{};
    // int mHeadSize{};
    int mTokensPerBlock{};
    int mMaxBlockPerSeq{};
    bool mIsCausalMask{};

    std::mt19937 gen;

    void SetUp() override
    {
        if (shouldSkip())
        {
            GTEST_SKIP() << "Skipping mla chunked prefill test";
        }
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        gen.seed(42U);
    }

    static bool shouldSkip()
    {
        return false;
    }

    void setDefaultParams()
    {
        mBatchSize = 2;
        // mMaxSeqLen = 128;
        mChunkSize = 32;
        mNumHeads = 16;
        mLoraSize = 512;
        mRopeSize = 64;
        mNopeSize = 128;
        mIsCausalMask = false;
        mMaxGenLength = 128;
        mTokensPerBlock = 16;
    }

    void memsetZeroHost(tensorrt_llm::runtime::BufferManager::ITensorPtr& tensor)
    {
        void* ptr = tensor->data();
        std::memset(ptr, 0, tensor->getSizeInBytes());
    }

    template <typename T>
    void showHostTensor(tensorrt_llm::runtime::BufferManager::ITensorPtr& tensor)
    {
        auto* const ptr = reinterpret_cast<T*>(tensor->data());
        for (int _ = 0; _ < tensor->getSize(); _++)
        {
            std::cout << static_cast<float>(ptr[_]) << " ";
        }
        std::cout << std::endl;
    }

    int generateRandomSizeSmallerThan(int a)
    {
        if (a <= 0)
        {
            return 0;
        }
        std::uniform_int_distribution<> distrib(0, a - 1);
        // Generate and return the random number
        return int{distrib(gen)};
    }

    float generateRandomFloat(float min, float max)
    {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);
    }

    template <typename T>
    void generateRandomData(T* data, int size)
    {
        for (int i = 0; i < size; i++)
        {
            data[i] = static_cast<T>(generateRandomFloat(-1.0f, 1.0f));
        }
    }

    template <typename T>
    void fillKVOffsetData(T* arr, size_t size, bool use_both_kv = true, int max_block_per_seq = 0)
    {
        if (use_both_kv)
        {
            for (int i = 0; i < size; i++)
            {
                arr[i] = static_cast<T>(i);
            }
        }
        else
        {
            int temp_idx = 0;
            for (int i = 0; i < size; i++)
            {
                bool is_v = (((i / max_block_per_seq) % 2) == 1);
                if (is_v)
                {
                    arr[i] = static_cast<T>(0);
                }
                else
                {
                    arr[i] = static_cast<T>(temp_idx);
                    temp_idx++;
                }
            }
        }
    }

    template <typename T>
    void fillArrayDataWithMod(T* arr, size_t size)
    {
        for (int i = 0; i < size; i++)
        {
            arr[i] = static_cast<T>(i % 448);
        }
    }

    bool allocateBuffers()
    {
        using tensorrt_llm::runtime::BufferManager;
        using tensorrt_llm::runtime::CudaStream;
        using tensorrt_llm::runtime::ITensor;
        using tensorrt_llm::runtime::bufferCast;

        auto dtype = nvinfer1::DataType::kHALF;
        if constexpr (std::is_same_v<DataType, float>)
        {
            dtype = nvinfer1::DataType::kFLOAT;
        }
        else if constexpr (std::is_same_v<DataType, half>)
        {
            dtype = nvinfer1::DataType::kHALF;
        }
        else if constexpr (std::is_same_v<DataType, __nv_bfloat16>)
        {
            dtype = nvinfer1::DataType::kBF16;
        }
        else
        {
            return false;
        }

        // cu lens
        this->h_cu_kv_seq_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize + 1}), nvinfer1::DataType::kINT64);
        this->h_cu_chunk_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize + 1}), nvinfer1::DataType::kINT64);
        this->h_cu_q_seq_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize + 1}), nvinfer1::DataType::kINT64);
        this->d_cu_kv_seq_lens = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_cu_kv_seq_lens->getShape(), nvinfer1::DataType::kINT64);
        this->d_cu_chunk_lens = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_cu_chunk_lens->getShape(), nvinfer1::DataType::kINT64);
        this->d_cu_q_seq_lens = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_cu_q_seq_lens->getShape(), nvinfer1::DataType::kINT64);
        {
            this->mMaxSeqLen = 0;
            this->mMaxQSeqLen = 0;
            this->mTotalQLen = 0;
            this->mTotalKVLen = 0;
            // we only initialize cu_seq_lens
            auto* cu_kv_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_kv_seq_lens));
            auto* cu_q_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_q_seq_lens));
            cu_kv_seq_lens_ptr[0] = 0;
            cu_q_seq_lens_ptr[0] = 0;
            for (int i = 0; i < this->mBatchSize; i++)
            {
                int temp_seq_len = this->generateRandomSizeSmallerThan(this->mMaxGenLength);
                if (temp_seq_len == 0)
                {
                    temp_seq_len = 1; // ensure at least one token
                }
                this->mMaxSeqLen = std::max(this->mMaxSeqLen, temp_seq_len);
                cu_kv_seq_lens_ptr[i + 1] = cu_kv_seq_lens_ptr[i] + temp_seq_len;
                auto temp_q_seq_len = temp_seq_len % this->mChunkSize;
                if (temp_q_seq_len == 0)
                {
                    temp_q_seq_len = this->mChunkSize; // ensure at least one chunk
                }
                cu_q_seq_lens_ptr[i + 1] = cu_q_seq_lens_ptr[i] + temp_q_seq_len;
                this->mMaxQSeqLen = std::max(this->mMaxQSeqLen, temp_q_seq_len);
                this->mTotalQLen += temp_q_seq_len;
                this->mTotalKVLen += temp_seq_len;
            }
            cudaMemcpy(this->d_cu_kv_seq_lens->data(), this->h_cu_kv_seq_lens->data(),
                this->h_cu_kv_seq_lens->getSizeInBytes(), cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_cu_q_seq_lens->data(), this->h_cu_q_seq_lens->data(),
                this->h_cu_q_seq_lens->getSizeInBytes(), cudaMemcpyHostToDevice);
        }
        // kv cache
        this->mMaxBlockPerSeq = (this->mMaxSeqLen + this->mTokensPerBlock - 1) / this->mTokensPerBlock;
        int maxChunkBlockPerSeq = (this->mChunkSize + this->mTokensPerBlock - 1) / this->mTokensPerBlock;
        this->h_kv_cache_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, 2, maxChunkBlockPerSeq, this->mNumHeads, this->mTokensPerBlock,
                this->mNopeSize + this->mRopeSize}),
            dtype);

        this->h_kv_cache_tensor_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, 2, maxChunkBlockPerSeq, this->mNumHeads, this->mTokensPerBlock,
                this->mNopeSize + this->mRopeSize}),
            dtype);

        this->h_compressed_kv_cache_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, 2, this->mMaxBlockPerSeq, this->mNumHeads, this->mTokensPerBlock,
                this->mLoraSize + this->mRopeSize}),
            dtype);
        this->h_compressed_offset_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, 2, this->mMaxBlockPerSeq + 1}), nvinfer1::DataType::kINT32);
        this->d_kv_cache_tensor
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_kv_cache_tensor->getShape(), dtype);
        this->d_compressed_kv_cache_tensor
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_compressed_kv_cache_tensor->getShape(), dtype);
        this->d_compressed_offset_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_compressed_offset_tensor->getShape(), nvinfer1::DataType::kINT32);

        {
            auto* compressed_kv_cache_ptr = bufferCast<DataType>(*(this->h_compressed_kv_cache_tensor));
            auto* offset_ptr = bufferCast<int32_t>(*(this->h_compressed_offset_tensor));

            this->memsetZeroHost(this->h_kv_cache_tensor);
            this->memsetZeroHost(this->h_kv_cache_tensor_ref);

            this->fillArrayDataWithMod(compressed_kv_cache_ptr, this->h_compressed_kv_cache_tensor->getSize());
            this->fillKVOffsetData(
                offset_ptr, this->h_compressed_offset_tensor->getSize(), false, this->mMaxBlockPerSeq);
            cudaMemcpy(this->d_kv_cache_tensor->data(), this->h_kv_cache_tensor->data(),
                this->h_kv_cache_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_compressed_kv_cache_tensor->data(), this->h_compressed_kv_cache_tensor->data(),
                this->h_compressed_kv_cache_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_compressed_offset_tensor->data(), this->h_compressed_offset_tensor->data(),
                this->h_compressed_offset_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
        }

        // tensor
        // kv, k_pe for invokeMLALoadChunkedKV (kernel 1)
        this->h_compressed_kv_output = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize * this->mChunkSize, 1, this->mLoraSize}), dtype);
        this->h_k_pe_output = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize * this->mChunkSize, 1, this->mRopeSize}), dtype);
        this->h_compressed_kv_output_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize * this->mChunkSize, 1, this->mLoraSize}), dtype);
        this->h_k_pe_output_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize * this->mChunkSize, 1, this->mRopeSize}), dtype);
        this->d_compressed_kv_output
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_compressed_kv_output->getShape(), dtype);
        this->d_k_pe_output = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_k_pe_output->getShape(), dtype);
        {
            this->memsetZeroHost(this->h_compressed_kv_output);
            this->memsetZeroHost(this->h_k_pe_output);
            this->memsetZeroHost(this->h_compressed_kv_output_ref);
            this->memsetZeroHost(this->h_k_pe_output_ref);

            cudaMemcpy(this->d_compressed_kv_output->data(), this->h_compressed_kv_output->data(),
                this->h_compressed_kv_output->getSizeInBytes(), cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_k_pe_output->data(), this->h_k_pe_output->data(), this->h_k_pe_output->getSizeInBytes(),
                cudaMemcpyHostToDevice);
        }

        // kv, k_pe for invokeMLASetChunkedKV (kernel 2)
        this->h_kv_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize * this->mChunkSize, 2, this->mNumHeads, this->mNopeSize}), dtype);
        this->h_k_pe_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize * this->mChunkSize, 1, this->mRopeSize}), dtype);
        this->d_kv_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_kv_tensor->getShape(), dtype);
        this->d_k_pe_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_k_pe_tensor->getShape(), dtype);
        {
            auto* kv_ptr = bufferCast<DataType>(*(this->h_kv_tensor));
            auto* k_pe_ptr = bufferCast<DataType>(*(this->h_k_pe_tensor));

            generateRandomData(kv_ptr, h_kv_tensor->getSize());
            generateRandomData(k_pe_ptr, h_k_pe_tensor->getSize());

            cudaMemcpyAsync(d_kv_tensor->data(), h_kv_tensor->data(), h_kv_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_k_pe_tensor->data(), h_k_pe_tensor->data(), h_k_pe_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
        }

        // invokeMergeAttnWithSoftmax, we just ignore rope_size here for simplicity

        this->m_h_q_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalQLen, this->mNumHeads, this->mNopeSize}), dtype);
        this->m_h_kv_full_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalKVLen, 2, this->mNumHeads, this->mNopeSize}), dtype);
        this->m_h_chunked_kv_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize * this->mChunkSize, 2, this->mNumHeads, this->mNopeSize}), dtype);
        this->m_h_output_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalQLen, this->mNumHeads, this->mNopeSize}), dtype);
        this->m_h_softmax_sum_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({2, this->mTotalQLen, this->mNumHeads}), nvinfer1::DataType::kFLOAT);
        this->m_h_softmax_sum_accum_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({2, this->mTotalQLen, this->mNumHeads}), nvinfer1::DataType::kFLOAT);
        this->m_h_output_tensor_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalQLen, this->mNumHeads, this->mNopeSize}), dtype);
        this->m_h_output_tensor_accum_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalQLen, this->mNumHeads, this->mNopeSize}), dtype);
        this->m_h_merge_op = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize}), nvinfer1::DataType::kINT64);
        this->m_d_q_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(this->m_h_q_tensor->getShape(), dtype);
        this->m_d_kv_full_tensor
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->m_h_kv_full_tensor->getShape(), dtype);
        this->m_d_chunked_kv_tensor
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->m_h_chunked_kv_tensor->getShape(), dtype);
        this->m_d_output_tensor
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->m_h_output_tensor->getShape(), dtype);
        this->m_d_softmax_sum_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->m_h_softmax_sum_tensor->getShape(), nvinfer1::DataType::kFLOAT);
        this->m_d_softmax_sum_accum_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->m_h_softmax_sum_accum_tensor->getShape(), nvinfer1::DataType::kFLOAT);
        this->m_d_output_tensor_ref
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->m_h_output_tensor_ref->getShape(), dtype);
        this->m_d_output_tensor_accum_ref
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->m_h_output_tensor_accum_ref->getShape(), dtype);
        this->m_d_merge_op
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->m_h_merge_op->getShape(), nvinfer1::DataType::kINT64);

        {
            auto* q_ptr = bufferCast<DataType>(*(this->m_h_q_tensor));
            auto* kv_ptr = bufferCast<DataType>(*(this->m_h_kv_full_tensor));

            generateRandomData(q_ptr, m_h_q_tensor->getSize());
            generateRandomData(kv_ptr, m_h_kv_full_tensor->getSize());
            this->memsetZeroHost(m_h_chunked_kv_tensor);
            this->memsetZeroHost(m_h_output_tensor);
            this->memsetZeroHost(m_h_softmax_sum_tensor);
            this->memsetZeroHost(m_h_softmax_sum_accum_tensor);
            this->memsetZeroHost(m_h_output_tensor_ref);
            this->memsetZeroHost(m_h_output_tensor_accum_ref);

            // Copy data to device
            cudaMemcpyAsync(m_d_q_tensor->data(), m_h_q_tensor->data(), m_h_q_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(m_d_kv_full_tensor->data(), m_h_kv_full_tensor->data(),
                m_h_kv_full_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(m_d_chunked_kv_tensor->data(), m_h_chunked_kv_tensor->data(),
                m_h_chunked_kv_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(m_d_output_tensor->data(), m_h_output_tensor->data(), m_h_output_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(m_d_softmax_sum_tensor->data(), m_h_softmax_sum_tensor->data(),
                m_h_softmax_sum_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(m_d_softmax_sum_accum_tensor->data(), m_h_softmax_sum_accum_tensor->data(),
                m_h_softmax_sum_accum_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(m_d_output_tensor_ref->data(), m_h_output_tensor_ref->data(),
                m_h_output_tensor_ref->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(m_d_output_tensor_accum_ref->data(), m_h_output_tensor_accum_ref->data(),
                m_h_output_tensor_accum_ref->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaStreamSynchronize(mStream->get());
        }
        return true;
    }

    void PerformNormalAttention()
    {
        using tensorrt_llm::runtime::bufferCast;

        auto* q_ptr = bufferCast<DataType>(*(this->m_h_q_tensor));
        auto* kv_ptr = bufferCast<DataType>(*(this->m_h_kv_full_tensor));
        auto* output_ptr = bufferCast<DataType>(*(this->m_h_output_tensor));
        auto* cu_q_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_q_seq_lens));
        auto* cu_kv_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_kv_seq_lens));
        selfAttentionRef(output_ptr, q_ptr, kv_ptr, this->mBatchSize, this->mNumHeads, cu_q_seq_lens_ptr,
            cu_kv_seq_lens_ptr, this->mNopeSize, false, nullptr, this->mIsCausalMask);
    }

    void PerformMergedAttention()
    {
        using tensorrt_llm::runtime::bufferCast;

        auto* h_q_ptr = bufferCast<DataType>(*(this->m_h_q_tensor));
        auto* h_kv_ptr = bufferCast<DataType>(*(this->m_h_kv_full_tensor));
        auto* h_chunked_kv_ptr = bufferCast<DataType>(*(this->m_h_chunked_kv_tensor));
        auto* h_output_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_ref));
        auto* h_output_accum_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_accum_ref));
        auto* h_softmax_sum_ptr = bufferCast<float>(*(this->m_h_softmax_sum_tensor));
        auto* h_softmax_sum_accum_ptr = bufferCast<float>(*(this->m_h_softmax_sum_accum_tensor));
        auto* h_cu_q_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_q_seq_lens));
        auto* h_cu_kv_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_kv_seq_lens));
        auto* h_cu_chunk_lens_ptr = bufferCast<int64_t>(*(this->h_cu_chunk_lens));
        auto* h_merge_op = bufferCast<int64_t>(*(this->m_h_merge_op));
        auto* d_kv_ptr = bufferCast<DataType>(*(this->m_d_kv_full_tensor));
        auto* d_chunked_kv_ptr = bufferCast<DataType>(*(this->m_d_chunked_kv_tensor));
        auto* d_softmax_sum_ptr = bufferCast<float>(*(this->m_d_softmax_sum_tensor));
        auto* d_softmax_sum_accum_ptr = bufferCast<float>(*(this->m_d_softmax_sum_accum_tensor));
        auto* d_output_ptr = bufferCast<DataType>(*(this->m_d_output_tensor_ref));
        auto* d_output_accum_ptr = bufferCast<DataType>(*(this->m_d_output_tensor_accum_ref));
        auto* d_merge_op = bufferCast<int64_t>(*(this->m_d_merge_op));
        auto* d_cu_q_seq_lens_ptr = bufferCast<int64_t>(*(this->d_cu_q_seq_lens));

        int const loop_count = (this->mMaxSeqLen + this->mChunkSize - 1) / this->mChunkSize;
        // do not apply mask
        for (int _ = 0; _ < loop_count - 1; _++)
        {
            // get chunked len for each request
            h_cu_chunk_lens_ptr[0] = 0;
            for (int b = 0; b < this->mBatchSize; b++)
            {
                int curr_kv_len = h_cu_kv_seq_lens_ptr[b + 1] - h_cu_kv_seq_lens_ptr[b];
                int used_kv_len = loop_count * this->mChunkSize;
                int curr_chunk_len = std::min(this->mChunkSize, curr_kv_len - used_kv_len);
                if (curr_chunk_len != this->mChunkSize)
                {
                    // last chunk, we should skip it.
                    curr_chunk_len = 0;
                }
                else
                {
                    if (used_kv_len + curr_chunk_len == curr_kv_len)
                    {
                        // last chunk, we should skip it.
                        curr_chunk_len = 0;
                    }
                }
                h_cu_chunk_lens_ptr[b + 1] = h_cu_chunk_lens_ptr[b] + curr_chunk_len;
                if (_ == 0 && curr_chunk_len > 0)
                {
                    h_merge_op[b] = 2; // only copy result
                }
                else if (curr_chunk_len > 0)
                {
                    h_merge_op[b] = 1; // merge result
                }
                else
                {
                    h_merge_op[b] = 0; // skip
                }
            }
            cudaMemcpy(d_merge_op, h_merge_op, this->m_h_merge_op->getSizeInBytes(), cudaMemcpyHostToDevice);
            // copy related kv chunk data
            copyRelatedChunkedKV(h_chunked_kv_ptr, h_kv_ptr, _, this->mChunkSize, this->mBatchSize, this->mNumHeads,
                h_cu_kv_seq_lens_ptr, h_cu_chunk_lens_ptr, this->mNopeSize);
            // attention
            selfAttentionRef<DataType>(h_output_ptr, h_q_ptr, h_chunked_kv_ptr, this->mBatchSize, this->mNumHeads,
                h_cu_q_seq_lens_ptr, h_cu_chunk_lens_ptr, this->mNopeSize, true, h_softmax_sum_ptr, false);
            // merge attention

            // copy curr_attn and softmax_sum to device
            cudaMemcpyAsync(d_softmax_sum_accum_ptr, h_softmax_sum_accum_ptr,
                this->m_h_softmax_sum_accum_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_softmax_sum_ptr, h_softmax_sum_ptr, this->m_h_softmax_sum_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_output_accum_ptr, h_output_accum_ptr, this->m_h_output_tensor_accum_ref->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            cudaMemcpyAsync(d_output_ptr, h_output_ptr, this->m_h_output_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice, mStream->get());
            sync_check_cuda_error(mStream->get());
            // merge softmax
            tensorrt_llm::kernels::invokeMergeAttnWithSoftmax<DataType>(d_output_accum_ptr, d_softmax_sum_accum_ptr,
                d_output_accum_ptr, d_softmax_sum_accum_ptr, d_output_ptr, d_softmax_sum_ptr, this->mBatchSize,
                d_cu_q_seq_lens_ptr, this->mMaxQSeqLen, d_merge_op, this->mNumHeads, this->mNopeSize, mStream->get());
            sync_check_cuda_error(mStream->get());
            // copy merged softmax sum back to host
            cudaMemcpyAsync(h_softmax_sum_accum_ptr, d_softmax_sum_accum_ptr,
                this->m_h_softmax_sum_tensor->getSizeInBytes(), cudaMemcpyDeviceToHost, mStream->get());
            cudaMemcpyAsync(h_output_accum_ptr, d_output_accum_ptr, this->m_h_output_tensor->getSizeInBytes(),
                cudaMemcpyDeviceToHost, mStream->get());
            sync_check_cuda_error(mStream->get());
        }
        // final round, apply causal mask.

        // copy the last chunked kv data
        copyFinalChunkedKV<DataType>(h_chunked_kv_ptr, h_kv_ptr, this->mChunkSize, this->mBatchSize, this->mNumHeads,
            h_cu_kv_seq_lens_ptr, h_cu_chunk_lens_ptr, this->mNopeSize, h_merge_op);
        // attention
        selfAttentionRef<DataType>(h_output_ptr, h_q_ptr, h_chunked_kv_ptr, this->mBatchSize, this->mNumHeads,
            h_cu_q_seq_lens_ptr, h_cu_chunk_lens_ptr, this->mNopeSize, true, h_softmax_sum_ptr, this->mIsCausalMask);
        // merge attention
        // copy curr_attn and softmax_sum to device
        cudaMemcpyAsync(d_softmax_sum_accum_ptr, h_softmax_sum_accum_ptr,
            this->m_h_softmax_sum_accum_tensor->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
        cudaMemcpyAsync(d_softmax_sum_ptr, h_softmax_sum_ptr, this->m_h_softmax_sum_tensor->getSizeInBytes(),
            cudaMemcpyHostToDevice, mStream->get());
        cudaMemcpyAsync(d_output_accum_ptr, h_output_accum_ptr, this->m_h_output_tensor_accum_ref->getSizeInBytes(),
            cudaMemcpyHostToDevice, mStream->get());
        cudaMemcpyAsync(d_output_ptr, h_output_ptr, this->m_h_output_tensor->getSizeInBytes(), cudaMemcpyHostToDevice,
            mStream->get());
        sync_check_cuda_error(mStream->get());
        tensorrt_llm::kernels::invokeMergeAttnWithSoftmax<DataType>(d_output_accum_ptr, d_softmax_sum_accum_ptr,
            d_output_accum_ptr, d_softmax_sum_accum_ptr, d_output_ptr, d_softmax_sum_ptr, this->mBatchSize,
            d_cu_q_seq_lens_ptr, this->mMaxQSeqLen, d_merge_op, this->mNumHeads, this->mNopeSize, mStream->get());
        sync_check_cuda_error(mStream->get());
        // copy merged softmax sum back to host
        cudaMemcpyAsync(h_softmax_sum_accum_ptr, d_softmax_sum_accum_ptr,
            this->m_h_softmax_sum_tensor->getSizeInBytes(), cudaMemcpyDeviceToHost, mStream->get());
        cudaMemcpyAsync(h_output_accum_ptr, d_output_accum_ptr, this->m_h_output_tensor->getSizeInBytes(),
            cudaMemcpyDeviceToHost, mStream->get());
        sync_check_cuda_error(mStream->get());
    }
};

using MLATypes = ::testing::Types<half, __nv_bfloat16, float>;
TYPED_TEST_SUITE(MlaChunkedPrefillTest, MLATypes);

TYPED_TEST(MlaChunkedPrefillTest, MlaChunkedPrefillDefault)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    this->setDefaultParams();
    this->allocateBuffers();

    sync_check_cuda_error(this->mStream->get());
    bool allEqual{true};

    this->PerformNormalAttention();
    sync_check_cuda_error(this->mStream->get());

    this->PerformMergedAttention();
    sync_check_cuda_error(this->mStream->get());

    // check result
    auto* output_ptr = bufferCast<DataType>(*(this->m_h_output_tensor));
    auto* output_ref_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_accum_ref));
    for (int i = 0; i < this->m_h_output_tensor->getSize(); i++)
    {
        if (std::abs(static_cast<float>(output_ptr[i]) - static_cast<float>(output_ref_ptr[i]))
            > getTolerance<DataType>(output_ptr[i]))
        {
            std::cout << "Output mismatch at index " << i << ": "
                      << "expected " << static_cast<float>(output_ref_ptr[i]) << ", got "
                      << static_cast<float>(output_ptr[i]) << std::endl;
            allEqual = false;
            break;
        }
    }
    ASSERT_TRUE(allEqual);
}

TYPED_TEST(MlaChunkedPrefillTest, MlaChunkedPrefillCausalMask)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    this->setDefaultParams();
    this->mIsCausalMask = true;
    this->allocateBuffers();

    sync_check_cuda_error(this->mStream->get());
    bool allEqual{true};

    this->PerformNormalAttention();
    sync_check_cuda_error(this->mStream->get());

    this->PerformMergedAttention();
    sync_check_cuda_error(this->mStream->get());

    // check result
    auto* output_ptr = bufferCast<DataType>(*(this->m_h_output_tensor));
    auto* output_ref_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_accum_ref));
    for (int i = 0; i < this->h_output_tensor->getSize(); i++)
    {
        if (std::abs(static_cast<float>(output_ptr[i]) - static_cast<float>(output_ref_ptr[i]))
            > getTolerance<DataType>(output_ptr[i]))
        {
            std::cout << "Output mismatch at index " << i << ": "
                      << "expected " << static_cast<float>(output_ref_ptr[i]) << ", got "
                      << static_cast<float>(output_ptr[i]) << std::endl;
            allEqual = false;
            break;
        }
    }
    ASSERT_TRUE(allEqual);
}
