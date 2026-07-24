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

// #define TRTLLM_MLA_CHUNKED_PREFILL_TEST_DBG

namespace
{
// kv_output {total_tokens, h=1, lora_size}
// k_pe_output {total_tokens, h=1, rope_size}
template <typename T, typename TCache>
void loadChunkedKVKernelRef(T* kv_output, T* k_pe_output, tensorrt_llm::kernels::KVBlockArray const& kv_cache,
    int num_contexts, int64_t const* cu_ctx_chunked_len, int64_t const* chunked_ld_global_offset, int const lora_size,
    int const rope_size, float const* kv_scale_quant_orig_ptr)
{
    int const head_size = lora_size + rope_size;
    float const kv_scale_quant_orig = kv_scale_quant_orig_ptr ? kv_scale_quant_orig_ptr[0] : 1.0f;
    for (int b = 0; b < num_contexts; b++)
    {
        int const chunked_len = cu_ctx_chunked_len[b + 1] - cu_ctx_chunked_len[b];
        for (int s = 0; s < chunked_len; s++)
        {
            int const local_token_idx = chunked_ld_global_offset[b] + s;
            int const st_token_offset = (cu_ctx_chunked_len[b] + s);

            auto const* kv_src = reinterpret_cast<TCache const*>(kv_cache.getKBlockPtr(b, local_token_idx));
            for (int d = 0; d < head_size; d++)
            {
                auto kv_block_idx = kv_cache.getKVLocalIdx(local_token_idx, 0, head_size, d);
                auto src_data = kv_src[kv_block_idx];
                T data;
                if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    data = T(float(src_data) * kv_scale_quant_orig);
                }
                else
                {
                    data = src_data;
                }
                if (d < lora_size)
                {
                    kv_output[st_token_offset * lora_size + d] = data;
                }
                else
                {
                    k_pe_output[st_token_offset * rope_size + (d - lora_size)] = data;
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
                    // P[s_q * curr_kv_len + s_kv] = std::exp(P[s_q * curr_kv_len + s_kv]);
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
void copyRelatedChunkedKV(T* chunked_kv, T* const kv, int64_t const* chunked_ld_global_offset, int batch_size,
    int num_heads, int64_t* const cu_kv_seq_len, int64_t* const cu_chunked_seq_len, int head_size)
{
    for (int b = 0; b < batch_size; b++)
    {
        int src_global_offset = (cu_kv_seq_len[b] + chunked_ld_global_offset[b]) * 2 * num_heads * head_size;
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

template <typename Typepair>
class MlaChunkedPrefillTest : public ::testing::Test
{
protected:
    using DataType = typename Typepair::first_type;
    using TCache = typename Typepair::second_type;
    static_assert(std::is_same_v<DataType, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as DataType or __nv_fp8_e4m3");
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    tensorrt_llm::runtime::BufferManager::ITensorPtr h_kv_cache_tensor{nullptr}, h_kv_cache_tensor_ref{nullptr},
        d_kv_cache_tensor{nullptr}, h_compressed_kv_cache_tensor{nullptr}, d_compressed_kv_cache_tensor{nullptr},
        h_compressed_offset_tensor{nullptr}, d_compressed_offset_tensor{nullptr}, h_cu_kv_seq_lens{nullptr},
        d_cu_kv_seq_lens{nullptr}, h_cu_chunk_lens{nullptr}, d_cu_chunk_lens{nullptr},
        h_chunked_ld_global_offset{nullptr}, d_chunked_ld_global_offset{nullptr}, h_cu_q_seq_lens{nullptr},
        d_cu_q_seq_lens{nullptr},

        // for kernel 1
        h_compressed_kv_output{nullptr}, d_compressed_kv_output{nullptr}, h_k_pe_output{nullptr},
        d_k_pe_output{nullptr}, h_compressed_kv_output_ref{nullptr}, h_k_pe_output_ref{nullptr},
        h_kv_scale_quant_orig{nullptr}, d_kv_scale_quant_orig{nullptr},

        // for merge attn {kv_full_tensor  = kv + k_pe}
        m_h_q_tensor{nullptr}, m_h_kv_full_tensor{nullptr}, m_h_chunked_kv_tensor{nullptr}, m_h_output_tensor{nullptr},
        m_h_softmax_sum_tensor{nullptr}, m_h_softmax_sum_accum_tensor{nullptr}, m_h_output_tensor_ref{nullptr},
        m_h_output_tensor_accum{nullptr}, m_d_q_tensor{nullptr}, m_d_kv_full_tensor{nullptr},
        m_d_chunked_kv_tensor{nullptr}, m_d_output_tensor{nullptr}, m_d_softmax_sum_tensor{nullptr},
        m_d_softmax_sum_accum_tensor{nullptr}, m_d_output_tensor_accum{nullptr}, m_h_merge_op{nullptr},
        m_d_merge_op{nullptr};

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

    // for chunked main loop
    std::vector<int> max_chunk_len_per_loop;

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
        mBatchSize = 16;
        // mMaxSeqLen = 128;
        mChunkSize = 16;
        mNumHeads = 16;
        mLoraSize = 512;
        mRopeSize = 64;
        mNopeSize = 128;
        mIsCausalMask = false;
        mMaxGenLength = 128;
        mTokensPerBlock = 16;
        assert(this->mChunkSize % this->mTokensPerBlock == 0);
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

    template <typename T>
    void showHostTensor(tensorrt_llm::runtime::BufferManager::ITensorPtr& tensor, std::string const& tensor_name)
    {
        std::cout << "Tensor: " << tensor_name << ": \n";
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
        auto cacheType = dtype;
        if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
        {
            cacheType = nvinfer1::DataType::kFP8;
            this->h_kv_scale_quant_orig
                = tensorrt_llm::runtime::BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
            this->d_kv_scale_quant_orig
                = tensorrt_llm::runtime::BufferManager::gpuSync(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
            auto* kv_scale_quant_orig_ptr = bufferCast<float>(*(this->h_kv_scale_quant_orig));
            float kv_scale_orig_quant = 2.0F;
            kv_scale_quant_orig_ptr[0] = 1.0 / kv_scale_orig_quant;
            cudaMemcpy(this->d_kv_scale_quant_orig->data(), this->h_kv_scale_quant_orig->data(),
                this->h_kv_scale_quant_orig->getSizeInBytes(), cudaMemcpyHostToDevice);
        }

        // cu lens
        this->h_cu_kv_seq_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize + 1}), nvinfer1::DataType::kINT64);
        this->h_cu_q_seq_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize + 1}), nvinfer1::DataType::kINT64);
        this->d_cu_kv_seq_lens = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_cu_kv_seq_lens->getShape(), nvinfer1::DataType::kINT64);
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
#ifdef TRTLLM_MLA_CHUNKED_PREFILL_TEST_DBG
            this->showHostTensor<int64_t>(this->h_cu_q_seq_lens, "cu_q_seq_lens");
            this->showHostTensor<int64_t>(this->h_cu_kv_seq_lens, "cu_kv_seq_lens");
#endif
        }
        int const total_chunk_size = this->mChunkSize * this->mBatchSize;
        int const total_cached_kv_len = this->mTotalKVLen - this->mTotalQLen;
        int const chunked_loop_num = (total_cached_kv_len + total_chunk_size - 1) / total_chunk_size;
        this->h_cu_chunk_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({chunked_loop_num + 1, this->mBatchSize + 1}), nvinfer1::DataType::kINT64);
        this->h_chunked_ld_global_offset = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({chunked_loop_num + 1, this->mBatchSize}), nvinfer1::DataType::kINT64);
        this->memsetZeroHost(this->h_chunked_ld_global_offset);
        this->d_cu_chunk_lens = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_cu_chunk_lens->getShape(), nvinfer1::DataType::kINT64);
        this->d_chunked_ld_global_offset = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_chunked_ld_global_offset->getShape(), nvinfer1::DataType::kINT64);

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
            cacheType);
        this->h_compressed_offset_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mBatchSize, 2, this->mMaxBlockPerSeq + 1}), nvinfer1::DataType::kINT32);
        this->d_kv_cache_tensor
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_kv_cache_tensor->getShape(), dtype);
        this->d_compressed_kv_cache_tensor
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->h_compressed_kv_cache_tensor->getShape(), cacheType);
        this->d_compressed_offset_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            this->h_compressed_offset_tensor->getShape(), nvinfer1::DataType::kINT32);

        {
            auto* compressed_kv_cache_ptr = bufferCast<TCache>(*(this->h_compressed_kv_cache_tensor));
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
        this->m_h_output_tensor_accum = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalQLen, this->mNumHeads, this->mNopeSize}), dtype);
        this->m_h_merge_op = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({chunked_loop_num + 1, this->mBatchSize}), nvinfer1::DataType::kINT64);
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
        this->m_d_output_tensor_accum
            = tensorrt_llm::runtime::BufferManager::gpuSync(this->m_h_output_tensor_accum->getShape(), dtype);
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
            this->memsetZeroHost(m_h_output_tensor_accum);

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
            cudaMemcpyAsync(m_d_output_tensor_accum->data(), m_h_output_tensor_accum->data(),
                m_h_output_tensor_accum->getSizeInBytes(), cudaMemcpyHostToDevice, mStream->get());
            cudaStreamSynchronize(mStream->get());
        }
        this->prepareChunkedPrefillMetaData();
        return true;
    }

    int prepareChunkedPrefillMetaData()
    {
        using tensorrt_llm::runtime::bufferCast;
        int const total_chunk_size = this->mChunkSize * this->mBatchSize;
        int chunked_loop_num = (this->mTotalKVLen - this->mTotalQLen + total_chunk_size - 1) / total_chunk_size;

        auto* h_merge_op = bufferCast<int64_t>(*(this->m_h_merge_op));             // {chunked_loop_num + 1, batch_size}
        auto* h_cu_q_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_q_seq_lens)); // {batch_size + 1}
        auto* h_cu_kv_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_kv_seq_lens)); // {batch_size + 1}
        auto* h_cu_chunk_lens_ptr
            = bufferCast<int64_t>(*(this->h_cu_chunk_lens));            // {chunked_loop_num + 1, batch_size + 1}
        auto* h_chunked_ld_global_offset_ptr
            = bufferCast<int64_t>(*(this->h_chunked_ld_global_offset)); // {chunked_loop_num + 1, batch_size}

        this->max_chunk_len_per_loop.clear();
        std::vector<int64_t> chunked_seq_len_vec((chunked_loop_num + 1) * (this->mBatchSize), 0);
        // 0 -> chunked_loop_num -1
        int remain_buffer_len = total_chunk_size;
        int curr_loop_idx = 0;
        int temp_max_chunk_len = 0;

#define chunked_seq_len(chunked_loop_idx, b_idx) chunked_seq_len_vec[(chunked_loop_idx) * (this->mBatchSize) + (b_idx)]
#define cu_chunked_seq_len(chunked_loop_idx, b_idx)                                                                    \
    h_cu_chunk_lens_ptr[(chunked_loop_idx) * (this->mBatchSize + 1) + (b_idx)]
#define chunked_ld_global_offset(chunked_loop_idx, b_idx)                                                              \
    h_chunked_ld_global_offset_ptr[(chunked_loop_idx) * (this->mBatchSize) + (b_idx)]

        for (int b = 0; b < this->mBatchSize; b++)
        {
            int temp_cached_kv_len = (h_cu_kv_seq_lens_ptr[b + 1] - h_cu_kv_seq_lens_ptr[b])
                - (h_cu_q_seq_lens_ptr[b + 1] - h_cu_q_seq_lens_ptr[b]);
            while (temp_cached_kv_len > 0)
            {
                auto used_buffer_len = std::min(remain_buffer_len, temp_cached_kv_len);
                remain_buffer_len -= used_buffer_len;
                temp_cached_kv_len -= used_buffer_len;
                temp_max_chunk_len = std::max(temp_max_chunk_len, used_buffer_len);
                chunked_seq_len(curr_loop_idx, b) = used_buffer_len;
                chunked_ld_global_offset(curr_loop_idx + 1, b)
                    = chunked_ld_global_offset(curr_loop_idx, b) + used_buffer_len;
                if (remain_buffer_len == 0)
                {
                    this->max_chunk_len_per_loop.push_back(temp_max_chunk_len);
                    temp_max_chunk_len = 0;
                    remain_buffer_len = total_chunk_size;
                    curr_loop_idx++;
                }
            }
        }
        if (this->max_chunk_len_per_loop.size() < chunked_loop_num)
        {
            this->max_chunk_len_per_loop.push_back(temp_max_chunk_len);
        }
        assert(this->max_chunk_len_per_loop.size() == chunked_loop_num);

        // for not cached part
        for (int b = 0; b < this->mBatchSize; b++)
        {
            int uncached_len = (h_cu_q_seq_lens_ptr[b + 1] - h_cu_q_seq_lens_ptr[b]);
            chunked_seq_len(chunked_loop_num, b) = uncached_len;
        }
        for (int loop_idx = 0; loop_idx < chunked_loop_num + 1; loop_idx++)
        {
            for (int b = 0; b < this->mBatchSize; b++)
            {
                cu_chunked_seq_len(loop_idx, b + 1) = cu_chunked_seq_len(loop_idx, b) + chunked_seq_len(loop_idx, b);
            }
        }
        // merge op
        for (int loop_idx = 0; loop_idx < chunked_loop_num; loop_idx++)
        {
            for (int b = 0; b < this->mBatchSize; b++)
            {
                if (chunked_seq_len(loop_idx, b) != 0 && (loop_idx == 0 || chunked_seq_len(loop_idx - 1, b) == 0))
                {
                    h_merge_op[loop_idx * (this->mBatchSize) + b] = 2; // copy
                }
                else if (chunked_seq_len(loop_idx, b) != 0)
                {
                    h_merge_op[loop_idx * (this->mBatchSize) + b] = 1; // merge
                }
                else
                {
                    h_merge_op[loop_idx * (this->mBatchSize) + b] = 0; // skip
                }
            }
        }
        // for the last uncached part
        for (int b = 0; b < this->mBatchSize; b++)
        {
            int temp_cached_kv_len = (h_cu_kv_seq_lens_ptr[b + 1] - h_cu_kv_seq_lens_ptr[b])
                - (h_cu_q_seq_lens_ptr[b + 1] - h_cu_q_seq_lens_ptr[b]);
            if (temp_cached_kv_len == 0)
            {
                h_merge_op[chunked_loop_num * (this->mBatchSize) + b] = 2; // copy
            }
            else
            {
                h_merge_op[chunked_loop_num * (this->mBatchSize) + b] = 1; // merge
            }
            chunked_ld_global_offset(chunked_loop_num, b) = temp_cached_kv_len;
        }

#undef chunked_seq_len
#undef cu_chunked_seq_len
#undef chunked_ld_global_offset
        // copy to device
        cudaMemcpy(this->d_cu_chunk_lens->data(), this->h_cu_chunk_lens->data(),
            this->h_cu_chunk_lens->getSizeInBytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_chunked_ld_global_offset->data(), this->h_chunked_ld_global_offset->data(),
            this->h_chunked_ld_global_offset->getSizeInBytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(this->m_d_merge_op->data(), this->m_h_merge_op->data(), this->m_h_merge_op->getSizeInBytes(),
            cudaMemcpyHostToDevice);
#ifdef TRTLLM_MLA_CHUNKED_PREFILL_TEST_DBG
        std::cout << "chunked_loop_num: " << chunked_loop_num << '\n';
        this->showHostTensor<int64_t>(this->m_h_merge_op, "merge_op");
        this->showHostTensor<int64_t>(this->h_chunked_ld_global_offset, "chunked_ld_global_offset");
        this->showHostTensor<int64_t>(this->h_cu_chunk_lens, "cu_chunk_lens");
#endif
        return chunked_loop_num;
    }

    void PerformNormalAttention()
    {
        using tensorrt_llm::runtime::bufferCast;

        auto* q_ptr = bufferCast<DataType>(*(this->m_h_q_tensor));
        auto* kv_ptr = bufferCast<DataType>(*(this->m_h_kv_full_tensor));
        auto* output_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_ref));
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
        auto* h_output_ptr = bufferCast<DataType>(*(this->m_h_output_tensor));
        auto* h_output_accum_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_accum));
        auto* h_softmax_sum_ptr = bufferCast<float>(*(this->m_h_softmax_sum_tensor));
        auto* h_softmax_sum_accum_ptr = bufferCast<float>(*(this->m_h_softmax_sum_accum_tensor));
        auto* h_cu_q_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_q_seq_lens));
        auto* h_cu_kv_seq_lens_ptr = bufferCast<int64_t>(*(this->h_cu_kv_seq_lens));
        auto* h_cu_chunk_lens_ptr = bufferCast<int64_t>(*(this->h_cu_chunk_lens));
        auto* h_chunked_ld_global_offset_ptr = bufferCast<int64_t>(*(this->h_chunked_ld_global_offset));
        auto* d_kv_ptr = bufferCast<DataType>(*(this->m_d_kv_full_tensor));
        auto* d_chunked_kv_ptr = bufferCast<DataType>(*(this->m_d_chunked_kv_tensor));
        auto* d_softmax_sum_ptr = bufferCast<float>(*(this->m_d_softmax_sum_tensor));
        auto* d_softmax_sum_accum_ptr = bufferCast<float>(*(this->m_d_softmax_sum_accum_tensor));
        auto* d_output_ptr = bufferCast<DataType>(*(this->m_d_output_tensor));
        auto* d_output_accum_ptr = bufferCast<DataType>(*(this->m_d_output_tensor_accum));
        auto* d_merge_op = bufferCast<int64_t>(*(this->m_d_merge_op));
        auto* d_cu_q_seq_lens_ptr = bufferCast<int64_t>(*(this->d_cu_q_seq_lens));

        int const total_chunk_size = this->mChunkSize * this->mBatchSize;
        int chunked_loop_num = (this->mTotalKVLen - this->mTotalQLen + total_chunk_size - 1) / total_chunk_size;
        // do not apply mask
        for (int _ = 0; _ < chunked_loop_num; _++)
        {
            // copy related kv chunk data
            copyRelatedChunkedKV(h_chunked_kv_ptr, h_kv_ptr, h_chunked_ld_global_offset_ptr, this->mBatchSize,
                this->mNumHeads, h_cu_kv_seq_lens_ptr, h_cu_chunk_lens_ptr, this->mNopeSize);
            // attention
            selfAttentionRef<DataType>(h_output_ptr, h_q_ptr, h_chunked_kv_ptr, this->mBatchSize, this->mNumHeads,
                h_cu_q_seq_lens_ptr, h_cu_chunk_lens_ptr, this->mNopeSize, true, h_softmax_sum_ptr, false);
            // merge attention

            // copy curr_attn and softmax_sum to device
            cudaMemcpy(d_softmax_sum_ptr, h_softmax_sum_ptr, this->m_h_softmax_sum_tensor->getSizeInBytes(),
                cudaMemcpyHostToDevice);
            cudaMemcpy(d_output_ptr, h_output_ptr, this->m_h_output_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
            // merge softmax
            tensorrt_llm::kernels::invokeMergeAttnWithSoftmax<DataType>(d_output_accum_ptr, d_softmax_sum_accum_ptr,
                d_output_accum_ptr, d_softmax_sum_accum_ptr, d_output_ptr, d_softmax_sum_ptr, this->mBatchSize,
                d_cu_q_seq_lens_ptr, this->mMaxQSeqLen, d_merge_op, this->mNumHeads, this->mNopeSize, mStream->get());
            cudaStreamSynchronize(mStream->get());
            // copy merged softmax sum back to host
            cudaMemcpy(h_softmax_sum_accum_ptr, d_softmax_sum_accum_ptr, this->m_h_softmax_sum_tensor->getSizeInBytes(),
                cudaMemcpyDeviceToHost);
            cudaMemcpy(h_output_accum_ptr, d_output_accum_ptr, this->m_h_output_tensor->getSizeInBytes(),
                cudaMemcpyDeviceToHost);
            // update merge op, ld global offset, cu chunk lens ptr.
            d_merge_op += this->mBatchSize;
            h_cu_chunk_lens_ptr += (this->mBatchSize + 1);
            h_chunked_ld_global_offset_ptr += this->mBatchSize;
        }
        // final round, apply causal mask.

        // copy the last chunked kv data
        copyRelatedChunkedKV(h_chunked_kv_ptr, h_kv_ptr, h_chunked_ld_global_offset_ptr, this->mBatchSize,
            this->mNumHeads, h_cu_kv_seq_lens_ptr, h_cu_chunk_lens_ptr, this->mNopeSize);
        // attention
        selfAttentionRef<DataType>(h_output_ptr, h_q_ptr, h_chunked_kv_ptr, this->mBatchSize, this->mNumHeads,
            h_cu_q_seq_lens_ptr, h_cu_chunk_lens_ptr, this->mNopeSize, true, h_softmax_sum_ptr, this->mIsCausalMask);
        // merge attention
        // copy curr_attn and softmax_sum to device
        cudaMemcpy(d_softmax_sum_ptr, h_softmax_sum_ptr, this->m_h_softmax_sum_tensor->getSizeInBytes(),
            cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_ptr, h_output_ptr, this->m_h_output_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
        tensorrt_llm::kernels::invokeMergeAttnWithSoftmax<DataType>(d_output_accum_ptr, d_softmax_sum_accum_ptr,
            d_output_accum_ptr, d_softmax_sum_accum_ptr, d_output_ptr, d_softmax_sum_ptr, this->mBatchSize,
            d_cu_q_seq_lens_ptr, this->mMaxQSeqLen, d_merge_op, this->mNumHeads, this->mNopeSize, mStream->get());
        cudaStreamSynchronize(mStream->get());
        // copy merged softmax sum back to host
        cudaMemcpy(h_softmax_sum_accum_ptr, d_softmax_sum_accum_ptr, this->m_h_softmax_sum_tensor->getSizeInBytes(),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_output_accum_ptr, d_output_accum_ptr, this->m_h_output_tensor->getSizeInBytes(), cudaMemcpyDeviceToHost);
        sync_check_cuda_error(mStream->get());
    }

    void PerformLoadChunkedKVRef(int chunk_idx)
    {
        using tensorrt_llm::runtime::bufferCast;

        auto* compressed_kv_output_ptr = bufferCast<DataType>(*(this->h_compressed_kv_output_ref));
        auto* k_pe_output_ptr = bufferCast<DataType>(*(this->h_k_pe_output_ref));
        auto* compressed_kv_cache_ptr = bufferCast<TCache>(*(this->h_compressed_kv_cache_tensor));
        auto* offset_ptr = bufferCast<int32_t>(*(this->h_compressed_offset_tensor));
        auto* h_cu_chunk_lens_ptr = bufferCast<int64_t>(*(this->h_cu_chunk_lens)) + chunk_idx * (this->mBatchSize + 1);
        auto* h_chunked_ld_global_offset_ptr
            = bufferCast<int64_t>(*(this->h_chunked_ld_global_offset)) + chunk_idx * this->mBatchSize;
        float* kv_scale_quant_orig_ptr = nullptr;
        if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
        {
            kv_scale_quant_orig_ptr = bufferCast<float>(*(this->h_kv_scale_quant_orig));
        }
        tensorrt_llm::kernels::KVBlockArray kv_cache(this->mBatchSize, this->mMaxBlockPerSeq, this->mTokensPerBlock,
            sizeof(TCache) * 1 * (this->mLoraSize + this->mRopeSize), 0, 0, 0, 0, compressed_kv_cache_ptr, nullptr,
            reinterpret_cast<tensorrt_llm::kernels::KVBlockArrayForContextFMHA::DataType*>(offset_ptr));

        loadChunkedKVKernelRef<DataType, TCache>(compressed_kv_output_ptr, k_pe_output_ptr, kv_cache, this->mBatchSize,
            h_cu_chunk_lens_ptr, h_chunked_ld_global_offset_ptr, this->mLoraSize, this->mRopeSize,
            kv_scale_quant_orig_ptr);
    }

    void PreformLoadChunkedKV(int chunk_idx)
    {
        using tensorrt_llm::runtime::bufferCast;

        auto* compressed_kv_output_ptr = bufferCast<DataType>(*(this->d_compressed_kv_output));
        auto* k_pe_output_ptr = bufferCast<DataType>(*(this->d_k_pe_output));
        auto* compressed_kv_cache_ptr = bufferCast<TCache>(*(this->d_compressed_kv_cache_tensor));
        auto* offset_ptr = bufferCast<int32_t>(*(this->d_compressed_offset_tensor));
        auto* d_cu_chunk_lens_ptr = bufferCast<int64_t>(*(this->d_cu_chunk_lens)) + chunk_idx * (this->mBatchSize + 1);
        auto* d_chunked_ld_global_offset_ptr
            = bufferCast<int64_t>(*(this->d_chunked_ld_global_offset)) + chunk_idx * this->mBatchSize;
        float* kv_scale_quant_orig_ptr = nullptr;
        if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
        {
            kv_scale_quant_orig_ptr = bufferCast<float>(*(this->d_kv_scale_quant_orig));
        }
        tensorrt_llm::kernels::KVBlockArray kv_cache(this->mBatchSize, this->mMaxBlockPerSeq, this->mTokensPerBlock,
            sizeof(TCache) * 1 * (this->mLoraSize + this->mRopeSize), 0, 0, 0, 0, compressed_kv_cache_ptr, nullptr,
            reinterpret_cast<tensorrt_llm::kernels::KVBlockArrayForContextFMHA::DataType*>(offset_ptr));
        // copy cu chunk lens to device
        cudaMemcpy(this->d_cu_chunk_lens->data(), this->h_cu_chunk_lens->data(),
            this->h_cu_chunk_lens->getSizeInBytes(), cudaMemcpyHostToDevice);
        tensorrt_llm::kernels::invokeMLALoadChunkedKV<DataType, TCache>(compressed_kv_output_ptr, k_pe_output_ptr,
            kv_cache, this->mBatchSize, d_cu_chunk_lens_ptr, d_chunked_ld_global_offset_ptr, this->mLoraSize,
            this->mRopeSize, this->max_chunk_len_per_loop[chunk_idx], kv_scale_quant_orig_ptr, mStream->get());
        cudaStreamSynchronize(this->mStream->get());
        // copy result back to host
        cudaMemcpy(this->h_compressed_kv_output->data(), compressed_kv_output_ptr,
            this->h_compressed_kv_output->getSizeInBytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(this->h_k_pe_output->data(), k_pe_output_ptr, this->h_k_pe_output->getSizeInBytes(),
            cudaMemcpyDeviceToHost);
        sync_check_cuda_error(this->mStream->get());
    }
};

using MLATypes
    = ::testing::Types<std::pair<half, half>, std::pair<__nv_bfloat16, __nv_bfloat16>, std::pair<float, float>,
        std::pair<half, __nv_fp8_e4m3>, std::pair<__nv_bfloat16, __nv_fp8_e4m3>, std::pair<float, __nv_fp8_e4m3>>;

TYPED_TEST_SUITE(MlaChunkedPrefillTest, MLATypes);

TYPED_TEST(MlaChunkedPrefillTest, MlaChunkedPrefillDefault)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    using TCache = typename TestFixture::TCache;
    if constexpr (std::is_same_v<DataType, TCache>)
    {
        this->setDefaultParams();
        this->allocateBuffers();

        sync_check_cuda_error(this->mStream->get());
        bool allEqual{true};

        this->PerformNormalAttention();
        sync_check_cuda_error(this->mStream->get());

        this->PerformMergedAttention();
        sync_check_cuda_error(this->mStream->get());

        // check result
        auto* output_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_accum));
        auto* output_ref_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_ref));
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
}

TYPED_TEST(MlaChunkedPrefillTest, MlaChunkedPrefillCausalMask)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    using TCache = typename TestFixture::TCache;
    if constexpr (std::is_same_v<DataType, TCache>)
    {
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
        auto* output_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_accum));
        auto* output_ref_ptr = bufferCast<DataType>(*(this->m_h_output_tensor_ref));
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
}

TYPED_TEST(MlaChunkedPrefillTest, MlaChunkedLoad)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    this->setDefaultParams();
    this->allocateBuffers();

    sync_check_cuda_error(this->mStream->get());
    bool allEqual{true};

    int const total_chunk_size = this->mChunkSize * this->mBatchSize;
    int const chunked_loop_num = (this->mTotalKVLen - this->mTotalQLen + total_chunk_size - 1) / total_chunk_size;
    for (int _ = 0; _ < chunked_loop_num - 1; _++)
    {
        this->PerformLoadChunkedKVRef(_);
        sync_check_cuda_error(this->mStream->get());
        this->PreformLoadChunkedKV(_);
        sync_check_cuda_error(this->mStream->get());

        // check result
        auto* compressed_kv_output_ptr = bufferCast<DataType>(*(this->h_compressed_kv_output_ref));
        auto* compressed_kv_output_ref_ptr = bufferCast<DataType>(*(this->h_compressed_kv_output));
        auto* k_pe_output_ptr = bufferCast<DataType>(*(this->h_k_pe_output));
        auto* k_pe_output_ref_ptr = bufferCast<DataType>(*(this->h_k_pe_output_ref));
        // check kv
        for (int i = 0; i < this->h_compressed_kv_output->getSize(); i++)
        {
            if (std::abs(static_cast<float>(compressed_kv_output_ptr[i])
                    - static_cast<float>(compressed_kv_output_ref_ptr[i]))
                > getTolerance<DataType>(compressed_kv_output_ptr[i]))
            {
                std::cout << "Compressed KV output mismatch at loop: " << _ << " index " << i << ": "
                          << "expected " << static_cast<float>(compressed_kv_output_ref_ptr[i]) << ", got "
                          << static_cast<float>(compressed_kv_output_ptr[i]) << std::endl;
                allEqual = false;
                break;
            }
        }
        // check k_pe
        for (int i = 0; i < this->h_k_pe_output->getSize(); i++)
        {
            if (std::abs(static_cast<float>(k_pe_output_ptr[i]) - static_cast<float>(k_pe_output_ref_ptr[i]))
                > getTolerance<DataType>(k_pe_output_ptr[i]))
            {
                std::cout << "kpe mismatch at loop: " << _ << " index " << i << ": "
                          << "expected " << static_cast<float>(k_pe_output_ref_ptr[i]) << ", got "
                          << static_cast<float>(k_pe_output_ptr[i]) << std::endl;
                allEqual = false;
                break;
            }
        }
    }
    ASSERT_TRUE(allEqual);
}
