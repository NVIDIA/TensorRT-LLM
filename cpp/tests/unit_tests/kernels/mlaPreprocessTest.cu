/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include "tensorrt_llm/kernels/mlaKernels.h"
#include <random>

namespace
{

// copy matched kv cache data to compressed_kv_output and k_pe_output
// compressed_kv_output {total_cached_token, lora_size}
// k_pe_output {total_cached_token, rope_size}
// compressed_kv_cache {batch, 1 (ignore v), max_seq_len / tokens_per_block, num_head, tokens_per_block, (lora_size +
// rope_size)}
template <typename T, typename TCache>
void loadPagedKvKernelRef(T* compressed_kv_output, T* k_pe_output,
    tensorrt_llm::kernels::KVBlockArray const& compressed_kv_cache, int num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int const lora_size, int const rope_size,
    float const* kv_scale_quant_orig_ptr)
{
    static_assert(std::is_same_v<T, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as T or __nv_fp8_e4m3");
    int const head_dim = lora_size + rope_size;
    float const kv_scale_quant_orig = kv_scale_quant_orig_ptr ? kv_scale_quant_orig_ptr[0] : 1.0f;
    for (int b = 0; b < num_contexts; b++)
    {
        int const global_token_offset = cu_ctx_cached_kv_lens[b];
        int const current_token_len = cu_ctx_cached_kv_lens[b + 1] - cu_ctx_cached_kv_lens[b];
        for (int s = 0; s < current_token_len; s++)
        {
            int const global_token_idx = global_token_offset + s;
            for (int d = 0; d < head_dim; d++)
            {
                auto const* kv_src = reinterpret_cast<TCache const*>(compressed_kv_cache.getKBlockPtr(b, s));
                auto kv_block_idx = compressed_kv_cache.getKVLocalIdx(s, 0, head_dim, d);

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
                    compressed_kv_output[global_token_idx * lora_size + d] = data;
                }
                else
                {
                    k_pe_output[global_token_idx * rope_size + (d - lora_size)] = data;
                }
            }
        }
    }
}

inline bool almostEqual(float a, float b, float atol = 1e-2, float rtol = 1e-3)
{
    if (isnan(a) || isnan(b))
    {
        return false;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

} // namespace

template <typename Typepair>
class MlaPreprocessTest : public testing::Test
{
protected:
    using DataType = typename Typepair::first_type;
    using TCache = typename Typepair::second_type;
    static_assert(std::is_same_v<DataType, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as DataType or __nv_fp8_e4m3");
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    // kv_cache shape {batch, 2(k or v), max_seq_len / tokens_per_block, num_head, tokens_per_block, head_size}
    // k, v, k_pe shape {total_token, num_head, head_size(lora_size or rope_size, or uncompressed_head_size)}
    // offset shape {batch, 2, max_seq_len / tokens_per_block}
    // for KVBlockArray, we only allocate primary pool.
    // you can infer the allocateBuffers function for more details.
    tensorrt_llm::runtime::BufferManager::ITensorPtr h_kv_cache_tensor{nullptr}, h_kv_cache_tensor_ref{nullptr},
        d_kv_cache_tensor{nullptr}, d_compressed_kv_cache_tensor{nullptr}, d_compressed_kv_cache_tensor_ref{nullptr},
        h_compressed_kv_cache_tensor{nullptr}, h_compressed_kv_cache_tensor_ref{nullptr}, d_offset_tensor{nullptr},
        d_compressed_offset_tensor{nullptr}, d_cu_ctx_cached_kv_lens{nullptr}, d_cu_seq_lens{nullptr},
        h_offset_tensor{nullptr}, h_compressed_offset_tensor{nullptr}, h_cu_ctx_cached_kv_lens{nullptr},
        h_cu_seq_lens{nullptr}, h_kv_scale_orig_quant{nullptr}, d_kv_scale_orig_quant{nullptr},
        h_kv_scale_quant_orig{nullptr}, d_kv_scale_quant_orig{nullptr},
        // for kernel 1
        d_compressed_kv_output{nullptr}, h_compressed_kv_output{nullptr}, h_compressed_kv_output_ref{nullptr},
        d_k_pe_output{nullptr}, h_k_pe_output{nullptr}, h_k_pe_output_ref{nullptr};

    int mNumRequests{};
    int mMaxSeqLen{};
    int mMaxCachedSeqLen{};
    int mMaxUncachedSeqLen{};
    int mMaxBlockPerSeq{};
    int mTokensPerBlock{};
    int mNumHeadsCompressed{};
    int mNumHeadsUncompressed{};
    int mTotalTokens{};
    int mTotalCachedTokens{};
    int mTotalUncachedTokens{};
    int mLoraSize{};
    int mRopeSize{};
    int mUncompressedHeadSize{};
    int64_t mKvTokenStride{};

    std::mt19937 gen;

    void SetUp() override
    {
        if (shouldSkip())
        {
            GTEST_SKIP() << "Skipping mla preprocess test";
        }
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
        gen.seed(42U);
    }

    static bool shouldSkip()
    {
        return false;
    }

    void setDefaultParams()
    {
        this->mTokensPerBlock = 64;
        this->mNumHeadsCompressed = 1;
        this->mNumHeadsUncompressed = 128;
        this->mLoraSize = 512;
        this->mRopeSize = 64;
        this->mUncompressedHeadSize = 128;
        this->mMaxSeqLen = 0;
        this->mMaxCachedSeqLen = 0;
        this->mMaxUncachedSeqLen = 0;
        this->mKvTokenStride = this->mNumHeadsUncompressed * this->mUncompressedHeadSize;
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

    template <typename T>
    void memsetZeroDevice(T* ptr, size_t size)
    {
        cudaMemset(ptr, 0, size * sizeof(T));
    }

    template <typename T>
    void memsetZeroHost(T* ptr, size_t size)
    {
        std::memset(ptr, 0, size * sizeof(T));
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
        auto cache_dtype = dtype;
        if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
        {
            cache_dtype = nvinfer1::DataType::kFP8;
            this->h_kv_scale_orig_quant
                = tensorrt_llm::runtime::BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
            this->d_kv_scale_orig_quant
                = tensorrt_llm::runtime::BufferManager::gpuSync(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
            this->h_kv_scale_quant_orig
                = tensorrt_llm::runtime::BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
            this->d_kv_scale_quant_orig
                = tensorrt_llm::runtime::BufferManager::gpuSync(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
            auto* kv_scale_orig_quant_ptr = bufferCast<float>(*(this->h_kv_scale_orig_quant));
            auto* kv_scale_quant_orig_ptr = bufferCast<float>(*(this->h_kv_scale_quant_orig));
            float kv_scale_orig_quant = 2.0f;
            kv_scale_orig_quant_ptr[0] = kv_scale_orig_quant;
            kv_scale_quant_orig_ptr[0] = 1.0 / kv_scale_orig_quant;
            cudaMemcpy(this->d_kv_scale_orig_quant->data(), this->h_kv_scale_orig_quant->data(),
                this->h_kv_scale_orig_quant->getSizeInBytes(), cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_kv_scale_quant_orig->data(), this->h_kv_scale_quant_orig->data(),
                this->h_kv_scale_quant_orig->getSizeInBytes(), cudaMemcpyHostToDevice);
        }
        else
        {
            static_assert(std::is_same_v<DataType, TCache>, "TCache must be the same type as DataType");
        }
        this->h_cu_seq_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests + 1}), nvinfer1::DataType::kINT64);
        this->h_cu_ctx_cached_kv_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests + 1}), nvinfer1::DataType::kINT64);
        this->d_cu_seq_lens = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mNumRequests + 1}), nvinfer1::DataType::kINT64);
        this->d_cu_ctx_cached_kv_lens = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mNumRequests + 1}), nvinfer1::DataType::kINT64);
        {
            // set random sequence length
            auto* cu_seq_lens_temp_ptr = bufferCast<int64_t>(*(this->h_cu_seq_lens));
            auto* cu_ctx_cached_kv_lens_temp_ptr = bufferCast<int64_t>(*(this->h_cu_ctx_cached_kv_lens));
            cu_seq_lens_temp_ptr[0] = 0;
            cu_ctx_cached_kv_lens_temp_ptr[0] = 0;
            for (int i = 1; i <= this->mNumRequests; i++)
            {
                int temp_seq_len = generateRandomSizeSmallerThan(512);
                if (temp_seq_len <= 0)
                {
                    temp_seq_len = 1; // at least 1 token
                }
                int cached_seq_len = generateRandomSizeSmallerThan(temp_seq_len);
                this->mMaxSeqLen = std::max(temp_seq_len, this->mMaxSeqLen);
                this->mMaxCachedSeqLen = std::max(cached_seq_len, this->mMaxCachedSeqLen);
                this->mMaxUncachedSeqLen = std::max(temp_seq_len - cached_seq_len, this->mMaxUncachedSeqLen);
                this->mTotalTokens += temp_seq_len;
                this->mTotalCachedTokens += cached_seq_len;
                this->mTotalUncachedTokens += temp_seq_len - cached_seq_len;
                cu_seq_lens_temp_ptr[i] = cu_seq_lens_temp_ptr[i - 1] + temp_seq_len;
                cu_ctx_cached_kv_lens_temp_ptr[i] = cu_ctx_cached_kv_lens_temp_ptr[i - 1] + cached_seq_len;
                // std::cout << "batch " << i << "seq len: " << temp_seq_len << ", cached len: " << cached_seq_len
                //           << ", uncached len: " << temp_seq_len - cached_seq_len << std::endl;
            }
            cudaMemcpy(this->d_cu_seq_lens->data(), this->h_cu_seq_lens->data(), this->h_cu_seq_lens->getSizeInBytes(),
                cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_cu_ctx_cached_kv_lens->data(), this->h_cu_ctx_cached_kv_lens->data(),
                this->h_cu_ctx_cached_kv_lens->getSizeInBytes(), cudaMemcpyHostToDevice);
        }

        // malloc kv_cache
        this->mMaxBlockPerSeq = (this->mMaxSeqLen + this->mTokensPerBlock - 1) / this->mTokensPerBlock;
        this->h_kv_cache_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq, this->mNumHeadsUncompressed,
                this->mTokensPerBlock, this->mUncompressedHeadSize + this->mRopeSize}),
            dtype);
        this->h_kv_cache_tensor_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq, this->mNumHeadsUncompressed,
                this->mTokensPerBlock, this->mUncompressedHeadSize + this->mRopeSize}),
            dtype);
        this->h_compressed_kv_cache_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 1, this->mMaxBlockPerSeq, this->mNumHeadsCompressed,
                this->mTokensPerBlock, this->mLoraSize + this->mRopeSize}),
            cache_dtype);
        this->h_compressed_kv_cache_tensor_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 1, this->mMaxBlockPerSeq, this->mNumHeadsCompressed,
                this->mTokensPerBlock, this->mLoraSize + this->mRopeSize}),
            cache_dtype);
        this->h_offset_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq}), nvinfer1::DataType::kINT32);
        this->h_compressed_offset_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq}), nvinfer1::DataType::kINT32);
        this->d_kv_cache_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq, this->mNumHeadsUncompressed,
                this->mTokensPerBlock, this->mUncompressedHeadSize + this->mRopeSize}),
            dtype);
        this->d_compressed_kv_cache_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mNumRequests, 1, this->mMaxBlockPerSeq, this->mNumHeadsCompressed,
                this->mTokensPerBlock, this->mLoraSize + this->mRopeSize}),
            cache_dtype);
        this->d_compressed_kv_cache_tensor_ref = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mNumRequests, 1, this->mMaxBlockPerSeq, this->mNumHeadsCompressed,
                this->mTokensPerBlock, this->mLoraSize + this->mRopeSize}),
            cache_dtype);
        this->d_offset_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq}), nvinfer1::DataType::kINT32);
        this->d_compressed_offset_tensor = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq}), nvinfer1::DataType::kINT32);
        {
            auto* kv_cache_ptr = bufferCast<DataType>(*(this->h_kv_cache_tensor));
            auto* kv_cache_ref_ptr = bufferCast<DataType>(*(this->h_kv_cache_tensor_ref));
            auto* compressed_kv_cache_ptr = bufferCast<TCache>(*(this->h_compressed_kv_cache_tensor));
            auto* compressed_kv_cache_ref_ptr = bufferCast<TCache>(*(this->h_compressed_kv_cache_tensor_ref));
            auto* offset_ptr = bufferCast<int32_t>(*(this->h_offset_tensor));
            auto* compressed_offset_ptr = bufferCast<int32_t>(*(this->h_compressed_offset_tensor));
            fillArrayDataWithMod(compressed_kv_cache_ptr, this->h_compressed_kv_cache_tensor->getSize());
            fillArrayDataWithMod(compressed_kv_cache_ref_ptr, this->h_compressed_kv_cache_tensor_ref->getSize());
            memsetZeroHost<DataType>(kv_cache_ptr, this->h_kv_cache_tensor->getSize());
            memsetZeroHost<DataType>(kv_cache_ref_ptr, this->h_kv_cache_tensor_ref->getSize());
            cudaMemcpy(this->d_kv_cache_tensor->data(), this->h_kv_cache_tensor->data(),
                this->h_kv_cache_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
            // fillArrayDataWithMod(offset_ptr, this->offset_tensor->getSize());
            fillKVOffsetData(
                compressed_offset_ptr, this->h_compressed_offset_tensor->getSize(), false, this->mMaxBlockPerSeq);
            cudaMemcpy(this->d_compressed_kv_cache_tensor->data(), this->h_compressed_kv_cache_tensor->data(),
                this->h_compressed_kv_cache_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_compressed_kv_cache_tensor_ref->data(), this->h_compressed_kv_cache_tensor_ref->data(),
                this->h_compressed_kv_cache_tensor_ref->getSizeInBytes(), cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_compressed_offset_tensor->data(), this->h_compressed_offset_tensor->data(),
                this->h_compressed_offset_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
            cudaMemcpy(this->d_offset_tensor->data(), this->h_offset_tensor->data(),
                this->h_offset_tensor->getSizeInBytes(), cudaMemcpyHostToDevice);
        }

        // compressed_kv_output + k_pe_output for loadPagedKvKernel (kernel 1)
        this->h_compressed_kv_output = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, this->mLoraSize}), dtype);
        this->h_compressed_kv_output_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, this->mLoraSize}), dtype);
        this->d_compressed_kv_output = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mTotalCachedTokens, this->mLoraSize}), dtype);
        this->h_k_pe_output = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, this->mRopeSize}), dtype);
        this->h_k_pe_output_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, this->mRopeSize}), dtype);
        this->d_k_pe_output = tensorrt_llm::runtime::BufferManager::gpuSync(
            ITensor::makeShape({this->mTotalCachedTokens, this->mRopeSize}), dtype);
        {
            auto* compressed_kv_output_ptr = bufferCast<DataType>(*(this->h_compressed_kv_output));
            auto* compressed_kv_output_ref_ptr = bufferCast<DataType>(*(this->h_compressed_kv_output_ref));
            memsetZeroHost<DataType>(compressed_kv_output_ptr, this->h_compressed_kv_output->getSize());
            memsetZeroHost<DataType>(compressed_kv_output_ref_ptr, this->h_compressed_kv_output_ref->getSize());
            cudaMemcpy(this->d_compressed_kv_output->data(), this->h_compressed_kv_output->data(),
                this->h_compressed_kv_output->getSizeInBytes(), cudaMemcpyHostToDevice);

            auto* k_pe_output_ptr = bufferCast<DataType>(*(this->h_k_pe_output));
            auto* k_pe_output_ref_ptr = bufferCast<DataType>(*(this->h_k_pe_output_ref));
            memsetZeroHost<DataType>(k_pe_output_ptr, this->h_k_pe_output->getSize());
            memsetZeroHost<DataType>(k_pe_output_ref_ptr, this->h_k_pe_output_ref->getSize());
            cudaMemcpy(this->d_k_pe_output->data(), this->h_k_pe_output->data(), this->h_k_pe_output->getSizeInBytes(),
                cudaMemcpyHostToDevice);
        }
        return true;
    }

    void PerformLoadPagedKV()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* compressed_kv_output_ptr = bufferCast<DataType>(*(this->d_compressed_kv_output));
        auto* k_pe_output_ptr = bufferCast<DataType>(*(this->d_k_pe_output));
        auto* compressed_kv_cache_ptr = bufferCast<TCache>(*(this->d_compressed_kv_cache_tensor));
        auto* offset_ptr = bufferCast<int32_t>(*(this->d_compressed_offset_tensor));
        auto* cu_ctx_cached_kv_lens_ptr = bufferCast<int64_t>(*(this->d_cu_ctx_cached_kv_lens));
        float* kv_scale_quant_orig_ptr = nullptr;
        if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
        {
            kv_scale_quant_orig_ptr = bufferCast<float>(*(this->d_kv_scale_quant_orig));
        }
        tensorrt_llm::kernels::KVBlockArray kv_cache(this->mNumRequests, this->mMaxBlockPerSeq, this->mTokensPerBlock,
            sizeof(TCache) * 1 * (this->mLoraSize + this->mRopeSize), 0, 0, 0, 0, compressed_kv_cache_ptr, nullptr,
            reinterpret_cast<tensorrt_llm::kernels::KVBlockArrayForContextFMHA::DataType*>(offset_ptr));
        tensorrt_llm::kernels::invokeMLALoadPagedKV<DataType, TCache>(compressed_kv_output_ptr, k_pe_output_ptr,
            kv_cache, this->mNumRequests, cu_ctx_cached_kv_lens_ptr, this->mMaxCachedSeqLen, this->mLoraSize,
            this->mRopeSize, kv_scale_quant_orig_ptr, this->mStream->get());
        cudaStreamSynchronize(this->mStream->get());
        cudaMemcpy(this->h_compressed_kv_output->data(), this->d_compressed_kv_output->data(),
            this->d_compressed_kv_output->getSizeInBytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(this->h_k_pe_output->data(), this->d_k_pe_output->data(), this->d_k_pe_output->getSizeInBytes(),
            cudaMemcpyDeviceToHost);
    }

    void PerformLoadPagedKVRef()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* compressed_kv_output_ptr = bufferCast<DataType>(*(this->h_compressed_kv_output_ref));
        auto* k_pe_output_ptr = bufferCast<DataType>(*(this->h_k_pe_output_ref));
        auto* compressed_kv_cache_ptr = bufferCast<TCache>(*(this->h_compressed_kv_cache_tensor));
        auto* offset_ptr = bufferCast<int32_t>(*(this->h_compressed_offset_tensor));
        auto* cu_ctx_cached_kv_lens_ptr = bufferCast<int64_t>(*(this->h_cu_ctx_cached_kv_lens));
        float* kv_scale_quant_orig_ptr = nullptr;
        if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
        {
            kv_scale_quant_orig_ptr = bufferCast<float>(*(this->h_kv_scale_quant_orig));
        }
        tensorrt_llm::kernels::KVBlockArray kv_cache(this->mNumRequests, this->mMaxBlockPerSeq, this->mTokensPerBlock,
            sizeof(TCache) * 1 * (this->mLoraSize + this->mRopeSize), 0, 0, 0, 0, compressed_kv_cache_ptr, nullptr,
            reinterpret_cast<tensorrt_llm::kernels::KVBlockArrayForContextFMHA::DataType*>(offset_ptr));
        loadPagedKvKernelRef<DataType, TCache>(compressed_kv_output_ptr, k_pe_output_ptr, kv_cache, this->mNumRequests,
            cu_ctx_cached_kv_lens_ptr, this->mLoraSize, this->mRopeSize, kv_scale_quant_orig_ptr);
    }

    template <typename T>
    bool CheckEqual(T const* expected, T const* output, size_t size)
    {
        for (int i = 0; i < size; i++)
        {
            auto e = static_cast<float>(expected[i]);
            auto o = static_cast<float>(output[i]);
            if (!almostEqual(e, o, 1e-3, 1e-3))
            {
                TLLM_LOG_ERROR(
                    "Mismatch input value. Position of inputs: %d, expected value: %f, output value: %f", i, e, o);
                return false;
            }
        }
        return true;
    }
};

using MLATypes
    = ::testing::Types<std::pair<half, half>, std::pair<__nv_bfloat16, __nv_bfloat16>, std::pair<float, float>,
        std::pair<half, __nv_fp8_e4m3>, std::pair<__nv_bfloat16, __nv_fp8_e4m3>, std::pair<float, __nv_fp8_e4m3>>;
TYPED_TEST_SUITE(MlaPreprocessTest, MLATypes);

TYPED_TEST(MlaPreprocessTest, MLAPreprocessDefault)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    using TCache = typename TestFixture::TCache;
    this->mNumRequests = 8;
    this->setDefaultParams();
    EXPECT_TRUE(this->allocateBuffers());

    sync_check_cuda_error(this->mStream->get());
    bool allEqual{true};

    {
        this->PerformLoadPagedKV();
        sync_check_cuda_error(this->mStream->get());
        this->PerformLoadPagedKVRef();
        auto* compressed_kv_output_ptr = bufferCast<DataType>(*(this->h_compressed_kv_output));
        auto* k_pe_output_ptr = bufferCast<DataType>(*(this->h_k_pe_output));
        auto* compressed_kv_output_ref_ptr = bufferCast<DataType>(*(this->h_compressed_kv_output_ref));
        auto* k_pe_output_ref_ptr = bufferCast<DataType>(*(this->h_k_pe_output_ref));
        allEqual = this->CheckEqual(
            compressed_kv_output_ref_ptr, compressed_kv_output_ptr, this->h_compressed_kv_output->getSize());
        EXPECT_TRUE(allEqual);
        allEqual = this->CheckEqual(k_pe_output_ref_ptr, k_pe_output_ptr, this->h_k_pe_output->getSize());
        EXPECT_TRUE(allEqual);
    }
}
