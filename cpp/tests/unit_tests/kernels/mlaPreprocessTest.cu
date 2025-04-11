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

// copy matched kv cache data to kv_output
// kv_output {total_cached_token, num_head = 1, head_size(lora_size + rope_size)}
// compressed_kv_cache {batch, 1 (ignore v), max_seq_len / tokens_per_block, num_head, tokens_per_block, (lora_size +
// rope_size)}
template <typename T>
void loadCompressedPagedKvKernelRef(T* kv_output, tensorrt_llm::kernels::KVBlockArray const& compressed_kv_cache,
    int num_contexts, int64_t const* cu_ctx_cached_kv_lens, int head_dim)
{

    for (int b = 0; b < num_contexts; b++)
    {
        int const global_token_offset = cu_ctx_cached_kv_lens[b];
        int const current_token_len = cu_ctx_cached_kv_lens[b + 1] - cu_ctx_cached_kv_lens[b];
        for (int s = 0; s < current_token_len; s++)
        {
            for (int d = 0; d < head_dim; d++)
            {
                auto const* kv_src = reinterpret_cast<T const*>(compressed_kv_cache.getKBlockPtr(b, s));
                auto kv_block_idx = compressed_kv_cache.getKVLocalIdx(s, 0, head_dim, d);

                int const global_token_idx = global_token_offset + s;
                int const dst_idx = global_token_idx * head_dim + d;
                kv_output[dst_idx] = kv_src[kv_block_idx];
            }
        }
    }
}

// k {total_token, h, uncompressed_h=128}, v {total_token, h, uncompressed_h}, k_pe {total_token, h, rope_h}
// output {b, 2, ceil(max_seq / kv_cache_tokens_per_block), h, kv_cache_tokens_per_block, (uncompressed_h + rope_h)}
// copy k, v, k_pe to a continuous memory space (then it will be packed to kv_cache)
template <typename T>
void setPagedKvCacheForMLAKernelRef(T* output, T* const k_ptr, T* const v_ptr, T* const k_pe_ptr, int num_requests,
    int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads, int uncompressed_head_size, int rope_size,
    int kv_cache_tokens_per_block)
{
    int const kv_cache_size_per_block = num_heads * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size);
    int const kv_cache_block_num_per_seq
        = (max_input_seq_len + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block;
    for (int b = 0; b < num_requests; b++)
    {
        int const global_token_offset = cu_seq_lens[b];
        int const current_token_len = cu_seq_lens[b + 1] - cu_seq_lens[b];
        for (int s = 0; s < current_token_len; s++)
        {
            int const global_token_idx = global_token_offset + s;
            int const kv_cache_block_offset_for_k
                = ((b * 2 * kv_cache_block_num_per_seq) + (s / kv_cache_tokens_per_block)) * kv_cache_size_per_block;
            int const kv_cache_block_offset_for_v
                = kv_cache_block_offset_for_k + (kv_cache_block_num_per_seq * kv_cache_size_per_block);
            for (int h = 0; h < num_heads; h++)
            {
                // copy k, v
                int const ld_kv_head_offset
                    = (global_token_idx * num_heads * uncompressed_head_size) + (h * uncompressed_head_size);
                int const ld_k_pe_head_offset = (global_token_idx * num_heads * rope_size) + (h * rope_size);
                for (int d = 0; d < uncompressed_head_size; d++)
                {
                    int const ld_kv_idx = ld_kv_head_offset + d;
                    int const st_k_idx = kv_cache_block_offset_for_k
                        + h * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size)
                        + (s % kv_cache_tokens_per_block) * (uncompressed_head_size + rope_size) + d;
                    int const st_v_idx = kv_cache_block_offset_for_v
                        + h * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size)
                        + (s % kv_cache_tokens_per_block) * (uncompressed_head_size + rope_size) + d;
                    output[st_k_idx] = k_ptr[ld_kv_idx];
                    output[st_v_idx] = v_ptr[ld_kv_idx];
                }
                // copy k_pe
                for (int d = 0; d < rope_size; d++)
                {
                    int const ld_k_pe_idx = ld_k_pe_head_offset + d;
                    int const st_k_pe_idx = kv_cache_block_offset_for_k
                        + h * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size)
                        + (s % kv_cache_tokens_per_block) * (uncompressed_head_size + rope_size) + d
                        + uncompressed_head_size;
                    output[st_k_pe_idx] = k_pe_ptr[ld_k_pe_idx];
                }
            }
        }
    }
}

// ck or cv {total_cached_token, h, uncompressed_h=128}, ck_pe {total_cached_token, h, rope_h}
// uk or uv {total_uncached_token, h, uncompressed_h}, uk_pe {total_uncached_token, h, rope_h}
// output {b, 2, ceil(max_seq / kv_cache_tokens_per_block), h, kv_cache_tokens_per_block, (uncompressed_h + rope_h)}
// copy k, v, k_pe to a continuous memory space (then it will be packed to kv_cache)
template <typename T>
void setPagedKvCacheForMLAKernelRefV2(T* output, T* const ck_ptr, T* const cv_ptr, T* const ck_pe_ptr, T* const nk_ptr,
    T* const nv_ptr, T* const nk_pe_ptr, int num_requests, int64_t const* cu_ctx_cached_kv_lens,
    int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads, int uncompressed_head_size, int rope_size,
    int kv_cache_tokens_per_block)
{
    int const kv_cache_size_per_block = num_heads * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size);
    int const kv_cache_block_num_per_seq
        = (max_input_seq_len + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block;
    for (int b = 0; b < num_requests; b++)
    {
        int const global_cached_token_offset = cu_ctx_cached_kv_lens[b];
        int const global_unchached_token_offset = cu_seq_lens[b] - cu_ctx_cached_kv_lens[b];
        int const current_token_len = cu_seq_lens[b + 1] - cu_seq_lens[b];
        int const current_cached_token_len = cu_ctx_cached_kv_lens[b + 1] - cu_ctx_cached_kv_lens[b];
        // int const current_uncached_token_len = current_token_len - current_cached_token_len;

        for (int s = 0; s < current_token_len; s++)
        {
            bool const is_cached = (s < current_cached_token_len);
            int const global_token_idx = is_cached ? global_cached_token_offset + s
                                                   : global_unchached_token_offset + (s - current_cached_token_len);
            int const kv_cache_block_offset_for_k
                = ((b * 2 * kv_cache_block_num_per_seq) + (s / kv_cache_tokens_per_block)) * kv_cache_size_per_block;
            int const kv_cache_block_offset_for_v
                = kv_cache_block_offset_for_k + (kv_cache_block_num_per_seq * kv_cache_size_per_block);
            auto const k_ptr = is_cached ? ck_ptr : nk_ptr;
            auto const v_ptr = is_cached ? cv_ptr : nv_ptr;
            auto const k_pe_ptr = is_cached ? ck_pe_ptr : nk_pe_ptr;
            for (int h = 0; h < num_heads; h++)
            {
                // copy k, v
                int const ld_kv_head_offset
                    = (global_token_idx * num_heads * uncompressed_head_size) + (h * uncompressed_head_size);
                int const ld_k_pe_head_offset = (global_token_idx * num_heads * rope_size) + (h * rope_size);
                for (int d = 0; d < uncompressed_head_size; d++)
                {
                    int const ld_kv_idx = ld_kv_head_offset + d;
                    int const st_k_idx = kv_cache_block_offset_for_k
                        + h * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size)
                        + (s % kv_cache_tokens_per_block) * (uncompressed_head_size + rope_size) + d;
                    int const st_v_idx = kv_cache_block_offset_for_v
                        + h * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size)
                        + (s % kv_cache_tokens_per_block) * (uncompressed_head_size + rope_size) + d;
                    output[st_k_idx] = k_ptr[ld_kv_idx];
                    output[st_v_idx] = v_ptr[ld_kv_idx];
                }
                // copy k_pe
                for (int d = 0; d < rope_size; d++)
                {
                    int const ld_k_pe_idx = ld_k_pe_head_offset + d;
                    int const st_k_pe_idx = kv_cache_block_offset_for_k
                        + h * kv_cache_tokens_per_block * (uncompressed_head_size + rope_size)
                        + (s % kv_cache_tokens_per_block) * (uncompressed_head_size + rope_size) + d
                        + uncompressed_head_size;
                    output[st_k_pe_idx] = k_pe_ptr[ld_k_pe_idx];
                }
            }
        }
    }
}

// compressed_kv_cache {batch, 1 (ignore v), max_seq_len / tokens_per_block, num_head=1, tokens_per_block, (lora_size +
// rope_size)}
// kv {total_uncached_tokens, h_k=1, lora_d}, k_pe {total_uncached_tokens, h_kpe=128, rope_d}
template <typename T>
void setCompressedPagedKvForMLAKernelRef(tensorrt_llm::kernels::KVBlockArray& kv_cache, T* const compressed_kv_ptr,
    T* const k_pe_ptr, int const num_requests, int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens,
    int k_pe_head_num, int lora_size, int rope_size)
{
    assert(k_pe_head_num == 1);
    for (int b = 0; b < num_requests; b++)
    {
        int const global_token_offset = cu_seq_lens[b] - cu_ctx_cached_kv_lens[b];
        int const cached_kv_len = cu_ctx_cached_kv_lens[b + 1] - cu_ctx_cached_kv_lens[b];
        int const uncached_token_len = cu_seq_lens[b + 1] - cu_seq_lens[b] - cached_kv_len;
        for (int s = 0; s < uncached_token_len; s++)
        {
            int const ld_kv_offset = (global_token_offset + s) * lora_size;
            int const ld_k_pe_offset = (global_token_offset + s) * k_pe_head_num * rope_size;
            auto* kv_cache_ptr = reinterpret_cast<T*>(kv_cache.getKBlockPtr(b, cached_kv_len + s));
            // copy kv
            for (int d = 0; d < lora_size; d++)
            {
                int const ld_kv_idx = ld_kv_offset + d;
                int const kv_cache_idx_in_block
                    = kv_cache.getKVLocalIdx(cached_kv_len + s, 0, lora_size + rope_size, d);
                kv_cache_ptr[kv_cache_idx_in_block] = compressed_kv_ptr[ld_kv_idx];
            }
            // copy k_pe (we only copy the first head)
            for (int d = 0; d < rope_size; d++)
            {
                int const ld_k_pe_idx = ld_k_pe_offset + d;
                int const kv_cache_idx_in_block
                    = kv_cache.getKVLocalIdx(cached_kv_len + s, 0, lora_size + rope_size, d + lora_size);
                kv_cache_ptr[kv_cache_idx_in_block] = k_pe_ptr[ld_k_pe_idx];
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

template <typename _DataType>
class MlaPreprocessTest : public testing::Test
{
protected:
    using DataType = _DataType;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    // kv_cache shape {batch, 2(k or v), max_seq_len / tokens_per_block, num_head, tokens_per_block, head_size}
    // k, v, k_pe shape {total_token, num_head, head_size(lora_size or rope_size, or uncompressed_head_size)}
    // offset shape {batch, 2, max_seq_len / tokens_per_block}
    // for KVBlockArray, we only allocate primary pool.
    // you can infer the allocateBuffers function for more details.
    tensorrt_llm::runtime::BufferManager::ITensorPtr kv_cache_tensor{nullptr}, kv_cache_tensor_ref{nullptr},
        compressed_kv_cache_tensor{nullptr}, compressed_kv_cache_tensor_ref{nullptr},
        // for kernel 1
        kv_k_pe_tensor{nullptr}, kv_k_pe_tensor_ref{nullptr},
        // for kernel 2
        k_tensor{nullptr}, v_tensor{nullptr}, k_pe_tensor{nullptr},
        // for kernel 2 (new)
        k_tensor_cached{nullptr}, v_tensor_cached{nullptr}, k_pe_tensor_cached{nullptr}, k_tensor_uncached{nullptr},
        v_tensor_uncached{nullptr}, k_pe_tensor_uncached{nullptr},
        // for kernel 3
        compressed_kv_tensor{nullptr}, k_pe_full_head_tensor{nullptr}, offset_tensor{nullptr},
        compressed_offset_tensor{nullptr}, cu_ctx_cached_kv_lens{nullptr}, cu_seq_lens{nullptr};

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
    void memsetZero(T* ptr, size_t size)
    {
        cudaMemset(ptr, 0, size * sizeof(T));
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
#ifdef ENABLE_BF16
        else if constexpr (std::is_same_v<DataType, __nv_bfloat16>)
        {
            dtype = nvinfer1::DataType::kBF16;
        }
#endif
        else
        {
            return false;
        }
        this->cu_seq_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests + 1}), nvinfer1::DataType::kINT64);
        this->cu_ctx_cached_kv_lens = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests + 1}), nvinfer1::DataType::kINT64);
        {
            // set random sequence length
            auto* cu_seq_lens_temp_ptr = bufferCast<int64_t>(*(this->cu_seq_lens));
            auto* cu_ctx_cached_kv_lens_temp_ptr = bufferCast<int64_t>(*(this->cu_ctx_cached_kv_lens));
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
        }

        // malloc kv_cache
        this->mMaxBlockPerSeq = (this->mMaxSeqLen + this->mTokensPerBlock - 1) / this->mTokensPerBlock;
        this->kv_cache_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq, this->mNumHeadsUncompressed,
                this->mTokensPerBlock, this->mUncompressedHeadSize + this->mRopeSize}),
            dtype);
        this->kv_cache_tensor_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq, this->mNumHeadsUncompressed,
                this->mTokensPerBlock, this->mUncompressedHeadSize + this->mRopeSize}),
            dtype);
        this->compressed_kv_cache_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 1, this->mMaxBlockPerSeq, this->mNumHeadsCompressed,
                this->mTokensPerBlock, this->mLoraSize + this->mRopeSize}),
            dtype);
        this->compressed_kv_cache_tensor_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 1, this->mMaxBlockPerSeq, this->mNumHeadsCompressed,
                this->mTokensPerBlock, this->mLoraSize + this->mRopeSize}),
            dtype);
        this->offset_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq}), nvinfer1::DataType::kINT32);
        this->compressed_offset_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mNumRequests, 2, this->mMaxBlockPerSeq}), nvinfer1::DataType::kINT32);

        {
            auto* kv_cache_ptr = bufferCast<DataType>(*(this->kv_cache_tensor));
            auto* kv_cache_ref_ptr = bufferCast<DataType>(*(this->kv_cache_tensor_ref));
            auto* compressed_kv_cache_ptr = bufferCast<DataType>(*(this->compressed_kv_cache_tensor));
            auto* compressed_kv_cache_ref_ptr = bufferCast<DataType>(*(this->compressed_kv_cache_tensor_ref));
            auto* offset_ptr = bufferCast<int32_t>(*(this->offset_tensor));
            auto* compressed_offset_ptr = bufferCast<int32_t>(*(this->compressed_offset_tensor));
            fillArrayDataWithMod(compressed_kv_cache_ptr, this->compressed_kv_cache_tensor->getSize());
            fillArrayDataWithMod(compressed_kv_cache_ref_ptr, this->compressed_kv_cache_tensor_ref->getSize());
            memsetZero<DataType>(kv_cache_ptr, this->kv_cache_tensor->getSize());
            memsetZero<DataType>(kv_cache_ref_ptr, this->kv_cache_tensor_ref->getSize());
            // fillArrayDataWithMod(offset_ptr, this->offset_tensor->getSize());
            fillKVOffsetData(
                compressed_offset_ptr, this->compressed_offset_tensor->getSize(), false, this->mMaxBlockPerSeq);
        }

        // kv + k_pe for loadCompressedPagedKvKernel (kernel 1)
        // std::cout << "kv_cache_tensor size: {" << this->mTotalCachedTokens << ", 1, " << this->mLoraSize +
        // this->mRopeSize <<  "}" << std::endl;
        this->kv_k_pe_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, 1, this->mLoraSize + this->mRopeSize}), dtype);
        this->kv_k_pe_tensor_ref = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, 1, this->mLoraSize + this->mRopeSize}), dtype);
        {
            auto* kv_k_pe_ptr = bufferCast<DataType>(*(this->kv_k_pe_tensor));
            auto* kv_k_pe_ref_ptr = bufferCast<DataType>(*(this->kv_k_pe_tensor_ref));
            memsetZero<DataType>(kv_k_pe_ptr, this->kv_k_pe_tensor->getSize());
            memsetZero<DataType>(kv_k_pe_ref_ptr, this->kv_k_pe_tensor_ref->getSize());
        }
        // k, v, k_pe for setPagedKvCacheForMLAKernel (kernel 2)
        this->k_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalTokens, this->mNumHeadsUncompressed, this->mUncompressedHeadSize}), dtype);
        this->v_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalTokens, this->mNumHeadsUncompressed, this->mUncompressedHeadSize}), dtype);
        this->k_pe_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalTokens, this->mNumHeadsUncompressed, this->mRopeSize}), dtype);
        {
            auto* k_ptr = bufferCast<DataType>(*(this->k_tensor));
            auto* v_ptr = bufferCast<DataType>(*(this->v_tensor));
            auto* k_pe_ptr = bufferCast<DataType>(*(this->k_pe_tensor));
            fillArrayDataWithMod(k_ptr, this->k_tensor->getSize());
            fillArrayDataWithMod(v_ptr, this->v_tensor->getSize());
            fillArrayDataWithMod(k_pe_ptr, this->k_pe_tensor->getSize());
        }
        // ck, cv, ck_pe, uk, uc, uk_pe for setPagedKvCacheForMLAKernelV2 (kernel 2)
        this->k_tensor_cached = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, this->mNumHeadsUncompressed, this->mUncompressedHeadSize}),
            dtype);
        this->v_tensor_cached = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, this->mNumHeadsUncompressed, this->mUncompressedHeadSize}),
            dtype);
        this->k_pe_tensor_cached = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalCachedTokens, this->mNumHeadsUncompressed, this->mRopeSize}), dtype);
        this->k_tensor_uncached = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalUncachedTokens, this->mNumHeadsUncompressed, this->mUncompressedHeadSize}),
            dtype);
        this->v_tensor_uncached = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalUncachedTokens, this->mNumHeadsUncompressed, this->mUncompressedHeadSize}),
            dtype);
        this->k_pe_tensor_uncached = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalUncachedTokens, this->mNumHeadsUncompressed, this->mRopeSize}), dtype);
        {
            auto* k_cached_ptr = bufferCast<DataType>(*(this->k_tensor_cached));
            auto* v_cached_ptr = bufferCast<DataType>(*(this->v_tensor_cached));
            auto* k_pe_cached_ptr = bufferCast<DataType>(*(this->k_pe_tensor_cached));
            auto* k_uncached_ptr = bufferCast<DataType>(*(this->k_tensor_uncached));
            auto* v_uncached_ptr = bufferCast<DataType>(*(this->v_tensor_uncached));
            auto* k_pe_uncached_ptr = bufferCast<DataType>(*(this->k_pe_tensor_uncached));
            fillArrayDataWithMod(k_cached_ptr, this->k_tensor_cached->getSize());
            fillArrayDataWithMod(v_cached_ptr, this->v_tensor_cached->getSize());
            fillArrayDataWithMod(k_pe_cached_ptr, this->k_pe_tensor_cached->getSize());
            fillArrayDataWithMod(k_uncached_ptr, this->k_tensor_uncached->getSize());
            fillArrayDataWithMod(v_uncached_ptr, this->v_tensor_uncached->getSize());
            fillArrayDataWithMod(k_pe_uncached_ptr, this->k_pe_tensor_uncached->getSize());
        }
        // compressed_kv, k_pe_full_head for setCompressedPagedKvForMLAKernel (kernel 3)
        this->compressed_kv_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalUncachedTokens, 1, this->mLoraSize}), dtype);
        this->k_pe_full_head_tensor = tensorrt_llm::runtime::BufferManager::pinned(
            ITensor::makeShape({this->mTotalUncachedTokens, 1, this->mRopeSize}), dtype);
        {
            auto* compressed_kv_ptr = bufferCast<DataType>(*(this->compressed_kv_tensor));
            auto* k_pe_full_head_ptr = bufferCast<DataType>(*(this->k_pe_full_head_tensor));
            fillArrayDataWithMod(compressed_kv_ptr, this->compressed_kv_tensor->getSize());
            fillArrayDataWithMod(k_pe_full_head_ptr, this->k_pe_full_head_tensor->getSize());
        }
        return true;
    }

    void PerformKernel1()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* kv_k_pe_ptr = bufferCast<DataType>(*(this->kv_k_pe_tensor));
        auto* compressed_kv_cache_ptr = bufferCast<DataType>(*(this->compressed_kv_cache_tensor));
        auto* offset_ptr = bufferCast<int32_t>(*(this->compressed_offset_tensor));
        auto* cu_ctx_cached_kv_lens_ptr = bufferCast<int64_t>(*(this->cu_ctx_cached_kv_lens));
        tensorrt_llm::kernels::KVBlockArray kv_cache(this->mNumRequests, this->mMaxBlockPerSeq, this->mTokensPerBlock,
            sizeof(DataType) * 1 * (this->mLoraSize + this->mRopeSize), 0, 0, 0, 0, compressed_kv_cache_ptr, nullptr,
            reinterpret_cast<tensorrt_llm::kernels::KVBlockArrayForContextFMHA::DataType*>(offset_ptr));
        tensorrt_llm::kernels::invokeLoadPagedKVKernel<DataType>(kv_k_pe_ptr, kv_cache, this->mNumRequests,
            cu_ctx_cached_kv_lens_ptr, this->mMaxCachedSeqLen, this->mLoraSize + this->mRopeSize, this->mStream->get());
    }

    void PerformKernel1Ref()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* kv_k_pe_ptr = bufferCast<DataType>(*(this->kv_k_pe_tensor_ref));
        auto* compressed_kv_cache_ptr = bufferCast<DataType>(*(this->compressed_kv_cache_tensor));
        auto* offset_ptr = bufferCast<int32_t>(*(this->compressed_offset_tensor));
        auto* cu_ctx_cached_kv_lens_ptr = bufferCast<int64_t>(*(this->cu_ctx_cached_kv_lens));
        tensorrt_llm::kernels::KVBlockArray kv_cache(this->mNumRequests, this->mMaxBlockPerSeq, this->mTokensPerBlock,
            sizeof(DataType) * 1 * (this->mLoraSize + this->mRopeSize), 0, 0, 0, 0, compressed_kv_cache_ptr, nullptr,
            reinterpret_cast<tensorrt_llm::kernels::KVBlockArrayForContextFMHA::DataType*>(offset_ptr));
        loadCompressedPagedKvKernelRef(
            kv_k_pe_ptr, kv_cache, this->mNumRequests, cu_ctx_cached_kv_lens_ptr, this->mLoraSize + this->mRopeSize);
    }

    void PerformKernel2()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* k_ptr = bufferCast<DataType>(*(this->k_tensor));
        auto* v_ptr = bufferCast<DataType>(*(this->v_tensor));
        auto* k_pe_ptr = bufferCast<DataType>(*(this->k_pe_tensor));
        auto* kv_cache_ptr = bufferCast<DataType>(*(this->kv_cache_tensor));
        auto* cu_seq_lens_ptr = bufferCast<int64_t>(*(this->cu_seq_lens));
        tensorrt_llm::kernels::invokeSetPagedKVKernel<DataType>(kv_cache_ptr, k_ptr, v_ptr, k_pe_ptr,
            this->mNumRequests, cu_seq_lens_ptr, this->mMaxSeqLen, this->mNumHeadsUncompressed,
            this->mUncompressedHeadSize, this->mRopeSize, this->mTokensPerBlock, this->mStream->get());
    }

    void PerformKernel2Ref()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* k_ptr = bufferCast<DataType>(*(this->k_tensor));
        auto* v_ptr = bufferCast<DataType>(*(this->v_tensor));
        auto* k_pe_ptr = bufferCast<DataType>(*(this->k_pe_tensor));
        auto* kv_cache_ptr = bufferCast<DataType>(*(this->kv_cache_tensor_ref));
        auto* cu_seq_lens_ptr = bufferCast<int64_t>(*(this->cu_seq_lens));
        setPagedKvCacheForMLAKernelRef(kv_cache_ptr, k_ptr, v_ptr, k_pe_ptr, this->mNumRequests, cu_seq_lens_ptr,
            this->mMaxSeqLen, this->mNumHeadsUncompressed, this->mUncompressedHeadSize, this->mRopeSize,
            this->mTokensPerBlock);
    }

    void PerformKernel2V2()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* k_cached_ptr = bufferCast<DataType>(*(this->k_tensor_cached));
        auto* v_cached_ptr = bufferCast<DataType>(*(this->v_tensor_cached));
        auto* k_pe_cached_ptr = bufferCast<DataType>(*(this->k_pe_tensor_cached));
        auto* k_uncached_ptr = bufferCast<DataType>(*(this->k_tensor_uncached));
        auto* v_uncached_ptr = bufferCast<DataType>(*(this->v_tensor_uncached));
        auto* k_pe_uncached_ptr = bufferCast<DataType>(*(this->k_pe_tensor_uncached));
        auto* cu_ctx_cached_kv_lens_ptr = bufferCast<int64_t>(*(this->cu_ctx_cached_kv_lens));
        auto* kv_cache_ptr = bufferCast<DataType>(*(this->kv_cache_tensor));
        auto* cu_seq_lens_ptr = bufferCast<int64_t>(*(this->cu_seq_lens));
        tensorrt_llm::kernels::invokeSetPagedKVKernelV2<DataType>(kv_cache_ptr, k_cached_ptr, v_cached_ptr,
            k_pe_cached_ptr, k_uncached_ptr, v_uncached_ptr, k_pe_uncached_ptr, this->mNumRequests,
            cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr, this->mMaxSeqLen, this->mNumHeadsUncompressed,
            this->mUncompressedHeadSize, this->mRopeSize, this->mTokensPerBlock, this->mStream->get());
    }

    void PerformKernel2V2Ref()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* k_cached_ptr = bufferCast<DataType>(*(this->k_tensor_cached));
        auto* v_cached_ptr = bufferCast<DataType>(*(this->v_tensor_cached));
        auto* k_pe_cached_ptr = bufferCast<DataType>(*(this->k_pe_tensor_cached));
        auto* k_uncached_ptr = bufferCast<DataType>(*(this->k_tensor_uncached));
        auto* v_uncached_ptr = bufferCast<DataType>(*(this->v_tensor_uncached));
        auto* k_pe_uncached_ptr = bufferCast<DataType>(*(this->k_pe_tensor_uncached));
        auto* cu_ctx_cached_kv_lens_ptr = bufferCast<int64_t>(*(this->cu_ctx_cached_kv_lens));
        auto* kv_cache_ptr = bufferCast<DataType>(*(this->kv_cache_tensor_ref));
        auto* cu_seq_lens_ptr = bufferCast<int64_t>(*(this->cu_seq_lens));
        setPagedKvCacheForMLAKernelRefV2(kv_cache_ptr, k_cached_ptr, v_cached_ptr, k_pe_cached_ptr, k_uncached_ptr,
            v_uncached_ptr, k_pe_uncached_ptr, this->mNumRequests, cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr,
            this->mMaxSeqLen, this->mNumHeadsUncompressed, this->mUncompressedHeadSize, this->mRopeSize,
            this->mTokensPerBlock);
    }

    void PerformKernel3()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* compressed_kv_ptr = bufferCast<DataType>(*(this->compressed_kv_tensor));
        auto* k_pe_full_head_ptr = bufferCast<DataType>(*(this->k_pe_full_head_tensor));
        auto* offset_ptr = bufferCast<int32_t>(*(this->compressed_offset_tensor));
        auto* compressed_kv_cache_ptr = bufferCast<DataType>(*(this->compressed_kv_cache_tensor));
        auto* cu_ctx_cached_kv_lens_ptr = bufferCast<int64_t>(*(this->cu_ctx_cached_kv_lens));
        auto* cu_seq_lens_ptr = bufferCast<int64_t>(*(this->cu_seq_lens));
        tensorrt_llm::kernels::KVBlockArray kv_cache(this->mNumRequests, this->mMaxBlockPerSeq, this->mTokensPerBlock,
            sizeof(DataType) * 1 * (this->mLoraSize + this->mRopeSize), 0, 0, 0, 0, compressed_kv_cache_ptr, nullptr,
            reinterpret_cast<tensorrt_llm::kernels::KVBlockArrayForContextFMHA::DataType*>(offset_ptr));
        tensorrt_llm::kernels::invokeSetCompressedPagedKVKernel<DataType>(kv_cache, compressed_kv_ptr,
            k_pe_full_head_ptr, this->mNumRequests, cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr,
            this->mMaxUncachedSeqLen, this->mLoraSize + this->mRopeSize, this->mStream->get());
    }

    void PerformKernel3Ref()
    {
        using tensorrt_llm::runtime::bufferCast;
        auto* compressed_kv_ptr = bufferCast<DataType>(*(this->compressed_kv_tensor));
        auto* k_pe_full_head_ptr = bufferCast<DataType>(*(this->k_pe_full_head_tensor));
        auto* offset_ptr = bufferCast<int32_t>(*(this->compressed_offset_tensor));
        auto* compressed_kv_cache_ptr = bufferCast<DataType>(*(this->compressed_kv_cache_tensor_ref));
        auto* cu_ctx_cached_kv_lens_ptr = bufferCast<int64_t>(*(this->cu_ctx_cached_kv_lens));
        auto* cu_seq_lens_ptr = bufferCast<int64_t>(*(this->cu_seq_lens));
        tensorrt_llm::kernels::KVBlockArray kv_cache(this->mNumRequests, this->mMaxBlockPerSeq, this->mTokensPerBlock,
            sizeof(DataType) * 1 * (this->mLoraSize + this->mRopeSize), 0, 0, 0, 0, compressed_kv_cache_ptr, nullptr,
            reinterpret_cast<tensorrt_llm::kernels::KVBlockArrayForContextFMHA::DataType*>(offset_ptr));
        // currently k_pe_head_num = 1
        setCompressedPagedKvForMLAKernelRef(kv_cache, compressed_kv_ptr, k_pe_full_head_ptr, this->mNumRequests,
            cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr, 1, this->mLoraSize, this->mRopeSize);
    }

    template <typename T>
    bool CheckEqual(T const* expected, T const* output, size_t size)
    {
        for (int i = 0; i < size; i++)
        {
            if (!almostEqual(expected[i], output[i], 1e-3, 1e-3))
            {
                TLLM_LOG_ERROR("Mismatch input value. Position of inputs: %d, expected value: %f, output value: %f", i,
                    static_cast<float>(expected[i]), static_cast<float>(output[i]));
                return false;
            }
        }
        return true;
    }
};

using MLATypes = ::testing::Types<half,
#ifdef ENABLE_BF16
    __nv_bfloat16,
#endif
    float>;
TYPED_TEST_SUITE(MlaPreprocessTest, MLATypes);

TYPED_TEST(MlaPreprocessTest, MLAPreprocessDefault)
{
    using tensorrt_llm::runtime::bufferCast;
    using DataType = typename TestFixture::DataType;
    this->mNumRequests = 8;
    this->setDefaultParams();
    this->allocateBuffers();

    sync_check_cuda_error(this->mStream->get());
    bool allEqual{true};

    this->PerformKernel1();
    sync_check_cuda_error(this->mStream->get());
    this->PerformKernel1Ref();
    auto* kv_k_pe_ptr = bufferCast<DataType>(*(this->kv_k_pe_tensor));
    auto* kv_k_pe_ref_ptr = bufferCast<DataType>(*(this->kv_k_pe_tensor_ref));
    allEqual = this->CheckEqual(kv_k_pe_ref_ptr, kv_k_pe_ptr, this->kv_k_pe_tensor->getSize());
    EXPECT_TRUE(allEqual);

    this->PerformKernel2V2();
    sync_check_cuda_error(this->mStream->get());
    this->PerformKernel2V2Ref();
    auto* kv_cache_ptr = bufferCast<DataType>(*(this->kv_cache_tensor));
    auto* kv_cache_ref_ptr = bufferCast<DataType>(*(this->kv_cache_tensor_ref));
    allEqual = this->CheckEqual(kv_cache_ref_ptr, kv_cache_ptr, this->kv_cache_tensor->getSize());
    EXPECT_TRUE(allEqual);

    this->PerformKernel3();
    sync_check_cuda_error(this->mStream->get());
    this->PerformKernel3Ref();
    auto* compressed_kv_cache_ptr = bufferCast<DataType>(*(this->compressed_kv_cache_tensor));
    auto* compressed_kv_cache_ref_ptr = bufferCast<DataType>(*(this->compressed_kv_cache_tensor_ref));
    allEqual = this->CheckEqual(
        compressed_kv_cache_ref_ptr, compressed_kv_cache_ptr, this->compressed_kv_cache_tensor->getSize());
    EXPECT_TRUE(allEqual);
}
