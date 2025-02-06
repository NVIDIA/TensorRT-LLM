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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <cstdint>
#include <cstdlib>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

namespace
{

template <typename T>
void randomInitVector(std::vector<T>& vec, float range)
{
    for (auto& v : vec)
    {
        float r = range * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        if (std::is_same_v<T, float>)
        {
            v = r;
        }
        else if (std::is_same_v<T, half>)
        {
            v = __float2half(r);
        }
    }
}

template void randomInitVector(std::vector<float>& vec, float scale);
template void randomInitVector(std::vector<half>& vec, float scale);

std::vector<KVCacheIndex> offsetsArrayFromPageTable(
    std::unordered_map<int, int> const& pageTable, int32_t batchSize, int32_t blocksPerSeq, int32_t blocksPerPool)
{
    auto const offsetsArrayElts = pageTable.size();
    std::vector<KVCacheIndex> offsets(2 * offsetsArrayElts, KVCacheIndex{0});
    for (int i = 0; i < offsetsArrayElts; ++i)
    {
        int const pageIdx = pageTable.find(i)->second;
        auto const kOffset = KVCacheIndex{pageIdx};
        auto const vOffset = KVCacheIndex{pageIdx + blocksPerPool};
        int const batchIdx = i / batchSize;
        int const seqIdx = i % blocksPerSeq;
        offsets[batchIdx * blocksPerSeq * 2 + 0 * blocksPerSeq + seqIdx] = kOffset;
        offsets[batchIdx * blocksPerSeq * 2 + 1 * blocksPerSeq + seqIdx] = vOffset;
    }
    return offsets;
}

template <typename T, typename T_DST>
T_DST castTo(T value)
{
    return value;
}

template <>
int8_t castTo(float value)
{
    auto const clipped = std::min(127.f, std::max(value, -128.f));
    auto const rounded = std::round(clipped);
    return static_cast<int8_t>(rounded);
}

template <>
__nv_fp8_e4m3 castTo(float value)
{
    return __nv_fp8_e4m3(value);
}

template <>
float castTo(__nv_fp8_e4m3 value)
{
    return float(value);
}

template <typename T, typename T_DST, typename KVCacheBuffer>
void verifyKVTransposed(int batchSize, int headsNum, int dimsPerHead, int seqLen, int maxSeqLen, KVCacheBuffer& buffer,
    std::vector<T> const& refKCacheVec, std::vector<T> const& vTransposedCacheVec, bool b8bitKVCache,
    float kvScaleOrigQuant)
{
    for (int bi = 0; bi < batchSize; ++bi)
    {
        for (int hi = 0; hi < headsNum; ++hi)
        {
            constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
            for (int di = 0; di < dimsPerHead / X_ELEMS; ++di)
            {
                for (int li = 0; li < seqLen; ++li)
                {
                    const T_DST* blockKPtr = reinterpret_cast<T_DST*>(buffer.getKBlockPtr(bi, li));
                    const T_DST* blockVPtr = reinterpret_cast<T_DST*>(buffer.getVBlockPtr(bi, li));

                    for (int xi = 0; xi < X_ELEMS; ++xi)
                    {
                        int const refKVIdx = bi * headsNum * seqLen * dimsPerHead + hi * seqLen * dimsPerHead
                            + li * dimsPerHead + di * X_ELEMS + xi;

                        int const kVIdx = buffer.getKVLocalIdx(li, hi, dimsPerHead, di * X_ELEMS + xi);

                        T refK = refKCacheVec[refKVIdx];
                        T refV = vTransposedCacheVec[refKVIdx];
                        if (b8bitKVCache)
                        {
                            refK = castTo<float, T>(castTo<T, float>(refK) * kvScaleOrigQuant);
                            refV = castTo<float, T>(castTo<T, float>(refV) * kvScaleOrigQuant);
                        }

                        const T_DST castedRefK = castTo<T, T_DST>(refK);
                        const T_DST castedRefV = castTo<T, T_DST>(refV);

                        auto const outK = blockKPtr[kVIdx];
                        auto const outV = blockVPtr[kVIdx];

                        // Since EXPECT_EQ does not support fp8, casting to float to compare
                        float const outK_float = castTo<T_DST, float>(outK);
                        float const outV_float = castTo<T_DST, float>(outV);
                        float const castedRefK_float = castTo<T_DST, float>(castedRefK);
                        float const castedRefV_float = castTo<T_DST, float>(castedRefV);
                        EXPECT_EQ(outK_float, castedRefK_float);
                        EXPECT_EQ(outV_float, castedRefV_float);
                    }
                }
            }
        }
    }
}

template <typename T, typename T_DST>
void testTransposeBatch4dPaged(bool multiQueryMode, bool int8KVCache, bool fp8KVCache)
{
    // Fix seed
    srand(42);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    constexpr int32_t tokensPerBlock{8};
    constexpr int32_t maxBlocksPerSeq{64};
    constexpr int32_t maxSeq{64};
    constexpr int32_t batchSize{2};
    int32_t const headsNum = multiQueryMode ? 1 : 8;
    constexpr int32_t seqLen{16};
    constexpr int32_t maxSeqLen{2 * seqLen};
    constexpr int32_t dimsPerHead{256};
    constexpr int8_t elemSize{sizeof(T_DST)};
    constexpr int32_t blockSize = tokensPerBlock * dimsPerHead;
    constexpr int32_t bytesPerBlock = blockSize * elemSize;
    constexpr int32_t bytesPerToken = dimsPerHead * elemSize;
    constexpr int32_t maxAttentionWindow = maxSeqLen;
    constexpr int32_t sinkTokenLen{0};
    constexpr int32_t onlyKorV{false};
    constexpr bool canUseOneMoreBlock{true};

    TLLM_CHECK_WITH_INFO(batchSize <= maxSeq, "Batch size is larger than max number of allowed sequence");
    TLLM_CHECK_WITH_INFO(headsNum * seqLen <= maxBlocksPerSeq * tokensPerBlock,
        "Total amount of tokens is less than max amount of tokens is cache per sequence");

    // Allocate for kv cache block pool
    auto const blocksPerPool = maxBlocksPerSeq * maxSeq;
    auto const kvPoolSize = 2 * bytesPerBlock * blocksPerPool;
    void* kvMemoryPool = nullptr;
    cudaMalloc(&kvMemoryPool, kvPoolSize);
    cudaMemset(kvMemoryPool, 0, kvPoolSize);

    // Allocate offsets array
    std::remove_const_t<KVBlockArray::DataType>* offsetsArray = nullptr;
    auto const offsetsArrayElts = maxSeq * maxBlocksPerSeq;
    auto const offsetsArraySize = 2 * offsetsArrayElts * sizeof(int64_t);
    cudaMalloc(&offsetsArray, offsetsArraySize);
    cudaMemset(offsetsArray, 0, offsetsArraySize);

    // Create page table
    std::unordered_map<int, int> mapIndicesTable;
    for (int i = 0; i < offsetsArrayElts; ++i)
    {
        int value;
        int idx = i;
        if (idx % 2 == 0)
        {
            value = idx / 2;
        }
        else
        {
            value = offsetsArrayElts / 2 + idx / 2;
        }

        mapIndicesTable[idx] = value;
    }

    // Init array of pointer from page table
    auto const offsets = offsetsArrayFromPageTable(mapIndicesTable, maxSeq, maxBlocksPerSeq, blocksPerPool);
    cudaMemcpy(offsetsArray, offsets.data(), offsetsArraySize, cudaMemcpyHostToDevice);

    auto blockArray = KVBlockArray(maxSeq, maxBlocksPerSeq, tokensPerBlock, bytesPerToken, maxAttentionWindow,
        maxAttentionWindow, sinkTokenLen, canUseOneMoreBlock, kvMemoryPool, nullptr, offsetsArray);

    float kvScaleOrigQuant = 1.0f;
    float* kvScaleOrigQuantPtr = nullptr;
    if (int8KVCache || fp8KVCache)
    {
        kvScaleOrigQuant = 0.1f;
        cudaMalloc(&kvScaleOrigQuantPtr, sizeof(float));
        cudaMemcpy(kvScaleOrigQuantPtr, &kvScaleOrigQuant, sizeof(float), cudaMemcpyHostToDevice);
    }
    int* sequenceLengths = nullptr;
    cudaMalloc(&sequenceLengths, sizeof(int) * batchSize);
    tensorrt_llm::common::deviceFill(sequenceLengths, batchSize, seqLen, streamPtr->get());

    // set up inputs
    std::vector<T> kTransposedCacheVec(batchSize * headsNum * seqLen * dimsPerHead);
    std::vector<T> vTransposedCacheVec(batchSize * headsNum * seqLen * dimsPerHead);
    randomInitVector(kTransposedCacheVec, 1.f / kvScaleOrigQuant);
    randomInitVector(vTransposedCacheVec, 1.f / kvScaleOrigQuant);

    // Copy inputs to GPU
    auto kTransposedCache = std::shared_ptr(manager.copyFrom(
        kTransposedCacheVec, ITensor::makeShape({batchSize, headsNum, seqLen, dimsPerHead}), MemoryType::kGPU));
    auto vTransposedCache = std::shared_ptr(manager.copyFrom(
        vTransposedCacheVec, ITensor::makeShape({batchSize, headsNum, seqLen, dimsPerHead}), MemoryType::kGPU));

    // Run inference
    KvCacheDataType const cache_type
        = int8KVCache ? KvCacheDataType::INT8 : (fp8KVCache ? KvCacheDataType::FP8 : KvCacheDataType::BASE);
    invokeTranspose4dBatchMajor(bufferCast<T>(*kTransposedCache), bufferCast<T>(*vTransposedCache), blockArray,
        batchSize, seqLen, maxSeqLen, dimsPerHead, headsNum, cache_type, kvScaleOrigQuantPtr, sequenceLengths,
        streamPtr->get());

    // Synchronize
    streamPtr->synchronize();

    // Copy pool to CPU
    std::vector<T_DST> kvMemoryPoolHost(kvPoolSize);
    cudaMemcpy(kvMemoryPoolHost.data(), kvMemoryPool, kvPoolSize, cudaMemcpyDeviceToHost);
    KVBlockArray blockArrayHost = blockArray;
    blockArrayHost.mPrimaryPoolPtr = kvMemoryPoolHost.data();

    // Init array of CPU pointers from page table
    auto offsetsHost = offsetsArrayFromPageTable(mapIndicesTable, maxSeq, maxBlocksPerSeq, blocksPerPool);
    blockArrayHost.data = offsetsHost.data();

    verifyKVTransposed<T, T_DST>(batchSize, headsNum, dimsPerHead, seqLen, maxSeqLen, blockArrayHost,
        kTransposedCacheVec, vTransposedCacheVec, int8KVCache || fp8KVCache, kvScaleOrigQuant);

    cudaFree(sequenceLengths);
    if (int8KVCache || fp8KVCache)
    {
        cudaFree(kvScaleOrigQuantPtr);
    }
}

template <typename T, typename T_DST>
void testTransposeBatch4dContiguous(bool multiQueryMode, bool int8KVCache, bool fp8KVCache)
{
    // Fix seed
    srand(42);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    constexpr int32_t batchSize{2};
    int32_t const headsNum = multiQueryMode ? 1 : 8;
    constexpr int32_t seqLen{16};
    constexpr int32_t maxSeqLen{2 * seqLen};
    constexpr int32_t dimsPerHead{256};
    constexpr int32_t maxAttentionWindow = maxSeqLen;
    constexpr int32_t sinkTokenLen{0};
    constexpr int32_t onlyKorV{false};

    KVLinearBuffer kvLinearBuffer(batchSize, maxSeqLen, dimsPerHead * headsNum * sizeof(T_DST), maxAttentionWindow,
        sinkTokenLen, onlyKorV, nullptr);

    // Allocate for kv cache pool
    auto const kvPoolElts = 2 * batchSize * maxSeqLen * dimsPerHead * headsNum;
    auto const kvPoolSize = kvPoolElts * sizeof(T_DST);
    cudaMalloc(&kvLinearBuffer.data, kvPoolSize);
    cudaMemset(kvLinearBuffer.data, 0, kvPoolSize);

    float kvScaleOrigQuant = 1.0f;
    float* kvScaleOrigQuantPtr = nullptr;
    if (int8KVCache || fp8KVCache)
    {
        kvScaleOrigQuant = 0.1f;
        cudaMalloc(&kvScaleOrigQuantPtr, sizeof(float));
        cudaMemcpy(kvScaleOrigQuantPtr, &kvScaleOrigQuant, sizeof(float), cudaMemcpyHostToDevice);
    }
    int* sequenceLengths = nullptr;
    cudaMalloc(&sequenceLengths, sizeof(int) * batchSize);
    tensorrt_llm::common::deviceFill(sequenceLengths, batchSize, seqLen, streamPtr->get());

    // set up inputs
    std::vector<T> kTransposedCacheVec(batchSize * headsNum * seqLen * dimsPerHead);
    std::vector<T> vTransposedCacheVec(batchSize * headsNum * seqLen * dimsPerHead);
    randomInitVector(kTransposedCacheVec, 1.f / kvScaleOrigQuant);
    randomInitVector(vTransposedCacheVec, 1.f / kvScaleOrigQuant);

    // Copy inputs to GPU
    auto kTransposedCache = std::shared_ptr(manager.copyFrom(
        kTransposedCacheVec, ITensor::makeShape({batchSize, headsNum, seqLen, dimsPerHead}), MemoryType::kGPU));
    auto vTransposedCache = std::shared_ptr(manager.copyFrom(
        vTransposedCacheVec, ITensor::makeShape({batchSize, headsNum, seqLen, dimsPerHead}), MemoryType::kGPU));

    // Run inference
    KvCacheDataType const cache_type
        = int8KVCache ? KvCacheDataType::INT8 : (fp8KVCache ? KvCacheDataType::FP8 : KvCacheDataType::BASE);
    invokeTranspose4dBatchMajor(bufferCast<T>(*kTransposedCache), bufferCast<T>(*vTransposedCache), kvLinearBuffer,
        batchSize, seqLen, maxSeqLen, dimsPerHead, headsNum, cache_type, kvScaleOrigQuantPtr, sequenceLengths,
        streamPtr->get());

    // Synchronize
    streamPtr->synchronize();

    // Copy pool to CPU
    std::vector<T_DST> kvMemoryPoolHost(kvPoolElts);
    cudaMemcpy(kvMemoryPoolHost.data(), kvLinearBuffer.data, kvPoolSize, cudaMemcpyDeviceToHost);
    KVLinearBuffer kvLinearBufferHost = kvLinearBuffer;

    // Init array of CPU pointers from page table
    kvLinearBufferHost.data = reinterpret_cast<int8_t*>(kvMemoryPoolHost.data());

    verifyKVTransposed<T, T_DST>(batchSize, headsNum, dimsPerHead, seqLen, maxSeqLen, kvLinearBufferHost,
        kTransposedCacheVec, vTransposedCacheVec, int8KVCache || fp8KVCache, kvScaleOrigQuant);

    cudaFree(sequenceLengths);
    if (int8KVCache || fp8KVCache)
    {
        cudaFree(kvScaleOrigQuantPtr);
    }
}

} // namespace

TEST(AttentionKernelTest, transposeBatch4dPagedFloat)
{
    testTransposeBatch4dPaged<float, float>(false, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dPagedHalf)
{
    testTransposeBatch4dPaged<half, half>(false, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dPagedMultiQuery)
{
    testTransposeBatch4dPaged<half, half>(true, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dPagedInt8)
{
    testTransposeBatch4dPaged<float, int8_t>(false, true, false);
}

#ifdef ENABLE_FP8
TEST(AttentionKernelTest, transposeBatch4dPagedFp8)
{
    testTransposeBatch4dPaged<float, __nv_fp8_e4m3>(false, false, true);
}
#endif

TEST(AttentionKernelTest, transposeBatch4dContiguousFloat)
{
    testTransposeBatch4dContiguous<float, float>(false, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dContiguousHalf)
{
    testTransposeBatch4dContiguous<half, half>(false, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dContiguousMultiQuery)
{
    testTransposeBatch4dContiguous<half, half>(true, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dContiguousInt8)
{
    testTransposeBatch4dContiguous<float, int8_t>(false, true, false);
}

#ifdef ENABLE_FP8
TEST(AttentionKernelTest, transposeBatch4dContiguousFp8)
{
    testTransposeBatch4dContiguous<float, __nv_fp8_e4m3>(false, false, true);
}
#endif
