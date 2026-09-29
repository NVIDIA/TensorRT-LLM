/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCodec.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

struct CudaDeleter
{
    void operator()(void* ptr) const noexcept
    {
        cudaFree(ptr);
    }
};

using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation allocateCuda(size_t numBytes)
{
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, numBytes) != cudaSuccess)
    {
        return CudaAllocation{};
    }
    return CudaAllocation{ptr};
}

CoalescedBuffer makeCoalescedBuffer(size_t size, LayerId layerId, std::string role)
{
    return CoalescedBuffer{size, {BufferId{layerId, std::move(role)}}};
}

PoolGroupDesc makePoolGroupDesc(void* pool0, size_t pool0Bytes, void* pool1, size_t pool1Bytes, SlotCount numSlots,
    PoolGroupIndex poolGroupIndex = PoolGroupIndex{0}, LifeCycleId firstLifeCycle = LifeCycleId{0})
{
    LifeCycleId const secondLifeCycle{firstLifeCycle.value() + 1};
    SlotDescVariant variant0{firstLifeCycle,
        TypedVec<PoolIndex, CoalescedBuffer>{makeCoalescedBuffer(pool0Bytes, firstLifeCycle.value(), "key"),
            makeCoalescedBuffer(pool1Bytes, firstLifeCycle.value(), "value")}};
    SlotDescVariant variant1{secondLifeCycle,
        TypedVec<PoolIndex, CoalescedBuffer>{makeCoalescedBuffer(pool0Bytes, secondLifeCycle.value(), "key"),
            makeCoalescedBuffer(pool1Bytes, secondLifeCycle.value(), "value")}};
    SlotDesc slotDesc{{std::move(variant0), std::move(variant1)}};

    return PoolGroupDesc{poolGroupIndex, numSlots, std::move(slotDesc),
        TypedVec<PoolIndex, PoolDesc>{PoolDesc{PoolIndex{0}, reinterpret_cast<MemAddress>(pool0), pool0Bytes},
            PoolDesc{PoolIndex{1}, reinterpret_cast<MemAddress>(pool1), pool1Bytes}}};
}

bool configureOne(IKvCacheColdPageCodec& codec, PoolGroupDesc const& gpuDesc)
{
    return codec.configure(&gpuDesc, PoolGroupIndex{1});
}

std::vector<uint8_t> makePattern(size_t slotBytes, size_t numSlots, uint8_t salt)
{
    std::vector<uint8_t> result(slotBytes * numSlots);
    for (size_t slot = 0; slot < numSlots; ++slot)
    {
        for (size_t byte = 0; byte < slotBytes; ++byte)
        {
            result[slot * slotBytes + byte]
                = static_cast<uint8_t>((slot * 17 + byte * 3 + static_cast<size_t>(salt)) & 0xFFU);
        }
    }
    return result;
}

TEST(KvCacheManagerV2DefaultColdPageCodecTest, ConfiguresAllPoolGroupsAtomically)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    CudaAllocation firstPool0 = allocateCuda(16);
    CudaAllocation firstPool1 = allocateCuda(16);
    CudaAllocation secondPool0 = allocateCuda(24);
    CudaAllocation secondPool1 = allocateCuda(16);
    ASSERT_NE(firstPool0, nullptr);
    ASSERT_NE(firstPool1, nullptr);
    ASSERT_NE(secondPool0, nullptr);
    ASSERT_NE(secondPool1, nullptr);

    PoolGroupDesc first
        = makePoolGroupDesc(firstPool0.get(), 16, firstPool1.get(), 16, 1, PoolGroupIndex{0}, LifeCycleId{0});
    PoolGroupDesc second
        = makePoolGroupDesc(secondPool0.get(), 24, secondPool1.get(), 16, 1, PoolGroupIndex{1}, LifeCycleId{2});
    PoolGroupDesc invalidSecond = second;
    invalidSecond.pools.at(PoolIndex{0}).poolIndex = PoolIndex{1};

    auto codec = createDefaultKvCacheColdPageCodec();
    ASSERT_NE(codec, nullptr);
    EXPECT_FALSE(codec->configure(nullptr, PoolGroupIndex{1}));
    EXPECT_FALSE(codec->configure(&first, PoolGroupIndex{0}));

    std::array<PoolGroupDesc, 2> gpuDescs{first, invalidSecond};
    EXPECT_FALSE(codec->configure(gpuDescs.data(), PoolGroupIndex{2}));
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{0}), 0);
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{2}), 0);

    gpuDescs[1] = second;
    ASSERT_TRUE(codec->configure(gpuDescs.data(), PoolGroupIndex{2}));
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{0}), 32);
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{1}), 32);
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{2}), 40);
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{3}), 40);
    EXPECT_EQ(codec->getBatchingLayerGroupId(LifeCycleId{0}), LifeCycleId{0});
    EXPECT_EQ(codec->getBatchingLayerGroupId(LifeCycleId{1}), LifeCycleId{0});
    EXPECT_EQ(codec->getBatchingLayerGroupId(LifeCycleId{2}), LifeCycleId{2});
    EXPECT_EQ(codec->getBatchingLayerGroupId(LifeCycleId{3}), LifeCycleId{2});
}

TEST(KvCacheManagerV2DefaultColdPageCodecTest, ConcatenatesAndRestoresLargeNonContiguousBatch)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kPool0Bytes = 64;
    constexpr size_t kPool1Bytes = 48;
    constexpr size_t kColdPageBytes = kPool0Bytes + kPool1Bytes;
    constexpr size_t kBatchSize = 65536;
    constexpr size_t kNumSlots = kBatchSize;

    CudaAllocation pool0 = allocateCuda(kPool0Bytes * kNumSlots);
    CudaAllocation pool1 = allocateCuda(kPool1Bytes * kNumSlots);
    CudaAllocation cold = allocateCuda(kColdPageBytes * kNumSlots);
    ASSERT_NE(pool0, nullptr);
    ASSERT_NE(pool1, nullptr);
    ASSERT_NE(cold, nullptr);

    std::vector<uint8_t> const pool0Input = makePattern(kPool0Bytes, kNumSlots, 11);
    std::vector<uint8_t> const pool1Input = makePattern(kPool1Bytes, kNumSlots, 29);
    ASSERT_EQ(cudaMemcpy(pool0.get(), pool0Input.data(), pool0Input.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(pool1.get(), pool1Input.data(), pool1Input.size(), cudaMemcpyHostToDevice), cudaSuccess);

    auto codec = createDefaultKvCacheColdPageCodec();
    ASSERT_NE(codec, nullptr);
    PoolGroupDesc const desc
        = makePoolGroupDesc(pool0.get(), kPool0Bytes, pool1.get(), kPool1Bytes, static_cast<SlotCount>(kNumSlots));
    ASSERT_TRUE(configureOne(*codec, desc));
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{0}), kColdPageBytes);
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{1}), kColdPageBytes);
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{2}), 0);
    EXPECT_EQ(codec->getBatchingLayerGroupId(LifeCycleId{0}), LifeCycleId{0});
    EXPECT_EQ(codec->getBatchingLayerGroupId(LifeCycleId{1}), LifeCycleId{0});
    EXPECT_EQ(codec->getBatchingLayerGroupId(LifeCycleId{2}), LifeCycleId{-1});
    EXPECT_EQ(codec->queryPageIndexLocation(LifeCycleId{0}), PageIndexLocation::kHost);
    EXPECT_EQ(codec->queryPageIndexLocation(LifeCycleId{1}), PageIndexLocation::kHost);
    EXPECT_EQ(codec->queryPageIndexLocation(LifeCycleId{2}), PageIndexLocation::kBadLocation);
    EXPECT_FALSE(configureOne(*codec, desc));

    std::vector<PageIndexPair> encodePageIndices(kBatchSize);
    std::vector<PageIndexPair> decodePageIndices(kBatchSize);
    for (size_t index = 0; index < kBatchSize; ++index)
    {
        int32_t const hotIndex = static_cast<int32_t>((index * 37) % kBatchSize);
        int32_t const coldIndex = static_cast<int32_t>((index * 7) % kBatchSize);
        encodePageIndices[index] = PageIndexPair{coldIndex, hotIndex};
        decodePageIndices[index] = PageIndexPair{hotIndex, coldIndex};
    }

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(codec->encode(LifeCycleId{0}, cold.get(), encodePageIndices.data(), kBatchSize, stream));
    ASSERT_EQ(cudaMemsetAsync(pool0.get(), 0, kPool0Bytes * kNumSlots, stream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(pool1.get(), 0, kPool1Bytes * kNumSlots, stream), cudaSuccess);
    ASSERT_TRUE(codec->decode(LifeCycleId{0}, cold.get(), decodePageIndices.data(), kBatchSize, stream));
    std::vector<uint8_t> pool0Output(kPool0Bytes * kNumSlots);
    std::vector<uint8_t> pool1Output(kPool1Bytes * kNumSlots);
    ASSERT_EQ(cudaMemcpyAsync(pool0Output.data(), pool0.get(), pool0Output.size(), cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(pool1Output.data(), pool1.get(), pool1Output.size(), cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    std::vector<uint8_t> pool0Expected(kPool0Bytes * kNumSlots);
    std::vector<uint8_t> pool1Expected(kPool1Bytes * kNumSlots);
    std::copy_n(pool0Input.begin(), kPool0Bytes * kBatchSize, pool0Expected.begin());
    std::copy_n(pool1Input.begin(), kPool1Bytes * kBatchSize, pool1Expected.begin());
    EXPECT_EQ(pool0Output, pool0Expected);
    EXPECT_EQ(pool1Output, pool1Expected);
}

TEST(KvCacheManagerV2DefaultColdPageCodecTest, RoundTripsBatchedCopies)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kNumSlots = 2;
    constexpr size_t kPool0Bytes = (5U << 20U) + 16;
    constexpr size_t kPool1Bytes = 16;
    constexpr size_t kColdPageBytes = kPool0Bytes + kPool1Bytes;
    CudaAllocation pool0 = allocateCuda(kPool0Bytes * kNumSlots);
    CudaAllocation pool1 = allocateCuda(kPool1Bytes * kNumSlots);
    CudaAllocation cold = allocateCuda(kColdPageBytes * kNumSlots);
    ASSERT_NE(pool0, nullptr);
    ASSERT_NE(pool1, nullptr);
    ASSERT_NE(cold, nullptr);

    std::vector<uint8_t> const pool0Input = makePattern(kPool0Bytes, kNumSlots, 7);
    std::vector<uint8_t> const pool1Input = makePattern(kPool1Bytes, kNumSlots, 13);
    ASSERT_EQ(cudaMemcpy(pool0.get(), pool0Input.data(), pool0Input.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(pool1.get(), pool1Input.data(), pool1Input.size(), cudaMemcpyHostToDevice), cudaSuccess);

    auto codec = createDefaultKvCacheColdPageCodec();
    ASSERT_NE(codec, nullptr);
    ASSERT_TRUE(configureOne(*codec, makePoolGroupDesc(pool0.get(), kPool0Bytes, pool1.get(), kPool1Bytes, kNumSlots)));

    std::array<PageIndexPair, kNumSlots> const indices{{PageIndexPair{0, 1}, PageIndexPair{1, 0}}};
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(codec->encode(LifeCycleId{0}, cold.get(), indices.data(), kNumSlots, stream));
    ASSERT_EQ(cudaMemsetAsync(pool0.get(), 0, pool0Input.size(), stream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(pool1.get(), 0, pool1Input.size(), stream), cudaSuccess);
    ASSERT_TRUE(codec->decode(LifeCycleId{0}, cold.get(), indices.data(), kNumSlots, stream));

    std::vector<uint8_t> pool0Output(pool0Input.size());
    std::vector<uint8_t> pool1Output(pool1Input.size());
    ASSERT_EQ(cudaMemcpyAsync(pool0Output.data(), pool0.get(), pool0Output.size(), cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(pool1Output.data(), pool1.get(), pool1Output.size(), cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_EQ(pool0Output, pool0Input);
    EXPECT_EQ(pool1Output, pool1Input);
}

TEST(KvCacheManagerV2DefaultColdPageCodecTest, AcceptsUnalignedPoolLayouts)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kAllocationBytes = 64;
    constexpr size_t kPool0Bytes = 17;
    constexpr size_t kPool1Bytes = 15;
    CudaAllocation pool0 = allocateCuda(kAllocationBytes);
    CudaAllocation pool1 = allocateCuda(kAllocationBytes);
    ASSERT_NE(pool0, nullptr);
    ASSERT_NE(pool1, nullptr);

    auto codec = createDefaultKvCacheColdPageCodec();
    ASSERT_NE(codec, nullptr);
    auto* const unalignedPool0 = static_cast<std::byte*>(pool0.get()) + 1;
    EXPECT_TRUE(configureOne(*codec, makePoolGroupDesc(unalignedPool0, kPool0Bytes, pool1.get(), kPool1Bytes, 1)));
    EXPECT_EQ(codec->queryColdPageBytes(LifeCycleId{0}), kPool0Bytes + kPool1Bytes);
}

TEST(KvCacheManagerV2DefaultColdPageCodecTest, ValidatesHostIndexArgumentsBeforeSubmission)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kPool0Bytes = 16;
    constexpr size_t kPool1Bytes = 16;
    CudaAllocation pool0 = allocateCuda(kPool0Bytes);
    CudaAllocation pool1 = allocateCuda(kPool1Bytes);
    CudaAllocation cold = allocateCuda(kPool0Bytes + kPool1Bytes);
    ASSERT_NE(pool0, nullptr);
    ASSERT_NE(pool1, nullptr);
    ASSERT_NE(cold, nullptr);

    auto codec = createDefaultKvCacheColdPageCodec();
    ASSERT_NE(codec, nullptr);
    ASSERT_TRUE(configureOne(*codec, makePoolGroupDesc(pool0.get(), kPool0Bytes, pool1.get(), kPool1Bytes, 1)));

    PageIndexPair const validIndex{0, 0};
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    EXPECT_FALSE(codec->encode(LifeCycleId{0}, cold.get(), nullptr, 1, stream));
    EXPECT_FALSE(codec->encode(LifeCycleId{2}, cold.get(), &validIndex, 1, stream));
    EXPECT_FALSE(codec->encode(LifeCycleId{0}, cold.get(), &validIndex, 1, nullptr));
    EXPECT_TRUE(codec->encode(LifeCycleId{0}, nullptr, nullptr, 0, nullptr));
    EXPECT_FALSE(codec->decode(LifeCycleId{0}, nullptr, &validIndex, 1, stream));
    EXPECT_FALSE(codec->decode(LifeCycleId{0}, cold.get(), nullptr, 1, stream));
    EXPECT_FALSE(codec->decode(LifeCycleId{2}, cold.get(), &validIndex, 1, stream));
    EXPECT_FALSE(codec->decode(LifeCycleId{0}, cold.get(), &validIndex, 1, nullptr));
    EXPECT_TRUE(codec->decode(LifeCycleId{0}, nullptr, nullptr, 0, nullptr));
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
