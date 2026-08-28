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

#include "kvCacheManagerV2TestUtils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/funcGuard.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeTieredConfig;
using tensorrt_llm::common::TllmException;

KVCacheManagerConfig makeDiskTieredConfig()
{
    auto config = makeConfig();
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    return config;
}

KVCacheManagerConfig makeGpuTieredConfig()
{
    auto config = makeTieredConfig();
    config.cacheTiers[1] = GpuCacheTierConfig{4 << 20};
    return config;
}

KVCacheManagerConfig makeSplitColdGroupingConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});

    AttentionLayerConfig first;
    first.layerId = 0;
    first.slidingWindowSize = 128;
    first.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(first));

    AttentionLayerConfig second;
    second.layerId = 1;
    second.slidingWindowSize = 256;
    second.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(second));
    return config;
}

class RejectingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    explicit RejectingColdPageCodec(int& destructionCount)
        : mDestructionCount(destructionCount)
    {
    }

    ~RejectingColdPageCodec() override
    {
        ++mDestructionCount;
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return false;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

private:
    int& mDestructionCount;
};

class SplitColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    explicit SplitColdPageCodec(bool batchTogether = false)
        : mBatchTogether(batchTogether)
    {
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId layerGroupId) const noexcept override
    {
        if (layerGroupId == LayerGroupId{0})
            return 1024;
        if (layerGroupId == LayerGroupId{1})
            return 2048;
        return 0;
    }

    LayerGroupId getBatchingLayerGroupId(LayerGroupId layerGroupId) const noexcept override
    {
        return mBatchTogether ? LayerGroupId{0} : layerGroupId;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return true;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return true;
    }

private:
    bool mBatchTogether;
};

class OversizedColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return std::numeric_limits<size_t>::max() / 3 + 1;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }
};

class MixedIndexLocationColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 1024;
    }

    LayerGroupId getBatchingLayerGroupId(LayerGroupId) const noexcept override
    {
        return LayerGroupId{0};
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId layerGroupId) const noexcept override
    {
        return layerGroupId == LayerGroupId{0} ? PageIndexLocation::kHost : PageIndexLocation::kDevice;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t) noexcept override
    {
        return false;
    }
};

class AsyncRejectingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    enum class Operation
    {
        kEncode,
        kDecode,
    };

    explicit AsyncRejectingColdPageCodec(Operation operation)
        : mOperation(operation)
    {
    }

    bool configure(PoolGroupDesc const*, PoolGroupIndex) noexcept override
    {
        return true;
    }

    size_t queryColdPageBytes(LayerGroupId) const noexcept override
    {
        return 2 << 20;
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId) const noexcept override
    {
        return PageIndexLocation::kHost;
    }

    bool encode(LayerGroupId, void*, PageIndexPair const*, size_t, cudaStream_t stream) noexcept override
    {
        return mOperation == Operation::kEncode ? reject(stream) : true;
    }

    bool decode(LayerGroupId, void const*, PageIndexPair const*, size_t, cudaStream_t stream) noexcept override
    {
        return mOperation == Operation::kDecode ? reject(stream) : true;
    }

    bool launched() const noexcept
    {
        return mLaunched.load(std::memory_order_acquire);
    }

    void release() noexcept
    {
        mRelease.store(true, std::memory_order_release);
    }

private:
    static void CUDART_CB waitForRelease(void* data)
    {
        auto& codec = *static_cast<AsyncRejectingColdPageCodec*>(data);
        while (!codec.mRelease.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }

    bool reject(cudaStream_t stream) noexcept
    {
        bool const launched = cudaLaunchHostFunc(stream, waitForRelease, this) == cudaSuccess;
        mLaunched.store(launched, std::memory_order_release);
        return false;
    }

    Operation mOperation;
    std::atomic<bool> mLaunched{false};
    std::atomic<bool> mRelease{false};
};

SharedPtr<CommittedPage> makeCommittedPage(KvCacheManager& manager, StorageManager& storage, CacheLevel level,
    Slot& slot, LifeCycleId lifeCycle = LifeCycleId{0}, Priority priority = kPriorityDefault, int tokenBase = 0)
{
    RootBlock& root = manager.radixTree().addOrGetExisting({});
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager.tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{tokenBase + token});
    }
    auto block = addOrGetExistingBlock(&root, std::move(tokens), /*knownNoDigest=*/true);
    auto page = makeShared<CommittedPage>(
        &storage, block, lifeCycle, level, static_cast<int>(block->tokens.size()), priority);
    page->setSlot(slot);
    block->storage[lifeCycle] = page.get();
    storage.scheduleForEviction(*page);
    return page;
}

TEST(KvCacheManagerV2ColdPageTest, ConstructionFailureDestroysCodec)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int destructionCount = 0;
    std::unique_ptr<IKvCacheColdPageCodec> codec = std::make_unique<RejectingColdPageCodec>(destructionCount);
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(makeConfig(), nullptr, std::move(codec));
            (void) manager;
        },
        TllmException);

    EXPECT_EQ(destructionCount, 1);
}

TEST(KvCacheManagerV2ColdPageTest, RejectsColdPageStagingSizeOverflow)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(
                makeDiskTieredConfig(), nullptr, std::make_unique<OversizedColdPageCodec>());
            (void) manager;
        },
        TllmException);
}

TEST(KvCacheManagerV2ColdPageTest, DoesNotSizePageStagingWithoutColdTier)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_NO_THROW({
        auto manager
            = std::make_shared<KvCacheManager>(makeConfig(), nullptr, std::make_unique<OversizedColdPageCodec>());
        (void) manager;
    });
}

TEST(KvCacheManagerV2ColdPageTest, ColdGpuTierSupportsSingleSlotRoundTrip)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeGpuTieredConfig());
    auto& storage = manager->storage();
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto streamGuard = FuncGuard([stream]() { cudaStreamDestroy(stream); });
    LifeCycleId const lifeCycle{0};
    CacheLevel const coldLevel{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto hotSlots = storage.newSlots(kHotLevel, oneSlot);
    auto coldSlots = storage.newSlots(coldLevel, oneSlot);
    ASSERT_EQ(hotSlots[lifeCycle].size(), 1);
    ASSERT_EQ(coldSlots[lifeCycle].size(), 1);

    Slot& hotSlot = hotSlots[lifeCycle].front();
    Slot& coldSlot = coldSlots[lifeCycle].front();
    PoolGroupIndex const hotPoolGroup = storage.getPoolGroupIndex(kHotLevel, lifeCycle);
    size_t const hotPageBytes = storage.slotSize(hotPoolGroup).at(PoolIndex{0});
    MemAddress const hotAddress
        = std::get<MemAddress>(storage.slotAddress(kHotLevel, hotPoolGroup, hotSlot.slotId(), PoolIndex{0}));
    constexpr uint8_t kPattern = 0xA7;
    ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(hotAddress), kPattern, hotPageBytes), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    storage.copySlotData(
        lifeCycle, coldLevel, kHotLevel, coldSlot.slotId(), hotSlot.slotId(), reinterpret_cast<CUstream>(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(hotAddress), 0, hotPageBytes), cudaSuccess);
    storage.copySlotData(
        lifeCycle, kHotLevel, coldLevel, hotSlot.slotId(), coldSlot.slotId(), reinterpret_cast<CUstream>(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<uint8_t> restoredPage(hotPageBytes);
    ASSERT_EQ(cudaMemcpy(
                  restoredPage.data(), reinterpret_cast<void const*>(hotAddress), hotPageBytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_TRUE(std::all_of(restoredPage.begin(), restoredPage.end(), [](uint8_t byte) { return byte == kPattern; }));

    storage.releaseSlot(lifeCycle, coldLevel, std::move(coldSlot));
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(hotSlot));
}

TEST(KvCacheManagerV2ColdPageTest, AsyncEncodeRejectionFencesRecycledColdSlotAndReschedulesSource)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>(AsyncRejectingColdPageCodec::Operation::kEncode);
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), nullptr, std::move(codec));
    auto releaseCodecGuard = FuncGuard([codecPtr]() { codecPtr->release(); });
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const coldLevel{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);

    auto coldSlots = storage.newSlots(coldLevel, oneSlot);
    Slot coldBlocker = std::move(coldSlots[lifeCycle].front());
    auto hotSlots = storage.newGpuSlots(twoSlots);
    Slot hotBlocker = std::move(hotSlots[lifeCycle].back());
    auto sourcePage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[lifeCycle].front());
    SlotId const sourceSlotId = sourcePage->slotId();

    EXPECT_THROW(storage.newGpuSlots(oneSlot), TllmException);
    ASSERT_TRUE(codecPtr->launched());
    EXPECT_EQ(sourcePage->cacheLevel, kHotLevel);
    EXPECT_EQ(sourcePage->slotId(), sourceSlotId);
    EXPECT_TRUE(sourcePage->scheduledForEviction());

    auto recycledSlots = storage.newSlots(coldLevel, oneSlot);
    Slot recycledColdSlot = std::move(recycledSlots[lifeCycle].front());
    EXPECT_FALSE(recycledColdSlot.queryReady());
    EXPECT_FALSE(sourcePage->queryReady());

    codecPtr->release();
    recycledColdSlot.readyEvent.synchronize();
    storage.releaseSlot(lifeCycle, coldLevel, std::move(recycledColdSlot));
    storage.releaseSlot(lifeCycle, coldLevel, std::move(coldBlocker));
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(hotBlocker));
}

TEST(KvCacheManagerV2ColdPageTest, ForceEvictFailureReschedulesFallenPage)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>(AsyncRejectingColdPageCodec::Operation::kEncode);
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), nullptr, std::move(codec));
    auto releaseCodecGuard = FuncGuard([codecPtr]() { codecPtr->release(); });
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);

    auto hotSlots = storage.newGpuSlots(oneSlot);
    auto sourcePage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[lifeCycle].front());
    SlotId const sourceSlotId = sourcePage->slotId();
    TypedVec<PoolGroupIndex, SlotCount> evictOne(storage.numPoolGroups(kHotLevel), 0);
    evictOne[storage.getPoolGroupIndex(kHotLevel, lifeCycle)] = 1;

    EXPECT_THROW(storage.forceEvict(kHotLevel, evictOne), TllmException);
    ASSERT_TRUE(codecPtr->launched());
    EXPECT_EQ(sourcePage->cacheLevel, kHotLevel);
    EXPECT_EQ(sourcePage->slotId(), sourceSlotId);
    EXPECT_TRUE(sourcePage->scheduledForEviction());
    EXPECT_FALSE(sourcePage->queryReady());

    codecPtr->release();
    sourcePage->readyEvent.synchronize();
}

TEST(KvCacheManagerV2ColdPageTest, AsyncDecodeRejectionFencesRecycledGpuSlot)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>(AsyncRejectingColdPageCodec::Operation::kDecode);
    auto* codecPtr = codec.get();
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig(), nullptr, std::move(codec));
    auto releaseCodecGuard = FuncGuard([codecPtr]() { codecPtr->release(); });
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    CacheLevel const coldLevel{1};
    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);

    auto hotSlots = storage.newGpuSlots(oneSlot);
    Slot hotBlocker = std::move(hotSlots[lifeCycle].front());
    auto coldSlots = storage.newSlots(coldLevel, oneSlot);
    auto sourcePage = makeCommittedPage(*manager, storage, coldLevel, coldSlots[lifeCycle].front());
    SlotId const sourceSlotId = sourcePage->slotId();
    storage.excludeFromEviction(*sourcePage);
    auto cache = manager->createKvCache();
    std::vector<BatchedLockTarget> targets{{sourcePage, kDefaultBeamIndex, BlockOrdinal{0}, lifeCycle}};

    EXPECT_THROW(storage.batchedMigrateToGpu(targets, *cache, {}), TllmException);
    ASSERT_TRUE(codecPtr->launched());
    EXPECT_EQ(sourcePage->cacheLevel, coldLevel);
    EXPECT_EQ(sourcePage->slotId(), sourceSlotId);

    auto recycledSlots = storage.newGpuSlots(oneSlot);
    Slot recycledGpuSlot = std::move(recycledSlots[lifeCycle].front());
    EXPECT_FALSE(recycledGpuSlot.queryReady());
    EXPECT_FALSE(sourcePage->queryReady());

    codecPtr->release();
    recycledGpuSlot.readyEvent.synchronize();
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(recycledGpuSlot));
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(hotBlocker));
    storage.scheduleForEviction(*sourcePage);
    cache->close();
}

TEST(KvCacheManagerV2ColdPageTest, RejectsBatchingClassWithDifferentColdPageSizes)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(
                makeSplitColdGroupingConfig(), nullptr, std::make_unique<SplitColdPageCodec>(true));
            (void) manager;
        },
        TllmException);
}

TEST(KvCacheManagerV2ColdPageTest, RejectsBatchingClassWithDifferentIndexLocations)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW(
        {
            auto manager = std::make_shared<KvCacheManager>(
                makeSplitColdGroupingConfig(), nullptr, std::make_unique<MixedIndexLocationColdPageCodec>());
            (void) manager;
        },
        TllmException);
}

TEST(KvCacheManagerV2ColdPageTest, ColdGroupingIsIndependentOfHotGrouping)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeSplitColdGroupingConfig();
    config.initialPoolRatio = std::vector<float>{0.25F, 0.75F};
    auto codec = std::make_unique<SplitColdPageCodec>();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    EXPECT_FALSE(codec);

    StorageManager const& storage = manager->storage();
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{2});
    EXPECT_EQ(storage.numPoolGroups(kHotLevel), PoolGroupIndex{1});
    EXPECT_EQ(storage.numPoolGroups(CacheLevel{1}), PoolGroupIndex{2});

    PoolGroupIndex const hotGroup0 = storage.getPoolGroupIndex(kHotLevel, LifeCycleId{0});
    PoolGroupIndex const hotGroup1 = storage.getPoolGroupIndex(kHotLevel, LifeCycleId{1});
    EXPECT_EQ(hotGroup0, hotGroup1);
    EXPECT_EQ(storage.getRatioList(kHotLevel), (TypedVec<PoolGroupIndex, float>{1.0F}));

    auto const coldRatio = storage.getRatioList(CacheLevel{1});
    ASSERT_EQ(coldRatio.size(), PoolGroupIndex{2});
    constexpr float kFirstColdByteRatio = (0.25F * 1024) / (0.25F * 1024 + 0.75F * 2048);
    EXPECT_NEAR(coldRatio[PoolGroupIndex{0}], kFirstColdByteRatio, 0.01F);
    EXPECT_NEAR(coldRatio[PoolGroupIndex{1}], 1.0F - kFirstColdByteRatio, 0.01F);

    auto const firstColdSlots = storage.numSlots(PoolGroupIndex{0}, CacheLevel{1});
    auto const secondColdSlots = storage.numSlots(PoolGroupIndex{1}, CacheLevel{1});
    EXPECT_NEAR(
        static_cast<float>(firstColdSlots) / static_cast<float>(firstColdSlots + secondColdSlots), 0.25F, 0.01F);

    for (LifeCycleId lifeCycle{0}; lifeCycle < LifeCycleId{2}; ++lifeCycle)
    {
        PoolGroupIndex const coldGroup = storage.getPoolGroupIndex(CacheLevel{1}, lifeCycle);
        EXPECT_NE(coldGroup, storage.getPoolGroupIndex(CacheLevel{1}, LifeCycleId{1 - lifeCycle.value()}));
        EXPECT_EQ(storage.numPools(CacheLevel{1}, coldGroup), PoolIndex{1});
        auto const coldSlotSizes = storage.slotSize(CacheLevel{1}, coldGroup);
        ASSERT_EQ(coldSlotSizes.size(), PoolIndex{1});
        size_t const expectedBytes = lifeCycle == LifeCycleId{0} ? 1024 : 2048;
        EXPECT_EQ(coldSlotSizes.at(PoolIndex{0}), expectedBytes);
    }
}

TEST(KvCacheManagerV2ColdPageTest, MigrationStatsUseColdPageBytes)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeSplitColdGroupingConfig();
    constexpr size_t kHotPageBytes = 2U << 20U;
    constexpr size_t kGpuQuota = 12U << 20U;
    auto& firstLayer = std::get<AttentionLayerConfig>(config.layers[0]);
    auto& secondLayer = std::get<AttentionLayerConfig>(config.layers[1]);
    firstLayer.buffers[0].size = kHotPageBytes;
    firstLayer.slidingWindowSize = std::nullopt;
    secondLayer.buffers[0].size = kHotPageBytes;
    secondLayer.slidingWindowSize = 8;
    config.cacheTiers[0] = GpuCacheTierConfig{kGpuQuota};
    config.initialPoolRatio = std::vector<float>{0.5F, 0.5F};

    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::make_unique<SplitColdPageCodec>());
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto streamGuard = FuncGuard([stream]() { cudaStreamDestroy(stream); });

    constexpr int kNumPages = 3;
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < kNumPages * manager->tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{token});
    }

    auto first = manager->createKvCache();
    ASSERT_TRUE(first->resume(reinterpret_cast<CUstream>(stream)));
    EXPECT_TRUE(first->resize(static_cast<int>(tokens.size())));
    first->commit(toSpan(tokens));
    first->suspend();
    manager->getAndResetIterationStats();

    auto second = manager->createKvCache();
    ASSERT_TRUE(second->resume(reinterpret_cast<CUstream>(stream)));
    EXPECT_TRUE(second->resize(static_cast<int>(tokens.size())));

    auto offloadStats = manager->getAndResetIterationStats();
    ASSERT_EQ(offloadStats.size(), 2);
    EXPECT_EQ(offloadStats.at(LifeCycleId{0}).iterOffloadBlocks, kNumPages);
    EXPECT_EQ(offloadStats.at(LifeCycleId{0}).iterOffloadBytes, kNumPages * 1024);
    EXPECT_EQ(offloadStats.at(LifeCycleId{1}).iterOffloadBlocks, kNumPages);
    EXPECT_EQ(offloadStats.at(LifeCycleId{1}).iterOffloadBytes, kNumPages * 2048);

    second->close();
    ASSERT_TRUE(first->resume(reinterpret_cast<CUstream>(stream)));
    auto onboardStats = manager->getAndResetIterationStats();
    ASSERT_EQ(onboardStats.size(), 2);
    EXPECT_EQ(onboardStats.at(LifeCycleId{0}).iterOnboardBlocks, kNumPages);
    EXPECT_EQ(onboardStats.at(LifeCycleId{0}).iterOnboardBytes, kNumPages * 1024);
    constexpr int kSwaOnboardPages = 2;
    EXPECT_EQ(onboardStats.at(LifeCycleId{1}).iterOnboardBlocks, kSwaOnboardPages);
    EXPECT_EQ(onboardStats.at(LifeCycleId{1}).iterOnboardBytes, kSwaOnboardPages * 2048);
    first->close();
}

TEST(KvCacheManagerV2ColdPageTest, EvictionRoutesLifecycleQueuesToDifferentColdPoolGroups)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(
        makeSplitColdGroupingConfig(), nullptr, std::make_unique<SplitColdPageCodec>());
    auto& storage = manager->storage();
    TypedVec<LifeCycleId, SlotCount> oneSlotPerLifeCycle(LifeCycleId{2}, 1);
    auto hotSlots = storage.newGpuSlots(oneSlotPerLifeCycle);
    auto firstPage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{0}].front(), LifeCycleId{0});
    auto secondPage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{1}].front(), LifeCycleId{1});

    TypedVec<PoolGroupIndex, SlotCount> evictBoth(storage.numPoolGroups(kHotLevel), 0);
    evictBoth[storage.getPoolGroupIndex(kHotLevel, LifeCycleId{0})] = 2;
    storage.forceEvict(kHotLevel, evictBoth);

    EXPECT_EQ(firstPage->cacheLevel, CacheLevel{1});
    EXPECT_EQ(secondPage->cacheLevel, CacheLevel{1});
    EXPECT_NE(storage.getPoolGroupIndex(CacheLevel{1}, firstPage->lifeCycle),
        storage.getPoolGroupIndex(CacheLevel{1}, secondPage->lifeCycle));
    EXPECT_TRUE(firstPage->scheduledForEviction());
    EXPECT_TRUE(secondPage->scheduledForEviction());
}

TEST(KvCacheManagerV2ColdPageTest, FallenPagesRetainHighestPriorityAcrossLifecycleQueues)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeSplitColdGroupingConfig();
    config.cacheTiers[1] = HostCacheTierConfig{4096};
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto& storage = manager->storage();
    CacheLevel const coldLevel{1};
    ASSERT_EQ(storage.numPoolGroups(kHotLevel), PoolGroupIndex{1});
    ASSERT_EQ(storage.numPoolGroups(coldLevel), PoolGroupIndex{1});
    ASSERT_EQ(storage.getStatistics(coldLevel).total, 1);

    TypedVec<LifeCycleId, SlotCount> oneSlotPerLifeCycle(LifeCycleId{2}, 1);
    auto hotSlots = storage.newGpuSlots(oneSlotPerLifeCycle);
    auto highPriorityPage = makeCommittedPage(
        *manager, storage, kHotLevel, hotSlots[LifeCycleId{0}].front(), LifeCycleId{0}, /*priority=*/100);
    auto lowPriorityPage = makeCommittedPage(
        *manager, storage, kHotLevel, hotSlots[LifeCycleId{1}].front(), LifeCycleId{1}, /*priority=*/1);

    TypedVec<PoolGroupIndex, SlotCount> evictBoth(storage.numPoolGroups(kHotLevel), 2);
    storage.forceEvict(kHotLevel, evictBoth);

    EXPECT_EQ(highPriorityPage->cacheLevel, coldLevel);
    EXPECT_EQ(lowPriorityPage->cacheLevel, kHotLevel);
}

TEST(KvCacheManagerV2ColdPageTest, RecursiveFallenPageMergeResortsByPriority)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeSplitColdGroupingConfig();
    config.cacheTiers[1] = HostCacheTierConfig{4096};
    config.cacheTiers.emplace_back(HostCacheTierConfig{4096});
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto& storage = manager->storage();
    CacheLevel const firstColdLevel{1};
    CacheLevel const lastColdLevel{2};
    ASSERT_EQ(storage.getStatistics(firstColdLevel).total, 1);
    ASSERT_EQ(storage.getStatistics(lastColdLevel).total, 1);

    TypedVec<LifeCycleId, SlotCount> oneSlotForSecondLifeCycle(LifeCycleId{2}, 0);
    oneSlotForSecondLifeCycle[LifeCycleId{1}] = 1;
    auto coldSlots = storage.newSlots(firstColdLevel, oneSlotForSecondLifeCycle);
    auto highestPriorityPage = makeCommittedPage(*manager, storage, firstColdLevel, coldSlots[LifeCycleId{1}].front(),
        LifeCycleId{1}, /*priority=*/100, /*tokenBase=*/0);

    TypedVec<LifeCycleId, SlotCount> oneSlotPerLifeCycle(LifeCycleId{2}, 1);
    auto hotSlots = storage.newGpuSlots(oneSlotPerLifeCycle);
    auto middlePriorityPage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{0}].front(),
        LifeCycleId{0}, /*priority=*/50, /*tokenBase=*/100);
    auto lowestPriorityPage = makeCommittedPage(*manager, storage, kHotLevel, hotSlots[LifeCycleId{1}].front(),
        LifeCycleId{1}, /*priority=*/1, /*tokenBase=*/200);

    TypedVec<PoolGroupIndex, SlotCount> evictBoth(storage.numPoolGroups(kHotLevel), 2);
    storage.forceEvict(kHotLevel, evictBoth);

    EXPECT_EQ(middlePriorityPage->cacheLevel, firstColdLevel);
    EXPECT_EQ(highestPriorityPage->cacheLevel, lastColdLevel);
    EXPECT_EQ(lowestPriorityPage->cacheLevel, kHotLevel);
}

} // namespace
