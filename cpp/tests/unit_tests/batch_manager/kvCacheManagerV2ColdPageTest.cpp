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
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/eventManager.h"
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
#include <cstring>
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

class ObservingColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    bool configure(PoolGroupDesc const* descriptors, PoolGroupIndex count) noexcept override
    {
        return mCodec->configure(descriptors, count);
    }

    size_t queryColdPageBytes(LayerGroupId layerGroup) const noexcept override
    {
        return mCodec->queryColdPageBytes(layerGroup);
    }

    LayerGroupId getBatchingLayerGroupId(LayerGroupId layerGroup) const noexcept override
    {
        return mCodec->getBatchingLayerGroupId(layerGroup);
    }

    PageIndexLocation queryPageIndexLocation(LayerGroupId layerGroup) const noexcept override
    {
        return mCodec->queryPageIndexLocation(layerGroup);
    }

    bool encode(LayerGroupId layerGroup, void* destination, PageIndexPair const* indices, size_t count,
        cudaStream_t stream) noexcept override
    {
        ++encodeCalls;
        encodedPages += count;
        encodeStream = stream;
        bool const submitted = mCodec->encode(layerGroup, destination, indices, count, stream);
        return submitted && encodeCalls != rejectEncodeCall;
    }

    bool decode(LayerGroupId layerGroup, void const* source, PageIndexPair const* indices, size_t count,
        cudaStream_t stream) noexcept override
    {
        return mCodec->decode(layerGroup, source, indices, count, stream);
    }

    size_t encodeCalls = 0;
    size_t encodedPages = 0;
    size_t rejectEncodeCall = 0;
    cudaStream_t encodeStream{};

private:
    std::unique_ptr<IKvCacheColdPageCodec> mCodec = createDefaultKvCacheColdPageCodec();
};

class StreamGate
{
public:
    ~StreamGate()
    {
        release();
        if (mStream)
        {
            cudaStreamSynchronize(mStream);
        }
    }

    cudaError_t enqueue(cudaStream_t stream)
    {
        mStream = stream;
        return cudaLaunchHostFunc(stream, wait, this);
    }

    void release() noexcept
    {
        mRelease.store(true, std::memory_order_release);
    }

private:
    static void CUDART_CB wait(void* data)
    {
        auto& gate = *static_cast<StreamGate*>(data);
        while (!gate.mRelease.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }

    std::atomic<bool> mRelease{false};
    cudaStream_t mStream{};
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
    // Every operation on the hot page must be ordered against the migrations, so issue the memsets
    // on `stream` too. cudaMemset() goes to the legacy NULL stream and is asynchronous with respect
    // to the host for device memory; because `stream` is cudaStreamNonBlocking it does not
    // implicitly synchronize with the legacy stream, so a plain cudaMemset() here would be free to
    // land after the migration that is supposed to overwrite it.
    ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(hotAddress), kPattern, hotPageBytes, stream), cudaSuccess);

    storage.copySlotData(
        lifeCycle, coldLevel, kHotLevel, coldSlot.slotId(), hotSlot.slotId(), reinterpret_cast<CUstream>(stream));
    ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(hotAddress), 0, hotPageBytes, stream), cudaSuccess);
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
    EXPECT_THROW(storage.batchedMigrate(kHotLevel, {sourcePage}, {}), TllmException);
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

class KvCacheManagerV2PageLockTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), cudaSuccess);
    }

    void TearDown() override
    {
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    static KVCacheManagerConfig sparseConfig()
    {
        auto config = makeTieredConfig();
        std::get<AttentionLayerConfig>(config.layers.front()).buffers.front().isSparse = true;
        return config;
    }

    static SharedPtr<CommittedPage> seedPrefix(
        KvCacheManager& manager, CacheLevel level, LifeCycleId lc = LifeCycleId{0})
    {
        auto& storage = manager.storage();
        TypedVec<LifeCycleId, SlotCount> counts(storage.numLifeCycles(), 0);
        counts[lc] = 1;
        auto slots = storage.newSlots(level, counts);
        return makeCommittedPage(manager, storage, level, slots[lc].front(), lc);
    }

    static SharedPtr<Page> pageAt(KvCache const& cache, int ordinal = 0, LifeCycleId lc = LifeCycleId{0})
    {
        return blockPageGetPage(cache.blocks().at(BlockOrdinal{ordinal}).pages.at(kDefaultBeamIndex).at(lc));
    }

    CUstream stream() const
    {
        return reinterpret_cast<CUstream>(mStream);
    }

    TokenSpan tokens(int length = 4) const
    {
        return {mTokens.data(), length};
    }

    std::vector<TokenIdExt> const mTokens{TokenIdExt{0}, TokenIdExt{1}, TokenIdExt{2}, TokenIdExt{3}};
    cudaStream_t mStream{};
};

class KvCacheManagerV2SparseOffloadTest : public KvCacheManagerV2PageLockTest
{
};

TEST_F(KvCacheManagerV2SparseOffloadTest, BatchesCompleteCoalescedPagesAndCountsPhysicalCopies)
{
    auto config = makeSplitColdGroupingConfig();
    config.enableStats = true;
    for (auto& layerConfig : config.layers)
    {
        auto& layer = std::get<AttentionLayerConfig>(layerConfig);
        layer.buffers.front().isSparse = true;
        layer.buffers.push_back({.role = "value", .size = 2048, .tokensPerBlockOverride = 2, .isSparse = true});
        layer.buffers.push_back({.role = "scale", .size = 128, .isSparse = true});
    }
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    ASSERT_EQ(storage.numLifeCycles(), LifeCycleId{2});
    ASSERT_EQ(storage.numPoolGroups(kHotLevel), PoolGroupIndex{1});
    ASSERT_GT(storage.numPools(PoolGroupIndex{0}), PoolIndex{1});
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(12, 8));

    std::vector<SharedPtr<Page>> pages;
    std::vector<std::vector<uint8_t>> expected;
    for (int ordinal = 0; ordinal < 2; ++ordinal)
    {
        for (LifeCycleId lc{0}; lc < storage.numLifeCycles(); ++lc)
        {
            auto page = pageAt(*cache, ordinal, lc);
            auto const pg = storage.getPoolGroupIndex(kHotLevel, lc);
            auto const& sizes = storage.slotSize(kHotLevel, pg);
            std::vector<uint8_t> bytes;
            for (PoolIndex pool{0}; pool < sizes.size(); ++pool)
            {
                auto const pattern = static_cast<uint8_t>(17 * pages.size() + pool.value() + 1);
                auto const address = std::get<MemAddress>(storage.slotAddress(kHotLevel, pg, page->slotId(), pool));
                ASSERT_EQ(
                    cudaMemsetAsync(reinterpret_cast<void*>(address), pattern, sizes[pool], mStream), cudaSuccess);
                bytes.insert(bytes.end(), sizes[pool], pattern);
            }
            pages.push_back(std::move(page));
            expected.push_back(std::move(bytes));
        }
    }
    auto const gpuFree = storage.getStatistics(kHotLevel).free;
    auto const hostFree = storage.getStatistics(kSparseHistoryLevel).free;
    auto targets = pages;
    targets.push_back(pages.front());
    cache->offloadSparsePages(targets);
    EXPECT_EQ(observer->encodeCalls, 1);
    EXPECT_EQ(observer->encodedPages, pages.size());
    EXPECT_EQ(observer->encodeStream, mStream);
    EXPECT_EQ(storage.getStatistics(kHotLevel).free, gpuFree + pages.size());
    EXPECT_EQ(storage.getStatistics(kSparseHistoryLevel).free, hostFree - pages.size());
    EXPECT_EQ(cache->pageStorageVersion(), pages.size());

    for (size_t i = 0; i < pages.size(); ++i)
    {
        auto const& page = pages[i];
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(page->status(), PageStatus::LOCKED);
        EXPECT_FALSE(page->scheduledForEviction());
        page->readyEvent.synchronize();
        auto const pg = storage.getPoolGroupIndex(kSparseHistoryLevel, page->lifeCycle);
        auto const address
            = std::get<MemAddress>(storage.slotAddress(kSparseHistoryLevel, pg, page->slotId(), PoolIndex{0}));
        EXPECT_EQ(std::memcmp(reinterpret_cast<void const*>(address), expected[i].data(), expected[i].size()), 0);
    }
    for (LifeCycleId lc{0}; lc < storage.numLifeCycles(); ++lc)
    {
        EXPECT_EQ(pageAt(*cache, 2, lc)->cacheLevel, kHotLevel);
        auto const indices = cache->getBasePageIndices(lc);
        for (int ordinal = 0; ordinal < 3; ++ordinal)
        {
            EXPECT_EQ(indices[ordinal], slotIdToPageIndexValue(pageAt(*cache, ordinal, lc)->slotId()));
        }
    }
    auto const stats = manager->getAndResetIterationStats();
    for (LifeCycleId lc{0}; lc < storage.numLifeCycles(); ++lc)
    {
        EXPECT_EQ(stats.at(lc).iterOffloadBlocks, 2);
        EXPECT_EQ(stats.at(lc).iterOffloadBytes, 2 * expected.front().size());
    }
    cache->offloadSparsePages(targets);
    EXPECT_EQ(observer->encodeCalls, 1);
    EXPECT_EQ(cache->pageStorageVersion(), pages.size());
    EXPECT_TRUE(manager->getAndResetIterationStats().empty());
}

TEST_F(KvCacheManagerV2SparseOffloadTest, SharedOwnersPublishHostIndicesAndKeepHistoryPinned)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto page = seedPrefix(*manager, kHotLevel);
    auto first = manager->createKvCache({}, tokens());
    auto second = manager->createKvCache({}, tokens());
    std::vector<int32_t> externalIndices(1, kBadPageIndex.value());
    auto closeCaches = FuncGuard(
        [&]()
        {
            first->close();
            second->close();
        });
    ASSERT_TRUE(first->resume(stream()));
    ASSERT_TRUE(second->resume(stream()));
    second->setBasePageIndexBuf(kDefaultBeamIndex, LifeCycleId{0}, externalIndices.data(), externalIndices.size());
    auto const gpuSlot = page->slotId();
    auto hostBlockers = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
    auto releaseBlockers = FuncGuard([&]()
        { storage.releaseSlot(LifeCycleId{0}, kSparseHistoryLevel, std::move(hostBlockers[LifeCycleId{0}].front())); });

    EXPECT_THROW(storage.batchedMigrate(kSparseHistoryLevel, {page}, {}), LogicError);
    first->offloadSparsePages({page, page});
    EXPECT_NE(page->slotId(), gpuSlot);
    EXPECT_EQ(first->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(page->slotId()));
    EXPECT_EQ(externalIndices[0], slotIdToPageIndexValue(page->slotId()));
    EXPECT_EQ(first->pageStorageVersion(), 1);
    EXPECT_EQ(second->pageStorageVersion(), 1);
    EXPECT_EQ(storage.getStatistics(kHotLevel).free, storage.getStatistics(kHotLevel).total);
    EXPECT_FALSE(storage.isEvictable(*page));
    EXPECT_THROW(storage.batchedMigrate(kHotLevel, {page}, {}), LogicError);
    first->suspend();
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    ASSERT_TRUE(first->resume());
    EXPECT_EQ(pageAt(*first), page);
    second->close();
    EXPECT_EQ(externalIndices[0], kBadPageIndex.value());
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
}

TEST_F(KvCacheManagerV2SparseOffloadTest, WaitsForLiveAndFinishedReadersBeforeRecyclingGpuSlot)
{
    for (bool const finishReader : {false, true})
    {
        SCOPED_TRACE(finishReader);
        auto manager = std::make_shared<KvCacheManager>(sparseConfig());
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        cudaStream_t readerStream{};
        ASSERT_EQ(cudaStreamCreateWithFlags(&readerStream, cudaStreamNonBlocking), cudaSuccess);
        auto destroyReaderStream = FuncGuard([&]() { cudaStreamDestroy(readerStream); });
        auto page = seedPrefix(*manager, kHotLevel);
        auto first = manager->createKvCache({}, tokens());
        auto second = manager->createKvCache({}, tokens());
        auto closeCaches = FuncGuard(
            [&]()
            {
                first->close();
                second->close();
            });
        ASSERT_TRUE(first->resume(stream()));
        ASSERT_TRUE(second->resume(reinterpret_cast<CUstream>(readerStream)));
        auto const lc = page->lifeCycle;
        auto const pg = storage.getPoolGroupIndex(kHotLevel, lc);
        size_t const bytes = storage.slotSize(kHotLevel, pg)[PoolIndex{0}];
        auto const gpuSlot = page->slotId();
        auto const gpuAddress = std::get<MemAddress>(storage.slotAddress(kHotLevel, pg, gpuSlot, PoolIndex{0}));
        constexpr uint8_t kPattern = 0xA6;
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(gpuAddress), kPattern, bytes, mStream), cudaSuccess);
        auto gpuBlocker = storage.newGpuSlots(TypedVec<LifeCycleId, SlotCount>{1});
        auto hostScratch = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseSlots = FuncGuard(
            [&]()
            {
                storage.releaseSlot(lc, kHotLevel, std::move(gpuBlocker[lc].front()));
                storage.releaseSlot(lc, kSparseHistoryLevel, std::move(hostScratch[lc].front()));
            });
        // Warm the codec's descriptor/index staging before deliberately blocking a stream.
        storage.copySlotData(lc, kSparseHistoryLevel, kHotLevel, hostScratch[lc].front().slotId(), gpuSlot, stream());
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        auto const readback = std::get<MemAddress>(storage.slotAddress(kSparseHistoryLevel,
            storage.getPoolGroupIndex(kSparseHistoryLevel, lc), hostScratch[lc].front().slotId(), PoolIndex{0}));
        StreamGate gate;
        ASSERT_EQ(gate.enqueue(readerStream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(readback), reinterpret_cast<void const*>(gpuAddress), bytes,
                      cudaMemcpyDeviceToHost, readerStream),
            cudaSuccess);
        if (finishReader)
        {
            second->suspend();
        }

        first->offloadSparsePages({page});
        EXPECT_FALSE(page->queryReady());
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
        auto recycled = storage.newGpuSlots(TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseRecycled
            = FuncGuard([&]() { storage.releaseSlot(lc, kHotLevel, std::move(recycled[lc].front())); });
        EXPECT_EQ(recycled[lc].front().slotId(), gpuSlot);
        EXPECT_FALSE(recycled[lc].front().queryReady());
        recycled[lc].front().readyEvent.waitInStream(reinterpret_cast<CudaStream>(mStream));
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(gpuAddress), 0, bytes, mStream), cudaSuccess);
        gate.release();
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        auto const* readBytes = reinterpret_cast<uint8_t const*>(readback);
        EXPECT_TRUE(std::all_of(readBytes, readBytes + bytes, [](uint8_t value) { return value == kPattern; }));
        page->readyEvent.synchronize();
        auto const hostAddress = std::get<MemAddress>(storage.slotAddress(
            kSparseHistoryLevel, storage.getPoolGroupIndex(kSparseHistoryLevel, lc), page->slotId(), PoolIndex{0}));
        auto const* hostBytes = reinterpret_cast<uint8_t const*>(hostAddress);
        EXPECT_TRUE(std::all_of(hostBytes, hostBytes + bytes, [](uint8_t value) { return value == kPattern; }));
    }
}

TEST_F(KvCacheManagerV2SparseOffloadTest, HostOomLeavesEntireBatchOnGpu)
{
    auto config = sparseConfig();
    config.cacheTiers[1] = HostCacheTierConfig{2 << 20};
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(8, 8));
    auto first = pageAt(*cache);
    auto second = pageAt(*cache, 1);
    auto const firstSlot = first->slotId();
    auto const secondSlot = second->slotId();
    EXPECT_THROW(cache->offloadSparsePages({first, second}), OutOfPagesError);
    EXPECT_EQ(observer->encodeCalls, 0);
    EXPECT_EQ(first->cacheLevel, kHotLevel);
    EXPECT_EQ(second->cacheLevel, kHotLevel);
    EXPECT_EQ(first->slotId(), firstSlot);
    EXPECT_EQ(second->slotId(), secondSlot);
    EXPECT_EQ(first->status(), PageStatus::LOCKED);
    EXPECT_EQ(second->status(), PageStatus::LOCKED);
    EXPECT_EQ(cache->pageStorageVersion(), 0);
    EXPECT_EQ(storage.getStatistics(kSparseHistoryLevel).free, 1);
    EXPECT_EQ(storage.getStatistics(kHotLevel).free, 0);
    EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(firstSlot));
    EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[1], slotIdToPageIndexValue(secondSlot));
}

TEST_F(KvCacheManagerV2SparseOffloadTest, AsynchronousRejectionFencesBothSlotsWithoutPublishingHostIndices)
{
    auto config = sparseConfig();
    config.enableStats = true;
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>(AsyncRejectingColdPageCodec::Operation::kEncode);
    auto* rejecting = codec.get();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 4));
    auto page = pageAt(*cache);
    auto const gpuSlot = page->slotId();
    auto const lc = page->lifeCycle;
    auto blocker = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
    auto releaseBlocker
        = FuncGuard([&]() { storage.releaseSlot(lc, kSparseHistoryLevel, std::move(blocker[lc].front())); });
    auto releaseCodec = FuncGuard([&]() { rejecting->release(); });
    EXPECT_THROW(cache->offloadSparsePages({page}), TllmException);
    ASSERT_TRUE(rejecting->launched());
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(page->slotId(), gpuSlot);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    EXPECT_FALSE(page->queryReady());
    EXPECT_EQ(cache->pageStorageVersion(), 0);
    EXPECT_EQ(cache->getBasePageIndices(lc)[0], slotIdToPageIndexValue(gpuSlot));
    auto recycled = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
    auto releaseRecycled
        = FuncGuard([&]() { storage.releaseSlot(lc, kSparseHistoryLevel, std::move(recycled[lc].front())); });
    EXPECT_FALSE(recycled[lc].front().queryReady());
    EXPECT_TRUE(manager->getAndResetIterationStats().empty());
    rejecting->release();
    recycled[lc].front().readyEvent.synchronize();
}

TEST_F(KvCacheManagerV2SparseOffloadTest, LaterCodecBatchFailurePreservesAllSourcePages)
{
    auto config = makeSplitColdGroupingConfig();
    for (auto& layer : config.layers)
    {
        std::get<AttentionLayerConfig>(layer).buffers.front().isSparse = true;
    }
    std::get<AttentionLayerConfig>(config.layers[1]).buffers.front().size *= 2;
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    observer->rejectEncodeCall = 2;
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    ASSERT_EQ(storage.numPoolGroups(kHotLevel), PoolGroupIndex{2});
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 4));
    auto first = pageAt(*cache, 0, LifeCycleId{0});
    auto second = pageAt(*cache, 0, LifeCycleId{1});
    auto const firstSlot = first->slotId();
    auto const secondSlot = second->slotId();
    EXPECT_THROW(cache->offloadSparsePages({first, second}), TllmException);
    EXPECT_EQ(observer->encodeCalls, 2);
    EXPECT_EQ(first->cacheLevel, kHotLevel);
    EXPECT_EQ(second->cacheLevel, kHotLevel);
    EXPECT_EQ(first->slotId(), firstSlot);
    EXPECT_EQ(second->slotId(), secondSlot);
    EXPECT_EQ(cache->pageStorageVersion(), 0);
    EXPECT_EQ(storage.getStatistics(kSparseHistoryLevel).free, storage.getStatistics(kSparseHistoryLevel).total);
    observer->rejectEncodeCall = 0;
    cache->offloadSparsePages({first, second});
    EXPECT_EQ(first->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(second->cacheLevel, kSparseHistoryLevel);
}

TEST_F(KvCacheManagerV2SparseOffloadTest, RejectsWritablePartialAndDensePagesBeforeAnyCopy)
{
    for (bool const sparse : {false, true})
    {
        for (int const history : {0, 2, 4})
        {
            SCOPED_TRACE(sparse);
            SCOPED_TRACE(history);
            auto config = sparse ? sparseConfig() : makeTieredConfig();
            auto codec = std::make_unique<ObservingColdPageCodec>();
            auto* observer = codec.get();
            auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
            auto const apiLock = manager->lockExclusive();
            auto cache = manager->createKvCache();
            auto closeCache = FuncGuard([&]() { cache->close(); });
            ASSERT_TRUE(cache->resume(stream()));
            ASSERT_TRUE(cache->resize(8, history));
            auto first = pageAt(*cache);
            auto input = pageAt(*cache, 1);
            EXPECT_THROW(cache->offloadSparsePages({first, input}), LogicError);
            EXPECT_EQ(observer->encodeCalls, 0);
            EXPECT_EQ(first->cacheLevel, kHotLevel);
            EXPECT_EQ(input->cacheLevel, kHotLevel);
            EXPECT_EQ(cache->pageStorageVersion(), 0);
        }
    }
}

TEST_F(KvCacheManagerV2SparseOffloadTest, RejectsPartialCommittedPagesAndNonOwners)
{
    auto config = sparseConfig();
    config.commitMinSnapshot = true;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto cache = manager->createKvCache();
    auto other = manager->createKvCache();
    auto closeCaches = FuncGuard(
        [&]()
        {
            cache->close();
            other->close();
        });
    EXPECT_THROW(cache->offloadSparsePages({}), LogicError);
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(other->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 2));
    cache->commit(tokens(2), /*isEnd=*/true);
    auto partial = pageAt(*cache);
    ASSERT_TRUE(partial->isCommitted());
    EXPECT_THROW(cache->offloadSparsePages({partial}), LogicError);
    EXPECT_EQ(partial->cacheLevel, kHotLevel);
    ASSERT_TRUE(other->resize(4, 4));
    auto complete = pageAt(*other);
    EXPECT_THROW(cache->offloadSparsePages({complete}), LogicError);
    EXPECT_EQ(complete->cacheLevel, kHotLevel);
    EXPECT_THROW(other->offloadSparsePages({nullptr}), LogicError);
}

TEST_F(KvCacheManagerV2SparseOffloadTest, EmitsCommittedTierChangeEvenWhenSlotIndexIsUnchanged)
{
    auto events = std::make_shared<EventManager>(128);
    auto manager = std::make_shared<KvCacheManager>(sparseConfig(), events);
    auto const apiLock = manager->lockExclusive();
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 4));
    cache->commit(tokens());
    auto page = pageAt(*cache);
    ASSERT_TRUE(page->isCommitted());
    auto const originalIndex = cache->getBasePageIndices(LifeCycleId{0})[0];
    events->flushIterationEvents();
    events->getLatestEvents(/*timeoutMs=*/0);

    cache->offloadSparsePages({page, page});
    EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[0], originalIndex);
    EXPECT_EQ(cache->pageStorageVersion(), 1);
    events->flushIterationEvents();
    auto const updates = events->getLatestEvents(/*timeoutMs=*/0);
    ASSERT_EQ(updates.size(), 1);
    EXPECT_EQ(updates.front().layerGroupId, 0);
    auto const* data = std::get_if<KVCacheUpdatedData>(&updates.front().data);
    ASSERT_NE(data, nullptr);
    ASSERT_TRUE(data->cacheLevel.has_value());
    EXPECT_EQ(data->cacheLevel->oldValue, kHotLevel.value());
    EXPECT_EQ(data->cacheLevel->newValue, kSparseHistoryLevel.value());
}

TEST_F(KvCacheManagerV2SparseOffloadTest, DemotedUncommittedHistoryCanCommitAndBeReused)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 4));
    cache->offloadSparsePages({pageAt(*cache)});
    auto const hostSlot = pageAt(*cache)->slotId();
    cache->commit(tokens());
    auto committed = pageAt(*cache);
    ASSERT_TRUE(committed->isCommitted());
    EXPECT_EQ(committed->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(committed->slotId(), hostSlot);
    auto reused = manager->createKvCache({}, tokens());
    auto closeReused = FuncGuard([&]() { reused->close(); });
    ASSERT_TRUE(reused->resume(stream()));
    EXPECT_EQ(pageAt(*reused), committed);
    cache->close();
    EXPECT_EQ(committed->status(), PageStatus::LOCKED);
    EXPECT_EQ(reused->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(hostSlot));
}

TEST_F(KvCacheManagerV2PageLockTest, SparseHostPrefixStaysPinnedAcrossReuseAndResume)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto page = seedPrefix(*manager, kSparseHistoryLevel);
    SlotId const hostSlot = page->slotId();
    auto cache = manager->createKvCache({}, tokens());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->prefetch(kHotLevel));
    ASSERT_TRUE(cache->resume(stream()));
    EXPECT_EQ(pageAt(*cache), page);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(page->slotId(), hostSlot);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    EXPECT_FALSE(page->scheduledForEviction());
    EXPECT_FALSE(storage.isEvictable(*page));
    EXPECT_THROW(storage.batchedMigrate(kHotLevel, {page}, {}), LogicError);

    auto second = manager->createKvCache({}, tokens());
    auto closeSecond = FuncGuard([&]() { second->close(); });
    ASSERT_TRUE(second->prefetch(kHotLevel));
    ASSERT_TRUE(second->resume(stream()));
    EXPECT_EQ(pageAt(*second), page);
    cache->suspend();
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    second->close();
    EXPECT_EQ(page->status(), PageStatus::HELD);
    ASSERT_TRUE(cache->resume());
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(hostSlot));
    cache->close();
    EXPECT_EQ(page->status(), PageStatus::DROPPABLE);
    EXPECT_TRUE(page->scheduledForEviction());

    // Destruction returns the slot to the host pool, leaving the GPU pool untouched.
    storage.excludeFromEviction(*page);
    page.reset();
    for (CacheLevel level : {kHotLevel, kSparseHistoryLevel})
    {
        auto const stats = manager->getStorageStatistics(level).at(PoolGroupIndex{0});
        EXPECT_EQ(stats.free, stats.total);
    }
}

TEST_F(KvCacheManagerV2PageLockTest, SparseGpuPrefixStaysOnGpuAcrossReuseAndResume)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kHotLevel);
    SlotId const gpuSlot = page->slotId();
    EXPECT_EQ(page->queryLockLevel(), kHotLevel);
    auto cache = manager->createKvCache({}, tokens());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->prefetch(kHotLevel));
    ASSERT_TRUE(cache->resume(stream()));
    EXPECT_EQ(pageAt(*cache), page);
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(page->slotId(), gpuSlot);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);

    cache->suspend();
    ASSERT_TRUE(cache->resume());
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(page->slotId(), gpuSlot);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    auto const hostStats = manager->getStorageStatistics(kSparseHistoryLevel).at(PoolGroupIndex{0});
    EXPECT_EQ(hostStats.free, hostStats.total);
}

TEST_F(KvCacheManagerV2PageLockTest, DensePrefixStillRequiresGpuLock)
{
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig());
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kSparseHistoryLevel);
    EXPECT_EQ(page->queryLockLevel(), kHotLevel);
    auto holder = page->hold();
    EXPECT_THROW(makeShared<UniqPageLock>(holder), LogicError);
    auto cache = manager->createKvCache({}, tokens());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
}

TEST_F(KvCacheManagerV2PageLockTest, PartialReuseCopiesSharedHostPrefixToPrivateGpuPage)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto page = seedPrefix(*manager, kSparseHistoryLevel);
    SlotId const hostSlot = page->slotId();
    PoolGroupIndex const hostPool = storage.getPoolGroupIndex(kSparseHistoryLevel, page->lifeCycle);
    auto const hostAddress
        = std::get<MemAddress>(storage.slotAddress(kSparseHistoryLevel, hostPool, hostSlot, PoolIndex{0}));
    size_t const bytes = storage.slotSize(kSparseHistoryLevel, hostPool).at(PoolIndex{0});
    page->readyEvent.synchronize();
    constexpr uint8_t kPattern = 0xA7;
    std::memset(reinterpret_cast<void*>(hostAddress), kPattern, bytes);

    auto full = manager->createKvCache({}, tokens());
    auto closeFull = FuncGuard([&]() { full->close(); });
    ASSERT_TRUE(full->resume(stream()));
    auto partial = manager->createKvCache({}, tokens(2));
    auto closePartial = FuncGuard([&]() { partial->close(); });
    ASSERT_EQ(partial->historyLength(), 2);
    ASSERT_TRUE(partial->resume(stream()));
    auto privatePage = pageAt(*partial);
    EXPECT_NE(privatePage, page);
    EXPECT_FALSE(privatePage->isCommitted());
    EXPECT_EQ(privatePage->cacheLevel, kHotLevel);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(page->slotId(), hostSlot);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    EXPECT_EQ(full->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(hostSlot));

    auto const gpuPool = storage.getPoolGroupIndex(kHotLevel, privatePage->lifeCycle);
    auto const gpuAddress
        = std::get<MemAddress>(storage.slotAddress(kHotLevel, gpuPool, privatePage->slotId(), PoolIndex{0}));
    std::vector<uint8_t> copied(bytes);
    ASSERT_EQ(cudaMemcpyAsync(
                  copied.data(), reinterpret_cast<void const*>(gpuAddress), bytes, cudaMemcpyDeviceToHost, mStream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_TRUE(std::all_of(copied.begin(), copied.end(), [](uint8_t byte) { return byte == kPattern; }));
}

TEST_F(KvCacheManagerV2PageLockTest, DiskSparsePrefixRestoresToHostBeforeLocking)
{
    auto config = sparseConfig();
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, CacheLevel{2});
    EXPECT_EQ(page->queryLockLevel(), kSparseHistoryLevel);
    EXPECT_THROW(makeShared<UniqPageLock>(page->hold()), LogicError);
    auto cache = manager->createKvCache({}, tokens());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    auto const gpuStats = manager->getStorageStatistics(kHotLevel).at(PoolGroupIndex{0});
    EXPECT_EQ(gpuStats.free, gpuStats.total);
}

TEST_F(KvCacheManagerV2PageLockTest, WritableSparsePageRestoresToGpuButFullHistoryStaysOnHost)
{
    for (int const historyLength : {0, 4})
    {
        SCOPED_TRACE(historyLength);
        auto manager = std::make_shared<KvCacheManager>(sparseConfig());
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        auto cache = manager->createKvCache();
        auto closeCache = FuncGuard([&]() { cache->close(); });
        ASSERT_TRUE(cache->resume(stream()));
        ASSERT_TRUE(cache->resize(4, historyLength));
        auto page = pageAt(*cache);
        // Tier-aware locking alone must not demote a GPU page.
        EXPECT_EQ(page->cacheLevel, kHotLevel);
        cache->suspend();
        TypedVec<PoolGroupIndex, SlotCount> evictOne(storage.numPoolGroups(kHotLevel), 1);
        storage.forceEvict(kHotLevel, evictOne);
        ASSERT_EQ(page->cacheLevel, kSparseHistoryLevel);
        ASSERT_TRUE(cache->resume());
        EXPECT_EQ(page->cacheLevel, historyLength == 0 ? kHotLevel : kSparseHistoryLevel);
        EXPECT_EQ(page->status(), PageStatus::LOCKED);
    }
}

TEST_F(KvCacheManagerV2PageLockTest, ResizeOomRestoresOriginalHostLock)
{
    auto config = sparseConfig();
    std::get<AttentionLayerConfig>(config.layers.front()).slidingWindowSize = 4;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kSparseHistoryLevel);
    auto cache = manager->createKvCache({}, tokens());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(8, 4));
    SlotId const hostSlot = page->slotId();
    auto const gpuFree = manager->getStorageStatistics(kHotLevel).at(PoolGroupIndex{0}).free;

    // Advancing history unlocks block 0 before the request for three GPU pages
    // exceeds the two-slot pool. Rollback must restore the original host lock.
    EXPECT_FALSE(cache->resize(20, 8));
    EXPECT_EQ(cache->capacity(), 8);
    EXPECT_EQ(cache->historyLength(), 4);
    EXPECT_EQ(pageAt(*cache), page);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(page->slotId(), hostSlot);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    EXPECT_FALSE(page->scheduledForEviction());
    EXPECT_EQ(manager->getStorageStatistics(kHotLevel).at(PoolGroupIndex{0}).free, gpuFree);
}

TEST_F(KvCacheManagerV2PageLockTest, MixedSparseAndDensePrefixUsesSeparateLockLevels)
{
    auto config = sparseConfig();
    auto dense = std::get<AttentionLayerConfig>(config.layers.front());
    dense.layerId = 1;
    dense.buffers.front().isSparse = false;
    config.layers.emplace_back(std::move(dense));
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto sparsePage = seedPrefix(*manager, kSparseHistoryLevel, LifeCycleId{0});
    auto densePage = seedPrefix(*manager, kSparseHistoryLevel, LifeCycleId{1});
    auto cache = manager->createKvCache({}, tokens());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    EXPECT_EQ(pageAt(*cache, 0, LifeCycleId{0}), sparsePage);
    EXPECT_EQ(pageAt(*cache, 0, LifeCycleId{1}), densePage);
    EXPECT_EQ(sparsePage->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(densePage->cacheLevel, kHotLevel);
    EXPECT_EQ(sparsePage->status(), PageStatus::LOCKED);
    EXPECT_EQ(densePage->status(), PageStatus::LOCKED);
}

TEST_F(KvCacheManagerV2PageLockTest, CommitRebasesOntoSharedHostPrefix)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kSparseHistoryLevel);
    auto first = manager->createKvCache({}, tokens());
    auto closeFirst = FuncGuard([&]() { first->close(); });
    ASSERT_TRUE(first->resume(stream()));
    auto second = manager->createKvCache();
    auto closeSecond = FuncGuard([&]() { second->close(); });
    ASSERT_TRUE(second->resume(stream()));
    ASSERT_TRUE(second->resize(4, 4));
    ASSERT_EQ(pageAt(*second)->cacheLevel, kHotLevel);
    second->commit(tokens());
    EXPECT_EQ(second->numCommittedBlocks(), 1);
    EXPECT_EQ(pageAt(*second), page);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    auto const gpuStats = manager->getStorageStatistics(kHotLevel).at(PoolGroupIndex{0});
    EXPECT_EQ(gpuStats.free, gpuStats.total);
}

TEST_F(KvCacheManagerV2PageLockTest, ScratchSlotReturnsToGpuPool)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kSparseHistoryLevel);
    auto cache = manager->createKvCache({}, tokens());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    auto const gpuFree = manager->getStorageStatistics(kHotLevel).at(PoolGroupIndex{0}).free;
    auto const hostFree = manager->getStorageStatistics(kSparseHistoryLevel).at(PoolGroupIndex{0}).free;
    auto slots = manager->storage().newGpuSlots(TypedVec<LifeCycleId, SlotCount>(LifeCycleId{1}, 1));
    ScratchSlotLock scratch(std::move(slots[LifeCycleId{0}].front()), *cache, LifeCycleId{0});
    EXPECT_EQ(manager->getStorageStatistics(kHotLevel).at(PoolGroupIndex{0}).free, gpuFree - 1);
    {
        auto scope = cache->recordEventScope();
        scratch.unlock();
    }
    EXPECT_EQ(manager->getStorageStatistics(kHotLevel).at(PoolGroupIndex{0}).free, gpuFree);
    EXPECT_EQ(manager->getStorageStatistics(kSparseHistoryLevel).at(PoolGroupIndex{0}).free, hostFree);
}

} // namespace
