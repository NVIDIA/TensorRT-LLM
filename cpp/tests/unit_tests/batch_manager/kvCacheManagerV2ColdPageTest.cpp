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
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/batch.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/eventManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/cudaEvent.h"
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
        ++decodeCalls;
        decodedPages += count;
        bool const submitted = mCodec->decode(layerGroup, source, indices, count, stream);
        return submitted && decodeCalls != rejectDecodeCall;
    }

    size_t encodeCalls = 0;
    size_t encodedPages = 0;
    size_t rejectEncodeCall = 0;
    size_t decodeCalls = 0;
    size_t decodedPages = 0;
    size_t rejectDecodeCall = 0;
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
        // Event merging must not create streams while a gate is held: stream creation can wait for host callbacks.
        CudaStreamPool::instance();
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

class KvCacheManagerV2PageStorageTest : public KvCacheManagerV2PageLockTest
{
};

class KvCacheManagerV2BatchTest : public KvCacheManagerV2PageLockTest
{
protected:
    CudaStream batchStream() const
    {
        return reinterpret_cast<CudaStream>(stream());
    }

    std::vector<int32_t> read(MemAddress address, size_t size)
    {
        cuCheck(cuStreamSynchronize(stream()));
        std::vector<int32_t> result(size);
        cuCheck(cuMemcpyDtoH(result.data(), address, size * sizeof(int32_t)));
        return result;
    }
};

TEST_F(KvCacheManagerV2BatchTest, PublishesRawMixedTierRowsAndSkipsUnchangedRows)
{
    auto config = makeSplitColdGroupingConfig();
    auto& sparse = std::get<AttentionLayerConfig>(config.layers.front());
    sparse.buffers.front().isSparse = true;
    sparse.buffers.push_back({.role = "value", .size = 2048, .tokensPerBlockOverride = 2, .isSparse = true});
    auto coalesced = sparse;
    coalesced.layerId = 2;
    config.layers.push_back(std::move(coalesced));
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    Batch batch(manager, 3, 4);
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(12, 6));
    EXPECT_EQ(batch.add(*cache, 2), 2);
    auto const sparseGroup = manager->getLayerGroupId(0);
    auto const denseGroup = manager->getLayerGroupId(1);
    auto const address = batch.pageTableAddress(sparseGroup);
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(read(batch.numBlocksAddress(sparseGroup), 3), (std::vector<int32_t>{0, 0, 0}));
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{}));
    EXPECT_FALSE(cache->pageStorageDirty());

    ASSERT_TRUE(cache->enterDecode());
    EXPECT_EQ(batch.dirtyRows(), (std::vector<int>{2}));
    EXPECT_THROW(batch.waitReady(batchStream()), LogicError);
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{2}));
    for (auto const group : {sparseGroup, denseGroup})
    {
        auto expected = std::vector<int32_t>(12, kBadPageIndex.value());
        auto const snapshot = cache->getPageStorageSnapshot(group);
        std::copy(snapshot.basePageIndices().begin(), snapshot.basePageIndices().end(), expected.begin() + 8);
        EXPECT_EQ(read(batch.pageTableAddress(group), 12), expected);
    }
    EXPECT_EQ(read(batch.numBlocksAddress(sparseGroup), 3), (std::vector<int32_t>{0, 0, 1}));
    EXPECT_EQ(read(batch.numBlocksAddress(denseGroup), 3), (std::vector<int32_t>{0, 0, 0}));
    EXPECT_EQ(batch.pageTableAddress(sparseGroup), address);

    cache->suspend();
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{2}));
    EXPECT_EQ(read(address, 12), (std::vector<int32_t>(12, kBadPageIndex.value())));
    EXPECT_EQ(read(batch.numBlocksAddress(sparseGroup), 3), (std::vector<int32_t>{0, 0, 0}));
    ASSERT_TRUE(cache->resume());
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{2}));
    EXPECT_EQ(read(batch.numBlocksAddress(sparseGroup), 3).back(), 1);
}

TEST_F(KvCacheManagerV2BatchTest, PublishesIncrementalOffloadAndClearsReusedRows)
{
    auto config = makeSplitColdGroupingConfig();
    auto& sparse = std::get<AttentionLayerConfig>(config.layers.front());
    sparse.buffers.front().isSparse = true;
    sparse.buffers.push_back({.role = "value", .size = 2048, .tokensPerBlockOverride = 2, .isSparse = true});
    auto coalesced = sparse;
    coalesced.layerId = 2;
    config.layers.push_back(std::move(coalesced));
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto const sparseGroup = manager->getLayerGroupId(0);
    auto const denseGroup = manager->getLayerGroupId(1);
    auto const gpuPool = storage.getPoolGroupIndex(kHotLevel, sparseGroup);
    auto const hostPool = storage.getPoolGroupIndex(kSparseHistoryLevel, sparseGroup);
    auto const& sizes = storage.slotSize(kHotLevel, gpuPool);
    ASSERT_EQ(sizes.size(), PoolIndex{1});
    size_t const bytes = sizes[PoolIndex{0}];
    Batch batch(manager, 2, 4);
    auto const tableAddress = batch.pageTableAddress(sparseGroup);
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(12, 6));
    for (int ordinal = 0; ordinal < 3; ++ordinal)
    {
        auto const page = pageAt(*cache, ordinal, sparseGroup);
        auto const address
            = std::get<MemAddress>(storage.slotAddress(kHotLevel, gpuPool, page->slotId(), PoolIndex{0}));
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), 0x40 + ordinal, bytes, mStream), cudaSuccess);
    }
    batch.add(*cache, 1);
    auto checkPublished = [&](int eligible)
    {
        for (auto const group : {sparseGroup, denseGroup})
        {
            auto expected = std::vector<int32_t>(8, kBadPageIndex.value());
            for (int ordinal = 0; ordinal < 3; ++ordinal)
            {
                auto const page = pageAt(*cache, ordinal, group);
                expected[4 + ordinal] = slotIdToPageIndexValue(page->slotId());
                EXPECT_EQ(
                    page->cacheLevel, group == sparseGroup && ordinal < eligible ? kSparseHistoryLevel : kHotLevel);
            }
            EXPECT_EQ(read(batch.pageTableAddress(group), 8), expected);
            EXPECT_EQ(
                read(batch.numBlocksAddress(group), 2), (std::vector<int32_t>{0, group == sparseGroup ? eligible : 0}));
        }
        for (int ordinal = 0; ordinal < eligible; ++ordinal)
        {
            auto const page = pageAt(*cache, ordinal, sparseGroup);
            auto const address = std::get<MemAddress>(
                storage.slotAddress(kSparseHistoryLevel, hostPool, page->slotId(), PoolIndex{0}));
            auto const* data = reinterpret_cast<uint8_t const*>(address);
            EXPECT_TRUE(std::all_of(data, data + bytes, [ordinal](uint8_t value) { return value == 0x40 + ordinal; }));
        }
    };
    batch.publish(batchStream());
    checkPublished(0);
    auto const freeGpuBefore = manager->getStorageStatistics(kHotLevel)[gpuPool].free;
    ASSERT_TRUE(cache->enterDecode());
    batch.publish(batchStream());
    checkPublished(1);
    auto const firstHostSlot = pageAt(*cache, 0, sparseGroup)->slotId();
    auto const version = cache->pageStorageVersion();
    EXPECT_EQ(batch.resize({std::nullopt, 12}, {std::nullopt, 7}, batchStream()),
        (std::vector<std::optional<bool>>{std::nullopt, true}));
    EXPECT_EQ(cache->pageStorageVersion(), version);
    EXPECT_EQ(observer->encodedPages, 1);
    checkPublished(1);
    EXPECT_EQ(batch.resize({std::nullopt, 12}, {std::nullopt, 8}, batchStream()),
        (std::vector<std::optional<bool>>{std::nullopt, true}));
    checkPublished(2);
    EXPECT_EQ(observer->encodedPages, 2);
    EXPECT_EQ(pageAt(*cache, 0, sparseGroup)->slotId(), firstHostSlot);
    EXPECT_EQ(manager->getStorageStatistics(kHotLevel)[gpuPool].free, freeGpuBefore + 2);
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{}));

    cache->close();
    auto replacement = manager->createKvCache();
    auto closeReplacement = FuncGuard([&]() { replacement->close(); });
    ASSERT_TRUE(replacement->resume(stream()));
    ASSERT_TRUE(replacement->resize(4, 0));
    batch.add(*replacement, 1);
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{1}));
    auto expected = std::vector<int32_t>(8, kBadPageIndex.value());
    expected[4] = slotIdToPageIndexValue(pageAt(*replacement, 0, sparseGroup)->slotId());
    EXPECT_EQ(read(tableAddress, 8), expected);
    EXPECT_EQ(read(batch.numBlocksAddress(sparseGroup), 2), (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(batch.pageTableAddress(sparseGroup), tableAddress);
}

TEST_F(KvCacheManagerV2BatchTest, MembershipCloseAndPublicationFailuresPreserveRows)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    Batch batch(manager, 3, 1);
    Batch other(manager, 3, 1);
    auto first = manager->createKvCache();
    auto second = manager->createKvCache();
    auto closeCaches = FuncGuard(
        [&]()
        {
            first->close();
            if (second)
            {
                second->close();
            }
        });
    EXPECT_EQ(batch.add(*first, 2), 2);
    EXPECT_EQ(batch.add(*first), 2);
    EXPECT_EQ(batch.add(*second), 0);
    EXPECT_THROW(other.add(*first), LogicError);
    EXPECT_THROW(first->bindPageStorageRow(1), LogicError);
    EXPECT_THROW(batch.add(*second, 2), LogicError);
    batch.publish(batchStream());
    first->close();
    EXPECT_EQ(batch.dirtyRows(), (std::vector<int>{2}));
    EXPECT_EQ(second->pageStorageRow(), 0);
    batch.remove(*second);
    EXPECT_EQ(other.add(*second, 1), 1);
    other.close();
    EXPECT_FALSE(second->pageStorageRow().has_value());
    EXPECT_FALSE(second->isClosed());
    EXPECT_EQ(batch.add(*second, 1), 1);
    ASSERT_TRUE(second->resume(stream()));
    ASSERT_TRUE(second->resize(8, 0));
    EXPECT_THROW(batch.publish(batchStream()), std::invalid_argument);
    EXPECT_THROW(batch.waitReady(batchStream()), LogicError);
    EXPECT_TRUE(second->pageStorageDirty());
    EXPECT_FALSE(batch.dirtyRows().empty());
    ASSERT_TRUE(second->resize(4, 0));
    batch.publish(batchStream());
    EXPECT_FALSE(second->pageStorageDirty());
    second.reset();
    EXPECT_EQ(batch.dirtyRows(), (std::vector<int>{1}));
    batch.publish(batchStream());
    EXPECT_EQ(read(batch.pageTableAddress(LifeCycleId{0}), 3), (std::vector<int32_t>(3, -1)));
    closeCaches.cancel();
}

TEST_F(KvCacheManagerV2BatchTest, BatchedResizePublishesFinalStatesAfterPartialOom)
{
    auto config = sparseConfig();
    config.cacheTiers[0] = GpuCacheTierConfig{8 << 20};
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    Batch batch(manager, 3, 4);
    auto first = manager->createKvCache();
    auto second = manager->createKvCache();
    auto closeCaches = FuncGuard(
        [&]()
        {
            first->close();
            second->close();
        });
    ASSERT_TRUE(first->resume(stream()));
    ASSERT_TRUE(second->resume(stream()));
    ASSERT_TRUE(first->resize(4, 0));
    ASSERT_TRUE(second->resize(4, 0));
    batch.add(*first, 0);
    batch.add(*second, 2);
    auto const results = batch.resize({8, std::nullopt, 12}, {0, std::nullopt, 0}, batchStream());
    EXPECT_EQ(results, (std::vector<std::optional<bool>>{true, std::nullopt, false}));
    EXPECT_EQ(first->capacity(), 8);
    EXPECT_EQ(second->capacity(), 4);
    EXPECT_TRUE(batch.dirtyRows().empty());
    auto expected = std::vector<int32_t>(12, -1);
    auto const firstIndices = first->getBasePageIndices(LifeCycleId{0});
    auto const secondIndices = second->getBasePageIndices(LifeCycleId{0});
    std::copy(firstIndices.data(), firstIndices.data() + 2, expected.begin());
    expected[8] = secondIndices[0];
    EXPECT_EQ(read(batch.pageTableAddress(LifeCycleId{0}), 12), expected);
}

TEST_F(KvCacheManagerV2BatchTest, SharedOffloadInvalidatesEveryOwner)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
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
    ASSERT_TRUE(second->resume(stream()));
    Batch batch(manager, 2, 1);
    batch.add(*first);
    batch.add(*second);
    batch.publish(batchStream());
    first->offloadSparsePages({page});
    EXPECT_EQ(batch.dirtyRows(), (std::vector<int>{0, 1}));
    ASSERT_TRUE(first->enterDecode());
    ASSERT_TRUE(second->enterDecode());
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{0, 1}));
    EXPECT_EQ(read(batch.pageTableAddress(LifeCycleId{0}), 2),
        (std::vector<int32_t>(2, slotIdToPageIndexValue(page->slotId()))));
    EXPECT_EQ(read(batch.numBlocksAddress(LifeCycleId{0}), 2), (std::vector<int32_t>{1, 1}));
}

TEST_F(KvCacheManagerV2BatchTest, DeferredHistoryGapClosesWithoutAdvancingWatermark)
{
    for (bool const closeOwner : {false, true})
    {
        SCOPED_TRACE(closeOwner);
        auto config = sparseConfig();
        config.cacheTiers[0] = GpuCacheTierConfig{8 << 20};
        auto codec = std::make_unique<ObservingColdPageCodec>();
        auto* observer = codec.get();
        auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
        auto const apiLock = manager->lockExclusive();
        auto page = seedPrefix(*manager, kHotLevel);
        auto decoder = manager->createKvCache({}, tokens());
        auto prefill = manager->createKvCache({}, tokens());
        auto closeCaches = FuncGuard(
            [&]()
            {
                decoder->close();
                prefill->close();
            });
        ASSERT_TRUE(decoder->resume(stream()));
        ASSERT_TRUE(prefill->resume(stream()));
        ASSERT_TRUE(decoder->resize(12, 8));
        Batch batch(manager, 1, 3);
        batch.add(*decoder);
        auto const group = manager->getLayerGroupId(0);
        ASSERT_TRUE(decoder->enterDecode());
        EXPECT_EQ(page->cacheLevel, kHotLevel);
        EXPECT_EQ(pageAt(*decoder, 1)->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(pageAt(*decoder, 2)->cacheLevel, kHotLevel);
        auto const snapshot = decoder->getPageStorageSnapshot(group);
        EXPECT_EQ(snapshot.eligibleHistoryBlocks(), 0);
        EXPECT_EQ(snapshot.cacheLevels(),
            (std::vector<std::optional<CacheLevel>>{kHotLevel, kSparseHistoryLevel, kHotLevel}));
        EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{0}));
        EXPECT_EQ(read(batch.pageTableAddress(group), 3), snapshot.basePageIndices());
        EXPECT_EQ(read(batch.numBlocksAddress(group), 1), (std::vector<int32_t>{0}));
        EXPECT_EQ(observer->encodedPages, 1);
        auto const hostSlot = pageAt(*decoder, 1)->slotId();
        auto const version = decoder->pageStorageVersion();

        if (closeOwner)
        {
            prefill->close();
        }
        else
        {
            prefill->suspend();
        }
        EXPECT_EQ(page->cacheLevel, kHotLevel);
        EXPECT_EQ(decoder->pageStorageVersion(), version);
        EXPECT_TRUE(batch.dirtyRows().empty());
        EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{0}));
        EXPECT_EQ(decoder->historyLength(), 8);
        EXPECT_GT(decoder->pageStorageVersion(), version);
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(pageAt(*decoder, 1)->slotId(), hostSlot);
        EXPECT_EQ(observer->encodedPages, 2);
        auto const finalSnapshot = decoder->getPageStorageSnapshot(group);
        EXPECT_EQ(finalSnapshot.eligibleHistoryBlocks(), 2);
        EXPECT_EQ(read(batch.numBlocksAddress(group), 1), (std::vector<int32_t>{2}));
        EXPECT_EQ(read(batch.pageTableAddress(group), 3), finalSnapshot.basePageIndices());
        EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{}));
        EXPECT_EQ(observer->encodeCalls, 2);
    }
}

TEST_F(KvCacheManagerV2BatchTest, DeferredOffloadRetainsWorkAfterHostOomAndCodecRejection)
{
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(sparseConfig(), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto page = seedPrefix(*manager, kHotLevel);
    auto decoder = manager->createKvCache({}, tokens());
    auto prefill = manager->createKvCache({}, tokens());
    auto closeCaches = FuncGuard(
        [&]()
        {
            decoder->close();
            prefill->close();
        });
    ASSERT_TRUE(decoder->resume(stream()));
    ASSERT_TRUE(prefill->resume(stream()));
    auto const freeHost = storage.getStatistics(kSparseHistoryLevel).free;
    auto blockers = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{freeHost});
    auto releaseBlockers = FuncGuard(
        [&]()
        {
            for (auto& slot : blockers[LifeCycleId{0}])
            {
                storage.releaseSlot(LifeCycleId{0}, kSparseHistoryLevel, std::move(slot));
            }
        });
    ASSERT_TRUE(decoder->enterDecode());
    Batch batch(manager, 1, 1);
    batch.add(*decoder);
    auto const group = manager->getLayerGroupId(0);
    batch.publish(batchStream());
    auto const gpuSlot = page->slotId();
    prefill->close();
    EXPECT_THROW(batch.publish(batchStream()), OutOfPagesError);
    EXPECT_EQ(observer->encodeCalls, 0);
    EXPECT_EQ(page->slotId(), gpuSlot);
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(read(batch.numBlocksAddress(group), 1), (std::vector<int32_t>{0}));
    releaseBlockers.run();

    observer->rejectEncodeCall = 1;
    EXPECT_THROW(batch.publish(batchStream()), TllmException);
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(page->slotId(), gpuSlot);
    EXPECT_TRUE(decoder->isDecoding());
    EXPECT_EQ(decoder->historyLength(), 4);
    EXPECT_EQ(decoder->getPageStorageSnapshot(group).eligibleHistoryBlocks(), 0);
    EXPECT_EQ(read(batch.pageTableAddress(group), 1), (std::vector<int32_t>{slotIdToPageIndexValue(gpuSlot)}));
    EXPECT_EQ(read(batch.numBlocksAddress(group), 1), (std::vector<int32_t>{0}));
    observer->rejectEncodeCall = 0;
    EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{0}));
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(read(batch.numBlocksAddress(group), 1), (std::vector<int32_t>{1}));
    EXPECT_EQ(storage.getStatistics(kSparseHistoryLevel).free, freeHost - 1);
}

TEST_F(KvCacheManagerV2BatchTest, RetainsStagingUntilUploadAndOrdersTableReuseAfterReaders)
{
    auto config = sparseConfig();
    std::get<AttentionLayerConfig>(config.layers.front()).buffers.front().size = 4096;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    Batch batch(manager, 1, 2);
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 0));
    batch.add(*cache);
    batch.publish(batchStream());
    auto const expected = read(batch.pageTableAddress(LifeCycleId{0}), 2);
    HostMem readback(2 * sizeof(int32_t));
    cudaStream_t readerStream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&readerStream, cudaStreamNonBlocking), cudaSuccess);
    auto destroyReader = FuncGuard([&]() { cudaStreamDestroy(readerStream); });
    StreamGate uploadGate;
    StreamGate readerGate;
    auto releaseGates = FuncGuard(
        [&]()
        {
            uploadGate.release();
            readerGate.release();
        });
    ASSERT_EQ(uploadGate.enqueue(mStream), cudaSuccess);
    EXPECT_THROW(cache->bindPageStorageRow(std::nullopt), LogicError);
    cache->setBasePageIndexBuf(kDefaultBeamIndex, LifeCycleId{0}, nullptr, 0);
    // Commit changes lock ownership/readiness even when the numeric slot stays unchanged.
    cache->commit(tokens());
    batch.publish(batchStream());
    batch.waitReady(reinterpret_cast<CudaStream>(readerStream));
    ASSERT_EQ(readerGate.enqueue(readerStream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(readback.address()),
                  reinterpret_cast<void const*>(batch.pageTableAddress(LifeCycleId{0})), 2 * sizeof(int32_t),
                  cudaMemcpyDeviceToHost, readerStream),
        cudaSuccess);
    batch.recordRead(reinterpret_cast<CudaStream>(readerStream));
    cache->close();
    batch.publish(batchStream());
    CachedCudaEvent cleared(batchStream());
    EXPECT_FALSE(cleared.queryComplete());
    uploadGate.release();
    EXPECT_FALSE(cleared.queryComplete());
    readerGate.release();
    cleared.synchronize();
    auto const* oldIndices = reinterpret_cast<int32_t const*>(readback.address());
    EXPECT_EQ(std::vector<int32_t>(oldIndices, oldIndices + 2), expected);
    EXPECT_EQ(read(batch.pageTableAddress(LifeCycleId{0}), 2), (std::vector<int32_t>{-1, -1}));
}

TEST_F(KvCacheManagerV2BatchTest, DeviceAddressesSurviveGraphReplayAcrossPublication)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    Batch batch(manager, 1, 2);
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 0));
    batch.add(*cache);
    batch.publish(batchStream());
    batch.waitReady(batchStream());
    auto const table = batch.pageTableAddress(LifeCycleId{0});
    void* output = nullptr;
    ASSERT_EQ(cudaMalloc(&output, 2 * sizeof(int32_t)), cudaSuccess);
    auto freeOutput = FuncGuard([&]() { cudaFree(output); });
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    auto destroyGraph = FuncGuard(
        [&]()
        {
            if (executable)
            {
                cudaGraphExecDestroy(executable);
            }
            if (graph)
            {
                cudaGraphDestroy(graph);
            }
        });
    ASSERT_EQ(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    EXPECT_THROW(batch.publish(batchStream()), LogicError);
    ASSERT_EQ(cudaMemcpyAsync(
                  output, reinterpret_cast<void const*>(table), 2 * sizeof(int32_t), cudaMemcpyDeviceToDevice, mStream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamEndCapture(mStream, &graph), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiateWithFlags(&executable, graph, 0), cudaSuccess);
    for (bool active : {true, false, true})
    {
        if (active && !cache->isActive())
        {
            ASSERT_TRUE(cache->resume());
        }
        if (!active)
        {
            cache->suspend();
        }
        batch.publish(batchStream());
        batch.waitReady(batchStream());
        ASSERT_EQ(cudaGraphLaunch(executable, mStream), cudaSuccess);
        batch.recordRead(batchStream());
        auto expected = std::vector<int32_t>{-1, -1};
        if (active)
        {
            expected[0] = cache->getBasePageIndices(LifeCycleId{0})[0];
        }
        EXPECT_EQ(read(reinterpret_cast<MemAddress>(output), 2), expected);
        EXPECT_EQ(batch.pageTableAddress(LifeCycleId{0}), table);
    }
}

TEST_F(KvCacheManagerV2BatchTest, PublicationWaitsForOffloadAndReaderFencesProtectHostSlots)
{
    for (bool commit : {false, true})
    {
        SCOPED_TRACE(commit);
        auto manager = std::make_shared<KvCacheManager>(sparseConfig());
        Batch batch(manager, 1, 1);
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        auto const lc = LifeCycleId{0};
        auto const gpuPool = storage.getPoolGroupIndex(kHotLevel, lc);
        auto const hostPool = storage.getPoolGroupIndex(kSparseHistoryLevel, lc);
        size_t const bytes = storage.slotSize(kHotLevel, gpuPool)[PoolIndex{0}];
        cudaStream_t readerStream{};
        ASSERT_EQ(cudaStreamCreateWithFlags(&readerStream, cudaStreamNonBlocking), cudaSuccess);
        auto destroyReader = FuncGuard([&]() { cudaStreamDestroy(readerStream); });
        void* gpuReadback = nullptr;
        void* hostReadback = nullptr;
        ASSERT_EQ(cudaMalloc(&gpuReadback, bytes), cudaSuccess);
        auto freeGpuReadback = FuncGuard([&]() { cudaFree(gpuReadback); });
        ASSERT_EQ(cudaMallocHost(&hostReadback, bytes), cudaSuccess);
        auto freeHostReadback = FuncGuard([&]() { cudaFreeHost(hostReadback); });
        auto cache = manager->createKvCache();
        auto closeCache = FuncGuard([&]() { cache->close(); });
        ASSERT_TRUE(cache->resume(stream()));
        ASSERT_TRUE(cache->resize(4, 4));
        auto const gpuSlot = pageAt(*cache)->slotId();
        auto const gpuAddress = std::get<MemAddress>(storage.slotAddress(kHotLevel, gpuPool, gpuSlot, PoolIndex{0}));
        constexpr uint8_t kPattern = 0xD3;
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(gpuAddress), kPattern, bytes, mStream), cudaSuccess);
        auto blocker = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseBlocker
            = FuncGuard([&]() { storage.releaseSlot(lc, kSparseHistoryLevel, std::move(blocker[lc].front())); });
        // Warm transfer staging before deliberately delaying the copy.
        storage.copySlotData(lc, kSparseHistoryLevel, kHotLevel, blocker[lc].front().slotId(), gpuSlot, stream());
        batch.add(*cache);
        batch.publish(batchStream());
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        StreamGate copyGate;
        StreamGate readerGate;
        auto releaseGates = FuncGuard(
            [&]()
            {
                copyGate.release();
                readerGate.release();
            });
        ASSERT_EQ(copyGate.enqueue(mStream), cudaSuccess);
        ASSERT_TRUE(cache->enterDecode());
        auto const snapshot = cache->getPageStorageSnapshot(lc);
        ASSERT_EQ(snapshot.eligibleHistoryBlocks(), 1);
        ASSERT_EQ(snapshot.readyEvents().size(), 1);
        EXPECT_FALSE(snapshot.readyEvents().front().queryComplete());
        batch.publish(reinterpret_cast<CudaStream>(readerStream));
        batch.waitReady(reinterpret_cast<CudaStream>(readerStream));
        CachedCudaEvent published(reinterpret_cast<CudaStream>(readerStream));
        EXPECT_FALSE(published.queryComplete());
        ASSERT_EQ(readerGate.enqueue(readerStream), cudaSuccess);
        auto const hostSlot = pageAt(*cache)->slotId();
        auto const hostAddress
            = std::get<MemAddress>(storage.slotAddress(kSparseHistoryLevel, hostPool, hostSlot, PoolIndex{0}));
        ASSERT_EQ(cudaMemcpyAsync(gpuReadback, reinterpret_cast<void const*>(hostAddress), bytes,
                      cudaMemcpyHostToDevice, readerStream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(hostReadback, gpuReadback, bytes, cudaMemcpyDeviceToHost, readerStream), cudaSuccess);
        batch.recordRead(reinterpret_cast<CudaStream>(readerStream));
        if (commit)
            cache->commit(tokens());
        cache->close();
        batch.publish(batchStream());
        manager->clearReusableBlocks();
        auto recycled = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseRecycled
            = FuncGuard([&]() { storage.releaseSlot(lc, kSparseHistoryLevel, std::move(recycled[lc].front())); });
        EXPECT_EQ(recycled[lc].front().slotId(), hostSlot);
        EXPECT_FALSE(recycled[lc].front().queryReady());
        copyGate.release();
        published.synchronize();
        EXPECT_FALSE(recycled[lc].front().queryReady());
        readerGate.release();
        recycled[lc].front().readyEvent.synchronize();
        EXPECT_EQ(read(batch.pageTableAddress(lc), 1), (std::vector<int32_t>{-1}));
        EXPECT_EQ(read(batch.numBlocksAddress(lc), 1), (std::vector<int32_t>{0}));
        auto const* readBytes = static_cast<uint8_t const*>(hostReadback);
        EXPECT_TRUE(std::all_of(readBytes, readBytes + bytes, [](uint8_t v) { return v == kPattern; }));
    }
}

TEST_F(KvCacheManagerV2PageStorageTest, QueriesSparseBuffersAndRejectsUnknownBuffers)
{
    auto config = makeSplitColdGroupingConfig();
    auto& sparse = std::get<AttentionLayerConfig>(config.layers[0]);
    sparse.buffers.front().isSparse = true;
    sparse.buffers.push_back({.role = "value", .size = 2048, .tokensPerBlockOverride = 2, .isSparse = true});
    config.layers.emplace_back(SsmLayerConfig{.layerId = 2, .buffers = {{"state", 4096}}});
    config.commitMinSnapshot = true;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    EXPECT_TRUE(manager->isSparse(0, "key"));
    EXPECT_TRUE(manager->isSparse(0, "value"));
    EXPECT_FALSE(manager->isSparse(1, "key"));
    EXPECT_FALSE(manager->isSparse(2, "state"));
    EXPECT_THROW(manager->isSparse(0, "missing"), std::out_of_range);
    EXPECT_THROW(manager->isSparse(3, "key"), std::out_of_range);
}

TEST_F(KvCacheManagerV2PageStorageTest, SnapshotsRawMixedTierIndicesAndDecodeEligibility)
{
    auto config = makeSplitColdGroupingConfig();
    auto& sparse = std::get<AttentionLayerConfig>(config.layers[0]);
    sparse.buffers.front().isSparse = true;
    sparse.buffers.push_back({.role = "value", .size = 2048, .tokensPerBlockOverride = 2, .isSparse = true});
    auto coalesced = sparse;
    coalesced.layerId = 2;
    config.layers.emplace_back(std::move(coalesced));
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto const sparseGroup = manager->getLayerGroupId(0);
    auto const denseGroup = manager->getLayerGroupId(1);
    EXPECT_EQ(manager->getLayerGroupId(2), sparseGroup);
    EXPECT_GT(manager->getPageIndexScale(0, "value"), 1);
    auto cache = manager->createKvCache();
    std::vector<int32_t> externalIndices(8, kBadPageIndex.value());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(12, 6));
    cache->setBasePageIndexBuf(kDefaultBeamIndex, sparseGroup, externalIndices.data(), externalIndices.size());
    auto const prefill = cache->getPageStorageSnapshot(sparseGroup);
    EXPECT_EQ(prefill.eligibleHistoryBlocks(), 0);
    EXPECT_EQ(prefill.cacheLevels(), (std::vector<std::optional<CacheLevel>>(3, kHotLevel)));

    ASSERT_TRUE(cache->enterDecode());
    auto const decode = cache->getPageStorageSnapshot(sparseGroup);
    EXPECT_EQ(decode.eligibleHistoryBlocks(), 1);
    ASSERT_EQ(decode.basePageIndices().size(), 3);
    EXPECT_EQ(decode.basePageIndices(), (std::vector<int>(externalIndices.begin(), externalIndices.begin() + 3)));
    EXPECT_EQ(
        decode.cacheLevels(), (std::vector<std::optional<CacheLevel>>{kSparseHistoryLevel, kHotLevel, kHotLevel}));
    EXPECT_EQ(cache->getPageStorageSnapshot(denseGroup).eligibleHistoryBlocks(), 0);
    EXPECT_EQ(prefill.cacheLevels()[0], kHotLevel);
    EXPECT_GT(decode.version(), prefill.version());
    for (int ord = 0; ord < 3; ++ord)
        EXPECT_EQ(decode.basePageIndices()[ord], slotIdToPageIndexValue(pageAt(*cache, ord, sparseGroup)->slotId()));

    cache->suspend();
    auto const suspended = cache->getPageStorageSnapshot(sparseGroup);
    EXPECT_EQ(suspended.eligibleHistoryBlocks(), 0);
    EXPECT_EQ(suspended.basePageIndices(), (std::vector<int>(3, kBadPageIndex.value())));
    EXPECT_EQ(suspended.cacheLevels(), (std::vector<std::optional<CacheLevel>>(3, std::nullopt)));
    EXPECT_TRUE(suspended.readyEvents().empty());
    ASSERT_TRUE(cache->resume());
    EXPECT_EQ(cache->getPageStorageSnapshot(sparseGroup).eligibleHistoryBlocks(), 1);
    ASSERT_TRUE(cache->resize(12, 8));
    EXPECT_EQ(cache->getPageStorageSnapshot(sparseGroup).eligibleHistoryBlocks(), 2);
    EXPECT_EQ(pageAt(*cache, 2, sparseGroup)->cacheLevel, kHotLevel);
}

TEST_F(KvCacheManagerV2PageStorageTest, SnapshotsSkipSharedPromptBeamRows)
{
    auto config = sparseConfig();
    config.enablePartialCommit = false;
    std::get<AttentionLayerConfig>(config.layers.front()).buffers.front().size = 4096;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto cache = manager->createKvCache({}, {}, std::nullopt, {}, /*expectedPromptLength=*/8);
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(12, 8));
    cache->setBeamWidth(BeamIndex{3});

    for (BeamIndex bi{0}; bi < cache->beamWidth(); ++bi)
    {
        auto const snapshot = cache->getPageStorageSnapshot(LifeCycleId{0}, bi);
        ASSERT_EQ(snapshot.basePageIndices().size(), 3U);
        ASSERT_EQ(snapshot.cacheLevels().size(), 3U);
        EXPECT_EQ(snapshot.eligibleHistoryBlocks(), 0);
        EXPECT_EQ(snapshot.readyEvents().size(), bi == kDefaultBeamIndex ? 3U : 1U);
        for (int ordinal = 0; ordinal < 3; ++ordinal)
        {
            if (bi != kDefaultBeamIndex && ordinal < 2)
            {
                EXPECT_EQ(snapshot.basePageIndices()[ordinal], kBadPageIndex.value());
                EXPECT_EQ(snapshot.cacheLevels()[ordinal], std::nullopt);
            }
            else
            {
                auto const& page = blockPageGetPage(cache->blocks()[BlockOrdinal{ordinal}].pages[bi][LifeCycleId{0}]);
                ASSERT_NE(page, nullptr);
                EXPECT_EQ(snapshot.basePageIndices()[ordinal], slotIdToPageIndexValue(page->slotId()));
                EXPECT_EQ(snapshot.cacheLevels()[ordinal], kHotLevel);
            }
        }
    }
}

TEST_F(KvCacheManagerV2PageStorageTest, DirtyAcknowledgmentTracksBindingsCommitAndRequestLifetime)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto cache = manager->createKvCache();
    std::vector<int32_t> externalIndices(2, kBadPageIndex.value());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    EXPECT_TRUE(cache->pageStorageDirty());
    EXPECT_FALSE(cache->pageStorageRow().has_value());
    EXPECT_THROW(cache->bindPageStorageRow(-1), LogicError);
    cache->bindPageStorageRow(7);
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(8, 4));
    auto const prefill = cache->getPageStorageSnapshot(LifeCycleId{0});
    EXPECT_EQ(prefill.row(), 7);
    ASSERT_TRUE(cache->acknowledgePageStorage(prefill.version()));
    EXPECT_FALSE(cache->pageStorageDirty());
    ASSERT_TRUE(cache->resize(8, 5));
    EXPECT_FALSE(cache->pageStorageDirty());
    ASSERT_TRUE(cache->enterDecode());
    EXPECT_TRUE(cache->pageStorageDirty());
    EXPECT_FALSE(cache->acknowledgePageStorage(prefill.version()));
    auto const decode = cache->getPageStorageSnapshot(LifeCycleId{0});
    ASSERT_TRUE(cache->acknowledgePageStorage(decode.version()));
    ASSERT_TRUE(cache->enterDecode());
    EXPECT_FALSE(cache->pageStorageDirty());

    cache->bindPageStorageRow(7);
    EXPECT_TRUE(cache->pageStorageDirty());
    EXPECT_FALSE(cache->acknowledgePageStorage(decode.version()));
    cache->bindPageStorageRow(9);
    cache->setBasePageIndexBuf(kDefaultBeamIndex, LifeCycleId{0}, externalIndices.data(), externalIndices.size());
    auto const rebound = cache->getPageStorageSnapshot(LifeCycleId{0});
    EXPECT_EQ(rebound.row(), 9);
    EXPECT_EQ(rebound.basePageIndices(), decode.basePageIndices());
    ASSERT_TRUE(cache->acknowledgePageStorage(rebound.version()));
    cache->setBasePageIndexBuf(kDefaultBeamIndex, LifeCycleId{0}, nullptr, 0);
    EXPECT_TRUE(cache->pageStorageDirty());

    auto const beforeCommit = cache->getPageStorageSnapshot(LifeCycleId{0});
    ASSERT_TRUE(cache->acknowledgePageStorage(beforeCommit.version()));
    cache->commit(tokens());
    EXPECT_TRUE(cache->pageStorageDirty());
    auto const committed = cache->getPageStorageSnapshot(LifeCycleId{0});
    EXPECT_EQ(committed.basePageIndices(), beforeCommit.basePageIndices());
    EXPECT_EQ(committed.cacheLevels(), beforeCommit.cacheLevels());
    ASSERT_TRUE(cache->acknowledgePageStorage(committed.version()));
    cache->suspend();
    EXPECT_TRUE(cache->pageStorageDirty());
    EXPECT_EQ(cache->pageStorageRow(), 9);
    ASSERT_TRUE(cache->resume());
    EXPECT_EQ(cache->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(), 1);
    cache->close();
    EXPECT_TRUE(cache->pageStorageDirty());
    EXPECT_FALSE(cache->pageStorageRow().has_value());
    auto const closed = cache->getPageStorageSnapshot(LifeCycleId{0});
    EXPECT_TRUE(closed.basePageIndices().empty());
    EXPECT_EQ(closed.eligibleHistoryBlocks(), 0);
    EXPECT_THROW(cache->bindPageStorageRow(9), LogicError);
    EXPECT_THROW(cache->recordPageStorageRead(reinterpret_cast<CudaStream>(mStream)), LogicError);

    auto reused = manager->createKvCache({}, tokens());
    auto closeReused = FuncGuard([&]() { reused->close(); });
    EXPECT_TRUE(reused->pageStorageDirty());
    EXPECT_FALSE(reused->pageStorageRow().has_value());
    EXPECT_EQ(reused->getPageStorageSnapshot(LifeCycleId{0}).basePageIndices(), (std::vector<int>{-1}));
    ASSERT_TRUE(reused->resume(stream(), true));
    EXPECT_EQ(reused->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(), 1);
}

TEST_F(KvCacheManagerV2PageStorageTest, ReadinessAndReaderFencesSurviveCommitAndClose)
{
    for (bool commit : {false, true})
    {
        SCOPED_TRACE(commit);
        auto manager = std::make_shared<KvCacheManager>(sparseConfig());
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        auto const lc = LifeCycleId{0};
        auto const gpuPool = storage.getPoolGroupIndex(kHotLevel, lc);
        auto const hostPool = storage.getPoolGroupIndex(kSparseHistoryLevel, lc);
        size_t const bytes = storage.slotSize(kHotLevel, gpuPool)[PoolIndex{0}];
        cudaStream_t readerStream{};
        ASSERT_EQ(cudaStreamCreateWithFlags(&readerStream, cudaStreamNonBlocking), cudaSuccess);
        auto destroyReader = FuncGuard([&]() { cudaStreamDestroy(readerStream); });
        void* gpuReadback = nullptr;
        void* hostReadback = nullptr;
        ASSERT_EQ(cudaMalloc(&gpuReadback, bytes), cudaSuccess);
        auto freeGpuReadback = FuncGuard([&]() { cudaFree(gpuReadback); });
        ASSERT_EQ(cudaMallocHost(&hostReadback, bytes), cudaSuccess);
        auto freeHostReadback = FuncGuard([&]() { cudaFreeHost(hostReadback); });
        auto cache = manager->createKvCache();
        auto closeCache = FuncGuard([&]() { cache->close(); });
        ASSERT_TRUE(cache->resume(stream()));
        ASSERT_TRUE(cache->resize(4, 4));
        auto const gpuSlot = pageAt(*cache)->slotId();
        auto const gpuAddress = std::get<MemAddress>(storage.slotAddress(kHotLevel, gpuPool, gpuSlot, PoolIndex{0}));
        constexpr uint8_t kPattern = 0xD3;
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(gpuAddress), kPattern, bytes, mStream), cudaSuccess);
        auto blocker = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseBlocker
            = FuncGuard([&]() { storage.releaseSlot(lc, kSparseHistoryLevel, std::move(blocker[lc].front())); });
        // Warm transfer staging before deliberately delaying the copy.
        storage.copySlotData(lc, kSparseHistoryLevel, kHotLevel, blocker[lc].front().slotId(), gpuSlot, stream());
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        StreamGate copyGate;
        StreamGate readerGate;
        auto releaseGates = FuncGuard(
            [&]()
            {
                copyGate.release();
                readerGate.release();
            });
        ASSERT_EQ(copyGate.enqueue(mStream), cudaSuccess);
        ASSERT_TRUE(cache->enterDecode());
        auto const snapshot = cache->getPageStorageSnapshot(lc);
        ASSERT_EQ(snapshot.eligibleHistoryBlocks(), 1);
        ASSERT_EQ(snapshot.readyEvents().size(), 1);
        EXPECT_FALSE(snapshot.readyEvents().front().queryComplete());
        snapshot.waitReady(reinterpret_cast<CudaStream>(readerStream));
        ASSERT_EQ(readerGate.enqueue(readerStream), cudaSuccess);
        auto const hostSlot = pageAt(*cache)->slotId();
        auto const hostAddress
            = std::get<MemAddress>(storage.slotAddress(kSparseHistoryLevel, hostPool, hostSlot, PoolIndex{0}));
        ASSERT_EQ(cudaMemcpyAsync(gpuReadback, reinterpret_cast<void const*>(hostAddress), bytes,
                      cudaMemcpyHostToDevice, readerStream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(hostReadback, gpuReadback, bytes, cudaMemcpyDeviceToHost, readerStream), cudaSuccess);
        cache->recordPageStorageRead(reinterpret_cast<CudaStream>(readerStream));
        if (commit)
            cache->commit(tokens());
        cache->close();
        manager->clearReusableBlocks();
        auto recycled = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseRecycled
            = FuncGuard([&]() { storage.releaseSlot(lc, kSparseHistoryLevel, std::move(recycled[lc].front())); });
        EXPECT_EQ(recycled[lc].front().slotId(), hostSlot);
        EXPECT_FALSE(recycled[lc].front().queryReady());
        copyGate.release();
        snapshot.readyEvents().front().synchronize();
        EXPECT_FALSE(recycled[lc].front().queryReady());
        readerGate.release();
        recycled[lc].front().readyEvent.synchronize();
        auto const* readBytes = static_cast<uint8_t const*>(hostReadback);
        EXPECT_TRUE(std::all_of(readBytes, readBytes + bytes, [](uint8_t v) { return v == kPattern; }));
    }
}

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
    auto const versionBeforeOffload = cache->pageStorageVersion();
    cache->offloadSparsePages(targets);
    EXPECT_EQ(observer->encodeCalls, 1);
    EXPECT_EQ(observer->encodedPages, pages.size());
    EXPECT_EQ(observer->encodeStream, mStream);
    EXPECT_EQ(storage.getStatistics(kHotLevel).free, gpuFree + pages.size());
    EXPECT_EQ(storage.getStatistics(kSparseHistoryLevel).free, hostFree - pages.size());
    EXPECT_GT(cache->pageStorageVersion(), versionBeforeOffload);

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
    auto const versionAfterOffload = cache->pageStorageVersion();
    cache->offloadSparsePages(targets);
    EXPECT_EQ(observer->encodeCalls, 1);
    EXPECT_EQ(cache->pageStorageVersion(), versionAfterOffload);
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
    auto const firstVersion = first->pageStorageVersion();
    auto const secondVersion = second->pageStorageVersion();
    ASSERT_TRUE(first->acknowledgePageStorage(firstVersion));
    ASSERT_TRUE(second->acknowledgePageStorage(secondVersion));
    first->offloadSparsePages({page, page});
    EXPECT_NE(page->slotId(), gpuSlot);
    EXPECT_EQ(first->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(page->slotId()));
    EXPECT_EQ(externalIndices[0], slotIdToPageIndexValue(page->slotId()));
    EXPECT_GT(first->pageStorageVersion(), firstVersion);
    EXPECT_GT(second->pageStorageVersion(), secondVersion);
    EXPECT_TRUE(first->pageStorageDirty());
    EXPECT_TRUE(second->pageStorageDirty());
    auto const snapshot = second->getPageStorageSnapshot(LifeCycleId{0});
    EXPECT_EQ(snapshot.basePageIndices(), externalIndices);
    EXPECT_EQ(snapshot.cacheLevels()[0], kSparseHistoryLevel);
    EXPECT_EQ(snapshot.eligibleHistoryBlocks(), 0);
    EXPECT_EQ(storage.getStatistics(kHotLevel).free, storage.getStatistics(kHotLevel).total);
    EXPECT_FALSE(storage.isEvictable(*page));
    EXPECT_THROW(storage.batchedMigrate(kHotLevel, {page}, {}), LogicError);
    ASSERT_TRUE(first->enterDecode());
    ASSERT_TRUE(second->enterDecode());
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
    for (auto const [deferred, finishReader] :
        {std::pair{false, false}, std::pair{false, true}, std::pair{true, false}, std::pair{true, true}})
    {
        SCOPED_TRACE(deferred);
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
        if (deferred)
        {
            ASSERT_TRUE(first->enterDecode());
            EXPECT_EQ(page->cacheLevel, kHotLevel);
        }
        StreamGate gate;
        ASSERT_EQ(gate.enqueue(readerStream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(readback), reinterpret_cast<void const*>(gpuAddress), bytes,
                      cudaMemcpyDeviceToHost, readerStream),
            cudaSuccess);
        if (finishReader)
        {
            second->suspend();
        }

        if (deferred)
        {
            ASSERT_TRUE(finishReader ? first->resize(4, 4) : second->enterDecode());
            EXPECT_EQ(first->historyLength(), 4);
        }
        else
        {
            first->offloadSparsePages({page});
        }
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
    auto const version = cache->pageStorageVersion();
    EXPECT_THROW(cache->offloadSparsePages({first, second}), OutOfPagesError);
    EXPECT_EQ(observer->encodeCalls, 0);
    EXPECT_EQ(first->cacheLevel, kHotLevel);
    EXPECT_EQ(second->cacheLevel, kHotLevel);
    EXPECT_EQ(first->slotId(), firstSlot);
    EXPECT_EQ(second->slotId(), secondSlot);
    EXPECT_EQ(first->status(), PageStatus::LOCKED);
    EXPECT_EQ(second->status(), PageStatus::LOCKED);
    EXPECT_EQ(cache->pageStorageVersion(), version);
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
    auto const version = cache->pageStorageVersion();
    EXPECT_THROW(cache->offloadSparsePages({page}), TllmException);
    ASSERT_TRUE(rejecting->launched());
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(page->slotId(), gpuSlot);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    EXPECT_FALSE(page->queryReady());
    EXPECT_GT(cache->pageStorageVersion(), version);
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
    auto const version = cache->pageStorageVersion();
    EXPECT_THROW(cache->offloadSparsePages({first, second}), TllmException);
    EXPECT_EQ(observer->encodeCalls, 2);
    EXPECT_EQ(first->cacheLevel, kHotLevel);
    EXPECT_EQ(second->cacheLevel, kHotLevel);
    EXPECT_EQ(first->slotId(), firstSlot);
    EXPECT_EQ(second->slotId(), secondSlot);
    EXPECT_GT(cache->pageStorageVersion(), version);
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
            auto const version = cache->pageStorageVersion();
            EXPECT_THROW(cache->offloadSparsePages({first, input}), LogicError);
            EXPECT_EQ(observer->encodeCalls, 0);
            EXPECT_EQ(first->cacheLevel, kHotLevel);
            EXPECT_EQ(input->cacheLevel, kHotLevel);
            EXPECT_EQ(cache->pageStorageVersion(), version);
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

    auto const version = cache->pageStorageVersion();
    cache->offloadSparsePages({page, page});
    EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[0], originalIndex);
    EXPECT_GT(cache->pageStorageVersion(), version);
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
    ASSERT_TRUE(cache->enterDecode());
    auto const hostSlot = pageAt(*cache)->slotId();
    cache->commit(tokens());
    auto committed = pageAt(*cache);
    ASSERT_TRUE(committed->isCommitted());
    EXPECT_EQ(committed->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(committed->slotId(), hostSlot);
    auto reused = manager->createKvCache({}, tokens());
    auto closeReused = FuncGuard([&]() { reused->close(); });
    ASSERT_TRUE(reused->resume(stream(), true));
    EXPECT_EQ(pageAt(*reused), committed);
    cache->close();
    EXPECT_EQ(committed->status(), PageStatus::LOCKED);
    EXPECT_EQ(reused->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(hostSlot));
}

class KvCacheManagerV2DecodeOffloadTest : public KvCacheManagerV2PageLockTest
{
};

TEST_F(KvCacheManagerV2DecodeOffloadTest, PrefillRetainsSparseSwaHistoryAndRestoresGpuStorage)
{
    auto config = sparseConfig();
    config.swaScratchReuse = SwaScratchReuseConfig{};
    auto& layer = std::get<AttentionLayerConfig>(config.layers.front());
    layer.slidingWindowSize = 4;
    layer.buffers.front().size = 4096;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    cache->setEnableSwaScratchReuse(true);
    ASSERT_TRUE(cache->resize(12, 0));
    EXPECT_FALSE(cache->hasScratchSlots());
    ASSERT_TRUE(cache->resize(12, 8));
    EXPECT_FALSE(cache->isDecoding());
    EXPECT_EQ(cache->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(), 0);
    for (int ord = 0; ord < 3; ++ord)
    {
        ASSERT_NE(pageAt(*cache, ord), nullptr);
        EXPECT_EQ(pageAt(*cache, ord)->cacheLevel, kHotLevel);
    }

    for (bool prefetch : {false, true})
    {
        cache->suspend();
        storage.forceEvict(kHotLevel, TypedVec<PoolGroupIndex, SlotCount>{3});
        for (int ord = 0; ord < 3; ++ord)
            EXPECT_EQ(pageAt(*cache, ord)->cacheLevel, kSparseHistoryLevel);
        if (prefetch)
            ASSERT_TRUE(cache->prefetch(kHotLevel));
        ASSERT_TRUE(cache->resume());
        for (int ord = 0; ord < 3; ++ord)
            EXPECT_EQ(pageAt(*cache, ord)->cacheLevel, kHotLevel);
    }
    ASSERT_TRUE(cache->resize(12, 12));
    EXPECT_EQ(cache->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(), 0);
    ASSERT_TRUE(cache->enterDecode());
    for (int ord = 0; ord < 3; ++ord)
    {
        EXPECT_EQ(pageAt(*cache, ord)->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(pageAt(*cache, ord)->status(), PageStatus::LOCKED);
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, EntryScansUnchangedWatermarkAndResumeKeepsHostHistory)
{
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(sparseConfig(), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(8, 8));
    EXPECT_EQ(observer->encodeCalls, 0);
    auto const prefillVersion = cache->pageStorageVersion();
    ASSERT_TRUE(cache->enterDecode());
    EXPECT_TRUE(cache->isDecoding());
    EXPECT_EQ(cache->historyLength(), 8);
    EXPECT_EQ(observer->encodeCalls, 1);
    EXPECT_EQ(observer->encodedPages, 2);
    EXPECT_GT(cache->pageStorageVersion(), prefillVersion);
    auto const decodeVersion = cache->pageStorageVersion();
    ASSERT_TRUE(cache->enterDecode());
    EXPECT_EQ(observer->encodeCalls, 1);
    EXPECT_EQ(cache->pageStorageVersion(), decodeVersion);
    EXPECT_THROW(cache->resize(8, 4), std::invalid_argument);
    cache->suspend();
    EXPECT_THROW(cache->resume(std::nullopt, false), std::invalid_argument);
    ASSERT_TRUE(cache->resume());
    EXPECT_EQ(observer->encodeCalls, 1);
    EXPECT_GT(cache->pageStorageVersion(), decodeVersion);
    for (int ord = 0; ord < 2; ++ord)
    {
        auto const page = pageAt(*cache, ord);
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[ord], slotIdToPageIndexValue(page->slotId()));
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, BeamExpansionOffloadsOnlyExistingHistoryRows)
{
    auto config = sparseConfig();
    config.enablePartialCommit = false;
    std::get<AttentionLayerConfig>(config.layers.front()).buffers.front().size = 4096;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto cache = manager->createKvCache({}, {}, std::nullopt, {}, /*expectedPromptLength=*/8);
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(12, 8));
    cache->setBeamWidth(BeamIndex{3});
    ASSERT_EQ(cache->blocks()[BlockOrdinal{0}].pages.size(), BeamIndex{1});
    ASSERT_EQ(cache->blocks()[BlockOrdinal{1}].pages.size(), BeamIndex{1});
    ASSERT_EQ(cache->blocks()[BlockOrdinal{2}].pages.size(), BeamIndex{3});

    ASSERT_TRUE(cache->enterDecode());
    EXPECT_TRUE(cache->isDecoding());
    for (int historyLength : {8, 12})
    {
        ASSERT_TRUE(cache->resize(16, historyLength));
        for (BlockOrdinal ordinal{0}; ordinal < cache->numBlocks(); ++ordinal)
        {
            auto const& pages = cache->blocks()[ordinal].pages;
            for (BeamIndex bi{0}; bi < pages.size(); ++bi)
            {
                auto const& page = blockPageGetPage(pages[bi][LifeCycleId{0}]);
                ASSERT_NE(page, nullptr);
                EXPECT_EQ(page->cacheLevel,
                    ordinal < BlockOrdinal{historyLength / cache->tokensPerBlock()} ? kSparseHistoryLevel : kHotLevel);
                EXPECT_EQ(page->status(), PageStatus::LOCKED);
                EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0}, bi)[ordinal.value()],
                    slotIdToPageIndexValue(page->slotId()));
            }
        }
        EXPECT_EQ(cache->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(),
            historyLength / cache->tokensPerBlock());
        EXPECT_EQ(cache->getPageStorageSnapshot(LifeCycleId{0}, BeamIndex{1}).eligibleHistoryBlocks(), 0);
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, SparseHistoryIsNotMarkedForTurnEndDrop)
{
    auto config = sparseConfig();
    std::get<AttentionLayerConfig>(config.layers.front()).slidingWindowSize = 4;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kHotLevel);
    auto cache = manager->createKvCache({}, tokens());
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream(), true));
    cache->stopCommitting();
    auto dropPlan = cache->planCommittedBlockDrop();
    EXPECT_EQ(page->plannedDropCount, 0);
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, CachedPrefillRestoresGpuOnResumePrefetchAndRebase)
{
    for (int path : {0, 1, 2})
    {
        SCOPED_TRACE(path);
        auto manager = std::make_shared<KvCacheManager>(sparseConfig());
        auto const apiLock = manager->lockExclusive();
        auto page = seedPrefix(*manager, kSparseHistoryLevel);
        auto cache = path == 2 ? manager->createKvCache() : manager->createKvCache({}, tokens());
        auto closeCache = FuncGuard([&]() { cache->close(); });
        if (path == 1)
        {
            ASSERT_TRUE(cache->prefetch(kHotLevel));
            EXPECT_EQ(page->cacheLevel, kHotLevel);
        }
        ASSERT_TRUE(cache->resume(stream()));
        if (path == 2)
        {
            ASSERT_TRUE(cache->resize(4, 4));
            cache->commit(tokens());
        }
        EXPECT_EQ(pageAt(*cache), page);
        EXPECT_EQ(page->cacheLevel, kHotLevel);
        EXPECT_EQ(cache->historyLength(), 4);
        EXPECT_FALSE(cache->isDecoding());
        EXPECT_EQ(cache->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(), 0);
        EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(page->slotId()));
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, MixedDenseSwaLocksAreRestoredAfterHostOom)
{
    auto config = sparseConfig();
    config.cacheTiers[0] = GpuCacheTierConfig{8 << 20};
    auto& sparse = std::get<AttentionLayerConfig>(config.layers.front());
    sparse.slidingWindowSize = 4;
    auto dense = sparse;
    dense.layerId = 1;
    dense.buffers.front().isSparse = false;
    config.layers.emplace_back(std::move(dense));
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(8, 4));
    auto const densePage = pageAt(*cache, 0, LifeCycleId{1});
    auto const denseSlot = densePage->slotId();
    ASSERT_TRUE(cache->enterDecode());
    EXPECT_EQ(densePage->cacheLevel, kHotLevel);
    EXPECT_EQ(pageAt(*cache)->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(pageAt(*cache, 1)->cacheLevel, kHotLevel);
    auto const version = cache->pageStorageVersion();
    auto const before = cache->getPageStorageSnapshot(LifeCycleId{0});
    ASSERT_TRUE(cache->acknowledgePageStorage(version));
    auto const freeHost = manager->getStorageStatistics(kSparseHistoryLevel)
                              .at(storage.getPoolGroupIndex(kSparseHistoryLevel, LifeCycleId{0}))
                              .free;
    auto blockers = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{freeHost, 0});
    auto releaseBlockers = FuncGuard(
        [&]()
        {
            for (auto& slot : blockers[LifeCycleId{0}])
                storage.releaseSlot(LifeCycleId{0}, kSparseHistoryLevel, std::move(slot));
        });
    EXPECT_FALSE(cache->resize(8, 8));
    EXPECT_EQ(cache->historyLength(), 4);
    EXPECT_GT(cache->pageStorageVersion(), version);
    EXPECT_TRUE(cache->pageStorageDirty());
    EXPECT_FALSE(cache->acknowledgePageStorage(version));
    auto const after = cache->getPageStorageSnapshot(LifeCycleId{0});
    EXPECT_EQ(after.basePageIndices(), before.basePageIndices());
    EXPECT_EQ(after.cacheLevels(), before.cacheLevels());
    EXPECT_EQ(after.eligibleHistoryBlocks(), before.eligibleHistoryBlocks());
    EXPECT_EQ(pageAt(*cache, 0, LifeCycleId{1}), densePage);
    EXPECT_EQ(densePage->status(), PageStatus::LOCKED);
    EXPECT_EQ(densePage->cacheLevel, kHotLevel);
    EXPECT_EQ(densePage->slotId(), denseSlot);
    EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{1})[0], slotIdToPageIndexValue(denseSlot));
    EXPECT_EQ(pageAt(*cache, 1)->cacheLevel, kHotLevel);
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, HistoryUpdatesOffloadOnlyNewFullPages)
{
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(sparseConfig(), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(8, 4));
    ASSERT_TRUE(cache->enterDecode());
    auto const version = cache->pageStorageVersion();
    auto const firstHostSlot = pageAt(*cache)->slotId();
    ASSERT_TRUE(cache->resize(8, 7));
    EXPECT_EQ(observer->encodedPages, 1);
    EXPECT_EQ(cache->pageStorageVersion(), version);
    EXPECT_EQ(pageAt(*cache, 1)->cacheLevel, kHotLevel);
    ASSERT_TRUE(cache->resize(8, 8));
    EXPECT_EQ(observer->encodeCalls, 2);
    EXPECT_EQ(observer->encodedPages, 2);
    EXPECT_GT(cache->pageStorageVersion(), version);
    EXPECT_EQ(pageAt(*cache)->slotId(), firstHostSlot);
    EXPECT_EQ(pageAt(*cache, 1)->cacheLevel, kSparseHistoryLevel);
    ASSERT_TRUE(cache->resize(12));
    ASSERT_TRUE(cache->resize(12, 9));
    EXPECT_EQ(pageAt(*cache, 2)->cacheLevel, kHotLevel);
    EXPECT_EQ(observer->encodedPages, 2);
}

TEST_F(KvCacheManagerV2BatchTest, PrefillPromotesOneSharedPageAndReleaseRetriesOffload)
{
    for (int path : {0, 1, 2, 3})
    {
        SCOPED_TRACE(path);
        auto codec = std::make_unique<ObservingColdPageCodec>();
        auto* observer = codec.get();
        auto manager = std::make_shared<KvCacheManager>(sparseConfig(), nullptr, std::move(codec));
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        auto page = seedPrefix(*manager, kHotLevel);
        auto first = manager->createKvCache({}, tokens());
        auto second = manager->createKvCache({}, tokens());
        std::shared_ptr<KvCache> prefill;
        Batch batch(manager, 3, 1);
        auto closeCaches = FuncGuard(
            [&]()
            {
                if (prefill)
                {
                    prefill->close();
                }
                first->close();
                second->close();
            });
        ASSERT_TRUE(first->resume(stream()));
        ASSERT_TRUE(second->resume(stream()));
        if (path == 1)
        {
            prefill = manager->createKvCache({}, tokens());
            ASSERT_TRUE(prefill->resume(stream()));
            prefill->suspend();
        }
        auto const lc = page->lifeCycle;
        auto const pg = storage.getPoolGroupIndex(kHotLevel, lc);
        size_t const bytes = storage.slotSize(kHotLevel, pg)[PoolIndex{0}];
        auto const originalAddress
            = std::get<MemAddress>(storage.slotAddress(kHotLevel, pg, page->slotId(), PoolIndex{0}));
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(originalAddress), 0x3C, bytes, mStream), cudaSuccess);
        ASSERT_TRUE(first->enterDecode());
        ASSERT_TRUE(second->enterDecode());
        ASSERT_EQ(page->cacheLevel, kSparseHistoryLevel);
        batch.add(*first, 0);
        batch.add(*second, 1);
        batch.publish(batchStream());
        auto const firstVersion = first->pageStorageVersion();
        auto const secondVersion = second->pageStorageVersion();

        if (!prefill)
        {
            prefill = path == 2 ? manager->createKvCache() : manager->createKvCache({}, tokens());
        }
        if (path == 3)
        {
            ASSERT_TRUE(prefill->prefetch(kHotLevel));
        }
        ASSERT_TRUE(prefill->resume(stream()));
        if (path == 2)
        {
            ASSERT_TRUE(prefill->resize(4, 4));
            prefill->commit(tokens());
        }
        EXPECT_EQ(page->cacheLevel, kHotLevel);
        EXPECT_EQ(observer->decodeCalls, 1);
        EXPECT_EQ(observer->decodedPages, 1);
        EXPECT_GT(first->pageStorageVersion(), firstVersion);
        EXPECT_GT(second->pageStorageVersion(), secondVersion);
        for (auto const& cache : {first, second, prefill})
        {
            EXPECT_EQ(pageAt(*cache), page);
            EXPECT_EQ(cache->getBasePageIndices(lc)[0], slotIdToPageIndexValue(page->slotId()));
        }
        EXPECT_EQ(storage.getStatistics(kHotLevel).free, storage.getStatistics(kHotLevel).total - 1);
        EXPECT_EQ(storage.getStatistics(kSparseHistoryLevel).free, storage.getStatistics(kSparseHistoryLevel).total);
        batch.add(*prefill, 2);
        EXPECT_EQ(batch.publish(batchStream()), (std::vector<int>{0, 1, 2}));
        EXPECT_EQ(
            read(batch.pageTableAddress(lc), 3), (std::vector<int32_t>(3, slotIdToPageIndexValue(page->slotId()))));
        EXPECT_EQ(read(batch.numBlocksAddress(lc), 3), (std::vector<int32_t>{0, 0, 0}));
        auto const address = std::get<MemAddress>(storage.slotAddress(kHotLevel, pg, page->slotId(), PoolIndex{0}));
        auto const data = read(address, bytes / sizeof(int32_t));
        EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](int32_t value) { return value == 0x3C3C3C3C; }));

        prefill->close();
        batch.publish(batchStream());
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(observer->encodeCalls, 2);
        EXPECT_EQ(storage.getStatistics(kHotLevel).free, storage.getStatistics(kHotLevel).total);
        EXPECT_EQ(read(batch.numBlocksAddress(lc), 3), (std::vector<int32_t>{1, 1, 0}));
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, SharedPrefillOwnerDefersDemotionAcrossDecodeResume)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
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
    ASSERT_TRUE(second->resume(stream()));
    auto const version = first->pageStorageVersion();
    ASSERT_TRUE(first->enterDecode());
    EXPECT_TRUE(first->isDecoding());
    EXPECT_GT(first->pageStorageVersion(), version);
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(first->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(), 0);
    second->suspend();
    ASSERT_TRUE(first->enterDecode());
    ASSERT_TRUE(second->resume());
    EXPECT_TRUE(second->isActive());
    EXPECT_FALSE(second->isDecoding());
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(pageAt(*second), page);
    first->suspend();
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    ASSERT_TRUE(first->resume());
    EXPECT_TRUE(first->isActive());
    EXPECT_TRUE(first->isDecoding());
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    second->close();
    ASSERT_TRUE(first->enterDecode());
    EXPECT_EQ(first->historyLength(), 4);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, PromotionGpuOomPreservesHostPageAndAllowsRetry)
{
    auto config = sparseConfig();
    config.maxUtilForResume = 1.0f;
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto page = seedPrefix(*manager, kHotLevel);
    auto decoder = manager->createKvCache({}, tokens());
    auto prefill = manager->createKvCache({}, tokens());
    auto closeCaches = FuncGuard(
        [&]()
        {
            prefill->close();
            decoder->close();
        });
    ASSERT_TRUE(decoder->resume(stream()));
    ASSERT_TRUE(decoder->enterDecode());
    auto const hostSlot = page->slotId();
    auto const version = decoder->pageStorageVersion();
    auto const lc = page->lifeCycle;
    auto blockers = storage.newGpuSlots(TypedVec<LifeCycleId, SlotCount>{storage.getStatistics(kHotLevel).free});
    auto releaseBlockers = FuncGuard(
        [&]()
        {
            for (auto& slot : blockers[lc])
            {
                storage.releaseSlot(lc, kHotLevel, std::move(slot));
            }
        });
    EXPECT_FALSE(prefill->resume(stream()));
    EXPECT_FALSE(prefill->isActive());
    EXPECT_TRUE(decoder->isActive());
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(page->slotId(), hostSlot);
    EXPECT_EQ(decoder->pageStorageVersion(), version);
    EXPECT_EQ(decoder->getBasePageIndices(lc)[0], slotIdToPageIndexValue(hostSlot));
    EXPECT_EQ(observer->decodeCalls, 0);
    releaseBlockers.run();
    ASSERT_TRUE(prefill->resume());
    EXPECT_EQ(pageAt(*prefill), page);
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(observer->decodedPages, 1);
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, PromotionCodecBatchFailurePreservesAllHostPages)
{
    auto config = makeSplitColdGroupingConfig();
    config.enableStats = true;
    for (auto& layer : config.layers)
    {
        std::get<AttentionLayerConfig>(layer).buffers.front().isSparse = true;
    }
    std::get<AttentionLayerConfig>(config.layers[1]).buffers.front().size *= 2;
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto decoder = manager->createKvCache();
    std::shared_ptr<KvCache> prefill;
    auto closeCaches = FuncGuard(
        [&]()
        {
            if (prefill)
            {
                prefill->close();
            }
            decoder->close();
        });
    ASSERT_TRUE(decoder->resume(stream()));
    ASSERT_TRUE(decoder->resize(4, 4));
    decoder->commit(tokens());
    ASSERT_TRUE(decoder->enterDecode());
    prefill = manager->createKvCache({}, tokens());
    auto const first = pageAt(*decoder, 0, LifeCycleId{0});
    auto const second = pageAt(*decoder, 0, LifeCycleId{1});
    auto const firstSlot = first->slotId();
    auto const secondSlot = second->slotId();
    auto const version = decoder->pageStorageVersion();
    manager->getAndResetIterationStats();
    observer->rejectDecodeCall = 2;
    EXPECT_THROW(prefill->resume(stream()), TllmException);
    EXPECT_FALSE(prefill->isActive());
    EXPECT_EQ(observer->decodeCalls, 2);
    EXPECT_EQ(first->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(second->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(first->slotId(), firstSlot);
    EXPECT_EQ(second->slotId(), secondSlot);
    EXPECT_EQ(decoder->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(firstSlot));
    EXPECT_EQ(decoder->getBasePageIndices(LifeCycleId{1})[0], slotIdToPageIndexValue(secondSlot));
    EXPECT_GT(decoder->pageStorageVersion(), version);
    EXPECT_EQ(storage.getStatistics(kHotLevel).free, storage.getStatistics(kHotLevel).total);
    EXPECT_TRUE(manager->getAndResetIterationStats().empty());
    observer->rejectDecodeCall = 0;
    ASSERT_TRUE(prefill->resume());
    EXPECT_EQ(pageAt(*prefill, 0, LifeCycleId{0}), first);
    EXPECT_EQ(pageAt(*prefill, 0, LifeCycleId{1}), second);
    EXPECT_EQ(first->cacheLevel, kHotLevel);
    EXPECT_EQ(second->cacheLevel, kHotLevel);
    EXPECT_EQ(storage.getStatistics(kSparseHistoryLevel).free, storage.getStatistics(kSparseHistoryLevel).total);
    auto const stats = manager->getAndResetIterationStats();
    EXPECT_EQ(stats.at(LifeCycleId{0}).iterOnboardBlocks, 1);
    EXPECT_EQ(stats.at(LifeCycleId{1}).iterOnboardBlocks, 1);
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, PromotionRejectionFencesSourceAndRecycledDestination)
{
    auto codec = std::make_unique<AsyncRejectingColdPageCodec>(AsyncRejectingColdPageCodec::Operation::kDecode);
    auto* rejecting = codec.get();
    auto manager = std::make_shared<KvCacheManager>(sparseConfig(), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto page = seedPrefix(*manager, kHotLevel);
    auto decoder = manager->createKvCache({}, tokens());
    auto prefill = manager->createKvCache({}, tokens());
    auto closeCaches = FuncGuard(
        [&]()
        {
            prefill->close();
            decoder->close();
        });
    ASSERT_TRUE(decoder->resume(stream()));
    ASSERT_TRUE(decoder->enterDecode());
    auto const lc = page->lifeCycle;
    auto const hostSlot = page->slotId();
    auto blocker = storage.newGpuSlots(TypedVec<LifeCycleId, SlotCount>{1});
    auto releaseBlocker = FuncGuard([&]() { storage.releaseSlot(lc, kHotLevel, std::move(blocker[lc].front())); });
    auto releaseCodec = FuncGuard([&]() { rejecting->release(); });
    EXPECT_THROW(prefill->resume(stream()), TllmException);
    ASSERT_TRUE(rejecting->launched());
    EXPECT_FALSE(prefill->isActive());
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(page->slotId(), hostSlot);
    EXPECT_FALSE(page->queryReady());
    EXPECT_EQ(decoder->getBasePageIndices(lc)[0], slotIdToPageIndexValue(hostSlot));
    auto recycled = storage.newGpuSlots(TypedVec<LifeCycleId, SlotCount>{1});
    auto releaseRecycled = FuncGuard([&]() { storage.releaseSlot(lc, kHotLevel, std::move(recycled[lc].front())); });
    EXPECT_FALSE(recycled[lc].front().queryReady());
    rejecting->release();
    recycled[lc].front().readyEvent.synchronize();
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, PromotionWaitsForLiveAndFinishedReadersBeforeRecyclingHostSlot)
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
        auto prefill = manager->createKvCache({}, tokens());
        auto closeCaches = FuncGuard(
            [&]()
            {
                prefill->close();
                first->close();
                second->close();
            });
        ASSERT_TRUE(first->resume(stream()));
        ASSERT_TRUE(second->resume(reinterpret_cast<CUstream>(readerStream)));
        auto const lc = page->lifeCycle;
        auto const gpuPool = storage.getPoolGroupIndex(kHotLevel, lc);
        auto const hostPool = storage.getPoolGroupIndex(kSparseHistoryLevel, lc);
        size_t const bytes = storage.slotSize(kHotLevel, gpuPool)[PoolIndex{0}];
        auto const originalAddress
            = std::get<MemAddress>(storage.slotAddress(kHotLevel, gpuPool, page->slotId(), PoolIndex{0}));
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(originalAddress), 0x3C, bytes, mStream), cudaSuccess);
        ASSERT_TRUE(first->enterDecode());
        ASSERT_TRUE(second->enterDecode());
        auto const hostSlot = page->slotId();
        auto const hostAddress
            = std::get<MemAddress>(storage.slotAddress(kSparseHistoryLevel, hostPool, hostSlot, PoolIndex{0}));
        auto readback = storage.newGpuSlots(TypedVec<LifeCycleId, SlotCount>{1});
        auto hostBlocker = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseSlots = FuncGuard(
            [&]()
            {
                storage.releaseSlot(lc, kHotLevel, std::move(readback[lc].front()));
                storage.releaseSlot(lc, kSparseHistoryLevel, std::move(hostBlocker[lc].front()));
            });
        auto const readbackAddress = std::get<MemAddress>(
            storage.slotAddress(kHotLevel, gpuPool, readback[lc].front().slotId(), PoolIndex{0}));
        // Warm the H2D codec before holding a CUDA callback.
        storage.copySlotData(lc, kHotLevel, kSparseHistoryLevel, readback[lc].front().slotId(), hostSlot, stream());
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::pair<MemAddress, size_t> overwrite{hostAddress, bytes};
        StreamGate gate;
        ASSERT_EQ(gate.enqueue(readerStream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(readbackAddress), reinterpret_cast<void const*>(hostAddress),
                      bytes, cudaMemcpyHostToDevice, readerStream),
            cudaSuccess);
        if (finishReader)
        {
            second->suspend();
        }
        ASSERT_TRUE(prefill->resume(stream()));
        EXPECT_EQ(page->cacheLevel, kHotLevel);
        EXPECT_FALSE(page->queryReady());
        EXPECT_EQ(pageAt(*prefill), page);
        auto recycled = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseRecycled
            = FuncGuard([&]() { storage.releaseSlot(lc, kSparseHistoryLevel, std::move(recycled[lc].front())); });
        EXPECT_EQ(recycled[lc].front().slotId(), hostSlot);
        EXPECT_FALSE(recycled[lc].front().queryReady());
        recycled[lc].front().readyEvent.waitInStream(reinterpret_cast<CudaStream>(mStream));
        ASSERT_EQ(cudaLaunchHostFunc(
                      mStream,
                      [](void* data)
                      {
                          auto const& [address, size] = *static_cast<std::pair<MemAddress, size_t> const*>(data);
                          std::memset(reinterpret_cast<void*>(address), 0, size);
                      },
                      &overwrite),
            cudaSuccess);
        gate.release();
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        auto const promotedAddress
            = std::get<MemAddress>(storage.slotAddress(kHotLevel, gpuPool, page->slotId(), PoolIndex{0}));
        for (auto const address : {readbackAddress, promotedAddress})
        {
            std::vector<uint8_t> data(bytes);
            cuCheck(cuMemcpyDtoH(data.data(), address, bytes));
            EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](uint8_t value) { return value == 0x3C; }));
        }
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, SharedOwnersEnterDecodeWithoutMutuallyBlocking)
{
    auto config = sparseConfig();
    config.enableStats = true;
    auto codec = std::make_unique<ObservingColdPageCodec>();
    auto* observer = codec.get();
    auto manager = std::make_shared<KvCacheManager>(std::move(config), nullptr, std::move(codec));
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kHotLevel);
    auto first = manager->createKvCache({}, tokens());
    auto second = manager->createKvCache({}, tokens());
    auto third = manager->createKvCache({}, tokens());
    auto closeCaches = FuncGuard(
        [&]()
        {
            first->close();
            second->close();
            third->close();
        });
    for (auto const& cache : {first, second, third})
    {
        ASSERT_TRUE(cache->resume(stream()));
        EXPECT_EQ(pageAt(*cache), page);
    }
    auto const gpuSlot = page->slotId();
    auto const hostFree = manager->storage().getStatistics(kSparseHistoryLevel).free;
    ASSERT_TRUE(first->enterDecode());
    ASSERT_TRUE(second->enterDecode());
    auto const version = first->pageStorageVersion();
    ASSERT_TRUE(third->resize(4, 4));
    ASSERT_TRUE(first->enterDecode());
    ASSERT_TRUE(second->resize(4, 4));
    EXPECT_EQ(first->pageStorageVersion(), version);
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(page->slotId(), gpuSlot);
    EXPECT_EQ(observer->encodeCalls, 0);
    EXPECT_EQ(manager->storage().getStatistics(kSparseHistoryLevel).free, hostFree);
    EXPECT_EQ(first->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(), 0);

    ASSERT_TRUE(third->enterDecode());
    EXPECT_EQ(observer->encodedPages, 1);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    for (auto const& cache : {first, second, third})
    {
        EXPECT_EQ(cache->historyLength(), 4);
        auto const snapshot = cache->getPageStorageSnapshot(LifeCycleId{0});
        EXPECT_EQ(snapshot.eligibleHistoryBlocks(), 1);
        EXPECT_EQ(snapshot.basePageIndices()[0], slotIdToPageIndexValue(page->slotId()));
        ASSERT_TRUE(cache->enterDecode());
    }
    EXPECT_EQ(observer->encodeCalls, 1);
    EXPECT_EQ(manager->getAndResetIterationStats().at(LifeCycleId{0}).iterOffloadBlocks, 1);
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, DeferredOwnerCanCloseBeforePrefillOwners)
{
    for (bool const closeAll : {false, true})
    {
        SCOPED_TRACE(closeAll);
        auto codec = std::make_unique<ObservingColdPageCodec>();
        auto* observer = codec.get();
        auto manager = std::make_shared<KvCacheManager>(sparseConfig(), nullptr, std::move(codec));
        auto const apiLock = manager->lockExclusive();
        auto page = seedPrefix(*manager, kHotLevel);
        auto decoder = manager->createKvCache({}, tokens());
        auto prefill = manager->createKvCache({}, tokens());
        auto closeCaches = FuncGuard(
            [&]()
            {
                decoder->close();
                prefill->close();
            });
        ASSERT_TRUE(decoder->resume(stream()));
        ASSERT_TRUE(prefill->resume(stream()));
        ASSERT_TRUE(decoder->enterDecode());
        decoder->close();
        if (closeAll)
        {
            prefill->close();
            EXPECT_EQ(observer->encodeCalls, 0);
            prefill = manager->createKvCache({}, tokens());
            ASSERT_TRUE(prefill->resume(stream()));
        }
        ASSERT_TRUE(prefill->enterDecode());
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(observer->encodedPages, 1);
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, HostOomDoesNotAdmitDecodeAndCanRetry)
{
    for (int admission : {0, 1, 2})
    {
        SCOPED_TRACE(admission);
        bool const suspended = admission != 0;
        bool const firstResume = admission == 2;
        int const history = firstResume ? 4 : 8;
        auto manager = std::make_shared<KvCacheManager>(sparseConfig());
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        auto prefix = firstResume ? seedPrefix(*manager, kHotLevel) : nullptr;
        auto cache = firstResume ? manager->createKvCache({}, tokens()) : manager->createKvCache();
        auto closeCache = FuncGuard([&]() { cache->close(); });
        if (!firstResume)
        {
            ASSERT_TRUE(cache->resume(stream()));
            ASSERT_TRUE(cache->resize(history, history));
            if (suspended)
                cache->suspend();
        }
        manager->getAndResetIterationSuspendResumeStats();
        auto blockers = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{firstResume ? 2 : 1});
        auto releaseBlockers = FuncGuard(
            [&]()
            {
                for (auto& slot : blockers[LifeCycleId{0}])
                    storage.releaseSlot(LifeCycleId{0}, kSparseHistoryLevel, std::move(slot));
            });
        EXPECT_FALSE(suspended ? cache->resume(stream(), true) : cache->enterDecode());
        EXPECT_EQ(cache->isActive(), !suspended);
        EXPECT_FALSE(cache->isDecoding());
        EXPECT_EQ(cache->historyLength(), history);
        EXPECT_EQ(cache->capacity(), history);
        EXPECT_EQ(cache->getPageStorageSnapshot(LifeCycleId{0}).eligibleHistoryBlocks(), 0);
        EXPECT_EQ(manager->getAndResetIterationSuspendResumeStats(), (std::pair<int64_t, int64_t>{0, 0}));
        for (int ord = 0; ord < history / 4; ++ord)
        {
            auto const page = pageAt(*cache, ord);
            EXPECT_EQ(page->cacheLevel, kHotLevel);
            EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[ord],
                suspended ? kBadPageIndex.value() : slotIdToPageIndexValue(page->slotId()));
        }
        releaseBlockers.run();
        ASSERT_TRUE(suspended ? cache->resume(std::nullopt, true) : cache->enterDecode());
        EXPECT_TRUE(cache->isDecoding());
        for (int ord = 0; ord < history / 4; ++ord)
            EXPECT_EQ(pageAt(*cache, ord)->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(manager->getAndResetIterationSuspendResumeStats().second, admission == 1 ? 1 : 0);
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, HistoryOomRollsBackShortcutGrowthAndShrink)
{
    for (auto const [oldCapacity, newCapacity] : {std::pair{8, 8}, std::pair{8, 12}, std::pair{12, 8}})
    {
        SCOPED_TRACE(newCapacity);
        SCOPED_TRACE(oldCapacity);
        auto manager = std::make_shared<KvCacheManager>(sparseConfig());
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        auto cache = manager->createKvCache();
        auto closeCache = FuncGuard([&]() { cache->close(); });
        ASSERT_TRUE(cache->resume(stream()));
        ASSERT_TRUE(cache->resize(8, 4));
        ASSERT_TRUE(cache->enterDecode());
        ASSERT_TRUE(cache->resize(oldCapacity));
        auto const page = pageAt(*cache, 1);
        auto const slotId = page->slotId();
        auto const version = cache->pageStorageVersion();
        auto const gpuFree = storage.getStatistics(kHotLevel).free;
        auto blockers = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{1});
        auto releaseBlockers = FuncGuard(
            [&]()
            {
                for (auto& slot : blockers[LifeCycleId{0}])
                    storage.releaseSlot(LifeCycleId{0}, kSparseHistoryLevel, std::move(slot));
            });
        EXPECT_FALSE(cache->resize(newCapacity, 8));
        EXPECT_EQ(cache->capacity(), oldCapacity);
        EXPECT_EQ(cache->historyLength(), 4);
        EXPECT_TRUE(cache->isDecoding());
        EXPECT_EQ(cache->pageStorageVersion(), version);
        EXPECT_EQ(pageAt(*cache, 1), page);
        EXPECT_EQ(page->slotId(), slotId);
        EXPECT_EQ(page->cacheLevel, kHotLevel);
        EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[1], slotIdToPageIndexValue(slotId));
        EXPECT_EQ(storage.getStatistics(kHotLevel).free, gpuFree);
        if (oldCapacity == 12)
            EXPECT_EQ(pageAt(*cache, 2)->status(), PageStatus::LOCKED);
        storage.releaseSlot(LifeCycleId{0}, kSparseHistoryLevel, std::move(blockers[LifeCycleId{0}].back()));
        blockers[LifeCycleId{0}].clear();
        ASSERT_TRUE(cache->resize(newCapacity, 8));
        EXPECT_EQ(cache->capacity(), newCapacity);
        EXPECT_EQ(cache->historyLength(), 8);
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
        EXPECT_GT(cache->pageStorageVersion(), version);
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, CodecRejectionDoesNotPublishEntryOrHistoryUpdate)
{
    for (bool entry : {false, true})
    {
        SCOPED_TRACE(entry);
        auto codec = std::make_unique<ObservingColdPageCodec>();
        auto* observer = codec.get();
        auto manager = std::make_shared<KvCacheManager>(sparseConfig(), nullptr, std::move(codec));
        auto const apiLock = manager->lockExclusive();
        auto cache = manager->createKvCache();
        auto closeCache = FuncGuard([&]() { cache->close(); });
        ASSERT_TRUE(cache->resume(stream()));
        int const history = entry ? 8 : 4;
        ASSERT_TRUE(cache->resize(8, history));
        if (!entry)
            ASSERT_TRUE(cache->enterDecode());
        auto const version = cache->pageStorageVersion();
        auto const before = cache->getPageStorageSnapshot(LifeCycleId{0});
        auto const page = pageAt(*cache, 1);
        auto const gpuSlot = page->slotId();
        observer->rejectEncodeCall = observer->encodeCalls + 1;
        EXPECT_THROW(entry ? cache->enterDecode() : cache->resize(12, 8), TllmException);
        EXPECT_EQ(cache->isDecoding(), !entry);
        EXPECT_EQ(cache->capacity(), 8);
        EXPECT_EQ(cache->historyLength(), history);
        EXPECT_GT(cache->pageStorageVersion(), version);
        auto const after = cache->getPageStorageSnapshot(LifeCycleId{0});
        EXPECT_EQ(after.basePageIndices(), before.basePageIndices());
        EXPECT_EQ(after.cacheLevels(), before.cacheLevels());
        EXPECT_EQ(after.eligibleHistoryBlocks(), before.eligibleHistoryBlocks());
        EXPECT_EQ(page->cacheLevel, kHotLevel);
        EXPECT_EQ(page->slotId(), gpuSlot);
        EXPECT_EQ(cache->getBasePageIndices(LifeCycleId{0})[1], slotIdToPageIndexValue(gpuSlot));
        observer->rejectEncodeCall = 0;
        ASSERT_TRUE(entry ? cache->enterDecode() : cache->resize(12, 8));
        EXPECT_TRUE(cache->isDecoding());
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    }
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, RebaseOffloadsOlderGpuHistoryAtUnchangedWatermark)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kHotLevel);
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 4));
    ASSERT_TRUE(cache->enterDecode());
    auto const version = cache->pageStorageVersion();
    cache->commit(tokens());
    EXPECT_EQ(pageAt(*cache), page);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(cache->historyLength(), 4);
    EXPECT_GT(cache->pageStorageVersion(), version);
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, PrefillRebasePromotesSharedHostPage)
{
    auto manager = std::make_shared<KvCacheManager>(sparseConfig());
    auto const apiLock = manager->lockExclusive();
    auto page = seedPrefix(*manager, kSparseHistoryLevel);
    auto decoder = manager->createKvCache({}, tokens());
    auto prefill = manager->createKvCache();
    auto closeCaches = FuncGuard(
        [&]()
        {
            decoder->close();
            prefill->close();
        });
    ASSERT_TRUE(decoder->resume(stream(), true));
    ASSERT_TRUE(prefill->resume(stream()));
    ASSERT_TRUE(prefill->resize(4, 4));
    EXPECT_NO_THROW(prefill->commit(tokens()));
    EXPECT_FALSE(prefill->isDecoding());
    EXPECT_EQ(prefill->numCommittedTokens(), 4);
    EXPECT_EQ(pageAt(*prefill), page);
    EXPECT_EQ(page->cacheLevel, kHotLevel);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    EXPECT_EQ(prefill->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(page->slotId()));
    EXPECT_EQ(decoder->getBasePageIndices(LifeCycleId{0})[0], slotIdToPageIndexValue(page->slotId()));
    EXPECT_EQ(manager->storage().getStatistics(kHotLevel).free, manager->storage().getStatistics(kHotLevel).total - 1);
    EXPECT_EQ(manager->storage().getStatistics(kSparseHistoryLevel).free,
        manager->storage().getStatistics(kSparseHistoryLevel).total);
    EXPECT_NO_THROW(prefill->close());
    ASSERT_TRUE(decoder->enterDecode());
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
}

TEST_F(KvCacheManagerV2DecodeOffloadTest, RebaseOomPreservesMissingLifecyclePagesAndCanRetryCommit)
{
    auto config = makeSplitColdGroupingConfig();
    for (auto& layer : config.layers)
        std::get<AttentionLayerConfig>(layer).buffers.front().isSparse = true;
    auto manager = std::make_shared<KvCacheManager>(std::move(config));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    auto existing = seedPrefix(*manager, kHotLevel, LifeCycleId{1});
    auto cache = manager->createKvCache();
    auto closeCache = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(stream()));
    ASSERT_TRUE(cache->resize(4, 4));
    ASSERT_TRUE(cache->enterDecode());
    auto first = pageAt(*cache, 0, LifeCycleId{0});
    auto second = pageAt(*cache, 0, LifeCycleId{1});
    auto const version = cache->pageStorageVersion();
    auto const before = cache->getPageStorageSnapshot(LifeCycleId{0});
    ASSERT_TRUE(cache->acknowledgePageStorage(version));
    auto const pool = storage.getPoolGroupIndex(kSparseHistoryLevel, LifeCycleId{1});
    auto const freeHost = manager->getStorageStatistics(kSparseHistoryLevel).at(pool).free;
    auto blockers = storage.newSlots(kSparseHistoryLevel, TypedVec<LifeCycleId, SlotCount>{0, freeHost});
    auto releaseBlockers = FuncGuard(
        [&]()
        {
            for (auto& slot : blockers[LifeCycleId{1}])
                storage.releaseSlot(LifeCycleId{1}, kSparseHistoryLevel, std::move(slot));
        });
    EXPECT_THROW(cache->commit(tokens()), OutOfPagesError);
    EXPECT_EQ(cache->numCommittedTokens(), 0);
    EXPECT_EQ(cache->numCommittedBlocks(), 0);
    EXPECT_EQ(cache->historyLength(), 4);
    EXPECT_GT(cache->pageStorageVersion(), version);
    EXPECT_TRUE(cache->pageStorageDirty());
    EXPECT_FALSE(cache->acknowledgePageStorage(version));
    auto const after = cache->getPageStorageSnapshot(LifeCycleId{0});
    EXPECT_EQ(after.basePageIndices(), before.basePageIndices());
    EXPECT_EQ(after.cacheLevels(), before.cacheLevels());
    EXPECT_EQ(after.eligibleHistoryBlocks(), before.eligibleHistoryBlocks());
    EXPECT_EQ(pageAt(*cache, 0, LifeCycleId{0}), first);
    EXPECT_EQ(pageAt(*cache, 0, LifeCycleId{1}), second);
    for (auto const& page : {first, second})
    {
        EXPECT_FALSE(page->isCommitted());
        EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
        EXPECT_EQ(page->status(), PageStatus::LOCKED);
        EXPECT_EQ(cache->getBasePageIndices(page->lifeCycle)[0], slotIdToPageIndexValue(page->slotId()));
    }
    EXPECT_EQ(existing->cacheLevel, kHotLevel);
    releaseBlockers.run();
    cache->commit(tokens());
    EXPECT_EQ(cache->numCommittedTokens(), 4);
    EXPECT_TRUE(pageAt(*cache, 0, LifeCycleId{0})->isCommitted());
    EXPECT_EQ(pageAt(*cache, 0, LifeCycleId{1}), existing);
    EXPECT_EQ(existing->cacheLevel, kSparseHistoryLevel);
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
    ASSERT_TRUE(cache->prefetch(kSparseHistoryLevel));
    ASSERT_TRUE(cache->resume(stream(), true));
    EXPECT_EQ(pageAt(*cache), page);
    EXPECT_EQ(page->cacheLevel, kSparseHistoryLevel);
    EXPECT_EQ(page->slotId(), hostSlot);
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    EXPECT_FALSE(page->scheduledForEviction());
    EXPECT_FALSE(storage.isEvictable(*page));
    EXPECT_THROW(storage.batchedMigrate(kHotLevel, {page}, {}), LogicError);

    auto second = manager->createKvCache({}, tokens());
    auto closeSecond = FuncGuard([&]() { second->close(); });
    ASSERT_TRUE(second->prefetch(kSparseHistoryLevel));
    ASSERT_TRUE(second->resume(stream(), true));
    EXPECT_EQ(pageAt(*second), page);
    cache->suspend();
    EXPECT_EQ(page->status(), PageStatus::LOCKED);
    second->close();
    EXPECT_EQ(page->status(), PageStatus::HELD);
    ASSERT_TRUE(cache->prefetch(kHotLevel));
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
    ASSERT_TRUE(full->resume(stream(), true));
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
    ASSERT_TRUE(cache->resume(stream(), true));
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
        ASSERT_TRUE(cache->resume(std::nullopt, true));
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
    ASSERT_TRUE(cache->resume(stream(), true));
    ASSERT_TRUE(cache->resize(8, 4));
    SlotId const hostSlot = page->slotId();
    auto const gpuFree = manager->getStorageStatistics(kHotLevel).at(PoolGroupIndex{0}).free;

    // Sparse SWA history stays pinned. A failed request for three GPU pages
    // must preserve the host lock and the previous eligible-history count.
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
    ASSERT_TRUE(cache->resume(stream(), true));
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
    ASSERT_TRUE(first->resume(stream(), true));
    auto second = manager->createKvCache();
    auto closeSecond = FuncGuard([&]() { second->close(); });
    ASSERT_TRUE(second->resume(stream()));
    ASSERT_TRUE(second->resize(4, 4));
    ASSERT_EQ(pageAt(*second)->cacheLevel, kHotLevel);
    ASSERT_TRUE(second->enterDecode());
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
    ASSERT_TRUE(cache->resume(stream(), true));
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
