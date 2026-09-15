/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kvCacheManagerV2TestUtils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/hostPageCopy.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace
{
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

KVCacheManagerConfig hostConfig()
{
    auto config = test::makeConfig();
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    config.maxUtilForResume = 1.0f;
    auto sparse = std::get<AttentionLayerConfig>(config.layers.front());
    sparse.residencyGroup = 1;
    sparse.buffers.push_back(BufferConfig{"value", 2048, std::nullopt});
    sparse.buffers.push_back(BufferConfig{"scales", 64, std::nullopt});
    sparse.layerId = 1;
    config.layers.emplace_back(sparse);
    sparse.layerId = 2;
    config.layers.emplace_back(sparse);
    return config;
}

// Gate GPU work without a timing assumption. Declared after the fixture's resources
// so fatal assertions still release the callback before those resources are destroyed.
class StreamGate
{
public:
    explicit StreamGate(cudaStream_t stream)
        : mStream(stream)
    {
        if (cudaLaunchHostFunc(
                stream,
                [](void* data)
                {
                    auto& gate = *static_cast<StreamGate*>(data);
                    while (!gate.mRelease.load(std::memory_order_acquire))
                        std::this_thread::yield();
                },
                this)
            != cudaSuccess)
            throw std::runtime_error("Could not gate CUDA stream");
    }

    ~StreamGate()
    {
        release();
        cudaStreamSynchronize(mStream);
    }

    void release()
    {
        mRelease.store(true, std::memory_order_release);
    }

private:
    cudaStream_t mStream;
    std::atomic<bool> mRelease{false};
};

class KvCacheManagerV2HostCopyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), cudaSuccess);
        reset(hostConfig());
    }

    void TearDown() override
    {
        mCache->close();
        mCache.reset();
        mManager->clearReusableBlocks();
        mManager.reset();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    void reset(KVCacheManagerConfig config, std::unique_ptr<IKvCacheColdPageCodec> codec = {}, int history = 16)
    {
        if (mCache)
            mCache->close();
        mCache.reset();
        mManager.reset();
        mManager = std::make_shared<KvCacheManager>(config, nullptr, std::move(codec));
        mCache = mManager->createKvCache();
        ASSERT_TRUE(mCache->resume(mStream));
        ASSERT_TRUE(mCache->resize(20, history));
    }

    SharedPtr<Page> page(BlockOrdinal ordinal = kPage, LifeCycleId group = kSparse)
    {
        return blockPageGetPage(mCache->blocks()[ordinal].pages[kDefaultBeamIndex][group]);
    }

    void fill(int value = 0x40)
    {
        auto& storage = mManager->storage();
        auto const group = storage.getPoolGroupIndex(kHotLevel, kSparse);
        auto const sizes = storage.slotSize(group);
        for (PoolIndex pool{0}; pool < sizes.size(); ++pool)
        {
            auto address = std::get<MemAddress>(storage.slotAddress(kHotLevel, group, page()->slotId(), pool));
            ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), value + pool.value(), sizes[pool], mStream),
                cudaSuccess);
        }
    }

    void checkHost(int value = 0x40)
    {
        auto read = mCache->acquireHostCopy(kSparse, kPage);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_TRUE(read->ready());
        auto const sizes = mManager->storage().slotSize(mManager->storage().getPoolGroupIndex(kSparse));
        auto const* bytes = reinterpret_cast<uint8_t const*>(read->copy().address());
        size_t offset = 0;
        for (PoolIndex pool{0}; pool < sizes.size(); ++pool)
        {
            EXPECT_TRUE(std::all_of(
                bytes + offset, bytes + offset + sizes[pool], [=](uint8_t x) { return x == value + pool.value(); }));
            offset += sizes[pool];
        }
        EXPECT_EQ(offset, read->copy().pageBytes());
    }

    void checkGpu(int value = 0x40)
    {
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        auto& storage = mManager->storage();
        auto group = storage.getPoolGroupIndex(kSparse);
        auto sizes = storage.slotSize(group);
        for (PoolIndex pool{0}; pool < sizes.size(); ++pool)
        {
            std::vector<uint8_t> bytes(sizes[pool]);
            auto address = std::get<MemAddress>(storage.slotAddress(kHotLevel, group, page()->slotId(), pool));
            ASSERT_EQ(
                cudaMemcpy(bytes.data(), reinterpret_cast<void const*>(address), bytes.size(), cudaMemcpyDeviceToHost),
                cudaSuccess);
            EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [=](uint8_t x) { return x == value + pool.value(); }));
        }
    }

    void holdHistory()
    {
        mCache->setResidencyWindow(kSparse, 5, 4);
    }

    static constexpr LifeCycleId kSparse{1};
    static constexpr BlockOrdinal kPage{1};
    cudaStream_t mStream{};
    std::shared_ptr<KvCacheManager> mManager;
    std::shared_ptr<KvCache> mCache;
};

TEST_F(KvCacheManagerV2HostCopyTest, RoundTripRetainsSameHostSlotForCoalescedLayersAndSeparateBuffers)
{
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    checkHost();
    auto const slot = page()->hostCopy()->slotId();
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    EXPECT_EQ(page()->cacheLevel, CacheLevel{1});
    EXPECT_EQ(page()->slotId(), slot);
    EXPECT_EQ(page(kPage, LifeCycleId{0})->cacheLevel, kHotLevel);
    EXPECT_EQ(mCache->getBasePageIndices(kSparse)[kPage.value()], kBadPageIndex.value());
    mCache->setResidencyWindow(kSparse, std::nullopt);
    checkGpu();
    EXPECT_EQ(page()->hostCopy()->slotId(), slot);
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    EXPECT_EQ(page()->slotId(), slot);
    checkHost();
}

TEST_F(KvCacheManagerV2HostCopyTest, BackupCountsTrafficOnceAndOffloadDoesNotCountAnotherCopy)
{
    fill();
    mManager->getAndResetIterationStats();
    mCache->backupToHost(kSparse, kPage, 4);
    auto const bytes = page()->hostCopy()->pageBytes();
    mCache->backupToHost(kSparse, kPage, 4);
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    auto stats = mManager->getAndResetIterationStats();
    EXPECT_EQ(stats.at(kSparse).iterOffloadBlocks, 1);
    EXPECT_EQ(stats.at(kSparse).iterOffloadBytes, bytes);
    mCache->setResidencyWindow(kSparse, std::nullopt);
    stats = mManager->getAndResetIterationStats();
    EXPECT_EQ(stats.at(kSparse).iterOnboardBlocks, 1);
    EXPECT_EQ(stats.at(kSparse).iterOnboardBytes, bytes);
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    EXPECT_TRUE(mManager->getAndResetIterationStats().empty());
}

TEST_F(KvCacheManagerV2HostCopyTest, PartialPrefixInvalidationAndRefreshKeepAddress)
{
    reset(hostConfig(), {}, 6);
    fill();
    EXPECT_THROW(mCache->backupToHost(kSparse, kPage, 3), std::invalid_argument);
    mCache->backupToHost(kSparse, kPage, 2);
    auto read = mCache->acquireHostCopy(kSparse, kPage);
    EXPECT_EQ(read->copy().validTokens(), 2);
    auto const address = read->copy().address();
    EXPECT_THROW(mCache->invalidateHostCopy(kSparse, kPage), LogicError);
    read->close();
    mCache->invalidateHostCopy(kSparse, kPage);
    EXPECT_FALSE(page()->hostCopy()->ready());
    EXPECT_THROW(mCache->acquireHostCopy(kSparse, kPage), LogicError);
    fill(0x70);
    mCache->setHistoryLength(8);
    mCache->backupToHost(kSparse, kPage, 4);
    EXPECT_EQ(page()->hostCopy()->address(), address);
    checkHost(0x70);
}

TEST_F(KvCacheManagerV2HostCopyTest, PendingBackupProtectsSourceAndDoesNotPublishReady)
{
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    StreamGate gate(mStream);
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    EXPECT_FALSE(page()->hostCopy()->ready());
    auto reader = mCache->acquireHostCopy(kSparse, kPage);
    EXPECT_EQ(reader->completedTokens(), 0);
    reader->close();
    auto const gpuSlot = page()->slotId();
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    auto& storage = mManager->storage();
    auto const group = storage.getPoolGroupIndex(kSparse);
    auto slots = storage.newSlotsForPoolGroup(kHotLevel, group, storage.getStatistics(kHotLevel, group).free);
    bool foundSource = false;
    for (auto& slot : slots)
    {
        if (slot.slotId() == gpuSlot)
        {
            foundSource = true;
            EXPECT_FALSE(slot.readyEvent.queryComplete());
        }
    }
    EXPECT_TRUE(foundSource);
    for (auto& slot : slots)
        storage.releaseSlot(kSparse, kHotLevel, std::move(slot));
    gate.release();
    checkHost();
}

TEST_F(KvCacheManagerV2HostCopyTest, ReaderSurvivesRequestCloseAndBlocksShutdown)
{
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    auto read = mCache->acquireHostCopy(kSparse, kPage);
    auto const address = read->copy().address();
    mCache->close();
    EXPECT_ANY_THROW(mManager->shutdown());
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_EQ(*reinterpret_cast<uint8_t const*>(address), 0x40);
    read->close();
    read->close();
    EXPECT_THROW(read->copy(), LogicError);
    EXPECT_NO_THROW(mManager->shutdown());
}

TEST_F(KvCacheManagerV2HostCopyTest, ReaderCompletionProtectsHostSlotReuse)
{
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    auto read = mCache->acquireHostCopy(kSparse, kPage);
    auto const hostSlot = read->copy().slotId();
    auto const group = read->copy().poolGroup();
    StreamGate gate(mStream);
    read->close();
    mCache->close();
    auto& storage = mManager->storage();
    auto slots = storage.newSlotsForPoolGroup(CacheLevel{1}, group, storage.getStatistics(CacheLevel{1}, group).free);
    bool foundHost = false;
    for (auto& slot : slots)
    {
        if (slot.slotId() == hostSlot)
        {
            foundHost = true;
            EXPECT_FALSE(slot.readyEvent.queryComplete());
        }
    }
    EXPECT_TRUE(foundHost);
    for (auto& slot : slots)
        storage.releaseSlot(kSparse, CacheLevel{1}, std::move(slot));
}

TEST_F(KvCacheManagerV2HostCopyTest, CommitAndPrefixSharingPreserveHostCopy)
{
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    auto const address = page()->hostCopy()->address();
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 16; ++i)
        tokens.emplace_back(i);
    mCache->commit(TokenSpan{tokens.data(), 16});
    EXPECT_TRUE(page()->isCommitted());
    EXPECT_EQ(page()->hostCopy()->address(), address);
    auto other = mManager->createKvCache({}, TokenSpan{tokens.data(), 16});
    ASSERT_TRUE(other->resume(mStream));
    auto read = other->acquireHostCopy(kSparse, kPage);
    EXPECT_EQ(read->copy().address(), address);
    mCache->close();
    read->close();
    other->close();
}

TEST_F(KvCacheManagerV2HostCopyTest, InvalidArgumentsAndLockedOffloadLeaveDataIntact)
{
    fill();
    EXPECT_THROW(mCache->backupToHost(kSparse, kPage, 0), std::invalid_argument);
    EXPECT_ANY_THROW(mCache->backupToHost(kSparse, kPage, 4, kHotLevel));
    EXPECT_THROW(mCache->backupToHost(kSparse, BlockOrdinal{10}, 4), std::out_of_range);
    EXPECT_THROW(mCache->backupToHost(LifeCycleId{10}, kPage, 4), std::invalid_argument);
    EXPECT_THROW(mCache->acquireHostCopy(kSparse, kPage), LogicError);
    mCache->backupToHost(kSparse, kPage, 3);
    EXPECT_THROW(mCache->offloadToHost(kSparse, kPage), LogicError);
    holdHistory();
    EXPECT_THROW(mCache->offloadToHost(kSparse, kPage), LogicError);
    mCache->backupToHost(kSparse, kPage, 4);
    mCache->offloadToHost(kSparse, kPage);
    checkHost();
}

TEST_F(KvCacheManagerV2HostCopyTest, PinnedHostPoolRejectsResizeUntilCleanup)
{
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    auto& storage = mManager->storage();
    auto const address = page()->hostCopy()->address();
    EXPECT_THROW(
        storage.adjustCacheLevel(CacheLevel{1}, 8 << 20, storage.getRatioList(CacheLevel{1}), nullptr), LogicError);
    EXPECT_EQ(page()->hostCopy()->address(), address);
    checkHost();
    mCache->close();
    EXPECT_NO_THROW(storage.adjustCacheLevel(CacheLevel{1}, 8 << 20, storage.getRatioList(CacheLevel{1}), nullptr));
}

TEST_F(KvCacheManagerV2HostCopyTest, ExhaustionDoesNotDropHeldCopiesOrReleaseGpuSource)
{
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    auto& storage = mManager->storage();
    auto const group = page()->hostCopy()->poolGroup();
    auto slots = storage.newSlotsForPoolGroup(CacheLevel{1}, group, storage.getStatistics(CacheLevel{1}, group).free);
    auto release = FuncGuard(
        [&]()
        {
            for (auto& slot : slots)
                storage.releaseSlot(kSparse, CacheLevel{1}, std::move(slot));
        });
    EXPECT_THROW(mCache->backupToHost(kSparse, BlockOrdinal{2}, 4), OutOfPagesError);
    EXPECT_EQ(page(BlockOrdinal{2})->cacheLevel, kHotLevel);
    EXPECT_FALSE(page(BlockOrdinal{2})->hostCopy());
    mCache->setResidencyWindow(kSparse, std::nullopt);
    checkGpu();
    holdHistory();
    EXPECT_NO_THROW(mCache->offloadToHost(kSparse, kPage));
    checkHost();
}

TEST_F(KvCacheManagerV2HostCopyTest, LastRequestReleasesExtraHostCopyOfCachedGpuPage)
{
    fill();
    auto& storage = mManager->storage();
    auto const group = storage.getPoolGroupIndex(CacheLevel{1}, kSparse);
    auto const freeBefore = storage.getStatistics(CacheLevel{1}, group).free;
    mCache->backupToHost(kSparse, kPage, 4);
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 16; ++i)
        tokens.emplace_back(i);
    mCache->commit(TokenSpan{tokens.data(), 16});
    auto cachedPage = page();
    ASSERT_TRUE(cachedPage->hostCopy());
    mCache->close();
    EXPECT_FALSE(cachedPage->hostCopy());
    EXPECT_EQ(cachedPage->cacheLevel, kHotLevel);
    EXPECT_EQ(storage.getStatistics(CacheLevel{1}, group).free, freeBefore);
}

TEST_F(KvCacheManagerV2HostCopyTest, ReaderKeepsOffloadedHostSlotUnavailableAfterRequestClose)
{
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 16; ++i)
        tokens.emplace_back(i);
    mCache->commit(TokenSpan{tokens.data(), 16});
    auto reader = mCache->acquireHostCopy(kSparse, kPage);
    auto const group = reader->copy().poolGroup();
    mCache->close();
    auto& storage = mManager->storage();
    EXPECT_EQ(storage.getStatistics(CacheLevel{1}, group).evictable, 0);
    auto const free = storage.getStatistics(CacheLevel{1}, group).free;
    auto slots = storage.newSlotsForPoolGroup(CacheLevel{1}, group, free);
    auto release = FuncGuard(
        [&]()
        {
            for (auto& slot : slots)
                storage.releaseSlot(kSparse, CacheLevel{1}, std::move(slot));
        });
    EXPECT_THROW(storage.newSlotsForPoolGroup(CacheLevel{1}, group, 1), OutOfPagesError);
    reader->close();
    EXPECT_EQ(storage.getStatistics(CacheLevel{1}, group).free, 1);
}

TEST_F(KvCacheManagerV2HostCopyTest, InactiveHostPageReleasesHostSlotWhenMovingToDisk)
{
    auto config = hostConfig();
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    reset(config);
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 16; ++i)
        tokens.emplace_back(i);
    mCache->commit(TokenSpan{tokens.data(), 16});
    auto cachedPage = page();
    mCache->close();
    auto& storage = mManager->storage();
    auto const group = storage.getPoolGroupIndex(CacheLevel{1}, kSparse);
    auto const freeBefore = storage.getStatistics(CacheLevel{1}, group).free;
    TypedVec<PoolGroupIndex, SlotCount> counts(storage.numPoolGroups(CacheLevel{1}), 0);
    counts[group] = 1;
    storage.forceEvict(CacheLevel{1}, counts);
    EXPECT_EQ(cachedPage->cacheLevel, CacheLevel{2});
    EXPECT_FALSE(cachedPage->hostCopy());
    EXPECT_EQ(storage.getStatistics(CacheLevel{1}, group).free, freeBefore + 1);
}

TEST_F(KvCacheManagerV2HostCopyTest, RestoresDiskPageBeforeMakingHostCopy)
{
    auto config = hostConfig();
    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    reset(config);
    fill();
    holdHistory();
    auto& storage = mManager->storage();
    TypedVec<PoolGroupIndex, SlotCount> gpuCounts(storage.numPoolGroups(kHotLevel), 0);
    gpuCounts[storage.getPoolGroupIndex(kSparse)] = 2;
    storage.forceEvict(kHotLevel, gpuCounts);
    EXPECT_EQ(page()->cacheLevel, CacheLevel{1});
    TypedVec<PoolGroupIndex, SlotCount> hostCounts(storage.numPoolGroups(CacheLevel{1}), 0);
    hostCounts[storage.getPoolGroupIndex(CacheLevel{1}, kSparse)] = 2;
    storage.forceEvict(CacheLevel{1}, hostCounts);
    ASSERT_EQ(page()->cacheLevel, CacheLevel{2});
    mCache->backupToHost(kSparse, kPage, 4);
    mCache->offloadToHost(kSparse, kPage);
    EXPECT_EQ(page()->cacheLevel, CacheLevel{1});
    checkHost();
}
} // namespace
