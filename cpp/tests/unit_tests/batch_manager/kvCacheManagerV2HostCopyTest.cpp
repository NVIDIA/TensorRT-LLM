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
#include <chrono>
#include <cstdint>
#include <future>
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
        // Lazy CUDA stream-pool creation can wait for host callbacks. Warm it
        // before blocking this stream so event merging remains asynchronous.
        {
            CachedCudaStream const warmup;
        }
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

    HostSourceView enableHostSources(int history = 16, int maxRequests = 4, int maxPages = 8)
    {
        mCache->close();
        mCache.reset();
        mManager->reserveHostSourceTable(maxRequests, maxPages);
        mCache = mManager->createKvCache({}, {}, RequestIdType{41});
        EXPECT_TRUE(mCache->resume(mStream));
        EXPECT_TRUE(mCache->resize(std::max(8, history), history));
        return mManager->hostSourceView();
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

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTablePublishesOnlyCompletedPrefixes)
{
    auto const view = enableHostSources(6);
    auto const index = view.pageIndex(0, 0, kSparse.value(), kPage.value());
    ASSERT_EQ(view.requestIds[0], 41);
    ASSERT_EQ(view.requestValid[0], 1);
    EXPECT_EQ(view.slotIds[index], -1);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    StreamGate gate(mStream);
    fill();
    mCache->backupToHost(kSparse, kPage, 2);
    mManager->refreshHostSourceTable();
    EXPECT_EQ(view.slotIds[index], -1);
    EXPECT_EQ(view.hostLevels[index], -1);
    EXPECT_EQ(view.completedTokens[index], 0);
    gate.release();
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    mManager->refreshHostSourceTable();
    EXPECT_EQ(view.slotIds[index], page()->hostCopy()->slotId().value());
    EXPECT_EQ(view.completedTokens[index], 2);
    EXPECT_EQ(view.hostLevels[index], 1);
    auto const slot = view.slotIds[index];
    auto const* metadata = view.poolMetadata + (view.numLifeCycles + kSparse.value()) * 4;
    CUdeviceptr deviceAddress = 0;
    ASSERT_EQ(cuMemHostGetDevicePointer(&deviceAddress, reinterpret_cast<void*>(page()->hostCopy()->address()), 0),
        CUDA_SUCCESS);
    EXPECT_EQ(metadata[0] + slot * metadata[1], deviceAddress);
    EXPECT_EQ(metadata[2], page()->hostCopy()->poolBytes());
    EXPECT_EQ(metadata[3], page()->hostCopy()->poolGroup().value());
    mCache->invalidateHostCopy(kSparse, kPage);
    EXPECT_EQ(view.slotIds[index], -1);
    EXPECT_EQ(view.completedTokens[index], 0);
    EXPECT_TRUE(mCache->resize(8, 8));
    fill(0x70);
    mCache->backupToHost(kSparse, kPage, 4);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    auto read = mManager->acquireHostSources(mStream);
    EXPECT_EQ(view.slotIds[index], slot);
    EXPECT_EQ(view.completedTokens[index], 4);
    EXPECT_THROW(mCache->invalidateHostCopy(kSparse, kPage), LogicError);
    EXPECT_THROW(mCache->close(), LogicError);
    read->close();
    EXPECT_NO_THROW(mCache->invalidateHostCopy(kSparse, kPage));
}

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTableTracksPendingOffloadAndSuspendResume)
{
    auto const view = enableHostSources();
    auto const index = view.pageIndex(0, 0, kSparse.value(), kPage.value());
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    StreamGate gate(mStream);
    fill();
    mCache->backupToHost(kSparse, kPage, 4);
    holdHistory();
    mCache->offloadToHost(kSparse, kPage);
    EXPECT_EQ(page()->cacheLevel, CacheLevel{1});
    EXPECT_EQ(view.slotIds[index], -1);
    gate.release();
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    mCache->suspend();
    auto const slot = view.slotIds[index];
    ASSERT_GE(slot, 0);
    EXPECT_EQ(view.completedTokens[index], 4);
    EXPECT_TRUE(mCache->resume(mStream));
    EXPECT_EQ(view.slotIds[index], slot);
    mCache->setResidencyWindow(kSparse, std::nullopt);
    EXPECT_EQ(page()->cacheLevel, kHotLevel);
    EXPECT_EQ(view.slotIds[index], slot);
    EXPECT_EQ(view.completedTokens[index], 4);
}

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTableFollowsCommitAndSharedPrefixOwnership)
{
    auto const view = enableHostSources();
    fill();
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    StreamGate gate(mStream);
    mCache->backupToHost(kSparse, kPage, 4);
    auto const slot = page()->hostCopy()->slotId().value();
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 16; ++i)
        tokens.emplace_back(i);
    mCache->commit(TokenSpan{tokens.data(), 16});
    auto const first = view.pageIndex(0, 0, kSparse.value(), kPage.value());
    EXPECT_EQ(view.slotIds[first], -1);
    auto other = mManager->createKvCache({}, TokenSpan{tokens.data(), 16}, RequestIdType{42});
    auto const cleanup = FuncGuard([&]() { other->close(); });
    EXPECT_EQ(mManager->hostSourceSlot(*other), 1);
    auto const second = view.pageIndex(1, 0, kSparse.value(), kPage.value());
    EXPECT_EQ(view.slotIds[second], -1);
    gate.release();
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    mManager->refreshHostSourceTable();
    EXPECT_EQ(view.slotIds[first], slot);
    EXPECT_EQ(view.slotIds[second], slot);
    EXPECT_TRUE(other->resume(mStream));
    mCache->close();
    EXPECT_EQ(view.requestValid[0], 0);
    EXPECT_EQ(view.slotIds[first], -1);
    EXPECT_EQ(view.slotIds[second], slot);
    auto cachedPage = blockPageGetPage(other->blocks()[kPage].pages[kDefaultBeamIndex][kSparse]);
    other->close();
    EXPECT_EQ(view.requestValid[1], 0);
    EXPECT_EQ(view.slotIds[second], -1);
    EXPECT_FALSE(cachedPage->hostCopy());
}

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTableDoesNotCarryPartialSourceIntoWritableClone)
{
    auto const view = enableHostSources(6);
    fill();
    mCache->backupToHost(kSparse, kPage, 2);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 6; ++i)
        tokens.emplace_back(i);
    mCache->commit(TokenSpan{tokens.data(), 6}, true);
    auto other = mManager->createKvCache({}, TokenSpan{tokens.data(), 6}, RequestIdType{42});
    auto const cleanup = FuncGuard([&]() { other->close(); });
    auto const first = view.pageIndex(0, 0, kSparse.value(), kPage.value());
    auto const second = view.pageIndex(1, 0, kSparse.value(), kPage.value());
    EXPECT_EQ(view.completedTokens[second], 2);
    EXPECT_EQ(view.slotIds[second], view.slotIds[first]);
    ASSERT_TRUE(other->resume(mStream));
    EXPECT_EQ(view.slotIds[second], -1);
    EXPECT_EQ(view.completedTokens[second], 0);
    EXPECT_EQ(view.completedTokens[first], 2);
}

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTableClearsRowsBeforeRequestSlotReuse)
{
    auto const view = enableHostSources(6, 1, 2);
    EXPECT_THROW(mManager->reserveHostSourceTable(1, 2), std::exception);
    EXPECT_THROW(mCache->resize(12, 6), std::out_of_range);
    EXPECT_EQ(mCache->capacity(), 8);
    EXPECT_THROW(mManager->createKvCache({}, {}, RequestIdType{41}), std::invalid_argument);
    EXPECT_THROW(mManager->createKvCache({}, {}, RequestIdType{42}), std::out_of_range);
    fill();
    mCache->backupToHost(kSparse, kPage, 2);
    auto pageRead = mCache->acquireHostCopy(kSparse, kPage);
    auto const generation = view.generations[0];
    auto const* slots = view.slotIds;
    mCache->close();
    EXPECT_EQ(view.requestValid[0], 0);
    EXPECT_EQ(mManager->hostSourceSlot(*mCache), -1);
    EXPECT_TRUE(std::all_of(
        view.completedTokens, view.completedTokens + 2 * view.numLifeCycles, [](int count) { return count == 0; }));
    EXPECT_ANY_THROW(mManager->shutdown()); // The existing per-page read still protects its source.
    auto other = mManager->createKvCache({}, {}, RequestIdType{41});
    auto const cleanup = FuncGuard([&]() { other->close(); });
    EXPECT_EQ(mManager->hostSourceSlot(*other), 0);
    EXPECT_EQ(view.requestIds[0], 41);
    EXPECT_EQ(view.generations[0], generation + 1);
    EXPECT_EQ(view.slotIds, slots);
    EXPECT_EQ(view.slotIds[view.pageIndex(0, 0, kSparse.value(), 1)], -1);
    pageRead->close();
}

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTablePoolMetadataTracksResize)
{
    auto const view = enableHostSources(6);
    auto const pool = (view.numLifeCycles + kSparse.value()) * 4;
    auto const initialBytes = view.poolMetadata[pool + 2];
    fill();
    mCache->backupToHost(kSparse, kPage, 2);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    mCache->suspend();
    EXPECT_FALSE(mManager->resize(CacheLevel{1}, 8 << 20));
    EXPECT_EQ(view.poolMetadata[pool + 2], initialBytes);
    mCache->close();
    ASSERT_TRUE(mManager->resize(CacheLevel{1}, 8 << 20));
    EXPECT_GT(view.poolMetadata[pool + 2], initialBytes);
    EXPECT_EQ(view.poolMetadata, mManager->hostSourceView().poolMetadata);
    EXPECT_EQ(view.requestValid[0], 0);
}

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTableTracksIdentityChangesWithoutStaleRows)
{
    auto const view = enableHostSources(6);
    fill();
    mCache->backupToHost(kSparse, kPage, 2);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    mManager->refreshHostSourceTable();
    auto const index = view.pageIndex(0, 0, kSparse.value(), 1);
    auto const slot = view.slotIds[index];
    auto const generation = view.generations[0];
    mCache->setId(RequestIdType{43});
    EXPECT_EQ(view.requestIds[0], 43);
    EXPECT_EQ(view.generations[0], generation + 1);
    EXPECT_EQ(view.slotIds[index], slot);
    mCache->setId(std::nullopt);
    EXPECT_EQ(mManager->hostSourceSlot(*mCache), -1);
    EXPECT_EQ(view.requestValid[0], 0);
    EXPECT_EQ(view.slotIds[index], -1);
    mCache->setId(RequestIdType{44});
    EXPECT_EQ(view.requestIds[0], 44);
    EXPECT_EQ(view.generations[0], generation + 2);
    EXPECT_EQ(view.slotIds[index], slot);
    auto other = mManager->createKvCache({}, {}, RequestIdType{45});
    auto const cleanup = FuncGuard([&]() { other->close(); });
    EXPECT_THROW(mCache->setId(RequestIdType{45}), std::invalid_argument);
    EXPECT_EQ(view.requestIds[0], 44);
    EXPECT_EQ(view.slotIds[index], slot);
}

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTableWaitsForClosedCrossStreamReaders)
{
    auto const view = enableHostSources(6);
    fill();
    mCache->backupToHost(kSparse, kPage, 2);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    cudaStream_t readStream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&readStream, cudaStreamNonBlocking), cudaSuccess);
    auto const freeStream = FuncGuard([&]() { cudaStreamDestroy(readStream); });
    auto read = mManager->acquireHostSources(readStream);
    auto anotherRead = mManager->acquireHostSources(mStream);
    StreamGate gate(readStream);
    read->close();
    EXPECT_THROW(mManager->refreshHostSourceTable(), LogicError);
    anotherRead->close();
    std::promise<void> started;
    auto startedFuture = started.get_future();
    auto update = std::async(std::launch::async,
        [&]()
        {
            if (cudaSetDevice(0) != cudaSuccess)
                throw std::runtime_error("Cannot set worker CUDA device");
            started.set_value();
            mCache->invalidateHostCopy(kSparse, kPage);
        });
    auto const release = FuncGuard(
        [&]()
        {
            gate.release();
            if (update.valid())
                update.wait();
        });
    startedFuture.wait();
    EXPECT_EQ(update.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    gate.release();
    EXPECT_NO_THROW(update.get());
    EXPECT_EQ(view.completedTokens[view.pageIndex(0, 0, kSparse.value(), 1)], 0);
}

TEST_F(KvCacheManagerV2HostCopyTest, HostSourceTableAddressesStayFixedAcrossGraphReplay)
{
    auto const view = enableHostSources(6);
    auto const index = view.pageIndex(0, 0, kSparse.value(), kPage.value());
    fill();
    mCache->backupToHost(kSparse, kPage, 2);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t* deviceCount = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&deviceCount), sizeof(int32_t)), cudaSuccess);
    auto const freeDevice = FuncGuard([&]() { cudaFree(deviceCount); });
    auto read = mManager->acquireHostSources(mStream);
    auto const source = reinterpret_cast<void const*>(view.gpuAddress(view.completedTokens + index));
    ASSERT_EQ(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(deviceCount, source, sizeof(int32_t), cudaMemcpyDefault, mStream), cudaSuccess);
    cudaGraph_t graph{};
    ASSERT_EQ(cudaStreamEndCapture(mStream, &graph), cudaSuccess);
    auto const freeGraph = FuncGuard([&]() { cudaGraphDestroy(graph); });
    cudaGraphExec_t exec{};
    ASSERT_EQ(cudaGraphInstantiateWithFlags(&exec, graph, 0), cudaSuccess);
    auto const freeExec = FuncGuard([&]() { cudaGraphExecDestroy(exec); });
    ASSERT_EQ(cudaGraphLaunch(exec, mStream), cudaSuccess);
    read->close();
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t count = 0;
    ASSERT_EQ(cudaMemcpy(&count, deviceCount, sizeof(count), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(count, 2);
    mCache->invalidateHostCopy(kSparse, kPage);
    read = mManager->acquireHostSources(mStream);
    ASSERT_EQ(cudaGraphLaunch(exec, mStream), cudaSuccess);
    read->close();
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&count, deviceCount, sizeof(count), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(view.completedTokens, mManager->hostSourceView().completedTokens);
    mCache->close();
    auto emptyRead = mManager->acquireHostSources(mStream);
    EXPECT_THROW(mManager->shutdown(), LogicError);
    emptyRead->close();
    EXPECT_NO_THROW(mManager->shutdown());
}

} // namespace
