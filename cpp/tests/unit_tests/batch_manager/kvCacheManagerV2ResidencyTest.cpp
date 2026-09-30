/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kvCacheManagerV2TestUtils.h"
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

KVCacheManagerConfig residencyConfig()
{
    auto config = test::makeConfig();
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    auto attention = std::get<AttentionLayerConfig>(config.layers.front());
    attention.layerId = 1;
    attention.residencyGroup = 1;
    config.layers.emplace_back(std::move(attention));
    config.maxUtilForResume = 1.0f;
    return config;
}

TEST(KvCacheManagerV2ResidencyTest, GroupIdentitySeparatesOtherwiseIdenticalLifecycles)
{
    auto config = residencyConfig();
    LifeCycleRegistry groups(config);
    ASSERT_EQ(groups.size(), LifeCycleId{2});
    EXPECT_NE(groups.getId(makeLifeCycle(config.layers[0], 4)), groups.getId(makeLifeCycle(config.layers[1], 4)));
    EXPECT_EQ(getStaleRange(groups[LifeCycleId{1}], 100, 4).length(), 0);

    std::get<AttentionLayerConfig>(config.layers[1]).residencyGroup = 0;
    EXPECT_EQ(LifeCycleRegistry(config).size(), LifeCycleId{1});
    std::get<AttentionLayerConfig>(config.layers[1]).residencyGroup = -1;
    EXPECT_THROW(LifeCycleRegistry{config}, std::invalid_argument);
}

class KvCacheManagerV2ResidencyGpuTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), cudaSuccess);
        mManager = std::make_shared<KvCacheManager>(residencyConfig());
        mCache = mManager->createKvCache();
        ASSERT_TRUE(mCache->resume(mStream));
        ASSERT_TRUE(mCache->resize(20, 0));
        for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{5}; ++ordinal)
        {
            auto const source = page(ordinal);
            auto& storage = mManager->storage();
            auto const group = storage.getPoolGroupIndex(kHotLevel, kSparse);
            auto const address
                = std::get<MemAddress>(storage.slotAddress(kHotLevel, group, source->slotId(), PoolIndex{0}));
            ASSERT_EQ(
                cudaMemsetAsync(reinterpret_cast<void*>(address), 0x30 + ordinal.value(), 4096, mStream), cudaSuccess);
        }
    }

    void TearDown() override
    {
        if (mCache)
            mCache->close();
        mCache.reset();
        if (mManager)
            mManager->clearReusableBlocks();
        mManager.reset();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    SharedPtr<Page> page(BlockOrdinal ordinal, LifeCycleId group = kSparse)
    {
        return blockPageGetPage(mCache->blocks()[ordinal].pages[kDefaultBeamIndex][group]);
    }

    bool locked(BlockOrdinal ordinal, LifeCycleId group = kSparse)
    {
        return std::holds_alternative<SharedPageLock>(mCache->blocks()[ordinal].pages[kDefaultBeamIndex][group]);
    }

    void makeColdHistory()
    {
        mCache->setResidencyWindow(kSparse, 5, 4);
        mCache->setHistoryLength(16);
    }

    void offloadColdHistory()
    {
        auto& storage = mManager->storage();
        TypedVec<PoolGroupIndex, SlotCount> counts(storage.numPoolGroups(), 0);
        counts[storage.getPoolGroupIndex(kSparse)] = 2;
        storage.forceEvict(kHotLevel, counts);
    }

    void checkPayload()
    {
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        auto& storage = mManager->storage();
        auto const group = storage.getPoolGroupIndex(kHotLevel, kSparse);
        for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{5}; ++ordinal)
        {
            ASSERT_EQ(page(ordinal)->cacheLevel, kHotLevel);
            auto const address
                = std::get<MemAddress>(storage.slotAddress(kHotLevel, group, page(ordinal)->slotId(), PoolIndex{0}));
            std::vector<uint8_t> bytes(4096);
            ASSERT_EQ(
                cudaMemcpy(bytes.data(), reinterpret_cast<void const*>(address), bytes.size(), cudaMemcpyDeviceToHost),
                cudaSuccess);
            EXPECT_TRUE(std::all_of(
                bytes.begin(), bytes.end(), [ordinal](uint8_t value) { return value == 0x30 + ordinal.value(); }));
        }
    }

    static constexpr LifeCycleId kSparse{1};
    cudaStream_t mStream{};
    std::shared_ptr<KvCacheManager> mManager;
    std::shared_ptr<KvCache> mCache;
};

TEST_F(KvCacheManagerV2ResidencyGpuTest, HoldsUncommittedHistoryAndProtectsScoringSinksWindowAndWrites)
{
    makeColdHistory();
    ASSERT_TRUE(mCache->isActive());
    EXPECT_EQ(mCache->numCommittedTokens(), 0);
    for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{5}; ++ordinal)
    {
        EXPECT_TRUE(locked(ordinal, LifeCycleId{0}));
        EXPECT_EQ(locked(ordinal), ordinal != BlockOrdinal{1} && ordinal != BlockOrdinal{2});
    }
    EXPECT_EQ(page(BlockOrdinal{1})->status(), PageStatus::HELD);
    EXPECT_FALSE(page(BlockOrdinal{1})->isCommitted());
    EXPECT_EQ(mCache->getBasePageIndices(kSparse)[1], kBadPageIndex.value());
    offloadColdHistory();
    EXPECT_NE(page(BlockOrdinal{1})->cacheLevel, kHotLevel);
    EXPECT_NE(page(BlockOrdinal{2})->cacheLevel, kHotLevel);
    mCache->stopCommitting();
    EXPECT_TRUE(page(BlockOrdinal{1}));
    mCache->setResidencyWindow(kSparse, std::nullopt);
    checkPayload();
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, SuspendResumePreservesColdHistoryAndReacquiresOnlyRequiredPages)
{
    makeColdHistory();
    offloadColdHistory();
    auto const coldSlot = page(BlockOrdinal{1})->slotId();
    mCache->suspend();
    EXPECT_FALSE(locked(BlockOrdinal{0}));
    ASSERT_TRUE(mCache->resume(mStream));
    EXPECT_TRUE(locked(BlockOrdinal{0}));
    EXPECT_FALSE(locked(BlockOrdinal{1}));
    EXPECT_EQ(page(BlockOrdinal{1})->slotId(), coldSlot);
    EXPECT_NE(page(BlockOrdinal{1})->cacheLevel, kHotLevel);
    mCache->setResidencyWindow(kSparse, std::nullopt);
    checkPayload();
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, HistoryAdvanceAndShrinkPreserveLiveColdPages)
{
    makeColdHistory();
    ASSERT_TRUE(mCache->resize(24, 20));
    EXPECT_FALSE(locked(BlockOrdinal{3}));
    EXPECT_TRUE(locked(BlockOrdinal{4}));
    EXPECT_TRUE(locked(BlockOrdinal{5}));
    ASSERT_TRUE(mCache->resize(20, 20));
    EXPECT_EQ(mCache->numBlocks(), BlockOrdinal{5});
    EXPECT_TRUE(page(BlockOrdinal{3}));
    mCache->setResidencyWindow(kSparse, std::nullopt);
    checkPayload();
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, CommitOffloadedPagesAndSharePrefixWithoutRelockingColdHistory)
{
    makeColdHistory();
    offloadColdHistory();
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < 16; ++token)
        tokens.emplace_back(token);
    mCache->commit(TokenSpan{tokens.data(), static_cast<int>(tokens.size())});
    EXPECT_TRUE(page(BlockOrdinal{1})->isCommitted());
    EXPECT_NE(page(BlockOrdinal{1})->cacheLevel, kHotLevel);
    EXPECT_FALSE(locked(BlockOrdinal{1}));

    auto other = mManager->createKvCache({}, TokenSpan{tokens.data(), static_cast<int>(tokens.size())});
    ASSERT_TRUE(other->resume(mStream));
    EXPECT_EQ(page(BlockOrdinal{1})->status(), PageStatus::LOCKED);
    EXPECT_FALSE(mManager->storage().isEvictable(*page(BlockOrdinal{1})));
    other->close();
    EXPECT_EQ(page(BlockOrdinal{1})->status(), PageStatus::HELD);
    mCache->setResidencyWindow(kSparse, std::nullopt);
    checkPayload();
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, CommitPreservesTransferCompletionFromAnotherStream)
{
    makeColdHistory();
    offloadColdHistory();
    cudaStream_t transferStream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking), cudaSuccess);
    std::atomic<bool> release{false};
    auto guard = FuncGuard(
        [&]()
        {
            release.store(true);
            cudaStreamSynchronize(transferStream);
            cudaStreamDestroy(transferStream);
        });
    ASSERT_EQ(cudaLaunchHostFunc(
                  transferStream,
                  [](void* state)
                  {
                      auto& released = *static_cast<std::atomic<bool>*>(state);
                      while (!released.load())
                          std::this_thread::yield();
                  },
                  &release),
        cudaSuccess);
    // Model a still-running transfer on the held page. The committing stream
    // does not wait on this event because commitment does not read the payload.
    page(BlockOrdinal{1})->readyEvent = CachedCudaEvent(reinterpret_cast<CudaStream>(transferStream));
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < 16; ++token)
        tokens.emplace_back(token);
    mCache->commit(TokenSpan{tokens.data(), static_cast<int>(tokens.size())});
    EXPECT_FALSE(page(BlockOrdinal{1})->queryReady());
    release.store(true);
    mCache->setResidencyWindow(kSparse, std::nullopt);
    checkPayload();
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, RebaseKeepsColdPagesHeld)
{
    makeColdHistory();
    offloadColdHistory();
    auto donor = mManager->createKvCache();
    ASSERT_TRUE(donor->resume(mStream));
    ASSERT_TRUE(donor->resize(20, 16));
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < 16; ++token)
        tokens.emplace_back(token);
    donor->commit(TokenSpan{tokens.data(), static_cast<int>(tokens.size())});
    mCache->commit(TokenSpan{tokens.data(), static_cast<int>(tokens.size())});
    EXPECT_FALSE(locked(BlockOrdinal{1}));
    EXPECT_EQ(
        page(BlockOrdinal{1}), blockPageGetPage(donor->blocks()[BlockOrdinal{1}].pages[kDefaultBeamIndex][kSparse]));
    EXPECT_EQ(page(BlockOrdinal{1})->status(), PageStatus::LOCKED);
    donor->close();
    EXPECT_EQ(page(BlockOrdinal{1})->status(), PageStatus::HELD);
    EXPECT_EQ(mCache->getBasePageIndices(kSparse)[1], kBadPageIndex.value());
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, ExternalPageTableTracksLockReleaseAndReacquisition)
{
    std::vector<int32_t> indices(6, kBadPageIndex.value());
    mCache->setBasePageIndexBuf(kDefaultBeamIndex, kSparse, indices.data(), static_cast<int>(indices.size()));
    auto guard = FuncGuard([&]() { mCache->setBasePageIndexBuf(kDefaultBeamIndex, kSparse, nullptr, 0); });
    makeColdHistory();
    EXPECT_EQ(indices[1], kBadPageIndex.value());
    EXPECT_NE(indices[0], kBadPageIndex.value());
    offloadColdHistory();
    mCache->setResidencyWindow(kSparse, std::nullopt);
    EXPECT_NE(indices[1], kBadPageIndex.value());
    mCache->suspend();
    // Suspension detaches the caller's buffer. Inspect the internal table, then
    // reattach after resume as the executor does.
    auto const suspended = mCache->getBasePageIndices(kSparse);
    EXPECT_TRUE(
        std::all_of(suspended.begin(), suspended.end(), [](int index) { return index == kBadPageIndex.value(); }));
    ASSERT_TRUE(mCache->resume(mStream));
    mCache->setBasePageIndexBuf(kDefaultBeamIndex, kSparse, indices.data(), static_cast<int>(indices.size()));
    EXPECT_NE(indices[1], kBadPageIndex.value());
    checkPayload();
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, SparsePolicyCoexistsWithSlidingWindowAndSsmLifecycles)
{
    auto config = residencyConfig();
    auto swa = std::get<AttentionLayerConfig>(config.layers[0]);
    swa.layerId = 2;
    swa.slidingWindowSize = 5;
    config.layers.emplace_back(swa);
    config.layers.emplace_back(SsmLayerConfig{3, {BufferConfig{"state", 4096, std::nullopt}}});
    config.commitMinSnapshot = true;
    auto manager = std::make_shared<KvCacheManager>(config);
    auto cache = manager->createKvCache();
    auto guard = FuncGuard([&]() { cache->close(); });
    auto const swaGroup = manager->lifeCycles().getId(makeLifeCycle(config.layers[2], 4));
    auto const ssmGroup = manager->lifeCycles().getId(makeLifeCycle(config.layers[3], 4));
    EXPECT_THROW(cache->setResidencyWindow(swaGroup, 5), std::invalid_argument);
    EXPECT_THROW(cache->setResidencyWindow(ssmGroup, 5), std::invalid_argument);
    ASSERT_TRUE(cache->resume(mStream));
    ASSERT_TRUE(cache->resize(20, 16));
    cache->stopCommitting();
    cache->setResidencyWindow(kSparse, 5, 4);
    EXPECT_TRUE(blockPageIsNull(cache->blocks()[BlockOrdinal{1}].pages[kDefaultBeamIndex][swaGroup]));
    EXPECT_FALSE(blockPageIsNull(cache->blocks()[BlockOrdinal{1}].pages[kDefaultBeamIndex][kSparse]));
    cache->suspend();
    cache->setResidencyWindow(kSparse, 9, 4);
    ASSERT_TRUE(cache->resume(mStream));
    EXPECT_TRUE(
        std::holds_alternative<SharedPageLock>(cache->blocks()[BlockOrdinal{2}].pages[kDefaultBeamIndex][kSparse]));
    EXPECT_TRUE(std::holds_alternative<SharedPtr<PageHolder>>(
        cache->blocks()[BlockOrdinal{1}].pages[kDefaultBeamIndex][kSparse]));
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, FailedExpansionPreservesResidencyAndRequestState)
{
    makeColdHistory();
    offloadColdHistory();
    auto& storage = mManager->storage();
    auto const group = storage.getPoolGroupIndex(kSparse);
    auto blockers = storage.newSlotsForPoolGroup(kHotLevel, group, storage.getStatistics(kHotLevel, group).free);
    EXPECT_THROW(mCache->setResidencyWindow(kSparse, std::nullopt), OutOfPagesError);
    EXPECT_TRUE(mCache->isActive());
    EXPECT_FALSE(locked(BlockOrdinal{1}));
    EXPECT_TRUE(locked(BlockOrdinal{3}));
    EXPECT_FALSE(mCache->resize(24, 20));
    EXPECT_EQ(mCache->historyLength(), 16);
    EXPECT_EQ(mCache->capacity(), 20);
    EXPECT_TRUE(locked(BlockOrdinal{3}));
    for (auto& slot : blockers)
        storage.releaseSlot(kSparse, kHotLevel, std::move(slot));
    ASSERT_TRUE(mCache->resize(24, 20));
    EXPECT_FALSE(locked(BlockOrdinal{3}));
}

TEST_F(KvCacheManagerV2ResidencyGpuTest, ValidatesResidencyWindowAndSupportsSuspendedConfiguration)
{
    EXPECT_THROW(mCache->setResidencyWindow(kSparse, 0), std::invalid_argument);
    EXPECT_THROW(mCache->setResidencyWindow(kSparse, -1), std::invalid_argument);
    EXPECT_THROW(mCache->setResidencyWindow(kSparse, 4, -1), std::invalid_argument);
    EXPECT_THROW(mCache->setResidencyWindow(kSparse, std::nullopt, 1), std::invalid_argument);
    mCache->setHistoryLength(16);
    mCache->suspend();
    mCache->setResidencyWindow(kSparse, 5, 4);
    ASSERT_TRUE(mCache->resume(mStream));
    EXPECT_FALSE(locked(BlockOrdinal{1}));
    EXPECT_TRUE(locked(BlockOrdinal{0}));
    mCache->close();
    EXPECT_THROW(mCache->setResidencyWindow(kSparse, 4), LogicError);
}

} // namespace
