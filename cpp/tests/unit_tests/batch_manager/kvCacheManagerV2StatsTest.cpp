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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/eventManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/pendingStats.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/stats.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <limits>
#include <memory>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

KVCacheManagerConfig makeConfig(bool enableStats = true)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    config.enableStats = enableStats;
    return config;
}

KVCacheManagerConfig makeTieredConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 4;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 2 << 20, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    return config;
}

TEST(KvCacheManagerV2StatsTest, StatsDeltaArithmetic)
{
    KVCacheStatsDelta stats{4, 3, 2, 1};
    KVCacheStatsDelta const delta{1, 2, 3, 4};
    stats.add(delta);
    EXPECT_EQ(stats.allocTotalBlocks, 5);
    EXPECT_EQ(stats.allocNewBlocks, 5);
    EXPECT_EQ(stats.reusedBlocks, 5);
    EXPECT_EQ(stats.missedBlocks, 5);

    KVCacheStatsDelta const copy = stats.copy();
    stats.subtract(delta);
    EXPECT_EQ(stats.allocTotalBlocks, 4);
    EXPECT_EQ(copy.allocTotalBlocks, 5);
    stats.clear();
    EXPECT_TRUE(stats.empty());
}

TEST(KvCacheManagerV2StatsTest, IterationStatsDeltaArithmeticAndHitRate)
{
    KVCacheIterationStatsDelta stats;
    stats.iterReusedBlocks = 3;
    stats.iterFullReusedBlocks = 2;
    stats.iterPartialReusedBlocks = 1;
    stats.iterMissedBlocks = 1;
    stats.iterOnboardBytes = 1024;
    EXPECT_DOUBLE_EQ(stats.iterCacheHitRate(), 0.75);

    KVCacheIterationStatsDelta delta = stats.copy();
    stats.add(delta);
    EXPECT_EQ(stats.iterReusedBlocks, 6);
    EXPECT_EQ(stats.iterOnboardBytes, 2048);
    stats.subtract(delta);
    EXPECT_EQ(stats.iterReusedBlocks, 3);
    stats.clear();
    EXPECT_TRUE(stats.empty());
    EXPECT_DOUBLE_EQ(stats.iterCacheHitRate(), 0.0);
}

TEST(KvCacheManagerV2StatsTest, PendingAllocationRangesAreReversibleAndScoped)
{
    PendingStats pending;
    EXPECT_TRUE(pending.recordAllocationRange(
        LifeCycleId{0}, BlockOrdinal{0}, BlockOrdinal{3}, /*beamWidth=*/2, /*countAsMissed=*/true));
    EXPECT_TRUE(pending.recordAllocationRange(LifeCycleId{1}, BlockOrdinal{3}, BlockOrdinal{5},
        /*beamWidth=*/1, /*countAsMissed=*/false, /*countAsGeneration=*/true));

    EXPECT_EQ(pending.globalStats().allocTotalBlocks, 8);
    EXPECT_EQ(pending.globalStats().allocNewBlocks, 8);
    EXPECT_EQ(pending.globalStats().missedBlocks, 6);
    EXPECT_EQ(pending.requestStats().allocTotalBlocks, 8);
    ASSERT_EQ(pending.iterationStatsByLifeCycle().size(), 2);
    EXPECT_EQ(pending.iterationStatsByLifeCycle().at(LifeCycleId{0}).iterMissedBlocks, 6);
    EXPECT_EQ(pending.iterationStatsByLifeCycle().at(LifeCycleId{1}).iterGenAllocBlocks, 2);

    EXPECT_TRUE(pending.subtractAllocationRange(BlockOrdinal{2}, BlockOrdinal{5}));
    EXPECT_EQ(pending.globalStats().allocTotalBlocks, 4);
    EXPECT_EQ(pending.globalStats().missedBlocks, 4);
    ASSERT_EQ(pending.iterationStatsByLifeCycle().size(), 1);
    EXPECT_EQ(pending.iterationStatsByLifeCycle().at(LifeCycleId{0}).iterAllocTotalBlocks, 4);

    EXPECT_TRUE(pending.subtractAllocationRange(BlockOrdinal{0}, BlockOrdinal{2}));
    EXPECT_TRUE(pending.empty());
}

TEST(KvCacheManagerV2StatsTest, PendingReuseSurvivesAllocationRollbackUntilClear)
{
    PendingStats pending;
    EXPECT_TRUE(pending.recordAllocationRange(
        LifeCycleId{0}, BlockOrdinal{0}, BlockOrdinal{1}, /*beamWidth=*/1, /*countAsMissed=*/true));
    EXPECT_TRUE(pending.recordReuse(LifeCycleId{0}, /*fullReusedBlocks=*/2, /*partialReusedBlocks=*/1));

    EXPECT_TRUE(pending.subtractAllocationRange(BlockOrdinal{0}, BlockOrdinal{1}));
    EXPECT_EQ(pending.globalStats().allocTotalBlocks, 0);
    EXPECT_EQ(pending.globalStats().reusedBlocks, 3);
    auto const& iteration = pending.iterationStatsByLifeCycle().at(LifeCycleId{0});
    EXPECT_EQ(iteration.iterReusedBlocks, 3);
    EXPECT_EQ(iteration.iterFullReusedBlocks, 2);
    EXPECT_EQ(iteration.iterPartialReusedBlocks, 1);

    pending.clear();
    EXPECT_TRUE(pending.empty());
}

TEST(KvCacheManagerV2StatsTest, ManagerCommitResetAndRequestIdTracking)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig());

    KVCacheStatsDelta globalStats{4, 3, 2, 1};
    KVCacheIterationStatsDelta iterationStats;
    iterationStats.iterAllocTotalBlocks = 4;
    iterationStats.iterReusedBlocks = 2;
    manager->commitStats(globalStats, {{LifeCycleId{0}, iterationStats}});

    EXPECT_EQ(manager->getCommittedStats().allocTotalBlocks, 4);
    auto firstIteration = manager->getAndResetIterationStats();
    ASSERT_EQ(firstIteration.size(), 1);
    EXPECT_EQ(firstIteration.at(LifeCycleId{0}).iterReusedBlocks, 2);
    EXPECT_TRUE(manager->getAndResetIterationStats().empty());

    manager->markStatsDirty(11);
    manager->markStatsDirty(std::nullopt);
    EXPECT_EQ(manager->getDirtyStatsKvCacheIds().count(11), 1);
    manager->markStatsExcluded(11);
    EXPECT_TRUE(manager->isStatsExcluded(11));
    EXPECT_TRUE(manager->getDirtyStatsKvCacheIds().empty());
    manager->clearStatsExcluded(11);
    EXPECT_FALSE(manager->isStatsExcluded(11));

    auto cache = manager->createKvCache({}, {}, 17, {}, 8);
    manager->markStatsDirty(17);
    EXPECT_TRUE(cache->commitPendingStats().empty());
    EXPECT_TRUE(manager->getDirtyStatsKvCacheIds().empty());
    cache->close();

    RequestIdType const cudaGraphDummyRequestId = std::numeric_limits<RequestIdType>::max();
    auto dummyCache = manager->createKvCache({}, {}, cudaGraphDummyRequestId);
    ASSERT_TRUE(dummyCache->id.has_value());
    EXPECT_EQ(*dummyCache->id, cudaGraphDummyRequestId);
    manager->markStatsDirty(cudaGraphDummyRequestId);
    EXPECT_EQ(manager->getDirtyStatsKvCacheIds(), std::unordered_set<RequestIdType>{cudaGraphDummyRequestId});
    manager->markStatsExcluded(cudaGraphDummyRequestId);
    EXPECT_TRUE(manager->isStatsExcluded(cudaGraphDummyRequestId));
    EXPECT_TRUE(manager->getDirtyStatsKvCacheIds().empty());
    dummyCache->close();
}

TEST(KvCacheManagerV2StatsTest, DisabledStatsSuppressManagerCommit)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig(false));
    manager->commitStats(KVCacheStatsDelta{4, 3, 2, 1});
    EXPECT_TRUE(manager->getCommittedStats().empty());
    EXPECT_TRUE(manager->getAndResetIterationStats().empty());
}

TEST(KvCacheManagerV2LifecycleTest, DetachedBlockRetainsTokensPerBlock)
{
    auto const config = makeConfig();
    LifeCycleRegistry const lifeCycles{config};
    BlockRadixTree tree{lifeCycles, config.tokensPerBlock};
    RootBlock& root = tree.addOrGetExisting({});

    auto block
        = addOrGetExistingBlock(&root, lifeCycles.size(), std::vector<TokenIdExt>(config.tokensPerBlock, TokenId{1}));
    auto detached = root.detachNext(block->key);
    ASSERT_EQ(detached, block);
    EXPECT_TRUE(detached->isOrphan());
    EXPECT_TRUE(detached->isFull());
    EXPECT_EQ(detached->tokensPerBlock(), config.tokensPerBlock);
    EXPECT_TRUE(detached->next.empty());
}

TEST(KvCacheManagerV2LifecycleTest, ResetPreventsLiveCacheFromRepopulatingReuse)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig());
    CachedCudaStream stream;
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(stream.handle()));
    ASSERT_TRUE(cache->resize(4 * manager->tokensPerBlock()));

    std::vector<TokenIdExt> const firstBlock(manager->tokensPerBlock(), TokenId{1});
    std::vector<TokenIdExt> const secondBlock(manager->tokensPerBlock(), TokenId{2});
    std::vector<TokenIdExt> const thirdBlock(manager->tokensPerBlock(), TokenId{3});
    std::vector<TokenIdExt> const fourthBlock(manager->tokensPerBlock(), TokenId{4});
    std::vector<TokenIdExt> committed = firstBlock;
    committed.insert(committed.end(), secondBlock.begin(), secondBlock.end());
    cache->commit(committed);
    EXPECT_EQ(manager->probeReuse({}, committed), 2 * manager->tokensPerBlock());
    auto const livePage = blockPageGetPage(cache->blocks()[BlockOrdinal{0}].pages[kDefaultBeamIndex][LifeCycleId{0}]);
    auto const committedLivePage = dynamicPointerCast<CommittedPage>(livePage);
    ASSERT_TRUE(committedLivePage);
    ASSERT_TRUE(committedLivePage->hasValidSlot());

    std::vector<TokenIdExt> const siblingTokens(manager->tokensPerBlock(), TokenId{9});
    auto siblingCache = manager->createKvCache();
    ASSERT_TRUE(siblingCache->resume(stream.handle()));
    ASSERT_TRUE(siblingCache->resize(manager->tokensPerBlock()));
    siblingCache->commit(siblingTokens);
    WeakPtr<Block> const siblingBlock = siblingCache->blocks()[BlockOrdinal{0}].treeBlock;
    siblingCache->close();
    EXPECT_FALSE(siblingBlock.expired());
    EXPECT_EQ(manager->storage().getStatistics().evictable, SlotCount{1});

    manager->clearReusableBlocks();
    EXPECT_EQ(cache->commitState(), KvCache::CommitState::VIRTUAL_STOP);
    ASSERT_EQ(cache->blocks().size(), BlockOrdinal{4});
    EXPECT_TRUE(cache->blocks()[BlockOrdinal{0}].treeBlock->isOrphan());
    EXPECT_TRUE(cache->blocks()[BlockOrdinal{1}].treeBlock->isOrphan());
    EXPECT_TRUE(cache->blocks()[BlockOrdinal{0}].treeBlock->next.empty());
    EXPECT_TRUE(cache->blocks()[BlockOrdinal{1}].treeBlock->next.empty());
    EXPECT_EQ(cache->blocks()[BlockOrdinal{0}].treeBlock->getPage(LifeCycleId{0}), nullptr);
    EXPECT_EQ(committedLivePage->block, nullptr);
    EXPECT_EQ(committedLivePage->status(), PageStatus::LOCKED);
    EXPECT_TRUE(committedLivePage->hasValidSlot());
    EXPECT_TRUE(siblingBlock.expired());
    EXPECT_EQ(manager->storage().getStatistics().evictable, SlotCount{0});
    EXPECT_TRUE(manager->radixTree().roots().empty());
    EXPECT_EQ(manager->probeReuse({}, committed), 0);
    EXPECT_NO_THROW(cache->commit(thirdBlock));
    EXPECT_EQ(cache->numCommittedTokens(), 3 * manager->tokensPerBlock());
    EXPECT_EQ(cache->historyLength(), cache->numCommittedTokens());
    EXPECT_NO_THROW(cache->commit(fourthBlock));
    EXPECT_EQ(cache->numCommittedTokens(), 4 * manager->tokensPerBlock());
    EXPECT_EQ(cache->historyLength(), cache->numCommittedTokens());
    EXPECT_EQ(cache->commitState(), KvCache::CommitState::VIRTUAL_STOP);
    EXPECT_TRUE(manager->radixTree().roots().empty());

    cache->stopCommitting();
    EXPECT_EQ(cache->commitState(), KvCache::CommitState::USER_STOP);
    cache->close();
    stream.synchronize();
    committed.insert(committed.end(), thirdBlock.begin(), thirdBlock.end());
    committed.insert(committed.end(), fourthBlock.begin(), fourthBlock.end());
    EXPECT_EQ(manager->probeReuse({}, committed), 0);
}

TEST(KvCacheManagerV2LifecycleTest, OrphanedParentStopsFullBlockReuseBeforeV1Event)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto eventManager
        = std::make_shared<EventManager>(1024, 0, std::nullopt, EventManager::AttentionDpGatherFn{}, "v1_block_key");
    auto manager = std::make_shared<KvCacheManager>(makeConfig(), eventManager);
    CachedCudaStream stream;
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(stream.handle()));
    ASSERT_TRUE(cache->resize(4 * manager->tokensPerBlock()));

    std::vector<TokenIdExt> const firstBlock(manager->tokensPerBlock(), TokenId{1});
    cache->commit(firstBlock);
    auto const orphan = cache->blocks()[BlockOrdinal{0}].treeBlock;
    ASSERT_TRUE(orphan);
    manager->radixTree().clear();
    ASSERT_TRUE(orphan->isOrphan());
    ASSERT_TRUE(manager->radixTree().roots().empty());

    EXPECT_NO_THROW(cache->commit(std::vector<TokenIdExt>(manager->tokensPerBlock(), TokenId{2})));
    EXPECT_EQ(cache->commitState(), KvCache::CommitState::VIRTUAL_STOP);
    EXPECT_EQ(cache->numCommittedBlocks(), 1);
    EXPECT_TRUE(orphan->next.empty());
    EXPECT_TRUE(manager->radixTree().roots().empty());
    EXPECT_NO_THROW(cache->commit(std::vector<TokenIdExt>(manager->tokensPerBlock(), TokenId{3})));
    EXPECT_NO_THROW(cache->commit(std::vector<TokenIdExt>(manager->tokensPerBlock(), TokenId{4})));
    EXPECT_EQ(cache->numCommittedTokens(), 4 * manager->tokensPerBlock());
    EXPECT_EQ(cache->historyLength(), cache->numCommittedTokens());

    cache->stopCommitting();
    cache->close();
    stream.synchronize();
    EXPECT_NO_THROW(manager->radixTree().clear());
    EXPECT_NO_THROW(manager->shutdown());
}

TEST(KvCacheManagerV2LifecycleTest, OrphanedParentStopsPartialSnapshotBeforeV1Event)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeConfig();
    config.commitMinSnapshot = true;
    auto eventManager
        = std::make_shared<EventManager>(1024, 0, std::nullopt, EventManager::AttentionDpGatherFn{}, "v1_block_key");
    auto manager = std::make_shared<KvCacheManager>(config, eventManager);
    CachedCudaStream stream;
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(stream.handle()));
    ASSERT_TRUE(cache->resize(2 * manager->tokensPerBlock()));

    cache->commit(std::vector<TokenIdExt>(manager->tokensPerBlock(), TokenId{1}));
    auto const orphan = cache->blocks()[BlockOrdinal{0}].treeBlock;
    ASSERT_TRUE(orphan);
    manager->radixTree().clear();
    ASSERT_TRUE(orphan->isOrphan());

    EXPECT_NO_THROW(cache->commit({TokenId{2}}));
    EXPECT_EQ(cache->commitState(), KvCache::CommitState::VIRTUAL_STOP);
    EXPECT_EQ(cache->numCommittedBlocks(), 1);
    EXPECT_TRUE(orphan->next.empty());
    EXPECT_TRUE(manager->radixTree().roots().empty());

    cache->close();
    stream.synchronize();
}

TEST(KvCacheManagerV2LifecycleTest, OrphanedParentStopsPartialFinalCommitBeforeV1Event)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto eventManager
        = std::make_shared<EventManager>(1024, 0, std::nullopt, EventManager::AttentionDpGatherFn{}, "v1_block_key");
    auto manager = std::make_shared<KvCacheManager>(makeConfig(), eventManager);
    CachedCudaStream stream;
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(stream.handle()));
    ASSERT_TRUE(cache->resize(2 * manager->tokensPerBlock()));

    cache->commit(std::vector<TokenIdExt>(manager->tokensPerBlock(), TokenId{1}));
    auto const orphan = cache->blocks()[BlockOrdinal{0}].treeBlock;
    ASSERT_TRUE(orphan);
    manager->radixTree().clear();
    ASSERT_TRUE(orphan->isOrphan());

    cache->commit({TokenId{2}});
    ASSERT_EQ(cache->commitState(), KvCache::CommitState::ALLOWED);
    EXPECT_NO_THROW(cache->stopCommitting());
    EXPECT_EQ(cache->commitState(), KvCache::CommitState::USER_STOP);
    EXPECT_EQ(cache->numCommittedBlocks(), 1);
    EXPECT_TRUE(orphan->next.empty());
    EXPECT_TRUE(manager->radixTree().roots().empty());

    cache->close();
    stream.synchronize();
}

TEST(KvCacheManagerV2LifecycleTest, ResetReleasesSwaPagesHeldOnlyForCommit)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeConfig();
    std::get<AttentionLayerConfig>(config.layers.front()).slidingWindowSize = 4;
    auto manager = std::make_shared<KvCacheManager>(config);
    CachedCudaStream stream;
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(stream.handle()));
    ASSERT_TRUE(cache->resize(12, 0));
    ASSERT_TRUE(cache->resize(12, 8));

    auto const& stalePage = cache->blocks()[BlockOrdinal{0}].pages[kDefaultBeamIndex][LifeCycleId{0}];
    ASSERT_TRUE(std::holds_alternative<SharedPtr<PageHolder>>(stalePage));
    manager->clearReusableBlocks();
    EXPECT_EQ(cache->commitState(), KvCache::CommitState::VIRTUAL_STOP);
    EXPECT_TRUE(blockPageIsNull(stalePage));

    EXPECT_NO_THROW(manager->clearReusableBlocks());
    EXPECT_TRUE(cache->resize(std::nullopt, 12));
    cache->close();
    stream.synchronize();
}

TEST(KvCacheManagerV2StatsTest, PeakBlockStatsResetStartsNextIntervalFromCurrentSnapshot)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};

    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    auto gpuSlots = storage.newGpuSlots(twoSlots);
    manager->commitStats({});

    RootBlock& root = manager->radixTree().addOrGetExisting({});
    std::vector<SharedPtr<Page>> pages;
    NodeBase* previous = &root;
    int token = 0;
    for (auto& slot : gpuSlots[lifeCycle])
    {
        std::vector<TokenIdExt> tokens;
        for (int i = 0; i < manager->tokensPerBlock(); ++i)
        {
            tokens.emplace_back(TokenId{token++});
        }
        auto block = addOrGetExistingBlock(previous, LifeCycleId{1}, std::move(tokens));
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
        page->setSlot(slot);
        block->storage[lifeCycle] = page.get();
        storage.scheduleForEviction(*page);
        pages.push_back(page);
        previous = block.get();
    }
    manager->commitStats({});

    TypedVec<LifeCycleId, SlotCount> oneSlot(LifeCycleId{1}, 1);
    auto hostSlots = storage.newSlots(CacheLevel{1}, oneSlot);
    manager->commitStats({});
    storage.releaseSlot(lifeCycle, CacheLevel{1}, std::move(hostSlots[lifeCycle].front()));
    manager->clearReusableBlocks();
    pages.clear();

    auto primaryPeak = manager->getAndResetIterationPeakBlockStats(kGpuLevel);
    auto secondaryPeak = manager->getAndResetIterationPeakBlockStats(CacheLevel{1});
    ASSERT_EQ(primaryPeak.size(), PoolGroupIndex{1});
    ASSERT_EQ(secondaryPeak.size(), PoolGroupIndex{1});
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].unavailable, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].evictable, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].unavailable, 1);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].evictable, 0);

    primaryPeak = manager->getAndResetIterationPeakBlockStats(kGpuLevel);
    secondaryPeak = manager->getAndResetIterationPeakBlockStats(CacheLevel{1});
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].unavailable, 0);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].evictable, 0);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].unavailable, 0);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].evictable, 0);

    auto nextIntervalSlots = storage.newSlots(kGpuLevel, oneSlot);
    manager->commitStats({});
    storage.releaseSlot(lifeCycle, kGpuLevel, std::move(nextIntervalSlots[lifeCycle].front()));
    primaryPeak = manager->getAndResetIterationPeakBlockStats(kGpuLevel);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].unavailable, 1);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].evictable, 0);
}

TEST(KvCacheManagerV2StatsTest, MigrationAndLastTierDropRecordersReceiveExactPages)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const lifeCycle{0};
    ASSERT_EQ(storage.getStatistics(kGpuLevel).total, 2);
    ASSERT_EQ(storage.getStatistics(CacheLevel{1}).total, 2);

    int offloaded = 0;
    int onboarded = 0;
    int dropped = 0;
    MigrationRecorder const migrationRecorder
        = [&](std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots, CacheLevel srcLevel,
              CacheLevel dstLevel)
    {
        EXPECT_EQ(pages.size(), slots.size());
        if (srcLevel == kGpuLevel && dstLevel == CacheLevel{1})
        {
            offloaded += static_cast<int>(pages.size());
        }
        else if (srcLevel == CacheLevel{1} && dstLevel == kGpuLevel)
        {
            onboarded += static_cast<int>(pages.size());
        }
    };
    DropRecorder const dropRecorder = [&](std::vector<SharedPtr<Page>> const& pages, CacheLevel level)
    {
        EXPECT_EQ(level, CacheLevel{1});
        dropped += static_cast<int>(pages.size());
    };

    RootBlock& root = manager->radixTree().addOrGetExisting({});
    int tokenBase = 0;
    auto makeCommittedPages = [&](std::vector<Slot> slots)
    {
        std::vector<SharedPtr<Page>> pages;
        NodeBase* previous = &root;
        for (auto& slot : slots)
        {
            std::vector<TokenIdExt> tokens;
            for (int i = 0; i < manager->tokensPerBlock(); ++i)
            {
                tokens.emplace_back(TokenId{tokenBase++});
            }
            auto block = addOrGetExistingBlock(previous, LifeCycleId{1}, std::move(tokens));
            auto page = makeShared<CommittedPage>(
                &storage, block, lifeCycle, kGpuLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
            page->setSlot(slot);
            block->storage[lifeCycle] = page.get();
            storage.scheduleForEviction(*page);
            pages.push_back(page);
            previous = block.get();
        }
        return pages;
    };

    TypedVec<LifeCycleId, SlotCount> twoSlots(LifeCycleId{1}, 2);
    auto initialSlots = storage.newGpuSlots(twoSlots);
    auto firstPages = makeCommittedPages(std::move(initialSlots[lifeCycle]));

    auto temporarySlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 2);
    EXPECT_EQ(onboarded, 0);
    EXPECT_EQ(dropped, 0);
    for (auto& slot : temporarySlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
    }

    auto cache = manager->createKvCache();
    std::vector<BatchedLockTarget> targets;
    for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{2}; ++ordinal)
    {
        auto const& page = firstPages[toSizeT(ordinal)];
        ASSERT_TRUE(page->scheduledForEviction());
        storage.excludeFromEviction(*page);
        targets.push_back({page, kDefaultBeamIndex, ordinal, lifeCycle});
    }
    storage.batchedMigrateToGpu(targets, *cache, migrationRecorder);
    EXPECT_EQ(onboarded, 2);
    for (auto const& page : firstPages)
    {
        storage.scheduleForEviction(*page);
    }

    temporarySlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 4);
    auto secondPages = makeCommittedPages(std::move(temporarySlots[lifeCycle]));
    (void) secondPages;
    firstPages.clear();
    targets.clear();

    auto finalSlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 6);
    EXPECT_EQ(onboarded, 2);
    EXPECT_EQ(dropped, 2);
    for (auto& slot : finalSlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, kGpuLevel, std::move(slot));
    }
    cache->close();
}

} // namespace
