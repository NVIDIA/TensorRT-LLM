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
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/pendingStats.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/stats.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeHybridTieredConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeTieredConfig;

class ReplayCacheTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
    }

    void TearDown() override
    {
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    static std::vector<TokenIdExt> makeTokens(int length)
    {
        std::vector<TokenIdExt> tokens;
        for (int i = 0; i < length; ++i)
        {
            tokens.emplace_back(TokenId{i});
        }
        return tokens;
    }

    static void addWindow(KVCacheManagerConfig& config, int id, int window, AttentionReusePolicy policy, int sinks = 0)
    {
        auto layer = std::get<AttentionLayerConfig>(config.layers.front());
        layer.layerId = id;
        layer.slidingWindowSize = window;
        layer.numSinkTokens = sinks;
        layer.reusePolicy = policy;
        config.layers.emplace_back(layer);
    }

    cudaStream_t mStream{};
};

TEST_F(ReplayCacheTest, GlobalReuseKeepsPrivateAttentionUnpublished)
{
    for (int length : {23, 24})
    {
        SCOPED_TRACE(length);
        auto config = makeConfig();
        addWindow(config, 1, 8, AttentionReusePolicy::PRIVATE);
        auto manager = std::make_shared<KvCacheManager>(config);
        auto const globalLc = manager->getLayerGroupId(LayerId{0});
        auto const privateLc = manager->getLayerGroupId(LayerId{1});
        auto const tokens = makeTokens(length);
        TokenSpan const tokenSpan{tokens.data(), static_cast<int>(tokens.size())};
        auto first = manager->createKvCache({}, tokenSpan, 1);
        ASSERT_TRUE(first->resume(mStream));
        ASSERT_TRUE(first->resize(length));
        first->commit(tokenSpan);
        first->stopCommitting();
        auto second = manager->createKvCache({}, tokenSpan, 2);
        EXPECT_EQ(second->numCommittedTokens(), length);
        auto& storage = manager->storage();
        auto const group = storage.getPoolGroupIndex(kHotLevel, privateLc);
        auto reservations
            = storage.newSlotsForPoolGroup(kHotLevel, group, storage.getStatistics(kHotLevel, group).free);
        for (int attempt = 0; attempt < 3; ++attempt)
        {
            EXPECT_FALSE(second->resume(mStream));
            EXPECT_FALSE(second->isActive());
            EXPECT_EQ(second->numCommittedTokens(), length);
        }
        for (auto& slot : reservations)
        {
            storage.releaseSlot(privateLc, kHotLevel, std::move(slot));
        }
        ASSERT_TRUE(second->resume(mStream));
        auto firstPrivate = first->getBasePageIndices(privateLc);
        auto secondPrivate = second->getBasePageIndices(privateLc);
        EXPECT_NE(firstPrivate[5], secondPrivate[5]);
        auto firstGlobal = first->getBasePageIndices(globalLc);
        auto secondGlobal = second->getBasePageIndices(globalLc);
        EXPECT_EQ(firstGlobal[0], secondGlobal[0]);
        EXPECT_EQ(firstGlobal[5] == secondGlobal[5], length % 4 == 0);
        {
            auto match = manager->radixTree().match({}, tokenSpan, false, true);
            EXPECT_EQ(match.numTokens, length);
            for (auto const* block : match.blocks)
            {
                EXPECT_EQ(block->getPage(privateLc), nullptr);
                EXPECT_NE(block->getPage(globalLc), nullptr);
            }
        }
        second->suspend();
        ASSERT_TRUE(second->resume(mStream));
        second->close();
        first->close();
        manager->shutdown();
    }
}

TEST_F(ReplayCacheTest, ConcurrentRebaseDoesNotResurrectRetiredPages)
{
    auto const tokens = makeTokens(24);
    TokenSpan const tokenSpan{tokens.data(), static_cast<int>(tokens.size())};
    for (auto policy : {AttentionReusePolicy::REQUIRED, AttentionReusePolicy::PRIVATE, AttentionReusePolicy::OPTIONAL})
    {
        SCOPED_TRACE(static_cast<int>(policy));
        auto config = makeConfig();
        if (policy != AttentionReusePolicy::REQUIRED)
        {
            addWindow(config, 1, 8, policy);
        }
        auto manager = std::make_shared<KvCacheManager>(config);
        auto first = manager->createKvCache({}, tokenSpan, 1);
        auto second = manager->createKvCache({}, tokenSpan, 2);
        ASSERT_TRUE(first->resume(mStream));
        ASSERT_TRUE(second->resume(mStream));
        ASSERT_TRUE(first->resize(24));
        ASSERT_TRUE(second->resize(24));
        auto const globalLc = manager->getLayerGroupId(LayerId{0});
        EXPECT_NE(first->getBasePageIndices(globalLc)[5], second->getBasePageIndices(globalLc)[5]);
        first->commit(tokenSpan);
        second->commit(tokenSpan);
        EXPECT_EQ(first->getBasePageIndices(globalLc)[5], second->getBasePageIndices(globalLc)[5]);
        if (policy != AttentionReusePolicy::REQUIRED)
        {
            auto const privateLc = manager->getLayerGroupId(LayerId{1});
            EXPECT_EQ(first->getBasePageIndices(privateLc)[5] == second->getBasePageIndices(privateLc)[5],
                policy == AttentionReusePolicy::OPTIONAL);
        }
        second->close();
        first->close();
        manager->shutdown();
    }
}

TEST_F(ReplayCacheTest, ShrinkReleasesRetiredLocksBeforeVariantChanges)
{
    for (bool privateState : {false, true})
    {
        SCOPED_TRACE(privateState);
        auto config = makeConfig();
        addWindow(config, 1, 8, privateState ? AttentionReusePolicy::PRIVATE : AttentionReusePolicy::REQUIRED);
        auto manager = std::make_shared<KvCacheManager>(config);
        auto cache = manager->createKvCache({}, {}, 1);
        ASSERT_TRUE(cache->resume(mStream));
        ASSERT_TRUE(cache->resize(40));
        for (int capacity = 39; capacity >= 1; --capacity)
        {
            ASSERT_TRUE(cache->resize(capacity));
        }
        // Grow again to check that retired slots and page indices were returned.
        ASSERT_TRUE(cache->resize(40));
        cache->close();
        manager->shutdown();
    }
}

TEST_F(ReplayCacheTest, SsmSnapshotLocksDoNotUseAttentionPageIndices)
{
    auto config = makeConfig();
    config.commitMinSnapshot = true;
    SsmLayerConfig ssm;
    ssm.layerId = 1;
    ssm.buffers.push_back(BufferConfig{"state", 4096, std::nullopt});
    config.layers.emplace_back(ssm);
    auto manager = std::make_shared<KvCacheManager>(config);
    auto const tokens = makeTokens(12);
    TokenSpan const tokenSpan{tokens.data(), static_cast<int>(tokens.size())};
    auto first = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(first->resume(mStream));
    ASSERT_TRUE(first->resize(12));
    first->commit(tokenSpan);
    first->suspend();
    ASSERT_TRUE(first->resume(mStream));
    first->close();
    auto reused = manager->createKvCache({}, tokenSpan, 2);
    EXPECT_EQ(reused->numCommittedTokens(), 12);
    ASSERT_TRUE(reused->resume(mStream));
    reused->close();
    manager->shutdown();
}

TEST_F(ReplayCacheTest, ClosingScratchCacheReleasesEverySlotExactlyOnce)
{
    auto config = makeConfig();
    config.swaScratchReuse = SwaScratchReuseConfig{};
    auto layer = std::get<AttentionLayerConfig>(config.layers.front());
    layer.slidingWindowSize = 8;
    config.layers.clear();
    for (int i = 0; i < 16; ++i)
    {
        layer.layerId = LayerId{i};
        config.layers.emplace_back(layer);
    }
    auto manager = std::make_shared<KvCacheManager>(config);
    auto const initial = manager->storage().getStatistics();
    for (int i = 0; i < 4; ++i)
    {
        auto cache = manager->createKvCache({}, {}, i);
        ASSERT_TRUE(cache->resume(mStream));
        ASSERT_TRUE(cache->resize(64));
        cache->close();
        EXPECT_EQ(manager->storage().getStatistics().free, initial.free);
    }
    manager->shutdown();
}

TEST_F(ReplayCacheTest, OptionalAndPrivateHaveIndependentGroups)
{
    auto config = makeConfig();
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    auto& encoder = std::get<AttentionLayerConfig>(config.layers.front());
    encoder.slidingWindowSize = 8;
    encoder.reusePolicy = AttentionReusePolicy::OPTIONAL;
    auto decoder = encoder;
    decoder.layerId = 1;
    decoder.reusePolicy = AttentionReusePolicy::PRIVATE;
    config.layers.emplace_back(decoder);
    auto required = decoder;
    required.layerId = 2;
    required.reusePolicy = AttentionReusePolicy::REQUIRED;
    config.layers.emplace_back(required);
    config.initialPoolRatio = std::vector<float>{0.25F, 0.5F, 0.25F};
    auto manager = std::make_shared<KvCacheManager>(config);
    EXPECT_NE(manager->getLayerGroupId(LayerId{0}), manager->getLayerGroupId(LayerId{1}));
    for (auto const level : {kHotLevel, CacheLevel{1}})
    {
        for (int first = 0; first < 3; ++first)
        {
            for (int second = first + 1; second < 3; ++second)
            {
                EXPECT_NE(manager->storage().getPoolGroupIndex(level, manager->getLayerGroupId(LayerId{first})),
                    manager->storage().getPoolGroupIndex(level, manager->getLayerGroupId(LayerId{second})));
            }
        }
    }
    manager->shutdown();
}

TEST_F(ReplayCacheTest, OptionalWindowClaimsSurviveIndependentPageRemoval)
{

    struct WindowCase
    {
        int prefix;
        int tail;
        int window;
        int sinks;
        std::vector<int> ordinals;
    };

    // Minimal retention before a partial tail, a single page, and a window
    // spanning three blocks with or without a separate sink.
    for (auto const& c : {WindowCase{8, 1, 2, 0, {1}}, WindowCase{8, 0, 4, 0, {1}}, WindowCase{20, 3, 10, 0, {2, 3, 4}},
             WindowCase{20, 3, 6, 4, {0, 3, 4}}})
    {
        for (int victim : c.ordinals)
        {
            SCOPED_TRACE(::testing::Message() << "window=" << c.window << " sinks=" << c.sinks << " victim=" << victim);
            auto config = makeConfig();
            config.commitMinSnapshot = true;
            for (int id = 1; id <= 2; ++id)
            {
                addWindow(config, id, id == 1 ? c.window : 3, AttentionReusePolicy::OPTIONAL, id == 1 ? c.sinks : 0);
            }
            auto manager = std::make_shared<KvCacheManager>(config);
            auto const encoder = manager->getLayerGroupId(LayerId{1});
            auto const peer = manager->getLayerGroupId(LayerId{2});
            std::vector<LifeCycleId> const selected{encoder, peer};
            auto const tokens = makeTokens(c.prefix + c.tail);
            TokenSpan const prefix{tokens.data(), c.prefix};
            auto source = manager->createKvCache({}, {}, 1);
            ASSERT_TRUE(source->resume(mStream));
            ASSERT_TRUE(source->resize(c.prefix + c.tail));
            std::vector<int> original;
            for (int ordinal : c.ordinals)
            {
                original.push_back(source->getBasePageIndices(encoder)[ordinal]);
            }
            ASSERT_TRUE(source->resize(std::nullopt, c.prefix + c.tail));
            source->commit(TokenSpan{tokens.data(), static_cast<int>(tokens.size())});
            source->stopCommitting();
            auto reader = manager->createKvCache({}, prefix, 2);
            EXPECT_EQ(reader->numCommittedTokens(), c.prefix);
            ASSERT_TRUE(reader->reuseStatus()[encoder.value()].complete);
            std::vector<int> covered;
            for (auto const& [begin, end] : reader->reuseStatus()[encoder.value()].coverage)
            {
                for (int i = begin / 4; i < end / 4; ++i)
                {
                    covered.push_back(i);
                }
            }
            EXPECT_EQ(covered, c.ordinals);
            // Declining the claim allocates writable pages without replacing
            // the candidates available to a later reader.
            auto declined = manager->createKvCache({}, prefix, 3);
            ASSERT_TRUE(declined->resume(mStream, std::nullopt, std::vector<LifeCycleId>{}));
            for (size_t i = 0; i < original.size(); ++i)
            {
                EXPECT_NE(declined->getBasePageIndices(encoder)[c.ordinals[i]], original[i]);
            }
            declined->close();
            source->blocks()[BlockOrdinal{victim}].treeBlock->unlinkPage(encoder);
            source->close();
            auto missing = manager->createKvCache({}, prefix, 4);
            EXPECT_EQ(missing->numCommittedTokens(), c.prefix);
            EXPECT_FALSE(missing->reuseStatus()[encoder.value()].complete);
            EXPECT_TRUE(missing->reuseStatus()[encoder.value()].coverage.empty());
            EXPECT_TRUE(missing->reuseStatus()[peer.value()].complete);
            EXPECT_THROW(missing->resume(mStream, std::nullopt, selected), std::invalid_argument);
            EXPECT_FALSE(missing->isActive());
            ASSERT_TRUE(missing->resume(mStream, std::nullopt, std::vector<LifeCycleId>{peer}));
            missing->close();
            // The earlier claim owns every candidate even after tree unlink.
            manager->getAndResetIterationReusedBlocksByLevel();
            ASSERT_TRUE(reader->resume(mStream, std::nullopt, selected));
            reader->commitPendingStats();
            auto const reusedByLevel = manager->getAndResetIterationReusedBlocksByLevel();
            ASSERT_NE(reusedByLevel.find(encoder), reusedByLevel.end());
            EXPECT_EQ(reusedByLevel.at(encoder).full.at(kHotLevel), c.ordinals.size());
            EXPECT_EQ(countsByLevelTotal(reusedByLevel.at(encoder).partial), 0);
            reader->suspend();
            ASSERT_TRUE(reader->resume(mStream));
            reader->commitPendingStats();
            EXPECT_TRUE(manager->getAndResetIterationReusedBlocksByLevel().empty());
            for (size_t i = 0; i < original.size(); ++i)
            {
                EXPECT_EQ(reader->getBasePageIndices(encoder)[c.ordinals[i]], original[i]);
            }
            reader->close();
            EXPECT_EQ(manager->radixTree().match({}, prefix).numTokens, c.prefix);
            manager->shutdown();
        }
    }
}

TEST_F(ReplayCacheTest, IndependentOptionalDropPreservesMigrationPeer)
{
    auto config = makeConfig();
    for (int i = 1; i <= 2; ++i)
    {
        addWindow(config, i, i == 1 ? 4 : 2, AttentionReusePolicy::OPTIONAL);
    }
    auto manager = std::make_shared<KvCacheManager>(config);
    auto const encoder = manager->getLayerGroupId(LayerId{1});
    auto const peer = manager->getLayerGroupId(LayerId{2});
    auto const tokens = makeTokens(8);
    TokenSpan const span{tokens.data(), static_cast<int>(tokens.size())};
    auto source = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(source->resume(mStream));
    ASSERT_TRUE(source->resize(8));
    source->commit(span);
    auto block = source->blocks()[BlockOrdinal{1}].treeBlock;
    source->close();
    auto& storage = manager->storage();
    // Like a migration batch, hold a raw page after dequeueing it, without a
    // PageHolder. Dropping another group must preserve this page's tree entry.
    auto staged = block->getPage(peer)->sharedFromThis();
    storage.excludeFromEviction(*staged);
    storage.excludeFromEviction(*block->getPage(encoder));
    EXPECT_EQ(block->getPage(peer), staged.get());
    storage.scheduleForEviction(*staged);
    EXPECT_TRUE(staged->scheduledForEviction());
    // Keep cleanup safe when running the regression against the old runtime.
    if (staged->scheduledForEviction())
    {
        storage.excludeFromEviction(*staged);
    }
    staged.reset();
    EXPECT_EQ(manager->radixTree().match({}, span).numTokens, 8);
    block.reset();
    manager->shutdown();
}

TEST_F(ReplayCacheTest, DecliningOptionalClaimReleasesCapacityBeforeAllocation)
{
    auto config = makeConfig();
    config.maxUtilForResume = 1.0f;
    addWindow(config, 1, 4, AttentionReusePolicy::OPTIONAL);
    auto manager = std::make_shared<KvCacheManager>(config);
    auto const encoder = manager->getLayerGroupId(LayerId{1});
    auto const tokens = makeTokens(8);
    TokenSpan const span{tokens.data(), static_cast<int>(tokens.size())};
    auto source = manager->createKvCache({}, {}, 1);
    ASSERT_TRUE(source->resume(mStream));
    ASSERT_TRUE(source->resize(8));
    source->commit(span);
    source->close();
    auto reader = manager->createKvCache({}, span, 2);
    ASSERT_TRUE(reader->reuseStatus()[encoder.value()].complete);
    auto& storage = manager->storage();
    auto const group = storage.getPoolGroupIndex(kHotLevel, encoder);
    TypedVec<PoolGroupIndex, SlotCount> goals(storage.numPoolGroups(kHotLevel), SlotCount{0});
    goals[group] = storage.getStatistics(kHotLevel, group).total - 1;
    storage.prepareFreeSlots(kHotLevel, goals);
    auto reservations = storage.newSlotsForPoolGroup(kHotLevel, group, storage.getStatistics(kHotLevel, group).free);
    // No host tier, no free slots: the rejected candidate is the only capacity
    // available to the request's fresh writable Encoder page.
    EXPECT_TRUE(reader->resume(mStream, std::nullopt, std::vector<LifeCycleId>{}));
    EXPECT_EQ(reader->numCommittedTokens(), 8);
    reader->close();
    for (auto& slot : reservations)
    {
        storage.releaseSlot(encoder, kHotLevel, std::move(slot));
    }
    manager->shutdown();
}

TEST_F(ReplayCacheTest, OptionalCommitAcceptsGlobalRebase)
{
    auto config = makeConfig();
    addWindow(config, 1, 4, AttentionReusePolicy::OPTIONAL);
    auto manager = std::make_shared<KvCacheManager>(config);
    auto const encoder = manager->getLayerGroupId(LayerId{1});
    auto const tokens = makeTokens(8);
    TokenSpan const span{tokens.data(), static_cast<int>(tokens.size())};
    auto first = manager->createKvCache({}, {}, 1);
    auto second = manager->createKvCache({}, {}, 2);
    ASSERT_TRUE(first->resume(mStream));
    ASSERT_TRUE(second->resume(mStream));
    ASSERT_TRUE(first->resize(8));
    ASSERT_TRUE(second->resize(8));
    auto const slot = second->getBasePageIndices(encoder)[1];
    first->commit(span);
    auto block = first->blocks()[BlockOrdinal{1}].treeBlock;
    ASSERT_NE(block->getPage(encoder), nullptr);
    block->unlinkPage(encoder);
    second->commit(span);
    ASSERT_NE(block->getPage(encoder), nullptr);
    EXPECT_EQ(second->getBasePageIndices(encoder)[1], slot);
    second->close();
    first->close();
    block.reset();
    manager->shutdown();
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

TEST(KvCacheManagerV2StatsTest, PendingReuseByLevelRidesAlongWithScalarCounts)
{
    PendingStats pending;
    ReusedBlocksByLevel firstMatch;
    firstMatch.full = {2, 0, 1};
    firstMatch.partial = {0, 1, 0};
    EXPECT_TRUE(pending.recordReuse(LifeCycleId{0}, /*fullReusedBlocks=*/3, /*partialReusedBlocks=*/1, firstMatch));

    // A second match on the same life cycle accumulates element-wise.
    ReusedBlocksByLevel secondMatch;
    secondMatch.full = {1, 4, 0};
    secondMatch.partial = {0, 0, 0};
    EXPECT_TRUE(pending.recordReuse(LifeCycleId{0}, /*fullReusedBlocks=*/5, /*partialReusedBlocks=*/0, secondMatch));

    auto const& byLevel = pending.reusedBlocksByLevelByLifeCycle().at(LifeCycleId{0});
    EXPECT_EQ(byLevel.full.raw(), (std::vector<int64_t>{3, 4, 1}));
    EXPECT_EQ(byLevel.partial.raw(), (std::vector<int64_t>{0, 1, 0}));
    // The by-level split must agree with the scalar counters it rides along with.
    auto const& iteration = pending.iterationStatsByLifeCycle().at(LifeCycleId{0});
    EXPECT_EQ(std::accumulate(byLevel.full.begin(), byLevel.full.end(), int64_t{0}), iteration.iterFullReusedBlocks);
    EXPECT_EQ(
        std::accumulate(byLevel.partial.begin(), byLevel.partial.end(), int64_t{0}), iteration.iterPartialReusedBlocks);

    // Discarding the request drops the by-level split together with the scalar counters.
    pending.clear();
    EXPECT_TRUE(pending.reusedBlocksByLevelByLifeCycle().empty());
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
        auto block = addOrGetExistingBlock(previous, std::move(tokens), /*knownNoDigest=*/true);
        auto page = makeShared<CommittedPage>(
            &storage, block, lifeCycle, kHotLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
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

    auto primaryPeak = manager->getAndResetIterationPeakBlockStats(kHotLevel);
    auto secondaryPeak = manager->getAndResetIterationPeakBlockStats(CacheLevel{1});
    ASSERT_EQ(primaryPeak.size(), PoolGroupIndex{1});
    ASSERT_EQ(secondaryPeak.size(), PoolGroupIndex{1});
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].unavailable, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].evictable, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].unavailable, 1);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].evictable, 0);

    primaryPeak = manager->getAndResetIterationPeakBlockStats(kHotLevel);
    secondaryPeak = manager->getAndResetIterationPeakBlockStats(CacheLevel{1});
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].unavailable, 0);
    EXPECT_EQ(primaryPeak[PoolGroupIndex{0}].evictable, 0);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].available, 2);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].unavailable, 0);
    EXPECT_EQ(secondaryPeak[PoolGroupIndex{0}].evictable, 0);

    auto nextIntervalSlots = storage.newSlots(kHotLevel, oneSlot);
    manager->commitStats({});
    storage.releaseSlot(lifeCycle, kHotLevel, std::move(nextIntervalSlots[lifeCycle].front()));
    primaryPeak = manager->getAndResetIterationPeakBlockStats(kHotLevel);
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
    ASSERT_EQ(storage.getStatistics(kHotLevel).total, 2);
    ASSERT_EQ(storage.getStatistics(CacheLevel{1}).total, 2);

    int offloaded = 0;
    int onboarded = 0;
    int dropped = 0;
    MigrationRecorder const migrationRecorder
        = [&](std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots, CacheLevel srcLevel,
              CacheLevel dstLevel)
    {
        EXPECT_EQ(pages.size(), slots.size());
        if (srcLevel == kHotLevel && dstLevel == CacheLevel{1})
        {
            offloaded += static_cast<int>(pages.size());
        }
        else if (srcLevel == CacheLevel{1} && dstLevel == kHotLevel)
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
            auto block = addOrGetExistingBlock(previous, std::move(tokens), /*knownNoDigest=*/true);
            auto page = makeShared<CommittedPage>(
                &storage, block, lifeCycle, kHotLevel, static_cast<int>(block->tokens.size()), kPriorityDefault);
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
    PoolGroupIndex const hotPoolGroup = storage.getPoolGroupIndex(kHotLevel, lifeCycle);
    size_t const hotPageBytes = storage.slotSize(hotPoolGroup).at(PoolIndex{0});
    std::array<uint8_t, 2> const pagePatterns{0x3C, 0xA7};
    for (size_t index = 0; index < firstPages.size(); ++index)
    {
        MemAddress const address = std::get<MemAddress>(
            storage.slotAddress(kHotLevel, hotPoolGroup, firstPages[index]->slotId(), PoolIndex{0}));
        ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(address), pagePatterns[index], hotPageBytes), cudaSuccess);
    }

    auto temporarySlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 2);
    EXPECT_EQ(onboarded, 0);
    EXPECT_EQ(dropped, 0);
    for (auto& slot : temporarySlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, kHotLevel, std::move(slot));
    }

    auto cache = manager->createKvCache();
    for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{2}; ++ordinal)
    {
        auto const& page = firstPages[toSizeT(ordinal)];
        ASSERT_TRUE(page->scheduledForEviction());
        storage.excludeFromEviction(*page);
    }
    storage.batchedMigrate(kHotLevel, firstPages, migrationRecorder);
    EXPECT_EQ(onboarded, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (size_t index = 0; index < firstPages.size(); ++index)
    {
        MemAddress const address = std::get<MemAddress>(
            storage.slotAddress(kHotLevel, hotPoolGroup, firstPages[index]->slotId(), PoolIndex{0}));
        std::vector<uint8_t> restoredPage(hotPageBytes);
        ASSERT_EQ(cudaMemcpy(restoredPage.data(), reinterpret_cast<void const*>(address), hotPageBytes,
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_TRUE(std::all_of(restoredPage.begin(), restoredPage.end(),
            [expected = pagePatterns[index]](uint8_t byte) { return byte == expected; }));
    }
    for (auto const& page : firstPages)
    {
        storage.scheduleForEviction(*page);
    }

    temporarySlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 4);
    auto secondPages = makeCommittedPages(std::move(temporarySlots[lifeCycle]));
    (void) secondPages;
    firstPages.clear();

    auto finalSlots = storage.newGpuSlots(twoSlots, migrationRecorder, dropRecorder);
    EXPECT_EQ(offloaded, 6);
    EXPECT_EQ(onboarded, 2);
    EXPECT_EQ(dropped, 2);
    for (auto& slot : finalSlots[lifeCycle])
    {
        storage.releaseSlot(lifeCycle, kHotLevel, std::move(slot));
    }
    cache->close();
}

TEST(KvCacheManagerV2StatsTest, SuspendResumeIterationCountersTrackPreemptionOnly)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig());
    auto cache = manager->createKvCache({}, {}, 1);

    // A freshly-created cache starts SUSPENDED and is activated by its first resume().
    // That is an admission, not a preemption recovery, so neither counter moves.
    ASSERT_TRUE(cache->resume(stream));
    auto [admittedSuspended, admittedResumed] = manager->getAndResetIterationSuspendResumeStats();
    EXPECT_EQ(admittedSuspended, 0);
    EXPECT_EQ(admittedResumed, 0);

    // A real ACTIVE->SUSPENDED->ACTIVE cycle is a preemption and does count.
    cache->suspend();
    EXPECT_FALSE(cache->isActive());
    ASSERT_TRUE(cache->resume(stream));
    EXPECT_TRUE(cache->isActive());
    auto [suspended, resumed] = manager->getAndResetIterationSuspendResumeStats();
    EXPECT_EQ(suspended, 1);
    EXPECT_EQ(resumed, 1);

    // The drain resets both counters for the next iteration window.
    auto [drainedSuspended, drainedResumed] = manager->getAndResetIterationSuspendResumeStats();
    EXPECT_EQ(drainedSuspended, 0);
    EXPECT_EQ(drainedResumed, 0);

    cache->close();
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(KvCacheManagerV2StatsTest, DisabledStatsSuppressSuspendResumeCounters)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeConfig(false));
    auto cache = manager->createKvCache({}, {}, 1);

    ASSERT_TRUE(cache->resume(stream));
    cache->suspend();
    ASSERT_TRUE(cache->resume(stream));

    auto [suspended, resumed] = manager->getAndResetIterationSuspendResumeStats();
    EXPECT_EQ(suspended, 0);
    EXPECT_EQ(resumed, 0);

    cache->close();
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Hybrid attention + SSM page-movement statistics.
//
// Iteration statistics are keyed by life cycle and must report recurrent (SSM) page
// movement alongside attention movement, otherwise KDA recurrent-state offload, onboard
// and drop are invisible to callers. The global cache-hit counters are the deliberate
// exception: they stay attention-only.
//
// The tiers below give the attention life cycle 4 GPU and 2 host slots of 1 MiB, and the
// SSM life cycle 1 GPU and 1 host slot of 2 MiB, so a second sequence evicts the first and
// a second eviction round overflows the host pool.
namespace
{
constexpr int kHybridBlocks = 3;
constexpr CacheLevel kHybridHostLevel{1};

int64_t slotBytesFor(StorageManager const& storage, CacheLevel level, LifeCycleId lifeCycle)
{
    int64_t bytes = 0;
    for (size_t const size : storage.slotSize(storage.getPoolGroupIndex(level, lifeCycle)))
    {
        bytes += static_cast<int64_t>(size);
    }
    return bytes;
}

std::vector<TokenIdExt> makeTokens(KvCacheManager const& manager, int firstToken)
{
    std::vector<TokenIdExt> tokens;
    for (int offset = 0; offset < kHybridBlocks * manager.tokensPerBlock(); ++offset)
    {
        tokens.emplace_back(TokenId{firstToken + offset});
    }
    return tokens;
}

// Fill a sequence, park it, then start a second one that needs the same slots. Returns the
// still-open second sequence so the caller can close it to trigger the onboard.
std::pair<std::shared_ptr<KvCache>, std::shared_ptr<KvCache>> evictFirstSequence(
    KvCacheManager& manager, cudaStream_t stream, int firstToken)
{
    auto const tokens = makeTokens(manager, firstToken);
    auto first = manager.createKvCache();
    EXPECT_TRUE(first->resume(reinterpret_cast<CUstream>(stream)));
    EXPECT_TRUE(first->resize(static_cast<int>(tokens.size())));
    first->commit(toSpan(tokens));
    first->suspend();

    auto second = manager.createKvCache();
    EXPECT_TRUE(second->resume(reinterpret_cast<CUstream>(stream)));
    EXPECT_TRUE(second->resize(static_cast<int>(tokens.size())));
    return {std::move(first), std::move(second)};
}
} // namespace

TEST(KvCacheManagerV2StatsTest, OffloadAndOnboardAreRecordedForAttentionAndSsmLifeCycles)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeHybridTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const attention{0};
    LifeCycleId const ssm{1};
    ASSERT_TRUE(std::holds_alternative<AttnLifeCycle>(manager->lifeCycles().getLifeCycle(attention)));
    ASSERT_FALSE(std::holds_alternative<AttnLifeCycle>(manager->lifeCycles().getLifeCycle(ssm)));

    manager->getAndResetIterationStats();
    auto [first, second] = evictFirstSequence(*manager, stream, 0);

    auto const offload = manager->getAndResetIterationStats();
    ASSERT_EQ(offload.size(), 2) << "both life cycles must report offload";
    for (LifeCycleId const lifeCycle : {attention, ssm})
    {
        auto const& stats = offload.at(lifeCycle);
        EXPECT_GT(stats.iterOffloadBlocks, 0) << "life cycle " << lifeCycle.value();
        EXPECT_EQ(stats.iterOffloadBytes, stats.iterOffloadBlocks * slotBytesFor(storage, kHotLevel, lifeCycle));
    }

    second->close();
    ASSERT_TRUE(first->resume(reinterpret_cast<CUstream>(stream)));

    auto const onboard = manager->getAndResetIterationStats();
    ASSERT_EQ(onboard.size(), 2) << "both life cycles must report onboard";
    for (LifeCycleId const lifeCycle : {attention, ssm})
    {
        auto const& stats = onboard.at(lifeCycle);
        EXPECT_GT(stats.iterOnboardBlocks, 0) << "life cycle " << lifeCycle.value();
        EXPECT_EQ(stats.iterOnboardBytes, stats.iterOnboardBlocks * slotBytesFor(storage, kHotLevel, lifeCycle));
    }

    first->close();
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(KvCacheManagerV2StatsTest, SsmOnboardLeavesGlobalAllocCountersToAttention)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeHybridTieredConfig());
    LifeCycleId const attention{0};
    LifeCycleId const ssm{1};

    manager->getAndResetIterationStats();
    auto [first, second] = evictFirstSequence(*manager, stream, 0);
    manager->getAndResetIterationStats();

    second->close();
    auto const allocTotalBefore = manager->getCommittedStats().allocTotalBlocks;
    auto const allocNewBefore = manager->getCommittedStats().allocNewBlocks;
    ASSERT_TRUE(first->resume(reinterpret_cast<CUstream>(stream)));
    auto const allocTotalDelta = manager->getCommittedStats().allocTotalBlocks - allocTotalBefore;
    auto const allocNewDelta = manager->getCommittedStats().allocNewBlocks - allocNewBefore;

    auto const onboard = manager->getAndResetIterationStats();
    ASSERT_EQ(onboard.size(), 2);
    auto const attentionOnboard = onboard.at(attention).iterAllocTotalBlocks;
    auto const ssmOnboard = onboard.at(ssm).iterAllocTotalBlocks;
    // Both life cycles onboard, so a global delta equal to the attention share alone is
    // only possible if the SSM share was excluded.
    ASSERT_GT(attentionOnboard, 0);
    ASSERT_GT(ssmOnboard, 0);
    EXPECT_EQ(allocTotalDelta, attentionOnboard);
    EXPECT_EQ(allocNewDelta, attentionOnboard);

    first->close();
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(KvCacheManagerV2StatsTest, HostDropIsRecordedForAttentionAndSsmLifeCycles)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeHybridTieredConfig());
    auto& storage = manager->storage();
    LifeCycleId const attention{0};
    LifeCycleId const ssm{1};

    // First round fills the host pools.
    auto [first, second] = evictFirstSequence(*manager, stream, 0);
    second->close();
    first->close();

    // Second round uses disjoint tokens, so nothing is reused and the host pools overflow.
    manager->getAndResetIterationStats();
    auto [third, fourth] = evictFirstSequence(*manager, stream, 1000);

    auto const dropped = manager->getAndResetIterationStats();
    ASSERT_EQ(dropped.size(), 2) << "both life cycles must report host drops";
    for (LifeCycleId const lifeCycle : {attention, ssm})
    {
        auto const& stats = dropped.at(lifeCycle);
        EXPECT_GT(stats.iterHostDroppedBlocks, 0) << "life cycle " << lifeCycle.value();
        EXPECT_EQ(stats.iterHostDroppedBytes,
            stats.iterHostDroppedBlocks * slotBytesFor(storage, kHybridHostLevel, lifeCycle));
    }

    fourth->close();
    third->close();
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
