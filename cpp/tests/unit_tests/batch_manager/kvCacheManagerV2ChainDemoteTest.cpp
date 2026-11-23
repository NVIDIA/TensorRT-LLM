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

// Evict-first demotion of idle committed chains: the eviction key (Page::evictionPriority()), the
// StorageManager::demote()/undemote() transitions, and ChainDemoteHandle planning and ownership rules.

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/common.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/exceptions.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/page.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

constexpr int kTokensPerBlock = 4;

//! One full-attention layer with 4 KiB pages, on a GPU tier and an optional host tier.
KVCacheManagerConfig fullAttentionConfig(size_t gpuQuota = 4UL << 20, size_t hostQuota = 0)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = kTokensPerBlock;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{gpuQuota});
    if (hostQuota > 0)
    {
        config.cacheTiers.emplace_back(HostCacheTierConfig{hostQuota});
    }
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(layer));
    return config;
}

//! A full-attention and a sliding-window layer with different page sizes, so they use two GPU pool groups.
KVCacheManagerConfig twoPoolGroupConfig()
{
    auto config = fullAttentionConfig();
    AttentionLayerConfig swa;
    swa.layerId = 1;
    swa.slidingWindowSize = 8;
    swa.buffers.push_back(BufferConfig{"key", 8192, std::nullopt});
    config.layers.emplace_back(std::move(swa));
    return config;
}

//! A committed page in its own root-level block, scheduled for eviction. Mirrors the ColdPage test helper.
SharedPtr<CommittedPage> makeCommittedPage(
    KvCacheManager& manager, CacheLevel level, Priority priority, int tokenBase, LifeCycleId lifeCycle = LifeCycleId{0})
{
    auto& storage = manager.storage();
    TypedVec<LifeCycleId, SlotCount> counts(storage.numLifeCycles(), 0);
    counts[lifeCycle] = 1;
    auto slots = storage.newSlots(level, counts);
    RootBlock& root = manager.radixTree().addOrGetExisting({});
    std::vector<TokenIdExt> tokens;
    for (int token = 0; token < manager.tokensPerBlock(); ++token)
    {
        tokens.emplace_back(TokenId{tokenBase + token});
    }
    auto block = addOrGetExistingBlock(&root, std::move(tokens), /*knownNoDigest=*/true);
    auto page = makeShared<CommittedPage>(
        &storage, block, lifeCycle, level, static_cast<int>(block->tokens.size()), priority);
    page->setSlot(slots[lifeCycle].front());
    block->storage[lifeCycle] = page.get();
    storage.scheduleForEviction(*page);
    return page;
}

//! Evict `count` pages of `lifeCycle`'s pool group at `level` (drop at the last level, fall otherwise).
void evict(StorageManager& storage, CacheLevel level, int count, LifeCycleId lifeCycle = LifeCycleId{0})
{
    TypedVec<PoolGroupIndex, SlotCount> numPages(storage.numPoolGroups(level), 0);
    numPages[storage.getPoolGroupIndex(level, lifeCycle)] = count;
    storage.forceEvict(level, numPages);
}

//! Indices of the expired references.
std::set<int> expiredIndices(std::vector<WeakPtr<CommittedPage>> const& pages)
{
    std::set<int> result;
    for (int i = 0; i < static_cast<int>(pages.size()); ++i)
    {
        if (pages[i].expired())
        {
            result.insert(i);
        }
    }
    return result;
}

class KvCacheManagerV2ChainDemoteTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), cudaSuccess);
    }

    void TearDown() override
    {
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    CUstream stream() const
    {
        return reinterpret_cast<CUstream>(mStream);
    }

    static std::vector<TokenIdExt> makeTokens(int base, int count)
    {
        std::vector<TokenIdExt> tokens;
        tokens.reserve(count);
        for (int i = 0; i < count; ++i)
        {
            tokens.emplace_back(TokenId{base + i});
        }
        return tokens;
    }

    static std::vector<TokenIdExt> concat(std::vector<TokenIdExt> lhs, std::vector<TokenIdExt> const& rhs)
    {
        lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        return lhs;
    }

    //! Commit `tokens` through a new cache, reusing whatever prefix the tree already holds. The cache is left
    //! open after stopCommitting(), ready for planning.
    std::shared_ptr<KvCache> commitOpen(KvCacheManager& manager, std::vector<TokenIdExt> const& tokens) const
    {
        auto cache = manager.createKvCache({}, toSpan(tokens));
        EXPECT_TRUE(cache->resume(stream()));
        int const size = static_cast<int>(tokens.size());
        int const reused = cache->numCommittedTokens();
        EXPECT_TRUE(cache->resize(size));
        if (reused < size)
        {
            cache->commit(TokenSpan{tokens.data() + reused, size - reused});
        }
        cache->stopCommitting();
        return cache;
    }

    //! The full-attention page of every block matched by `tokens`.
    static std::vector<CommittedPage*> chainPages(KvCacheManager& manager, std::vector<TokenIdExt> const& tokens)
    {
        auto const apiLock = manager.lockExclusive();
        auto const match = manager.matchReuse({}, toSpan(tokens), /*knownNoDigest=*/true);
        std::vector<CommittedPage*> pages;
        for (Block* block : match.blocks)
        {
            pages.push_back(block->getPage(LifeCycleId{0}));
        }
        return pages;
    }

    cudaStream_t mStream{};
};

TEST_F(KvCacheManagerV2ChainDemoteTest, EvictionKeyFollowsDemotion)
{
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    std::vector<WeakPtr<CommittedPage>> pages;
    for (int i = 0; i < 6; ++i)
    {
        pages.emplace_back(makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 100 * i));
    }

    EXPECT_TRUE(storage.demote(*pages[3].lock()));
    EXPECT_TRUE(storage.demote(*pages[1].lock()));
    EXPECT_FALSE(storage.demote(*pages[1].lock())); // already demoted
    EXPECT_EQ(pages[1].lock()->evictionPriority(), kPriorityDemoted);
    EXPECT_EQ(pages[1].lock()->priority, kPriorityDefault);
    EXPECT_EQ(pages[0].lock()->evictionPriority(), kPriorityDefault);

    // Removed under the demoted key and pushed back under its own: page 4 moves to the back of its list.
    EXPECT_TRUE(storage.demote(*pages[4].lock()));
    storage.undemote(*pages[4].lock());
    EXPECT_FALSE(pages[4].lock()->demoted);
    EXPECT_TRUE(pages[4].lock()->scheduledForEviction());

    // Demoted pages first, FIFO by demotion time, then LRU for the rest.
    std::set<int> dropped;
    for (int expected : {3, 1, 0, 2, 5, 4})
    {
        evict(storage, kHotLevel, 1);
        dropped.insert(expected);
        EXPECT_EQ(expiredIndices(pages), dropped) << "expected page " << expected;
    }
    EXPECT_EQ(manager->demotionStats().destroyedDemotedPages, 2U);
    EXPECT_EQ(manager->demotionStats().undemotedPages, 1U);
}

TEST_F(KvCacheManagerV2ChainDemoteTest, HoldUndemotesAndRequeuesAtBack)
{
    // Non-last level: a HELD page stays scheduled, so the hold must clear the mark and re-queue it.
    {
        auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig(4UL << 20, 4UL << 20));
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        auto a = makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 0);
        auto b = makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 100);
        auto c = makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 200);
        ASSERT_TRUE(storage.demote(*a));

        auto holder = a->hold();
        EXPECT_FALSE(a->demoted);
        EXPECT_EQ(a->status(), PageStatus::HELD);
        EXPECT_TRUE(a->scheduledForEviction());
        EXPECT_EQ(manager->demotionStats().undemotedPages, 1U);

        // a is now behind b and c.
        evict(storage, kHotLevel, 1);
        EXPECT_EQ(b->cacheLevel, CacheLevel{1});
        EXPECT_EQ(a->cacheLevel, kHotLevel);
        EXPECT_EQ(c->cacheLevel, kHotLevel);
        evict(storage, kHotLevel, 1);
        EXPECT_EQ(c->cacheLevel, CacheLevel{1});
        EXPECT_EQ(a->cacheLevel, kHotLevel);
        holder.reset();
    }
    // Last level: a HELD page is unscheduled while held and goes to the back of its own list on release.
    {
        auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig());
        auto const apiLock = manager->lockExclusive();
        auto& storage = manager->storage();
        std::vector<WeakPtr<CommittedPage>> pages;
        auto a = makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 0);
        pages.emplace_back(a);
        pages.emplace_back(makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 100));
        ASSERT_TRUE(storage.demote(*a));

        auto holder = a->hold();
        EXPECT_FALSE(a->demoted);
        EXPECT_FALSE(a->scheduledForEviction());
        holder.reset();
        EXPECT_TRUE(a->scheduledForEviction());
        EXPECT_FALSE(a->demoted);
        a.reset();

        evict(storage, kHotLevel, 1);
        EXPECT_EQ(expiredIndices(pages), std::set<int>{1});
        evict(storage, kHotLevel, 1);
        EXPECT_EQ(expiredIndices(pages), (std::set<int>{0, 1}));
        EXPECT_EQ(manager->demotionStats().destroyedDemotedPages, 0U);
    }
}

TEST_F(KvCacheManagerV2ChainDemoteTest, CommitOfExistingBlockUndemotes)
{
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig());
    auto const tokens = makeTokens(0, 2 * kTokensPerBlock);

    auto first = commitOpen(*manager, tokens);
    auto handle = first->planCommittedChainDemote(0);
    ASSERT_TRUE(handle != nullptr);
    first->close();
    EXPECT_EQ(handle->demote().demotedPages, 2);
    for (auto* page : chainPages(*manager, tokens))
    {
        EXPECT_TRUE(page->demoted);
    }

    // A fresh cache commits the same tokens itself, without an input match: _commitBlock finds existing blocks.
    auto second = manager->createKvCache();
    ASSERT_TRUE(second->resume(stream()));
    ASSERT_TRUE(second->resize(static_cast<int>(tokens.size())));
    second->commit(toSpan(tokens));
    for (auto* page : chainPages(*manager, tokens))
    {
        EXPECT_FALSE(page->demoted);
    }
    EXPECT_EQ(manager->demotionStats().undemotedPages, 2U);
    second->close();
}

TEST_F(KvCacheManagerV2ChainDemoteTest, DemotedPageFallsToHostStaysFirst)
{
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig(4UL << 20, 4UL << 20));
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    CacheLevel const hostLevel{1};
    std::vector<WeakPtr<CommittedPage>> pages;
    for (int i = 0; i < 3; ++i)
    {
        pages.emplace_back(makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 100 * i));
    }
    ASSERT_TRUE(storage.demote(*pages[2].lock()));

    // The demoted page is the first GPU victim and keeps its mark at the host.
    evict(storage, kHotLevel, 1);
    EXPECT_EQ(pages[2].lock()->cacheLevel, hostLevel);
    EXPECT_TRUE(pages[2].lock()->demoted);
    EXPECT_TRUE(pages[2].lock()->scheduledForEviction());
    EXPECT_EQ(pages[0].lock()->cacheLevel, kHotLevel);
    EXPECT_EQ(pages[1].lock()->cacheLevel, kHotLevel);

    // The others fall after it; the host still evicts the demoted page first.
    evict(storage, kHotLevel, 2);
    EXPECT_EQ(pages[0].lock()->cacheLevel, hostLevel);
    EXPECT_EQ(pages[1].lock()->cacheLevel, hostLevel);
    evict(storage, hostLevel, 1);
    EXPECT_EQ(expiredIndices(pages), std::set<int>{2});
    EXPECT_EQ(manager->demotionStats().destroyedDemotedPages, 1U);
}

TEST_F(KvCacheManagerV2ChainDemoteTest, RollbackKeepsDemotedKey)
{
    auto manager = std::make_shared<KvCacheManager>(twoPoolGroupConfig());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    ASSERT_EQ(storage.numPoolGroups(kHotLevel), PoolGroupIndex{2});
    // Pages go to the life cycle of pool group 0, which evict() visits first.
    LifeCycleId const first
        = storage.getPoolGroupIndex(kHotLevel, LifeCycleId{0}) == PoolGroupIndex{0} ? LifeCycleId{0} : LifeCycleId{1};
    LifeCycleId const second = first == LifeCycleId{0} ? LifeCycleId{1} : LifeCycleId{0};

    std::vector<WeakPtr<CommittedPage>> pages;
    for (int i = 0; i < 3; ++i)
    {
        pages.emplace_back(makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 100 * i, first));
    }
    ASSERT_TRUE(storage.demote(*pages[0].lock()));
    ASSERT_TRUE(storage.demote(*pages[1].lock()));

    // Group 0 gives up page 0, then group 1 has nothing to evict: the rollback re-queues page 0 at the front
    // of the list its eviction key names.
    TypedVec<PoolGroupIndex, SlotCount> numPages(storage.numPoolGroups(kHotLevel), 0);
    numPages[storage.getPoolGroupIndex(kHotLevel, first)] = 1;
    numPages[storage.getPoolGroupIndex(kHotLevel, second)] = 1;
    EXPECT_THROW(storage.forceEvict(kHotLevel, numPages), OutOfPagesError);
    EXPECT_TRUE(pages[0].lock()->demoted);
    EXPECT_TRUE(pages[0].lock()->scheduledForEviction());
    EXPECT_TRUE(expiredIndices(pages).empty());

    // Removing page 0 must find it in the demoted list; it then goes behind page 2.
    storage.undemote(*pages[0].lock());
    evict(storage, kHotLevel, 1, first);
    EXPECT_EQ(expiredIndices(pages), std::set<int>{1});
    evict(storage, kHotLevel, 1, first);
    EXPECT_EQ(expiredIndices(pages), (std::set<int>{1, 2}));
    evict(storage, kHotLevel, 1, first);
    EXPECT_EQ(expiredIndices(pages), (std::set<int>{0, 1, 2}));
}

TEST_F(KvCacheManagerV2ChainDemoteTest, ChainDemoteSkipsSharedAndHeld)
{
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig());
    auto const parent = makeTokens(0, 2 * kTokensPerBlock);
    auto const child = concat(parent, makeTokens(1000, 2 * kTokensPerBlock));
    auto const sibling = concat(concat(parent, makeTokens(1000, kTokensPerBlock)), makeTokens(2000, kTokensPerBlock));
    commitOpen(*manager, parent)->close();

    // The child plans from the end of its first-request match, so the parent's blocks are never planned.
    auto childCache = commitOpen(*manager, child);
    EXPECT_EQ(childCache->numCommittedTokens(), static_cast<int>(child.size()));
    auto handle = childCache->planCommittedChainDemote(static_cast<int>(parent.size()) / kTokensPerBlock);
    ASSERT_TRUE(handle != nullptr);
    EXPECT_EQ(handle->numRefs(), 2);
    childCache->close();

    // A sibling continues the child's first block, which then has a child outside the plan.
    commitOpen(*manager, sibling)->close();
    auto const result = handle->demote();
    EXPECT_FALSE(handle->valid());
    EXPECT_EQ(result.planBlocks, 2);
    EXPECT_EQ(result.demotedBlocks, 1);
    EXPECT_EQ(result.demotedPages, 1);
    EXPECT_EQ(result.demotedPagesHot, 1);
    EXPECT_EQ(result.keptSharedBlocks, 1);
    EXPECT_EQ(result.keptHeldBlocks, 0);
    auto const childPages = chainPages(*manager, child);
    ASSERT_EQ(childPages.size(), 4U);
    EXPECT_FALSE(childPages[0]->demoted);
    EXPECT_FALSE(childPages[1]->demoted);
    EXPECT_FALSE(childPages[2]->demoted);
    EXPECT_TRUE(childPages[3]->demoted);
    auto const siblingPages = chainPages(*manager, sibling);
    ASSERT_EQ(siblingPages.size(), 4U);
    EXPECT_FALSE(siblingPages[3]->demoted);

    // An open cache that matched a planned chain holds its pages: nothing is demoted.
    auto const other = concat(parent, makeTokens(3000, 2 * kTokensPerBlock));
    auto otherCache = commitOpen(*manager, other);
    auto otherHandle = otherCache->planCommittedChainDemote(static_cast<int>(parent.size()) / kTokensPerBlock);
    ASSERT_TRUE(otherHandle != nullptr);
    otherCache->close();
    auto reader = manager->createKvCache({}, toSpan(other));
    ASSERT_TRUE(reader->resume(stream()));
    ASSERT_GT(reader->numCommittedTokens(), static_cast<int>(parent.size()));
    auto const held = otherHandle->demote();
    EXPECT_EQ(held.planBlocks, 2);
    EXPECT_EQ(held.demotedPages, 0);
    EXPECT_GE(held.keptHeldBlocks, 1);
    EXPECT_EQ(held.keptHeldBlocks + held.keptSharedBlocks, 2);
    reader->close();
}

TEST_F(KvCacheManagerV2ChainDemoteTest, HandleDestructorAndReleaseHaveNoEffect)
{
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig());
    auto const tokens = makeTokens(0, 3 * kTokensPerBlock);
    auto cache = commitOpen(*manager, tokens);
    {
        auto dropped = cache->planCommittedChainDemote(0);
        ASSERT_TRUE(dropped != nullptr);
        EXPECT_TRUE(dropped->valid());
        EXPECT_EQ(dropped->numRefs(), 3);
    }
    auto released = cache->planCommittedChainDemote(0);
    ASSERT_TRUE(released != nullptr);
    cache->close();
    released->release();
    EXPECT_FALSE(released->valid());
    EXPECT_EQ(released->numRefs(), 0);
    EXPECT_THROW(released->release(), LogicError);
    EXPECT_THROW(released->demote(), LogicError);

    auto const pages = chainPages(*manager, tokens);
    ASSERT_EQ(pages.size(), 3U);
    for (auto* page : pages)
    {
        EXPECT_FALSE(page->demoted);
        EXPECT_EQ(page->plannedDropCount, 0);
        EXPECT_TRUE(page->scheduledForEviction());
    }
    EXPECT_EQ(manager->demotionStats().destroyedDemotedPages, 0U);
    EXPECT_EQ(manager->demotionStats().undemotedPages, 0U);

    // Planner preconditions: stopCommitting() first, and nothing past the committed blocks.
    auto open = manager->createKvCache();
    ASSERT_TRUE(open->resume(stream()));
    ASSERT_TRUE(open->resize(kTokensPerBlock));
    open->commit(TokenSpan{tokens.data(), kTokensPerBlock});
    EXPECT_THROW(open->planCommittedChainDemote(0), LogicError);
    open->stopCommitting();
    EXPECT_TRUE(open->planCommittedChainDemote(5) == nullptr);
    open->close();
}

TEST_F(KvCacheManagerV2ChainDemoteTest, AbsorbPrunesExpired)
{
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig());
    auto const turn1 = makeTokens(0, 2 * kTokensPerBlock);
    auto first = commitOpen(*manager, turn1);
    auto h1 = first->planCommittedChainDemote(0);
    ASSERT_TRUE(h1 != nullptr);
    first->close();

    // Turn 1's pages are the only evictable ones: drop them, so h1 holds two expired references.
    {
        auto const apiLock = manager->lockExclusive();
        evict(manager->storage(), kHotLevel, 2);
    }
    EXPECT_EQ(manager->probeReuse({}, toSpan(turn1), /*knownNoDigest=*/true), 0);

    auto const turn2 = makeTokens(500, 2 * kTokensPerBlock);
    auto second = commitOpen(*manager, turn2);
    auto h2 = second->planCommittedChainDemote(0);
    ASSERT_TRUE(h2 != nullptr);
    second->close();

    EXPECT_EQ(h1->numRefs(), 2);
    h1->absorb(*h2);
    EXPECT_EQ(h1->numRefs(), 2);
    EXPECT_FALSE(h2->valid());
    EXPECT_THROW(h1->absorb(*h1), LogicError);
    EXPECT_THROW(h1->absorb(*h2), LogicError);
    EXPECT_THROW(h2->absorb(*h1), LogicError);

    auto const result = h1->demote();
    EXPECT_EQ(result.planBlocks, 2);
    EXPECT_EQ(result.demotedPages, 2);
    EXPECT_EQ(result.keptSharedBlocks, 0);
}

TEST_F(KvCacheManagerV2ChainDemoteTest, DeepChain30kBlocksIterative)
{
    constexpr int kNumBlocks = 30000;
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig(256UL << 20));
    auto const tokens = makeTokens(0, kNumBlocks * kTokensPerBlock);
    auto cache = commitOpen(*manager, tokens);
    auto handle = cache->planCommittedChainDemote(0);
    ASSERT_TRUE(handle != nullptr);
    EXPECT_EQ(handle->numRefs(), kNumBlocks);
    cache->close();

    auto const result = handle->demote();
    EXPECT_EQ(result.planBlocks, kNumBlocks);
    EXPECT_EQ(result.demotedBlocks, kNumBlocks);
    EXPECT_EQ(result.demotedPages, kNumBlocks);
    EXPECT_EQ(result.keptSharedBlocks, 0);
    EXPECT_EQ(result.keptHeldBlocks, 0);

    // The tail was queued first, so each drop removes a leaf: the chain is torn down without recursing
    // through a deep subtree.
    {
        auto const apiLock = manager->lockExclusive();
        evict(manager->storage(), kHotLevel, 1);
    }
    EXPECT_EQ(manager->probeReuse({}, toSpan(tokens), /*knownNoDigest=*/true), (kNumBlocks - 1) * kTokensPerBlock);
    {
        auto const apiLock = manager->lockExclusive();
        evict(manager->storage(), kHotLevel, kNumBlocks - 1);
    }
    EXPECT_EQ(manager->probeReuse({}, toSpan(tokens), /*knownNoDigest=*/true), 0);
    EXPECT_EQ(manager->demotionStats().destroyedDemotedPages, static_cast<uint64_t>(kNumBlocks));
}

TEST_F(KvCacheManagerV2ChainDemoteTest, DestroyedAndUndemotedCounters)
{
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    std::vector<WeakPtr<CommittedPage>> pages;
    for (int i = 0; i < 3; ++i)
    {
        pages.emplace_back(makeCommittedPage(*manager, kHotLevel, kPriorityDefault, 100 * i));
    }
    ASSERT_TRUE(storage.demote(*pages[0].lock()));
    ASSERT_TRUE(storage.demote(*pages[1].lock()));
    storage.undemote(*pages[2].lock()); // not demoted: no-op
    EXPECT_EQ(manager->demotionStats().undemotedPages, 0U);

    {
        auto holder = pages[0].lock()->hold();
        EXPECT_EQ(manager->demotionStats().undemotedPages, 1U);
    }
    evict(storage, kHotLevel, 1);
    EXPECT_EQ(expiredIndices(pages), std::set<int>{1});
    EXPECT_EQ(manager->demotionStats().destroyedDemotedPages, 1U);
    evict(storage, kHotLevel, 2);
    EXPECT_EQ(expiredIndices(pages), (std::set<int>{0, 1, 2}));
    EXPECT_EQ(manager->demotionStats().destroyedDemotedPages, 1U);
    EXPECT_EQ(manager->demotionStats().undemotedPages, 1U);
}

TEST_F(KvCacheManagerV2ChainDemoteTest, OffPathPopOrderUnchanged)
{
    auto manager = std::make_shared<KvCacheManager>(fullAttentionConfig());
    auto const apiLock = manager->lockExclusive();
    auto& storage = manager->storage();
    std::vector<WeakPtr<CommittedPage>> pages;
    std::vector<Priority> const priorities{kPriorityDefault, kPriorityMin, 70, kPriorityDefault, kPriorityMin};
    for (int i = 0; i < static_cast<int>(priorities.size()); ++i)
    {
        pages.emplace_back(makeCommittedPage(*manager, kHotLevel, priorities[i], 100 * i));
        EXPECT_EQ(pages.back().lock()->evictionPriority(), priorities[i]);
    }

    // Lowest priority first, LRU within a priority: the order without the feature.
    std::set<int> dropped;
    for (int expected : {1, 4, 0, 3, 2})
    {
        evict(storage, kHotLevel, 1);
        dropped.insert(expected);
        EXPECT_EQ(expiredIndices(pages), dropped) << "expected page " << expected;
    }
    EXPECT_EQ(manager->demotionStats().destroyedDemotedPages, 0U);
    EXPECT_EQ(manager->demotionStats().undemotedPages, 0U);
}

} // namespace
