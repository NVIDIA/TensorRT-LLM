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
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/exceptions.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/localityDomain.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

std::vector<SlotCount> freeSlotsPerDomain(StorageManager const& storage)
{
    std::vector<SlotCount> freeSlots;
    for (LocalityDomainId localityDomainId{0}; localityDomainId < storage.numLocalityDomains(); ++localityDomainId)
    {
        SlotCount free = 0;
        for (PoolGroupIndex pgIdx{0}; pgIdx < storage.numPoolGroups(kHotLevel); ++pgIdx)
        {
            free += storage.getStatistics(kHotLevel, pgIdx, localityDomainId).free;
        }
        freeSlots.push_back(free);
    }
    return freeSlots;
}

// ---------------------------------------------------------------------------
// SlotIdMap / SlotAllocator (no GPU)
// ---------------------------------------------------------------------------

TEST(KvCacheManagerV2LocalityTest, SlotIdMapRoundTripsAndRejectsForeignChunks)
{
    SlotIdMap map{SlotCount{4}};
    map.append(2);
    map.append(0);

    EXPECT_EQ(map.slotId(0), SlotId{8});
    EXPECT_EQ(map.slotId(3), SlotId{11});
    EXPECT_EQ(map.slotId(4), SlotId{0});
    EXPECT_EQ(map.localIndex(SlotId{9}), SlotCount{1});
    EXPECT_EQ(map.localIndex(SlotId{3}), SlotCount{7});
    EXPECT_FALSE(map.localIndex(SlotId{4}).has_value());
    EXPECT_FALSE(map.localIndex(SlotId{-1}).has_value());

    EXPECT_EQ(map.pop(), 0U);
    EXPECT_FALSE(map.localIndex(SlotId{0}).has_value());
}

TEST(KvCacheManagerV2LocalityTest, MappedAllocatorsIssueDisjointGlobalIdsAndStampTheDomain)
{
    SlotIdMap map0{SlotCount{4}};
    SlotIdMap map1{SlotCount{4}};
    map0.append(0);
    map1.append(1);
    SlotAllocator alloc0{SlotCount{4}, &map0, LocalityDomainId{0}};
    SlotAllocator alloc1{SlotCount{4}, &map1, LocalityDomainId{1}};

    std::vector<Slot> slots0;
    std::vector<Slot> slots1;
    for (int i = 0; i < 4; ++i)
    {
        slots0.push_back(alloc0.allocate());
        slots1.push_back(alloc1.allocate());
    }
    std::set<SlotId> ids;
    for (auto const& slot : slots0)
    {
        EXPECT_EQ(slot.localityDomainId(), std::optional<LocalityDomainId>{0});
        EXPECT_LT(slot.slotId(), SlotId{4});
        ids.insert(slot.slotId());
    }
    for (auto const& slot : slots1)
    {
        EXPECT_EQ(slot.localityDomainId(), std::optional<LocalityDomainId>{1});
        EXPECT_GE(slot.slotId(), SlotId{4});
        ids.insert(slot.slotId());
    }
    EXPECT_EQ(ids.size(), 8U);
    EXPECT_EQ(alloc0.numFreeSlots(), 0);

    // A handle carrying the other domain's id is foreign to this allocator; the real slot stays
    // with alloc1 and is released below.
    {
        Slot foreign;
        foreign.setSlotId(slots1[0].slotId());
        foreign.setLocalityDomainId(LocalityDomainId{1});
        EXPECT_THROW(alloc0.release(std::move(foreign)), LogicError);
    }
    for (auto& slot : slots0)
    {
        Slot moved;
        moved.setSlot(slot);
        alloc0.release(std::move(moved));
    }
    for (auto& slot : slots1)
    {
        Slot moved;
        moved.setSlot(slot);
        alloc1.release(std::move(moved));
    }
    EXPECT_EQ(alloc0.numFreeSlots(), 4);
    EXPECT_EQ(alloc1.numFreeSlots(), 4);
    alloc0.synchronize();
    alloc1.synchronize();
    alloc0.prepareForShrink(SlotCount{0});
    EXPECT_TRUE(alloc0.finishShrink());
    alloc1.prepareForShrink(SlotCount{0});
    EXPECT_TRUE(alloc1.finishShrink());
}

TEST(KvCacheManagerV2LocalityTest, MappedAllocatorShrinksByLocalIndex)
{
    // Chunk 5 backs local indices [0, 4): global ids 20..23. Shrinking to 2 local slots must treat
    // global ids 22 and 23 as overflow, not ids >= 2.
    SlotIdMap map{SlotCount{4}};
    map.append(5);
    SlotAllocator alloc{SlotCount{4}, &map, LocalityDomainId{0}};
    auto slots = std::vector<Slot>{};
    for (int i = 0; i < 4; ++i)
    {
        slots.push_back(alloc.allocate());
    }
    for (auto& slot : slots)
    {
        Slot moved;
        moved.setSlot(slot);
        alloc.release(std::move(moved));
    }
    alloc.synchronize();
    alloc.prepareForShrink(SlotCount{2});
    EXPECT_EQ(alloc.numOverflowSlots(), 2);
    EXPECT_TRUE(alloc.finishShrink());
    EXPECT_EQ(alloc.numSlots(), 2);
    auto kept = alloc.allocate();
    EXPECT_LT(kept.slotId(), SlotId{22});
    Slot moved;
    moved.setSlot(kept);
    alloc.release(std::move(moved));
    alloc.synchronize();
    alloc.prepareForShrink(SlotCount{0});
    EXPECT_TRUE(alloc.finishShrink());
}

// ---------------------------------------------------------------------------
// Manager under TRT_LLM_MOCK_LOCALIZATION_SUPPORT=1 (GPU, ordinary device memory)
// ---------------------------------------------------------------------------

class KvCacheManagerV2MockLocalityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mSavedMode = std::getenv("TRT_LLM_MOCK_LOCALIZATION_SUPPORT") == nullptr
            ? std::nullopt
            : std::optional<std::string>{std::getenv("TRT_LLM_MOCK_LOCALIZATION_SUPPORT")};
        setenv("TRT_LLM_MOCK_LOCALIZATION_SUPPORT", "1", 1);
    }

    void TearDown() override
    {
        if (mSavedMode.has_value())
        {
            setenv("TRT_LLM_MOCK_LOCALIZATION_SUPPORT", mSavedMode->c_str(), 1);
        }
        else
        {
            unsetenv("TRT_LLM_MOCK_LOCALIZATION_SUPPORT");
        }
        cudaStreamDestroy(mStream);
    }

    static KVCacheManagerConfig makeLocalizedConfig(bool enableLocalityDomains = true)
    {
        KVCacheManagerConfig config;
        config.tokensPerBlock = 32;
        GpuCacheTierConfig gpu{256ULL << 20};
        gpu.enableLocalityDomains = enableLocalityDomains;
        config.cacheTiers.emplace_back(gpu);
        for (int layerId = 0; layerId < 2; ++layerId)
        {
            AttentionLayerConfig layer;
            layer.layerId = layerId;
            layer.buffers.push_back(BufferConfig{"key", 8192, std::nullopt});
            layer.buffers.push_back(BufferConfig{"value", 8192, std::nullopt});
            config.layers.emplace_back(std::move(layer));
        }
        return config;
    }

    static ReuseScope scopeFor(std::optional<int> localityDomainId)
    {
        return ReuseScope{std::nullopt, std::nullopt, std::optional<LocalityDomainId>(localityDomainId)};
    }

    std::vector<TokenIdExt> tokens(int count) const
    {
        std::vector<TokenIdExt> result;
        result.reserve(count);
        for (int i = 0; i < count; ++i)
        {
            result.emplace_back(TokenId{i + 1});
        }
        return result;
    }

    void commitAndClose(KvCache& cache, std::vector<TokenIdExt> const& allTokens, int capacity)
    {
        ASSERT_TRUE(cache.resume(reinterpret_cast<CUstream>(mStream)));
        ASSERT_TRUE(cache.resize(capacity));
        int const committed = cache.numCommittedTokens();
        if (committed < static_cast<int>(allTokens.size()))
        {
            cache.commit(TokenSpan{allTokens.data() + committed, static_cast<int>(allTokens.size()) - committed});
        }
        cache.stopCommitting();
        cache.close();
    }

    cudaStream_t mStream{};
    std::optional<std::string> mSavedMode;
};

TEST_F(KvCacheManagerV2MockLocalityTest, MockReportsTwoDomainsOnlyWhenEnabled)
{
    EXPECT_EQ(std::make_shared<KvCacheManager>(makeLocalizedConfig(false))->numLocalityDomains(), 1);
    EXPECT_EQ(std::make_shared<KvCacheManager>(makeLocalizedConfig(true))->numLocalityDomains(), 2);
    EXPECT_EQ(resolveNumLocalityDomains(false), 1);
}

TEST_F(KvCacheManagerV2MockLocalityTest, FreeSlotsAreTrackedPerDomain)
{
    auto manager = std::make_shared<KvCacheManager>(makeLocalizedConfig());
    auto const before = freeSlotsPerDomain(manager->storage());
    ASSERT_EQ(before.size(), 2U);
    EXPECT_GT(before[0], 0);
    EXPECT_EQ(before[0], before[1]);

    auto cache = manager->createKvCache(scopeFor(1));
    EXPECT_EQ(cache->localityDomainId(), std::optional<LocalityDomainId>{1});
    ASSERT_TRUE(cache->resume(reinterpret_cast<CUstream>(mStream)));
    ASSERT_TRUE(cache->resize(64));
    auto const during = freeSlotsPerDomain(manager->storage());
    EXPECT_EQ(during[0], before[0]);
    EXPECT_LT(during[1], before[1]);

    cache->close();
    EXPECT_EQ(freeSlotsPerDomain(manager->storage()), before);

    EXPECT_THROW(manager->createKvCache(scopeFor(2)), tensorrt_llm::common::TllmException);
    EXPECT_THROW(manager->createKvCache(scopeFor(std::nullopt)), tensorrt_llm::common::TllmException);
    auto plain = std::make_shared<KvCacheManager>(makeLocalizedConfig(false));
    EXPECT_THROW(plain->createKvCache(scopeFor(0)), tensorrt_llm::common::TllmException);
}

TEST_F(KvCacheManagerV2MockLocalityTest, PageIndexUpperBoundCoversBothDomains)
{
    auto manager = std::make_shared<KvCacheManager>(makeLocalizedConfig());
    auto cache0 = manager->createKvCache(scopeFor(0));
    auto cache1 = manager->createKvCache(scopeFor(1));
    ASSERT_TRUE(cache0->resume(reinterpret_cast<CUstream>(mStream)));
    ASSERT_TRUE(cache1->resume(reinterpret_cast<CUstream>(mStream)));
    ASSERT_TRUE(cache0->resize(128));
    ASSERT_TRUE(cache1->resize(128));

    auto const upperBound = static_cast<int>(manager->getPageIndexUpperBound(0, "key"));
    int maxIndex = -1;
    for (auto const* cache : {cache0.get(), cache1.get()})
    {
        for (int index : cache->getBasePageIndices(LayerGroupId{0}))
        {
            if (index != kBadPageIndex.value())
            {
                maxIndex = std::max(maxIndex, index);
            }
        }
    }
    EXPECT_GE(maxIndex, 0);
    EXPECT_LT(maxIndex, upperBound);

    // Both domains map into one address range; the mapped extents cover every page.
    size_t mappedPages = 0;
    for (auto const& [address, numPages] : manager->getMappedBufferViews(0, "key"))
    {
        EXPECT_NE(address, 0U);
        mappedPages += numPages;
    }
    EXPECT_GT(mappedPages, 0U);
    cache0->close();
    cache1->close();
}

TEST_F(KvCacheManagerV2MockLocalityTest, PrefixReuseStaysWithinADomain)
{
    auto manager = std::make_shared<KvCacheManager>(makeLocalizedConfig());
    auto const all = tokens(64);
    TokenSpan const prefix{all.data(), static_cast<int>(all.size()) - 1};

    auto first = manager->createKvCache(scopeFor(0), prefix);
    commitAndClose(*first, all, 128);

    auto sameDomain = manager->createKvCache(scopeFor(0), prefix);
    EXPECT_GT(sameDomain->numCommittedTokens(), 0);
    commitAndClose(*sameDomain, all, 128);

    auto otherDomain = manager->createKvCache(scopeFor(1), prefix);
    EXPECT_EQ(otherDomain->numCommittedTokens(), 0);
    commitAndClose(*otherDomain, all, 128);

    auto otherDomainAgain = manager->createKvCache(scopeFor(1), prefix);
    EXPECT_GT(otherDomainAgain->numCommittedTokens(), 0);
    otherDomainAgain->close();
}

TEST_F(KvCacheManagerV2MockLocalityTest, ResizeKeepsDomainsIndependent)
{
    auto manager = std::make_shared<KvCacheManager>(makeLocalizedConfig());
    auto const all = tokens(64);
    TokenSpan const prefix{all.data(), static_cast<int>(all.size()) - 1};
    for (int domain : {0, 1})
    {
        auto cache = manager->createKvCache(scopeFor(domain), prefix);
        commitAndClose(*cache, all, static_cast<int>(all.size()));
    }

    auto& storage = manager->storage();
    PoolGroupIndex const pg{0};
    SlotCount const before0 = storage.numSlots(pg);
    storage.shrinkPoolGroup(kHotLevel, pg, SlotCount{1}, {}, LocalityDomainId{1});
    EXPECT_EQ(storage.numSlots(pg), before0);
    EXPECT_EQ(freeSlotsPerDomain(storage)[1], 0);

    // Domain 0's cached prefix survived domain 1's shrink.
    auto cache = manager->createKvCache(scopeFor(0), prefix);
    EXPECT_GT(cache->numCommittedTokens(), 0);
    cache->close();

    storage.expandPoolGroup(kHotLevel, pg, before0, LocalityDomainId{1});
    EXPECT_GT(freeSlotsPerDomain(storage)[1], 0);
}

// A slot size with only a 16-byte power-of-two factor needs 2 MiB / 16 slots per chunk, about 16 GiB, so the
// manager keeps one domain instead of stranding that much memory per domain.
TEST_F(KvCacheManagerV2MockLocalityTest, OversizedChunkFallsBackToOneDomain)
{
    auto makeConfig = [](size_t slotSize)
    {
        KVCacheManagerConfig config;
        config.tokensPerBlock = 32;
        GpuCacheTierConfig gpu{256ULL << 20};
        gpu.enableLocalityDomains = true;
        config.cacheTiers.emplace_back(gpu);
        AttentionLayerConfig layer;
        layer.layerId = 0;
        layer.buffers.push_back(BufferConfig{"key", slotSize, std::nullopt});
        config.layers.emplace_back(std::move(layer));
        return config;
    };
    EXPECT_EQ(std::make_shared<KvCacheManager>(makeConfig(16384))->numLocalityDomains(), 2);
    EXPECT_EQ(std::make_shared<KvCacheManager>(makeConfig(8191 * 16))->numLocalityDomains(), 1);
}

// One 16 KiB-slot pool, 2 MiB physical grains, two domains: the per-domain share is the raw
// quota halved and rounded down to a grain, both at construction and on resize.
TEST_F(KvCacheManagerV2MockLocalityTest, QuotaIsSharedPerDomainConsistentlyOnInitAndResize)
{
    auto makeConfig = [](size_t quota)
    {
        KVCacheManagerConfig config;
        config.tokensPerBlock = 32;
        GpuCacheTierConfig gpu{quota};
        gpu.enableLocalityDomains = true;
        config.cacheTiers.emplace_back(gpu);
        AttentionLayerConfig layer;
        layer.layerId = 0;
        layer.buffers.push_back(BufferConfig{"key", 16384, std::nullopt});
        config.layers.emplace_back(std::move(layer));
        return config;
    };
    constexpr size_t kMiB = 1ULL << 20;
    constexpr SlotCount kSlotsPerGrain = 128;
    PoolGroupIndex const pg{0};

    for (auto const& [quotaMiB, grainsPerDomain] : std::vector<std::pair<size_t, SlotCount>>{{6, 1}, {7, 1}, {8, 2}})
    {
        auto manager = std::make_shared<KvCacheManager>(makeConfig(quotaMiB * kMiB));
        auto& storage = manager->storage();
        EXPECT_EQ(storage.numSlots(pg), grainsPerDomain * kSlotsPerGrain) << quotaMiB << " MiB at init";
        for (auto const& [resizeMiB, resizeGrains] : std::vector<std::pair<size_t, SlotCount>>{{8, 2}, {7, 1}, {6, 1}})
        {
            storage.adjustCacheLevel(kHotLevel, resizeMiB * kMiB, storage.getRatioList(kHotLevel), nullptr);
            EXPECT_EQ(storage.numSlots(pg), resizeGrains * kSlotsPerGrain) << resizeMiB << " MiB on resize";
            auto const free = freeSlotsPerDomain(storage);
            EXPECT_EQ(free[0], free[1]);
            EXPECT_EQ(free[0], resizeGrains * kSlotsPerGrain);
        }
    }
}

TEST_F(KvCacheManagerV2MockLocalityTest, ManagerResizeDecidesPerDomain)
{
    constexpr size_t kFullQuota = 256ULL << 20;
    constexpr size_t kMidQuota = 192ULL << 20;
    auto manager = std::make_shared<KvCacheManager>(makeLocalizedConfig());
    auto& storage = manager->storage();
    PoolGroupIndex const pg{0};
    auto const ratio = storage.getRatioList(kHotLevel);
    SlotCount const full = storage.numSlots(pg);
    ASSERT_TRUE(manager->resize(kHotLevel, kMidQuota));
    SlotCount const mid = storage.numSlots(pg);
    ASSERT_LT(mid, full);
    ASSERT_TRUE(manager->resize(kHotLevel, kFullQuota));
    ASSERT_EQ(storage.numSlots(pg), full);

    // Domain 0 above the target and domain 1 below it: one shrinks while the other grows.
    storage.shrinkPoolGroup(kHotLevel, pg, mid / 2, {}, LocalityDomainId{1});
    ASSERT_EQ(freeSlotsPerDomain(storage)[1], mid / 2);
    storage.adjustCacheLevel(kHotLevel, kMidQuota, ratio, nullptr);
    EXPECT_EQ(freeSlotsPerDomain(storage), (std::vector<SlotCount>{mid, mid}));

    // Domain 1 already at the target: only domain 0 moves.
    storage.shrinkPoolGroup(kHotLevel, pg, mid / 2, {}, LocalityDomainId{0});
    storage.adjustCacheLevel(kHotLevel, kMidQuota, ratio, nullptr);
    EXPECT_EQ(freeSlotsPerDomain(storage), (std::vector<SlotCount>{mid, mid}));

    EXPECT_THROW(storage.expandPoolGroup(kHotLevel, pg, mid, LocalityDomainId{0}), std::exception);
    EXPECT_THROW(storage.shrinkPoolGroup(kHotLevel, pg, mid, {}, LocalityDomainId{1}), std::exception);
    EXPECT_EQ(freeSlotsPerDomain(storage), (std::vector<SlotCount>{mid, mid}));
}

// A 3 MiB slot does not divide the 2 MiB grain, so each domain's backing rounds up on its own.
// Re-applying the reported quota must not lose slots.
TEST_F(KvCacheManagerV2MockLocalityTest, QuotaRoundTripKeepsCoarseSlots)
{
    constexpr size_t kMiB = 1ULL << 20;
    KVCacheManagerConfig config;
    config.tokensPerBlock = 32;
    GpuCacheTierConfig gpu{32 * kMiB};
    gpu.enableLocalityDomains = true;
    config.cacheTiers.emplace_back(gpu);
    AttentionLayerConfig layer;
    layer.layerId = 0;
    layer.buffers.push_back(BufferConfig{"key", 3 * kMiB, std::nullopt});
    config.layers.emplace_back(std::move(layer));

    auto manager = std::make_shared<KvCacheManager>(config);
    auto& storage = manager->storage();
    PoolGroupIndex const pg{0};
    SlotCount const initial = storage.numSlots(pg);
    ASSERT_GT(initial, 0);
    size_t const perDomainBytes = (static_cast<size_t>(initial) * 3 * kMiB + 2 * kMiB - 1) / (2 * kMiB) * (2 * kMiB);
    EXPECT_EQ(manager->getQuota(kHotLevel), 2 * perDomainBytes);
    for (int i = 0; i < 2; ++i)
    {
        manager->adjust();
        EXPECT_EQ(storage.numSlots(pg), initial);
        EXPECT_EQ(freeSlotsPerDomain(storage), (std::vector<SlotCount>{initial, initial}));
        EXPECT_EQ(manager->getQuota(kHotLevel), 2 * perDomainBytes);
    }
}

TEST_F(KvCacheManagerV2MockLocalityTest, SecondaryGpuTierIsRejectedWhenLocalized)
{
    auto config = makeLocalizedConfig();
    config.cacheTiers.emplace_back(GpuCacheTierConfig{64ULL << 20});
    try
    {
        KvCacheManager manager(config);
        FAIL() << "a localized manager accepted a secondary GPU tier";
    }
    catch (std::invalid_argument const& e)
    {
        EXPECT_NE(std::string(e.what()).find("Locality domains"), std::string::npos) << e.what();
    }
}

TEST_F(KvCacheManagerV2MockLocalityTest, UtilizationIsReportedPerDomain)
{
    auto manager = std::make_shared<KvCacheManager>(makeLocalizedConfig());
    auto& storage = manager->storage();
    auto const capacity = storage.numSlots(PoolGroupIndex{0});
    auto cache = manager->createKvCache(scopeFor(1));
    ASSERT_TRUE(cache->resume(reinterpret_cast<CUstream>(mStream)));
    ASSERT_TRUE(cache->resize(static_cast<int>(capacity) / 2 * 32));

    EXPECT_FLOAT_EQ(storage.getUtilization(kHotLevel, LocalityDomainId{0}).at(PoolGroupIndex{0}), 0.f);
    EXPECT_GT(storage.getUtilization(kHotLevel, LocalityDomainId{1}).at(PoolGroupIndex{0}), 0.4f);
    EXPECT_LT(storage.getUtilization(kHotLevel).at(PoolGroupIndex{0}), 0.35f);
    cache->close();
}

TEST_F(KvCacheManagerV2MockLocalityTest, NonLocalizedScopesHashAsBefore)
{
    // The third ReuseScope field must not change the key of scopes that do not set it.
    ReuseScope const plain{7, "11"};
    ReuseScope const explicitNone{7, "11", std::nullopt};
    EXPECT_EQ(Hasher(plain).digest(), Hasher(explicitNone).digest());
    EXPECT_NE(Hasher(plain).digest(), Hasher(ReuseScope{7, "11", LocalityDomainId{0}}).digest());
    EXPECT_NE(Hasher(ReuseScope{7, "11", LocalityDomainId{0}}).digest(),
        Hasher(ReuseScope{7, "11", LocalityDomainId{1}}).digest());
}

} // namespace
