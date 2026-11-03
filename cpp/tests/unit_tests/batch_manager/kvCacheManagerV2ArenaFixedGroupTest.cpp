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

// Copy-based rebalancing (CoReba) with pool groups of fixed size: a group holding host-backed
// (sparse) pools -- sparse attention KV cache offload -- is fixed at its initial slot count and
// lives beside the GPU arena, which rebalances the other groups. The fixed group's memory never
// moves, and every block's bytes survive the rebalances.

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/introspection.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

constexpr int kTokensPerBlock = 4;
constexpr int kBlocksPerRequest = 6;
constexpr int kNumRequests = 4;

// Layer 0 (dense) and layer 1 (sparse) are both full attention, so they share a life cycle and a pool group with
// two pools: the sparse one host-backed. Layers 2 and 3 are sliding-window layers of different slot sizes, so each
// has a pool group of its own: the two groups the arena rebalances.
constexpr LayerId kDense = 0;
constexpr LayerId kSparse = 1;
constexpr LayerId kSwaA = 2;
constexpr LayerId kSwaB = 3;
constexpr size_t kDenseBytes = 64 << 10;
constexpr size_t kSparseBytes = 128 << 10;
constexpr size_t kSwaABytes = 32 << 10;
constexpr size_t kSwaBBytes = 16 << 10;
// With bound pools, layer 2 also has a smaller "value" buffer: its group then has two pools, the second with slot
// ids of its own, which takes the arena's multi-pool path.
constexpr size_t kSwaAValueBytes = 8 << 10;

KVCacheManagerConfig makeSparseCoRebaConfig(bool withElasticGroups = true, bool withBoundPools = false)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = kTokensPerBlock;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{64UL << 20});
    config.layers.emplace_back(
        AttentionLayerConfig{.layerId = kDense, .buffers = {BufferConfig{.role = "key", .size = kDenseBytes}}});
    AttentionLayerConfig sparse{
        .layerId = kSparse, .buffers = {BufferConfig{.role = "key", .size = kSparseBytes, .isSparse = true}}};
    sparse.sparseCache = SparseCacheConfig{/*numSelectedBlocks=*/1, /*maxCachedUnits=*/2, /*unitTokens=*/4};
    config.layers.emplace_back(std::move(sparse));
    if (withElasticGroups)
    {
        AttentionLayerConfig swaA{
            .layerId = kSwaA, .buffers = {BufferConfig{.role = "key", .size = kSwaABytes}}, .slidingWindowSize = 8};
        if (withBoundPools)
            swaA.buffers.push_back(BufferConfig{.role = "value", .size = kSwaAValueBytes});
        config.layers.emplace_back(std::move(swaA));
        config.layers.emplace_back(AttentionLayerConfig{
            .layerId = kSwaB, .buffers = {BufferConfig{.role = "key", .size = kSwaBBytes}}, .slidingWindowSize = 16});
    }
    config.poolRebalance.copyBasedRebalancing = true;
    return config;
}

struct Buffer
{
    LayerId layer;
    char const* role;
    size_t bytes;
    int tag; // distinguishes the buffers' patterns
};

// One buffer per pool here, so a pool's slot is the buffer and the per-layer base addresses slot 0. A bound pool
// (its own slot ids) is addressed through its bindings, as the kernels' consumers translate it.
std::uintptr_t blockAddress(KvCacheManager const& manager, Buffer const& buffer, int slot)
{
    auto const base = manager.getMemPoolBaseAddress(buffer.layer, buffer.role, PageIndexMode::PER_LAYER);
    auto const bindings = manager.getPageIndexBindings(buffer.layer, buffer.role);
    int const id = bindings.has_value() ? bindings->at(static_cast<size_t>(slot)) : slot;
    return static_cast<std::uintptr_t>(base) + static_cast<std::uintptr_t>(id) * buffer.bytes;
}

std::uint8_t pattern(int request, int block, int tag)
{
    return static_cast<std::uint8_t>(1 + (request * 31 + block * 7 + tag * 3) % 251);
}

class ArenaFixedGroupTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
    }

    void TearDown() override
    {
        for (auto& [id, kv] : mLive)
            kv->close();
        mLive.clear();
        mManager.reset();
        cudaStreamDestroy(mStream);
    }

    std::vector<Buffer> buffers() const
    {
        std::vector<Buffer> out{{kDense, "key", kDenseBytes, 0}, {kSparse, "key", kSparseBytes, 1}};
        if (mWithElasticGroups)
        {
            out.push_back({kSwaA, "key", kSwaABytes, 2});
            out.push_back({kSwaB, "key", kSwaBBytes, 3});
        }
        if (mWithBoundPools)
            out.push_back({kSwaA, "value", kSwaAValueBytes, 4});
        return out;
    }

    // Page indices per layer of a live request (the layer's life cycle's base page indices).
    std::vector<int> slots(KvCache const& kv, LayerId layer) const
    {
        auto const span = kv.getBasePageIndices(mManager->getLayerGroupId(layer));
        return {span.begin(), span.end()};
    }

    void addRequest(int id)
    {
        std::vector<TokenIdExt> tokens;
        for (int t = 0; t < kBlocksPerRequest * kTokensPerBlock; ++t)
            tokens.emplace_back(TokenId{static_cast<TokenId>(1000 * id + t)});
        auto kv = mManager->createKvCache({}, TokenSpan{tokens.data(), static_cast<int>(tokens.size())}, id);
        ASSERT_TRUE(kv->resume(reinterpret_cast<CUstream>(mStream)));
        // As on a decode-only (offload) worker: the request arrives with its history, so capacity and history are
        // set together and nothing is in flight in the sparse cache.
        int const length = static_cast<int>(tokens.size());
        ASSERT_TRUE(kv->resize(length, length));
        for (auto const& buffer : buffers())
        {
            auto const ids = slots(*kv, buffer.layer);
            for (int block = 0; block < static_cast<int>(ids.size()); ++block)
            {
                if (ids[block] < 0)
                    continue; // out of a sliding window
                ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(blockAddress(*mManager, buffer, ids[block])),
                              pattern(id, block, buffer.tag), buffer.bytes),
                    cudaSuccess);
            }
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        mLive.emplace(id, std::move(kv));
    }

    // Every live block still holds its pattern at its (possibly moved) slot.
    void checkLive()
    {
        std::vector<std::uint8_t> host;
        for (auto const& [id, kv] : mLive)
        {
            for (auto const& buffer : buffers())
            {
                auto const ids = slots(*kv, buffer.layer);
                for (int block = 0; block < static_cast<int>(ids.size()); ++block)
                {
                    if (ids[block] < 0)
                        continue;
                    host.assign(buffer.bytes, 0);
                    ASSERT_EQ(
                        cudaMemcpy(host.data(), reinterpret_cast<void*>(blockAddress(*mManager, buffer, ids[block])),
                            host.size(), cudaMemcpyDefault),
                        cudaSuccess);
                    auto const want = pattern(id, block, buffer.tag);
                    for (size_t i = 0; i < host.size(); i += 4093)
                        ASSERT_EQ(host[i], want) << "request " << id << " layer " << buffer.layer << " " << buffer.role
                                                 << " block " << block;
                }
            }
        }
    }

    void rebalance(TypedVec<PoolGroupIndex, float> const& target)
    {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (auto& [id, kv] : mLive)
            kv->suspend();
        KvCacheIntrospection::setNumSampledKvCaches(*mManager, 1 << 20);
        KvCacheIntrospection::setLastAdjustmentTime(*mManager, 0.0);
        KvCacheIntrospection::setTargetRatioListGpu(*mManager, target);
        mManager->adjust();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (auto& [id, kv] : mLive)
            ASSERT_TRUE(kv->resume(reinterpret_cast<CUstream>(mStream)));
    }

    void runFixedWhileRebalancing();

    PoolGroupIndex poolGroupOf(LayerId layer) const
    {
        auto const lifeCycle = static_cast<LifeCycleId>(mManager->getLayerGroupId(layer));
        return mManager->getLifeCyclePoolGroupIndices()[lifeCycle];
    }

    bool mWithElasticGroups = true;
    bool mWithBoundPools = false;
    cudaStream_t mStream{};
    std::shared_ptr<KvCacheManager> mManager;
    std::map<int, std::shared_ptr<KvCache>> mLive;
};

TEST_F(ArenaFixedGroupTest, SparseGroupIsFixedWhileTheArenaRebalances)
{
    runFixedWhileRebalancing();
}

TEST_F(ArenaFixedGroupTest, SparseGroupIsFixedWhileBoundPoolsRebalance)
{
    mWithBoundPools = true;
    runFixedWhileRebalancing();
}

void ArenaFixedGroupTest::runFixedWhileRebalancing()
{
    mManager = std::make_shared<KvCacheManager>(makeSparseCoRebaConfig(true, mWithBoundPools));
    if (mWithBoundPools)
    {
        ASSERT_TRUE(mManager->getPageIndexBindings(kSwaA, "value").has_value()) << "expected a bound pool";
    }
    PoolGroupIndex const fixed = poolGroupOf(kDense);
    ASSERT_EQ(poolGroupOf(kSparse), fixed);
    PoolGroupIndex const a = poolGroupOf(kSwaA);
    PoolGroupIndex const b = poolGroupOf(kSwaB);
    ASSERT_NE(a, fixed);
    ASSERT_NE(b, fixed);
    ASSERT_NE(a, b);

    auto const fixedWindows = KvCacheIntrospection::gpuPoolWindows(*mManager)[toSizeT(fixed)];
    auto const denseBase = mManager->getMemPoolBaseAddress(kDense, "key", PageIndexMode::PER_LAYER);
    auto const sparseBase = mManager->getMemPoolBaseAddress(kSparse, "key", PageIndexMode::PER_LAYER);
    // A fixed group has an allocation of its own: slot ids from 0, the same for both pools.
    for (auto const& window : fixedWindows)
    {
        EXPECT_EQ(window.first, 0);
        EXPECT_EQ(window.second, fixedWindows.front().second);
    }

    for (int id = 0; id < kNumRequests; ++id)
        addRequest(id);
    checkLive();

    auto const before = KvCacheIntrospection::gpuSlotWindows(*mManager);
    int moved = 0;
    for (auto const& shares : {std::pair{0.9F, 0.1F}, std::pair{0.1F, 0.9F}, std::pair{0.5F, 0.5F}})
    {
        TypedVec<PoolGroupIndex, float> target(PoolGroupIndex{3}, 0.F);
        target[fixed] = 0.05F; // ignored: a fixed group keeps its share
        target[a] = shares.first;
        target[b] = shares.second;
        rebalance(target);
        checkLive();
        moved += KvCacheIntrospection::gpuSlotWindows(*mManager) != before ? 1 : 0;
        auto const windows = KvCacheIntrospection::gpuPoolWindows(*mManager);
        EXPECT_EQ(windows[toSizeT(fixed)], fixedWindows);
        EXPECT_EQ(mManager->getMemPoolBaseAddress(kDense, "key", PageIndexMode::PER_LAYER), denseBase);
        EXPECT_EQ(mManager->getMemPoolBaseAddress(kSparse, "key", PageIndexMode::PER_LAYER), sparseBase);
    }
    // The elastic groups did move.
    EXPECT_GT(moved, 0);
}

TEST_F(ArenaFixedGroupTest, ElasticWindowsTileTheArenaAndFollowTheTarget)
{
    mManager = std::make_shared<KvCacheManager>(makeSparseCoRebaConfig());
    PoolGroupIndex const fixed = poolGroupOf(kDense);
    PoolGroupIndex const a = poolGroupOf(kSwaA);
    PoolGroupIndex const b = poolGroupOf(kSwaB);
    auto sizeOf = [&](PoolGroupIndex pg)
    {
        auto const w = KvCacheIntrospection::gpuSlotWindows(*mManager)[toSizeT(pg)];
        return w.second - w.first;
    };
    TypedVec<PoolGroupIndex, float> target(PoolGroupIndex{3}, 0.F);
    target[fixed] = 0.3F;
    target[a] = 0.6F;
    target[b] = 0.1F;
    rebalance(target);
    auto const aBig = sizeOf(a) * static_cast<SlotCount>(kSwaABytes);
    auto const bSmall = sizeOf(b) * static_cast<SlotCount>(kSwaBBytes);
    target[a] = 0.1F;
    target[b] = 0.6F;
    rebalance(target);
    auto const aSmall = sizeOf(a) * static_cast<SlotCount>(kSwaABytes);
    auto const bBig = sizeOf(b) * static_cast<SlotCount>(kSwaBBytes);
    EXPECT_GT(aBig, aSmall);
    EXPECT_GT(bBig, bSmall);
    // The two elastic groups split the arena: their windows do not overlap in bytes.
    auto const windows = KvCacheIntrospection::gpuSlotWindows(*mManager);
    auto const wa = windows[toSizeT(a)];
    auto const wb = windows[toSizeT(b)];
    auto const aEnd = static_cast<size_t>(wa.second) * kSwaABytes;
    auto const bBegin = static_cast<size_t>(wb.first) * kSwaBBytes;
    auto const bEnd = static_cast<size_t>(wb.second) * kSwaBBytes;
    auto const aBegin = static_cast<size_t>(wa.first) * kSwaABytes;
    EXPECT_TRUE(aEnd <= bBegin || bEnd <= aBegin);
}

TEST_F(ArenaFixedGroupTest, GpuArenaExtentCoversOnlyTheArena)
{
    // Disaggregated serving registers the extent once with NIXL, and fixed groups separately: the extent must not
    // reach past what the arena maps into the fixed groups' own allocations.
    mManager = std::make_shared<KvCacheManager>(makeSparseCoRebaConfig());
    auto const extent = mManager->getGpuArenaExtent();
    ASSERT_TRUE(extent.has_value());
    auto const [base, bytes, hasBoundPools] = *extent;
    EXPECT_FALSE(hasBoundPools);
    auto inArena = [&, base = base, bytes = bytes](LayerId layer)
    {
        auto const addr = mManager->getMemPoolBaseAddress(layer, "key", PageIndexMode::PER_LAYER);
        return base <= addr && addr < base + bytes;
    };
    EXPECT_TRUE(inArena(kSwaA));
    EXPECT_TRUE(inArena(kSwaB));
    EXPECT_FALSE(inArena(kDense));
    EXPECT_FALSE(inArena(kSparse));
    // The level's quota also pays for the fixed group's device pool.
    EXPECT_LT(bytes, mManager->getQuota(kHotLevel));
}

TEST_F(ArenaFixedGroupTest, OnlyFixedGroupsFallsBackToTheVmmStorage)
{
    // One pool group, holding the sparse pool: nothing to rebalance, so the arena is not built.
    mWithElasticGroups = false;
    mManager = std::make_shared<KvCacheManager>(makeSparseCoRebaConfig(/*withElasticGroups=*/false));
    addRequest(0);
    TypedVec<PoolGroupIndex, float> target(PoolGroupIndex{1}, 1.F);
    rebalance(target);
    checkLive();
}

} // namespace
