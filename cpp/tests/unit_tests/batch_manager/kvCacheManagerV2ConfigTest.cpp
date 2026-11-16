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
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/storage/config.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeReplayRegionConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeTieredConfig;

AttentionLayerConfig makeAttentionLayer(LayerId layerId, bool isSparse, size_t bufferSize = 4096)
{
    AttentionLayerConfig layer{
        .layerId = layerId, .buffers = {BufferConfig{.role = "key", .size = bufferSize, .isSparse = isSparse}}};
    if (isSparse)
    {
        layer.sparseCache = SparseCacheConfig{/*numSelectedBlocks=*/1, /*maxCachedUnits=*/2, /*unitTokens=*/4};
    }
    return layer;
}

TEST(KvCacheManagerV2ConfigTest, ExistingBufferConfigurationDefaultsToDense)
{
    BufferConfig const buffer{"key", 4096, 2};
    EXPECT_FALSE(buffer.isSparse);
    EXPECT_EQ(buffer.tokensPerBlockOverride, 2);
    EXPECT_NO_THROW(makeConfig().validate());
}

TEST(KvCacheManagerV2ConfigTest, SparseCacheCapacityRespectsProbeLimitAfterPageRounding)
{
    constexpr int kTokensPerBlock = 128;

    struct Boundary
    {
        int unitTokens;
        int maxCachedUnits;
    };

    constexpr Boundary kBoundaries[] = {{128, 16'382}, {1, 16'256}, {4, 16'352}};
    for (auto const& boundary : kBoundaries)
    {
        SCOPED_TRACE(boundary.unitTokens);
        SparseCacheConfig cache{
            .numSelectedBlocks = 2, .maxCachedUnits = boundary.maxCachedUnits, .unitTokens = boundary.unitTokens};
        EXPECT_NO_THROW(cache.validate(kTokensPerBlock));
        ++cache.maxCachedUnits;
        EXPECT_THROW(cache.validate(kTokensPerBlock), std::runtime_error);
    }
}

TEST(KvCacheManagerV2ConfigTest, OversizedSparseCacheFailsManagerConfigValidation)
{
    auto config = makeConfig();
    config.tokensPerBlock = 128;
    auto layer = makeAttentionLayer(0, true);
    layer.buffers.push_back(BufferConfig{.role = "index", .size = 256});
    // Eight selections of 2048 tokens plus one in-flight token round to 16,512 entries.
    layer.sparseCache = SparseCacheConfig{.numSelectedBlocks = 2048, .maxCachedUnits = 16'512, .unitTokens = 1};
    config.layers = {layer};
    try
    {
        config.validate();
        FAIL() << "An oversized sparse cache must fail before manager construction";
    }
    catch (std::runtime_error const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("maxCachedUnits=16512"), std::string::npos);
        EXPECT_NE(message.find("supported capacity of 16256"), std::string::npos);
        EXPECT_NE(message.find("reduce gpu_cache_factor"), std::string::npos);
    }
}

TEST(KvCacheManagerV2ConfigTest, SparseCacheCapacityValidationHandlesLargeInputs)
{
    constexpr int kTokensPerBlock = 128;
    SparseCacheConfig const cache{
        .numSelectedBlocks = 2, .maxCachedUnits = std::numeric_limits<int>::max(), .unitTokens = 1};
    EXPECT_THROW(cache.validate(kTokensPerBlock), std::runtime_error);
}

TEST(KvCacheManagerV2ConfigTest, SparseCacheCapacityValidationRejectsNonPositivePageSize)
{
    SparseCacheConfig const cache{.numSelectedBlocks = 2, .maxCachedUnits = 4, .unitTokens = 1};
    EXPECT_THROW(cache.validate(0), std::runtime_error);
    EXPECT_THROW(cache.validate(-1), std::runtime_error);
}

TEST(KvCacheManagerV2ConfigTest, OneLayerMayMixSparseAndDenseBuffers)
{
    // Sparsity picks a buffer's pool and nothing above it, so an indexer and the data it
    // selects over stay in the one layer they belong to.
    auto config = makeTieredConfig();
    for (bool const firstIsSparse : {false, true})
    {
        auto layer = makeAttentionLayer(0, firstIsSparse);
        layer.buffers.push_back(BufferConfig{.role = "value", .size = 4096, .isSparse = !firstIsSparse});
        config.layers = {layer};
        ASSERT_NO_THROW(config.validate());
        auto const storage = createStorageConfig(config);
        auto const attributes = storage.bufferAttributes();
        auto const& key = attributes.at(BufferId{0, "key"});
        auto const& value = attributes.at(BufferId{0, "value"});
        // One life cycle, one pool group, but a pool each: equal sizes do not coalesce
        // across sparsity, because the two pools differ in medium.
        EXPECT_EQ(key.lifeCycleId, value.lifeCycleId);
        EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{1});
        EXPECT_NE(key.poolIndex, value.poolIndex);
    }
}

TEST(KvCacheManagerV2ConfigTest, SparseSsmBuffersAreRejected)
{
    auto config = makeTieredConfig();
    config.commitMinSnapshot = true;
    config.layers
        = {SsmLayerConfig{.layerId = 0, .buffers = {BufferConfig{.role = "state", .size = 4096, .isSparse = true}}}};
    EXPECT_THROW(config.validate(), std::invalid_argument);
    EXPECT_THROW(createStorageConfig(config), std::invalid_argument);
}

TEST(KvCacheManagerV2ConfigTest, SparseAndDenseLayersShareALifeCycleButNotAPool)
{
    auto config = makeTieredConfig();
    config.layers = {makeAttentionLayer(0, false), makeAttentionLayer(1, true), makeAttentionLayer(2, false),
        makeAttentionLayer(3, true)};
    ASSERT_NO_THROW(config.validate());

    LifeCycleRegistry const registry(config);
    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const grouping = storage.lifeCycleGrouping();
    auto const& dense = attributes.at(BufferId{0, "key"});
    auto const& sparse = attributes.at(BufferId{1, "key"});

    // Sparsity is not part of a life cycle's identity, so layers alike in window and sinks
    // share one however their buffers are backed.
    EXPECT_EQ(registry.size(), LifeCycleId{1});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{1});
    EXPECT_EQ(dense.lifeCycleId, sparse.lifeCycleId);
    EXPECT_EQ(grouping[dense.lifeCycleId], grouping[sparse.lifeCycleId]);
    // Separate pools, because only one of them is host-backed.
    EXPECT_NE(dense.poolIndex, sparse.poolIndex);

    auto const& otherDense = attributes.at(BufferId{2, "key"});
    auto const& otherSparse = attributes.at(BufferId{3, "key"});
    EXPECT_EQ(dense.poolIndex, otherDense.poolIndex);
    EXPECT_EQ(sparse.poolIndex, otherSparse.poolIndex);
    EXPECT_EQ(otherSparse.offset, sparse.offset + sparse.size);
    EXPECT_EQ(otherDense.offset, dense.offset + dense.size);
}

TEST(KvCacheManagerV2ConfigTest, MatchingSlotShapesSharePoolsAcrossWindows)
{
    auto config = makeTieredConfig();
    config.layers.clear();
    for (LayerId layerId = 0; layerId < 4; ++layerId)
    {
        auto layer = makeAttentionLayer(layerId, layerId >= 2);
        layer.slidingWindowSize = layerId % 2 == 0 ? 128 : 256;
        config.layers.push_back(layer);
    }
    ASSERT_NO_THROW(config.validate());

    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const grouping = storage.lifeCycleGrouping();
    auto poolGroup = [&](LayerId layerId) { return grouping[attributes.at(BufferId{layerId, "key"}).lifeCycleId]; };
    // Two life cycles, one per window; sparsity is not part of their identity, so layer 0
    // (dense) and layer 2 (sparse) share one. Both life cycles present the same pools --
    // a dense 4096 and a sparse 4096 -- so one pool group serves them.
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{2});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{1});
    EXPECT_EQ(poolGroup(0), poolGroup(1));
    EXPECT_EQ(poolGroup(0), poolGroup(2));
    EXPECT_EQ(poolGroup(2), poolGroup(3));
    // Sparsity still separates the pools within that group.
    auto poolOf = [&](LayerId layerId) { return attributes.at(BufferId{layerId, "key"}).poolIndex; };
    EXPECT_NE(poolOf(0), poolOf(2));
}

TEST(KvCacheManagerV2ConfigTest, SinkRequiredOnlyWindowsShareRequiredPools)
{
    auto config = makeTieredConfig();
    auto sinkOnly = makeAttentionLayer(1, false);
    sinkOnly.slidingWindowSize = 128;
    sinkOnly.numSinkTokens = 4;
    sinkOnly.reusePolicy = ReusePolicy::SINK_REQUIRED_ONLY;
    auto optional = sinkOnly;
    optional.layerId = 2;
    optional.reusePolicy = ReusePolicy::OPTIONAL;
    config.layers = {makeAttentionLayer(0, false), sinkOnly, optional};
    ASSERT_NO_THROW(config.validate());

    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const grouping = storage.lifeCycleGrouping();
    auto const& full = attributes.at(BufferId{0, "key"});
    auto const& window = attributes.at(BufferId{1, "key"});
    auto const& optionalWindow = attributes.at(BufferId{2, "key"});
    // Three life cycles of one page shape: the SINK_REQUIRED_ONLY window shares the REQUIRED
    // layer's pool group, and OPTIONAL keeps its own.
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{3});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{2});
    EXPECT_EQ(grouping[full.lifeCycleId], grouping[window.lifeCycleId]);
    EXPECT_NE(grouping[full.lifeCycleId], grouping[optionalWindow.lifeCycleId]);
}

TEST(KvCacheManagerV2ConfigTest, SparseBuffersCoalesceBySize)
{
    // A sparse buffer's `size` is its whole page, as for any other buffer, so two of equal
    // size coalesce however finely each divides its page into selection units.
    auto config = makeTieredConfig();
    auto firstLayer = makeAttentionLayer(0, true, 4096);
    firstLayer.sparseCache = SparseCacheConfig{/*numSelectedBlocks=*/1, /*maxCachedUnits=*/2, /*unitTokens=*/2};
    config.layers = {firstLayer, makeAttentionLayer(1, true, 4096)};
    ASSERT_NO_THROW(config.validate());

    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const& first = attributes.at(BufferId{0, "key"});
    auto const& second = attributes.at(BufferId{1, "key"});
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{1});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{1});
    EXPECT_EQ(first.lifeCycleId, second.lifeCycleId);
    EXPECT_EQ(first.poolIndex, second.poolIndex);
    EXPECT_EQ(first.size, second.size);
    EXPECT_EQ(second.offset, first.offset + first.size);
    // tokensPerBlockOverride is the index converter's, and a sparse buffer does not use it.
    EXPECT_EQ(first.expansion, 1);
    EXPECT_EQ(second.expansion, 1);
}

TEST(KvCacheManagerV2ConfigTest, AttentionBufferPoliciesOverrideLayerDefault)
{
    auto config = makeConfig();
    auto& buffer = std::get<AttentionLayerConfig>(config.layers.front()).buffers.front();
    buffer.reusePolicy = ReusePolicy::REQUIRED;
    EXPECT_NO_THROW(config.validate());
    EXPECT_NO_THROW(createStorageConfig(config));
    for (auto const policy : {ReusePolicy::PRIVATE, ReusePolicy::OPTIONAL})
    {
        buffer.reusePolicy = policy;
        EXPECT_NO_THROW(config.validate());
        auto const storage = createStorageConfig(config);
        LifeCycleRegistry const registry(config);
        auto const attr = storage.bufferAttributes().at(BufferId{0, buffer.role});
        EXPECT_EQ(std::get<AttnLifeCycle>(registry[attr.lifeCycleId]).reusePolicy, policy);
    }
}

TEST(KvCacheManagerV2ConfigTest, SsmBufferReusePoliciesSplitLayerLifecycles)
{
    auto config = makeTieredConfig();
    config.commitMinSnapshot = true;
    for (LayerId layerId = 1; layerId <= 2; ++layerId)
    {
        config.layers.push_back(SsmLayerConfig{.layerId = layerId,
            .buffers = {BufferConfig{.role = "state", .size = 4096},
                BufferConfig{.role = "replay", .size = 2048, .reusePolicy = ReusePolicy::PRIVATE},
                BufferConfig{.role = "draft", .size = 2048, .reusePolicy = ReusePolicy::OPTIONAL}}});
    }
    // A layer-level PRIVATE layer shares the lifecycle of PRIVATE buffer overrides.
    config.layers.push_back(SsmLayerConfig{
        .layerId = 3, .buffers = {BufferConfig{.role = "replay", .size = 2048}}, .reusePolicy = ReusePolicy::PRIVATE});
    ASSERT_NO_THROW(config.validate());

    LifeCycleRegistry const registry(config);
    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto lifeCycle = [&](LayerId layerId, DataRole const& role) {
        return attributes.at(BufferId{layerId, role}).lifeCycleId;
    };
    auto const required = lifeCycle(1, "state");
    auto const privateLc = lifeCycle(1, "replay");
    auto const optional = lifeCycle(1, "draft");

    // Lifecycles are registered in buffer order, after the attention layer's.
    EXPECT_EQ(registry.size(), LifeCycleId{4});
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{4});
    EXPECT_EQ(required, LifeCycleId{1});
    EXPECT_EQ(privateLc, LifeCycleId{2});
    EXPECT_EQ(optional, LifeCycleId{3});
    EXPECT_EQ(registry.ssmLifeCycleIds(), (std::vector<LifeCycleId>{required, privateLc, optional}));
    EXPECT_EQ(registry.requiredSsmLifeCycleId(), required);
    EXPECT_EQ(std::get<SsmLifeCycle>(registry[privateLc]).reusePolicy, ReusePolicy::PRIVATE);
    EXPECT_EQ(std::get<SsmLifeCycle>(registry[optional]).reusePolicy, ReusePolicy::OPTIONAL);
    EXPECT_EQ(lifeCycle(2, "state"), required);
    EXPECT_EQ(lifeCycle(2, "replay"), privateLc);
    EXPECT_EQ(lifeCycle(2, "draft"), optional);
    EXPECT_EQ(lifeCycle(3, "replay"), privateLc);

    auto const layerLifeCycles = storage.layerToLifeCycleIds();
    EXPECT_EQ(layerLifeCycles.size(), 4U);
    EXPECT_EQ(layerLifeCycles.at(0), (std::vector<LifeCycleId>{LifeCycleId{0}}));
    EXPECT_EQ(layerLifeCycles.at(1), (std::vector<LifeCycleId>{required, privateLc, optional}));
    EXPECT_EQ(layerLifeCycles.at(2), (std::vector<LifeCycleId>{required, privateLc, optional}));
    EXPECT_EQ(layerLifeCycles.at(3), (std::vector<LifeCycleId>{privateLc}));

    auto const grouping = storage.lifeCycleGrouping();
    EXPECT_NE(grouping[required], grouping[privateLc]);
    EXPECT_NE(grouping[required], grouping[optional]);
    EXPECT_NE(grouping[privateLc], grouping[optional]);
}

TEST(KvCacheManagerV2ConfigTest, RegionsKeepLayerLifecycleIdsAndBufferAttributes)
{
    // The replay-region layout against the same layers with no region: every layer buffer keeps its
    // lifecycle id and slot offset, and the region's lifecycle is numbered after all of them.
    auto const withRegion = makeReplayRegionConfig(32 << 20, std::nullopt);
    auto withoutRegion = withRegion;
    withoutRegion.regions.clear();
    ASSERT_NO_THROW(withRegion.validate());
    ASSERT_NO_THROW(withoutRegion.validate());

    LifeCycleRegistry const plain(withoutRegion);
    LifeCycleRegistry const registry(withRegion);
    ASSERT_EQ(plain.size(), LifeCycleId{3});
    ASSERT_EQ(registry.size(), LifeCycleId{4});
    for (LifeCycleId id{0}; id < plain.size(); ++id)
    {
        SCOPED_TRACE(id.value());
        EXPECT_EQ(registry[id], plain[id]);
    }
    LifeCycleId const region{3};
    EXPECT_EQ(registry.numRegions(), 1);
    EXPECT_EQ(registry.regionLifeCycleId(0), region);
    EXPECT_EQ(registry.regionLifeCycleId(std::string{"replay"}), region);
    EXPECT_THROW(static_cast<void>(registry.regionLifeCycleId(std::string{"missing"})), std::out_of_range);
    EXPECT_EQ(registry[region], (LifeCycle{SsmLifeCycle{ReusePolicy::PRIVATE, 0}}));
    EXPECT_EQ(registry.ssmLifeCycleIds(), (std::vector<LifeCycleId>{LifeCycleId{1}, region}));
    EXPECT_EQ(registry.requiredSsmLifeCycleId(), LifeCycleId{1});

    auto const plainStorage = createStorageConfig(withoutRegion);
    auto const storage = createStorageConfig(withRegion);
    auto const attributes = storage.bufferAttributes();
    for (auto const& [buffer, attr] : plainStorage.bufferAttributes())
    {
        SCOPED_TRACE(std::to_string(buffer.layerId) + "/" + buffer.role);
        auto const& other = attributes.at(buffer);
        EXPECT_EQ(other.lifeCycleId, attr.lifeCycleId);
        EXPECT_EQ(other.offset, attr.offset);
        EXPECT_EQ(other.size, attr.size);
    }
    // Both layers' old_x buffers coalesce into one region slot, in layer order.
    for (LayerId const layerId : {1, 2})
    {
        auto const& attr = attributes.at(BufferId{layerId, "old_x"});
        EXPECT_EQ(attr.lifeCycleId, region);
        EXPECT_EQ(attr.offset, layerId == 1 ? 0U : 2048U);
        EXPECT_EQ(attr.size, 2048U);
        EXPECT_EQ(storage.layerToLifeCycleIds().at(layerId), (std::vector<LifeCycleId>{LifeCycleId{1}, region}));
    }
    // A region without a fixed slot count joins the pool group of equal policy and slot sizes.
    auto const grouping = storage.lifeCycleGrouping();
    EXPECT_EQ(storage.slotDescList.size(), plainStorage.slotDescList.size());
    EXPECT_EQ(grouping[region], grouping[LifeCycleId{2}]);
    for (auto const& fixed : storage.fixedSlotCounts)
    {
        EXPECT_FALSE(fixed.has_value());
    }
}

TEST(KvCacheManagerV2ConfigTest, RegionLifecyclesAreNeverMerged)
{
    // A PRIVATE override on layer 1 gives SsmLifeCycle{PRIVATE}; two PRIVATE request regions with the same
    // policy, one fixed and one not, sit on layer 2. All three stay separate lifecycles.
    auto config = makeTieredConfig();
    config.commitMinSnapshot = true;
    config.layers.push_back(SsmLayerConfig{.layerId = 1,
        .buffers = {BufferConfig{.role = "state", .size = 4096},
            BufferConfig{.role = "replay", .size = 2048, .reusePolicy = ReusePolicy::PRIVATE}}});
    config.layers.push_back(SsmLayerConfig{.layerId = 2, .buffers = {BufferConfig{.role = "state", .size = 4096}}});
    config.regions.push_back(RegionConfig{.name = "first",
        .buffers = {RegionBuffer{.layerId = 2, .buffer = BufferConfig{.role = "replay", .size = 2048}}},
        .fixedSlotCount = 5});
    config.regions.push_back(RegionConfig{.name = "second",
        .buffers = {RegionBuffer{.layerId = 2, .buffer = BufferConfig{.role = "count", .size = 2048}}}});
    config.regions.push_back(RegionConfig{.name = "third",
        .buffers = {RegionBuffer{.layerId = 1, .buffer = BufferConfig{.role = "count", .size = 2048}}},
        .fixedSlotCount = 7});
    ASSERT_NO_THROW(config.validate());

    LifeCycleRegistry const registry(config);
    std::vector<LifeCycle> const expected{AttnLifeCycle{}, SsmLifeCycle{ReusePolicy::REQUIRED},
        SsmLifeCycle{ReusePolicy::PRIVATE}, SsmLifeCycle{ReusePolicy::PRIVATE, 0},
        SsmLifeCycle{ReusePolicy::PRIVATE, 1}, SsmLifeCycle{ReusePolicy::PRIVATE, 2}};
    ASSERT_EQ(registry.size(), LifeCycleId{static_cast<int>(expected.size())});
    for (LifeCycleId id{0}; id < registry.size(); ++id)
    {
        SCOPED_TRACE(id.value());
        EXPECT_EQ(registry[id], expected.at(id.value()));
    }

    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    EXPECT_EQ(attributes.at(BufferId{1, "replay"}).lifeCycleId, LifeCycleId{2});
    EXPECT_EQ(attributes.at(BufferId{2, "replay"}).lifeCycleId, LifeCycleId{3});
    EXPECT_EQ(attributes.at(BufferId{2, "count"}).lifeCycleId, LifeCycleId{4});
    EXPECT_EQ(attributes.at(BufferId{1, "count"}).lifeCycleId, LifeCycleId{5});

    // The override lifecycle and the unfixed region have equal policy and slot sizes and share a pool group;
    // each fixed region has one of its own, holding only its lifecycle and keeping its own count.
    auto const grouping = storage.lifeCycleGrouping();
    EXPECT_EQ(grouping[LifeCycleId{2}], grouping[LifeCycleId{4}]);
    EXPECT_NE(grouping[LifeCycleId{3}], grouping[LifeCycleId{2}]);
    EXPECT_NE(grouping[LifeCycleId{5}], grouping[LifeCycleId{2}]);
    EXPECT_NE(grouping[LifeCycleId{3}], grouping[LifeCycleId{5}]);
    ASSERT_EQ(storage.fixedSlotCounts.size(), storage.slotDescList.size());
    for (PoolGroupIndex pgIdx{0}; pgIdx < storage.slotDescList.size(); ++pgIdx)
    {
        SCOPED_TRACE(pgIdx.value());
        std::optional<SlotCount> expectedFixed;
        if (pgIdx == grouping[LifeCycleId{3}])
        {
            expectedFixed = 5;
        }
        else if (pgIdx == grouping[LifeCycleId{5}])
        {
            expectedFixed = 7;
        }
        EXPECT_EQ(storage.fixedSlotCounts[pgIdx], expectedFixed);
        if (expectedFixed.has_value())
        {
            EXPECT_EQ(storage.slotDescList[pgIdx].variants.size(), 1U);
        }
    }
}

TEST(KvCacheManagerV2ConfigTest, FixedRegionGetsAPoolGroupOfItsOwn)
{
    LifeCycleId const privateAttention{2};
    auto const elastic = createStorageConfig(makeReplayRegionConfig(32 << 20, std::nullopt));
    EXPECT_EQ(elastic.lifeCycleGrouping().raw(),
        (std::vector<PoolGroupIndex>{PoolGroupIndex{0}, PoolGroupIndex{1}, PoolGroupIndex{2}, PoolGroupIndex{2}}));

    auto config = makeReplayRegionConfig(32 << 20, 9);
    // A SINK_REQUIRED_ONLY windowed layer beside the region stays ratio-sized.
    config.layers.emplace_back(AttentionLayerConfig{.layerId = 4,
        .buffers = {BufferConfig{.role = "key", .size = 4096}},
        .slidingWindowSize = 12,
        .numSinkTokens = 4,
        .reusePolicy = ReusePolicy::SINK_REQUIRED_ONLY});
    ASSERT_NO_THROW(config.validate());
    auto const fixed = createStorageConfig(config);
    auto const grouping = fixed.lifeCycleGrouping();
    // The SINK_REQUIRED_ONLY layer is a layer lifecycle, so the region still follows every layer lifecycle.
    LifeCycleId const sinkLayer{3};
    LifeCycleId const fixedRegion{4};
    EXPECT_EQ(LifeCycleRegistry(config).regionLifeCycleId(0), fixedRegion);
    EXPECT_NE(grouping[fixedRegion], grouping[privateAttention]);
    EXPECT_NE(grouping[fixedRegion], grouping[sinkLayer]);
    EXPECT_EQ(fixed.slotDescList[grouping[fixedRegion]].variants.size(), 1U);
    EXPECT_EQ(fixed.slotDescList[grouping[fixedRegion]].slotSizeList(),
        fixed.slotDescList[grouping[privateAttention]].slotSizeList());
    ASSERT_EQ(fixed.fixedSlotCounts.size(), fixed.slotDescList.size());
    for (PoolGroupIndex pgIdx{0}; pgIdx < fixed.slotDescList.size(); ++pgIdx)
    {
        SCOPED_TRACE(pgIdx.value());
        EXPECT_EQ(fixed.fixedSlotCounts[pgIdx],
            pgIdx == grouping[fixedRegion] ? std::optional<SlotCount>{9} : std::optional<SlotCount>{});
    }
}

TEST(KvCacheManagerV2ConfigTest, InvalidRegionsAreRejected)
{
    auto const base = makeReplayRegionConfig(32 << 20, 4);
    ASSERT_NO_THROW(base.validate());
    auto const expectRejected = [&](std::string const& what, auto&& mutate)
    {
        SCOPED_TRACE(what);
        auto config = base;
        mutate(config);
        EXPECT_THROW(config.validate(), std::invalid_argument);
    };
    expectRejected("empty name", [](KVCacheManagerConfig& c) { c.regions[0].name.clear(); });
    expectRejected("token-block region",
        [](KVCacheManagerConfig& c) { c.regions[0].allocationUnit = RegionAllocationUnit::kTokenBlock; });
    for (auto const policy : {ReusePolicy::REQUIRED, ReusePolicy::OPTIONAL, ReusePolicy::SINK_REQUIRED_ONLY})
    {
        expectRejected("non-PRIVATE policy", [policy](KVCacheManagerConfig& c) { c.regions[0].reusePolicy = policy; });
    }
    expectRejected("no buffers", [](KVCacheManagerConfig& c) { c.regions[0].buffers.clear(); });
    for (std::int64_t const count : {0, -1})
    {
        expectRejected(
            "non-positive fixed slot count", [count](KVCacheManagerConfig& c) { c.regions[0].fixedSlotCount = count; });
    }
    expectRejected("buffer-level reuse policy",
        [](KVCacheManagerConfig& c) { c.regions[0].buffers[0].buffer.reusePolicy = ReusePolicy::PRIVATE; });
    expectRejected("tokens-per-block override",
        [](KVCacheManagerConfig& c) { c.regions[0].buffers[0].buffer.tokensPerBlockOverride = 2; });
    expectRejected("sparse buffer", [](KVCacheManagerConfig& c) { c.regions[0].buffers[0].buffer.isSparse = true; });
    expectRejected("buffer declared twice",
        [](KVCacheManagerConfig& c) { c.regions[0].buffers.push_back(c.regions[0].buffers[0]); });
    expectRejected(
        "buffer on an attention layer", [](KVCacheManagerConfig& c) { c.regions[0].buffers[0].layerId = 0; });
    expectRejected("buffer on a missing layer", [](KVCacheManagerConfig& c) { c.regions[0].buffers[0].layerId = 9; });
    expectRejected(
        "buffer a layer config owns", [](KVCacheManagerConfig& c) { c.regions[0].buffers[0].buffer.role = "state"; });
    expectRejected("buffer another region owns",
        [](KVCacheManagerConfig& c)
        {
            auto other = c.regions[0];
            other.name = "other";
            c.regions.push_back(std::move(other));
        });
    expectRejected("duplicate region name",
        [](KVCacheManagerConfig& c)
        {
            auto other = c.regions[0];
            other.buffers = {RegionBuffer{.layerId = 1, .buffer = BufferConfig{.role = "count", .size = 64}}};
            c.regions.push_back(std::move(other));
        });
}

} // namespace
