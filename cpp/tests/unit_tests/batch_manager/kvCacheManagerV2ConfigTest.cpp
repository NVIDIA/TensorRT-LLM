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

    // The layer lifecycles first (attention, the REQUIRED SSM state, layer 3's layer-level PRIVATE),
    // then the OPTIONAL one only an override creates; the PRIVATE overrides join layer 3's lifecycle.
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

TEST(KvCacheManagerV2ConfigTest, BufferOverridesNumberTheirLifecyclesAfterLayerLifecycles)
{
    // Full attention, two SSM layers, windowed attention and a layer-level PRIVATE SSM layer, with no
    // buffer overrides; then the same config with PRIVATE replay buffers on the first two SSM layers
    // (a group no layer has on its own) and OPTIONAL ones on the second (a new group too), placed
    // before the later layers' buffers.
    auto const makeLayers = [](bool overrides)
    {
        auto config = makeTieredConfig();
        config.commitMinSnapshot = true;
        for (LayerId layerId = 1; layerId <= 2; ++layerId)
        {
            SsmLayerConfig ssm{.layerId = layerId, .buffers = {BufferConfig{.role = "state", .size = 4096}}};
            if (overrides)
            {
                ssm.buffers.push_back(BufferConfig{.role = "old_x", .size = 2048, .reusePolicy = ReusePolicy::PRIVATE});
                if (layerId == 2)
                {
                    ssm.buffers.push_back(
                        BufferConfig{.role = "draft", .size = 2048, .reusePolicy = ReusePolicy::OPTIONAL});
                }
            }
            // An override equal to the layer's policy is no override.
            ssm.buffers.push_back(BufferConfig{.role = "conv", .size = 1024, .reusePolicy = ReusePolicy::REQUIRED});
            config.layers.emplace_back(std::move(ssm));
        }
        config.layers.emplace_back(AttentionLayerConfig{.layerId = 3,
            .buffers = {BufferConfig{.role = "key", .size = 4096}},
            .slidingWindowSize = 8,
            .reusePolicy = ReusePolicy::PRIVATE});
        config.layers.emplace_back(SsmLayerConfig{.layerId = 4,
            .buffers = {BufferConfig{.role = "state", .size = 4096}},
            .reusePolicy = ReusePolicy::OPTIONAL});
        return config;
    };
    auto const plain = makeLayers(false);
    auto const split = makeLayers(true);
    ASSERT_NO_THROW(plain.validate());
    ASSERT_NO_THROW(split.validate());
    LifeCycleRegistry const plainRegistry(plain);
    LifeCycleRegistry const splitRegistry(split);

    // Every lifecycle of the plain config keeps its id, and the override-only ones follow in buffer order.
    std::vector<LifeCycle> const expected{AttnLifeCycle{}, SsmLifeCycle{ReusePolicy::REQUIRED},
        AttnLifeCycle::make(8, std::nullopt, split.tokensPerBlock, ReusePolicy::PRIVATE),
        SsmLifeCycle{ReusePolicy::OPTIONAL}, SsmLifeCycle{ReusePolicy::PRIVATE}};
    ASSERT_EQ(plainRegistry.size(), LifeCycleId{4});
    ASSERT_EQ(splitRegistry.size(), LifeCycleId{5});
    for (LifeCycleId id{0}; id < splitRegistry.size(); ++id)
    {
        SCOPED_TRACE(id.value());
        EXPECT_EQ(splitRegistry[id], expected.at(id.value()));
        if (id < plainRegistry.size())
        {
            EXPECT_EQ(plainRegistry[id], splitRegistry[id]);
        }
    }
    EXPECT_EQ(
        splitRegistry.ssmLifeCycleIds(), (std::vector<LifeCycleId>{LifeCycleId{1}, LifeCycleId{3}, LifeCycleId{4}}));

    // The storage config assigns the buffers the same ids, and the OPTIONAL draft buffer joins layer 4's group.
    auto const plainStorage = createStorageConfig(plain);
    auto const splitStorage = createStorageConfig(split);
    for (auto const& [buffer, attr] : plainStorage.bufferAttributes())
    {
        SCOPED_TRACE(std::to_string(buffer.layerId) + "/" + buffer.role);
        EXPECT_EQ(splitStorage.bufferAttributes().at(buffer).lifeCycleId, attr.lifeCycleId);
    }
    auto const lifeCycle = [&](LayerId layerId, DataRole const& role) {
        return splitStorage.bufferAttributes().at(BufferId{layerId, role}).lifeCycleId;
    };
    EXPECT_EQ(lifeCycle(1, "old_x"), LifeCycleId{4});
    EXPECT_EQ(lifeCycle(2, "old_x"), LifeCycleId{4});
    EXPECT_EQ(lifeCycle(2, "draft"), LifeCycleId{3});
    EXPECT_EQ(lifeCycle(1, "conv"), LifeCycleId{1});
}

TEST(KvCacheManagerV2ConfigTest, MaxLiveKvCachesMustBePositive)
{
    auto config = makeConfig();
    EXPECT_FALSE(config.maxLiveKvCaches.has_value());
    for (int const value : {1, 64})
    {
        config.maxLiveKvCaches = value;
        EXPECT_NO_THROW(config.validate());
    }
    for (int const value : {0, -1})
    {
        config.maxLiveKvCaches = value;
        EXPECT_THROW(config.validate(), std::invalid_argument);
    }

    // The fixed private SSM capacity divides by max_util_for_resume.
    config.maxLiveKvCaches = 8;
    for (float const util : {0.0F, -0.5F, 1.5F})
    {
        config.maxUtilForResume = util;
        EXPECT_THROW(config.validate(), std::invalid_argument);
    }
    config.maxUtilForResume = 1.0F;
    EXPECT_NO_THROW(config.validate());
}

TEST(KvCacheManagerV2ConfigTest, MaxLiveKvCachesGivesPrivateSsmLifecycleAFixedPoolGroup)
{
    using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeReservedSsmConfig;
    LifeCycleId const attention{0};
    LifeCycleId const requiredSsm{1};
    LifeCycleId const privateAttention{2};
    LifeCycleId const privateSsm{3};

    // Without the bound the PRIVATE SSM and PRIVATE attention lifecycles share a pool group of equal slots.
    auto const elastic = createStorageConfig(makeReservedSsmConfig(32 << 20, std::nullopt));
    auto const elasticGrouping = elastic.lifeCycleGrouping();
    EXPECT_EQ(elasticGrouping.raw(),
        (std::vector<PoolGroupIndex>{PoolGroupIndex{0}, PoolGroupIndex{1}, PoolGroupIndex{2}, PoolGroupIndex{2}}));
    ASSERT_EQ(elastic.fixedSlotCounts.size(), PoolGroupIndex{3});
    for (auto const& fixed : elastic.fixedSlotCounts)
    {
        EXPECT_FALSE(fixed.has_value());
    }

    // ceil(N / maxUtilForResume) slots, in a group of its own; the other groups keep their order.
    for (auto const& [maxUtilForResume, expectedSlots] : {std::pair{0.97F, 9}, std::pair{0.8F, 10}, std::pair{1.0F, 8}})
    {
        SCOPED_TRACE(maxUtilForResume);
        auto config = makeReservedSsmConfig(32 << 20, 8);
        config.maxUtilForResume = maxUtilForResume;
        ASSERT_NO_THROW(config.validate());
        auto const reserved = createStorageConfig(config);
        auto const grouping = reserved.lifeCycleGrouping();
        EXPECT_EQ(reserved.slotDescList.size(), PoolGroupIndex{4});
        EXPECT_EQ(grouping[attention], PoolGroupIndex{0});
        EXPECT_EQ(grouping[requiredSsm], PoolGroupIndex{1});
        EXPECT_EQ(grouping[privateAttention], PoolGroupIndex{2});
        EXPECT_EQ(grouping[privateSsm], PoolGroupIndex{3});
        EXPECT_EQ(reserved.slotDescList[grouping[privateSsm]].variants.size(), 1U);
        EXPECT_EQ(reserved.slotDescList[grouping[privateSsm]].slotSizeList(),
            reserved.slotDescList[grouping[privateAttention]].slotSizeList());
        ASSERT_EQ(reserved.fixedSlotCounts.size(), PoolGroupIndex{4});
        for (PoolGroupIndex pgIdx{0}; pgIdx < reserved.fixedSlotCounts.size(); ++pgIdx)
        {
            if (pgIdx == grouping[privateSsm])
            {
                EXPECT_EQ(reserved.fixedSlotCounts[pgIdx], std::optional<SlotCount>{expectedSlots});
            }
            else
            {
                EXPECT_FALSE(reserved.fixedSlotCounts[pgIdx].has_value());
            }
        }
    }

    // The bound alone changes nothing without a PRIVATE SSM lifecycle.
    auto config = makeReservedSsmConfig(32 << 20, 8);
    for (auto& layer : config.layers)
    {
        if (auto* ssm = std::get_if<SsmLayerConfig>(&layer))
        {
            ssm->buffers.back().reusePolicy.reset();
        }
    }
    auto const withoutPrivateSsm = createStorageConfig(config);
    EXPECT_EQ(withoutPrivateSsm.slotDescList.size(), PoolGroupIndex{3});
    for (auto const& fixed : withoutPrivateSsm.fixedSlotCounts)
    {
        EXPECT_FALSE(fixed.has_value());
    }
}

TEST(KvCacheManagerV2ConfigTest, SinkRequiredOnlyGroupsStayElasticBesideTheFixedPrivateSsmGroup)
{
    // The reserved-SSM layout with its windowed layer (3) on SINK_REQUIRED_ONLY, then two REQUIRED windowed
    // layers whose key buffers override to SINK_REQUIRED_ONLY: layer 4 with another window (a lifecycle only
    // the override creates) and layer 5 with layer 3's window (the override joins layer 3's lifecycle).
    using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeReservedSsmConfig;
    auto config = makeReservedSsmConfig(32 << 20, 8);
    auto& windowed = std::get<AttentionLayerConfig>(config.layers.back());
    windowed.numSinkTokens = 4;
    windowed.reusePolicy = ReusePolicy::SINK_REQUIRED_ONLY;
    for (auto const& [layerId, window] : {std::pair<LayerId, int>{4, 12}, std::pair<LayerId, int>{5, 8}})
    {
        config.layers.emplace_back(AttentionLayerConfig{.layerId = layerId,
            .buffers = {BufferConfig{.role = "key", .size = 4096, .reusePolicy = ReusePolicy::SINK_REQUIRED_ONLY}},
            .slidingWindowSize = window,
            .numSinkTokens = 4});
    }
    ASSERT_NO_THROW(config.validate());

    // Layer lifecycles first, the SINK_REQUIRED_ONLY layer among them although it is declared after the SSM
    // layers; then the override-only lifecycles in buffer order, the PRIVATE SSM replay group before layer 4's.
    LifeCycleRegistry const registry(config);
    std::vector<LifeCycle> const expected{AttnLifeCycle{}, SsmLifeCycle{ReusePolicy::REQUIRED},
        AttnLifeCycle::make(8, 4, config.tokensPerBlock, ReusePolicy::SINK_REQUIRED_ONLY),
        SsmLifeCycle{ReusePolicy::PRIVATE},
        AttnLifeCycle::make(12, 4, config.tokensPerBlock, ReusePolicy::SINK_REQUIRED_ONLY)};
    ASSERT_EQ(registry.size(), LifeCycleId{static_cast<int>(expected.size())});
    for (LifeCycleId id{0}; id < registry.size(); ++id)
    {
        SCOPED_TRACE(id.value());
        EXPECT_EQ(registry[id], expected.at(id.value()));
    }

    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const lifeCycle = [&](LayerId layerId, DataRole const& role) {
        return attributes.at(BufferId{layerId, role}).lifeCycleId;
    };
    LifeCycleId const sinkLayer{2};
    LifeCycleId const privateSsm{3};
    LifeCycleId const sinkOverride{4};
    EXPECT_EQ(lifeCycle(3, "key"), sinkLayer);
    EXPECT_EQ(lifeCycle(5, "key"), sinkLayer);
    EXPECT_EQ(lifeCycle(4, "key"), sinkOverride);
    EXPECT_EQ(lifeCycle(1, "old_x"), privateSsm);
    EXPECT_EQ(lifeCycle(2, "old_x"), privateSsm);

    // Only the PRIVATE SSM group has a fixed capacity, ceil(8 / 0.97); the SINK_REQUIRED_ONLY groups are
    // ratio-sized.
    auto const grouping = storage.lifeCycleGrouping();
    EXPECT_NE(grouping[sinkLayer], grouping[privateSsm]);
    EXPECT_NE(grouping[sinkOverride], grouping[privateSsm]);
    EXPECT_EQ(storage.slotDescList[grouping[privateSsm]].variants.size(), 1U);
    ASSERT_EQ(storage.fixedSlotCounts.size(), storage.slotDescList.size());
    for (PoolGroupIndex pgIdx{0}; pgIdx < storage.fixedSlotCounts.size(); ++pgIdx)
    {
        SCOPED_TRACE(pgIdx.value());
        EXPECT_EQ(storage.fixedSlotCounts[pgIdx],
            pgIdx == grouping[privateSsm] ? std::optional<SlotCount>{9} : std::optional<SlotCount>{});
    }
}

} // namespace
