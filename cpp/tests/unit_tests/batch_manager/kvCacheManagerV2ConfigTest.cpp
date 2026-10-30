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

} // namespace
