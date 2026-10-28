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
    return AttentionLayerConfig{
        .layerId = layerId, .buffers = {BufferConfig{.role = "key", .size = bufferSize, .isSparse = isSparse}}};
}

TEST(KvCacheManagerV2ConfigTest, ExistingBufferConfigurationDefaultsToDense)
{
    BufferConfig const buffer{"key", 4096, 2};
    EXPECT_FALSE(buffer.isSparse);
    EXPECT_EQ(buffer.tokensPerBlockOverride, 2);
    EXPECT_NO_THROW(makeConfig().validate());
}

TEST(KvCacheManagerV2ConfigTest, SparseBuffersRequireHostMemoryAtLevelOne)
{
    auto config = makeConfig();
    config.layers = {makeAttentionLayer(0, true)};
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.cacheTiers[1] = GpuCacheTierConfig{4 << 20};
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.cacheTiers[1] = HostCacheTierConfig{4 << 20};
    EXPECT_NO_THROW(config.validate());
}

TEST(KvCacheManagerV2ConfigTest, MixedBuffersInOneLayerAreRejected)
{
    auto config = makeTieredConfig();
    for (bool const firstIsSparse : {false, true})
    {
        auto layer = makeAttentionLayer(0, firstIsSparse);
        layer.buffers.push_back(BufferConfig{.role = "value", .size = 4096, .isSparse = !firstIsSparse});
        config.layers = {layer};
        EXPECT_THROW(config.validate(), std::invalid_argument);
        EXPECT_THROW(createStorageConfig(config), std::invalid_argument);
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

TEST(KvCacheManagerV2ConfigTest, SparseAndDenseBuffersHaveSeparateLifecyclesAndPools)
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

    EXPECT_EQ(registry.size(), LifeCycleId{2});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{2});
    EXPECT_NE(dense.lifeCycleId, sparse.lifeCycleId);
    EXPECT_NE(grouping[dense.lifeCycleId], grouping[sparse.lifeCycleId]);
    EXPECT_FALSE(std::get<AttnLifeCycle>(registry[dense.lifeCycleId]).isSparse);
    EXPECT_TRUE(std::get<AttnLifeCycle>(registry[sparse.lifeCycleId]).isSparse);
    EXPECT_FALSE(registry[dense.lifeCycleId] == registry[sparse.lifeCycleId]);

    auto const& otherDense = attributes.at(BufferId{2, "key"});
    auto const& otherSparse = attributes.at(BufferId{3, "key"});
    EXPECT_EQ(dense.lifeCycleId, otherDense.lifeCycleId);
    EXPECT_EQ(sparse.lifeCycleId, otherSparse.lifeCycleId);
    EXPECT_EQ(sparse.poolIndex, otherSparse.poolIndex);
    EXPECT_EQ(otherSparse.offset, sparse.offset + sparse.size);
    EXPECT_EQ(otherDense.offset, dense.offset + dense.size);
}

TEST(KvCacheManagerV2ConfigTest, MatchingSlotShapesSharePoolsWithinEachSparsity)
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
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{4});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{2});
    EXPECT_EQ(poolGroup(0), poolGroup(1));
    EXPECT_EQ(poolGroup(2), poolGroup(3));
    EXPECT_NE(poolGroup(0), poolGroup(2));
}

TEST(KvCacheManagerV2ConfigTest, SparseBuffersCoalesceByExpandedSize)
{
    auto config = makeTieredConfig();
    auto firstLayer = makeAttentionLayer(0, true, 2048);
    firstLayer.buffers.front().tokensPerBlockOverride = 2;
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
    EXPECT_EQ(first.expansion, 2);
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
    LifeCycleId const privateSsm{2};
    LifeCycleId const privateAttention{3};

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

} // namespace
