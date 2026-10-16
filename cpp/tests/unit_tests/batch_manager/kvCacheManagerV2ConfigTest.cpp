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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/lifeCycleRegistry.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/config.h"

#include <gtest/gtest.h>

namespace
{
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

TEST(KvCacheManagerV2ConfigTest, AttentionLifecycleFactoryPreservesCacheDomain)
{
    auto const target = AttnLifeCycle::make(128, 5, 4);
    auto const draft = AttnLifeCycle::make(128, 5, 4, CacheDomain::kStandaloneDraft);
    EXPECT_EQ(target.numSinkBlocks, 2);
    EXPECT_EQ(target.cacheDomain, CacheDomain::kTarget);
    EXPECT_EQ(draft.cacheDomain, CacheDomain::kStandaloneDraft);
    EXPECT_FALSE(target == draft);
    EXPECT_TRUE(target < draft || draft < target);
    EXPECT_TRUE(draft == AttnLifeCycle::make(128, 8, 4, CacheDomain::kStandaloneDraft));
}

TEST(KvCacheManagerV2ConfigTest, MatchingSlotShapesSharePoolsOnlyWithinEachCacheDomain)
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = 32;
    config.cacheTiers
        = {GpuCacheTierConfig{4 << 20}, HostCacheTierConfig{4 << 20}, DiskCacheTierConfig{4 << 20, "/tmp"}};
    for (LayerId layerId = 0; layerId < 4; ++layerId)
    {
        AttentionLayerConfig layer;
        layer.layerId = layerId;
        layer.buffers = {BufferConfig{"key", 4096, std::nullopt}};
        layer.slidingWindowSize = layerId % 2 == 0 ? 128 : 256;
        layer.cacheDomain = layerId < 2 ? CacheDomain::kTarget : CacheDomain::kStandaloneDraft;
        config.layers.push_back(layer);
    }
    ASSERT_NO_THROW(config.validate());

    LifeCycleRegistry const registry(config);
    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const grouping = storage.lifeCycleGrouping();
    auto lifeCycle = [&](LayerId layerId) { return attributes.at(BufferId{layerId, "key"}).lifeCycleId; };
    auto poolGroup = [&](LayerId layerId) { return grouping[lifeCycle(layerId)]; };
    EXPECT_EQ(registry.size(), LifeCycleId{4});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{2});
    for (LayerId first = 0; first < 4; ++first)
    {
        auto const& attention = std::get<AttnLifeCycle>(registry[lifeCycle(first)]);
        EXPECT_EQ(attention.cacheDomain, first < 2 ? CacheDomain::kTarget : CacheDomain::kStandaloneDraft);
        for (LayerId second = first + 1; second < 4; ++second)
        {
            EXPECT_NE(lifeCycle(first), lifeCycle(second));
            EXPECT_FALSE(registry[lifeCycle(first)] == registry[lifeCycle(second)]);
            EXPECT_EQ(poolGroup(first) == poolGroup(second), first / 2 == second / 2);
        }
    }
}

} // namespace
