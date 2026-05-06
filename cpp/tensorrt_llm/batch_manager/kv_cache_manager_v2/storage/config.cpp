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

#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/utils/math.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// StorageConfig methods
// ---------------------------------------------------------------------------

int StorageConfig::numLifeCycles() const
{
    int count = 0;
    for (auto const& sd : slotDescList)
    {
        count += static_cast<int>(sd.variants.size());
    }
    return count;
}

std::vector<PoolGroupIndex> StorageConfig::lifeCycleGrouping() const
{
    int n = numLifeCycles();
    std::vector<PoolGroupIndex> ret(static_cast<size_t>(n), -1);
    for (int pgIdx = 0; pgIdx < static_cast<int>(slotDescList.size()); ++pgIdx)
    {
        for (auto const& variant : slotDescList[pgIdx].variants)
        {
            ret[static_cast<size_t>(variant.lifeCycleId)] = pgIdx;
        }
    }
    return ret;
}

std::map<BufferId, BufferAttr> StorageConfig::bufferAttributes() const
{
    std::map<BufferId, BufferAttr> ret;
    for (auto const& sd : slotDescList)
    {
        for (auto const& variant : sd.variants)
        {
            LifeCycleId lcId = variant.lifeCycleId;
            for (int poolIdx = 0; poolIdx < static_cast<int>(variant.coalescedBuffers.size()); ++poolIdx)
            {
                auto const& cb = variant.coalescedBuffers[poolIdx];
                size_t offset = 0;
                for (auto const& bufId : cb.bufferIds)
                {
                    int exp = 1;
                    auto it = expansion.find(bufId);
                    if (it != expansion.end())
                        exp = it->second;

                    ret[bufId] = BufferAttr{lcId, poolIdx, offset, cb.singleBufferSize, exp};
                    offset += cb.singleBufferSize;
                }
            }
        }
    }
    return ret;
}

std::unordered_map<LayerId, LifeCycleId> StorageConfig::layerToLifeCycleIds() const
{
    std::unordered_map<LayerId, LifeCycleId> map;
    for (auto const& [bufId, attr] : bufferAttributes())
    {
        auto [it, inserted] = map.emplace(bufId.layerId, attr.lifeCycleId);
        if (!inserted)
        {
            assert(it->second == attr.lifeCycleId);
        }
    }
    return map;
}

// ---------------------------------------------------------------------------
// createStorageConfig — factory function.
// Mirrors _storage/_config.py::create_storage_config.
// ---------------------------------------------------------------------------
StorageConfig createStorageConfig(KVCacheManagerConfig const& config)
{
    LifeCycleRegistry registry{config};
    int tokensPerBlock = config.tokensPerBlock;

    // Map: lifeCycleId → (bufferSize → list of BufferId).
    // Outer map key is LifeCycleId; inner map key is expanded buffer size.
    // NOTE: Python uses insertion-ordered dict/defaultdict here. The std::map
    // grouping below intentionally remains sorted, which can make observable
    // pool-group indices/layout order differ from Python. Later lookups use
    // StorageConfig mappings, so this is not expected to affect correctness.
    std::map<LifeCycleId, std::map<size_t, std::vector<BufferId>>> bufferGroups;
    std::unordered_map<BufferId, int, BufferIdHash> expansionMap;

    for (auto const& layer : config.layers)
    {
        LifeCycle lc = makeLifeCycle(layer, tokensPerBlock);
        LifeCycleId lcId = registry.getId(lc);

        std::visit(
            [&](auto const& cfg)
            {
                for (auto const& buf : cfg.buffers)
                {
                    int tpbo = buf.tokensPerBlockOverride.value_or(tokensPerBlock);
                    int exp = exactDiv(tokensPerBlock, tpbo);
                    size_t expandedSize = buf.size * static_cast<size_t>(exp);

                    BufferId bid{cfg.layerId, buf.role};
                    expansionMap[bid] = exp;
                    bufferGroups[lcId][expandedSize].push_back(bid);
                }
            },
            layer);
    }

    // Build one SlotDescVariant per life cycle.
    // Each variant: coalesced buffers sorted by size descending.
    std::vector<SlotDescVariant> slotGroups;
    slotGroups.reserve(bufferGroups.size());
    for (auto const& [lcId, sizeMap] : bufferGroups)
    {
        SlotDescVariant var;
        var.lifeCycleId = lcId;
        for (auto const& [sz, bufIds] : sizeMap)
        {
            CoalescedBuffer cb;
            cb.singleBufferSize = sz;
            cb.bufferIds = bufIds;
            var.coalescedBuffers.push_back(std::move(cb));
        }
        // Sort descending by size.
        std::sort(var.coalescedBuffers.begin(), var.coalescedBuffers.end(),
            [](CoalescedBuffer const& a, CoalescedBuffer const& b) { return a.size() > b.size(); });
        slotGroups.push_back(std::move(var));
    }

    // Merge SlotDescVariants that share the same slotSizeList.
    // Key: tuple of sizes (sorted desc).
    std::map<std::vector<size_t>, std::vector<SlotDescVariant>> poolGroupsBySizes;
    for (auto& sg : slotGroups)
    {
        poolGroupsBySizes[sg.slotSizeList()].push_back(std::move(sg));
    }

    StorageConfig out;
    out.cacheTiers = config.cacheTiers;
    out.expansion = expansionMap;

    for (auto& [sizes, variants] : poolGroupsBySizes)
    {
        SlotDesc sd;
        sd.variants = std::move(variants);
        out.slotDescList.push_back(std::move(sd));
    }

    return out;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
