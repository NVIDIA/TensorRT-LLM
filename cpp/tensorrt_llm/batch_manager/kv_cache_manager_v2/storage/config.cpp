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
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// StorageConfig methods
// ---------------------------------------------------------------------------

LifeCycleId StorageConfig::numLifeCycles() const
{
    LifeCycleId count{0};
    for (auto const& sd : slotDescList)
    {
        count += static_cast<int>(sd.variants.size());
    }
    return count;
}

TypedVec<LifeCycleId, PoolGroupIndex> StorageConfig::lifeCycleGrouping() const
{
    LifeCycleId n = numLifeCycles();
    TypedVec<LifeCycleId, PoolGroupIndex> ret(n, PoolGroupIndex{-1});
    for (PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < slotDescList.size(); ++poolGroupIndex)
    {
        for (auto const& variant : slotDescList[poolGroupIndex].variants)
        {
            ret[variant.lifeCycleId] = poolGroupIndex;
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
            for (PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
            {
                auto const& cb = variant.coalescedBuffers[poolIndex];
                size_t offset = 0;
                for (auto const& bufId : cb.bufferIds)
                {
                    int exp = 1;
                    auto it = expansion.find(bufId);
                    if (it != expansion.end())
                        exp = it->second;

                    ret[bufId] = BufferAttr{lcId, poolIndex, offset, cb.singleBufferSize, exp};
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
            TLLM_CHECK_DEBUG(it->second == attr.lifeCycleId);
        }
    }
    return map;
}

std::map<LayerId, LayerAttr> StorageConfig::layerAttributes() const
{
    std::map<LayerId, LayerAttr> ret;
    for (auto const& pg : slotDescList)
    {
        for (auto const& variant : pg.variants)
        {
            LifeCycleId lcId = variant.lifeCycleId;
            for (PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
            {
                auto const& cb = variant.coalescedBuffers[poolIndex];
                // Count how many buffers per layer in this coalesced buffer.
                std::unordered_map<LayerId, int> slotUtilPerLayer;
                for (auto const& bufId : cb.bufferIds)
                {
                    slotUtilPerLayer[bufId.layerId] += 1;
                }
                int buffersPerSlot = cb.numBuffers();

                for (auto const& [layerId, count] : slotUtilPerLayer)
                {
                    auto it = ret.find(layerId);
                    if (it == ret.end())
                    {
                        LayerAttr attr;
                        attr.lifeCycleId = lcId;
                        attr.slotUtil.resize(variant.coalescedBuffers.size(), 0);
                        attr.slotUtilFracMax = Rational{0, 1};
                        it = ret.emplace(layerId, std::move(attr)).first;
                    }
                    auto& attr = it->second;
                    TLLM_CHECK_DEBUG(attr.lifeCycleId == lcId);
                    attr.slotUtil[poolIndex] = count;
                    Rational frac{count, buffersPerSlot};
                    if (frac > attr.slotUtilFracMax)
                    {
                        attr.slotUtilFracMax = frac;
                    }
                }
            }
        }
    }
    return ret;
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
        auto sizes = sg.slotSizeList();
        poolGroupsBySizes[sizes.raw()].push_back(std::move(sg));
    }

    StorageConfig out;
    out.cacheTiers = TypedVec<CacheLevel, CacheTierConfig>{config.cacheTiers};
    out.expansion = expansionMap;

    for (auto& [sizes, variants] : poolGroupsBySizes)
    {
        SlotDesc sd;
        sd.variants = std::move(variants);
        out.slotDescList.push_back(std::move(sd));
    }

    // A21: Assert all life_cycle_ids across all SlotDescVariants are unique.
    // Mirrors Python StorageConfig.__post_init__:
    //   all_life_cycle_ids = [lc_id for variant in self.slot_desc_list for lc_id in variant.life_cycle_ids]
    //   assert len(all_life_cycle_ids) == len(set(all_life_cycle_ids))
    if (TLLM_UNLIKELY(gDebug))
    {
        std::unordered_set<LifeCycleId> allLcIds;
        for (auto const& sd : out.slotDescList)
        {
            for (auto const& variant : sd.variants)
            {
                [[maybe_unused]] bool inserted = allLcIds.insert(variant.lifeCycleId).second;
                TLLM_CHECK_WITH_INFO(inserted, "Duplicate life_cycle_id across SlotDescVariants");
            }
        }
    }

    return out;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
