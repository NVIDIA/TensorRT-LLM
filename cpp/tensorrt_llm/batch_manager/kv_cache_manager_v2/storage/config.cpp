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
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>

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
                std::unordered_map<LayerId, int> buffersPerLayer;
                for (auto const& bufId : cb.bufferIds)
                {
                    ++buffersPerLayer[bufId.layerId];
                }
                size_t offset = 0;
                for (auto const& bufId : cb.bufferIds)
                {
                    int exp = 1;
                    auto it = expansion.find(bufId);
                    if (it != expansion.end())
                        exp = it->second;

                    ret[bufId] = BufferAttr{
                        lcId, poolIndex, offset, cb.singleBufferSize, exp, buffersPerLayer.at(bufId.layerId)};
                    offset += cb.singleBufferSize;
                }
            }
        }
    }
    return ret;
}

std::unordered_map<LayerId, std::vector<LifeCycleId>> StorageConfig::layerToLifeCycleIds() const
{
    std::unordered_map<LayerId, std::vector<LifeCycleId>> map;
    for (auto const& [bufId, attr] : bufferAttributes())
    {
        auto& ids = map[bufId.layerId];
        if (std::find(ids.begin(), ids.end(), attr.lifeCycleId) == ids.end())
        {
            ids.push_back(attr.lifeCycleId);
        }
    }
    for (auto& [layerId, ids] : map)
    {
        std::sort(ids.begin(), ids.end());
    }
    return map;
}

namespace
{

// Fixed GPU slot count of a lifecycle, or nullopt for one sized by ratio. A PRIVATE SSM lifecycle never
// publishes pages to the radix tree and holds one slot per living KvCache (beam width 1), so a bound on
// living caches fixes its demand. The headroom keeps its pool at or below maxUtilForResume, above which
// KvCache::resume() fails.
std::optional<SlotCount> fixedSlotCount(KVCacheManagerConfig const& config, LifeCycle const& lifeCycle)
{
    if (!config.maxLiveKvCaches.has_value() || !std::holds_alternative<SsmLifeCycle>(lifeCycle)
        || !isPrivateLifeCycle(lifeCycle))
    {
        return std::nullopt;
    }
    double const numSlots
        = std::ceil(static_cast<double>(*config.maxLiveKvCaches) / static_cast<double>(config.maxUtilForResume));
    if (!(numSlots >= 1.0 && numSlots <= static_cast<double>(std::numeric_limits<int>::max())))
    {
        throw std::invalid_argument("max_live_kv_caches=" + std::to_string(*config.maxLiveKvCaches)
            + " with max_util_for_resume=" + std::to_string(config.maxUtilForResume)
            + " does not give a valid fixed slot count for the private SSM layer group");
    }
    return static_cast<SlotCount>(numSlots);
}

} // namespace

// ---------------------------------------------------------------------------
// createStorageConfig — factory function.
// Mirrors _storage/_config.py::create_storage_config.
// ---------------------------------------------------------------------------
StorageConfig createStorageConfig(KVCacheManagerConfig const& config)
{
    LifeCycleRegistry registry{config};
    int tokensPerBlock = config.tokensPerBlock;

    // Map: lifeCycleId → ((coalescingGroup, expanded buffer size) → list of BufferId).
    // NOTE: Python uses insertion-ordered dict/defaultdict here. The std::map
    // grouping below intentionally remains sorted, which can make observable
    // pool-group indices/layout order differ from Python. Later lookups use
    // StorageConfig mappings, so this is not expected to affect correctness.
    using CoalescingKey = std::pair<int, size_t>;
    std::map<LifeCycleId, std::map<CoalescingKey, std::vector<BufferId>>> bufferGroups;
    std::unordered_map<BufferId, int, BufferIdHash> expansionMap;

    for (auto const& layer : config.layers)
    {
        LayerId const layerId = std::visit([](auto const& cfg) { return cfg.layerId; }, layer);
        for (auto const& buf : layerBuffers(layer))
        {
            LifeCycleId const lcId = registry.getId(makeLifeCycle(layer, buf, tokensPerBlock));
            int tpbo = buf.tokensPerBlockOverride.value_or(tokensPerBlock);
            int exp = exactDiv(tokensPerBlock, tpbo);
            size_t expandedSize = buf.size * static_cast<size_t>(exp);

            BufferId bid{layerId, buf.role};
            expansionMap[bid] = exp;
            bufferGroups[lcId][{buf.coalescingGroup, expandedSize}].push_back(bid);
        }
    }

    // Build one SlotDescVariant per life cycle.
    // Each variant: coalesced buffers sorted by size descending.
    std::vector<SlotDescVariant> slotGroups;
    slotGroups.reserve(bufferGroups.size());
    for (auto const& [lcId, coalescingMap] : bufferGroups)
    {
        SlotDescVariant var;
        var.lifeCycleId = lcId;
        for (auto const& [key, bufIds] : coalescingMap)
        {
            CoalescedBuffer cb;
            cb.singleBufferSize = key.second;
            cb.bufferIds = bufIds;
            var.coalescedBuffers.push_back(std::move(cb));
        }
        // Sort descending by size.
        std::sort(var.coalescedBuffers.begin(), var.coalescedBuffers.end(),
            [](CoalescedBuffer const& a, CoalescedBuffer const& b) { return a.size() > b.size(); });
        slotGroups.push_back(std::move(var));
    }

    // Distinct sparsity and reuse policies need separate GPU pools and eviction queues. A lifecycle with a
    // fixed capacity gets a pool group of its own, so that no ratio-sized lifecycle shares its slots.
    struct PoolGroupKey
    {
        ReusePolicy reusePolicy = ReusePolicy::REQUIRED;
        std::vector<size_t> slotSizes;
        bool isSparse = false;
        bool fixedCapacity = false;

        bool operator<(PoolGroupKey const& other) const
        {
            return std::tie(reusePolicy, slotSizes, isSparse, fixedCapacity)
                < std::tie(other.reusePolicy, other.slotSizes, other.isSparse, other.fixedCapacity);
        }
    };

    std::map<PoolGroupKey, std::vector<SlotDescVariant>> poolGroups;
    for (auto& sg : slotGroups)
    {
        auto const& lifeCycle = registry.getLifeCycle(sg.lifeCycleId);
        auto const sizes = sg.slotSizeList();
        auto const* attn = std::get_if<AttnLifeCycle>(&lifeCycle);
        bool const isSparse = attn != nullptr && attn->isSparse;
        auto const policy = reusePolicy(lifeCycle);
        bool const fixedCapacity = fixedSlotCount(config, lifeCycle).has_value();
        poolGroups
            [{.reusePolicy = policy, .slotSizes = sizes.raw(), .isSparse = isSparse, .fixedCapacity = fixedCapacity}]
                .push_back(std::move(sg));
    }

    StorageConfig out;
    out.cacheTiers = TypedVec<CacheLevel, CacheTierConfig>{config.cacheTiers};
    out.expansion = expansionMap;

    for (auto& [key, variants] : poolGroups)
    {
        std::optional<SlotCount> fixedSlots;
        if (key.fixedCapacity)
        {
            // SSM lifecycles are deduplicated by policy, so the PRIVATE SSM lifecycle is the only fixed one.
            TLLM_CHECK_WITH_INFO(variants.size() == 1, "A fixed-capacity pool group must hold exactly one lifecycle");
            fixedSlots = fixedSlotCount(config, registry.getLifeCycle(variants.front().lifeCycleId));
        }
        SlotDesc sd;
        sd.variants = std::move(variants);
        out.slotDescList.push_back(std::move(sd));
        out.fixedSlotCounts.push_back(fixedSlots);
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
