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
#include <tuple>
#include <unordered_set>
#include <utility>

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

// ---------------------------------------------------------------------------
// createStorageConfig — factory function.
// Mirrors _storage/_config.py::create_storage_config.
// ---------------------------------------------------------------------------
StorageConfig createStorageConfig(KVCacheManagerConfig const& config)
{
    LifeCycleRegistry registry{config};
    int tokensPerBlock = config.tokensPerBlock;

    // Map: lifeCycleId → ((coalescingGroup, expanded buffer size, sparsity) → list of BufferId).
    // NOTE: Python uses insertion-ordered dict/defaultdict here. The std::map
    // grouping below intentionally remains sorted, which can make observable
    // pool-group indices/layout order differ from Python. Later lookups use
    // StorageConfig mappings, so this is not expected to affect correctness.
    // Keyed on sparsity as well as group and size: a pool is uniformly sparse or uniformly
    // dense, so one layer's buffers may land in both.
    using CoalescingKey = std::tuple<int, size_t, bool>;
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
            bufferGroups[lcId][{buf.coalescingGroup, expandedSize, buf.isSparse}].push_back(bid);
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
            [[maybe_unused]] auto const& [coalescingGroup, sz, isSparse] = key;
            CoalescedBuffer cb;
            cb.singleBufferSize = sz;
            cb.isSparse = isSparse;
            cb.bufferIds = bufIds;
            var.coalescedBuffers.push_back(std::move(cb));
        }
        // Sort descending by size. Stable, so pools of equal size keep the order their
        // keys gave them -- by coalescing group, then sparsity -- which `size()` alone
        // does not distinguish.
        std::ranges::stable_sort(var.coalescedBuffers,
            [](CoalescedBuffer const& a, CoalescedBuffer const& b) { return a.size() > b.size(); });
        slotGroups.push_back(std::move(var));
    }

    // Distinct reuse policies need separate GPU pools and eviction queues. Keying on
    // per-pool sparsity too keeps a shared pool uniformly host-backed or device-backed.
    struct PoolGroupKey
    {
        ReusePolicy reusePolicy = ReusePolicy::REQUIRED;
        std::vector<size_t> slotSizes;
        std::vector<bool> sparsity;

        bool operator<(PoolGroupKey const& other) const
        {
            return std::tie(reusePolicy, slotSizes, sparsity)
                < std::tie(other.reusePolicy, other.slotSizes, other.sparsity);
        }
    };

    std::map<PoolGroupKey, std::vector<SlotDescVariant>> poolGroups;
    for (auto& sg : slotGroups)
    {
        auto const sizes = sg.slotSizeList();
        auto const policy = reusePolicy(registry.getLifeCycle(sg.lifeCycleId));
        TypedVec<PoolIndex, bool> sparsity;
        sparsity.reserve(sizes.size());
        for (PoolIndex pool{0}; pool < sizes.size(); ++pool)
        {
            sparsity.push_back(sg.isSparse(pool));
        }
        poolGroups[{.reusePolicy = policy, .slotSizes = sizes.raw(), .sparsity = sparsity.raw()}].push_back(
            std::move(sg));
    }

    StorageConfig out;
    out.cacheTiers = TypedVec<CacheLevel, CacheTierConfig>{config.cacheTiers};
    out.hostSparsePoolQuota = config.hostSparsePoolQuota;
    out.expansion = expansionMap;

    for (auto& [key, variants] : poolGroups)
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
