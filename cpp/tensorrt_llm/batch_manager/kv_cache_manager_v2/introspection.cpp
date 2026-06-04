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

#include "kv_cache_manager_v2/introspection.h"

#include "kv_cache_manager_v2/blockRadixTree.h"

#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storage/core.h"
#include "kv_cache_manager_v2/storageManager.h"

#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

bool allBlockPagesDroppable(Block const& block)
{
    for (auto const* page : block.storage)
    {
        if (page != nullptr && page->status() != PageStatus::DROPPABLE)
        {
            return false;
        }
    }

    for (auto const& [_, child] : block.next)
    {
        if (!allBlockPagesDroppable(*child))
        {
            return false;
        }
    }
    return true;
}

} // namespace

KvCacheIntrospection::ActivePageStats KvCacheIntrospection::activePageStats(KvCache const& kvCache)
{
    auto& storageMgr = kvCache.manager().storage();
    int const numTiers = storageMgr.numCacheLevels();
    std::vector<int> counts(static_cast<size_t>(numTiers), 0);
    std::vector<int> unscheduledEvictable(static_cast<size_t>(numTiers), 0);

    for (auto const& activePage : kvCache._activePages())
    {
        auto page = kvCache._page(activePage.ordinal, activePage.beamIdx, activePage.lcId);
        if (!page)
        {
            continue;
        }

        CacheLevel const level = page->cacheLevel;
        auto const levelIdx = static_cast<size_t>(level);
        counts.at(levelIdx) += 1;
        if (storageMgr.isEvictable(*page) && !page->scheduledForEviction())
        {
            unscheduledEvictable.at(levelIdx) += 1;
        }
    }

    return {std::move(counts), std::move(unscheduledEvictable)};
}

bool KvCacheIntrospection::isCommitAllowed(KvCache const& kvCache)
{
    return kvCache.commitState() == KvCache::CommitState::ALLOWED;
}

std::vector<float> KvCacheIntrospection::currentGpuRatio(KvCacheManager& manager)
{
    return manager.storage().getRatioList(kGpuLevel);
}

std::vector<StorageStatistics> KvCacheIntrospection::storageStatistics(KvCacheManager& manager, CacheLevel level)
{
    std::vector<StorageStatistics> result;
    int const numPoolGroups = manager.storage().numPoolGroups();
    result.reserve(static_cast<size_t>(numPoolGroups));
    for (int pg = 0; pg < numPoolGroups; ++pg)
    {
        result.push_back(manager.storage().getStatistics(level, static_cast<PoolGroupIndex>(pg)));
    }
    return result;
}

std::vector<float> KvCacheIntrospection::storageUtilization(KvCacheManager& manager, CacheLevel level)
{
    return manager.storage().getUtilization(level);
}

int64_t KvCacheIntrospection::grainsForSlots(int numSlots, std::vector<int> const& slotSizeList, int granularity)
{
    return CacheLevelStorage::grainsForSlots(numSlots, slotSizeList, granularity);
}

std::tuple<int, int64_t> KvCacheIntrospection::grainsToSlots(
    int64_t pgGrains, std::vector<int> const& slotSizeList, int granularity)
{
    auto [slots, used] = CacheLevelStorage::grainsToSlots(pgGrains, slotSizeList, granularity);
    return {slots, used};
}

std::vector<int> KvCacheIntrospection::ratioToSlotCountList(size_t quota,
    std::vector<std::vector<int>> const& slotSizeLists, std::vector<float> const& ratioList, int granularity,
    std::vector<int> const& minSlots)
{
    return CacheLevelStorage::ratioToSlotCountList(quota, slotSizeLists, ratioList, granularity, minSlots);
}

bool KvCacheIntrospection::allTreePagesDroppable(KvCacheManager& manager)
{
    for (auto const& [_, root] : manager.radixTree().roots())
    {
        for (auto const& [__, block] : root->next)
        {
            if (!allBlockPagesDroppable(*block))
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
