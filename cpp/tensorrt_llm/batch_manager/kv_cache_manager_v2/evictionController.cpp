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

#include "kv_cache_manager_v2/evictionController.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/page.h"

#include <cassert>
#include <numeric>
#include <set>
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// LRUEvictionPolicy
// ---------------------------------------------------------------------------

NodeRef LRUEvictionPolicy::push(std::shared_ptr<Page> page, bool evictFirst)
{
    assert(!page->nodeRef.has_value() && "page must not already be scheduled for eviction");
    if (evictFirst)
    {
        mQueue.push_front(std::move(page));
        return mQueue.begin();
    }
    else
    {
        mQueue.push_back(std::move(page));
        auto it = mQueue.end();
        --it;
        return it;
    }
}

std::shared_ptr<Page> LRUEvictionPolicy::pop()
{
    assert(!mQueue.empty());
    auto page = mQueue.front();
    mQueue.pop_front();
    return page;
}

std::shared_ptr<Page> LRUEvictionPolicy::remove(NodeRef node)
{
    auto page = *node;
    assert(page->nodeRef.has_value() && page->nodeRef.value() == node && "node's page must reference this node");
    mQueue.erase(node);
    return page;
}

// ---------------------------------------------------------------------------
// PrioritizedEvictionPolicy
// ---------------------------------------------------------------------------

LRUEvictionPolicy& PrioritizedEvictionPolicy::getOrCreate(Priority p)
{
    return mPolicies[p]; // std::map default-constructs if not present
}

NodeRef PrioritizedEvictionPolicy::push(std::shared_ptr<Page> page, bool evictFirst)
{
    Priority p = page->priority;
    LRUEvictionPolicy& policy = getOrCreate(p);
    return policy.push(std::move(page), evictFirst);
}

std::shared_ptr<Page> PrioritizedEvictionPolicy::pop()
{
    assert(!mPolicies.empty());
    // Lowest priority key evicted first (std::map iterates in ascending key order)
    auto it = mPolicies.begin();
    auto page = it->second.pop();
    if (it->second.empty())
    {
        mPolicies.erase(it);
    }
    return page;
}

std::shared_ptr<Page> PrioritizedEvictionPolicy::remove(NodeRef node)
{
    auto page = *node;
    Priority p = page->priority;
    auto it = mPolicies.find(p);
    assert(it != mPolicies.end());
    it->second.remove(node);
    if (it->second.empty())
    {
        mPolicies.erase(it);
    }
    return page;
}

size_t PrioritizedEvictionPolicy::size() const noexcept
{
    size_t total = 0;
    for (auto const& [p, policy] : mPolicies)
    {
        total += policy.size();
    }
    return total;
}

// allPages() removed — use begin()/end() iterator instead.

// ---------------------------------------------------------------------------
// PerLevelEvictionController
// ---------------------------------------------------------------------------

PerLevelEvictionController::PerLevelEvictionController(
    std::vector<PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cacheLevel)
    : mCacheLevel(cacheLevel)
    , mLifeCycleGrouping(lifeCycleGrouping)
{
    // Compute number of pool groups = max(grouping) + 1
    PoolGroupIndex numPoolGroups = 0;
    for (auto g : mLifeCycleGrouping)
    {
        if (g + 1 > numPoolGroups)
            numPoolGroups = g + 1;
    }
    // Pool group indices must be contiguous 0..N-1.
    assert(numPoolGroups
        == static_cast<PoolGroupIndex>(
            std::set<PoolGroupIndex>(mLifeCycleGrouping.begin(), mLifeCycleGrouping.end()).size()));
    mPolicies.resize(static_cast<size_t>(numPoolGroups));
}

PerLevelEvictionController::~PerLevelEvictionController()
{
    if (!gNdebug)
    {
        bool allEmpty = true;
        for (auto const& p : mPolicies)
        {
            if (!p.empty())
            {
                allEmpty = false;
                break;
            }
        }
        assert(allEmpty && "Eviction controller is not empty on destruction");
    }
}

PrioritizedEvictionPolicy& PerLevelEvictionController::getPolicy(LifeCycleId lcId)
{
    PoolGroupIndex pgIdx = mLifeCycleGrouping.at(static_cast<size_t>(lcId));
    return mPolicies.at(static_cast<size_t>(pgIdx));
}

void PerLevelEvictionController::scheduleForEviction(Page& page, bool evictFirst)
{
    assert(page.nodeRef == std::nullopt);
    assert(page.cacheLevel == mCacheLevel);
    auto sharedPage = page.shared_from_this();
    NodeRef ref = getPolicy(page.lifeCycle).push(sharedPage, evictFirst);
    page.nodeRef = ref;
    assert(*ref == sharedPage && "stored iterator must dereference to this page");
}

std::vector<std::vector<std::shared_ptr<Page>>> PerLevelEvictionController::evict(std::vector<int> const& minNumPages)
{
    assert(static_cast<int>(minNumPages.size()) == numPoolGroups());

    std::vector<std::vector<std::shared_ptr<Page>>> ret(static_cast<size_t>(numPoolGroups()));

    try
    {
        for (int pgIdx = 0; pgIdx < numPoolGroups(); ++pgIdx)
        {
            auto& policy = mPolicies.at(static_cast<size_t>(pgIdx));
            int count = minNumPages.at(static_cast<size_t>(pgIdx));
            int available = static_cast<int>(policy.size()) + static_cast<int>(ret[pgIdx].size());
            if (available < count)
            {
                throw OutOfPagesError("Not enough pages to evict in group " + std::to_string(pgIdx));
            }
            while (static_cast<int>(ret[pgIdx].size()) < count)
            {
                auto page = policy.pop();
                page->nodeRef = std::nullopt;
                ret[static_cast<size_t>(pgIdx)].push_back(page);
                // @TODO: evict dependencies (like Python _evict_dependencies)
            }
        }
    }
    catch (...)
    {
        // Re-queue evicted pages in reverse order (push to front so they are evicted first next time)
        for (int i = static_cast<int>(ret.size()) - 1; i >= 0; --i)
        {
            auto& group = ret[static_cast<size_t>(i)];
            for (int j = static_cast<int>(group.size()) - 1; j >= 0; --j)
            {
                scheduleForEviction(*group[static_cast<size_t>(j)], /*evictFirst=*/true);
            }
        }
        throw;
    }

    if (!gNdebug)
    {
        for (auto const& group : ret)
        {
            for (auto const& p : group)
            {
                assert(p->cacheLevel == mCacheLevel && "Corrupted eviction controller");
            }
        }
    }

    return ret;
}

void PerLevelEvictionController::remove(NodeRef node)
{
    auto page = *node;
    assert(page->nodeRef.has_value() && page->nodeRef.value() == node
        && "page's nodeRef must match the node being removed");
    getPolicy(page->lifeCycle).remove(node);
    page->nodeRef = std::nullopt;
}

int PerLevelEvictionController::numEvictablePages(PoolGroupIndex pgIdx) const
{
    return static_cast<int>(mPolicies.at(static_cast<size_t>(pgIdx)).size());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
