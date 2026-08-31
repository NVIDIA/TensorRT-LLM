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

#pragma once

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include <cstddef>
#include <list>
#include <map>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declaration: Page is defined in page.h, but evictionController.h
// must not include page.h to avoid circular dependencies.
// We use a forward-declared abstract interface instead.
class Page;

// ---------------------------------------------------------------------------
// NodeRef — stable iterator into an LRU list.
// Using list<shared_ptr<Page>>: erasing any element does not invalidate
// other iterators (std::list guarantee). This is the C++ equivalent of the
// Python dllistnode.
//
// IMPORTANT: A NodeRef is always tied to the specific list that issued it.
// Never mix NodeRefs from different LRUEvictionPolicy instances.
// ---------------------------------------------------------------------------
using EvictionList = std::list<SharedPtr<Page>>;
using NodeRef = EvictionList::iterator;

// ---------------------------------------------------------------------------
// LRUEvictionPolicy — LRU queue backed by std::list.
// Mirrors _eviction_controller.py::LRUEvictionPolicy.
// push() → O(1), pop() → O(1), remove(NodeRef) → O(1).
// ---------------------------------------------------------------------------
class LRUEvictionPolicy
{
public:
    // Push a page into the eviction queue.
    // evictFirst=true puts it at the front (will be evicted first).
    // Returns a stable iterator (NodeRef) for later O(1) removal.
    NodeRef push(SharedPtr<Page> page, bool evictFirst = false);

    // Remove and return the front (least-recently-used) page.
    SharedPtr<Page> pop();

    // Remove an arbitrary page via its iterator. O(1).
    SharedPtr<Page> remove(NodeRef node);

    size_t size() const noexcept
    {
        return mQueue.size();
    }

    bool empty() const noexcept
    {
        return mQueue.empty();
    }

    EvictionList::const_iterator cbegin() const
    {
        return mQueue.cbegin();
    }

    EvictionList::const_iterator cend() const
    {
        return mQueue.cend();
    }

private:
    EvictionList mQueue;
};

// ---------------------------------------------------------------------------
// PrioritizedEvictionPolicy — wraps per-priority LRU sub-queues.
// Mirrors _eviction_controller.py::PrioritizedEvictionPolicy.
// Lower priority key = evicted first.
// ---------------------------------------------------------------------------
class PrioritizedEvictionPolicy
{
public:
    NodeRef push(SharedPtr<Page> page, bool evictFirst = false);
    SharedPtr<Page> pop();
    SharedPtr<Page> remove(NodeRef node);

    SlotCount size() const noexcept;

    bool empty() const noexcept
    {
        return size() == 0;
    }

    // Generator: returns a mutable lambda that yields shared_ptr<Page> const*
    // in eviction order (lowest priority first, LRU within). Returns nullptr when exhausted.
    // No extra strong refs — the pointer references the shared_ptr inside the list node.
    [[nodiscard]] auto pageGenerator() const
    {
        using MapIt = std::map<Priority, LRUEvictionPolicy>::const_iterator;
        MapIt mapIt = mPolicies.cbegin();
        MapIt mapEnd = mPolicies.cend();
        EvictionList::const_iterator listIt;
        // Advance to first non-empty sub-queue.
        while (mapIt != mapEnd && mapIt->second.empty())
            ++mapIt;
        if (mapIt != mapEnd)
            listIt = mapIt->second.cbegin();
        return [=]() mutable -> SharedPtr<Page> const*
        {
            if (mapIt == mapEnd)
                return nullptr;
            auto* result = &(*listIt);
            ++listIt;
            if (listIt == mapIt->second.cend())
            {
                ++mapIt;
                while (mapIt != mapEnd && mapIt->second.empty())
                    ++mapIt;
                if (mapIt != mapEnd)
                    listIt = mapIt->second.cbegin();
            }
            return result;
        };
    }

private:
    LRUEvictionPolicy& getOrCreate(Priority p);

    // std::map keeps keys sorted. We evict from the lowest-priority key first.
    std::map<Priority, LRUEvictionPolicy> mPolicies;
};

// ---------------------------------------------------------------------------
// PerLevelEvictionController — one eviction controller per cache level.
// Holds one PrioritizedEvictionPolicy per pool group.
// Mirrors _eviction_controller.py::PerLevelEvictionController.
// ---------------------------------------------------------------------------
class PerLevelEvictionController
{
public:
    // lifeCycleGrouping: maps LifeCycleId → PoolGroupIndex.
    // cacheLevel: the level this controller manages.
    PerLevelEvictionController(TypedVec<LifeCycleId, PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cacheLevel);

    ~PerLevelEvictionController();

    void scheduleForEviction(Page& page, bool evictFirst = false);

    // Evict at least minNumPages[pgIdx] pages per pool group.
    // Returns evicted pages per pool group.
    // On failure, re-queues any already-evicted pages and throws OutOfPagesError.
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> evict(
        TypedVec<PoolGroupIndex, SlotCount> const& minNumPages);

    // Remove a page from the queue by its NodeRef.
    void remove(NodeRef node);

    [[nodiscard]] SlotCount numEvictablePages(PoolGroupIndex pgIdx) const;

    [[nodiscard]] PoolGroupIndex numPoolGroups() const noexcept
    {
        return mPolicies.size();
    }

    // All pages in eviction order for a pool group.
    [[nodiscard]] auto pageGenerator(PoolGroupIndex pgIdx) const
    {
        return mPolicies.at(pgIdx).pageGenerator();
    }

private:
    PrioritizedEvictionPolicy& getPolicy(LifeCycleId lcId);

    CacheLevel mCacheLevel;
    TypedVec<LifeCycleId, PoolGroupIndex> mLifeCycleGrouping;
    TypedVec<PoolGroupIndex, PrioritizedEvictionPolicy> mPolicies; // one per pool group
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
