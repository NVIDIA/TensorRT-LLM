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

#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/kvCache.h"        // for KvCache
#include "kv_cache_manager_v2/storageManager.h" // for StorageManager

#include <cassert>
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

Page::Page(std::shared_ptr<StorageManager> mgr, LifeCycleId lc, CacheLevel level, Priority prio)
    : manager(mgr)
    , lifeCycle(lc)
    , cacheLevel(level)
    , priority(prio)
    , nodeRef(std::nullopt)
{
}

Page::~Page()
{
    assert(gNdebug
        || (status() == PageStatus::DROPPABLE && !scheduledForEviction()
            && "Page destroyed while still held or scheduled for eviction"));
    if (hasValidSlot())
    {
        if (auto mgr = manager.lock())
        {
            Slot s;
            s.setSlotId(slotId());
            s.readyEvent = std::move(readyEvent);
            resetSlot();
            mgr->releaseSlot(lifeCycle, cacheLevel, std::move(s));
        }
    }
}

PageStatus Page::status() const noexcept
{
    auto h = holder.lock();
    if (!h)
        return PageStatus::DROPPABLE;
    if (h->uniqLock.expired())
        return PageStatus::HELD;
    return PageStatus::LOCKED;
}

std::shared_ptr<PageHolder> Page::hold()
{
    // Return existing holder if any.
    auto h = holder.lock();
    if (h)
        return h;

    auto self = shared_from_this();
    h = std::make_shared<PageHolder>(self);
    holder = h;

    // If we were scheduled for eviction but are no longer evictable (just got held), remove.
    if (scheduledForEviction())
    {
        if (auto mgr = manager.lock())
        {
            if (!mgr->isEvictable(*this))
            {
                mgr->excludeFromEviction(*this);
                assert(!scheduledForEviction());
            }
        }
    }
    return h;
}

SharedPageLock Page::lock(KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lc, bool skipWait)
{
    return hold()->lock(kvCache, beamIndex, ordinal, lc, skipWait);
}

// ---------------------------------------------------------------------------
// CommittedPage
// ---------------------------------------------------------------------------

CommittedPage::CommittedPage(
    std::shared_ptr<StorageManager> mgr, std::shared_ptr<Block> blk, LifeCycleId lc, CacheLevel level, Priority prio)
    : Page(mgr, lc, level, prio)
    , block(blk)
{
}

CommittedPage::~CommittedPage()
{
    auto blk = block.lock();
    if (blk)
    {
        if (auto mgr = manager.lock())
        {
            LifeCycle const& lc = mgr->lifeCycles().getLifeCycle(lifeCycle);
            blk->unsetPage(lifeCycle, lc);
        }
    }
    // Delegate slot release to Page::~Page().
}

// ---------------------------------------------------------------------------
// UncommittedPage
// ---------------------------------------------------------------------------

UncommittedPage::UncommittedPage(KvCache& kvc, BlockOrdinal ord, LifeCycleId lc, CacheLevel level, BeamIndex bi)
    : Page(kvc.storageManager().lock(), lc, level, kvc.getPriority(ord, lc))
    , kvCache(kvc.weak_from_this())
    , ordinal(ord)
    , beamIndex(bi)
{
}

UncommittedPage::~UncommittedPage()
{
    // Mirrors Python UncommittedPage.__del__: for attention LCs, the page must be either:
    //  - part of an SSM lifecycle (different rules),
    //  - at an ordinal beyond the current block list (block already removed),
    //  - the slot at this position is null, CommittedPage, or this page itself (self-destruction).
    // The "p == this" condition is C++-specific: std::variant destroys the old value before
    // switching to monostate, so during destruction the slot still references this page.
    if (!gNdebug)
    {
        if (auto mgr = manager.lock())
        {
            auto ssmLcId = mgr->lifeCycles().ssmLifeCycleId();
            bool isSsm = ssmLcId.has_value() && lifeCycle == *ssmLcId;
            if (!isSsm)
            {
                if (auto kvc = kvCache.lock())
                {
                    [[maybe_unused]] bool blockRemoved = static_cast<int>(kvc->blocks().size()) <= ordinal;
                    [[maybe_unused]] bool pageOk = true;
                    if (!blockRemoved)
                    {
                        auto const& bp = kvc->blocks()[static_cast<size_t>(ordinal)]
                                             .pages[static_cast<size_t>(beamIndex)][static_cast<size_t>(lifeCycle)];
                        auto page = blockPageGetPage(bp);
                        pageOk = blockPageIsNull(bp) || page.get() == this
                            || std::dynamic_pointer_cast<CommittedPage>(page) != nullptr;
                    }
                    assert((blockRemoved || pageOk)
                        && "UncommittedPage destroyed but slot still holds a different uncommitted page");
                }
            }
        }
    }
    // Delegate slot release to Page::~Page().
}

std::shared_ptr<CommittedPage> UncommittedPage::convertToCommitted(std::shared_ptr<Block> blk, CachedCudaEvent readyEv)
{
    assert(!scheduledForEviction());
    assert(blk->storage.at(static_cast<size_t>(lifeCycle)).expired()
        && "Block slot for this lifecycle already has a committed page");
    assert(status() == PageStatus::DROPPABLE && "Release holder/lock before converting");

    auto mgr = manager.lock();
    assert(mgr);

    // Set the ready event before transfer (matches Python: self.ready_event = ready_event).
    this->readyEvent = std::move(readyEv);

    auto committed = std::make_shared<CommittedPage>(mgr, blk, lifeCycle, cacheLevel, priority);
    // Move slot id to the committed page; invalidate our slot.
    committed->setSlotId(slotId()); // asserts valid
    committed->readyEvent = std::move(readyEvent);
    resetSlot();
    readyEvent = CachedCudaEvent::makeNull();

    assert(!hasValidSlot() && readyEvent.isNull());
    assert(committed->hasValidSlot() && "committed page must have a valid slot after transfer");

    // Register in block storage.
    blk->storage.at(static_cast<size_t>(lifeCycle)) = committed;

    return committed;
}

// ---------------------------------------------------------------------------
// PageHolder
// ---------------------------------------------------------------------------

PageHolder::PageHolder(std::shared_ptr<Page> p)
    : page(std::move(p))
{
}

PageHolder::~PageHolder()
{
    assert(gNdebug || (uniqLock.expired() && "PageHolder destroyed while lock still active"));

    page->holder.reset(); // clear back-reference

    // If it's a committed page, schedule for eviction (if evictable).
    if (page->isCommitted())
    {
        auto mgr = page->manager.lock();
        if (mgr)
        {
            if (!page->scheduledForEviction())
                mgr->scheduleForEviction(*page);

            // If the block is orphan, exclude from eviction immediately.
            auto* cp = dynamic_cast<CommittedPage*>(page.get());
            if (cp)
            {
                auto blk = cp->block.lock();
                if (!blk || blk->isOrphan())
                    mgr->excludeFromEviction(*page);
            }
        }
    }
    else
    {
        // Uncommitted page: if scheduled for eviction, remove it.
        if (page->scheduledForEviction())
        {
            if (auto mgr = page->manager.lock())
                mgr->excludeFromEviction(*page);
        }
    }
}

SharedPageLock PageHolder::lock(
    KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lc, bool skipWait)
{
    // Create or reuse UniqPageLock.
    auto ul = uniqLock.lock();
    if (!ul)
    {
        ul = std::make_shared<UniqPageLock>(shared_from_this());
        uniqLock = ul;
    }

    // Remove from eviction queue if scheduled.
    if (page->scheduledForEviction())
    {
        if (auto mgr = page->manager.lock())
            mgr->excludeFromEviction(*page);
        assert(!page->scheduledForEviction());
    }

    return ul->share(kvCache, beamIndex, ordinal, lc, skipWait);
}

// ---------------------------------------------------------------------------
// UniqPageLock
// ---------------------------------------------------------------------------

UniqPageLock::UniqPageLock(std::shared_ptr<PageHolder> h)
    : holder(std::move(h))
{
    if (holder->page->cacheLevel != kGpuLevel)
        throw LogicError("Lock can only be applied to GPU-memory pages");
}

UniqPageLock::~UniqPageLock()
{
    Page& p = *page();
    assert(gNdebug || (p.cacheLevel == kGpuLevel && !p.scheduledForEviction()));
    // Merge finish events and set on the page (mirrors Python's merge_events()).
    p.readyEvent = mergeEvents(finishEvents);

    // Clear the holder's lock reference.
    assert(holder);
    holder->uniqLock.reset();

    // Optimized path (mirrors Python): set holder=nullptr, then check if still evictable.
    auto holderCopy = std::move(holder);
    holder = nullptr;

    // If the page is not droppable (still held by someone else) and evictable,
    // schedule for eviction.
    if (p.status() != PageStatus::DROPPABLE)
    {
        if (auto mgr = p.manager.lock())
        {
            if (mgr->isEvictable(p))
                mgr->scheduleForEviction(p);
        }
    }
}

void UniqPageLock::notifyFinish(CachedCudaEvent event)
{
    finishEvents.push_back(std::move(event));
    // Avoid unbounded growth for system prompt pages shared by all requests.
    if (finishEvents.size() > 32)
    {
        CachedCudaEvent merged = mergeEvents(finishEvents);
        finishEvents.clear();
        finishEvents.push_back(std::move(merged));
    }
}

std::shared_ptr<Page> const& UniqPageLock::page() const
{
    assert(holder && holder->page);
    return holder->page;
}

SharedPageLock UniqPageLock::share(
    KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lc, bool skipWait)
{
    return SharedPageLock(shared_from_this(), kvCache, beamIndex, ordinal, lc, skipWait);
}

// ---------------------------------------------------------------------------
// SharedPageLock
// ---------------------------------------------------------------------------

SharedPageLock::SharedPageLock(std::shared_ptr<UniqPageLock> ul, KvCache& kvCache, BeamIndex beamIndex,
    BlockOrdinal ordinal, LifeCycleId lc, bool skipWait)
    : mUniqLock(std::move(ul))
    , mUser{kvCache.weak_from_this(), beamIndex, ordinal, lc}
{
    if (!skipWait)
        page()->readyEvent.waitInStream(reinterpret_cast<CudaStream>(kvCache.cudaStream()));

    acquirePageIndex();
}

SharedPageLock::~SharedPageLock()
{
    if (mUniqLock)
        unlock();
}

SharedPageLock::SharedPageLock(SharedPageLock&& other) noexcept
    : mUniqLock(std::move(other.mUniqLock))
    , mUser(std::move(other.mUser))
{
}

SharedPageLock& SharedPageLock::operator=(SharedPageLock&& other) noexcept
{
    if (this != &other)
    {
        if (mUniqLock)
            unlock();
        mUniqLock = std::move(other.mUniqLock);
        mUser = std::move(other.mUser);
    }
    return *this;
}

std::shared_ptr<Page> const& SharedPageLock::page() const
{
    assert(mUniqLock);
    return mUniqLock->page();
}

std::shared_ptr<Page> SharedPageLock::unlock()
{
    assert(mUniqLock);

    // Record finish event from the KvCache stream.
    // Use lock() instead of unwrap(): during ~KvCache, member destruction
    // destroys SharedPageLocks after the KvCache's weak_ptr has expired.
    // Throwing from a destructor would call std::terminate().
    if (auto kvc = mUser.kvCache.lock())
    {
        mUniqLock->notifyFinish(kvc->finishEvent());
    }

    releasePageIndex();
    auto p = page(); // copy shared_ptr before reset
    mUniqLock.reset();
    return p;
}

void SharedPageLock::acquirePageIndex()
{
    auto kvc = unwrap(mUser.kvCache);
    auto& pg = *page();
    int old = kvc->updateBasePageIndex(mUser.beamIndex, mUser.ordinal, mUser.lifeCycle, pg.slotId());
    // Mirrors Python assertion: old base index must be BAD (prevents double-locking same slot).
    assert(old == kBadPageIndex && "Double-lock: page index already acquired for this (beam, ordinal, lc)");
    (void) old;
}

void SharedPageLock::releasePageIndex()
{
    // Use lock() instead of unwrap() — may be called from destructor path.
    if (auto kvc = mUser.kvCache.lock())
    {
        int oldBaseIndex = kvc->updateBasePageIndex(mUser.beamIndex, mUser.ordinal, mUser.lifeCycle, /*BAD=*/-1);
        // Mirrors Python assertion: old base index must match this page's slot ID.
        assert(oldBaseIndex == page()->slotId());
        (void) oldBaseIndex;
    }
}

// ---------------------------------------------------------------------------
// batchedLockToGpu
// ---------------------------------------------------------------------------

std::vector<SharedPageLock> batchedLockToGpu(KvCache& kvCache, std::vector<BatchedLockTarget> const& targets)
{
    auto storeMgr = kvCache.storageManager().lock();
    assert(storeMgr);
    // All pages must belong to the same storage manager.
    assert(targets.empty()
        || std::all_of(targets.begin(), targets.end(),
            [&](auto const& t) { return t.page->manager.lock().get() == storeMgr.get(); }));

    // Determine how many GPU slots are needed per pool group.
    std::vector<int> requirements(static_cast<size_t>(storeMgr->numPoolGroups()), 0);
    std::vector<bool> wasScheduled(targets.size(), false);

    for (size_t i = 0; i < targets.size(); ++i)
    {
        auto const& t = targets[i];
        wasScheduled[i] = t.page->scheduledForEviction();
        if (wasScheduled[i])
            storeMgr->excludeFromEviction(*t.page);
        if (t.page->cacheLevel != kGpuLevel)
        {
            int pgIdx = storeMgr->getPoolGroupIndex(t.lifeCycle);
            requirements[static_cast<size_t>(pgIdx)] += 1;
        }
    }

    try
    {
        storeMgr->prepareFreeSlots(kGpuLevel, requirements);
        // Migrate non-GPU pages.
        storeMgr->batchedMigrateToGpu(targets, kvCache);
    }
    catch (...)
    {
        // Restore eviction scheduling.
        for (size_t i = 0; i < targets.size(); ++i)
            if (wasScheduled[i])
                storeMgr->scheduleForEviction(*targets[i].page);
        throw;
    }

    // Wait for all ready events on KvCache's stream (deduplicated).
    {
        std::vector<CachedCudaEvent const*> readyEvents;
        readyEvents.reserve(targets.size());
        for (auto const& t : targets)
            readyEvents.push_back(&t.page->readyEvent);
        streamWaitEvents(reinterpret_cast<CudaStream>(kvCache.cudaStream()), readyEvents);
    }

    // Lock all pages.
    std::vector<SharedPageLock> locks;
    locks.reserve(targets.size());
    for (auto const& t : targets)
        locks.emplace_back(t.page->lock(kvCache, t.beamIndex, t.ordinal, t.lifeCycle,
            /*skipWait=*/true));
    return locks;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
