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

#include "tensorrt_llm/common/assert.h"
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

Page::Page(StorageManager* mgr, LifeCycleId lc, CacheLevel level, Priority prio)
    : manager(mgr)
    , lifeCycle(lc)
    , cacheLevel(level)
    , priority(prio)
    , nodeRef(std::nullopt)
{
}

Page::~Page()
{
    TLLM_CHECK_DEBUG_WITH_INFO(status() == PageStatus::DROPPABLE && !scheduledForEviction(),
        "Page destroyed while still held or scheduled for eviction");
    if (hasValidSlot())
    {
        Slot s;
        s.setSlotId(slotId());
        s.readyEvent = std::move(readyEvent);
        resetSlot();
        manager->releaseSlot(lifeCycle, cacheLevel, std::move(s));
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

SharedPtr<PageHolder> Page::hold()
{
    // Return existing holder if any.
    auto h = holder.lock();
    if (h)
        return h;

    auto self = sharedFromThis();
    h = makeShared<PageHolder>(self);
    holder = h;

    // If we were scheduled for eviction but are no longer evictable (just got held), remove.
    if (scheduledForEviction())
    {
        if (!manager->isEvictable(*this))
        {
            manager->excludeFromEviction(*this);
            TLLM_CHECK_DEBUG(!scheduledForEviction());
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

CommittedPage::CommittedPage(StorageManager* mgr, SharedPtr<Block> blk, LifeCycleId lc, CacheLevel level, Priority prio)
    : Page(mgr, lc, level, prio)
    , block(blk.get())
{
}

SsmCommittedPage::SsmCommittedPage(
    StorageManager* mgr, SharedPtr<Block> blk, LifeCycleId lc, CacheLevel level, Priority prio, int numTokensInBlock_)
    : CommittedPage(mgr, std::move(blk), lc, level, prio)
    , numTokensInBlock(numTokensInBlock_)
{
    TLLM_CHECK_DEBUG(numTokensInBlock_ > 0);
}

CommittedPage::~CommittedPage()
{
    if (block != nullptr)
    {
        // unlinkPage nulls storage[lc]->block (i.e. our member), so capture
        // the block pointer first. Pass `this` as the expected page: if a newer
        // page already replaced us in the slot (e.g. a larger SSM snapshot), the
        // slot is left alone and prev is nullptr — skip stale-block cleanup then.
        Block* blk = block;
        auto* prev = blk->unlinkPage(lifeCycle, this);
        if (prev != nullptr)
        {
            TLLM_CHECK_DEBUG_WITH_INFO(prev == this, "unlinkPage returned unexpected page");
            LifeCycle const& lc = manager->lifeCycles().getLifeCycle(lifeCycle);
            Block::clearStaleBlocksAfterPageUnlink(*blk, lifeCycle, lc);
        }
    }
    // Delegate slot release to Page::~Page().
}

// ---------------------------------------------------------------------------
// UncommittedPage
// ---------------------------------------------------------------------------

UncommittedPage::UncommittedPage(KvCache& kvc, BlockOrdinal ord, LifeCycleId lc, CacheLevel level, BeamIndex bi)
    : Page(kvc.storageManager(), lc, level, kvc.getPriority(ord, lc))
    , kvCache(&kvc)
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
    if (TLLM_UNLIKELY(gDebug))
    {
        auto ssmLcId = manager->lifeCycles().ssmLifeCycleId();
        bool isSsm = ssmLcId.has_value() && lifeCycle == *ssmLcId;
        if (!isSsm)
        {
            [[maybe_unused]] bool blockRemoved = kvCache->blocks().size() <= ordinal;
            [[maybe_unused]] bool pageOk = true;
            if (!blockRemoved)
            {
                auto const& bp = kvCache->blocks()[ordinal].pages[beamIndex][lifeCycle];
                auto page = blockPageGetPage(bp);
                pageOk
                    = blockPageIsNull(bp) || page.get() == this || dynamicPointerCast<CommittedPage>(page) != nullptr;
            }
            TLLM_CHECK_WITH_INFO(
                blockRemoved || pageOk, "UncommittedPage destroyed but slot still holds a different uncommitted page");
        }
    }
    // Delegate slot release to Page::~Page().
}

SharedPtr<CommittedPage> UncommittedPage::convertToCommitted(SharedPtr<Block> blk, CachedCudaEvent readyEv)
{
    TLLM_CHECK_DEBUG(!scheduledForEviction());
    TLLM_CHECK_DEBUG_WITH_INFO(
        blk->storage.at(lifeCycle) == nullptr, "Block slot for this lifecycle already has a committed page");
    TLLM_CHECK_DEBUG_WITH_INFO(status() == PageStatus::DROPPABLE, "Release holder/lock before converting");

    // Set the ready event before transfer (matches Python: self.ready_event = ready_event).
    this->readyEvent = std::move(readyEv);

    auto committed = makeShared<CommittedPage>(manager, blk, lifeCycle, cacheLevel, priority);
    // Move slot id to the committed page; invalidate our slot.
    committed->setSlotId(slotId()); // asserts valid
    committed->readyEvent = std::move(readyEvent);
    resetSlot();
    readyEvent = CachedCudaEvent::makeNull();

    TLLM_CHECK_DEBUG(!hasValidSlot() && readyEvent.isClosed());
    TLLM_CHECK_DEBUG_WITH_INFO(committed->hasValidSlot(), "committed page must have a valid slot after transfer");

    // Register in block storage.
    blk->storage.at(lifeCycle) = committed.get();

    return committed;
}

SharedPtr<SsmCommittedPage> UncommittedPage::convertToSsmCommitted(
    SharedPtr<Block> blk, CachedCudaEvent readyEv, int numTokensInBlock)
{
    TLLM_CHECK_DEBUG(!scheduledForEviction());
    TLLM_CHECK_DEBUG_WITH_INFO(
        blk->storage.at(lifeCycle) == nullptr, "Block slot for this lifecycle already has a committed page");
    TLLM_CHECK_DEBUG_WITH_INFO(status() == PageStatus::DROPPABLE, "Release holder/lock before converting");

    this->readyEvent = std::move(readyEv);

    auto committed = makeShared<SsmCommittedPage>(manager, blk, lifeCycle, cacheLevel, priority, numTokensInBlock);
    committed->setSlotId(slotId()); // asserts valid
    committed->readyEvent = std::move(readyEvent);
    resetSlot();
    readyEvent = CachedCudaEvent::makeNull();

    TLLM_CHECK_DEBUG(!hasValidSlot() && readyEvent.isClosed());
    TLLM_CHECK_DEBUG_WITH_INFO(committed->hasValidSlot(), "committed page must have a valid slot after transfer");

    blk->storage.at(lifeCycle) = committed.get();

    return committed;
}

// ---------------------------------------------------------------------------
// PageHolder
// ---------------------------------------------------------------------------

PageHolder::PageHolder(SharedPtr<Page> p)
    : page(std::move(p))
{
}

PageHolder::~PageHolder()
{
    TLLM_CHECK_DEBUG_WITH_INFO(uniqLock.expired(), "PageHolder destroyed while lock still active");

    page->holder.reset(); // clear back-reference
    auto const manager = page->manager;

    // If it's a committed page, schedule for eviction (if evictable).
    if (page->isCommitted())
    {
        if (!page->scheduledForEviction())
            manager->scheduleForEviction(*page);

        // If the block is orphan, exclude from eviction immediately.
        auto* cp = dynamic_cast<CommittedPage*>(page.get());
        if (cp)
        {
            if (cp->block == nullptr || cp->block->isOrphan())
                manager->excludeFromEviction(*page);
        }
    }
    else
    {
        // Uncommitted page: if scheduled for eviction, remove it.
        if (page->scheduledForEviction())
            manager->excludeFromEviction(*page);
    }
}

SharedPageLock PageHolder::lock(
    KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lc, bool skipWait)
{
    // Create or reuse UniqPageLock.
    auto ul = uniqLock.lock();
    if (!ul)
    {
        ul = makeShared<UniqPageLock>(sharedFromThis());
        uniqLock = ul;
    }

    // Remove from eviction queue if scheduled.
    if (page->scheduledForEviction())
    {
        page->manager->excludeFromEviction(*page);
        TLLM_CHECK_DEBUG(!page->scheduledForEviction());
    }

    return ul->share(kvCache, beamIndex, ordinal, lc, skipWait);
}

// ---------------------------------------------------------------------------
// UniqPageLock
// ---------------------------------------------------------------------------

UniqPageLock::UniqPageLock(SharedPtr<PageHolder> h)
    : holder(std::move(h))
{
    if (holder->page->cacheLevel != kGpuLevel)
        throw LogicError("Lock can only be applied to GPU-memory pages");
}

UniqPageLock::~UniqPageLock()
{
    Page& p = *page();
    TLLM_CHECK_DEBUG(p.cacheLevel == kGpuLevel && !p.scheduledForEviction());
    // Set readyEvent to the merged finish events of all readers. For committed (read-only)
    // pages, this means the next reader will wait for prior reads to complete, which is
    // unnecessary but correct. See the CommittedPage comment in page.h for rationale.
    p.readyEvent = mergeEvents(finishEvents);

    // Clear the holder's lock reference.
    TLLM_CHECK_DEBUG(holder);
    holder->uniqLock.reset();

    // Optimized path (mirrors Python): set holder=nullptr, then check if still evictable.
    auto holderCopy = std::move(holder);
    holder = nullptr;

    // If the page is not droppable (still held by someone else) and evictable,
    // schedule for eviction.
    if (p.status() != PageStatus::DROPPABLE)
    {
        auto const manager = p.manager;
        if (manager->isEvictable(p))
            manager->scheduleForEviction(p);
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

SharedPtr<Page> const& UniqPageLock::page() const
{
    TLLM_CHECK_DEBUG(holder && holder->page);
    return holder->page;
}

SharedPageLock UniqPageLock::share(
    KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lc, bool skipWait)
{
    return SharedPageLock(sharedFromThis(), kvCache, beamIndex, ordinal, lc, skipWait);
}

// ---------------------------------------------------------------------------
// SharedPageLock
// ---------------------------------------------------------------------------

SharedPageLock::SharedPageLock(SharedPtr<UniqPageLock> ul, KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal,
    LifeCycleId lc, bool skipWait)
    : mUniqLock(std::move(ul))
    , mUser{&kvCache, beamIndex, ordinal, lc}
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

SharedPtr<Page> const& SharedPageLock::page() const
{
    TLLM_CHECK_DEBUG(mUniqLock);
    return mUniqLock->page();
}

SharedPtr<Page> SharedPageLock::unlock()
{
    TLLM_CHECK_DEBUG(mUniqLock);

    // Record finish event from the KvCache stream.
    mUniqLock->notifyFinish(mUser.kvCache->finishEvent());

    releasePageIndex();
    auto p = page(); // copy shared_ptr before reset
    mUniqLock.reset();
    return p;
}

void SharedPageLock::acquirePageIndex()
{
    auto* kvc = mUser.kvCache;
    auto& pg = *page();
    int old = kvc->updateBasePageIndex(
        mUser.beamIndex, mUser.ordinal, mUser.lifeCycle, slotIdToPageIndexValue(pg.slotId()));
    // Mirrors Python assertion: old base index must be BAD (prevents double-locking same slot).
    TLLM_CHECK_DEBUG_WITH_INFO(
        old == kBadPageIndex.value(), "Double-lock: page index already acquired for this (beam, ordinal, lc)");
    (void) old;
}

void SharedPageLock::releasePageIndex()
{
    int oldBaseIndex
        = mUser.kvCache->updateBasePageIndex(mUser.beamIndex, mUser.ordinal, mUser.lifeCycle, kBadPageIndex.value());
    // Mirrors Python assertion: old base index must match this page's slot ID.
    TLLM_CHECK_DEBUG(oldBaseIndex == slotIdToPageIndexValue(page()->slotId()));
    (void) oldBaseIndex;
}

// ---------------------------------------------------------------------------
// batchedLockToGpu
// ---------------------------------------------------------------------------

std::vector<SharedPageLock> batchedLockToGpu(KvCache& kvCache, std::vector<BatchedLockTarget> const& targets)
{
    auto* storeMgr = kvCache.storageManager();
    TLLM_CHECK_DEBUG(storeMgr);
    // All pages must belong to the same storage manager.
    TLLM_CHECK_DEBUG(targets.empty()
        || std::all_of(targets.begin(), targets.end(), [&](auto const& t) { return t.page->manager == storeMgr; }));

    // Determine how many GPU slots are needed per pool group.
    TypedVec<PoolGroupIndex, SlotCount> requirements(storeMgr->numPoolGroups(), 0);
    std::vector<bool> wasScheduled(targets.size(), false);

    for (size_t i = 0; i < targets.size(); ++i)
    {
        auto const& t = targets[i];
        wasScheduled[i] = t.page->scheduledForEviction();
        if (wasScheduled[i])
            storeMgr->excludeFromEviction(*t.page);
        if (t.page->cacheLevel != kGpuLevel)
        {
            PoolGroupIndex pgIdx = storeMgr->getPoolGroupIndex(t.lifeCycle);
            requirements[pgIdx] += 1;
        }
    }

    try
    {
        MigrationRecorder const migrationRecorder
            = [&kvCache](std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots, CacheLevel srcLevel,
                  CacheLevel dstLevel) { kvCache._recordMigratedSlots(pages, slots, srcLevel, dstLevel); };
        DropRecorder const dropRecorder = [&kvCache](std::vector<SharedPtr<Page>> const& pages, CacheLevel cacheLevel)
        { kvCache._recordDroppedPages(pages, cacheLevel); };
        storeMgr->prepareFreeSlots(kGpuLevel, requirements, migrationRecorder, dropRecorder);
        // Migrate non-GPU pages.
        storeMgr->batchedMigrateToGpu(targets, kvCache, migrationRecorder);
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

// ---------------------------------------------------------------------------
// ScratchSlotLock
// ---------------------------------------------------------------------------

ScratchSlotLock::ScratchSlotLock(Slot slot, KvCache& owner, LifeCycleId lifeCycle, bool skipWait)
    : mOwner(&owner)
    , mLifeCycle(lifeCycle)
{
    if (!skipWait)
    {
        slot.readyEvent.waitInStream(reinterpret_cast<CudaStream>(owner.cudaStream()));
    }
    mSlot.setSlot(slot);
}

ScratchSlotLock::~ScratchSlotLock()
{
    if (mSlot.hasValidSlot())
    {
        try
        {
            unlock();
        }
        catch (...)
        {
        }
    }
}

ScratchSlotLock::ScratchSlotLock(ScratchSlotLock&& other) noexcept
    : mSlot(std::move(other.mSlot))
    , mOwner(other.mOwner)
    , mLifeCycle(other.mLifeCycle)
{
    // Invalidate moved-from: mSlot move only transfers readyEvent, not slotId (trivially-copyable).
    other.mSlot.resetSlot();
    other.mOwner = nullptr;
}

ScratchSlotLock& ScratchSlotLock::operator=(ScratchSlotLock&& other) noexcept
{
    if (this != &other)
    {
        if (mSlot.hasValidSlot())
        {
            try
            {
                unlock();
            }
            catch (...)
            {
            }
        }
        mSlot = std::move(other.mSlot);
        other.mSlot.resetSlot(); // Invalidate moved-from slotId.
        mOwner = other.mOwner;
        mLifeCycle = other.mLifeCycle;
        other.mOwner = nullptr;
    }
    return *this;
}

Slot ScratchSlotLock::detachSlot()
{
    TLLM_CHECK_DEBUG(mSlot.hasValidSlot());
    Slot result;
    result.setSlot(mSlot);
    return result;
}

void ScratchSlotLock::unlock()
{
    TLLM_CHECK_DEBUG(mSlot.hasValidSlot());
    mSlot.readyEvent = mOwner->finishEvent();
    mOwner->storageManager()->releaseSlot(mLifeCycle, kGpuLevel, std::move(mSlot));
    TLLM_CHECK_DEBUG(!mSlot.hasValidSlot());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
