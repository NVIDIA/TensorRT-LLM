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

#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/evictionController.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/storage/core.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include <functional>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declarations to break circular includes.
class StorageManager;
class KvCache;
class PageHolder;
class UniqPageLock;
class SharedPageLock;

// ---------------------------------------------------------------------------
// Page — base class for all KV-cache pages.
// Inherits from Slot (holds slotId + readyEvent).
// Mirrors Python's Page(Slot) dataclass.
// ---------------------------------------------------------------------------
class Page : public Slot, public EnableSharedFromThis<Page>
{
public:
    StorageManager* manager;
    LifeCycleId lifeCycle;
    CacheLevel cacheLevel;
    Priority priority;
    WeakPtr<PageHolder> holder;     // empty → DROPPABLE
    std::optional<NodeRef> nodeRef; // present → scheduled for eviction

    Page(StorageManager* mgr, LifeCycleId lc, CacheLevel level, Priority prio);

    virtual ~Page();

    virtual bool isCommitted() const = 0;

    PageStatus status() const noexcept;

    bool scheduledForEviction() const noexcept
    {
        return nodeRef.has_value();
    }

    // Prevent the page from being dropped (returns/creates a PageHolder).
    SharedPtr<PageHolder> hold();

    // Acquire a shared lock (migrates to GPU if needed).
    // skip_wait: caller guarantees the page is ready on kvCache's stream.
    SharedPageLock lock(
        KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lifeCycle, bool skipWait = false);
};

// ---------------------------------------------------------------------------
// CommittedPage — page associated with a Block in the radix tree.
//
// A committed page is immutable — all access after commit is read-only.
//
// We intentionally do not add a separate read event to track read completion.
// The inherited Slot::readyEvent serves double duty: after commit or migration
// it represents write completion; after UniqPageLock is destroyed it is set to
// the merged finish events of all prior readers.  This means a new reader may
// unnecessarily wait for a prior reader (read-after-read on immutable data),
// but this is functionally correct, only occurs when the lock is fully released
// between reuses, and saves one event field per committed page — a worthwhile
// tradeoff given the potentially huge number of committed pages in the system.
// ---------------------------------------------------------------------------
class CommittedPage : public Page
{
public:
    Block* block;

    CommittedPage(StorageManager* mgr, SharedPtr<Block> blk, LifeCycleId lc, CacheLevel level, Priority prio);

    ~CommittedPage() override;

    bool isCommitted() const override
    {
        return true;
    }
};

// ---------------------------------------------------------------------------
// UncommittedPage — page associated with a live KvCache sequence.
// ---------------------------------------------------------------------------
class UncommittedPage : public Page
{
public:
    KvCache* kvCache;
    BlockOrdinal ordinal;
    BeamIndex beamIndex;
    std::vector<TokenIdExt> tokens;

    UncommittedPage(KvCache& kvc, BlockOrdinal ord, LifeCycleId lc, CacheLevel level, BeamIndex bi = kDefaultBeamIndex);

    ~UncommittedPage() override;

    bool isCommitted() const override
    {
        return false;
    }

    // Convert this UncommittedPage into a CommittedPage and attach to `block`.
    // The UncommittedPage becomes invalid (slot transferred to CommittedPage).
    SharedPtr<CommittedPage> convertToCommitted(SharedPtr<Block> block, CachedCudaEvent readyEvent);
};

// ---------------------------------------------------------------------------
// PageHolder — prevents a page from being dropped (HELD status).
// Mirrors Python's _PageHolder.
// ---------------------------------------------------------------------------
class PageHolder : public EnableSharedFromThis<PageHolder>
{
public:
    explicit PageHolder(SharedPtr<Page> page);
    ~PageHolder();

    PageHolder(PageHolder const&) = delete;
    PageHolder& operator=(PageHolder const&) = delete;

    // Acquire a shared lock (creates or reuses the UniqPageLock).
    SharedPageLock lock(
        KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lifeCycle, bool skipWait = false);

    SharedPtr<Page> page;
    WeakPtr<UniqPageLock> uniqLock; // non-null → LOCKED
};

// ---------------------------------------------------------------------------
// UniqPageLock — locks a page to prevent eviction (LOCKED status).
// Owns finish events from all SharedPageLocks it issued.
// Mirrors Python's _UniqPageLock.
// ---------------------------------------------------------------------------
class UniqPageLock : public EnableSharedFromThis<UniqPageLock>
{
public:
    explicit UniqPageLock(SharedPtr<PageHolder> holder);
    ~UniqPageLock();

    UniqPageLock(UniqPageLock const&) = delete;
    UniqPageLock& operator=(UniqPageLock const&) = delete;

    // Issue a SharedPageLock to a specific (kvCache, beam, ordinal, lifecycle).
    SharedPageLock share(
        KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lifeCycle, bool skipWait);

    SharedPtr<Page> const& page() const;

    // Append a finish event, merging when count exceeds 32 to prevent unbounded growth.
    void notifyFinish(CachedCudaEvent event);

    SharedPtr<PageHolder> holder;
    std::vector<CachedCudaEvent> finishEvents;
};

// ---------------------------------------------------------------------------
// LockOwner — identifies who holds a SharedPageLock.
// ---------------------------------------------------------------------------
struct LockOwner
{
    KvCache* kvCache;
    BeamIndex beamIndex;
    BlockOrdinal ordinal;
    LifeCycleId lifeCycle;
};

// ---------------------------------------------------------------------------
// SharedPageLock — one user's hold on an active page lock.
// Mirrors Python's _SharedPageLock.
// ---------------------------------------------------------------------------
class SharedPageLock
{
public:
    SharedPageLock(SharedPtr<UniqPageLock> uniqLock, KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal,
        LifeCycleId lifeCycle, bool skipWait);

    ~SharedPageLock();

    SharedPageLock(SharedPageLock&&) noexcept;
    SharedPageLock& operator=(SharedPageLock&&) noexcept;

    SharedPageLock(SharedPageLock const&) = delete;
    SharedPageLock& operator=(SharedPageLock const&) = delete;

    // Explicitly release the lock (called by destructor if not already released).
    SharedPtr<Page> unlock();

    SharedPtr<Page> const& page() const;

    bool isValid() const noexcept
    {
        return mUniqLock != nullptr;
    }

private:
    // Internal helpers that update KvCache page index tables.
    void acquirePageIndex();
    void releasePageIndex();

    SharedPtr<UniqPageLock> mUniqLock;
    LockOwner mUser;
};

// ---------------------------------------------------------------------------
// BatchedLockTarget — input for batched_lock_to_gpu.
// ---------------------------------------------------------------------------
struct BatchedLockTarget
{
    SharedPtr<Page> page;
    BeamIndex beamIndex;
    BlockOrdinal ordinal;
    LifeCycleId lifeCycle;
};

// ---------------------------------------------------------------------------
// batchedLockToGpu — migrate pages to GPU then lock them.
// Returns one SharedPageLock per target.
// Mirrors Python's batched_lock_to_gpu().
// ---------------------------------------------------------------------------
std::vector<SharedPageLock> batchedLockToGpu(KvCache& kvCache, std::vector<BatchedLockTarget> const& targets);

// ---------------------------------------------------------------------------
// ScratchSlotLock — manages a scratch slot for SWA prefill memory reuse.
// Wraps a Slot with owner (KvCache) and lifecycle references.
// On destruction, releases the slot back to the StorageManager.
// Mirrors _page.py::ScratchSlotLock.
// ---------------------------------------------------------------------------
class ScratchSlotLock
{
public:
    ScratchSlotLock(Slot slot, KvCache& owner, LifeCycleId lifeCycle, bool skipWait = false);
    ~ScratchSlotLock();

    ScratchSlotLock(ScratchSlotLock&& other) noexcept;
    ScratchSlotLock& operator=(ScratchSlotLock&& other) noexcept;

    ScratchSlotLock(ScratchSlotLock const&) = delete;
    ScratchSlotLock& operator=(ScratchSlotLock const&) = delete;

    // Detach and return the slot (transfers ownership to caller).
    Slot detachSlot();

    // Release the slot back to storage manager.
    void unlock();

    Slot const& slot() const noexcept
    {
        return mSlot;
    }

private:
    Slot mSlot;
    KvCache* mOwner;
    LifeCycleId mLifeCycle;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
