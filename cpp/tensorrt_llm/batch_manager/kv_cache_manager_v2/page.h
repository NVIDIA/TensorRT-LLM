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
class SparsePageBacking;

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
    // Immutable: PrioritizedEvictionPolicy locates a scheduled page's sub-queue by this value,
    // so changing it while the page is scheduled would erase from the wrong list.
    Priority const priority;
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

    //! Return the lock destination for read-only use, preserving pages already at the hot GPU level.
    //! Cold sparse pages use host memory; callers must force writable pages to GPU.
    [[nodiscard]] CacheLevel queryLockLevel() const;

    // Acquire a shared lock at the current level. The caller handles any migration.
    // skip_wait: caller guarantees the page is ready on kvCache's stream.
    SharedPageLock lock(
        KvCache& kvCache, BeamIndex beamIndex, BlockOrdinal ordinal, LifeCycleId lifeCycle, bool skipWait = false);
};

// ---------------------------------------------------------------------------
// CommittedPage — immutable block payload, canonical or request-private.
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

    // Token count recorded for this page. It is usually block->tokens.size(), but a
    // snapshot taken at an earlier token boundary may live in a block that spans more
    // tokens — see addOrGetExistingBlock() and KvCache::_snapshotPartialBlockToTree().
    //
    // Attention and SSM life cycles interpret it differently:
    //   * for attention pages, it is the number of leading tokens with valid per-token KV,
    //     so the page is reusable for any prefix up to that count (compare with `>=`);
    //   * for an SSM page, it is the exact recurrent-state checkpoint, so reuse must be
    //     truncated to exactly that boundary.
    int numTokensInBlock;

    // Number of outstanding PlannedDropHandles that intend to drop this page.
    // Mirrors Python's CommittedPage.planned_drop_count.
    int plannedDropCount{0};

    CommittedPage(StorageManager* mgr, SharedPtr<Block> blk, LifeCycleId lc, CacheLevel level, int numTokensInBlock,
        Priority prio);

    ~CommittedPage() override;

    bool isCommitted() const override
    {
        return true;
    }

    //! Claim a GPU allocation for one request, or report that a private copy is required.
    bool claimSparseGpu(KvCache& kvCache);

    //! Shared host backing for a complete immutable sparse block; partial pages have none.
    SharedPtr<SparsePageBacking> sparseBacking();

private:
    friend class SparsePageBacking;
    friend class StorageManager;
    friend class UniqPageLock;
    friend class UncommittedPage;
    std::weak_ptr<KvCache> mSparseGpuOwner;
    SharedPtr<SparsePageBacking> mSparseBacking;
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

    UncommittedPage(KvCache& kvc, BlockOrdinal ord, LifeCycleId lc, CacheLevel level, BeamIndex bi = kDefaultBeamIndex);

    ~UncommittedPage() override;

    bool isCommitted() const override
    {
        return false;
    }

    // Convert this UncommittedPage into a CommittedPage. A private sparse copy
    // keeps its GPU slot and shares the existing block's host-backing record.
    // The UncommittedPage becomes invalid (slot transferred to CommittedPage).
    //
    // `numTokensInBlock` records the page's token count. See
    // CommittedPage::numTokensInBlock for its attention and SSM interpretations.
    SharedPtr<CommittedPage> convertToCommitted(
        SharedPtr<Block> block, CachedCudaEvent readyEvent, int numTokensInBlock, bool privateSparseCopy = false);
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

    //! Pin storage without installing an execution mapping in a request.
    SharedPtr<UniqPageLock> pin();

    SharedPtr<Page> page;
    WeakPtr<UniqPageLock> uniqLock; // non-null → LOCKED
};

//! Identifies one live SharedPageLock independently of the lock object's address.
struct LockOwner
{
    KvCache* kvCache;
    BeamIndex beamIndex;
    BlockOrdinal ordinal;
    LifeCycleId lifeCycle;

    bool operator==(LockOwner const&) const = default;
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

    //! Validate complete sparse history owned by the requesting cache.
    void prepareSparseOffload(KvCache const& requestingCache);

    //! Record a copy ordered after page readiness, finished readers, and all live owners' prior work.
    void recordOffloadEvent(CachedCudaEvent const& event);

    //! Publish the host slot to every owner and return the fenced GPU slot. Caller holds the API lock.
    [[nodiscard]] Slot moveToSparseHistory(Slot&& hostSlot);

    //! Reserve owner storage before replacing a request's execution binding.
    void reserveOwners(size_t count);

    //! Release a private GPU allocation after all execution bindings have been removed.
    void releaseSparseGpuSlot();

    std::vector<LockOwner> const& owners() const noexcept
    {
        return mOwners;
    }

    SharedPtr<PageHolder> holder;
    std::vector<CachedCudaEvent> finishEvents;

private:
    friend class SharedPageLock;
    void removeOwner(LockOwner const& owner);

    std::vector<LockOwner> mOwners;
};

//! Host backing shared by request-private copies of one full immutable block.
//! The host page does not retain this object, avoiding a storage ownership cycle.
class SparsePageBacking
{
public:
    explicit SparsePageBacking(SharedPtr<Block> const& block);

    SharedPtr<UniqPageLock> hostLock(LifeCycleId lifeCycle);
    void publishHost(SharedPtr<UniqPageLock> const& lock);

private:
    WeakPtr<Block> mBlock;
    SharedPtr<UniqPageLock> mHostLock;
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

    [[nodiscard]] SharedPtr<Page> const& page() const;

    [[nodiscard]] bool isValid() const noexcept
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
// BatchedLockTarget — page, owner, and intended storage level for a lock.
// ---------------------------------------------------------------------------
struct BatchedLockTarget
{
    SharedPtr<Page> page;
    BeamIndex beamIndex;
    BlockOrdinal ordinal;
    LifeCycleId lifeCycle;
    CacheLevel cacheLevel = kHotLevel;
};

// ---------------------------------------------------------------------------
// batchedLockPages — restore pages to their intended levels, then lock them.
// Returns one SharedPageLock per target.
// Sparse GPU destinations are private to the receiving request.
// ---------------------------------------------------------------------------
std::vector<SharedPageLock> batchedLockPages(KvCache& kvCache, std::vector<BatchedLockTarget> const& targets);

// ---------------------------------------------------------------------------
// ScratchSlotLock — manages a scratch slot for SWA prefill memory reuse.
// Wraps a Slot with owner (KvCache) and lifecycle references.
// GPU-only: on destruction, releases the slot back to the GPU storage pool.
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
