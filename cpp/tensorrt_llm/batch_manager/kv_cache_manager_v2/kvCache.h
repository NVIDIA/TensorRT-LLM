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
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/movingAverage.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/pendingStats.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"

#include "tensorrt_llm/common/assert.h"
#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declarations.
class KvCacheIntrospection;
class KvCacheManager;
class StorageManager;
struct ScratchDesc;

// ---------------------------------------------------------------------------
// BlockPage — what a SeqBlock holds per (beamIndex, lifeCycleId):
//   - nullptr             → no page (block not allocated for this lifecycle/beam)
//   - SharedPageLock      → locked (ACTIVE inference)
//   - shared_ptr<PageHolder> → held (suspended, waiting for activation)
// ---------------------------------------------------------------------------
using BlockPage = std::variant<std::monostate, // nullptr
    SharedPageLock,                            // locked
    SharedPtr<PageHolder>                      // held
    >;

inline bool blockPageIsNull(BlockPage const& bp) noexcept
{
    return std::holds_alternative<std::monostate>(bp);
}

inline SharedPtr<Page> const& blockPageGetPage(BlockPage const& bp) noexcept
{
    static SharedPtr<Page> const sNull{};
    if (auto* lock = std::get_if<SharedPageLock>(&bp))
        return lock->isValid() ? lock->page() : sNull;
    if (auto* holder = std::get_if<SharedPtr<PageHolder>>(&bp))
        return holder->get() ? (*holder)->page : sNull;
    return sNull;
}

// ---------------------------------------------------------------------------
// SeqBlock — one block-slot in a KvCache sequence.
// pages[beamIdx][lcId] tracks who holds/locks each page.
// treeBlock: non-null only for committed blocks (strong ref in rare cases).
// ---------------------------------------------------------------------------
using LifeCycleBlockPages = TypedVec<LifeCycleId, BlockPage>;
using BeamBlockPages = TypedVec<BeamIndex, LifeCycleBlockPages>;

struct SeqBlock
{
    BeamBlockPages pages;
    SharedPtr<Block> treeBlock; // non-null iff committed

    bool isCommitted() const noexcept
    {
        bool ret = treeBlock != nullptr;
        if (TLLM_UNLIKELY(gDebug))
        {
            // When committed: must have 1 beam, all non-null pages must be CommittedPage.
            if (ret)
            {
                TLLM_CHECK(pages.size() == BeamIndex{1});
                for (auto const& beamBlock : pages)
                    for (auto const& bp : beamBlock)
                        if (!blockPageIsNull(bp))
                        {
                            auto pg = blockPageGetPage(bp);
                            TLLM_CHECK(!pg || dynamicPointerCast<CommittedPage>(pg));
                        }
            }
            else
            {
                // When not committed: all non-null pages must be UncommittedPage.
                for (auto const& beamBlock : pages)
                    for (auto const& bp : beamBlock)
                        if (!blockPageIsNull(bp))
                        {
                            auto pg = blockPageGetPage(bp);
                            TLLM_CHECK(!pg || dynamicPointerCast<UncommittedPage>(pg));
                        }
            }
        }
        return ret;
    }
};

// ---------------------------------------------------------------------------
// Span<T> — non-owning view into a contiguous buffer.
// Supports operator[] for uniform access with std::vector<int>.
// ---------------------------------------------------------------------------
template <typename T>
struct Span
{
    T* ptr;
    int32_t len;

    T& operator[](int idx)
    {
        return ptr[idx];
    }

    T operator[](int idx) const
    {
        return ptr[idx];
    }

    int size() const noexcept
    {
        return len;
    }

    T* data() const noexcept
    {
        return ptr;
    }

    T* begin() const noexcept
    {
        return ptr;
    }

    T* end() const noexcept
    {
        return ptr + len;
    }
};

// ---------------------------------------------------------------------------
// PlannedDropHandle — tracks committed pages planned for dropping without
// owning them. Mirrors Python's PlannedDropHandle in _core/_kv_cache.py.
//
// The handle stores weak references and does not keep pages alive. Dropping it
// decrements each live page's planned-drop count and removes an already-
// droppable page from eviction tracking when no plans remain.
// ---------------------------------------------------------------------------
class PlannedDropHandle
{
public:
    // Deduplicates `pages` by identity, stores weak references, and increments
    // each page's plannedDropCount.
    explicit PlannedDropHandle(std::vector<CommittedPage*> const& pages);

    // Mirrors Python's __del__: applies the plan if not already dropped.
    ~PlannedDropHandle();

    PlannedDropHandle(PlannedDropHandle const&) = delete;
    PlannedDropHandle& operator=(PlannedDropHandle const&) = delete;

    // Apply this drop plan and invalidate the handle.
    //
    // A live page is removed from eviction tracking only when this is its final
    // plan and it is already droppable and queued for eviction. Calling this
    // method twice throws (translated to Python ValueError).
    void drop();

private:
    // nullopt once dropped (mirrors Python's `_page_refs is None`).
    std::optional<std::vector<WeakPtr<CommittedPage>>> mPageRefs;
};

// ---------------------------------------------------------------------------
// KvCache — manages the per-sequence KV cache state.
// Mirrors Python's _KVCache.
// ---------------------------------------------------------------------------
class KvCache : public std::enable_shared_from_this<KvCache>
{
public:
    enum class Status
    {
        ACTIVE,
        SUSPENDED,
        CLOSED
    };
    enum class CommitState
    {
        ALLOWED,
        VIRTUAL_STOP,
        USER_STOP
    };

    // Priority callback: (blockOrdinal, lifeCycleId) → Priority.
    using PriorityCb = std::function<Priority(BlockOrdinal, LifeCycleId)>;

    KvCache(KvCacheManager& manager, ReuseScope reuseScope, std::vector<TokenIdExt> const& inputTokens,
        std::optional<RequestIdType> id, PriorityCb priorityCb, std::optional<int> expectedPromptLength = std::nullopt);

    ~KvCache();

    KvCache(KvCache const&) = delete;
    KvCache& operator=(KvCache const&) = delete;

    // ---- State machine -----------------------------------------------------

    // Resume: check utilization and lock all pages to GPU.
    // Optionally sets a new CUDA stream; if nullopt, uses the existing one.
    // Returns false if utilization too high or out of memory.
    bool resume(std::optional<CUstream> stream = std::nullopt);

    // Suspend: detach from CUDA stream, unlock pages → PageHolder.
    void suspend();

    // Close: release all blocks back to KvCacheManager.
    void close();

    // Commit or discard request-local statistics accumulated since the previous scheduler commit.
    KVCacheStatsDelta commitPendingStats();
    void discardPendingStats();

    // Best-effort prefetch active pages to the target cache level.
    bool prefetch(CacheLevel target);

    // ---- Capacity / history ------------------------------------------------

    // Resize capacity and/or history_length.
    // Returns true if the resize was a no-op shortcut.
    bool resize(std::optional<int> capacity, std::optional<int> historyLength = std::nullopt);

    // Convenience: set only capacity or history length.
    void setCapacity(int capacity);
    void setHistoryLength(int historyLength);

    // ---- Committing tokens -------------------------------------------------

    // Commit tokens: finalises the oldest uncommitted block and makes it
    // available for reuse by other KvCaches.
    // tokens must contain exactly tokensPerBlock tokens per call (until the last).
    // is_end: if true, records a final reusable snapshot and stops committing.
    // This is a terminal-memory contract: callers must not perform later writes
    // to this KvCache's memory. The final live pages may be moved into the radix
    // tree instead of copied (SSM state and the last partial block).
    void commit(std::vector<TokenIdExt> const& tokens, bool isEnd = false);

    // Stop committing (called by close() automatically).
    void stopCommitting();

    // ---- Page index queries ------------------------------------------------

    // Get base page indices (slot_id) for beamIdx × layerGroupId.
    // Returns a non-owning Span into the internal page-index buffer.
    Span<int const> getBasePageIndices(LayerGroupId lgId, BeamIndex beamIdx = kDefaultBeamIndex) const;

    // Get aggregated (slot-level) page indices for one layer group + beam.
    // Returns one entry per block; bad blocks yield kBadPageIndex.
    // If valid_only=true, bad-index blocks are skipped entirely.
    std::vector<int> getAggregatedPageIndices(
        LayerGroupId lgId, BeamIndex beamIdx = kDefaultBeamIndex, bool validOnly = false) const;

    // Zero-copy page index buffer: copy current base indices into [buf, buf+len)
    // and arrange that future updateBasePageIndex calls write there too.
    // Pass buf=nullptr / len=0 to revert to the internal vector.
    void setBasePageIndexBuf(BeamIndex beamIdx, LayerGroupId lgId, int32_t* buf, int len);

    // ---- Introspection -----------------------------------------------------

    Status status() const noexcept
    {
        return mStatus;
    }

    CommitState commitState() const noexcept
    {
        return mCommitState;
    }

    bool isActive() const noexcept
    {
        return mStatus == Status::ACTIVE;
    }

    bool isClosed() const noexcept
    {
        return mStatus == Status::CLOSED;
    }

    BlockOrdinal numBlocks() const noexcept
    {
        return mBlocks.size();
    }

    TypedVec<BlockOrdinal, SeqBlock> const& blocks() const noexcept
    {
        return mBlocks;
    }

    int numCommittedBlocks() const noexcept
    {
        return mNumCommittedBlocks;
    }

    int numCommittedTokens() const noexcept
    {
        return static_cast<int>(mCommittedTokens.size());
    }

    std::vector<TokenIdExt> const& committedTokens() const noexcept
    {
        return mCommittedTokens;
    }

    ReuseScope const& reuseScope() const noexcept
    {
        return mReuseScope;
    }

    // Plan dropping SWA blocks needed only by the next conversation turn.
    //
    // The plan covers committed pages in each SWA life cycle's current attention
    // window. Full-attention and attention-sink blocks are excluded because
    // later turns may still need them. SSM state is not yet supported. Must be
    // called after stopCommitting(). Returns nullptr without creating a plan if
    // any required SWA page is unavailable. Mirrors Python's
    // _KVCache.plan_committed_block_drop().
    std::shared_ptr<PlannedDropHandle> planCommittedBlockDrop();

    int historyLength() const noexcept
    {
        return mHistoryLength;
    }

    int capacity() const noexcept
    {
        return mCapacity;
    }

    int tokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    BeamIndex beamWidth() const noexcept
    {
        return mBeamWidth;
    }

    CUstream cudaStream() const;

    // Mirrors Python's cuda_stream setter: if already on a stream AND active,
    // make the new stream wait for the old one before switching (cross-stream sync).
    void setCudaStream(CUstream stream)
    {
        if (mCudaStream.has_value())
        {
            if (mStatus == Status::ACTIVE)
            {
                CachedCudaEvent ev(reinterpret_cast<CudaStream>(*mCudaStream));
                ev.waitInStream(reinterpret_cast<CudaStream>(stream));
            }
        }
        else
        {
            TLLM_CHECK_DEBUG(mStatus == Status::SUSPENDED && !mFinishEvent.has_value());
        }
        mCudaStream = stream;
    }

    CachedCudaEvent finishEvent() const;

    // RAII scope guard for _record_event() context manager.
    // Sets mFinishEvent on construction, clears it on destruction (= Python's finally).
    [[nodiscard]] auto recordEventScope()
    {
        TLLM_CHECK_DEBUG(!mFinishEvent.has_value());
        // When mCudaStream is nullopt the cache was never resumed — no GPU work
        // was performed, so no CUDA event synchronization is needed.  Blocks only
        // contain PageHolders (not SharedPageLocks) whose destructors do not read
        // finishEvent.  Mirrors Python's _record_event() early-return path.
        if (mCudaStream.has_value())
        {
            mFinishEvent = CachedCudaEvent(reinterpret_cast<CudaStream>(*mCudaStream));
        }
        return FuncGuard([this]() { mFinishEvent.reset(); });
    }

    // Priority for (blockOrdinal, lifeCycleId) based on the callback.
    Priority getPriority(BlockOrdinal ordinal, LifeCycleId lc) const;

    // Reference to StorageManager (for page acquisition/release).
    StorageManager* storageManager() const;

    KvCacheManager& manager() const noexcept
    {
        return *mManager;
    }

    // ---- SSM support --------------------------------------------------------

    // Return the slot ID for the SSM block at the given layer group / beam.
    // Returns kBadPageIndex if no SSM blocks are allocated.
    int getSsmBlockBaseIndex(LayerGroupId lgId, BeamIndex beamIdx = kDefaultBeamIndex) const;

    // ---- SWA scratch slot management ------------------------------------------

    // Return scratch metadata for a layer group, or nullopt if no scratch blocks.
    std::optional<ScratchDesc> getScratchDesc(LayerGroupId lgId) const;

    // True if any lifecycle has scratch slots allocated.
    bool hasScratchSlots() const;

    // Whether SWA scratch reuse is enabled for this KvCache.
    bool isSwaScratchReuseEnabled() const noexcept;

    // Enable or disable SWA scratch reuse. Throws if the transition is invalid.
    void setEnableSwaScratchReuse(bool enable);

    // Whether the given page index mode is supported (SHARED requires no scratch slots).
    bool supportsIndexMode(PageIndexMode mode) const;

    // ---- Internal callbacks (called by SharedPageLock) ----------------------

    int updateBasePageIndex(BeamIndex bi, BlockOrdinal ord, LifeCycleId lc, int value);

    std::optional<RequestIdType> id; // opaque identifier (mirrors Python's id field)

private:
    friend class KvCacheIntrospection;
    friend std::vector<SharedPageLock> batchedLockToGpu(
        KvCache& kvCache, std::vector<BatchedLockTarget> const& targets);

    // Activate: lock all pages to GPU. mCudaStream must already be set.
    // Internal — called by resume(). Not public (mirrors Python where activate() doesn't exist).
    void activate();

    // Internal helpers.
    void _setupForReuse(std::vector<TokenIdExt> const& inputTokens);
    void _clearBlocks();
    // Copy `srcPage` into a new committed page attached to `treeBlock` for lifecycle
    // `lcIdx`. When `ssmNumTokensInBlock` is set, the copy is an SsmCommittedPage
    // covering that many tokens; otherwise a plain attention CommittedPage. No-op if
    // the block already holds a page for this lifecycle, or on OOM in all levels.
    void _copyPageToTreeBlock(SharedPtr<Block> const& treeBlock, LifeCycleId lcIdx, SharedPtr<Page> const& srcPage,
        std::optional<int> ssmNumTokensInBlock = std::nullopt);

    // Snapshot live SSM state to `treeBlock` reusable at `numTokens` committed tokens.
    // If `move`, the live SSM page is moved (not copied) into the tree — the caller
    // must guarantee no later writes to this KvCache's memory.
    void _snapshotSsmToTreeBlock(
        SharedPtr<Block> const& treeBlock, LifeCycleId ssmLcId, int numTokens, bool move = false);

    // Snapshot a partial (non-full) final block at `ordinal` into the radix tree,
    // copying partial attention pages and optionally the SSM snapshot.
    void _snapshotPartialBlockToTree(BlockOrdinal ordinal, bool commitSsm);
    // Returns [stale_begin, stale_end) block ordinal range for a SWA lifecycle.
    HalfOpenRange<BlockOrdinal> _getStaleRange(int historyLength, LifeCycle const& lc) const;

    // Backup entry for rollback after _unlockStaleBlocks.
    struct StaleBackup
    {
        BlockOrdinal ordinal;
        BeamIndex beamIdx;
        LifeCycleId lcId;
        SharedPtr<PageHolder> holder;
    };

    // Unlock stale SWA blocks. Returns backup holders for rollback.
    std::vector<StaleBackup> _unlockStaleBlocks(int historyLength);

    // Re-lock previously unlocked stale blocks (rollback on OOM).
    void _lockHeldBlocks(std::vector<StaleBackup> const& backup);

    // Iterator over (ordinal, beamIdx, lcIdx) tuples for active (non-stale) pages.
    // Mirrors Python's _active_pages(). Used by activate() for efficient lock.
    struct ActivePage
    {
        BlockOrdinal ordinal;
        BeamIndex beamIdx;
        LifeCycleId lcId;
    };

    std::vector<ActivePage> _activePages() const;
    SharedPtr<Page> _page(BlockOrdinal ordinal, BeamIndex beamIdx, LifeCycleId lcId) const;

    bool _shortcutSetCapacity(int capacity);
    bool _shortcutSetHistoryLength(int historyLength);
    bool _shouldRecordStats() const;
    void _refreshStatsDirtyState();
    void _recordDirectIterationStats(LifeCycleId lifeCycle, KVCacheIterationStatsDelta const& iterationStats);
    void _recordMigratedSlots(std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots,
        CacheLevel srcLevel, CacheLevel dstLevel);
    void _recordDroppedPages(std::vector<SharedPtr<Page>> const& pages, CacheLevel cacheLevel);
    void _refreshGenerationAllocReady();
    void _recordResizePendingAllocations(BlockOrdinal blockBegin, BlockOrdinal blockEnd,
        TypedVec<LifeCycleId, HalfOpenRange<BlockOrdinal>> const& excludedRanges, bool countAsGeneration);
    void _subtractPendingAllocationRange(BlockOrdinal blockBegin, BlockOrdinal blockEnd);
    static bool _hasReuseSource(BlockPage const& page);
    void _increaseCapacity(BlockOrdinal newNumBlocks, int newHistoryLength);
    void _decreaseCapacity(BlockOrdinal newNumBlocks);

    void _evictOutOfWindowBlocks(int historyLength)
    {
        (void) historyLength;
    } // handled by _unlockStaleBlocks

    // Release stale held uncommitted pages for SWA layers after committing stops.
    // Mirrors Python's _on_stop_committing().
    void _onStopCommitting();

    // Commit a single block at ordinal `ord`.
    // `isLast` mirrors Python's is_last parameter: when True (or on VIRTUAL_STOP),
    // transitions to USER_STOP and calls _onStopCommitting() internally.
    // Caller must have recordEventScope() open so finishEvent() works.
    // `commitSsm` snapshots the current SSM state for this block; `moveSsm`
    // moves (vs copies) the live SSM page into the tree (caller must guarantee
    // no later writes to this KvCache's memory). Mirrors Python's _commit_block.
    void _commitBlock(int ord, bool isLast, bool commitSsm = false, bool moveSsm = false);

    struct TakenPage
    {
        SharedPtr<UncommittedPage> page;
        bool locked;
    };

    // Extract uncommitted pages from a SeqBlock, resetting block page entries.
    // Returns one TakenPage per lifecycle. Mirrors Python's _take_uncommitted_page().
    TypedVec<LifeCycleId, TakenPage> _takeUncommittedPage(
        SeqBlock& sb, BeamIndex beamIdx, std::optional<LifeCycleId> skipLc = std::nullopt);

    // Get and validate the tree block at a committed ordinal.
    // Mirrors Python's _get_tree_block(). Asserts committed pages reference the correct block.
    SharedPtr<Block> const& _getTreeBlock(BlockOrdinal ordinal) const;

    // Comprehensive sanity check of KvCache invariants.
    // Mirrors Python's _check_sanity(). Returns true on success (asserts internally).
    bool _checkSanity() const;

    // ---- SWA scratch private helpers ------------------------------------------

    // Compute the scratch block range for a lifecycle.
    HalfOpenRange<BlockOrdinal> _getScratchRange(LifeCycle const& lc, std::optional<int> hlOverride = std::nullopt,
        std::optional<int> capOverride = std::nullopt) const;

    // Result of _takeExcessScratchSlots: excess locks, per-lc delta counts, and scratch ranges.
    struct DeltaScratchSlots
    {
        TypedVec<LifeCycleId, std::vector<ScratchSlotLock>> excess;
        TypedVec<LifeCycleId, int> deltaCnt;
        TypedVec<LifeCycleId, HalfOpenRange<BlockOrdinal>> scratchRanges;
    };

    // Compute and remove excess scratch slots for a new capacity/historyLength.
    DeltaScratchSlots _takeExcessScratchSlots(int capacity, int historyLength);

    // Recover previously taken excess scratch slots back into mScratchSlots.
    void _recoverExcessScratchSlots(TypedVec<LifeCycleId, std::vector<ScratchSlotLock>>& excess);

    // Release all scratch slots back to storage.
    void _freeScratchSlots();

    // Whether any lifecycle would require scratch blocks at the current state.
    bool _wouldUseSwaScratchBlocks() const;
    int _swaScratchMaxRewindLen() const;

    // Page index table management.
    // _basePageIndices[beamIdx][lcId][blockOrdinal] = slotId or BAD
    void _checkPageIndexBufferCapacity(BlockOrdinal newNumBlocks) const;
    void _resizePageIndexBuffers(BlockOrdinal newNumBlocks);

    std::shared_ptr<KvCacheManager> mManager;
    ReuseScope mReuseScope;
    PriorityCb mPriorityCb;
    std::optional<CUstream> mCudaStream;
    Status mStatus;
    CommitState mCommitState;
    BeamIndex mBeamWidth;
    int mCapacity;
    int mHistoryLength;
    std::optional<int> mExpectedPromptLength;
    bool mGenerationAllocReady = false;

    // Page index tables: [beamIdx][lcId] → either an internal vector or an external span.
    // Mirrors Python's IndexSeq = array.array | memoryview.
    using PageIndexBuf = std::variant<std::vector<int>, Span<int>>;
    using LifeCyclePageIndexBuffers = TypedVec<LifeCycleId, PageIndexBuf>;
    using BeamPageIndexBuffers = TypedVec<BeamIndex, LifeCyclePageIndexBuffers>;
    BeamPageIndexBuffers mBasePageIndices;

    TypedVec<BlockOrdinal, SeqBlock> mBlocks;

    std::vector<TokenIdExt> mCommittedTokens;
    int mNumCommittedBlocks;
    std::optional<CachedCudaEvent> mFinishEvent;
    int mTokensPerBlock;
    Average mAvgHistoryLength;
    Average mAvgCapacity;

    // SSM pages: [beamIdx][lcId] — always initialized (empty entries = monostate).
    BeamBlockPages mSsmBlocks;
    bool mNeverResumed = true;

    PendingStats mPendingStats;

    // SWA scratch slot support.
    bool mEnableSwaScratchReuse = false;
    TypedVec<LifeCycleId, std::vector<ScratchSlotLock>> mScratchSlots;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
