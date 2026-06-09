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
        std::optional<int64_t> id, PriorityCb priorityCb);

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
    void commit(std::vector<TokenIdExt> const& tokens);

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

    std::optional<int64_t> id; // opaque identifier (mirrors Python's id field)

private:
    friend class KvCacheIntrospection;

    // Activate: lock all pages to GPU. mCudaStream must already be set.
    // Internal — called by resume(). Not public (mirrors Python where activate() doesn't exist).
    void activate();

    // Internal helpers.
    void _setupForReuse(std::vector<TokenIdExt> const& inputTokens);
    void _clearBlocks();
    // Snapshot live SSM state to a new page and attach to radix tree block.
    void _snapshotSsmToTreeBlock(SharedPtr<Block> const& treeBlock, LifeCycleId ssmLcId, BeamIndex beamIdx);
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
    // Mirrors Python's _commit_block(ordinal, is_last).
    void _commitBlock(int ord, bool isLast);

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

    // SWA scratch slot support.
    bool mEnableSwaScratchReuse = false;
    TypedVec<LifeCycleId, std::vector<ScratchSlotLock>> mScratchSlots;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
