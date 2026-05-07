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

#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declarations.
class KvCacheManager;
class StorageManager;

// ---------------------------------------------------------------------------
// BlockPage — what a SeqBlock holds per (beamIndex, lifeCycleId):
//   - nullptr             → no page (block not allocated for this lifecycle/beam)
//   - SharedPageLock      → locked (ACTIVE inference)
//   - shared_ptr<PageHolder> → held (suspended, waiting for activation)
// ---------------------------------------------------------------------------
using BlockPage = std::variant<std::monostate, // nullptr
    SharedPageLock,                            // locked
    std::shared_ptr<PageHolder>                // held
    >;

inline bool blockPageIsNull(BlockPage const& bp) noexcept
{
    return std::holds_alternative<std::monostate>(bp);
}

inline std::shared_ptr<Page> const& blockPageGetPage(BlockPage const& bp) noexcept
{
    static const std::shared_ptr<Page> sNull{};
    if (auto* lock = std::get_if<SharedPageLock>(&bp))
        return lock->isValid() ? lock->page() : sNull;
    if (auto* holder = std::get_if<std::shared_ptr<PageHolder>>(&bp))
        return holder->get() ? (*holder)->page : sNull;
    return sNull;
}

// ---------------------------------------------------------------------------
// SeqBlock — one block-slot in a KvCache sequence.
// pages[beamIdx][lcId] tracks who holds/locks each page.
// treeBlock: non-null only for committed blocks (strong ref in rare cases).
// ---------------------------------------------------------------------------
struct SeqBlock
{
    // [beamIdx][lcId]
    std::vector<std::vector<BlockPage>> pages;
    std::shared_ptr<Block> treeBlock; // non-null iff committed

    bool isCommitted() const noexcept
    {
        bool ret = treeBlock != nullptr;
        if (!gNdebug)
        {
            // When committed: must have 1 beam, all non-null pages must be CommittedPage.
            if (ret)
            {
                assert(pages.size() == 1);
                for (auto const& beamBlock : pages)
                    for (auto const& bp : beamBlock)
                        if (!blockPageIsNull(bp))
                        {
                            auto pg = blockPageGetPage(bp);
                            assert(!pg || std::dynamic_pointer_cast<CommittedPage>(pg));
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
                            assert(!pg || std::dynamic_pointer_cast<UncommittedPage>(pg));
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

    KvCache(KvCacheManager& manager, std::optional<int64_t> loraTaskId, std::vector<TokenIdExt> const& inputTokens,
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
    Span<int const> getBasePageIndices(LayerGroupId lgId, BeamIndex beamIdx = 0) const;

    // Get aggregated (slot-level) page indices for one layer group + beam.
    // Returns one entry per block; bad blocks yield kBadPageIndex.
    // If valid_only=true, bad-index blocks are skipped entirely.
    std::vector<int> getAggregatedPageIndices(LayerGroupId lgId, BeamIndex beamIdx = 0, bool validOnly = false) const;

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

    int numBlocks() const noexcept
    {
        return static_cast<int>(mBlocks.size());
    }

    std::vector<SeqBlock> const& blocks() const noexcept
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

    int beamWidth() const noexcept
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
            assert(mStatus == Status::SUSPENDED && !mFinishEvent.has_value());
        }
        mCudaStream = stream;
    }

    CachedCudaEvent finishEvent() const;

    // RAII scope guard for _record_event() context manager.
    // Sets mFinishEvent on construction, clears it on destruction (= Python's finally).
    [[nodiscard]] auto recordEventScope()
    {
        assert(!mFinishEvent.has_value());
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
    int getSsmBlockBaseIndex(LayerGroupId lgId, BeamIndex beamIdx = 0) const;

    // ---- Internal callbacks (called by SharedPageLock) ----------------------

    int updateBasePageIndex(BeamIndex bi, BlockOrdinal ord, LifeCycleId lc, int value);

    std::optional<int64_t> id; // opaque identifier (mirrors Python's id field)

private:
    // Activate: lock all pages to GPU. mCudaStream must already be set.
    // Internal — called by resume(). Not public (mirrors Python where activate() doesn't exist).
    void activate();

    // Internal helpers.
    void _setupForReuse(std::vector<TokenIdExt> const& inputTokens);
    void _clearBlocks();
    // Snapshot live SSM state to a new page and attach to radix tree block.
    void _snapshotSsmToTreeBlock(std::shared_ptr<Block> const& treeBlock, LifeCycleId ssmLcId, BeamIndex beamIdx);
    // Returns [stale_begin, stale_end) block ordinal range for a SWA lifecycle.
    HalfOpenRange _getStaleRange(int historyLength, LifeCycle const& lc) const;

    // Backup entry for rollback after _unlockStaleBlocks.
    struct StaleBackup
    {
        BlockOrdinal ordinal;
        BeamIndex beamIdx;
        LifeCycleId lcId;
        std::shared_ptr<PageHolder> holder;
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

    bool _shortcutSetCapacity(int capacity);
    bool _shortcutSetHistoryLength(int historyLength);
    void _increaseCapacity(int newNumBlocks, int newHistoryLength);
    void _decreaseCapacity(int newNumBlocks);

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
        std::shared_ptr<UncommittedPage> page;
        bool locked;
    };

    // Extract uncommitted pages from a SeqBlock, resetting block page entries.
    // Returns one TakenPage per lifecycle. Mirrors Python's _take_uncommitted_page().
    std::vector<TakenPage> _takeUncommittedPage(
        SeqBlock& sb, BeamIndex beamIdx, std::optional<LifeCycleId> skipLc = std::nullopt);

    // Get and validate the tree block at a committed ordinal.
    // Mirrors Python's _get_tree_block(). Asserts committed pages reference the correct block.
    std::shared_ptr<Block> const& _getTreeBlock(BlockOrdinal ordinal) const;

    // Comprehensive sanity check of KvCache invariants.
    // Mirrors Python's _check_sanity(). Returns true on success (asserts internally).
    bool _checkSanity() const;

    // Page index table management.
    // _basePageIndices[beamIdx][lcId][blockOrdinal] = slotId or BAD
    void _resizePageIndexBuffers(int newNumBlocks);

    KvCacheManager* mManager;
    std::optional<int64_t> mLoraTaskId;
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
    std::vector<std::vector<PageIndexBuf>> mBasePageIndices; // [beamIdx][lcId]

    std::vector<SeqBlock> mBlocks;

    std::vector<TokenIdExt> mCommittedTokens;
    int mNumCommittedBlocks;
    std::optional<CachedCudaEvent> mFinishEvent;
    int mTokensPerBlock;
    Average mAvgHistoryLength;
    Average mAvgCapacity;

    // SSM pages: [beamIdx][lcId] — always initialized (empty entries = monostate).
    std::vector<std::vector<BlockPage>> mSsmBlocks;
    bool mNeverResumed = true;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
