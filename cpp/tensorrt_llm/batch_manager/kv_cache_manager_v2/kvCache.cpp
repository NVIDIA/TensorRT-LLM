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

#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

int64_t sumSlotBytes(StorageManager const& storage, CacheLevel level, LifeCycleId lifeCycle)
{
    PoolGroupIndex const poolGroup = storage.getPoolGroupIndex(level, lifeCycle);
    int64_t pageSize = 0;
    for (size_t const size : storage.slotSize(level, poolGroup))
    {
        pageSize += static_cast<int64_t>(size);
    }
    return pageSize;
}

} // namespace

// ---------------------------------------------------------------------------
// KvCache constructor
// ---------------------------------------------------------------------------

KvCache::KvCache(KvCacheManager& manager, ReuseScope reuseScope, std::optional<BlockRadixTree::ReuseMatch> reuseMatch,
    std::optional<RequestIdType> mId, PriorityCb priorityCb, std::optional<int> expectedPromptLength,
    std::optional<bool> textOnly, bool enableRequestStats)
    : id(mId)
    , mManager(manager.shared_from_this())
    , mReuseScope(std::move(reuseScope))
    , mPriorityCb(priorityCb ? std::move(priorityCb) : [](BlockOrdinal, LifeCycleId) { return kPriorityDefault; })
    , mStatus(Status::SUSPENDED)
    , mCommitState(CommitState::ALLOWED)
    , mBeamWidth(BeamIndex{1})
    , mCapacity(0)
    , mHistoryLength(0)
    , mExpectedPromptLength(
          expectedPromptLength.has_value() ? std::optional<int>{std::max(*expectedPromptLength, 0)} : std::nullopt)
    , mNumTokensBeforeHybridPruning(reuseMatch.has_value() ? reuseMatch->numTokensBeforeHybridPruning : 0)
    , mEnableRequestStats(enableRequestStats)
    , mNumCommittedBlocks(0)
    , mTokensPerBlock(manager.tokensPerBlock())
{
    LifeCycleId numLc = manager.storage().numLifeCycles();

    // Initialise page index buffers: [beamIdx][lcId] = empty vector
    mBasePageIndices.resize(mBeamWidth);
    for (auto& beamIndices : mBasePageIndices)
    {
        beamIndices.resize(numLc, PageIndexBuf{std::vector<int>{}});
    }

    // Always initialise mSsmBlocks (matching Python: no longer optional).
    mSsmBlocks.resize(mBeamWidth);
    for (auto& beam : mSsmBlocks)
    {
        beam.resize(numLc); // default-constructs to monostate
    }

    mEnableSwaScratchReuse = manager.isSwaScratchReuseEnabled();
    mScratchSlots.resize(manager.storage().numLifeCycles());

    TLLM_CHECK_WITH_INFO(!(manager.textOnly() && textOnly == std::optional<bool>{false}),
        "text_only=false is not allowed when the manager is configured text_only=true");
    mTextOnly = textOnly.value_or(manager.textOnly());

    if (reuseMatch.has_value())
    {
        _setupForReuse(*reuseMatch);
    }

    _refreshGenerationAllocReady();

    mAvgHistoryLength.update(static_cast<double>(mHistoryLength));

    mManager->registerKvCache(this);
    mManager->updateAvgReusedLength(static_cast<double>(mHistoryLength));
    TLLM_CHECK_DEBUG(_checkSanity());
}

KvCache::~KvCache()
{
    try
    {
        close();
    }
    catch (...)
    {
        // Destructors must not propagate exceptions (implicitly noexcept in C++11).
        // close() should not throw in normal usage; if it does, suppress and accept leak.
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

CUstream KvCache::cudaStream() const
{
    TLLM_CHECK_DEBUG_WITH_INFO(mCudaStream.has_value(), "No CUDA stream attached");
    return *mCudaStream;
}

CachedCudaEvent KvCache::finishEvent() const
{
    return mFinishEvent.value();
}

Priority KvCache::getPriority(BlockOrdinal ordinal, LifeCycleId lc) const
{
    return mPriorityCb(ordinal, lc);
}

StorageManager* KvCache::storageManager() const
{
    return &mManager->storage();
}

std::vector<KvCache::ActivePage> KvCache::_activePages() const
{
    std::vector<ActivePage> result;
    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    auto const& lcs = mManager->lifeCycles();
    LifeCycleId numLc = mManager->storage().numLifeCycles();

    for (LifeCycleId lcId{0}; lcId < numLc; ++lcId)
    {
        // SSM lifecycle → yield from mSsmBlocks (check individual entries).
        if (ssmLcId.has_value() && lcId == *ssmLcId)
        {
            for (BeamIndex bi{0}; bi < mBeamWidth; ++bi)
            {
                if (!blockPageIsNull(mSsmBlocks[bi][*ssmLcId]))
                    result.push_back({kBadBlockOrdinal, bi, lcId});
            }
            continue;
        }

        // Attention lifecycle: yield non-stale blocks (sink + window).
        LifeCycle const& lc = lcs.getLifeCycle(lcId);
        auto staleRange = _getStaleRange(mHistoryLength, lc);
        BlockOrdinal staleBeg = staleRange.beg;
        BlockOrdinal staleEnd = staleRange.end;
        auto scratchRange = _getScratchRange(lc);

        // Sink blocks: [0, staleBeg)
        for (BlockOrdinal ord{0}; ord < staleBeg; ++ord)
            for (BeamIndex bi{0}; bi < mBeamWidth; ++bi)
                result.push_back({ord, bi, lcId});

        // Window blocks: [staleEnd, numBlocks) — skip scratch blocks.
        for (BlockOrdinal ord{staleEnd}; ord < mBlocks.size(); ++ord)
        {
            auto& block = mBlocks[ord];
            for (BeamIndex bi{0}; bi < mBeamWidth; ++bi)
            {
                bool isScratch = scratchRange.contains(ord);
                TLLM_CHECK_DEBUG(isScratch == blockPageIsNull(block.pages[bi][lcId]));
                if (!isScratch)
                    result.push_back({ord, bi, lcId});
            }
        }
    }
    return result;
}

SharedPtr<Page> KvCache::_page(BlockOrdinal ordinal, BeamIndex beamIdx, LifeCycleId lcId) const
{
    bool const isSsm = ordinal == kBadBlockOrdinal;
    auto const ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    TLLM_CHECK_DEBUG((ssmLcId.has_value() && lcId == *ssmLcId) == isSsm);
    auto const& blockPage = isSsm ? mSsmBlocks.at(beamIdx).at(lcId) : mBlocks.at(ordinal).pages.at(beamIdx).at(lcId);
    return blockPageGetPage(blockPage);
}

void KvCache::activate()
{
    TLLM_CHECK_DEBUG(mStatus == Status::SUSPENDED);
    TLLM_CHECK_DEBUG_WITH_INFO(mCudaStream.has_value(), "cuda_stream must be set before activate()");

    mFinishEvent.reset();

    // Lock only active (non-stale) pages to GPU — mirrors Python's _active_pages().
    auto activePages = _activePages();
    std::vector<BatchedLockTarget> targets;
    targets.reserve(activePages.size());

    for (auto const& ap : activePages)
    {
        BlockPage* bp = nullptr;
        if (ap.ordinal == kBadBlockOrdinal)
        {
            bp = &mSsmBlocks[ap.beamIdx][ap.lcId];
        }
        else
        {
            bp = &mBlocks[ap.ordinal].pages[ap.beamIdx][ap.lcId];
        }
        auto& holder = std::get<SharedPtr<PageHolder>>(*bp);
        TLLM_CHECK_DEBUG(holder);
        targets.push_back({holder->page, ap.beamIdx, ap.ordinal, ap.lcId});
    }

    {
        auto locks = batchedLockToGpu(*this, targets);
        size_t idx = 0;
        for (auto& t : targets)
        {
            TLLM_CHECK_DEBUG(t.page == locks[idx].page());
            BeamIndex bi = t.beamIndex;
            LifeCycleId lc = t.lifeCycle;
            if (t.ordinal == kBadBlockOrdinal)
                mSsmBlocks[bi][lc] = std::move(locks[idx++]);
            else
                mBlocks[t.ordinal].pages[bi][lc] = std::move(locks[idx++]);
        }
    }
}

bool KvCache::resume(std::optional<CUstream> stream)
{
    TLLM_CHECK_DEBUG(mStatus == Status::SUSPENDED);

    // Set stream first (mirrors Python: self.cuda_stream = cuda_stream).
    if (stream.has_value())
    {
        setCudaStream(*stream);
    }
    TLLM_CHECK_DEBUG_WITH_INFO(mCudaStream.has_value(), "cuda_stream is never set");
    TLLM_CHECK_DEBUG(!mFinishEvent.has_value());

    // Check utilization against threshold.
    auto const utilizations = mManager->storage().getUtilization(kHotLevel);
    float const utilization = utilizations.empty() ? 0.f : *std::max_element(utilizations.begin(), utilizations.end());
    if (utilization > mManager->config().maxUtilForResume)
    {
        return false;
    }

    auto& storageMgr = mManager->storage();
    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    LifeCycleId numLc = storageMgr.numLifeCycles();

    // Pre-allocate GPU slots for deferred copies (partial blocks + SSM) and scratch slots
    // before locking, so we never end up in a state where pages are locked but we can't allocate.
    TypedVec<LifeCycleId, std::optional<Slot>> deferredSlots(numLc);

    // Compute scratch slot deltas UNCONDITIONALLY (mirrors Python: _take_excess_scratch_slots
    // is called outside _never_resumed).
    auto [excessScratch, scratchDeltaCounts, scratchRanges] = _takeExcessScratchSlots(mCapacity, mHistoryLength);
    TLLM_CHECK_DEBUG(excessScratch.size() == numLc
        && std::all_of(excessScratch.begin(), excessScratch.end(), [](auto const& s) { return s.empty(); }));

    TypedVec<LifeCycleId, SlotCount> numSlotsNeeded(numLc, 0);
    bool hasPartial = false;
    if (mNeverResumed)
    {
        TLLM_CHECK_DEBUG(mBeamWidth == BeamIndex{1});
        hasPartial = numCommittedTokens() % mTokensPerBlock != 0;
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            bool isSsm = ssmLcId.has_value() && lc == *ssmLcId;
            if (isSsm || hasPartial)
                numSlotsNeeded[lc] += 1;
        }
    }

    // Add scratch slot needs UNCONDITIONALLY (mirrors Python: delta loop is outside _never_resumed).
    for (LifeCycleId lc{0}; lc < numLc; ++lc)
        numSlotsNeeded[lc] += std::max(0, scratchDeltaCounts[lc]);

    // Only allocate if any slots are needed.
    bool anyNeeded = std::any_of(numSlotsNeeded.begin(), numSlotsNeeded.end(), [](SlotCount n) { return n > 0; });
    if (anyNeeded)
    {
        TypedVec<LifeCycleId, std::vector<Slot>> tmpSlots;
        try
        {
            MigrationRecorder const migrationRecorder
                = [this](std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots, CacheLevel srcLevel,
                      CacheLevel dstLevel) { _recordMigratedSlots(pages, slots, srcLevel, dstLevel); };
            DropRecorder const dropRecorder = [this](std::vector<SharedPtr<Page>> const& pages, CacheLevel cacheLevel)
            { _recordDroppedPages(pages, cacheLevel); };
            tmpSlots = storageMgr.newGpuSlots(numSlotsNeeded, migrationRecorder, dropRecorder);
        }
        catch (OutOfPagesError const&)
        {
            return false;
        }

        // Separate deferred vs scratch slots, and collect scratch ready events.
        std::vector<CachedCudaEvent const*> scratchReadyEvents;
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            if (!tmpSlots[lc].empty())
            {
                // Mirrors Python: `if self._never_resumed and (... SsmLifeCycle or has_partial):`
                bool needsDeferred = mNeverResumed && ((ssmLcId.has_value() && lc == *ssmLcId) || hasPartial);
                if (needsDeferred)
                {
                    // Python uses pop() here: reserve one slot for deferred copy, then treat the rest as scratch.
                    deferredSlots[lc] = std::move(tmpSlots[lc].back());
                    tmpSlots[lc].pop_back();
                }
                // Remaining slots are scratch slots.
                auto& scratchSlots = mScratchSlots[lc];
                scratchSlots.reserve(scratchSlots.size() + tmpSlots[lc].size());
                for (auto& slot : tmpSlots[lc])
                {
                    scratchSlots.emplace_back(std::move(slot), *this, lc,
                        /*skipWait=*/true);
                    scratchReadyEvents.push_back(&scratchSlots.back().slot().readyEvent);
                }
            }
        }

        // Wait only for newly-added scratch slots (mirrors Python's
        // stream_wait_events for scratch_slots_to_add).
        if (!scratchReadyEvents.empty())
            streamWaitEvents(reinterpret_cast<CudaStream>(cudaStream()), scratchReadyEvents);
    }

    try
    {
        activate();
    }
    catch (OutOfPagesError const&)
    {
        // Release pre-allocated deferred slots on failure.
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            if (deferredSlots[lc].has_value())
                storageMgr.releaseSlot(lc, kHotLevel, std::move(*deferredSlots[lc]));
        }
        // Scratch slots stay in mScratchSlots — they'll be freed by close() inside
        // a recordEventScope, matching Python behavior.
        return false;
    }

    // Deferred copy: for partial blocks and SSM, copy from now-locked source pages
    // to pre-allocated GPU slots, then unlock sources and replace with new pages.
    if (mNeverResumed)
    {
        BeamIndex beamIdx = kDefaultBeamIndex;
        auto const lastOrdinal = BlockOrdinal{mBlocks.empty() ? 0 : (numCommittedTokens() - 1) / mTokensPerBlock};
        CUstream cudaStr = cudaStream();
        bool const recordManagerStats = _shouldRecordManagerStats();
        bool const recordRequestStats = _shouldRecordRequestStats();

        // Wait for all new slots to be ready (deduplicated).
        {
            std::vector<CachedCudaEvent const*> slotEvents;
            for (auto& optSlot : deferredSlots)
            {
                if (optSlot.has_value())
                    slotEvents.push_back(&optSlot->readyEvent);
            }
            streamWaitEvents(reinterpret_cast<CudaStream>(cudaStr), slotEvents);
        }

        // Phase 1: Copy GPU→GPU from locked source pages to pre-allocated slots.
        std::vector<SharedPageLock*> srcLocks;
        for (LifeCycleId lcIdx{0}; lcIdx < numLc; ++lcIdx)
        {
            if (!deferredSlots[lcIdx].has_value())
                continue;
            auto& newSlot = *deferredSlots[lcIdx];

            BlockPage* sourcePage = nullptr;
            if (ssmLcId.has_value() && lcIdx == *ssmLcId)
            {
                if (numCommittedTokens() == 0)
                    continue; // fresh SSM — no source to copy from
                sourcePage = &mSsmBlocks[beamIdx][lcIdx];
            }
            else
            {
                sourcePage = &mBlocks[lastOrdinal].pages[beamIdx][lcIdx];
            }
            auto* lock = std::get_if<SharedPageLock>(sourcePage);
            TLLM_CHECK_DEBUG(lock && lock->isValid());
            bool const hasPartialReuseSource = _hasReuseSource(*sourcePage);
            srcLocks.push_back(lock);

            storageMgr.copySlotData(lcIdx, kHotLevel, kHotLevel, newSlot.slotId(), lock->page()->slotId(), cudaStr);
            if ((!ssmLcId.has_value() || lcIdx != *ssmLcId) && (recordManagerStats || recordRequestStats))
            {
                bool const changed = mPendingStats.recordAllocationRange(lcIdx, lastOrdinal, lastOrdinal + 1,
                    /*beamWidth=*/1, /*countAsMissed=*/!hasPartialReuseSource, /*countAsGeneration=*/false,
                    recordManagerStats, recordRequestStats);
                if (changed)
                {
                    mManager->markStatsDirty(id);
                }
            }
            KVCacheIterationStatsDelta iterationStats;
            iterationStats.iterIntraDeviceCopyBlocks = 1;
            iterationStats.iterIntraDeviceCopyBytes = sumSlotBytes(storageMgr, kHotLevel, lcIdx);
            _recordDirectIterationStats(lcIdx, iterationStats);
        }

        // Unlock source pages — recordEventScope captures all prior CUDA work
        // so the original pages know when we're done reading from them.
        if (!srcLocks.empty())
        {
            auto scope = recordEventScope();
            for (auto* lock : srcLocks)
                lock->unlock();
        }

        // Phase 2: Replace with new UncommittedPages (both copied and fresh SSM).
        for (LifeCycleId lcIdx{0}; lcIdx < numLc; ++lcIdx)
        {
            if (!deferredSlots[lcIdx].has_value())
                continue;
            auto& newSlot = *deferredSlots[lcIdx];

            BlockPage* targetBp;
            BlockOrdinal blockOrdinal;
            if (ssmLcId.has_value() && lcIdx == *ssmLcId)
            {
                targetBp = &mSsmBlocks[beamIdx][lcIdx];
                blockOrdinal = kBadBlockOrdinal;
            }
            else
            {
                targetBp = &mBlocks[lastOrdinal].pages[beamIdx][lcIdx];
                blockOrdinal = lastOrdinal;
            }

            auto newPage = makeShared<UncommittedPage>(*this, blockOrdinal, lcIdx, kHotLevel, beamIdx);
            newPage->setSlot(newSlot);
            auto newLock = newPage->lock(*this, beamIdx, blockOrdinal, lcIdx, /*skipWait=*/true);
            *targetBp = std::move(newLock);
        }

        // Clear treeBlock for partial last block (mirrors Python: partial block is uncommitted).
        if (numCommittedTokens() % mTokensPerBlock != 0)
            mBlocks[lastOrdinal].treeBlock = nullptr;
    }

    // A freshly-created cache starts SUSPENDED and is activated by this same
    // resume() call, so gate the counter on mNeverResumed: only a cache that was
    // previously ACTIVE and got suspended counts as a preemption recovery.
    // Without this, the counter would track request admissions, not preemption.
    bool const firstActivation = mNeverResumed;
    mNeverResumed = false;
    mStatus = Status::ACTIVE;
    if (!firstActivation && _shouldRecordStats())
    {
        mManager->recordRequestResumed();
    }
    return true;
}

bool KvCache::prefetch(CacheLevel target)
{
    TLLM_CHECK_DEBUG(mStatus == Status::SUSPENDED);
    auto& storageMgr = mManager->storage();
    CacheLevel const numTiers = storageMgr.numCacheLevels();
    TLLM_CHECK_DEBUG(kHotLevel <= target && target < numTiers);

    LifeCycleId const numLifeCycles = storageMgr.numLifeCycles();
    TypedVec<LifeCycleId, TypedVec<CacheLevel, std::vector<SharedPtr<Page>>>> allPages(
        numLifeCycles, TypedVec<CacheLevel, std::vector<SharedPtr<Page>>>(numTiers));

    for (auto const& activePage : _activePages())
    {
        auto page = _page(activePage.ordinal, activePage.beamIdx, activePage.lcId);
        if (!page)
        {
            continue;
        }
        CacheLevel const level = page->cacheLevel;
        if (level < target)
        {
            continue;
        }
        allPages.at(activePage.lcId).at(level).push_back(std::move(page));
    }

    try
    {
        storageMgr.prefetch(target, allPages);
    }
    catch (OutOfPagesError const&)
    {
        return false;
    }
    return true;
}

void KvCache::suspend()
{
    TLLM_CHECK_DEBUG(mStatus == Status::ACTIVE);
    TLLM_CHECK_DEBUG(_checkSanity());
    TLLM_CHECK_DEBUG(!mFinishEvent.has_value());

    // Copy data from external buffers back to internal vectors (mirrors Python's suspend).
    for (BeamIndex bi{0}; bi < mBeamWidth; ++bi)
        for (LifeCycleId lcId{0}; lcId < mBasePageIndices[bi].size(); ++lcId)
            if (std::holds_alternative<Span<int>>(mBasePageIndices[bi][lcId]))
                setBasePageIndexBuf(bi, lcId, nullptr, 0);

    // Record event scope — mirrors Python's `with self._record_event()`.
    // SharedPageLock destructors inside the scope use finishEvent() to synchronize.
    {
        auto scope = recordEventScope();
        auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();

        // Convert SharedPageLocks → PageHolders for active (non-stale) pages only.
        // Mirrors Python: for ordinal, beam_idx, lc_idx in self._active_pages()
        for (auto const& ap : _activePages())
        {
            auto& bp = (ap.lcId != ssmLcId) ? mBlocks[ap.ordinal].pages[ap.beamIdx][ap.lcId]
                                            : mSsmBlocks[ap.beamIdx][ap.lcId];
            // expect_type(_SharedPageLock, beam_block[lc_idx]) → std::get raises on wrong type
            auto& lock = std::get<SharedPageLock>(bp);
            auto holder = lock.page()->hold();
            bp = std::move(holder); // ~SharedPageLock calls unlock() → notifyFinish(finishEvent())
        }
        // Free scratch slots inside scope — unlock() needs finishEvent().
        // Mirrors Python: _free_scratch_slots() inside `with self._record_event()`.
        _freeScratchSlots();
    }
    mStatus = Status::SUSPENDED;
    if (_shouldRecordStats())
    {
        mManager->recordRequestSuspended();
    }
}

void KvCache::close()
{
    TLLM_CHECK_DEBUG(_checkSanity());
    if (mStatus == Status::CLOSED)
        return;

    discardPendingStats();
    stopCommitting();
    TLLM_CHECK_DEBUG(_checkSanity());

    // Dummy/warmup caches are reserved at the model's full declared context, not at a realistic
    // sequence length, and mAvgSqrCapacity is an RMS -- so a handful of them dominates the
    // statistic outright and the tuner sizes pools for sequences that never arrive. They are
    // already tracked as stats-excluded at creation; honour that here too.
    if (mCapacity > 0 && !mManager->isStatsExcluded(id))
    {
        mAvgCapacity.update(static_cast<double>(mCapacity));
        mManager->updateAvgSqrCapacity(mAvgCapacity.value() * mAvgCapacity.value());
        mManager->updateAvgSqrHistoryLength(mAvgHistoryLength.value() * mAvgHistoryLength.value());
        mManager->incrementNumSampledKvCaches();
        mManager->tryUpdateTargetRatios();
    }

    // Record event scope — mirrors Python's `with self._record_event()`.
    // Python always enters _record_event() here; _cuda_stream is valid for both ACTIVE and SUSPENDED.
    {
        auto scope = recordEventScope();
        _clearBlocks();
    }
    mStatus = Status::CLOSED;
    mManager->unregisterKvCache(this);
}

KVCacheStatsDelta KvCache::commitPendingStats()
{
    if (!_shouldRecordStats())
    {
        discardPendingStats();
        return {};
    }

    if (_shouldRecordManagerStats())
    {
        mManager->commitStats(mPendingStats.globalStats(), mPendingStats.iterationStatsByLifeCycle());
        mManager->commitSsmSnapshotIterationStats(mPendingStats.ssmSnapshotIterationStatsByLifeCycle());
    }
    KVCacheStatsDelta const requestStats
        = _shouldRecordRequestStats() ? mPendingStats.requestStats().copy() : KVCacheStatsDelta{};
    mPendingStats.clear();
    mManager->clearStatsDirty(id);
    return requestStats;
}

void KvCache::discardPendingStats()
{
    mPendingStats.clear();
    mManager->clearStatsDirty(id);
}

bool KvCache::_shouldRecordStats() const
{
    return _shouldRecordManagerStats() || _shouldRecordRequestStats();
}

bool KvCache::_shouldRecordManagerStats() const
{
    return mManager->config().enableStats && !mManager->isStatsExcluded(id);
}

bool KvCache::_shouldRecordRequestStats() const
{
    // Manager stats historically also produced the request delta returned by
    // commitPendingStats().  Preserve that contract while allowing callers to
    // opt in to request-only accounting through mEnableRequestStats.
    return (mManager->config().enableStats || mEnableRequestStats) && !mManager->isStatsExcluded(id);
}

void KvCache::_refreshStatsDirtyState()
{
    if (!mPendingStats.empty())
    {
        mManager->markStatsDirty(id);
    }
    else
    {
        mManager->clearStatsDirty(id);
    }
}

void KvCache::_recordDirectIterationStats(LifeCycleId lifeCycle, KVCacheIterationStatsDelta const& iterationStats)
{
    // Every lifecycle is reported, including SSM / recurrent ones: iteration
    // statistics are keyed by lifecycle, so recurrent page movement stays
    // distinguishable from attention movement downstream.
    if (!_shouldRecordManagerStats() || iterationStats.empty())
    {
        return;
    }
    IterationStatsByLifeCycle iterationStatsByLifeCycle;
    iterationStatsByLifeCycle.emplace(lifeCycle, iterationStats.copy());
    mManager->commitStats({}, iterationStatsByLifeCycle);
}

void KvCache::_recordMigratedSlots(
    std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots, CacheLevel srcLevel, CacheLevel dstLevel)
{
    if (!_shouldRecordManagerStats())
    {
        return;
    }
    TLLM_CHECK_DEBUG(pages.size() == slots.size());
    for (auto const& page : pages)
    {
        LifeCycleId const lifeCycle = page->lifeCycle;
        bool const isAttention = std::holds_alternative<AttnLifeCycle>(mManager->lifeCycles().getLifeCycle(lifeCycle));

        KVCacheStatsDelta stats;
        KVCacheIterationStatsDelta iterationStats;
        if (srcLevel == kHotLevel && dstLevel > kHotLevel)
        {
            iterationStats.iterOffloadBlocks = 1;
            iterationStats.iterOffloadBytes = sumSlotBytes(mManager->storage(), dstLevel, lifeCycle);
        }
        else if (dstLevel == kHotLevel)
        {
            // Global cache-hit accounting is attention-only. SSM movement is
            // reported by lifecycle/pool-group iteration statistics instead.
            if (isAttention)
            {
                stats.allocTotalBlocks = 1;
                stats.allocNewBlocks = 1;
            }
            iterationStats.iterAllocTotalBlocks = 1;
            iterationStats.iterAllocNewBlocks = 1;
            if (srcLevel > kHotLevel)
            {
                iterationStats.iterOnboardBlocks = 1;
                iterationStats.iterOnboardBytes = sumSlotBytes(mManager->storage(), srcLevel, lifeCycle);
            }
            else if (srcLevel == kHotLevel)
            {
                iterationStats.iterIntraDeviceCopyBlocks = 1;
                iterationStats.iterIntraDeviceCopyBytes = sumSlotBytes(mManager->storage(), kHotLevel, lifeCycle);
            }
        }

        if (!stats.empty() || !iterationStats.empty())
        {
            IterationStatsByLifeCycle iterationStatsByLifeCycle;
            iterationStatsByLifeCycle.emplace(lifeCycle, iterationStats);
            mManager->commitStats(stats, iterationStatsByLifeCycle);
        }
    }
}

void KvCache::_recordDroppedPages(std::vector<SharedPtr<Page>> const& pages, CacheLevel cacheLevel)
{
    if (!_shouldRecordManagerStats())
    {
        return;
    }
    for (auto const& page : pages)
    {
        LifeCycleId const lifeCycle = page->lifeCycle;
        KVCacheIterationStatsDelta iterationStats;
        iterationStats.iterHostDroppedBlocks = 1;
        iterationStats.iterHostDroppedBytes = sumSlotBytes(mManager->storage(), cacheLevel, lifeCycle);
        _recordDirectIterationStats(lifeCycle, iterationStats);
    }
}

void KvCache::_recordResizePendingAllocations(BlockOrdinal blockBegin, BlockOrdinal blockEnd,
    TypedVec<LifeCycleId, HalfOpenRange<BlockOrdinal>> const& excludedRanges, bool countAsGeneration)
{
    bool const recordManagerStats = _shouldRecordManagerStats();
    bool const recordRequestStats = _shouldRecordRequestStats();
    if ((!recordManagerStats && !recordRequestStats) || blockBegin >= blockEnd)
    {
        return;
    }

    bool changed = false;
    for (auto const& [lifeCycle, unused] : mManager->lifeCycles().attentionLifeCycles())
    {
        (void) unused;
        auto const& excluded = excludedRanges[lifeCycle];
        BlockOrdinal const firstEnd = std::min(blockEnd, excluded.beg);
        if (blockBegin < firstEnd)
        {
            changed |= mPendingStats.recordAllocationRange(lifeCycle, blockBegin, firstEnd, mBeamWidth.value(),
                !countAsGeneration, countAsGeneration, recordManagerStats, recordRequestStats);
        }
        BlockOrdinal const secondBegin = std::max(blockBegin, excluded.end);
        if (secondBegin < blockEnd)
        {
            changed |= mPendingStats.recordAllocationRange(lifeCycle, secondBegin, blockEnd, mBeamWidth.value(),
                !countAsGeneration, countAsGeneration, recordManagerStats, recordRequestStats);
        }
    }
    if (changed)
    {
        mManager->markStatsDirty(id);
    }
}

void KvCache::_subtractPendingAllocationRange(BlockOrdinal blockBegin, BlockOrdinal blockEnd)
{
    if (mPendingStats.subtractAllocationRange(blockBegin, blockEnd))
    {
        _refreshStatsDirtyState();
    }
}

bool KvCache::_hasReuseSource(BlockPage const& page)
{
    // `block` is set only while the page occupies its block's slot, so this asks whether
    // the tree still offers the page.
    auto const committedPage = dynamicPointerCast<CommittedPage>(blockPageGetPage(page));
    return committedPage && committedPage->block != nullptr;
}

// ---------------------------------------------------------------------------
// _clearBlocks
// ---------------------------------------------------------------------------

void KvCache::_clearBlocks()
{
    // Drop last block first (mirrors Python: while self._blocks: self._blocks.pop()).
    while (!mBlocks.empty())
        mBlocks.pop_back();
    _freeScratchSlots();
    // Clear SSM blocks.
    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    if (ssmLcId.has_value())
    {
        for (auto& beamBlock : mSsmBlocks)
            beamBlock[*ssmLcId] = std::monostate{};
    }
}

// ---------------------------------------------------------------------------
// _copyPageToTreeBlock: copy a page into a new committed (or SSM committed) page
// attached to a radix tree block. Mirrors Python's _copy_page_to_tree_block.
// ---------------------------------------------------------------------------

CommittedPage* KvCache::_copyPageToTreeBlock(
    SharedPtr<Block> const& treeBlock, LifeCycleId lcIdx, SharedPtr<Page> const& srcPage, int numTokensInBlock)
{
    if (!treeBlock->canReplacePage(lcIdx, numTokensInBlock))
    {
        return treeBlock->getPage(lcIdx);
    }

    auto& storageMgr = mManager->storage();

    for (CacheLevel lvl = srcPage->cacheLevel; lvl < storageMgr.numCacheLevels(); ++lvl)
    {
        PoolGroupIndex const pgIdx = storageMgr.getPoolGroupIndex(lvl, lcIdx);
        Slot newSlot;
        try
        {
            auto slots = storageMgr.newSlotsForPoolGroup(lvl, pgIdx, 1);
            newSlot = std::move(slots[0]);
        }
        catch (OutOfPagesError const&)
        {
            continue;
        }

        CUstream stream = cudaStream();
        newSlot.readyEvent.waitInStream(reinterpret_cast<CudaStream>(stream));
        storageMgr.copySlotData(lcIdx, lvl, srcPage->cacheLevel, newSlot.slotId(), srcPage->slotId(), stream);

        newSlot.readyEvent = CachedCudaEvent(reinterpret_cast<CudaStream>(stream));
        auto committed = makeShared<CommittedPage>(
            &storageMgr, treeBlock, lcIdx, lvl, numTokensInBlock, getPriority(treeBlock->ordinal(), lcIdx));
        committed->setSlot(newSlot);
        // Drops the superseded page, deferred until the copy is issued: an
        // OutOfPagesError above must not destroy a usable shorter snapshot.
        treeBlock->replacePage(lcIdx, committed.get());

        // Schedule for eviction so eviction controller keeps a strong reference,
        // preventing the page from being destroyed.
        storageMgr.scheduleForEviction(*committed);
        return committed.get(); // success
    }
    // No pages available in any level, silently skip snapshot (matches Python).
    return nullptr;
}

// ---------------------------------------------------------------------------
// _snapshotSsmToTreeBlock: snapshot live SSM state to a radix tree block for the
// given committed token count. Mirrors Python's _snapshot_ssm_to_tree_block.
// ---------------------------------------------------------------------------

void KvCache::_snapshotSsmToTreeBlock(SharedPtr<Block> const& treeBlock, LifeCycleId ssmLcId, int numTokens, bool move)
{
    int const numTokensInBlock = numTokens - treeBlock->ordinal().value() * mTokensPerBlock;
    TLLM_CHECK_DEBUG(0 < numTokensInBlock && numTokensInBlock <= mTokensPerBlock);

    if (treeBlock->pageCoverage(ssmLcId) >= numTokensInBlock)
    {
        return;
    }

    auto& ssmBlock = mSsmBlocks[kDefaultBeamIndex];
    auto* ssmLock = std::get_if<SharedPageLock>(&ssmBlock[ssmLcId]);
    TLLM_CHECK_DEBUG(ssmLock && ssmLock->isValid());
    auto srcPage = ssmLock->page();
    if (move)
    {
        auto unlocked = ssmLock->unlock();
        auto up = dynamicPointerCast<UncommittedPage>(unlocked);
        TLLM_CHECK_DEBUG(up != nullptr);
        ssmBlock[ssmLcId] = std::monostate{};
        // convertToCommitted() installs via replacePage(), dropping any shorter snapshot.
        auto committed = up->convertToCommitted(treeBlock, finishEvent(), numTokensInBlock);
        mManager->storage().scheduleForEviction(*committed);
        return;
    }

    _copyPageToTreeBlock(treeBlock, ssmLcId, srcPage, numTokensInBlock);
}

void KvCache::_reattachOrphanTreeBlocks(BlockOrdinal lastOrdinal, RootBlock& root)
{
    if (lastOrdinal < 0)
    {
        return;
    }

    // Every block at or below `lastOrdinal` is committed, so `treeBlock` is non-null:
    // the walk below relies on that, since a null treeBlock would stop it early and then
    // be dereferenced as `prevNode`.
    auto isDetached = [this](BlockOrdinal ord)
    {
        auto const& sb = mBlocks[ord];
        TLLM_CHECK_DEBUG_WITH_INFO(sb.treeBlock, "committed block must have a tree block");
        return sb.treeBlock->isOrphan();
    };

    // Fast path: the block we are about to use as `prev` is still attached.
    if (!isDetached(lastOrdinal))
    {
        return;
    }

    // Walk back to the deepest ancestor that is still in the tree (or the root).
    BlockOrdinal first = lastOrdinal;
    while (first >= 0 && isDetached(first))
    {
        --first;
    }

    NodeBase* prevNode
        = (first < 0) ? static_cast<NodeBase*>(&root) : static_cast<NodeBase*>(mBlocks[first].treeBlock.get());

    for (BlockOrdinal ord = first + 1; ord <= lastOrdinal; ++ord)
    {
        auto& sb = mBlocks[ord];

        // detachNext() only cleared `prev` and the parent's map entry, so ordinal, tokens
        // and surviving pages are intact -- re-attaching is enough, nothing to rebuild.
        bool attached = false;
        SharedPtr<Block> inTree = attachOrGetExistingBlock(prevNode, sb.treeBlock, &attached);
        TLLM_CHECK_DEBUG(inTree);

        if (attached && inTree->eventSink)
        {
            // Balances the addRemovedBlock() from detachNext(). If some other block was
            // attached instead, it was already announced when it was created.
            inTree->eventSink->addStoredBlock(*inTree);
        }

        sb.treeBlock = inTree;
        prevNode = inTree.get();
    }
}

// ---------------------------------------------------------------------------
// _snapshotPartialBlockToTree: snapshot a partial final block into the radix
// tree. Mirrors Python's _snapshot_partial_block_to_tree.
// ---------------------------------------------------------------------------

void KvCache::_snapshotPartialBlockToTree(BlockOrdinal ordinal, bool commitSsm)
{
    int const start = ordinal.value() * mTokensPerBlock;
    int const end = std::min(start + mTokensPerBlock, static_cast<int>(mCommittedTokens.size()));
    std::vector<TokenIdExt> tokens(mCommittedTokens.begin() + start, mCommittedTokens.begin() + end);
    int const numTokens = static_cast<int>(tokens.size());
    TLLM_CHECK_DEBUG(0 < numTokens && numTokens < mTokensPerBlock);

    NodeBase* prevNode = nullptr;
    RootBlock& root = mManager->radixTree().addOrGetExisting(mReuseScope);
    if (ordinal == BlockOrdinal{0})
    {
        prevNode = &root;
    }
    else
    {
        _reattachOrphanTreeBlocks(ordinal - 1, root);
        prevNode = _getTreeBlock(ordinal - 1).get();
    }

    bool isNew = false;
    SharedPtr<Block> treeBlock = addOrGetExistingBlock(prevNode, tokens, textOnly(), &isNew);
    TLLM_CHECK_DEBUG(treeBlock);
    // When not new we either matched exactly or were given a longer sibling that covers
    // these tokens, so our tokens are a prefix of the block we got back either way.
    TLLM_CHECK_DEBUG(isNew
        || (treeBlock->tokens.size() >= tokens.size()
            && std::equal(tokens.begin(), tokens.end(), treeBlock->tokens.begin())));

    auto& beamBlock = mBlocks.at(ordinal).pages[kDefaultBeamIndex];
    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    // `treeBlock` may be a longer existing sibling that covers these tokens. Attaching the
    // partial attention pages to it is still correct because each page records the token
    // span it covers, and prefix matching honours that span.
    std::vector<LifeCycleId> attachedLcs;
    for (auto const& [lcIdx, attn] : mManager->lifeCycles().attentionLifeCycles())
    {
        (void) attn;
        auto& bp = beamBlock[lcIdx];
        if (blockPageIsNull(bp) || treeBlock->pageCoverage(lcIdx) >= numTokens)
        {
            continue;
        }
        if (_copyPageToTreeBlock(treeBlock, lcIdx, blockPageGetPage(bp), numTokens) != nullptr)
        {
            attachedLcs.push_back(lcIdx);
        }
    }
    if (commitSsm)
    {
        TLLM_CHECK_DEBUG(ssmLcId.has_value());
        _snapshotSsmToTreeBlock(treeBlock, *ssmLcId, start + numTokens);
    }
    if (treeBlock->eventSink)
    {
        if (isNew)
        {
            treeBlock->eventSink->addStoredBlock(*treeBlock);
        }
        else
        {
            // The block was already announced, so report just the life cycles this snapshot
            // added. The event manager itself drops the ones whose page does not span the
            // whole block: the payload carries the block's full token list and cannot
            // express a shorter valid prefix.
            for (auto lcIdx : attachedLcs)
            {
                treeBlock->eventSink->addStoredLifeCycle(*treeBlock, lcIdx);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// resize
// ---------------------------------------------------------------------------

bool KvCache::resize(std::optional<int> capacity, std::optional<int> historyLength)
{
    TLLM_CHECK_DEBUG(mStatus == Status::ACTIVE);
    TLLM_CHECK_DEBUG(mBlocks.size() == BlockOrdinal{divUp(mCapacity, mTokensPerBlock)});

    int newCap = capacity.value_or(mCapacity);
    int newHist = historyLength.value_or(mHistoryLength);

    if (capacity.has_value())
        mAvgCapacity.update(static_cast<double>(newCap));
    if (historyLength.has_value())
        mAvgHistoryLength.update(static_cast<double>(newHist));

    if (newHist < mHistoryLength)
        throw std::invalid_argument("History length cannot be decreased");
    if (newCap < newHist)
        throw std::invalid_argument("History length cannot exceed capacity");

    // Scratch reuse: enforce constraint.
    bool enableScratch = mEnableSwaScratchReuse;
    if (TLLM_UNLIKELY(gDebug) && enableScratch && newCap != mCapacity)
    {
        int const maxRewindLen = _swaScratchMaxRewindLen();
        int const minHistoryLength = std::max(0, mCapacity - maxRewindLen);
        bool const validSwaScratchHistory = minHistoryLength <= newHist && newHist <= mCapacity;
        TLLM_CHECK_WITH_INFO(validSwaScratchHistory,
            "SWA scratch requires old_capacity - max_rewind_len <= history_length <= old_capacity");
        (void) validSwaScratchHistory;
    }

    bool const recordGenerationAllocStats = mGenerationAllocReady && newCap > mCapacity;
    if (!enableScratch && _shortcutSetCapacity(newCap) && _shortcutSetHistoryLength(newHist))
    {
        _refreshGenerationAllocReady();
        return true;
    }

    BlockOrdinal oldNumBlocks{divUp(mCapacity, mTokensPerBlock)};
    BlockOrdinal newNumBlocks{divUp(newCap, mTokensPerBlock)};
    LifeCycleId numLc = mManager->storage().numLifeCycles();
    auto const& lcs = mManager->lifeCycles();

    _checkPageIndexBufferCapacity(newNumBlocks);

    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    auto backupHolders = _unlockStaleBlocks(newHist);

    if (newNumBlocks < oldNumBlocks)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(!hasScratchSlots(), "Cannot shrink while scratch slots exist");
        _subtractPendingAllocationRange(newNumBlocks, oldNumBlocks);
        auto scope = recordEventScope();
        _decreaseCapacity(newNumBlocks);
    }

    // Compute scratch deltas.
    auto [excessScratchSlots, deltaScratchSlots, scratchRanges] = _takeExcessScratchSlots(newCap, newHist);

    if (newNumBlocks >= oldNumBlocks)
    {
        // Compute new normal slots needed per lifecycle.
        TypedVec<LifeCycleId, SlotCount> numNewSlots(numLc, 0);
        TypedVec<LifeCycleId, HalfOpenRange<BlockOrdinal>> staleRanges(numLc);
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            if (ssmLcId.has_value() && lc == *ssmLcId)
                continue;
            staleRanges[lc] = _getStaleRange(newHist, lcs.getLifeCycle(lc));

            if (enableScratch)
            {
                HalfOpenRange<BlockOrdinal> const newBlockRange{oldNumBlocks, newNumBlocks};
                int const numNewBlocksUsingScratch = intersect(scratchRanges[lc], newBlockRange).length();
                int const numNewNormalBlocks = newBlockRange.length() - numNewBlocksUsingScratch;
                numNewSlots[lc] = static_cast<SlotCount>(numNewNormalBlocks) * mBeamWidth.value();
            }
            else
            {
                auto [staleBeg, staleEnd] = staleRanges[lc];
                int numNewBlocksToAdd;
                if (oldNumBlocks < staleBeg)
                {
                    TLLM_CHECK_DEBUG(newNumBlocks >= staleEnd);
                    numNewBlocksToAdd = (staleBeg - oldNumBlocks) + (newNumBlocks - staleEnd);
                }
                else
                {
                    numNewBlocksToAdd = newNumBlocks - std::max(staleEnd, oldNumBlocks);
                }
                numNewSlots[lc] = static_cast<SlotCount>(numNewBlocksToAdd) * mBeamWidth.value();
            }
        }

        // Compute net allocation counts (normal + scratch delta).
        TypedVec<LifeCycleId, SlotCount> netAllocCounts(numLc, 0);
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
            netAllocCounts[lc] = numNewSlots[lc] + static_cast<SlotCount>(deltaScratchSlots[lc]);

        // Allocate new slots.
        TypedVec<LifeCycleId, std::vector<Slot>> newSlots;
        bool anyPositive = std::any_of(netAllocCounts.begin(), netAllocCounts.end(), [](SlotCount c) { return c > 0; });
        if (anyPositive)
        {
            TypedVec<LifeCycleId, SlotCount> allocCounts(numLc);
            for (LifeCycleId lc{0}; lc < numLc; ++lc)
                allocCounts[lc] = std::max(SlotCount{0}, netAllocCounts[lc]);
            try
            {
                MigrationRecorder const migrationRecorder
                    = [this](std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots,
                          CacheLevel srcLevel, CacheLevel dstLevel)
                { _recordMigratedSlots(pages, slots, srcLevel, dstLevel); };
                DropRecorder const dropRecorder
                    = [this](std::vector<SharedPtr<Page>> const& pages, CacheLevel cacheLevel)
                { _recordDroppedPages(pages, cacheLevel); };
                newSlots = mManager->storage().newGpuSlots(allocCounts, migrationRecorder, dropRecorder);
            }
            catch (OutOfPagesError const&)
            {
                _recoverExcessScratchSlots(excessScratchSlots);
                _lockHeldBlocks(backupHolders);
                return false;
            }
        }
        else
        {
            newSlots.resize(numLc);
        }

        // Wait on newly allocated slots.
        {
            std::vector<CachedCudaEvent const*> readyEvents;
            for (auto const& lcSlots : newSlots)
                for (auto const& slot : lcSlots)
                    readyEvents.push_back(&slot.readyEvent);
            if (!readyEvents.empty())
                streamWaitEvents(reinterpret_cast<CudaStream>(cudaStream()), readyEvents);
        }

        // Combine: new slots + excess scratch detached slots.
        TypedVec<LifeCycleId, std::vector<Slot>> slots(numLc);
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            auto& combined = slots[lc];
            combined = std::move(newSlots[lc]);
            for (auto& lock : excessScratchSlots[lc])
                combined.push_back(lock.detachSlot());
            excessScratchSlots[lc].clear();
        }

        // Release excess if net is negative.
        if (std::any_of(netAllocCounts.begin(), netAllocCounts.end(), [](SlotCount c) { return c < 0; }))
        {
            auto scope = recordEventScope();
            for (LifeCycleId lc{0}; lc < numLc; ++lc)
            {
                for (SlotCount i = 0; i < -netAllocCounts[lc]; ++i)
                {
                    auto slot = std::move(slots[lc].back());
                    slots[lc].pop_back();
                    slot.readyEvent = finishEvent();
                    mManager->storage().releaseSlot(lc, kHotLevel, std::move(slot));
                }
            }
        }

        // Assert correct combined slot count.
        if (TLLM_UNLIKELY(gDebug))
        {
            for (LifeCycleId lc{0}; lc < numLc; ++lc)
            {
                SlotCount const expected
                    = numNewSlots[lc] + std::max(SlotCount{0}, static_cast<SlotCount>(deltaScratchSlots[lc]));
                TLLM_CHECK(static_cast<SlotCount>(slots[lc].size()) == expected);
            }
        }

        // Fulfill additional scratch slots (pop from end of combined slots).
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            for (int i = 0; i < deltaScratchSlots[lc]; ++i)
            {
                auto slot = std::move(slots[lc].back());
                slots[lc].pop_back();
                mScratchSlots[lc].emplace_back(std::move(slot), *this, lc, /*skipWait=*/true);
            }
        }

        // Wait for all slots that will back new pages. This includes detached excess scratch slots,
        // which are not covered by the earlier wait on newly allocated slots.
        {
            std::vector<CachedCudaEvent const*> readyEvents;
            for (auto const& lcSlots : slots)
                for (auto const& slot : lcSlots)
                    readyEvents.push_back(&slot.readyEvent);
            if (!readyEvents.empty())
                streamWaitEvents(reinterpret_cast<CudaStream>(cudaStream()), readyEvents);
        }

        // Scratch and stale blocks do not consume per-request KV pages, so exclude them from allocation stats.
        auto const& excludedRanges = enableScratch ? scratchRanges : staleRanges;
        _recordResizePendingAllocations(oldNumBlocks, newNumBlocks, excludedRanges, recordGenerationAllocStats);

        // Resize page index buffers.
        TLLM_CHECK_DEBUG(std::all_of(mBasePageIndices.begin(), mBasePageIndices.end(),
            [oldNumBlocks](auto const& beamIndices)
            {
                return std::all_of(beamIndices.begin(), beamIndices.end(),
                    [oldNumBlocks](auto const& buf)
                    {
                        auto const* vec = std::get_if<std::vector<int>>(&buf);
                        return !vec || vec->size() == toSizeT(oldNumBlocks);
                    });
            }));

        // Create SeqBlocks for new ordinals (pop slots from end, matching Python).
        _resizePageIndexBuffers(newNumBlocks);
        for (BlockOrdinal ord = oldNumBlocks; ord < newNumBlocks; ++ord)
        {
            SeqBlock sb;
            sb.pages.resize(mBeamWidth);
            for (auto& row : sb.pages)
                row.resize(numLc);
            for (BeamIndex bi{0}; bi < mBeamWidth; ++bi)
            {
                for (LifeCycleId lc{0}; lc < numLc; ++lc)
                {
                    if (ssmLcId.has_value() && lc == *ssmLcId)
                        continue; // SSM pages live in mSsmBlocks, not in mBlocks
                    if (enableScratch)
                    {
                        if (scratchRanges[lc].contains(ord))
                            continue; // Scratch block — no per-block page allocation.
                    }
                    else
                    {
                        auto [staleBeg, staleEnd] = staleRanges[lc];
                        if (staleBeg <= ord && ord < staleEnd)
                            continue;
                    }
                    auto slot = std::move(slots[lc].back());
                    slots[lc].pop_back();
                    auto page = makeShared<UncommittedPage>(*this, ord, lc, kHotLevel, bi);
                    page->setSlot(slot);
                    sb.pages[bi][lc] = page->lock(*this, bi, ord, lc, /*skipWait=*/true);
                }
            }
            mBlocks.push_back(std::move(sb));
        }
        TLLM_CHECK_DEBUG(std::all_of(slots.begin(), slots.end(), [](auto const& vec) { return vec.empty(); }));
    }

    mCapacity = newCap;
    mHistoryLength = newHist;
    _refreshGenerationAllocReady();
    _evictOutOfWindowBlocks(newHist);
    TLLM_CHECK_DEBUG(_checkSanity());
    return true;
}

void KvCache::setCapacity(int cap)
{
    if (mEnableSwaScratchReuse)
    {
        throw std::invalid_argument(
            "Cannot use capacity setter when SWA scratch reuse is enabled. "
            "Use resize(capacity, history_length) instead.");
    }
    if (!resize(cap, std::nullopt))
        throw OutOfPagesError("Not enough pages in GPU memory");
}

void KvCache::setHistoryLength(int hist)
{
    bool success = resize(std::nullopt, hist);
    TLLM_CHECK(success);
    (void) success;
}

bool KvCache::_shortcutSetCapacity(int newCap)
{
    if (newCap == mCapacity)
        return true;
    // No shortcut if block count changes.
    if (divUp(newCap, mTokensPerBlock) != divUp(mCapacity, mTokensPerBlock))
        return false;
    mCapacity = newCap;
    return true;
}

bool KvCache::_shortcutSetHistoryLength(int newHist)
{
    if (newHist == mHistoryLength)
        return true;
    // Check if stale range changes for any lifecycle.
    for (auto [lcId, lc] : mManager->lifeCycles())
    {
        bool changed = std::visit(
            [&](auto const& v) -> bool
            {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, SsmLifeCycle>)
                {
                    // history_length change does not impact blocks at all.
                    return false;
                }
                else
                {
                    static_assert(std::is_same_v<T, AttnLifeCycle>);
                    if (!v.windowSize.has_value())
                        return false;
                    return v.getStaleRange(newHist, mTokensPerBlock)
                        != v.getStaleRange(mHistoryLength, mTokensPerBlock);
                }
            },
            lc);
        if (changed)
            return false;
    }
    mHistoryLength = newHist;
    return true;
}

void KvCache::_refreshGenerationAllocReady()
{
    if (mExpectedPromptLength.has_value() && mHistoryLength >= *mExpectedPromptLength)
    {
        mGenerationAllocReady = true;
    }
}

// ---------------------------------------------------------------------------
// Capacity management
// ---------------------------------------------------------------------------

void KvCache::_increaseCapacity(BlockOrdinal newNumBlocks, int newHistoryLength)
{
    BlockOrdinal curNumBlocks = mBlocks.size();
    LifeCycleId numLc = mManager->storage().numLifeCycles();
    auto const& lcs = mManager->lifeCycles();
    auto ssmLcId = lcs.ssmLifeCycleId();

    // Compute stale ranges using new history length so stale SWA blocks get no pages.
    TypedVec<LifeCycleId, HalfOpenRange<BlockOrdinal>> staleRanges(numLc);
    TypedVec<LifeCycleId, SlotCount> numSlotsPerLc(numLc, 0);
    for (LifeCycleId lc{0}; lc < numLc; ++lc)
    {
        // SSM slots are handled separately below — don't allocate block-level slots for SSM.
        if (ssmLcId.has_value() && lc == *ssmLcId)
            continue;
        staleRanges[lc] = _getStaleRange(newHistoryLength, lcs.getLifeCycle(lc));
        auto [staleBeg, staleEnd] = staleRanges[lc];
        int numNewBlocks;
        if (curNumBlocks < staleBeg)
        {
            TLLM_CHECK_DEBUG(newNumBlocks >= staleEnd);
            numNewBlocks = (staleBeg - curNumBlocks) + (newNumBlocks - staleEnd);
        }
        else
        {
            numNewBlocks = newNumBlocks - std::max(staleEnd, curNumBlocks);
        }
        numSlotsPerLc[lc] = static_cast<SlotCount>(numNewBlocks) * mBeamWidth.value();
    }

    // SSM slots are now allocated lazily in resume() via deferred copy, not here.

    MigrationRecorder const migrationRecorder
        = [this](std::vector<SharedPtr<Page>> const& pages, std::vector<Slot> const& slots, CacheLevel srcLevel,
              CacheLevel dstLevel) { _recordMigratedSlots(pages, slots, srcLevel, dstLevel); };
    DropRecorder const dropRecorder = [this](std::vector<SharedPtr<Page>> const& pages, CacheLevel cacheLevel)
    { _recordDroppedPages(pages, cacheLevel); };
    auto allSlots = mManager->storage().newGpuSlots(numSlotsPerLc, migrationRecorder, dropRecorder);

    // Assert that internal index buffer sizes match expected old_num_blocks (mirrors Python line ~463).
    TLLM_CHECK_DEBUG(std::all_of(mBasePageIndices.begin(), mBasePageIndices.end(),
        [curNumBlocks](auto const& beamIndices)
        {
            return std::all_of(beamIndices.begin(), beamIndices.end(),
                [curNumBlocks](auto const& buf)
                {
                    auto const* vec = std::get_if<std::vector<int>>(&buf);
                    return !vec || vec->size() == toSizeT(curNumBlocks);
                });
        }));

    // Create SeqBlocks for the new ordinals.
    TypedVec<LifeCycleId, size_t> slotCounters(numLc, 0);
    _resizePageIndexBuffers(newNumBlocks);
    for (BlockOrdinal ord = curNumBlocks; ord < newNumBlocks; ++ord)
    {
        SeqBlock sb;
        sb.pages.resize(mBeamWidth);
        for (auto& row : sb.pages)
            row.resize(numLc); // default-constructs to monostate
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            // SSM pages live in mSsmBlocks, not in _blocks.
            if (ssmLcId.has_value() && lc == *ssmLcId)
                continue;

            auto [staleBeg, staleEnd] = staleRanges[lc];
            if (staleBeg <= ord && ord < staleEnd)
                continue; // stale block for this lc — no page allocated

            size_t si = slotCounters[lc]++;
            auto& slot = allSlots[lc][si];
            auto page = makeShared<UncommittedPage>(*this, ord, lc, kHotLevel, kDefaultBeamIndex);
            page->setSlot(slot);
            sb.pages[kDefaultBeamIndex][lc] = page->lock(*this, kDefaultBeamIndex, ord, lc);
        }
        mBlocks.push_back(std::move(sb));
    }
    // Assert all allocated slots were consumed (mirrors Python line ~488).
    if (TLLM_UNLIKELY(gDebug))
    {
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
            TLLM_CHECK(slotCounters[lc] == allSlots[lc].size());
    }
}

void KvCache::_decreaseCapacity(BlockOrdinal newNumBlocks)
{
    while (mBlocks.size() > newNumBlocks)
    {
        auto& sb = mBlocks.back();
        for (auto& beamPages : sb.pages)
            for (auto& bp : beamPages)
                bp = std::monostate{};
        sb.treeBlock.reset();
        mBlocks.pop_back();
    }
    _resizePageIndexBuffers(newNumBlocks);
}

HalfOpenRange<BlockOrdinal> KvCache::_getStaleRange(int historyLength, LifeCycle const& lc) const
{
    return kv_cache_manager_v2::getStaleRange(lc, historyLength, mTokensPerBlock);
}

std::vector<KvCache::StaleBackup> KvCache::_unlockStaleBlocks(int newHistoryLength)
{
    std::vector<StaleBackup> ret;
    if (newHistoryLength == mHistoryLength)
        return ret;

    auto scope = recordEventScope();
    auto const& lcs = mManager->lifeCycles();
    LifeCycleId numLc = mManager->storage().numLifeCycles();

    for (LifeCycleId lcIdx{0}; lcIdx < numLc; ++lcIdx)
    {
        LifeCycle const& lc = lcs.getLifeCycle(lcIdx);
        // SSM pages live in mSsmBlocks, not _blocks — skip.
        if (std::holds_alternative<SsmLifeCycle>(lc))
            continue;
        // Full-attention (no SWA) has empty stale range — skip.
        auto const& alc = std::get<AttnLifeCycle>(lc);
        if (!alc.windowSize.has_value())
            continue;

        auto oldRange = _getStaleRange(mHistoryLength, lc);
        auto newRange = _getStaleRange(newHistoryLength, lc);

        BlockOrdinal unlockStart = std::max(oldRange.end, newRange.beg);
        BlockOrdinal unlockEnd = std::min(mBlocks.size(), newRange.end);

        for (BlockOrdinal ord = unlockStart; ord < unlockEnd; ++ord)
        {
            auto& sb = mBlocks[ord];
            bool isCommitted = sb.isCommitted();
            bool holdForCommit
                = !mManager->commitMinSnapshot() && !isCommitted && (mCommitState == CommitState::ALLOWED);

            for (BeamIndex bi{0}; bi < sb.pages.size(); ++bi)
            {
                auto& bp = sb.pages[bi][lcIdx];
                if (blockPageIsNull(bp))
                {
                    // No page to unlock: scratch block, commit_min_snapshot early
                    // release, or a stale block created unallocated by resize().
                    continue;
                }
                TLLM_CHECK_DEBUG(std::holds_alternative<SharedPageLock>(bp));
                auto holder = blockPageGetPage(bp)->hold();
                ret.push_back({ord, bi, lcIdx, holder});
                bp = holdForCommit ? BlockPage{std::move(holder)} : BlockPage{std::monostate{}};
            }
        }
    }
    return ret;
}

void KvCache::_lockHeldBlocks(std::vector<StaleBackup> const& backup)
{
    std::vector<BatchedLockTarget> targets;
    targets.reserve(backup.size());
    for (auto const& b : backup)
        targets.push_back({b.holder->page, b.beamIdx, b.ordinal, b.lcId});

    auto locks = batchedLockToGpu(*this, targets);
    for (size_t i = 0; i < locks.size(); ++i)
    {
        auto const& t = backup[i];
        mBlocks[t.ordinal].pages[t.beamIdx][t.lcId] = std::move(locks[i]);
    }
}

// ---------------------------------------------------------------------------
// _takeUncommittedPage — extract uncommitted pages from a SeqBlock.
// Mirrors Python's _take_uncommitted_page().
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, KvCache::TakenPage> KvCache::_takeUncommittedPage(
    SeqBlock& sb, BeamIndex beamIdx, std::optional<LifeCycleId> skipLc)
{
    LifeCycleId numLc = mManager->storage().numLifeCycles();
    TypedVec<LifeCycleId, TakenPage> result(numLc, TakenPage{nullptr, false});
    for (LifeCycleId lc{0}; lc < numLc; ++lc)
    {
        if (skipLc.has_value() && lc == *skipLc)
            continue;
        auto& bp = sb.pages[beamIdx][lc];
        if (auto* lock = std::get_if<SharedPageLock>(&bp))
        {
            auto up = dynamicPointerCast<UncommittedPage>(lock->page());
            TLLM_CHECK_DEBUG_WITH_INFO(up, "page must be UncommittedPage");
            result[lc] = {up, true};
        }
        else if (auto* holder = std::get_if<SharedPtr<PageHolder>>(&bp))
        {
            TLLM_CHECK_DEBUG(*holder);
            auto up = dynamicPointerCast<UncommittedPage>((*holder)->page);
            TLLM_CHECK_DEBUG_WITH_INFO(up, "page must be UncommittedPage");
            result[lc] = {up, false};
        }
        bp = std::monostate{};
    }
    return result;
}

// ---------------------------------------------------------------------------
// _commitBlock — shared logic for committing a single block.
// Mirrors Python's _commit_block(ordinal, is_last).
// Caller must have recordEventScope() open so finishEvent() works.
// On VIRTUAL_STOP or when isLast is true, transitions to USER_STOP and
// calls _onStopCommitting() — callers do not need post-call cleanup.
// ---------------------------------------------------------------------------

void KvCache::_commitBlock(int ord, bool isLast, bool commitSsm, bool moveSsm)
{
    TLLM_CHECK_DEBUG(mCommitState == CommitState::ALLOWED);
    TLLM_CHECK_DEBUG(ord == mNumCommittedBlocks);

    auto& sb = mBlocks.at(BlockOrdinal{ord});
    TLLM_CHECK_DEBUG_WITH_INFO(sb.pages.size() == BeamIndex{1}, "Must have 1 beam only");

    // Build token block — always slice up to tokens_per_block; is_full tells us
    // whether we got a full block's worth.  Mirrors Python's:
    //   tokens = self._committed_tokens[start : start + tokens_per_block]
    //   is_full = len(tokens) == tokens_per_block
    int start = ord * mTokensPerBlock;
    int end = std::min(start + mTokensPerBlock, static_cast<int>(mCommittedTokens.size()));
    std::vector<TokenIdExt> tokenBlock(mCommittedTokens.begin() + start, mCommittedTokens.begin() + end);
    int const numTokens = static_cast<int>(tokenBlock.size());
    bool const isFull = (numTokens == mTokensPerBlock);

    if (!isLast && !isFull)
        throw LogicError("Cannot commit block that is not full except last block");

    // Prev node lookup (root or previous committed block).
    RootBlock& root = mManager->radixTree().addOrGetExisting(mReuseScope);
    LifeCycleId numLc = mManager->storage().numLifeCycles();

    NodeBase* prevNode = &root;
    if (ord > 0)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(mBlocks[BlockOrdinal{ord - 1}].treeBlock, "prev block must be committed");
        _reattachOrphanTreeBlocks(BlockOrdinal{ord - 1}, root);
        prevNode = mBlocks[BlockOrdinal{ord - 1}].treeBlock.get();
    }

    // Find or create a block in the radix tree. A non-new result is either an exact match
    // or a longer sibling that covers these tokens; both are usable here.
    bool blockIsNew = false;
    SharedPtr<Block> newBlock = addOrGetExistingBlock(prevNode, tokenBlock, textOnly(), &blockIsNew);
    TLLM_CHECK_DEBUG(newBlock);
    TLLM_CHECK_DEBUG(newBlock->tokensPerBlock() == mTokensPerBlock);
    // In reuse case, verify token match (mirrors Python: tree_block.tokens[:num_tokens] == tokens).
    TLLM_CHECK_DEBUG(blockIsNew
        || (newBlock->tokens.size() >= tokenBlock.size()
            && std::equal(tokenBlock.begin(), tokenBlock.end(), newBlock->tokens.begin())));

    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    bool didCommit = false;

    if (blockIsNew)
    {
        // New block: take uncommitted pages, convert to committed.
        // Mirrors Python's _take_uncommitted_page + convert path.
        auto taken = _takeUncommittedPage(sb, kDefaultBeamIndex, ssmLcId);
        sb.treeBlock = newBlock;
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            auto& [up, locked] = taken[lc];
            if (!up)
                continue;
            auto committed = up->convertToCommitted(newBlock, finishEvent(), numTokens);
            if (locked)
                sb.pages[kDefaultBeamIndex][lc]
                    = committed->lock(*this, kDefaultBeamIndex, static_cast<BlockOrdinal>(ord), lc);
            else
                sb.pages[kDefaultBeamIndex][lc] = committed->hold();
        }
        TLLM_CHECK_DEBUG(_getTreeBlock(static_cast<BlockOrdinal>(ord)) == newBlock);
        ++mNumCommittedBlocks;
        if (newBlock->eventSink)
        {
            newBlock->eventSink->addStoredBlock(*newBlock);
        }
        didCommit = true;
    }
    else if (newBlock->isFull() && mManager->allowSeqRebasing() && isFull)
    {
        // Existing block: rebase — reuse existing block's committed pages.
        // Mirrors Python's `elif tree_block.is_full and allow_seq_rebasing and is_full` path.
        std::vector<BatchedLockTarget> reuseTasks;
        for (LifeCycleId lc{0}; lc < numLc; ++lc)
        {
            if (ssmLcId.has_value() && lc == *ssmLcId)
                continue;
            auto& bp = sb.pages[kDefaultBeamIndex][lc];
            if (blockPageIsNull(bp))
                continue;
            // A page covering fewer tokens than this block spans (moved in from a shorter
            // sibling) is NOT a substitute for our own full page: adopting it would feed
            // uninitialized KV for the uncovered tail into a live request.
            auto* existingPage = newBlock->getPage(lc);
            if (existingPage != nullptr && existingPage->numTokensInBlock < numTokens)
            {
                existingPage = nullptr;
            }
            bool isLocked = std::holds_alternative<SharedPageLock>(bp);
            if (existingPage == nullptr)
            {
                // Existing page gone — put our uncommitted page into the tree block.
                if (auto* lock = std::get_if<SharedPageLock>(&bp))
                {
                    auto up = dynamicPointerCast<UncommittedPage>(lock->page());
                    if (up)
                    {
                        bp = std::monostate{};
                        auto committed = up->convertToCommitted(newBlock, finishEvent(), numTokens);
                        if (newBlock->eventSink)
                        {
                            newBlock->eventSink->addStoredLifeCycle(*newBlock, lc);
                        }
                        bp = isLocked
                            ? BlockPage{committed->lock(*this, kDefaultBeamIndex, static_cast<BlockOrdinal>(ord), lc)}
                            : BlockPage{committed->hold()};
                    }
                }
                else if (auto* holder = std::get_if<SharedPtr<PageHolder>>(&bp))
                {
                    if (*holder)
                    {
                        auto up = dynamicPointerCast<UncommittedPage>((*holder)->page);
                        if (up)
                        {
                            bp = std::monostate{};
                            auto committed = up->convertToCommitted(newBlock, finishEvent(), numTokens);
                            if (newBlock->eventSink)
                            {
                                newBlock->eventSink->addStoredLifeCycle(*newBlock, lc);
                            }
                            bp = committed->hold();
                        }
                    }
                }
            }
            else
            {
                // Downgrade lock to holder for our page; reuse the existing page.
                if (isLocked)
                {
                    auto holder = blockPageGetPage(bp)->hold();
                    bp = std::move(holder);
                }
                reuseTasks.push_back(
                    {existingPage->sharedFromThis(), kDefaultBeamIndex, static_cast<BlockOrdinal>(ord), lc});
            }
        }
        if (!reuseTasks.empty())
        {
            auto locks = batchedLockToGpu(*this, reuseTasks);
            for (size_t ri = 0; ri < reuseTasks.size(); ++ri)
            {
                LifeCycleId lc = reuseTasks[ri].lifeCycle;
                sb.pages[kDefaultBeamIndex][lc] = std::move(locks[ri]);
            }
        }
        // Don't clear SSM storage on rebase — the existing block may have a valid snapshot.
        sb.treeBlock = newBlock;
        TLLM_CHECK_DEBUG(_getTreeBlock(static_cast<BlockOrdinal>(ord)) == newBlock);
        ++mNumCommittedBlocks;
        didCommit = true;
    }
    else
    {
        // Can't commit and can't reuse existing block. Just stop committing.
        mCommitState = CommitState::VIRTUAL_STOP;
    }

    if (didCommit && commitSsm)
    {
        TLLM_CHECK_DEBUG(ssmLcId.has_value());
        _snapshotSsmToTreeBlock(newBlock, *ssmLcId, start + static_cast<int>(tokenBlock.size()), moveSsm);
    }

    if (sb.isCommitted())
    {
        auto const& lifeCycles = mManager->lifeCycles();
        for (LifeCycleId lcIdx{0}; lcIdx < numLc; ++lcIdx)
        {
            if (ssmLcId.has_value() && lcIdx == *ssmLcId)
            {
                continue;
            }
            LifeCycle const& lc = lifeCycles.getLifeCycle(lcIdx);
            if (!std::holds_alternative<AttnLifeCycle>(lc))
            {
                continue;
            }
            auto const staleRange = _getStaleRange(mHistoryLength, lc);
            if (staleRange.contains(BlockOrdinal{ord}))
            {
                for (auto& beamBlock : sb.pages)
                {
                    beamBlock[lcIdx] = std::monostate{};
                }
            }
        }
    }

    // Mirrors Python's tail of _commit_block:
    //   if is_last or self._commit_state == self.CommitState.VIRTUAL_STOP:
    //       self._commit_state = self.CommitState.USER_STOP
    //       self._on_stop_committing()
    if (isLast || mCommitState == CommitState::VIRTUAL_STOP)
    {
        mCommitState = CommitState::USER_STOP;
        _onStopCommitting();
    }
}

// ---------------------------------------------------------------------------
// commit
// ---------------------------------------------------------------------------

void KvCache::commit(TokenSpan tokens, bool isEnd)
{
    TLLM_CHECK_DEBUG(mStatus == Status::ACTIVE);
    if (mBeamWidth != BeamIndex{1})
        throw LogicError("Not implemented yet for beam search");
    if (tokens.size() == 0)
    {
        if (isEnd)
            stopCommitting();
        return;
    }
    if (mCommitState == CommitState::USER_STOP)
        throw LogicError("Cannot commit tokens after stop_committing()");

    TLLM_CHECK_DEBUG_WITH_INFO(
        !mTextOnly || std::none_of(tokens.begin(), tokens.end(), [](TokenIdExt const& t) { return t.isDigest(); }),
        "Cannot commit digest tokens to a text-only KV cache");

    bool const commitMinSnapshot = mManager->commitMinSnapshot();
    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    int const numCommitted = static_cast<int>(mCommittedTokens.size()) + static_cast<int>(tokens.size());
    if (commitMinSnapshot)
    {
        if (mHistoryLength != static_cast<int>(mCommittedTokens.size()) && mHistoryLength != numCommitted)
        {
            throw AssertionError("commit_min_snapshot requires commit() to start or end at history_length");
        }
    }

    // Update history before mutating the committed-token state so an invalid growth leaves commit() retryable.
    if (mCommitState != CommitState::VIRTUAL_STOP && mHistoryLength < numCommitted)
        setHistoryLength(numCommitted);

    // Append tokens to committed list.
    mCommittedTokens.insert(mCommittedTokens.end(), tokens.begin(), tokens.end());
    if (mCommitState == CommitState::VIRTUAL_STOP)
    {
        if (isEnd)
            mCommitState = CommitState::USER_STOP;
        return;
    }

    int const numCommittedBlocksBefore = mNumCommittedBlocks;
    int const newNumFullBlocks = numCommitted / mTokensPerBlock;
    bool const hasPartialSnapshot = commitMinSnapshot && (numCommitted % mTokensPerBlock != 0) && numCommitted > 0;
    bool const hasNewFullBlocks = newNumFullBlocks > numCommittedBlocksBefore;
    if (hasNewFullBlocks || hasPartialSnapshot)
    {
        // Block whose end is the last committed token — where the SSM snapshot lives.
        int const ssmSnapshotOrdinal = (numCommitted - 1) / mTokensPerBlock;
        // Wrapped in recordEventScope() so SharedPageLock::unlock() shares one finish
        // event (mirrors Python's `with self._record_event()`).
        auto scope = recordEventScope();
        for (int ordinal = numCommittedBlocksBefore; ordinal < newNumFullBlocks; ++ordinal)
        {
            bool const commitSsm = commitMinSnapshot && ssmLcId.has_value() && ordinal == ssmSnapshotOrdinal;
            _commitBlock(ordinal, /*isLast=*/false, commitSsm, /*moveSsm=*/isEnd);
            // _commitBlock transitions to USER_STOP on VIRTUAL_STOP internally.
            if (mCommitState != CommitState::ALLOWED)
                break;
        }
        if (hasPartialSnapshot && mCommitState == CommitState::ALLOWED)
        {
            BlockOrdinal const partialOrdinal{newNumFullBlocks};
            if (isEnd)
            {
                _commitBlock(newNumFullBlocks, /*isLast=*/true, /*commitSsm=*/ssmLcId.has_value(),
                    /*moveSsm=*/ssmLcId.has_value());
            }
            else
            {
                _snapshotPartialBlockToTree(partialOrdinal, /*commitSsm=*/ssmLcId.has_value());
            }
        }
    }

    if (isEnd && mCommitState != CommitState::USER_STOP)
        stopCommitting();
}

void KvCache::stopCommitting()
{
    TLLM_CHECK_DEBUG(mStatus != Status::CLOSED);
    if (mCommitState == CommitState::USER_STOP)
        return;
    TLLM_CHECK_DEBUG(_checkSanity());

    // Mirrors Python's stop_committing() which calls _commit_block(ordinal, True).
    if (mCommitState == CommitState::VIRTUAL_STOP)
    {
        mCommitState = CommitState::USER_STOP;
        return;
    }

    TLLM_CHECK_DEBUG(mCommitState == CommitState::ALLOWED);

    int tokensLeft = static_cast<int>(mCommittedTokens.size()) - mNumCommittedBlocks * mTokensPerBlock;
    if (tokensLeft > 0)
    {
        TLLM_CHECK_DEBUG(BlockOrdinal{mNumCommittedBlocks} < mBlocks.size());
        auto scope = recordEventScope();
        // isLast=true: _commitBlock handles USER_STOP + _onStopCommitting() internally.
        _commitBlock(mNumCommittedBlocks, /*isLast=*/true);
    }
    else
    {
        mCommitState = CommitState::USER_STOP;
        _onStopCommitting();
    }
    TLLM_CHECK_DEBUG(mCommitState == CommitState::USER_STOP);
}

// ---------------------------------------------------------------------------
// PlannedDropHandle — mirrors Python's PlannedDropHandle.
// ---------------------------------------------------------------------------

PlannedDropHandle::PlannedDropHandle(std::vector<CommittedPage*> const& pages)
{
    // Deduplicate by identity (mirrors Python's {id(page): page} dict).
    std::vector<CommittedPage*> unique;
    std::unordered_set<CommittedPage*> seen;
    unique.reserve(pages.size());
    for (auto* page : pages)
    {
        if (seen.insert(page).second)
            unique.push_back(page);
    }

    std::vector<WeakPtr<CommittedPage>> refs;
    refs.reserve(unique.size());
    for (auto* page : unique)
    {
        refs.emplace_back(dynamicPointerCast<CommittedPage>(page->sharedFromThis()));
        page->plannedDropCount += 1;
    }
    mPageRefs = std::move(refs);
}

void PlannedDropHandle::drop()
{
    if (!mPageRefs.has_value())
        throw std::invalid_argument("Planned drop handle has already been dropped");

    std::vector<SharedPtr<CommittedPage>> pages;
    for (auto const& ref : *mPageRefs)
    {
        auto page = ref.lock();
        if (page)
        {
            if (page->plannedDropCount <= 0)
                throw std::invalid_argument("Committed page has no planned drop");
            pages.push_back(std::move(page));
        }
    }

    mPageRefs.reset();
    for (auto const& page : pages)
    {
        page->plannedDropCount -= 1;
        if (page->plannedDropCount == 0 && page->status() == PageStatus::DROPPABLE && page->scheduledForEviction())
        {
            page->manager->excludeFromEviction(*page);
        }
    }
}

PlannedDropHandle::~PlannedDropHandle()
{
    if (mPageRefs.has_value())
    {
        // Mirror Python's __del__: apply the plan if not already dropped.
        // Destructors must not throw; swallow any error.
        try
        {
            drop();
        }
        catch (...)
        {
        }
    }
}

std::unique_ptr<PlannedDropHandle> KvCache::planCommittedBlockDrop()
{
    if (mCommitState != CommitState::USER_STOP)
        throw LogicError("plan_committed_block_drop() requires stop_committing()");

    // A cache with no committed tokens has no preceding turn to drop.
    if (numCommittedTokens() == 0)
        return nullptr;

    auto const match = mManager->matchReuse(mReuseScope, toSpan(mCommittedTokens), textOnly());
    if (match.numTokens != numCommittedTokens() || match.blocks.empty())
        return nullptr;

    BlockOrdinal const end = match.blocks.size();
    std::vector<CommittedPage*> pagesToDrop;
    for (auto const item : mManager->lifeCycles())
    {
        LifeCycleId const lcIdx = item.id;
        LifeCycle const& lc = item.lc;
        BlockOrdinal windowStart;
        if (auto const* attn = std::get_if<AttnLifeCycle>(&lc))
        {
            // Full-attention blocks may still be needed by later turns.
            if (!attn->windowSize.has_value())
                continue;
            auto const staleRange = _getStaleRange(numCommittedTokens(), lc);
            windowStart = std::min(staleRange.end, end);
        }
        else
        {
            // SSM: only the last committed block carries the reusable snapshot.
            windowStart = BlockOrdinal{end.value() - 1};
        }
        for (BlockOrdinal ordinal = windowStart; ordinal < end; ++ordinal)
        {
            CommittedPage* page = match.blocks[ordinal]->getPage(lcIdx);
            if (page == nullptr)
                return nullptr;
            pagesToDrop.push_back(page);
        }
    }
    return std::make_unique<PlannedDropHandle>(pagesToDrop);
}

// ---------------------------------------------------------------------------
// _onStopCommitting: release stale held uncommitted pages for SWA layers.
// Mirrors Python's _on_stop_committing().
// ---------------------------------------------------------------------------

void KvCache::_onStopCommitting()
{
    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    auto const& lcs = mManager->lifeCycles();

    for (auto [lcIdx, lc] : lcs)
    {
        if (ssmLcId.has_value() && lcIdx == *ssmLcId)
            continue; // SSM pages live in _ssm_blocks, not in _blocks

        auto staleRange = _getStaleRange(mHistoryLength, lc);
        BlockOrdinal start = std::max(staleRange.beg, BlockOrdinal{mNumCommittedBlocks});
        BlockOrdinal end = staleRange.end;

        TLLM_CHECK_DEBUG(end <= mBlocks.size());
        for (BlockOrdinal ord = start; ord < end; ++ord)
        {
            auto& sb = mBlocks[ord];
            TLLM_CHECK_DEBUG(!sb.isCommitted());
            for (auto& beamPages : sb.pages)
            {
                auto& bp = beamPages[lcIdx];
                if (blockPageIsNull(bp))
                {
                    // Nothing to release: scratch block, commit_min_snapshot early
                    // release, or a stale block created unallocated by resize().
                    continue;
                }
                TLLM_CHECK_DEBUG(std::holds_alternative<SharedPtr<PageHolder>>(bp));
                bp = std::monostate{};
            }
        }
    }
    TLLM_CHECK_DEBUG(_checkSanity());
}

// ---------------------------------------------------------------------------
// _setupForReuse: find existing blocks in radix tree matching input tokens.
// ---------------------------------------------------------------------------

void KvCache::_setupForReuse(BlockRadixTree::ReuseMatch const& match)
{
    auto const& matched = match.blocks;
    int const numTokens = match.numTokens;

    auto& lifeCycles = mManager->lifeCycles();
    auto const& allLc = lifeCycles.getAll();
    LifeCycleId numLc = lifeCycles.size();
    auto ssmLcId = lifeCycles.ssmLifeCycleId();
    BlockOrdinal const fullReusedEnd{numTokens / mTokensPerBlock};
    bool const hasPartialMatch = numTokens % mTokensPerBlock != 0;
    bool const recordManagerStats = _shouldRecordManagerStats();
    bool const recordRequestStats = _shouldRecordRequestStats();
    bool const shouldRecordStats = recordManagerStats || recordRequestStats;

    // --- Build blocks with stale range handling ---

    BlockOrdinal const numMatchedBlocks = matched.size();
    _resizePageIndexBuffers(numMatchedBlocks);

    for (BlockOrdinal i{0}; i < numMatchedBlocks; ++i)
    {
        SeqBlock sb;
        sb.treeBlock = matched[i]->sharedFromThis();
        sb.pages.resize(BeamIndex{1});
        sb.pages[kDefaultBeamIndex].resize(numLc);
        mBlocks.push_back(std::move(sb));
    }

    BeamIndex beamIdx = kDefaultBeamIndex;

    for (LifeCycleId lcId{0}; lcId < numLc; ++lcId)
    {
        // SSM is handled separately below.
        if (ssmLcId.has_value() && lcId == *ssmLcId)
            continue;

        auto staleRange = getStaleRange(allLc[lcId], numTokens, mTokensPerBlock);
        BlockOrdinal staleStart = staleRange.beg;
        BlockOrdinal staleEnd = staleRange.end;
        bool const isAttention = std::holds_alternative<AttnLifeCycle>(allLc[lcId]);
        int fullReusedBlocks = 0;
        int partialReusedBlocks = 0;

        // Process a non-stale ordinal: hold the page.
        // For partial blocks (last block, not full), defer the copy to first resume().
        auto processOrdinal = [&](BlockOrdinal ordinal)
        {
            auto& blk = *matched.at(ordinal);
            auto* page = blk.storage.at(lcId);
            TLLM_CHECK_DEBUG_WITH_INFO(page, "Expected page in non-stale block");
            auto& bpSlot = mBlocks[ordinal].pages[beamIdx][lcId];
            bpSlot = page->hold();
            if (shouldRecordStats && isAttention)
            {
                if (ordinal < fullReusedEnd)
                {
                    ++fullReusedBlocks;
                }
                else if (hasPartialMatch && ordinal == fullReusedEnd && _hasReuseSource(bpSlot))
                {
                    partialReusedBlocks = 1;
                }
            }
        };

        for (BlockOrdinal ord{0}; ord < staleStart; ++ord)
            processOrdinal(ord);
        for (BlockOrdinal ord = staleEnd; ord < numMatchedBlocks; ++ord)
            processOrdinal(ord);

        if (shouldRecordStats && isAttention
            && mPendingStats.recordReuse(
                lcId, fullReusedBlocks, partialReusedBlocks, recordManagerStats, recordRequestStats))
        {
            mManager->markStatsDirty(id);
        }
    }

    // SSM reuse: hold the snapshot from the last matched block. Copy is deferred to first resume().
    if (ssmLcId.has_value() && !matched.empty())
    {
        auto& snapshotBlock = *matched.back();
        auto* snapshotPage = snapshotBlock.storage[*ssmLcId];
        TLLM_CHECK_DEBUG_WITH_INFO(snapshotPage, "Last matched block must have SSM snapshot after truncation");
        mSsmBlocks[kDefaultBeamIndex][*ssmLcId] = snapshotPage->hold();
    }
    // Record one SSM snapshot lookup for this reuse-match onboarding (a miss when
    // nothing matched). Mirrors Python's record_ssm_snapshot_lookup call.
    if (_shouldRecordManagerStats() && ssmLcId.has_value())
    {
        if (mPendingStats.recordSsmSnapshotLookup(*ssmLcId, match.numLookupTokens, numTokens, mTokensPerBlock))
        {
            mManager->markStatsDirty(id);
        }
    }

    // Append matched tokens (reconstructed from the matched blocks).
    mCommittedTokens = _getMatchedTokens(match);
    mNumCommittedBlocks = numTokens / mTokensPerBlock;
    mHistoryLength = numTokens;
    mCapacity = numTokens;
}

// ---------------------------------------------------------------------------
// _getMatchedTokens — reconstruct the committed token sequence from a match.
// Mirrors Python's _get_matched_tokens().
// ---------------------------------------------------------------------------

std::vector<TokenIdExt> KvCache::_getMatchedTokens(BlockRadixTree::ReuseMatch const& match) const
{
    std::vector<TokenIdExt> ret;
    ret.reserve(static_cast<size_t>(match.numTokens));
    int remaining = match.numTokens;
    for (auto const* block : match.blocks)
    {
        TLLM_CHECK_DEBUG(remaining > 0);
        int const numBlockTokens = std::min(remaining, static_cast<int>(block->tokens.size()));
        ret.insert(ret.end(), block->tokens.begin(), block->tokens.begin() + numBlockTokens);
        remaining -= numBlockTokens;
    }
    TLLM_CHECK_DEBUG(remaining == 0);
    return ret;
}

// ---------------------------------------------------------------------------
// _getTreeBlock — get and validate the tree block at a committed ordinal.
// Mirrors Python's _get_tree_block().
// ---------------------------------------------------------------------------

SharedPtr<Block> const& KvCache::_getTreeBlock(BlockOrdinal ordinal) const
{
    TLLM_CHECK_DEBUG(mBlocks[ordinal].isCommitted());
    auto const& ret = mBlocks[ordinal].treeBlock;
    TLLM_CHECK_DEBUG(ret);
    if (TLLM_UNLIKELY(gDebug))
    {
        auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
        auto const& beamBlock = mBlocks[ordinal].pages[kDefaultBeamIndex];
        for (LifeCycleId lcId{0}; lcId < beamBlock.size(); ++lcId)
        {
            if (ssmLcId.has_value() && lcId == *ssmLcId)
            {
                TLLM_CHECK_WITH_INFO(blockPageIsNull(beamBlock[lcId]), "SSM pages live in mSsmBlocks");
            }
            else if (!blockPageIsNull(beamBlock[lcId]))
            {
                auto page = blockPageGetPage(beamBlock[lcId]);
                // committed->block may differ from `ret`: a page can be moved to a longer
                // sibling block, or replaced there by one with larger token coverage.
                auto committed = dynamicPointerCast<CommittedPage>(page);
                TLLM_CHECK(committed);
            }
        }
    }
    return ret;
}

// ---------------------------------------------------------------------------
// _checkSanity — comprehensive invariant check.
// Mirrors Python's _check_sanity().
// ---------------------------------------------------------------------------

bool KvCache::_checkSanity() const
{
    if (mStatus == Status::CLOSED)
        return numBlocks() == BlockOrdinal{0};

    TLLM_CHECK_DEBUG(numCommittedTokens() <= mHistoryLength && mHistoryLength <= mCapacity);
    TLLM_CHECK_DEBUG(numBlocks() == BlockOrdinal{divUp(mCapacity, mTokensPerBlock)});

    auto const& lcs = mManager->lifeCycles();
    LifeCycleId numLc = mManager->storage().numLifeCycles();
    auto ssmLcId = lcs.ssmLifeCycleId();

    // Precompute stale and scratch ranges for each lifecycle.
    TypedVec<LifeCycleId, HalfOpenRange<BlockOrdinal>> staleRanges(numLc);
    TypedVec<LifeCycleId, HalfOpenRange<BlockOrdinal>> scratchRangesVec(numLc);
    for (LifeCycleId lc{0}; lc < numLc; ++lc)
    {
        auto const& lifecycle = lcs.getLifeCycle(lc);
        staleRanges[lc] = _getStaleRange(mHistoryLength, lifecycle);
        scratchRangesVec[lc] = _getScratchRange(lifecycle);
    }

    for (BlockOrdinal ordinal{0}; ordinal < numBlocks(); ++ordinal)
    {
        auto const& block = mBlocks[ordinal];
        bool isCommitted = mNeverResumed || ordinal < BlockOrdinal{mNumCommittedBlocks};
        TLLM_CHECK_DEBUG(isCommitted == block.isCommitted());

        for (auto const& beamBlock : block.pages)
        {
            TLLM_CHECK_DEBUG(beamBlock.size() == numLc);
            for (LifeCycleId lc{0}; lc < numLc; ++lc)
            {
                auto const& bp = beamBlock[lc];
                if (ssmLcId.has_value() && lc == *ssmLcId)
                {
                    // SSM pages live in mSsmBlocks, not in mBlocks.
                    // When mNeverResumed and SSM snapshot is held, the block is committed
                    // but SSM page entry remains null (SSM is in mSsmBlocks).
                    TLLM_CHECK_DEBUG(blockPageIsNull(bp));
                    continue;
                }

                auto const& staleRange = staleRanges[lc];
                auto const& scratchRange = scratchRangesVec[lc];

                if (scratchRange.contains(ordinal))
                {
                    // Scratch blocks have no per-block pages.
                    TLLM_CHECK_DEBUG(blockPageIsNull(bp));
                }
                else if (staleRange.beg <= ordinal && ordinal < staleRange.end)
                {
                    if (isCommitted || mCommitState != CommitState::ALLOWED)
                    {
                        TLLM_CHECK_DEBUG(blockPageIsNull(bp));
                    }
                    else if (_getScratchRange(lcs.getLifeCycle(lc), 0).contains(ordinal))
                    {
                        // It is uncertain whether this block should hold a page: it may have
                        // been scratch-allocated by an earlier chunk, but the prior history
                        // length and capacity are not retained.
                    }
                    else
                    {
                        // For the decoder-side disagg case, for the first step, we will skip the
                        // out-of-window blocks.
                        TLLM_CHECK_DEBUG(std::holds_alternative<SharedPtr<PageHolder>>(bp)
                            || (blockPageIsNull(bp) && (mCommittedTokens.empty() || mManager->commitMinSnapshot())));
                    }
                }
                else
                {
                    // The page must be present, and locked exactly when the cache is
                    // active; a suspended cache holds it instead.
                    TLLM_CHECK_DEBUG(!std::holds_alternative<std::monostate>(bp)
                        && ((mStatus == Status::ACTIVE) == std::holds_alternative<SharedPageLock>(bp)));
                }

                if (!blockPageIsNull(bp))
                {
                    auto page = blockPageGetPage(bp);
                    TLLM_CHECK_DEBUG(isCommitted == (dynamicPointerCast<CommittedPage>(page) != nullptr));
                }
            }
        }
    }

    // Check SSM blocks (mirrors Python lines 1342-1353).
    if (ssmLcId.has_value())
    {
        for (BeamIndex bi{0}; bi < mBeamWidth; ++bi)
        {
            auto const& bp = mSsmBlocks[bi][*ssmLcId];
            if (!blockPageIsNull(bp))
            {
                if (mNeverResumed)
                {
                    // Deferred copy: SSM holds CommittedPage from matched snapshot.
                    TLLM_CHECK_DEBUG(std::holds_alternative<SharedPtr<PageHolder>>(bp));
                    auto page = blockPageGetPage(bp);
                    TLLM_CHECK_DEBUG(dynamicPointerCast<CommittedPage>(page) != nullptr);
                }
                else
                {
                    TLLM_CHECK_DEBUG(std::holds_alternative<SharedPageLock>(bp));
                    auto page = blockPageGetPage(bp);
                    TLLM_CHECK_DEBUG(dynamicPointerCast<UncommittedPage>(page) != nullptr);
                }
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Page index tables
// ---------------------------------------------------------------------------

void KvCache::_checkPageIndexBufferCapacity(BlockOrdinal newNumBlocks) const
{
    for (auto const& beamIndices : mBasePageIndices)
    {
        for (auto const& buf : beamIndices)
        {
            if (auto const* ext = std::get_if<Span<int>>(&buf))
            {
                if (ext->len < newNumBlocks.value())
                {
                    throw std::invalid_argument("User-provided base page indices is too short");
                }
            }
        }
    }
}

void KvCache::_resizePageIndexBuffers(BlockOrdinal newNumBlocks)
{
    for (BeamIndex bi{0}; bi < mBeamWidth; ++bi)
    {
        for (LifeCycleId lcId{0}; lcId < mBasePageIndices[bi].size(); ++lcId)
        {
            auto& buf = mBasePageIndices[bi][lcId];
            if (auto* vec = std::get_if<std::vector<int>>(&buf))
            {
                // When shrinking, assert tail entries are already BAD (mirrors Python line ~432).
                auto const newSize = toSizeT(newNumBlocks);
                TLLM_CHECK_DEBUG(newSize >= vec->size()
                    || std::all_of(vec->begin() + static_cast<ptrdiff_t>(newSize), vec->end(),
                        [](int idx) { return idx == kBadPageIndex.value(); }));
                // Growing fills new entries with kBadPageIndex; shrinking truncates.
                vec->resize(newSize, kBadPageIndex.value());
            }
            else
            {
                // Span<int>: caller-provided buffer must be large enough,
                // and tail beyond active blocks must already be BAD
                // (lock destructors set indices via updateBasePageIndex).
                auto& ext = std::get<Span<int>>(buf);
                int const newLen = newNumBlocks.value();
                if (ext.len < newLen)
                {
                    throw std::invalid_argument("User-provided base page indices is too short");
                }
                for (int i = newLen; i < ext.len; ++i)
                    TLLM_CHECK_DEBUG(ext[i] == kBadPageIndex.value());
            }
        }
    }
}

int KvCache::updateBasePageIndex(BeamIndex bi, BlockOrdinal ord, LifeCycleId lc, int value)
{
    if (ord == kBadBlockOrdinal)
        return kBadPageIndex.value(); // SSM pages use BAD_BLOCK_ORDINAL
    auto& buf = mBasePageIndices[bi][lc];
    return std::visit(
        [&](auto& b) -> int
        {
            using T = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, std::vector<int>>)
            {
                TLLM_CHECK_DEBUG(b.size() > toSizeT(ord));
                int old = b[toSizeT(ord)];
                b[toSizeT(ord)] = value;
                return old;
            }
            else
            {
                TLLM_CHECK_DEBUG(ord < b.len);
                int old = b[ord.value()];
                b[ord.value()] = value;
                return old;
            }
        },
        buf);
}

Span<int const> KvCache::getBasePageIndices(LayerGroupId lgId, BeamIndex beamIdx) const
{
    auto const& buf = mBasePageIndices.at(beamIdx).at(lgId);
    auto result = std::visit(
        [](auto const& b) -> Span<int const> {
            return {b.data(), static_cast<int32_t>(b.size())};
        },
        buf);
    // Cross-validate cached indices against freshly computed reference (mirrors Python lines ~350-354).
    if (TLLM_UNLIKELY(gDebug) && isActive())
    {
        auto ref = getAggregatedPageIndices(lgId, beamIdx);
        auto len = static_cast<size_t>(std::min(result.len, static_cast<int32_t>(ref.size())));
        TLLM_CHECK(std::equal(result.data(), result.data() + len, ref.begin()));
    }
    return result;
}

std::vector<int> KvCache::getAggregatedPageIndices(LayerGroupId lgId, BeamIndex beamIdx, bool validOnly) const
{
    std::vector<int> result;
    result.reserve(mBlocks.stdSize());
    for (auto const& sb : mBlocks)
    {
        auto const& pg = blockPageGetPage(sb.pages[beamIdx][lgId]);
        if (!pg)
        {
            if (!validOnly)
                result.push_back(kBadPageIndex.value());
        }
        else
        {
            result.push_back(slotIdToPageIndexValue(pg->slotId()));
        }
    }
    return result;
}

void KvCache::setBasePageIndexBuf(BeamIndex beamIdx, LayerGroupId lgId, int32_t* buf, int len)
{
    auto& slot = mBasePageIndices[beamIdx][lgId];
    BlockOrdinal const numBlocks = mBlocks.size();

    if (!buf || len == 0)
    {
        // Revert to internal vector: copy current data out of the old buffer.
        auto const& old = slot;
        if (auto const* ext = std::get_if<Span<int>>(&old))
        {
            auto const n = std::min(toSizeT(numBlocks), static_cast<size_t>(ext->len));
            std::vector<int> vec(ext->ptr, ext->ptr + n);
            slot = std::move(vec);
        }
        // If already a vector, nothing to do.
        return;
    }

    if (len < numBlocks.value())
    {
        throw std::invalid_argument("setBasePageIndexBuf: buffer length must be >= num_blocks");
    }

    // Copy current indices into the external buffer (mirrors Python: buf[:length] = old_indices[:length]).
    int const* oldData = nullptr;
    int oldLen = 0;
    if (auto* vec = std::get_if<std::vector<int>>(&slot))
    {
        oldData = vec->data();
        oldLen = static_cast<int>(vec->size());
    }
    else
    {
        auto& span = std::get<Span<int>>(slot);
        oldData = span.data();
        oldLen = span.len;
    }
    auto const copyLen = std::min(static_cast<size_t>(oldLen), toSizeT(numBlocks));
    std::copy(oldData, oldData + copyLen, buf);
    std::fill(buf + copyLen, buf + len, kBadPageIndex.value());
    slot = Span<int>{buf, len};
}

int KvCache::getSsmBlockBaseIndex(LayerGroupId lgId, BeamIndex beamIdx) const
{
    auto const& bp = mSsmBlocks.at(beamIdx).at(lgId);
    if (blockPageIsNull(bp))
        return kBadPageIndex.value();
    TLLM_CHECK_DEBUG(std::holds_alternative<SharedPageLock>(bp));
    auto const& pg = blockPageGetPage(bp);
    TLLM_CHECK_DEBUG_WITH_INFO(pg, "SSM block must have a valid page");
    return slotIdToPageIndexValue(pg->slotId()); // asserts valid slot
}

// ---------------------------------------------------------------------------
// SWA scratch slot methods
// ---------------------------------------------------------------------------

bool KvCache::hasScratchSlots() const
{
    return std::any_of(mScratchSlots.begin(), mScratchSlots.end(), [](auto const& v) { return !v.empty(); });
}

bool KvCache::isSwaScratchReuseEnabled() const noexcept
{
    return mEnableSwaScratchReuse;
}

bool KvCache::supportsIndexMode(PageIndexMode mode) const
{
    switch (mode)
    {
    case PageIndexMode::PER_LAYER: return true;
    case PageIndexMode::SHARED: return !hasScratchSlots();
    }
    return false;
}

HalfOpenRange<BlockOrdinal> KvCache::_getScratchRange(
    LifeCycle const& lc, std::optional<int> hlOverride, std::optional<int> capOverride) const
{
    if (!mEnableSwaScratchReuse)
        return {0, 0};
    int hist = hlOverride.value_or(mHistoryLength);
    int cap = capOverride.value_or(mCapacity);
    return computeScratchRange(lc, hist, cap, mTokensPerBlock, _swaScratchMaxRewindLen());
}

bool KvCache::_wouldUseSwaScratchBlocks() const
{
    int const maxRewindLen = _swaScratchMaxRewindLen();
    for (auto const& [lcId, lc] : mManager->lifeCycles())
    {
        if (computeScratchRange(lc, mHistoryLength, mCapacity, mTokensPerBlock, maxRewindLen))
            return true;
    }
    return false;
}

int KvCache::_swaScratchMaxRewindLen() const
{
    auto const& cfg = mManager->config().swaScratchReuse;
    TLLM_CHECK_DEBUG(cfg.has_value());
    return cfg->maxRewindLen;
}

std::optional<ScratchDesc> KvCache::getScratchDesc(LayerGroupId lgId) const
{
    auto const& lc = mManager->lifeCycles().getLifeCycle(lgId);
    auto sr = _getScratchRange(lc);
    if (!sr)
        return std::nullopt;
    std::vector<int> slotIds;
    slotIds.reserve(mScratchSlots[lgId].size());
    for (auto const& lock : mScratchSlots[lgId])
        slotIds.push_back(slotIdToPageIndexValue(lock.slot().slotId()));
    return ScratchDesc{sr, std::move(slotIds)};
}

void KvCache::setEnableSwaScratchReuse(bool enable)
{
    if (enable == mEnableSwaScratchReuse)
        return;
    if (enable)
    {
        if (!mManager->isSwaScratchReuseEnabled())
            throw std::invalid_argument(
                "Cannot enable SWA scratch reuse for a request when it is disabled in KV cache manager config");
        if (_wouldUseSwaScratchBlocks())
            throw std::invalid_argument(
                "Cannot enable SWA scratch reuse while the current request state would need scratch blocks");
        mEnableSwaScratchReuse = true;
        return;
    }
    if (_wouldUseSwaScratchBlocks())
        throw std::invalid_argument("Cannot disable SWA scratch reuse while scratch blocks are needed");
    TLLM_CHECK_DEBUG(!hasScratchSlots());
    mEnableSwaScratchReuse = false;
}

bool KvCache::textOnly() const noexcept
{
    return mTextOnly;
}

void KvCache::setTextOnly(bool textOnly)
{
    // A text-only deployment is a hard guarantee: a request may not opt out.
    if (!textOnly && mManager->textOnly())
    {
        throw std::invalid_argument(
            "Cannot set text_only=false for a request when the KV cache manager is configured text_only=true");
    }
    // Claiming text-only is a fast-path claim; verify the committed tokens are digest-free.
    if (textOnly)
    {
        bool const hasDigest = std::any_of(
            mCommittedTokens.begin(), mCommittedTokens.end(), [](TokenIdExt const& t) { return t.isDigest(); });
        if (hasDigest)
        {
            throw std::invalid_argument("Cannot set text_only=true: this sequence has already committed digest tokens");
        }
    }
    mTextOnly = textOnly;
}

KvCache::DeltaScratchSlots KvCache::_takeExcessScratchSlots(int capacity, int historyLength)
{
    LifeCycleId numLc = mManager->storage().numLifeCycles();
    DeltaScratchSlots result;
    result.excess.resize(numLc);
    result.deltaCnt.resize(numLc, 0);
    result.scratchRanges.resize(numLc);

    for (auto const& [lcIdx, lc] : mManager->lifeCycles())
    {
        auto scratchRange = _getScratchRange(lc, historyLength, capacity);
        result.scratchRanges[lcIdx] = scratchRange;
        int numScratchBlocks = scratchRange.length();
        auto const& fracMax = mManager->storage().slotUtilFracMax(lcIdx);
        int neededSlots = fracMax.ceilMul(numScratchBlocks);
        int existingSlots = static_cast<int>(mScratchSlots[lcIdx].size());
        int delta = neededSlots - existingSlots;
        result.deltaCnt[lcIdx] = delta;

        if (delta < 0)
        {
            for (int i = 0; i < -delta; ++i)
            {
                result.excess[lcIdx].push_back(std::move(mScratchSlots[lcIdx].back()));
                mScratchSlots[lcIdx].pop_back();
            }
        }
    }
    return result;
}

void KvCache::_recoverExcessScratchSlots(TypedVec<LifeCycleId, std::vector<ScratchSlotLock>>& excess)
{
    for (LifeCycleId lcId{0}; lcId < excess.size(); ++lcId)
    {
        for (auto& lock : excess[lcId])
        {
            mScratchSlots[lcId].push_back(std::move(lock));
        }
        excess[lcId].clear();
    }
}

void KvCache::_freeScratchSlots()
{
    for (auto& lcSlots : mScratchSlots)
    {
        for (auto& lock : lcSlots)
            lock.unlock();
        lcSlots.clear();
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
