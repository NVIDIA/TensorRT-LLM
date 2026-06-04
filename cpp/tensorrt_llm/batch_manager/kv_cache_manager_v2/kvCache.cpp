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
#include "kv_cache_manager_v2/copyEngine.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/math.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

// Copy one slot's data across all pools in a pool group.
// Used by resume() (GPU→GPU partial block copy) and _snapshotSsmToTreeBlock().
void copySlotData(StorageManager& storageMgr, CacheLevel dstLevel, CacheLevel srcLevel, PoolGroupIndex pgIdx,
    SlotId dstSlotId, SlotId srcSlotId, CUstream stream)
{
    auto slotSizes = storageMgr.slotSize(pgIdx);
    CacheTier dstTier = storageMgr.cacheTier(dstLevel);
    CacheTier srcTier = storageMgr.cacheTier(srcLevel);
    int nPools = static_cast<int>(slotSizes.size());
    for (int poolIdx = 0; poolIdx < nPools; ++poolIdx)
    {
        Address dst = storageMgr.slotAddress(dstLevel, pgIdx, dstSlotId, poolIdx);
        Address src = storageMgr.slotAddress(srcLevel, pgIdx, srcSlotId, poolIdx);
        batchedCopy(dstTier, srcTier, static_cast<size_t>(slotSizes[poolIdx]), {{dst, src}}, stream);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// KvCache constructor
// ---------------------------------------------------------------------------

KvCache::KvCache(KvCacheManager& manager, ReuseScope reuseScope, std::vector<TokenIdExt> const& inputTokens,
    std::optional<int64_t> mId, PriorityCb priorityCb)
    : id(mId)
    , mManager(manager.shared_from_this())
    , mReuseScope(std::move(reuseScope))
    , mPriorityCb(priorityCb ? std::move(priorityCb) : [](BlockOrdinal, LifeCycleId) { return kPriorityDefault; })
    , mStatus(Status::SUSPENDED)
    , mCommitState(CommitState::ALLOWED)
    , mBeamWidth(1)
    , mCapacity(0)
    , mHistoryLength(0)
    , mNumCommittedBlocks(0)
    , mTokensPerBlock(manager.tokensPerBlock())
{
    int numLc = manager.storage().numLifeCycles();

    // Initialise page index buffers: [beamIdx][lcId] = empty vector
    mBasePageIndices.assign(
        static_cast<size_t>(mBeamWidth), std::vector<PageIndexBuf>(static_cast<size_t>(numLc), std::vector<int>{}));

    // Always initialise mSsmBlocks (matching Python: no longer optional).
    mSsmBlocks.resize(static_cast<size_t>(mBeamWidth));
    for (auto& beam : mSsmBlocks)
        beam.resize(static_cast<size_t>(numLc)); // default-constructs to monostate

    mEnableSwaScratchReuse = manager.isSwaScratchReuseEnabled();
    mScratchSlots.resize(static_cast<size_t>(manager.storage().numLifeCycles()));

    if (!inputTokens.empty())
        _setupForReuse(inputTokens);

    mAvgHistoryLength.update(static_cast<double>(mHistoryLength));

    mManager->registerKvCache(this);
    mManager->updateAvgReusedLength(static_cast<double>(mHistoryLength));
    assert(gNdebug || _checkSanity());
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
    assert(mCudaStream.has_value() && "No CUDA stream attached");
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
    int numLc = mManager->storage().numLifeCycles();

    for (int lcIdx = 0; lcIdx < numLc; ++lcIdx)
    {
        LifeCycleId lcId = static_cast<LifeCycleId>(lcIdx);
        // SSM lifecycle → yield from mSsmBlocks (check individual entries).
        if (ssmLcId.has_value() && lcIdx == *ssmLcId)
        {
            for (int bi = 0; bi < mBeamWidth; ++bi)
            {
                if (!blockPageIsNull(mSsmBlocks[bi][static_cast<size_t>(*ssmLcId)]))
                    result.push_back({kBadBlockOrdinal, static_cast<BeamIndex>(bi), lcId});
            }
            continue;
        }

        // Attention lifecycle: yield non-stale blocks (sink + window).
        LifeCycle const& lc = lcs.getLifeCycle(lcId);
        auto staleRange = _getStaleRange(mHistoryLength, lc);
        int staleBeg = staleRange.beg;
        int staleEnd = staleRange.end;
        auto scratchRange = _getScratchRange(lc);

        // Sink blocks: [0, staleBeg)
        for (int ord = 0; ord < staleBeg; ++ord)
            for (int bi = 0; bi < mBeamWidth; ++bi)
                result.push_back({static_cast<BlockOrdinal>(ord), static_cast<BeamIndex>(bi), lcId});

        // Window blocks: [staleEnd, numBlocks) — skip scratch blocks.
        for (int ord = staleEnd; ord < static_cast<int>(mBlocks.size()); ++ord)
        {
            auto& block = mBlocks[static_cast<size_t>(ord)];
            for (int bi = 0; bi < mBeamWidth; ++bi)
            {
                bool isScratch = scratchRange.contains(ord);
                assert(isScratch == blockPageIsNull(block.pages[bi][static_cast<size_t>(lcIdx)]));
                if (!isScratch)
                    result.push_back({static_cast<BlockOrdinal>(ord), static_cast<BeamIndex>(bi), lcId});
            }
        }
    }
    return result;
}

SharedPtr<Page> KvCache::_page(BlockOrdinal ordinal, BeamIndex beamIdx, LifeCycleId lcId) const
{
    bool const isSsm = ordinal == kBadBlockOrdinal;
    auto const ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    assert((ssmLcId.has_value() && lcId == *ssmLcId) == isSsm);
    auto const& blockPage = isSsm
        ? mSsmBlocks.at(static_cast<size_t>(beamIdx)).at(static_cast<size_t>(lcId))
        : mBlocks.at(static_cast<size_t>(ordinal)).pages.at(static_cast<size_t>(beamIdx)).at(static_cast<size_t>(lcId));
    return blockPageGetPage(blockPage);
}

void KvCache::activate()
{
    assert(mStatus == Status::SUSPENDED);
    assert(mCudaStream.has_value() && "cuda_stream must be set before activate()");

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
            bp = &mSsmBlocks[ap.beamIdx][static_cast<size_t>(ap.lcId)];
        }
        else
        {
            bp = &mBlocks[static_cast<size_t>(ap.ordinal)].pages[ap.beamIdx][static_cast<size_t>(ap.lcId)];
        }
        auto& holder = std::get<SharedPtr<PageHolder>>(*bp);
        assert(holder);
        targets.push_back({holder->page, ap.beamIdx, ap.ordinal, ap.lcId});
    }

    {
        auto locks = batchedLockToGpu(*this, targets);
        size_t idx = 0;
        for (auto& t : targets)
        {
            assert(gNdebug || t.page == locks[idx].page());
            int bi = static_cast<int>(t.beamIndex);
            int lc = static_cast<int>(t.lifeCycle);
            if (t.ordinal == kBadBlockOrdinal)
                mSsmBlocks[bi][lc] = std::move(locks[idx++]);
            else
                mBlocks[t.ordinal].pages[bi][lc] = std::move(locks[idx++]);
        }
    }
}

bool KvCache::resume(std::optional<CUstream> stream)
{
    assert(mStatus == Status::SUSPENDED);

    // Set stream first (mirrors Python: self.cuda_stream = cuda_stream).
    if (stream.has_value())
    {
        setCudaStream(*stream);
    }
    assert(mCudaStream.has_value() && "cuda_stream is never set");
    assert(!mFinishEvent.has_value());

    // Check utilization against threshold.
    auto const utilizations = mManager->storage().getUtilization(kGpuLevel);
    float const utilization = utilizations.empty() ? 0.f : *std::max_element(utilizations.begin(), utilizations.end());
    if (utilization > mManager->config().maxUtilForResume)
    {
        return false;
    }

    auto& storageMgr = mManager->storage();
    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    int numLc = storageMgr.numLifeCycles();

    // Pre-allocate GPU slots for deferred copies (partial blocks + SSM) and scratch slots
    // before locking, so we never end up in a state where pages are locked but we can't allocate.
    std::vector<std::optional<Slot>> deferredSlots(static_cast<size_t>(numLc));

    // Compute scratch slot deltas UNCONDITIONALLY (mirrors Python: _take_excess_scratch_slots
    // is called outside _never_resumed).
    auto [excessScratch, scratchDeltaCounts, scratchRanges] = _takeExcessScratchSlots(mCapacity, mHistoryLength);
    assert(static_cast<int>(excessScratch.size()) == numLc
        && std::all_of(excessScratch.begin(), excessScratch.end(), [](auto const& s) { return s.empty(); }));

    std::vector<int> numSlotsNeeded(static_cast<size_t>(numLc), 0);
    bool hasPartial = false;
    if (mNeverResumed)
    {
        assert(mBeamWidth == 1);
        hasPartial = numCommittedTokens() % mTokensPerBlock != 0;
        for (int lc = 0; lc < numLc; ++lc)
        {
            bool isSsm = ssmLcId.has_value() && lc == *ssmLcId;
            if (isSsm || hasPartial)
                numSlotsNeeded[lc] += 1;
        }
    }

    // Add scratch slot needs UNCONDITIONALLY (mirrors Python: delta loop is outside _never_resumed).
    for (int lc = 0; lc < numLc; ++lc)
        numSlotsNeeded[lc] += std::max(0, scratchDeltaCounts[static_cast<size_t>(lc)]);

    // Only allocate if any slots are needed.
    bool anyNeeded = std::any_of(numSlotsNeeded.begin(), numSlotsNeeded.end(), [](int n) { return n > 0; });
    if (anyNeeded)
    {
        std::vector<std::vector<Slot>> tmpSlots;
        try
        {
            tmpSlots = storageMgr.newGpuSlots(numSlotsNeeded);
        }
        catch (OutOfPagesError const&)
        {
            return false;
        }

        // Separate deferred vs scratch slots, and collect scratch ready events.
        std::vector<CachedCudaEvent const*> scratchReadyEvents;
        for (int lc = 0; lc < numLc; ++lc)
        {
            if (!tmpSlots[lc].empty())
            {
                // Mirrors Python: `if self._never_resumed and (... SsmLifeCycle or has_partial):`
                bool needsDeferred = mNeverResumed && ((ssmLcId.has_value() && lc == *ssmLcId) || hasPartial);
                if (needsDeferred)
                    deferredSlots[lc] = std::move(tmpSlots[lc][0]);
                // Remaining slots are scratch slots.
                int scratchStart = needsDeferred ? 1 : 0;
                for (size_t si = static_cast<size_t>(scratchStart); si < tmpSlots[lc].size(); ++si)
                {
                    mScratchSlots[static_cast<size_t>(lc)].emplace_back(
                        std::move(tmpSlots[lc][si]), *this, static_cast<LifeCycleId>(lc), /*skipWait=*/true);
                    scratchReadyEvents.push_back(&mScratchSlots[static_cast<size_t>(lc)].back().slot().readyEvent);
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
        for (int lc = 0; lc < numLc; ++lc)
        {
            if (deferredSlots[lc].has_value())
                storageMgr.releaseSlot(static_cast<LifeCycleId>(lc), kGpuLevel, std::move(*deferredSlots[lc]));
        }
        // Scratch slots stay in mScratchSlots — they'll be freed by close() inside
        // a recordEventScope, matching Python behavior.
        return false;
    }

    // Deferred copy: for partial blocks and SSM, copy from now-locked source pages
    // to pre-allocated GPU slots, then unlock sources and replace with new pages.
    if (mNeverResumed)
    {
        BeamIndex beamIdx = 0;
        int lastOrdinal = mBlocks.empty() ? 0 : (numCommittedTokens() - 1) / mTokensPerBlock;
        CUstream cudaStr = cudaStream();

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
        for (int lcIdx = 0; lcIdx < numLc; ++lcIdx)
        {
            if (!deferredSlots[lcIdx].has_value())
                continue;
            auto& newSlot = *deferredSlots[lcIdx];

            SharedPageLock* lock = nullptr;
            if (ssmLcId.has_value() && lcIdx == *ssmLcId)
            {
                if (numCommittedTokens() == 0)
                    continue; // fresh SSM — no source to copy from
                lock = std::get_if<SharedPageLock>(&mSsmBlocks[beamIdx][static_cast<size_t>(lcIdx)]);
            }
            else
            {
                lock = std::get_if<SharedPageLock>(&mBlocks[lastOrdinal].pages[beamIdx][static_cast<size_t>(lcIdx)]);
            }
            assert(lock && lock->isValid());
            srcLocks.push_back(lock);

            PoolGroupIndex pgIdx
                = static_cast<PoolGroupIndex>(storageMgr.getPoolGroupIndex(static_cast<LifeCycleId>(lcIdx)));
            copySlotData(storageMgr, kGpuLevel, kGpuLevel, pgIdx, newSlot.slotId(), lock->page()->slotId(), cudaStr);
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
        for (int lcIdx = 0; lcIdx < numLc; ++lcIdx)
        {
            if (!deferredSlots[lcIdx].has_value())
                continue;
            auto& newSlot = *deferredSlots[lcIdx];

            BlockPage* targetBp;
            BlockOrdinal blockOrdinal;
            if (ssmLcId.has_value() && lcIdx == *ssmLcId)
            {
                targetBp = &mSsmBlocks[beamIdx][static_cast<size_t>(lcIdx)];
                blockOrdinal = kBadBlockOrdinal;
            }
            else
            {
                targetBp = &mBlocks[lastOrdinal].pages[beamIdx][static_cast<size_t>(lcIdx)];
                blockOrdinal = static_cast<BlockOrdinal>(lastOrdinal);
            }

            auto newPage
                = makeShared<UncommittedPage>(*this, blockOrdinal, static_cast<LifeCycleId>(lcIdx), kGpuLevel, beamIdx);
            newPage->setSlot(newSlot);
            auto newLock
                = newPage->lock(*this, beamIdx, blockOrdinal, static_cast<LifeCycleId>(lcIdx), /*skipWait=*/true);
            *targetBp = std::move(newLock);
        }

        // Clear treeBlock for partial last block (mirrors Python: partial block is uncommitted).
        if (numCommittedTokens() % mTokensPerBlock != 0)
            mBlocks[static_cast<size_t>(lastOrdinal)].treeBlock = nullptr;
    }

    mNeverResumed = false;
    mStatus = Status::ACTIVE;
    return true;
}

bool KvCache::prefetch(CacheLevel target)
{
    assert(mStatus == Status::SUSPENDED);
    auto& storageMgr = mManager->storage();
    int const numTiers = storageMgr.numCacheLevels();
    assert(kGpuLevel <= target && target < numTiers);

    int const numPoolGroups = storageMgr.numPoolGroups();
    std::vector<std::vector<std::vector<SharedPtr<Page>>>> allPages(
        static_cast<size_t>(numPoolGroups), std::vector<std::vector<SharedPtr<Page>>>(static_cast<size_t>(numTiers)));

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
        auto const pgIdx = static_cast<PoolGroupIndex>(storageMgr.getPoolGroupIndex(activePage.lcId));
        allPages.at(static_cast<size_t>(pgIdx)).at(static_cast<size_t>(level)).push_back(std::move(page));
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
    assert(mStatus == Status::ACTIVE);
    assert(_checkSanity());
    assert(!mFinishEvent.has_value());

    // Copy data from external buffers back to internal vectors (mirrors Python's suspend).
    for (int bi = 0; bi < mBeamWidth; ++bi)
        for (int lc = 0; lc < static_cast<int>(mBasePageIndices[bi].size()); ++lc)
            if (std::holds_alternative<Span<int>>(mBasePageIndices[bi][lc]))
                setBasePageIndexBuf(static_cast<BeamIndex>(bi), static_cast<LayerGroupId>(lc), nullptr, 0);

    // Record event scope — mirrors Python's `with self._record_event()`.
    // SharedPageLock destructors inside the scope use finishEvent() to synchronize.
    {
        auto scope = recordEventScope();
        auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();

        // Convert SharedPageLocks → PageHolders for active (non-stale) pages only.
        // Mirrors Python: for ordinal, beam_idx, lc_idx in self._active_pages()
        for (auto const& ap : _activePages())
        {
            auto& bp = (ap.lcId != ssmLcId)
                ? mBlocks[static_cast<size_t>(ap.ordinal)].pages[ap.beamIdx][static_cast<size_t>(ap.lcId)]
                : mSsmBlocks[static_cast<size_t>(ap.beamIdx)][static_cast<size_t>(ap.lcId)];
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
}

void KvCache::close()
{
    assert(gNdebug || _checkSanity());
    if (mStatus == Status::CLOSED)
        return;

    stopCommitting();
    assert(gNdebug || _checkSanity());

    if (mCapacity > 0)
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
            beamBlock[static_cast<size_t>(*ssmLcId)] = std::monostate{};
    }
}

// ---------------------------------------------------------------------------
// _snapshotSsmToTreeBlock: copy live SSM state to a new page and attach to radix tree block.
// Called at ssm_reuse_interval boundaries during commit.
// ---------------------------------------------------------------------------

void KvCache::_snapshotSsmToTreeBlock(SharedPtr<Block> const& treeBlock, LifeCycleId ssmLcId, BeamIndex beamIdx)
{
    auto& storageMgr = mManager->storage();
    auto* ssmLock = std::get_if<SharedPageLock>(&mSsmBlocks[beamIdx][static_cast<size_t>(ssmLcId)]);
    assert(ssmLock && ssmLock->isValid());
    auto srcPage = ssmLock->page();
    PoolGroupIndex pgIdx = static_cast<PoolGroupIndex>(storageMgr.getPoolGroupIndex(ssmLcId));

    for (int lvlInt = static_cast<int>(srcPage->cacheLevel); lvlInt < storageMgr.numCacheLevels(); ++lvlInt)
    {
        CacheLevel lvl = static_cast<CacheLevel>(lvlInt);

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
        copySlotData(storageMgr, lvl, srcPage->cacheLevel, pgIdx, newSlot.slotId(), srcPage->slotId(), stream);

        CachedCudaEvent readyEv(reinterpret_cast<CudaStream>(stream));
        assert(mTokensPerBlock * (treeBlock->ordinal() + 1) == static_cast<int>(mCommittedTokens.size()));

        auto tempPage = makeShared<UncommittedPage>(*this, treeBlock->ordinal(), ssmLcId, lvl, beamIdx);
        tempPage->setSlot(newSlot);
        auto committed = tempPage->convertToCommitted(treeBlock, std::move(readyEv));

        // Schedule for eviction so eviction controller keeps a strong reference,
        // preventing the page from being destroyed.
        storageMgr.scheduleForEviction(*committed);
        return; // success
    }
    // No pages available in any level, silently skip snapshot (matches Python).
}

// ---------------------------------------------------------------------------
// resize
// ---------------------------------------------------------------------------

bool KvCache::resize(std::optional<int> capacity, std::optional<int> historyLength)
{
    assert(mStatus == Status::ACTIVE);
    assert(divUp(mCapacity, mTokensPerBlock) == static_cast<int>(mBlocks.size()));

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
    if (!gNdebug && enableScratch && newCap != mCapacity)
    {
        int const maxRewindLen = _swaScratchMaxRewindLen();
        int const minHistoryLength = std::max(0, mCapacity - maxRewindLen);
        bool const validSwaScratchHistory = minHistoryLength <= newHist && newHist <= mCapacity;
        assert(validSwaScratchHistory
            && "SWA scratch requires old_capacity - max_rewind_len <= history_length <= old_capacity");
        (void) validSwaScratchHistory;
    }

    if (!enableScratch && _shortcutSetCapacity(newCap) && _shortcutSetHistoryLength(newHist))
        return true;

    int oldNumBlocks = divUp(mCapacity, mTokensPerBlock);
    int newNumBlocks = divUp(newCap, mTokensPerBlock);
    int numLc = mManager->storage().numLifeCycles();
    auto const& lcs = mManager->lifeCycles();

    _checkPageIndexBufferCapacity(newNumBlocks);

    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
    auto backupHolders = _unlockStaleBlocks(newHist);

    if (newNumBlocks < oldNumBlocks)
    {
        assert(!hasScratchSlots() && "Cannot shrink while scratch slots exist");
        auto scope = recordEventScope();
        _decreaseCapacity(newNumBlocks);
    }

    // Compute scratch deltas.
    auto [excessScratchSlots, deltaScratchSlots, scratchRanges] = _takeExcessScratchSlots(newCap, newHist);

    if (newNumBlocks >= oldNumBlocks)
    {
        // Compute new normal slots needed per lifecycle.
        std::vector<int> numNewSlots(static_cast<size_t>(numLc), 0);
        std::vector<HalfOpenRange> staleRanges(static_cast<size_t>(numLc));
        for (int lc = 0; lc < numLc; ++lc)
        {
            if (ssmLcId.has_value() && lc == *ssmLcId)
                continue;
            staleRanges[static_cast<size_t>(lc)] = _getStaleRange(newHist, lcs.getLifeCycle(lc));

            if (enableScratch)
            {
                HalfOpenRange const newBlockRange{oldNumBlocks, newNumBlocks};
                int const numNewBlocksUsingScratch
                    = intersect(scratchRanges[static_cast<size_t>(lc)], newBlockRange).length();
                int const numNewNormalBlocks = newBlockRange.length() - numNewBlocksUsingScratch;
                numNewSlots[static_cast<size_t>(lc)] = numNewNormalBlocks * mBeamWidth;
            }
            else
            {
                auto [staleBeg, staleEnd] = staleRanges[static_cast<size_t>(lc)];
                int numNewBlocksToAdd;
                if (oldNumBlocks < staleBeg)
                {
                    assert(newNumBlocks >= staleEnd);
                    numNewBlocksToAdd = (staleBeg - oldNumBlocks) + (newNumBlocks - staleEnd);
                }
                else
                {
                    numNewBlocksToAdd = newNumBlocks - std::max(staleEnd, oldNumBlocks);
                }
                numNewSlots[static_cast<size_t>(lc)] = numNewBlocksToAdd * mBeamWidth;
            }
        }

        // Compute net allocation counts (normal + scratch delta).
        std::vector<int> netAllocCounts(static_cast<size_t>(numLc), 0);
        for (int lc = 0; lc < numLc; ++lc)
            netAllocCounts[static_cast<size_t>(lc)]
                = numNewSlots[static_cast<size_t>(lc)] + deltaScratchSlots[static_cast<size_t>(lc)];

        // Allocate new slots.
        std::vector<std::vector<Slot>> newSlots;
        bool anyPositive = std::any_of(netAllocCounts.begin(), netAllocCounts.end(), [](int c) { return c > 0; });
        if (anyPositive)
        {
            std::vector<int> allocCounts(static_cast<size_t>(numLc));
            for (int lc = 0; lc < numLc; ++lc)
                allocCounts[static_cast<size_t>(lc)] = std::max(0, netAllocCounts[static_cast<size_t>(lc)]);
            try
            {
                newSlots = mManager->storage().newGpuSlots(allocCounts);
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
            newSlots.resize(static_cast<size_t>(numLc));
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
        std::vector<std::vector<Slot>> slots(static_cast<size_t>(numLc));
        for (int lc = 0; lc < numLc; ++lc)
        {
            auto& combined = slots[static_cast<size_t>(lc)];
            combined = std::move(newSlots[static_cast<size_t>(lc)]);
            for (auto& lock : excessScratchSlots[static_cast<size_t>(lc)])
                combined.push_back(lock.detachSlot());
            excessScratchSlots[static_cast<size_t>(lc)].clear();
        }

        // Release excess if net is negative.
        if (std::any_of(netAllocCounts.begin(), netAllocCounts.end(), [](int c) { return c < 0; }))
        {
            auto scope = recordEventScope();
            for (int lc = 0; lc < numLc; ++lc)
            {
                for (int i = 0; i < -netAllocCounts[static_cast<size_t>(lc)]; ++i)
                {
                    auto slot = std::move(slots[static_cast<size_t>(lc)].back());
                    slots[static_cast<size_t>(lc)].pop_back();
                    slot.readyEvent = finishEvent();
                    mManager->storage().releaseSlot(static_cast<LifeCycleId>(lc), kGpuLevel, std::move(slot));
                }
            }
        }

        // Assert correct combined slot count.
        if (!gNdebug)
        {
            for (int lc = 0; lc < numLc; ++lc)
            {
                int expected
                    = numNewSlots[static_cast<size_t>(lc)] + std::max(0, deltaScratchSlots[static_cast<size_t>(lc)]);
                assert(static_cast<int>(slots[static_cast<size_t>(lc)].size()) == expected);
            }
        }

        // Fulfill additional scratch slots (pop from end of combined slots).
        for (int lc = 0; lc < numLc; ++lc)
        {
            for (int i = 0; i < deltaScratchSlots[static_cast<size_t>(lc)]; ++i)
            {
                auto slot = std::move(slots[static_cast<size_t>(lc)].back());
                slots[static_cast<size_t>(lc)].pop_back();
                mScratchSlots[static_cast<size_t>(lc)].emplace_back(
                    std::move(slot), *this, static_cast<LifeCycleId>(lc), /*skipWait=*/true);
            }
        }

        // Resize page index buffers.
        if (!gNdebug)
        {
            for (auto const& beamIndices : mBasePageIndices)
                for (auto const& buf : beamIndices)
                    if (auto const* vec = std::get_if<std::vector<int>>(&buf))
                        assert(static_cast<int>(vec->size()) == oldNumBlocks);
        }

        // Create SeqBlocks for new ordinals (pop slots from end, matching Python).
        _resizePageIndexBuffers(newNumBlocks);
        for (int ord = oldNumBlocks; ord < newNumBlocks; ++ord)
        {
            SeqBlock sb;
            sb.pages.resize(static_cast<size_t>(mBeamWidth));
            for (auto& row : sb.pages)
                row.resize(static_cast<size_t>(numLc));
            for (int bi = 0; bi < mBeamWidth; ++bi)
            {
                for (int lc = 0; lc < numLc; ++lc)
                {
                    if (ssmLcId.has_value() && lc == *ssmLcId)
                        continue; // SSM pages live in mSsmBlocks, not in mBlocks
                    if (enableScratch)
                    {
                        if (scratchRanges[static_cast<size_t>(lc)].contains(ord))
                            continue; // Scratch block — no per-block page allocation.
                    }
                    else
                    {
                        auto [staleBeg, staleEnd] = staleRanges[static_cast<size_t>(lc)];
                        if (staleBeg <= ord && ord < staleEnd)
                            continue;
                    }
                    auto slot = std::move(slots[static_cast<size_t>(lc)].back());
                    slots[static_cast<size_t>(lc)].pop_back();
                    auto page = makeShared<UncommittedPage>(
                        *this, static_cast<BlockOrdinal>(ord), static_cast<LifeCycleId>(lc), kGpuLevel, bi);
                    page->setSlot(slot);
                    sb.pages[bi][static_cast<size_t>(lc)]
                        = page->lock(*this, bi, static_cast<BlockOrdinal>(ord), static_cast<LifeCycleId>(lc));
                }
            }
            mBlocks.push_back(std::move(sb));
        }
        if (!gNdebug)
        {
            for (int lc = 0; lc < numLc; ++lc)
                assert(slots[static_cast<size_t>(lc)].empty());
        }
    }

    mCapacity = newCap;
    mHistoryLength = newHist;
    _evictOutOfWindowBlocks(newHist);
    assert(gNdebug || _checkSanity());
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
    assert(success);
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

// ---------------------------------------------------------------------------
// Capacity management
// ---------------------------------------------------------------------------

void KvCache::_increaseCapacity(int newNumBlocks, int newHistoryLength)
{
    int curNumBlocks = static_cast<int>(mBlocks.size());
    int numLc = mManager->storage().numLifeCycles();
    auto const& lcs = mManager->lifeCycles();
    auto ssmLcId = lcs.ssmLifeCycleId();

    // Compute stale ranges using new history length so stale SWA blocks get no pages.
    std::vector<HalfOpenRange> staleRanges(static_cast<size_t>(numLc));
    std::vector<int> numSlotsPerLc(static_cast<size_t>(numLc), 0);
    for (int lc = 0; lc < numLc; ++lc)
    {
        // SSM slots are handled separately below — don't allocate block-level slots for SSM.
        if (ssmLcId.has_value() && lc == *ssmLcId)
            continue;
        staleRanges[static_cast<size_t>(lc)] = _getStaleRange(newHistoryLength, lcs.getLifeCycle(lc));
        auto [staleBeg, staleEnd] = staleRanges[static_cast<size_t>(lc)];
        int numNewBlocks;
        if (curNumBlocks < staleBeg)
        {
            assert(newNumBlocks >= staleEnd);
            numNewBlocks = (staleBeg - curNumBlocks) + (newNumBlocks - staleEnd);
        }
        else
        {
            numNewBlocks = newNumBlocks - std::max(staleEnd, curNumBlocks);
        }
        numSlotsPerLc[static_cast<size_t>(lc)] = numNewBlocks * mBeamWidth;
    }

    // SSM slots are now allocated lazily in resume() via deferred copy, not here.

    auto allSlots = mManager->storage().newGpuSlots(numSlotsPerLc);

    // Assert that internal index buffer sizes match expected old_num_blocks (mirrors Python line ~463).
    if (!gNdebug)
    {
        for (auto const& beamIndices : mBasePageIndices)
        {
            for (auto const& buf : beamIndices)
            {
                if (auto const* vec = std::get_if<std::vector<int>>(&buf))
                    assert(static_cast<int>(vec->size()) == curNumBlocks);
            }
        }
    }

    // Create SeqBlocks for the new ordinals.
    std::vector<size_t> slotCounters(static_cast<size_t>(numLc), 0);
    _resizePageIndexBuffers(newNumBlocks);
    for (int ord = curNumBlocks; ord < newNumBlocks; ++ord)
    {
        SeqBlock sb;
        sb.pages.resize(static_cast<size_t>(mBeamWidth));
        for (auto& row : sb.pages)
            row.resize(static_cast<size_t>(numLc)); // default-constructs to monostate
        for (int lc = 0; lc < numLc; ++lc)
        {
            // SSM pages live in mSsmBlocks, not in _blocks.
            if (ssmLcId.has_value() && lc == *ssmLcId)
                continue;

            auto [staleBeg, staleEnd] = staleRanges[static_cast<size_t>(lc)];
            if (staleBeg <= ord && ord < staleEnd)
                continue; // stale block for this lc — no page allocated

            size_t si = slotCounters[static_cast<size_t>(lc)]++;
            auto& slot = allSlots[static_cast<size_t>(lc)][si];
            auto page = makeShared<UncommittedPage>(
                *this, static_cast<BlockOrdinal>(ord), static_cast<LifeCycleId>(lc), kGpuLevel, /*bi=*/0);
            page->setSlot(slot);
            sb.pages[0][static_cast<size_t>(lc)]
                = page->lock(*this, 0, static_cast<BlockOrdinal>(ord), static_cast<LifeCycleId>(lc));
        }
        mBlocks.push_back(std::move(sb));
    }
    // Assert all allocated slots were consumed (mirrors Python line ~488).
    if (!gNdebug)
    {
        for (int lc = 0; lc < numLc; ++lc)
            assert(slotCounters[static_cast<size_t>(lc)] == allSlots[static_cast<size_t>(lc)].size());
    }
}

void KvCache::_decreaseCapacity(int newNumBlocks)
{
    while (static_cast<int>(mBlocks.size()) > newNumBlocks)
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

HalfOpenRange KvCache::_getStaleRange(int historyLength, LifeCycle const& lc) const
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
    int numLc = mManager->storage().numLifeCycles();

    for (int lcIdx = 0; lcIdx < numLc; ++lcIdx)
    {
        LifeCycle const& lc = lcs.getLifeCycle(static_cast<LifeCycleId>(lcIdx));
        // SSM pages live in mSsmBlocks, not _blocks — skip.
        if (std::holds_alternative<SsmLifeCycle>(lc))
            continue;
        // Full-attention (no SWA) has empty stale range — skip.
        auto const& alc = std::get<AttnLifeCycle>(lc);
        if (!alc.windowSize.has_value())
            continue;

        auto oldRange = _getStaleRange(mHistoryLength, lc);
        int oldStaleEnd = oldRange.end;
        auto newRange = _getStaleRange(newHistoryLength, lc);
        int newStaleBeg = newRange.beg;
        int newStaleEnd = newRange.end;

        int unlockStart = std::max(oldStaleEnd, newStaleBeg);
        int unlockEnd = std::min(static_cast<int>(mBlocks.size()), newStaleEnd);

        for (int ord = unlockStart; ord < unlockEnd; ++ord)
        {
            auto& sb = mBlocks[static_cast<size_t>(ord)];
            bool isCommitted = sb.isCommitted();
            bool holdForCommit = !isCommitted && (mCommitState == CommitState::ALLOWED);

            for (int bi = 0; bi < static_cast<int>(sb.pages.size()); ++bi)
            {
                auto& bp = sb.pages[bi][static_cast<size_t>(lcIdx)];
                if (blockPageIsNull(bp))
                {
                    assert(mEnableSwaScratchReuse);
                    continue; // Scratch block — already null.
                }
                assert(std::holds_alternative<SharedPageLock>(bp));
                auto holder = blockPageGetPage(bp)->hold();
                ret.push_back({static_cast<BlockOrdinal>(ord), static_cast<BeamIndex>(bi),
                    static_cast<LifeCycleId>(lcIdx), holder});
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
        mBlocks[static_cast<size_t>(t.ordinal)].pages[t.beamIdx][static_cast<size_t>(t.lcId)] = std::move(locks[i]);
    }
}

// ---------------------------------------------------------------------------
// _takeUncommittedPage — extract uncommitted pages from a SeqBlock.
// Mirrors Python's _take_uncommitted_page().
// ---------------------------------------------------------------------------

std::vector<KvCache::TakenPage> KvCache::_takeUncommittedPage(
    SeqBlock& sb, BeamIndex beamIdx, std::optional<LifeCycleId> skipLc)
{
    int numLc = mManager->storage().numLifeCycles();
    std::vector<TakenPage> result(static_cast<size_t>(numLc), {nullptr, false});
    for (int lc = 0; lc < numLc; ++lc)
    {
        if (skipLc.has_value() && lc == *skipLc)
            continue;
        auto& bp = sb.pages[beamIdx][lc];
        if (auto* lock = std::get_if<SharedPageLock>(&bp))
        {
            auto up = dynamicPointerCast<UncommittedPage>(lock->page());
            assert(up && "page must be UncommittedPage");
            result[lc] = {up, true};
        }
        else if (auto* holder = std::get_if<SharedPtr<PageHolder>>(&bp))
        {
            assert(*holder);
            auto up = dynamicPointerCast<UncommittedPage>((*holder)->page);
            assert(up && "page must be UncommittedPage");
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

void KvCache::_commitBlock(int ord, bool isLast)
{
    assert(mCommitState == CommitState::ALLOWED);
    assert(ord == mNumCommittedBlocks);

    auto& sb = mBlocks.at(static_cast<size_t>(ord));
    assert(sb.pages.size() == 1 && "Must have 1 beam only");

    // Build token block — always slice up to tokens_per_block; is_full tells us
    // whether we got a full block's worth.  Mirrors Python's:
    //   tokens = self._committed_tokens[start : start + tokens_per_block]
    //   is_full = len(tokens) == tokens_per_block
    int start = ord * mTokensPerBlock;
    int end = std::min(start + mTokensPerBlock, static_cast<int>(mCommittedTokens.size()));
    std::vector<TokenIdExt> tokenBlock(mCommittedTokens.begin() + start, mCommittedTokens.begin() + end);
    bool isFull = static_cast<int>(tokenBlock.size()) == mTokensPerBlock;

    if (!isLast && !isFull)
        throw LogicError("Cannot commit block that is not full except last block");

    // Prev node lookup (root or previous committed block).
    RootBlock& root = mManager->radixTree().addOrGetExisting(mReuseScope);
    int numLc = mManager->storage().numLifeCycles();

    NodeBase* prevNode = &root;
    if (ord > 0)
    {
        assert(mBlocks[ord - 1].treeBlock && "prev block must be committed");
        prevNode = mBlocks[ord - 1].treeBlock.get();
    }

    // Try to find or create a block in the radix tree.
    // Mirrors Python's try/except UselessBlockError pattern.
    // TODO: Replace with if-condition once Python is removed and C++ is the primary codebase.
    bool blockIsNew = false;
    SharedPtr<Block> newBlock;
    try
    {
        newBlock = addOrGetExistingBlock(prevNode, numLc, tokenBlock, &blockIsNew);
    }
    catch (UselessBlockError const& e)
    {
        newBlock = e.block;
        blockIsNew = false;
    }
    assert(newBlock);
    assert(newBlock->tokensPerBlock() == mTokensPerBlock);
    // In reuse case, verify token match (mirrors Python: tree_block.tokens[:num_tokens] == tokens).
    assert(blockIsNew || std::equal(tokenBlock.begin(), tokenBlock.end(), newBlock->tokens.begin()));

    auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();

    if (blockIsNew)
    {
        // New block: take uncommitted pages, convert to committed.
        // Mirrors Python's _take_uncommitted_page + convert path.
        auto taken = _takeUncommittedPage(sb, 0, ssmLcId);
        sb.treeBlock = newBlock;
        for (int lc = 0; lc < numLc; ++lc)
        {
            auto& [up, locked] = taken[lc];
            if (!up)
                continue;
            auto committed = up->convertToCommitted(newBlock, finishEvent());
            if (locked)
                sb.pages[0][lc]
                    = committed->lock(*this, 0, static_cast<BlockOrdinal>(ord), static_cast<LifeCycleId>(lc));
            else
                sb.pages[0][lc] = committed->hold();
        }
        // SSM snapshot: copy live SSM state at interval boundaries.
        if (ssmLcId.has_value())
        {
            int numCommitted = static_cast<int>(mCommittedTokens.size());
            int blockEnd = (ord + 1) * mTokensPerBlock;
            if (blockEnd == numCommitted && numCommitted > 0 && numCommitted % mManager->ssmReuseInterval() == 0)
            {
                _snapshotSsmToTreeBlock(newBlock, *ssmLcId, 0);
            }
            else
            {
                newBlock->storage[static_cast<size_t>(*ssmLcId)] = nullptr;
            }
        }
        assert(gNdebug || _getTreeBlock(static_cast<BlockOrdinal>(ord)) == newBlock);
        ++mNumCommittedBlocks;
    }
    else if (newBlock->isFull() && mManager->allowSeqRebasing() && isFull)
    {
        // Existing block: rebase — reuse existing block's committed pages.
        // Mirrors Python's `elif tree_block.is_full and allow_seq_rebasing and is_full` path.
        std::vector<BatchedLockTarget> reuseTasks;
        for (int lc = 0; lc < numLc; ++lc)
        {
            if (ssmLcId.has_value() && lc == *ssmLcId)
                continue;
            auto& bp = sb.pages[0][lc];
            if (blockPageIsNull(bp))
                continue;
            auto* existingPage = newBlock->storage.at(static_cast<size_t>(lc));
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
                        auto committed = up->convertToCommitted(newBlock, finishEvent());
                        bp = isLocked ? BlockPage{committed->lock(
                                 *this, 0, static_cast<BlockOrdinal>(ord), static_cast<LifeCycleId>(lc))}
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
                            auto committed = up->convertToCommitted(newBlock, finishEvent());
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
                reuseTasks.push_back({existingPage->sharedFromThis(), static_cast<BeamIndex>(0),
                    static_cast<BlockOrdinal>(ord), static_cast<LifeCycleId>(lc)});
            }
        }
        if (!reuseTasks.empty())
        {
            auto locks = batchedLockToGpu(*this, reuseTasks);
            for (size_t ri = 0; ri < reuseTasks.size(); ++ri)
            {
                int lc = static_cast<int>(reuseTasks[ri].lifeCycle);
                sb.pages[0][lc] = std::move(locks[ri]);
            }
        }
        // Don't clear SSM storage on rebase — the existing block may have a valid snapshot.
        sb.treeBlock = newBlock;
        assert(gNdebug || _getTreeBlock(static_cast<BlockOrdinal>(ord)) == newBlock);
        ++mNumCommittedBlocks;
    }
    else
    {
        // Can't commit and can't reuse existing block. Just stop committing.
        mCommitState = CommitState::VIRTUAL_STOP;
    }

    if (sb.isCommitted())
    {
        auto const& lifeCycles = mManager->lifeCycles();
        for (int lcIdx = 0; lcIdx < numLc; ++lcIdx)
        {
            if (ssmLcId.has_value() && lcIdx == *ssmLcId)
            {
                continue;
            }
            LifeCycle const& lc = lifeCycles.getLifeCycle(static_cast<LifeCycleId>(lcIdx));
            if (!std::holds_alternative<AttnLifeCycle>(lc))
            {
                continue;
            }
            auto const staleRange = _getStaleRange(mHistoryLength, lc);
            if (staleRange.contains(ord))
            {
                for (auto& beamBlock : sb.pages)
                {
                    beamBlock[static_cast<size_t>(lcIdx)] = std::monostate{};
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

void KvCache::commit(std::vector<TokenIdExt> const& tokens)
{
    assert(mStatus == Status::ACTIVE);
    if (mBeamWidth != 1)
        throw LogicError("Not implemented yet for beam search");
    if (tokens.empty())
        return;
    if (mCommitState == CommitState::USER_STOP)
        throw LogicError("Cannot commit tokens after stop_committing()");
    if (mCommitState == CommitState::VIRTUAL_STOP)
    {
        mCommittedTokens.insert(mCommittedTokens.end(), tokens.begin(), tokens.end());
        return;
    }

    // Append tokens to committed list.
    mCommittedTokens.insert(mCommittedTokens.end(), tokens.begin(), tokens.end());

    // Commit full blocks — wrapped in recordEventScope() so SharedPageLock::unlock()
    // shares one finish event (mirrors Python's `with self._record_event()`).
    {
        int numCommittedBefore = mNumCommittedBlocks;
        int newNumFullBlocks = static_cast<int>(mCommittedTokens.size()) / mTokensPerBlock;
        if (newNumFullBlocks > numCommittedBefore)
        {
            auto scope = recordEventScope();
            while (static_cast<int>(mCommittedTokens.size()) >= mTokensPerBlock * (mNumCommittedBlocks + 1))
            {
                _commitBlock(mNumCommittedBlocks, /*isLast=*/false);
                // _commitBlock transitions to USER_STOP on VIRTUAL_STOP internally.
                if (mCommitState != CommitState::ALLOWED)
                    break;
            }
        }
    }

    // Bump history_length to cover newly committed tokens (mirrors Python's behavior).
    int numCommitted = static_cast<int>(mCommittedTokens.size());
    if (mHistoryLength < numCommitted)
        setHistoryLength(numCommitted);
}

void KvCache::stopCommitting()
{
    assert(mStatus != Status::CLOSED);
    if (mCommitState == CommitState::USER_STOP)
        return;
    assert(gNdebug || _checkSanity());

    // Mirrors Python's stop_committing() which calls _commit_block(ordinal, True).
    if (mCommitState == CommitState::VIRTUAL_STOP)
    {
        mCommitState = CommitState::USER_STOP;
        return;
    }

    assert(mCommitState == CommitState::ALLOWED);

    int tokensLeft = static_cast<int>(mCommittedTokens.size()) - mNumCommittedBlocks * mTokensPerBlock;
    if (tokensLeft > 0)
    {
        assert(mNumCommittedBlocks < static_cast<int>(mBlocks.size()));
        auto scope = recordEventScope();
        // isLast=true: _commitBlock handles USER_STOP + _onStopCommitting() internally.
        _commitBlock(mNumCommittedBlocks, /*isLast=*/true);
    }
    else
    {
        mCommitState = CommitState::USER_STOP;
        _onStopCommitting();
    }
    assert(mCommitState == CommitState::USER_STOP);
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
        int start = std::max(staleRange.beg, mNumCommittedBlocks);
        int end = staleRange.end;

        assert(end <= static_cast<int>(mBlocks.size()));
        for (int ord = start; ord < end; ++ord)
        {
            auto& sb = mBlocks[static_cast<size_t>(ord)];
            assert(!sb.isCommitted());
            for (auto& beamPages : sb.pages)
            {
                auto& bp = beamPages[static_cast<size_t>(lcIdx)];
                if (blockPageIsNull(bp))
                {
                    assert(mEnableSwaScratchReuse);
                    continue; // Scratch block — already handled
                }
                assert(gNdebug || std::holds_alternative<SharedPtr<PageHolder>>(bp));
                bp = std::monostate{};
            }
        }
    }
    assert(gNdebug || _checkSanity());
}

// ---------------------------------------------------------------------------
// _setupForReuse: find existing blocks in radix tree matching input tokens.
// ---------------------------------------------------------------------------

void KvCache::_setupForReuse(std::vector<TokenIdExt> const& inputTokens)
{
    auto match = mManager->radixTree().match(mReuseScope, inputTokens, mManager->enablePartialMatch());
    auto const& matched = match.blocks;
    int const numTokens = match.numTokens;

    auto& lifeCycles = mManager->lifeCycles();
    auto const& allLc = lifeCycles.getAll();
    int numLc = lifeCycles.size();
    auto ssmLcId = lifeCycles.ssmLifeCycleId();

    // --- Build blocks with stale range handling ---

    _resizePageIndexBuffers(static_cast<int>(matched.size()));

    for (size_t i = 0; i < matched.size(); ++i)
    {
        SeqBlock sb;
        sb.treeBlock = matched[i]->sharedFromThis();
        sb.pages.resize(1);
        sb.pages[0].resize(static_cast<size_t>(numLc));
        mBlocks.push_back(std::move(sb));
    }

    BeamIndex beamIdx = 0;

    for (int lcId = 0; lcId < numLc; ++lcId)
    {
        // SSM is handled separately below.
        if (ssmLcId.has_value() && lcId == *ssmLcId)
            continue;

        auto staleRange = getStaleRange(allLc[lcId], numTokens, mTokensPerBlock);
        int staleStart = staleRange.beg;
        int staleEnd = staleRange.end;

        // Process a non-stale ordinal: hold the page.
        // For partial blocks (last block, not full), defer the copy to first resume().
        auto processOrdinal = [&](int ordinal)
        {
            auto& blk = *matched[ordinal];
            auto* page = blk.storage.at(static_cast<size_t>(lcId));
            assert(page && "Expected page in non-stale block");
            auto& bpSlot = mBlocks[ordinal].pages[beamIdx][lcId];
            bpSlot = page->hold();
        };

        for (int ord = 0; ord < staleStart; ++ord)
            processOrdinal(ord);
        for (int ord = staleEnd; ord < static_cast<int>(matched.size()); ++ord)
            processOrdinal(ord);
    }

    // SSM reuse: hold the snapshot from the last matched block. Copy is deferred to first resume().
    if (ssmLcId.has_value() && !matched.empty())
    {
        auto& snapshotBlock = *matched.back();
        auto* snapshotPage = snapshotBlock.storage[static_cast<size_t>(*ssmLcId)];
        assert(snapshotPage && "Last matched block must have SSM snapshot after truncation");
        mSsmBlocks[0][static_cast<size_t>(*ssmLcId)] = snapshotPage->hold();
    }

    // Append matched tokens.
    mCommittedTokens.assign(
        inputTokens.begin(), inputTokens.begin() + std::min(numTokens, static_cast<int>(inputTokens.size())));
    mNumCommittedBlocks = numTokens / mTokensPerBlock;
    mHistoryLength = numTokens;
    mCapacity = numTokens;
}

// ---------------------------------------------------------------------------
// _getTreeBlock — get and validate the tree block at a committed ordinal.
// Mirrors Python's _get_tree_block().
// ---------------------------------------------------------------------------

SharedPtr<Block> const& KvCache::_getTreeBlock(BlockOrdinal ordinal) const
{
    assert(mBlocks[static_cast<size_t>(ordinal)].isCommitted());
    auto const& ret = mBlocks[static_cast<size_t>(ordinal)].treeBlock;
    assert(ret);
    if (!gNdebug)
    {
        auto ssmLcId = mManager->lifeCycles().ssmLifeCycleId();
        auto const& beamBlock = mBlocks[static_cast<size_t>(ordinal)].pages[0];
        for (int lc = 0; lc < static_cast<int>(beamBlock.size()); ++lc)
        {
            if (ssmLcId.has_value() && lc == *ssmLcId)
            {
                assert(blockPageIsNull(beamBlock[lc]) && "SSM pages live in mSsmBlocks");
            }
            else if (!blockPageIsNull(beamBlock[lc]))
            {
                auto page = blockPageGetPage(beamBlock[lc]);
                auto committed = dynamicPointerCast<CommittedPage>(page);
                assert(committed && committed->block == ret.get());
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
        return numBlocks() == 0;

    assert(numCommittedTokens() <= mHistoryLength && mHistoryLength <= mCapacity);
    assert(numBlocks() == divUp(mCapacity, mTokensPerBlock));

    auto const& lcs = mManager->lifeCycles();
    int numLc = mManager->storage().numLifeCycles();
    auto ssmLcId = lcs.ssmLifeCycleId();

    // Precompute stale and scratch ranges for each lifecycle.
    std::vector<HalfOpenRange> staleRanges(static_cast<size_t>(numLc));
    std::vector<HalfOpenRange> scratchRangesVec(static_cast<size_t>(numLc));
    for (int lc = 0; lc < numLc; ++lc)
    {
        auto const& lifecycle = lcs.getLifeCycle(static_cast<LifeCycleId>(lc));
        staleRanges[lc] = _getStaleRange(mHistoryLength, lifecycle);
        scratchRangesVec[lc] = _getScratchRange(lifecycle);
    }

    for (int ordinal = 0; ordinal < numBlocks(); ++ordinal)
    {
        auto const& block = mBlocks[static_cast<size_t>(ordinal)];
        bool isCommitted = mNeverResumed || ordinal < mNumCommittedBlocks;
        assert(isCommitted == block.isCommitted());

        for (auto const& beamBlock : block.pages)
        {
            assert(static_cast<int>(beamBlock.size()) == numLc);
            for (int lc = 0; lc < numLc; ++lc)
            {
                auto const& bp = beamBlock[lc];
                if (ssmLcId.has_value() && lc == *ssmLcId)
                {
                    // SSM pages live in mSsmBlocks, not in mBlocks.
                    // When mNeverResumed and SSM snapshot is held, the block is committed
                    // but SSM page entry remains null (SSM is in mSsmBlocks).
                    assert(blockPageIsNull(bp));
                    continue;
                }

                auto const& staleRange = staleRanges[lc];
                auto const& scratchRange = scratchRangesVec[lc];

                if (scratchRange.contains(ordinal))
                {
                    // Scratch blocks have no per-block pages.
                    assert(blockPageIsNull(bp));
                }
                else if (staleRange.beg <= ordinal && ordinal < staleRange.end)
                {
                    if (isCommitted || mCommitState != CommitState::ALLOWED)
                    {
                        assert(blockPageIsNull(bp));
                    }
                    else
                    {
                        // For the decoder-side disagg case, for the first step, we will skip the
                        // out-of-window blocks.
                        assert(std::holds_alternative<SharedPtr<PageHolder>>(bp)
                            || (blockPageIsNull(bp) && mCommittedTokens.empty()));
                    }
                }
                else
                {
                    if (mStatus == Status::ACTIVE)
                        assert(std::holds_alternative<SharedPageLock>(bp));
                    else
                        assert(std::holds_alternative<SharedPtr<PageHolder>>(bp));
                }

                if (!blockPageIsNull(bp))
                {
                    auto page = blockPageGetPage(bp);
                    assert(isCommitted == (dynamicPointerCast<CommittedPage>(page) != nullptr));
                }
            }
        }
    }

    // Check SSM blocks (mirrors Python lines 1342-1353).
    if (ssmLcId.has_value())
    {
        for (int bi = 0; bi < mBeamWidth; ++bi)
        {
            auto const& bp = mSsmBlocks[static_cast<size_t>(bi)][static_cast<size_t>(*ssmLcId)];
            if (!blockPageIsNull(bp))
            {
                if (mNeverResumed)
                {
                    // Deferred copy: SSM holds CommittedPage from matched snapshot.
                    assert(std::holds_alternative<SharedPtr<PageHolder>>(bp));
                    auto page = blockPageGetPage(bp);
                    assert(dynamicPointerCast<CommittedPage>(page) != nullptr);
                }
                else
                {
                    assert(std::holds_alternative<SharedPageLock>(bp));
                    auto page = blockPageGetPage(bp);
                    assert(dynamicPointerCast<UncommittedPage>(page) != nullptr);
                }
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Page index tables
// ---------------------------------------------------------------------------

void KvCache::_checkPageIndexBufferCapacity(int newNumBlocks) const
{
    for (auto const& beamIndices : mBasePageIndices)
    {
        for (auto const& buf : beamIndices)
        {
            if (auto const* ext = std::get_if<Span<int>>(&buf))
            {
                if (ext->len < newNumBlocks)
                {
                    throw std::invalid_argument("User-provided base page indices is too short");
                }
            }
        }
    }
}

void KvCache::_resizePageIndexBuffers(int newNumBlocks)
{
    for (int bi = 0; bi < mBeamWidth; ++bi)
    {
        for (int lc = 0; lc < static_cast<int>(mBasePageIndices[bi].size()); ++lc)
        {
            auto& buf = mBasePageIndices[bi][lc];
            if (auto* vec = std::get_if<std::vector<int>>(&buf))
            {
                // When shrinking, assert tail entries are already BAD (mirrors Python line ~432).
                if (!gNdebug && newNumBlocks < static_cast<int>(vec->size()))
                {
                    for (int i = newNumBlocks; i < static_cast<int>(vec->size()); ++i)
                        assert((*vec)[i] == kBadPageIndex);
                }
                // Growing fills new entries with kBadPageIndex; shrinking truncates.
                vec->resize(static_cast<size_t>(newNumBlocks), kBadPageIndex);
            }
            else
            {
                // Span<int>: caller-provided buffer must be large enough,
                // and tail beyond active blocks must already be BAD
                // (lock destructors set indices via updateBasePageIndex).
                auto& ext = std::get<Span<int>>(buf);
                if (ext.len < newNumBlocks)
                {
                    throw std::invalid_argument("User-provided base page indices is too short");
                }
                for (int i = newNumBlocks; i < ext.len; ++i)
                    assert(ext[i] == kBadPageIndex);
            }
        }
    }
}

int KvCache::updateBasePageIndex(BeamIndex bi, BlockOrdinal ord, LifeCycleId lc, int value)
{
    if (ord == kBadBlockOrdinal)
        return kBadPageIndex; // SSM pages use BAD_BLOCK_ORDINAL
    auto& buf = mBasePageIndices[bi][lc];
    return std::visit(
        [&](auto& b) -> int
        {
            using T = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, std::vector<int>>)
            {
                assert(static_cast<int>(b.size()) > ord);
            }
            else
            {
                assert(ord < b.len);
            }
            int old = b[ord];
            b[ord] = value;
            return old;
        },
        buf);
}

Span<int const> KvCache::getBasePageIndices(LayerGroupId lgId, BeamIndex beamIdx) const
{
    auto const& buf = mBasePageIndices.at(static_cast<size_t>(beamIdx)).at(static_cast<size_t>(lgId));
    auto result = std::visit(
        [](auto const& b) -> Span<int const> {
            return {b.data(), static_cast<int32_t>(b.size())};
        },
        buf);
    // Cross-validate cached indices against freshly computed reference (mirrors Python lines ~350-354).
    if (!gNdebug && isActive())
    {
        auto ref = getAggregatedPageIndices(lgId, beamIdx);
        int len = std::min(result.len, static_cast<int32_t>(ref.size()));
        for (int i = 0; i < len; ++i)
            assert(result[i] == ref[i]);
    }
    return result;
}

std::vector<int> KvCache::getAggregatedPageIndices(LayerGroupId lgId, BeamIndex beamIdx, bool validOnly) const
{
    std::vector<int> result;
    result.reserve(mBlocks.size());
    for (auto const& sb : mBlocks)
    {
        auto const& pg = blockPageGetPage(sb.pages[static_cast<size_t>(beamIdx)][static_cast<size_t>(lgId)]);
        if (!pg)
        {
            if (!validOnly)
                result.push_back(kBadPageIndex);
        }
        else
        {
            result.push_back(static_cast<int>(pg->slotId()));
        }
    }
    return result;
}

void KvCache::setBasePageIndexBuf(BeamIndex beamIdx, LayerGroupId lgId, int32_t* buf, int len)
{
    auto& slot = mBasePageIndices[static_cast<size_t>(beamIdx)][static_cast<size_t>(lgId)];
    int numBlocks = static_cast<int>(mBlocks.size());

    if (!buf || len == 0)
    {
        // Revert to internal vector: copy current data out of the old buffer.
        auto const& old = slot;
        if (auto const* ext = std::get_if<Span<int>>(&old))
        {
            int n = std::min(numBlocks, ext->len);
            std::vector<int> vec(ext->ptr, ext->ptr + n);
            slot = std::move(vec);
        }
        // If already a vector, nothing to do.
        return;
    }

    if (len < numBlocks)
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
    int copyLen = std::min(oldLen, numBlocks);
    std::copy(oldData, oldData + copyLen, buf);
    std::fill(buf + copyLen, buf + len, kBadPageIndex);
    slot = Span<int>{buf, len};
}

int KvCache::getSsmBlockBaseIndex(LayerGroupId lgId, BeamIndex beamIdx) const
{
    auto const& bp = mSsmBlocks.at(static_cast<size_t>(beamIdx)).at(static_cast<size_t>(lgId));
    if (blockPageIsNull(bp))
        return kBadPageIndex;
    assert(std::holds_alternative<SharedPageLock>(bp));
    auto const& pg = blockPageGetPage(bp);
    assert(pg && "SSM block must have a valid page");
    return static_cast<int>(pg->slotId()); // asserts valid slot
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

HalfOpenRange KvCache::_getScratchRange(
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
    assert(cfg.has_value());
    return cfg->maxRewindLen;
}

std::optional<ScratchDesc> KvCache::getScratchDesc(LayerGroupId lgId) const
{
    auto const& lc = mManager->lifeCycles().getLifeCycle(lgId);
    auto sr = _getScratchRange(lc);
    if (!sr)
        return std::nullopt;
    std::vector<int> slotIds;
    slotIds.reserve(mScratchSlots[static_cast<size_t>(lgId)].size());
    for (auto const& lock : mScratchSlots[static_cast<size_t>(lgId)])
        slotIds.push_back(lock.slot().slotId());
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
    assert(!hasScratchSlots());
    mEnableSwaScratchReuse = false;
}

KvCache::DeltaScratchSlots KvCache::_takeExcessScratchSlots(int capacity, int historyLength)
{
    int numLc = mManager->storage().numLifeCycles();
    DeltaScratchSlots result;
    result.excess.resize(static_cast<size_t>(numLc));
    result.deltaCnt.resize(static_cast<size_t>(numLc), 0);
    result.scratchRanges.resize(static_cast<size_t>(numLc));

    for (auto const& [lcIdx, lc] : mManager->lifeCycles())
    {
        auto scratchRange = _getScratchRange(lc, historyLength, capacity);
        result.scratchRanges[static_cast<size_t>(lcIdx)] = scratchRange;
        int numScratchBlocks = scratchRange.length();
        auto const& fracMax = mManager->storage().slotUtilFracMax(lcIdx);
        int neededSlots = fracMax.ceilMul(numScratchBlocks);
        int existingSlots = static_cast<int>(mScratchSlots[static_cast<size_t>(lcIdx)].size());
        int delta = neededSlots - existingSlots;
        result.deltaCnt[static_cast<size_t>(lcIdx)] = delta;

        if (delta < 0)
        {
            for (int i = 0; i < -delta; ++i)
            {
                result.excess[static_cast<size_t>(lcIdx)].push_back(
                    std::move(mScratchSlots[static_cast<size_t>(lcIdx)].back()));
                mScratchSlots[static_cast<size_t>(lcIdx)].pop_back();
            }
        }
    }
    return result;
}

void KvCache::_recoverExcessScratchSlots(std::vector<std::vector<ScratchSlotLock>>& excess)
{
    int numLc = static_cast<int>(excess.size());
    for (int lc = 0; lc < numLc; ++lc)
    {
        for (auto& lock : excess[static_cast<size_t>(lc)])
            mScratchSlots[static_cast<size_t>(lc)].push_back(std::move(lock));
        excess[static_cast<size_t>(lc)].clear();
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
