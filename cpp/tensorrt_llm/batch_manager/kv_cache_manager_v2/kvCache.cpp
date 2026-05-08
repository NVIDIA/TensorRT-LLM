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

KvCache::KvCache(KvCacheManager& manager, std::optional<int64_t> loraTaskId, std::vector<TokenIdExt> const& inputTokens,
    std::optional<int64_t> mId, PriorityCb priorityCb)
    : id(mId)
    , mManager(manager.shared_from_this())
    , mLoraTaskId(loraTaskId)
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

    if (!inputTokens.empty())
        _setupForReuse(inputTokens);

    mAvgHistoryLength.update(static_cast<double>(mHistoryLength));
    mAvgCapacity.update(static_cast<double>(mCapacity));

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

        // Sink blocks: [0, staleBeg)
        for (int ord = 0; ord < staleBeg; ++ord)
            for (int bi = 0; bi < mBeamWidth; ++bi)
                result.push_back({static_cast<BlockOrdinal>(ord), static_cast<BeamIndex>(bi), lcId});

        // Window blocks: [staleEnd, numBlocks)
        for (int ord = staleEnd; ord < static_cast<int>(mBlocks.size()); ++ord)
            for (int bi = 0; bi < mBeamWidth; ++bi)
                result.push_back({static_cast<BlockOrdinal>(ord), static_cast<BeamIndex>(bi), lcId});
    }
    return result;
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
        auto& holder = std::get<std::shared_ptr<PageHolder>>(*bp);
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

    // Pre-allocate GPU slots for deferred copies (partial blocks + SSM) before locking,
    // so we never end up in a state where pages are locked but we can't allocate for the copy.
    std::vector<std::optional<Slot>> deferredSlots(static_cast<size_t>(numLc));
    if (mNeverResumed)
    {
        assert(mBeamWidth == 1);
        bool hasPartial = numCommittedTokens() % mTokensPerBlock != 0;
        std::vector<int> numSlotsNeeded(static_cast<size_t>(numLc), 0);
        for (int lc = 0; lc < numLc; ++lc)
        {
            bool isSsm = ssmLcId.has_value() && lc == *ssmLcId;
            if (isSsm || hasPartial)
                numSlotsNeeded[lc] = 1;
        }
        // Only allocate if any slots are needed (i.e., there's something to copy).
        bool anyNeeded = false;
        for (int n : numSlotsNeeded)
        {
            if (n > 0)
            {
                anyNeeded = true;
                break;
            }
        }
        if (anyNeeded)
        {
            try
            {
                auto tmpSlots = storageMgr.newGpuSlots(numSlotsNeeded);
                for (int lc = 0; lc < numLc; ++lc)
                {
                    if (!tmpSlots[lc].empty())
                        deferredSlots[lc] = std::move(tmpSlots[lc][0]);
                }
            }
            catch (OutOfPagesError const&)
            {
                return false;
            }
        }
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

            auto newPage = std::make_shared<UncommittedPage>(
                *this, blockOrdinal, static_cast<LifeCycleId>(lcIdx), kGpuLevel, beamIdx);
            newPage->setSlot(newSlot);
            auto newLock
                = newPage->lock(*this, beamIdx, blockOrdinal, static_cast<LifeCycleId>(lcIdx), /*skipWait=*/true);
            *targetBp = std::move(newLock);
        }
    }

    mNeverResumed = false;
    mStatus = Status::ACTIVE;
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
            auto& bp = (ap.ordinal != kBadBlockOrdinal)
                ? mBlocks[static_cast<size_t>(ap.ordinal)].pages[ap.beamIdx][static_cast<size_t>(ap.lcId)]
                : mSsmBlocks[static_cast<size_t>(ap.beamIdx)][static_cast<size_t>(ap.lcId)];
            // expect_type(_SharedPageLock, beam_block[lc_idx]) → std::get raises on wrong type
            auto& lock = std::get<SharedPageLock>(bp);
            auto holder = lock.page()->hold();
            bp = std::move(holder); // ~SharedPageLock calls unlock() → notifyFinish(finishEvent())
        }
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

    mManager->updateAvgSqrCapacity(mAvgCapacity.value() * mAvgCapacity.value());
    mManager->updateAvgSqrHistoryLength(mAvgHistoryLength.value() * mAvgHistoryLength.value());
    mManager->tryUpdateTargetRatios();

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

void KvCache::_snapshotSsmToTreeBlock(std::shared_ptr<Block> const& treeBlock, LifeCycleId ssmLcId, BeamIndex beamIdx)
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
        assert(mTokensPerBlock * (treeBlock->ordinal + 1) == static_cast<int>(mCommittedTokens.size()));

        auto tempPage = std::make_shared<UncommittedPage>(*this, treeBlock->ordinal, ssmLcId, lvl, beamIdx);
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

    if (_shortcutSetCapacity(newCap) && _shortcutSetHistoryLength(newHist))
        return true;

    auto backupHolders = _unlockStaleBlocks(newHist);

    int oldNumBlocks = divUp(mCapacity, mTokensPerBlock);
    int newNumBlocks = divUp(newCap, mTokensPerBlock);

    if (newNumBlocks < oldNumBlocks)
    {
        auto scope = recordEventScope();
        _decreaseCapacity(newNumBlocks);
    }
    else if (newNumBlocks > oldNumBlocks)
    {
        try
        {
            _increaseCapacity(newNumBlocks, newHist);
        }
        catch (OutOfPagesError const&)
        {
            _lockHeldBlocks(backupHolders);
            return false;
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
            auto page = std::make_shared<UncommittedPage>(
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
            auto up = std::dynamic_pointer_cast<UncommittedPage>(lock->page());
            assert(up && "page must be UncommittedPage");
            result[lc] = {up, true};
        }
        else if (auto* holder = std::get_if<std::shared_ptr<PageHolder>>(&bp))
        {
            assert(*holder);
            auto up = std::dynamic_pointer_cast<UncommittedPage>((*holder)->page);
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

    // Parent block lookup (root or previous committed block).
    RootBlock& root = mManager->radixTree().addOrGetExisting(mLoraTaskId);
    int numLc = mManager->storage().numLifeCycles();

    std::shared_ptr<Block> parentBlock;
    std::unordered_map<BlockKey, std::shared_ptr<Block>>* parentNext = &root.next;
    BlockKey const* parentKey = &root.key;
    if (ord > 0)
    {
        assert(mBlocks[ord - 1].treeBlock && "prev block must be committed");
        parentBlock = mBlocks[ord - 1].treeBlock;
        parentNext = &parentBlock->next;
        parentKey = &parentBlock->key;
    }

    // Try to find or create a block in the radix tree.
    // Mirrors Python's try/except UselessBlockError pattern.
    // TODO: Replace with if-condition once Python is removed and C++ is the primary codebase.
    bool blockIsNew = false;
    std::shared_ptr<Block> newBlock;
    try
    {
        newBlock = addOrGetExistingBlock(*parentNext, *parentKey, numLc, mTokensPerBlock, tokenBlock,
            ord == 0 ? &root : nullptr, parentBlock.get(), &blockIsNew);
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
                newBlock->storage[static_cast<size_t>(*ssmLcId)].reset();
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
            auto existingPage = newBlock->storage.at(static_cast<size_t>(lc)).lock();
            bool isLocked = std::holds_alternative<SharedPageLock>(bp);
            if (!existingPage)
            {
                // Existing page gone — put our uncommitted page into the tree block.
                if (auto* lock = std::get_if<SharedPageLock>(&bp))
                {
                    auto up = std::dynamic_pointer_cast<UncommittedPage>(lock->page());
                    if (up)
                    {
                        bp = std::monostate{};
                        auto committed = up->convertToCommitted(newBlock, finishEvent());
                        bp = isLocked ? BlockPage{committed->lock(
                                 *this, 0, static_cast<BlockOrdinal>(ord), static_cast<LifeCycleId>(lc))}
                                      : BlockPage{committed->hold()};
                    }
                }
                else if (auto* holder = std::get_if<std::shared_ptr<PageHolder>>(&bp))
                {
                    if (*holder)
                    {
                        auto up = std::dynamic_pointer_cast<UncommittedPage>((*holder)->page);
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
                reuseTasks.push_back({existingPage, static_cast<BeamIndex>(0), static_cast<BlockOrdinal>(ord),
                    static_cast<LifeCycleId>(lc)});
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
                assert(gNdebug || std::holds_alternative<std::shared_ptr<PageHolder>>(bp) || blockPageIsNull(bp));
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
    auto matched = mManager->radixTree().match(mLoraTaskId, inputTokens, mManager->enablePartialMatch());
    // Assert all non-last matched blocks are full (mirrors Python line ~1113).
    if (!gNdebug && matched.size() > 1)
    {
        for (size_t i = 0; i + 1 < matched.size(); ++i)
            assert(matched[i].numMatchedTokens == mTokensPerBlock);
    }
    auto& lifeCycles = mManager->lifeCycles();
    auto const& allLc = lifeCycles.getAll();
    int numLc = lifeCycles.size();
    auto ssmLcId = lifeCycles.ssmLifeCycleId();

    // Helper: compute num matched tokens from current matched list.
    auto getNumMatchedTokens = [&]() -> int
    {
        return matched.empty()
            ? 0
            : mTokensPerBlock * (static_cast<int>(matched.size()) - 1) + matched.back().numMatchedTokens;
    };

    auto hasPage = [](Block const& blk, LifeCycleId lcId) -> bool
    { return !blk.storage.at(static_cast<size_t>(lcId)).expired(); };

    // Use attentionLifeCycles() for full-attention and SWA checks.
    auto attnLcs = lifeCycles.attentionLifeCycles();

    // --- Trim matched blocks based on page availability ---

    // Check for full attention layers: trim if blocks lack pages.
    {
        std::vector<LifeCycleId> fullAttnLcList;
        for (auto [lcId, attn] : attnLcs)
        {
            if (!attn->windowSize.has_value())
                fullAttnLcList.push_back(lcId);
        }
        if (!fullAttnLcList.empty())
        {
            int n = findIndex(matched.begin(), matched.end(),
                [&](auto const& m)
                {
                    for (auto lcId : fullAttnLcList)
                    {
                        if (!hasPage(*m.block, lcId))
                            return true;
                    }
                    return false;
                });
            matched.resize(static_cast<size_t>(n));
        }
    }

    // Collect SWA lifecycles.
    std::vector<std::pair<LifeCycleId, AttnLifeCycle const*>> swaLcs;
    for (auto [lcId, attn] : attnLcs)
    {
        if (attn->windowSize.has_value())
            swaLcs.push_back({lcId, attn});
    }

    // Check for SWA sink blocks.
    for (auto [lcId, attn] : swaLcs)
    {
        int sinkBlocks = attn->numSinkBlocks;
        int limit = std::min(sinkBlocks, static_cast<int>(matched.size()));
        int n = findIndex(
            matched.begin(), matched.begin() + limit, [&](auto const& m) { return !hasPage(*m.block, lcId); });
        if (n < sinkBlocks)
            matched.resize(static_cast<size_t>(n));
    }

    // Check SWA window and SSM snapshot constraints together.
    // SSM is checked first (intervals are large, so it prunes more).
    while (!matched.empty())
    {
        // SSM truncation: truncate to the last block with an SSM snapshot.
        if (ssmLcId.has_value())
        {
            int ssmTrunc = 0;
            for (int i = static_cast<int>(matched.size()) - 1; i >= 0; --i)
            {
                if (hasPage(*matched[i].block, *ssmLcId))
                {
                    ssmTrunc = i + 1;
                    break;
                }
            }
            matched.resize(static_cast<size_t>(ssmTrunc));
            if (matched.empty())
                break;
        }

        // SWA window check.
        int numTok = getNumMatchedTokens();
        bool trimmed = false;
        for (auto [lcId, attn] : swaLcs)
        {
            if (!attn->windowSize.has_value())
                continue;

            // Check tail: first block from end that HAS a page.
            int n = findIndex(matched.rbegin(), matched.rend(), [&](auto const& m) { return hasPage(*m.block, lcId); });
            if (n != 0)
            {
                matched.resize(matched.size() - static_cast<size_t>(n));
                trimmed = true;
                break;
            }

            // Check stale region: blocks after staleEnd should have pages.
            auto staleRange = attn->getStaleRange(numTok, mTokensPerBlock);
            auto staleEnd = staleRange.end;
            if (staleEnd < static_cast<int>(matched.size()))
            {
                auto tailBegin = matched.begin() + staleEnd;
                int nMissing = findIndex(std::make_reverse_iterator(matched.end()),
                    std::make_reverse_iterator(tailBegin), [&](auto const& m) { return !hasPage(*m.block, lcId); });
                if (static_cast<int>(matched.size()) - nMissing > staleEnd)
                {
                    matched.resize(matched.size() - static_cast<size_t>(nMissing));
                    trimmed = true;
                    break;
                }
            }
        }
        if (!trimmed)
            break;
    }

    int numTokens = getNumMatchedTokens();

    // --- Build blocks with stale range handling ---

    _resizePageIndexBuffers(static_cast<int>(matched.size()));

    for (size_t i = 0; i < matched.size(); ++i)
    {
        SeqBlock sb;
        bool isFullBlock = (matched[i].numMatchedTokens == mTokensPerBlock);
        sb.treeBlock = isFullBlock ? matched[i].block->shared_from_this() : nullptr;
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
            auto& blk = *matched[ordinal].block;
            auto page = blk.storage.at(static_cast<size_t>(lcId)).lock();
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
        auto& snapshotBlock = *matched.back().block;
        auto snapshotPage = snapshotBlock.storage[static_cast<size_t>(*ssmLcId)].lock();
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

std::shared_ptr<Block> const& KvCache::_getTreeBlock(BlockOrdinal ordinal) const
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
                auto committed = std::dynamic_pointer_cast<CommittedPage>(page);
                assert(committed && committed->block.lock() == ret);
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

    // Precompute stale ranges for each lifecycle.
    std::vector<HalfOpenRange> staleRanges(static_cast<size_t>(numLc));
    for (int lc = 0; lc < numLc; ++lc)
    {
        auto const& lifecycle = lcs.getLifeCycle(static_cast<LifeCycleId>(lc));
        staleRanges[lc] = _getStaleRange(mHistoryLength, lifecycle);
    }

    for (int ordinal = 0; ordinal < numBlocks(); ++ordinal)
    {
        auto const& block = mBlocks[static_cast<size_t>(ordinal)];
        bool isCommitted = ordinal < mNumCommittedBlocks;
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
                    assert(blockPageIsNull(bp));
                    continue;
                }

                auto const& staleRange = staleRanges[lc];
                if (staleRange.beg <= ordinal && ordinal < staleRange.end)
                {
                    if (isCommitted || mCommitState != CommitState::ALLOWED)
                    {
                        assert(blockPageIsNull(bp));
                    }
                    else
                    {
                        // For the decoder-side disagg case, for the first step, we will skip the
                        // out-of-window blocks.
                        assert(std::holds_alternative<std::shared_ptr<PageHolder>>(bp)
                            || (blockPageIsNull(bp) && mCommittedTokens.empty()));
                    }
                }
                else
                {
                    if (mStatus == Status::ACTIVE)
                        assert(std::holds_alternative<SharedPageLock>(bp));
                    else
                        assert(std::holds_alternative<std::shared_ptr<PageHolder>>(bp));
                }

                if (!blockPageIsNull(bp))
                {
                    auto page = blockPageGetPage(bp);
                    assert(isCommitted == (std::dynamic_pointer_cast<CommittedPage>(page) != nullptr));
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Page index tables
// ---------------------------------------------------------------------------

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
                assert(ext.len >= newNumBlocks);
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

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
