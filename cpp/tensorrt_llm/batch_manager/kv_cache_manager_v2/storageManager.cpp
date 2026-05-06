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

#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/copyEngine.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/utils/hostMem.h"
#include "kv_cache_manager_v2/utils/math.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <set>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// CacheLevelManager
// ---------------------------------------------------------------------------

CacheLevelManager::CacheLevelManager(std::vector<PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cl,
    CacheTierConfig const& tierConfig, StorageConfig const& storageConfig, std::vector<int> const& slotCountList)
    : cacheLevel(cl)
    , cacheTier(CacheTier(tierConfig.index()))
    , controller(lifeCycleGrouping, cl)
{
    storage = createCacheLevelStorage(tierConfig, storageConfig, slotCountList);
}

int CacheLevelManager::cacheTierGranularity(CacheTier tier, size_t quota)
{
    switch (tier)
    {
    case CacheTier::GPU_MEM:
    {
        constexpr size_t kPageSize = 2ULL << 20;
        return static_cast<int>(
            kPageSize << std::min(4, std::max(0, static_cast<int>(std::log2(quota / (kPageSize * 512))))));
    }
    case CacheTier::HOST_MEM: return static_cast<int>(HostMem::kAlignment); // 4 KiB
    case CacheTier::DISK: return 2 << 20; // DiskCacheLevelStorage::POOL_SIZE_GRANULARITY
    default: throw std::invalid_argument("Invalid cache tier");
    }
}

// ---------------------------------------------------------------------------
// StorageManager constructor helpers
// ---------------------------------------------------------------------------

namespace
{

// Compute the slot-to-page-indices scale factors.
// For each (lcId, poolIdx), scale = numBuffersInCoalescedSlot.
// Python: _slot_to_page_indices[lc_id][pool_idx] = numBuffers
std::vector<std::vector<int>> computeSlotToPageIndices(StorageConfig const& config)
{
    int numLc = config.numLifeCycles();
    std::vector<std::vector<int>> result(static_cast<size_t>(numLc));

    auto const& slotDescList = config.slotDescList;
    auto const& grouping = config.lifeCycleGrouping();

    for (int lc = 0; lc < numLc; ++lc)
    {
        PoolGroupIndex pgIdx = grouping[lc];
        SlotDesc const& sd = slotDescList.at(static_cast<size_t>(pgIdx));
        // Find the variant that corresponds to this lifecycle.
        for (auto const& variant : sd.variants)
        {
            if (variant.lifeCycleId == lc)
            {
                // Each coalesced buffer contributes its numBuffers as the scale.
                result[static_cast<size_t>(lc)].reserve(variant.coalescedBuffers.size());
                for (auto const& cb : variant.coalescedBuffers)
                    result[static_cast<size_t>(lc)].push_back(cb.numBuffers());
                break;
            }
        }
        if (result[static_cast<size_t>(lc)].empty())
            result[static_cast<size_t>(lc)].assign(1, 1); // fallback
    }
    return result;
}

} // namespace

// ---------------------------------------------------------------------------
// StorageManager
// ---------------------------------------------------------------------------

StorageManager::StorageManager(LifeCycleRegistry const& lifeCycles, StorageConfig const& config, int tokensPerBlock,
    std::optional<BatchDesc> const& typicalBatch, std::vector<BatchDesc> const& constraints)
    : mLifeCycles(lifeCycles)
    , mStorageConfig(config)
{
    mLifeCycleGrouping = config.lifeCycleGrouping();
    mLayerToLifeCycleIds = config.layerToLifeCycleIds();
    mSlotToPageIndices = computeSlotToPageIndices(config);
    mBufferAttr = config.bufferAttributes();
    mSlotDescList = config.slotDescList;

    assert(std::all_of(mLifeCycleGrouping.begin(), mLifeCycleGrouping.end(),
        [this](PoolGroupIndex pg) { return pg < numPoolGroups(); }));
    assert(numPoolGroups()
        == static_cast<int>(std::set<PoolGroupIndex>(mLifeCycleGrouping.begin(), mLifeCycleGrouping.end()).size()));

    // Build one CacheLevelManager per tier.
    assert(!config.cacheTiers.empty());
    assert(std::holds_alternative<GpuCacheTierConfig>(config.cacheTiers[0]) && "First cache tier must be GPU");

    // Compute slot size lists for all pool groups.
    std::vector<std::vector<int>> slotSizeLists;
    slotSizeLists.reserve(mSlotDescList.size());
    for (auto const& sd : mSlotDescList)
    {
        auto sizeList = sd.slotSizeList();
        slotSizeLists.emplace_back(sizeList.begin(), sizeList.end());
    }

    size_t gpuQuota = cacheTierQuota(config.cacheTiers[0]);
    int gpuGranularity = CacheLevelManager::cacheTierGranularity(CacheTier::GPU_MEM, gpuQuota);

    // Compute min_slots from constraints.
    mMinSlots = computeMinSlotsFromConstraints(constraints, tokensPerBlock);

    // Compute init_ratio from typical_batch, constraints, or fallback.
    std::vector<float> initRatio;
    if (typicalBatch.has_value())
    {
        initRatio = ratioFromBatch(*typicalBatch, tokensPerBlock, gpuGranularity);
    }
    else if (!constraints.empty())
    {
        // Use the constraint slot counts as the ratio basis.
        auto minBytes = slotsToBytes(mMinSlots, gpuGranularity);
        initRatio = normalizeToRatio(minBytes);
    }
    else
    {
        // Fallback: average history length 2048.
        BatchDesc fallback;
        fallback.kvCaches.push_back(KVCacheDesc{2049, 2048});
        initRatio = ratioFromBatch(fallback, tokensPerBlock, gpuGranularity);
    }

    int numLevels = static_cast<int>(config.cacheTiers.size());
    mLevels.reserve(config.cacheTiers.size());
    for (int i = 0; i < numLevels; ++i)
    {
        auto slotCountList = computeSlotCountForLevel(config.cacheTiers[i], slotSizeLists, initRatio);
        mLevels.emplace_back(
            mLifeCycleGrouping, static_cast<CacheLevel>(i), config.cacheTiers[i], config, slotCountList);
    }

    assert(mLevels.empty()
        || numPoolGroups()
            == getUniformAttribute(mLevels, [](auto const& lvl) { return lvl.storage->numPoolGroups(); }));
}

StorageManager::~StorageManager()
{
    destroy();
}

void StorageManager::destroy()
{
    for (auto& lvl : mLevels)
        if (lvl.storage)
            lvl.storage->destroy();
    mLevels.clear();
}

// ---------------------------------------------------------------------------
// newSlots
// ---------------------------------------------------------------------------

std::vector<std::vector<Slot>> StorageManager::newSlots(CacheLevel level, std::vector<int> const& numSlotsPerLc)
{
    assert(static_cast<int>(numSlotsPerLc.size()) == numLifeCycles());
    auto& storage = *mLevels.at(static_cast<size_t>(level)).storage;

    // Aggregate by pool group.
    std::vector<int> pgNumSlots(static_cast<size_t>(numPoolGroups()), 0);
    for (int lc = 0; lc < numLifeCycles(); ++lc)
        pgNumSlots[static_cast<size_t>(mLifeCycleGrouping[lc])] += numSlotsPerLc[lc];

    // Prepare free slots if needed.
    bool needMore = false;
    for (int pg = 0; pg < numPoolGroups(); ++pg)
        if (pgNumSlots[pg] > storage.numFreeSlots(static_cast<PoolGroupIndex>(pg)))
        {
            needMore = true;
            break;
        }

    if (needMore)
        prepareFreeSlots(level, pgNumSlots);

    // A14: post-condition — free-slot counts satisfy requirements.
    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        assert(pgNumSlots[pg] <= storage.numFreeSlots(static_cast<PoolGroupIndex>(pg))
            && "Free slot count does not satisfy requirement after prepareFreeSlots");
    }

    // Allocate.
    std::vector<std::vector<Slot>> ret(static_cast<size_t>(numLifeCycles()));
    try
    {
        for (int lc = 0; lc < numLifeCycles(); ++lc)
        {
            PoolGroupIndex pg = mLifeCycleGrouping[lc];
            ret[lc] = storage.allocateMultiple(pg, numSlotsPerLc[lc]);
        }
    }
    catch (...)
    {
        for (int lc = 0; lc < numLifeCycles(); ++lc)
        {
            PoolGroupIndex pg = mLifeCycleGrouping[lc];
            for (auto& s : ret[lc])
                storage.release(pg, std::move(s));
        }
        throw;
    }
    return ret;
}

std::vector<std::vector<Slot>> StorageManager::newGpuSlots(std::vector<int> const& numSlotsPerLc)
{
    return newSlots(kGpuLevel, numSlotsPerLc);
}

std::vector<Slot> StorageManager::newSlotsForPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, int numSlots)
{
    auto& storage = *mLevels.at(static_cast<size_t>(level)).storage;
    if (numSlots > storage.numFreeSlots(pgIdx))
    {
        std::vector<int> requirements(static_cast<size_t>(numPoolGroups()), 0);
        requirements.at(static_cast<size_t>(pgIdx)) = numSlots;
        prepareFreeSlots(level, requirements);
    }
    assert(numSlots <= storage.numFreeSlots(pgIdx));
    return storage.allocateMultiple(pgIdx, numSlots);
}

Address StorageManager::slotAddress(CacheLevel level, PoolGroupIndex pgIdx, SlotId slotId, int poolIdx) const
{
    return mLevels.at(static_cast<size_t>(level)).storage->slotAddress(pgIdx, slotId).at(static_cast<size_t>(poolIdx));
}

CacheTier StorageManager::cacheTier(CacheLevel level) const
{
    return mLevels.at(static_cast<size_t>(level)).cacheTier;
}

void StorageManager::releaseSlot(LifeCycleId lc, CacheLevel level, Slot slot)
{
    PoolGroupIndex pg = mLifeCycleGrouping.at(static_cast<size_t>(lc));
    mLevels.at(static_cast<size_t>(level)).storage->release(pg, std::move(slot));
}

// ---------------------------------------------------------------------------
// isEvictable
// ---------------------------------------------------------------------------

bool StorageManager::isEvictable(Page const& page, std::optional<CacheLevel> level) const noexcept
{
    PageStatus s = page.status();
    CacheLevel lvl = level.value_or(page.cacheLevel);
    return (s == PageStatus::DROPPABLE && page.isCommitted())
        || (s == PageStatus::HELD && lvl < static_cast<CacheLevel>(numCacheLevels() - 1));
}

// ---------------------------------------------------------------------------
// scheduleForEviction / excludeFromEviction
// ---------------------------------------------------------------------------

void StorageManager::scheduleForEviction(Page& page)
{
    if (isEvictable(page))
        mLevels.at(static_cast<size_t>(page.cacheLevel)).controller.scheduleForEviction(page);
}

void StorageManager::excludeFromEviction(Page& page)
{
    assert(page.nodeRef.has_value());
    mLevels.at(static_cast<size_t>(page.cacheLevel)).controller.remove(*page.nodeRef);
}

// ---------------------------------------------------------------------------
// prepareFreeSlots
// ---------------------------------------------------------------------------

void StorageManager::prepareFreeSlots(CacheLevel level, std::vector<int> const& requirements)
{
    // goals[level][pgIdx] = how many free slots we need at `level` for `pgIdx`.
    std::vector<std::vector<int>> goals(
        static_cast<size_t>(numCacheLevels()), std::vector<int>(static_cast<size_t>(numPoolGroups()), 0));
    for (int pg = 0; pg < numPoolGroups(); ++pg)
        goals.at(static_cast<size_t>(level)).at(static_cast<size_t>(pg)) = requirements.at(static_cast<size_t>(pg));

    std::vector<std::vector<std::shared_ptr<Page>>> fallenPages(static_cast<size_t>(numPoolGroups()));
    _prepareFreeSlots(goals, level, fallenPages);
}

void StorageManager::forceEvict(CacheLevel level, std::vector<int> const& minNumPages)
{
    auto evicted = mLevels.at(static_cast<size_t>(level)).controller.evict(minNumPages);

    if (isLastLevel(level))
    {
        // Last level: all evicted pages must be DROPPABLE (they get dropped, not migrated).
        for (auto const& pages : evicted)
        {
            for (auto const& page : pages)
            {
                assert(page->status() == PageStatus::DROPPABLE && "Corrupted eviction controller");
            }
        }
        return;
    }

    std::vector<std::vector<int>> goals(
        static_cast<size_t>(numCacheLevels()), std::vector<int>(static_cast<size_t>(numPoolGroups()), 0));
    CacheLevel nextLvl = static_cast<CacheLevel>(level + 1);

    std::vector<std::vector<std::shared_ptr<Page>>> fallen(static_cast<size_t>(numPoolGroups()));
    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        for (auto& sp : evicted.at(static_cast<size_t>(pg)))
            fallen.at(static_cast<size_t>(pg)).push_back(sp);
    }
    _prepareFreeSlots(goals, nextLvl, fallen);
}

// ---------------------------------------------------------------------------
// _prepareFreeSlots (recursive)
// ---------------------------------------------------------------------------

void StorageManager::_prepareFreeSlots(std::vector<std::vector<int>>& goals, CacheLevel lvlId,
    std::vector<std::vector<std::shared_ptr<Page>>>& fallenPages)
{
    // A7: goals dimensions must match [numCacheLevels][numPoolGroups].
    if (!gNdebug)
    {
        assert(static_cast<int>(goals.size()) == numCacheLevels() && "goals.rows must equal numCacheLevels");
        for (auto const& row : goals)
        {
            assert(static_cast<int>(row.size()) == numPoolGroups() && "goals.cols must equal numPoolGroups");
        }
    }

    // A8: all fallen pages must come from upper cache levels (cache_level < lvlId).
    if (!gNdebug)
    {
        for (auto const& pages : fallenPages)
        {
            for (auto const& p : pages)
            {
                assert(p->cacheLevel < lvlId && "Fallen pages must come from upper cache levels");
            }
        }
    }

    auto& lvl = mLevels.at(static_cast<size_t>(lvlId));
    auto& storage = *lvl.storage;
    auto& ctrl = lvl.controller;
    bool isLast = isLastLevel(lvlId);

    std::vector<int> numToEvict(static_cast<size_t>(numPoolGroups()), 0);
    std::vector<std::vector<std::shared_ptr<Page>>> heldPages(static_cast<size_t>(numPoolGroups()));

    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        int goal = goals.at(static_cast<size_t>(lvlId)).at(static_cast<size_t>(pg));
        int fallen = static_cast<int>(fallenPages.at(static_cast<size_t>(pg)).size());
        int oldFree = storage.numFreeSlots(static_cast<PoolGroupIndex>(pg));
        int evictable = ctrl.numEvictablePages(static_cast<PoolGroupIndex>(pg));
        numToEvict.at(static_cast<size_t>(pg)) = std::max(0, std::min(goal + fallen - oldFree, evictable));

        int fallenHeld = 0;
        if (isLast)
        {
            // Separate held pages from fallen_pages (mirrors Python's remove_if).
            auto& fp = fallenPages.at(static_cast<size_t>(pg));
            heldPages.at(static_cast<size_t>(pg))
                = stealIf(fp, [](std::shared_ptr<Page> const& p) { return p->status() == PageStatus::HELD; });
            fallenHeld = static_cast<int>(heldPages.at(static_cast<size_t>(pg)).size());

            if (fallenHeld > oldFree + evictable)
                throw OutOfPagesError(
                    "Too many held pages falling to last-level cache for group " + std::to_string(pg));
        }

        if (oldFree + evictable - fallenHeld < goal)
            throw OutOfPagesError(
                "Impossible to meet free-slot goal " + std::to_string(goal) + " for group " + std::to_string(pg));
    }

    auto evicted = ctrl.evict(numToEvict);
    std::vector<std::vector<std::shared_ptr<Page>>> acceptedPages(static_cast<size_t>(numPoolGroups()));

    if (isLast)
    {
        for (int pg = 0; pg < numPoolGroups(); ++pg)
        {
            auto& ev = evicted.at(static_cast<size_t>(pg));
            int oldFree = storage.numFreeSlots(static_cast<PoolGroupIndex>(pg));
            int numEvicted = static_cast<int>(ev.size());
            // A9: all evicted pages at last level must be DROPPABLE.
            if (!gNdebug)
            {
                for (auto const& p : ev)
                {
                    assert(p->status() == PageStatus::DROPPABLE && "Evicted page at last level must be DROPPABLE");
                }
            }
            // Drop droppable evicted pages (GC).
            ev.clear();
            int newFree = storage.numFreeSlots(static_cast<PoolGroupIndex>(pg));
            assert(newFree >= numEvicted + oldFree);

            // A10: held_pages count must not exceed new_free.
            assert(static_cast<int>(heldPages.at(static_cast<size_t>(pg)).size()) <= newFree
                && "held_pages count exceeds new free slot count");

            // Add held pages from upper levels.
            auto& hp = heldPages.at(static_cast<size_t>(pg));
            auto& fp = fallenPages.at(static_cast<size_t>(pg));
            fp.insert(fp.end(), hp.begin(), hp.end());
            hp.clear();

            int goal = goals.at(static_cast<size_t>(lvlId)).at(static_cast<size_t>(pg));
            int numAccepted = std::min(newFree - goal, static_cast<int>(fp.size()));
            // A11: numAccepted must be non-negative (last-level path).
            assert(numAccepted >= 0 && "numAccepted must be >= 0");
            if (numAccepted > 0)
            {
                acceptedPages.at(static_cast<size_t>(pg)).assign(fp.end() - numAccepted, fp.end());
            }
            fp.clear();
        }
    }
    else
    {
        // A12: no held pages at non-last level.
        if (!gNdebug)
        {
            for (auto const& hp : heldPages)
            {
                assert(hp.empty() && "held_pages must be empty at non-last level");
            }
        }

        CacheLevel nextLvl = static_cast<CacheLevel>(lvlId + 1);
        for (int pg = 0; pg < numPoolGroups(); ++pg)
        {
            auto& ev = evicted.at(static_cast<size_t>(pg));
            int oldFree = storage.numFreeSlots(static_cast<PoolGroupIndex>(pg));
            int numEvicted = static_cast<int>(ev.size());
            auto& fp = fallenPages.at(static_cast<size_t>(pg));
            fp.insert(fp.begin(), ev.begin(), ev.end()); // prepend evicted to fallen (preserving order)
            ev.clear();

            int goal = goals.at(static_cast<size_t>(lvlId)).at(static_cast<size_t>(pg));
            int numAccepted = std::min(oldFree + numEvicted - goal, static_cast<int>(fp.size()));
            // A11: numAccepted must be non-negative (non-last-level path).
            assert(numAccepted >= 0 && "numAccepted must be >= 0");
            if (numAccepted > 0)
            {
                acceptedPages.at(static_cast<size_t>(pg)).assign(fp.end() - numAccepted, fp.end());
                fp.erase(fp.end() - numAccepted, fp.end());
            }
        }
        _prepareFreeSlots(goals, nextLvl, fallenPages);
    }

    // A13: all fallen pages must have been consumed.
    if (!gNdebug)
    {
        for (auto const& fp : fallenPages)
        {
            assert(fp.empty() && "All fallen pages must be consumed after level loop");
        }
    }

    // Migrate accepted pages into lvlId.
    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        // Group by source level (mirrors Python's partition()).
        auto bySrcLevel = partition(
            acceptedPages.at(static_cast<size_t>(pg)), [](std::shared_ptr<Page> const& p) { return p->cacheLevel; });

        for (auto& [srcLvl, pages] : bySrcLevel)
        {
            _batchedMigrate(static_cast<PoolGroupIndex>(pg), lvlId, srcLvl, pages, /*updateSrc=*/true);
            for (auto const& p : pages)
            {
                if (isLast && p->status() == PageStatus::HELD)
                    continue;
                lvl.controller.scheduleForEviction(*p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// _batchedMigrate
// ---------------------------------------------------------------------------

void StorageManager::_batchedMigrate(PoolGroupIndex pgIdx, CacheLevel dstLevel, CacheLevel srcLevel,
    std::vector<std::shared_ptr<Page>> const& srcPages, bool updateSrc, bool defrag)
{
    assert(defrag || dstLevel != srcLevel);
    int numSlots = static_cast<int>(srcPages.size());
    if (numSlots == 0)
        return;

    auto& srcPoolGroup = poolGroup(srcLevel, pgIdx);
    auto& dstPoolGroup = poolGroup(dstLevel, pgIdx);

    if (dstPoolGroup.numFreeSlots() < numSlots)
        throw OutOfPagesError("Not enough free slots for migration");

    auto dstSlots = dstPoolGroup.allocateMultiple(numSlots);
    // A15: allocated slot count must match the request.
    assert(static_cast<int>(dstSlots.size()) == numSlots && "dst_slots size mismatch");
    try
    {
        CacheTier dstTier = mLevels.at(static_cast<size_t>(dstLevel)).cacheTier;
        CacheTier srcTier = mLevels.at(static_cast<size_t>(srcLevel)).cacheTier;

        int numPools = mNumPools(pgIdx);

        // Build copy tasks per pool.
        std::vector<std::vector<CopyTask>> tasksPerPool(static_cast<size_t>(numPools));
        for (int i = 0; i < numSlots; ++i)
        {
            auto const& src = srcPages.at(static_cast<size_t>(i));
            auto const& dst = dstSlots.at(static_cast<size_t>(i));
            // Fix #8: assert non-defrag migrations only accept pages not scheduled for eviction.
            assert(defrag || !src->scheduledForEviction());
            for (int pi = 0; pi < numPools; ++pi)
            {
                Address dstAddr = dstPoolGroup.slotAddress(dst.slotId()).at(static_cast<size_t>(pi));
                Address srcAddr = srcPoolGroup.slotAddress(src->slotId()).at(static_cast<size_t>(pi));
                tasksPerPool.at(static_cast<size_t>(pi)).push_back({dstAddr, srcAddr});
            }
        }

        // Collect prior events (src + dst ready events) — mirrors Python's prior_events set.
        std::vector<CachedCudaEvent const*> priorEvents;
        priorEvents.reserve(static_cast<size_t>(2 * numSlots));
        for (int i = 0; i < numSlots; ++i)
        {
            priorEvents.push_back(&srcPages.at(static_cast<size_t>(i))->readyEvent);
            priorEvents.push_back(&dstSlots.at(static_cast<size_t>(i)).readyEvent);
        }

        // Create a temporary CUDA stream that waits for all prior events before copying.
        TemporaryCudaStream tempStream(priorEvents);
        {
            auto scope = tempStream.enter();
            CUstream stream = tempStream.get();
            auto slotSizes = slotSize(pgIdx);
            for (int pi = 0; pi < numPools; ++pi)
            {
                batchedCopy(dstTier, srcTier, static_cast<size_t>(slotSizes.at(static_cast<size_t>(pi))),
                    tasksPerPool.at(static_cast<size_t>(pi)), stream);
            }
        } // ~Scope records finish event

        CachedCudaEvent finishEvent = tempStream.takeFinishEvent();
        for (int i = 0; i < numSlots; ++i)
        {
            dstSlots.at(static_cast<size_t>(i)).readyEvent = finishEvent;
            // Fix #6: set src.ready_event unconditionally — compulsory for the next owner
            // getting this slot from the pool. Mirrors Python: `src.ready_event = finish_event`.
            srcPages.at(static_cast<size_t>(i))->readyEvent = finishEvent;
            if (updateSrc)
            {
                bool wasScheduled = srcPages.at(static_cast<size_t>(i))->scheduledForEviction();
                if (wasScheduled)
                    excludeFromEviction(*srcPages.at(static_cast<size_t>(i)));
                // Extract source slot from the page and release it back to the pool.
                Slot srcSlot;
                srcSlot.setSlotId(srcPages.at(static_cast<size_t>(i))->slotId()); // asserts valid
                srcSlot.readyEvent = finishEvent;
                srcPages.at(static_cast<size_t>(i))->resetSlot();
                srcPoolGroup.release(std::move(srcSlot));
                // Transfer dst slot ownership to the page.
                srcPages.at(static_cast<size_t>(i))->setSlot(dstSlots.at(static_cast<size_t>(i)));
                srcPages.at(static_cast<size_t>(i))->cacheLevel = dstLevel;
                if (wasScheduled)
                    scheduleForEviction(*srcPages.at(static_cast<size_t>(i)));
            }
        }
    }
    catch (...)
    {
        for (auto& s : dstSlots)
            dstPoolGroup.release(std::move(s));
        throw;
    }
}

// ---------------------------------------------------------------------------
// batchedMigrateToGpu
// ---------------------------------------------------------------------------

void StorageManager::batchedMigrateToGpu(std::vector<BatchedLockTarget> const& targets, KvCache& /*kvCache*/)
{
    // Group by (srcLevel, pgIdx).
    std::map<std::pair<CacheLevel, PoolGroupIndex>, std::vector<std::shared_ptr<Page>>> groups;
    for (auto const& t : targets)
    {
        if (t.page->cacheLevel == kGpuLevel)
            continue;
        PoolGroupIndex pg = mLifeCycleGrouping.at(static_cast<size_t>(t.lifeCycle));
        groups[{t.page->cacheLevel, pg}].push_back(t.page);
    }
    for (auto& [key, pages] : groups)
        _batchedMigrate(key.second, kGpuLevel, key.first, pages, /*updateSrc=*/true);
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

LifeCycle const& StorageManager::getLifeCycle(LifeCycleId lc) const
{
    return mLifeCycles[lc];
}

int StorageManager::getPoolGroupIndex(LifeCycleId lc) const
{
    return mLifeCycleGrouping.at(static_cast<size_t>(lc));
}

int StorageManager::mNumPools(PoolGroupIndex pgIdx) const
{
    if (mLevels.empty())
        return 0;
    return getUniformAttribute(mLevels, [pgIdx](auto const& lvl) { return lvl.storage->numPools(pgIdx); });
}

int StorageManager::numPools(PoolGroupIndex pgIdx) const
{
    return mNumPools(pgIdx);
}

std::vector<int> StorageManager::slotSize(PoolGroupIndex pgIdx) const
{
    auto szList = mSlotDescList.at(static_cast<size_t>(pgIdx)).slotSizeList();
    std::vector<int> result;
    result.reserve(szList.size());
    for (auto s : szList)
        result.push_back(static_cast<int>(s));
    return result;
}

PoolGroupBase& StorageManager::poolGroup(CacheLevel lvl, PoolGroupIndex pgIdx)
{
    return mLevels.at(static_cast<size_t>(lvl)).storage->poolGroup(pgIdx);
}

MemAddress StorageManager::getMemPoolBaseAddress(LayerId layerId, DataRole role) const
{
    auto it = mBufferAttr.find(BufferId{layerId, role});
    if (it == mBufferAttr.end())
        throw std::out_of_range("Unknown BufferId");
    auto const& attr = it->second;
    PoolGroupIndex pgIdx = mLifeCycleGrouping.at(static_cast<size_t>(attr.lifeCycleId));
    return mLevels[0].storage->getBaseAddress(pgIdx, attr.poolIndex, SlotId(0)) + attr.offset;
}

int StorageManager::numSlots(PoolGroupIndex pgIdx, CacheLevel level) const
{
    return mLevels.at(static_cast<size_t>(level)).storage->numSlots(pgIdx);
}

StorageStatistics StorageManager::getStatistics(CacheLevel level, PoolGroupIndex pgIdx) const
{
    auto const& lvl = mLevels.at(static_cast<size_t>(level));
    int freeSlots = lvl.storage->numFreeSlots(pgIdx);
    int totalSlots = lvl.storage->numSlots(pgIdx);
    int evictable = lvl.controller.numEvictablePages(pgIdx);
    auto sizes = lvl.storage->slotSize(pgIdx);
    return StorageStatistics{sizes, totalSlots, freeSlots, evictable};
}

std::vector<float> StorageManager::getUtilization(CacheLevel level) const
{
    std::vector<float> result;
    result.reserve(static_cast<size_t>(numPoolGroups()));
    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        auto const s = getStatistics(level, static_cast<PoolGroupIndex>(pg));
        result.push_back(s.total > 0 ? static_cast<float>(s.unavailable()) / static_cast<float>(s.total) : 0.f);
    }
    return result;
}

float StorageManager::getOverallUtilization(CacheLevel level) const
{
    float num = 0.f, den = 0.f;
    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        auto s = getStatistics(level, static_cast<PoolGroupIndex>(pg));
        float sz = 0.f;
        for (auto v : s.slotSizes)
            sz += static_cast<float>(v);
        num += sz * static_cast<float>(s.unavailable());
        den += sz * static_cast<float>(s.total);
    }
    return den > 0.f ? num / den : 0.f;
}

// ---------------------------------------------------------------------------
// expandPoolGroup
// ---------------------------------------------------------------------------

void StorageManager::expandPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, int newNumSlots)
{
    auto& pg = poolGroup(level, pgIdx);
    assert(newNumSlots > pg.numSlots());
    pg.resizePools(newNumSlots);
    pg.slotAllocator().expand(newNumSlots);
}

// ---------------------------------------------------------------------------
// shrinkPoolGroup — mirrors Python _storage_manager.py::shrink_pool_group
// ---------------------------------------------------------------------------

void StorageManager::shrinkPoolGroup(
    CacheLevel level, PoolGroupIndex pgIdx, int newNumSlots, std::vector<std::shared_ptr<Page>> const& persistentPages)
{
    auto& pg = poolGroup(level, pgIdx);
    auto& allocator = pg.slotAllocator();
    auto& ctrl = mLevels.at(static_cast<size_t>(level)).controller;
    assert(newNumSlots < pg.numSlots());

    // A16: persistent_pages preconditions.
    assert(static_cast<int>(persistentPages.size()) <= newNumSlots && "Not enough slots to hold all persistent pages");
    if (!gNdebug)
    {
        for (auto const& p : persistentPages)
        {
            assert(p->cacheLevel == level && "Persistent page cache level mismatch");
            assert(mLifeCycleGrouping.at(static_cast<size_t>(p->lifeCycle)) == pgIdx
                && "Persistent page pool group mismatch");
        }
    }

    // Find overflow pages: scheduled pages with slot_id >= newNumSlots.
    auto gen = ctrl.pageGenerator(pgIdx);
    std::deque<std::pair<int, std::shared_ptr<Page>>> overflowSlots;
    {
        int idx = 0;
        while (auto const* page = gen())
        {
            if ((*page)->slotId() >= newNumSlots)
                overflowSlots.emplace_back(idx, *page);
            ++idx;
        }
    }

    // Persistent pages in overflow range.
    std::vector<std::shared_ptr<Page>> overflowPersistent;
    for (auto const& p : persistentPages)
    {
        if (p->slotId() >= newNumSlots)
            overflowPersistent.push_back(p);
    }
    int numOverflowPersistent = static_cast<int>(overflowPersistent.size());

    // A2: RUNTIME check — persistent overflow pages must fit in the new capacity.
    if (numOverflowPersistent > newNumSlots)
    {
        throw OutOfPagesError("Not enough slots to hold all persistent pages");
    }

    // Mark the allocator for shrink.
    allocator.prepareForShrink(newNumSlots);

    // Calculate minimum number of lowest-priority pages to evict.
    // Need numEvictedOverflowSlots because evicted overflow pages won't become free,
    // because only free non-overflow slots can be used for defragmentation.
    int minNumEvicted = 0;
    int numEvictedOverflowSlots = 0;
    while (!overflowSlots.empty()
        && static_cast<int>(overflowSlots.size()) + numOverflowPersistent
            > std::min(newNumSlots, overflowSlots.front().first + allocator.numFreeSlots() - numEvictedOverflowSlots))
    {
        minNumEvicted = overflowSlots.front().first + 1;
        overflowSlots.pop_front();
        ++numEvictedOverflowSlots;
    }

    // Force-evict the required pages.
    std::vector<int> evictReqs(static_cast<size_t>(numPoolGroups()), 0);
    evictReqs[static_cast<size_t>(pgIdx)] = minNumEvicted;
    forceEvict(level, evictReqs);

    // Remaining overflow pages to defragment.
    std::vector<std::shared_ptr<Page>> overflowPages;
    overflowPages.reserve(overflowSlots.size() + overflowPersistent.size());
    for (auto& [idx, p] : overflowSlots)
        overflowPages.push_back(p);
    for (auto& p : overflowPersistent)
        overflowPages.push_back(p);

    // Ensure free slots for the overflow pages.
    std::vector<int> reqs(static_cast<size_t>(numPoolGroups()), 0);
    reqs[static_cast<size_t>(pgIdx)] = static_cast<int>(overflowPages.size());
    prepareFreeSlots(level, reqs);

    // A17: all overflow pages must be at the expected cache level.
    if (!gNdebug)
    {
        for (auto const& p : overflowPages)
        {
            assert(p->cacheLevel == level && "Overflow page cache level mismatch");
        }
    }

    // Defragment: migrate overflow pages to free slots within the same level.
    _batchedMigrate(pgIdx, level, level, overflowPages, /*updateSrc=*/true, /*defrag=*/true);

    // A18: post-defrag overflow assertion — overflow slot count matches expectations.
    assert(allocator.numOverflowSlots() == allocator.numActiveSlots() - allocator.targetCapacity()
        && "Post-defrag overflow slot count mismatch");

    // Finalize shrink and resize pools.
    allocator.finishShrink();
    pg.resizePools(newNumSlots);
}

// ---------------------------------------------------------------------------
// adjustCacheLevel — mirrors Python _storage_manager.py::adjust_cache_level
// ---------------------------------------------------------------------------

void StorageManager::adjustCacheLevel(CacheLevel level, std::optional<size_t> newQuota,
    std::vector<float> const& ratioList, std::vector<std::vector<std::shared_ptr<Page>>> const* persistentPages)
{
    auto& lvlStorage = *mLevels.at(static_cast<size_t>(level)).storage;
    auto oldNumSlots = lvlStorage.slotCountList();
    size_t quota = newQuota.has_value()
        ? roundUp(newQuota.value(), static_cast<size_t>(lvlStorage.poolSizeGranularity()))
        : lvlStorage.totalQuota();
    size_t minQuota = minQuotaForLevel(lvlStorage.slotSizeLists(), lvlStorage.poolSizeGranularity());
    if (quota < minQuota)
    {
        throw std::invalid_argument("Quota " + std::to_string(quota)
            + " is insufficient for min_slots constraints (requires at least " + std::to_string(minQuota) + ")");
    }
    auto newNumSlots = lvlStorage.computeSlotCountList(ratioList, mMinSlots, quota);

    if (!isLastLevel(level))
        assert(persistentPages == nullptr);

    // Shrink first.
    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        if (newNumSlots[pg] >= oldNumSlots[pg])
            continue;
        std::vector<std::shared_ptr<Page>> pages;
        if (persistentPages)
            pages = (*persistentPages)[pg];
        shrinkPoolGroup(level, static_cast<PoolGroupIndex>(pg), newNumSlots[pg], pages);
    }
    // Then expand.
    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        if (newNumSlots[pg] <= oldNumSlots[pg])
            continue;
        expandPoolGroup(level, static_cast<PoolGroupIndex>(pg), newNumSlots[pg]);
    }
    lvlStorage.postResize();
}

std::vector<float> StorageManager::getRatioList(CacheLevel level) const
{
    return mLevels.at(static_cast<size_t>(level)).storage->ratioList();
}

std::vector<float> StorageManager::ratioFromLength(int tokensPerBlock, int historyLength, int capacity) const
{
    int numBlocks = divUp(capacity, tokensPerBlock);
    std::vector<size_t> numBytes(static_cast<size_t>(numPoolGroups()), 0);
    auto ssmLcId = mLifeCycles.ssmLifeCycleId();
    auto const& lifecycles = mLifeCycles.getAll();
    for (int lcIdx = 0; lcIdx < numLifeCycles(); ++lcIdx)
    {
        PoolGroupIndex pgIdx = mLifeCycleGrouping[lcIdx];
        auto ss = slotSize(pgIdx);
        int slotSizeSum = 0;
        for (auto s : ss)
            slotSizeSum += s;
        int numRequiredBlocks;
        if (ssmLcId.has_value() && lcIdx == *ssmLcId)
        {
            numRequiredBlocks = 1;
        }
        else
        {
            auto stale = getStaleRange(lifecycles[lcIdx], historyLength, tokensPerBlock);
            numRequiredBlocks = numBlocks - stale.length();
        }
        numBytes[static_cast<size_t>(pgIdx)]
            += static_cast<size_t>(numRequiredBlocks) * static_cast<size_t>(slotSizeSum);
    }
    return normalizeToRatio(numBytes);
}

// ---------------------------------------------------------------------------
// ratioFromBatch
// ---------------------------------------------------------------------------

std::vector<float> StorageManager::ratioFromBatch(BatchDesc const& batch, int tokensPerBlock, int granularity) const
{
    auto numSlots = computeSlotsForBatch(batch, tokensPerBlock);
    auto numBytes = slotsToBytes(numSlots, granularity);
    return normalizeToRatio(numBytes);
}

// ---------------------------------------------------------------------------
// computeMinSlotsFromConstraints
// ---------------------------------------------------------------------------

std::vector<int> StorageManager::computeMinSlotsFromConstraints(
    std::vector<BatchDesc> const& constraints, int tokensPerBlock) const
{
    // Default floor: 1 slot per life cycle in each pool group.
    std::vector<int> maxSlots(static_cast<size_t>(numPoolGroups()), 0);
    for (auto pgIdx : mLifeCycleGrouping)
        maxSlots[static_cast<size_t>(pgIdx)] += 1;
    for (auto const& batch : constraints)
    {
        auto slots = computeSlotsForBatch(batch, tokensPerBlock);
        for (int pg = 0; pg < numPoolGroups(); ++pg)
            maxSlots[pg] = std::max(maxSlots[pg], slots[pg]);
    }
    return maxSlots;
}

// ---------------------------------------------------------------------------
// computeSlotsForBatch
// ---------------------------------------------------------------------------

std::vector<int> StorageManager::computeSlotsForBatch(BatchDesc const& batch, int tokensPerBlock) const
{
    std::vector<int> numSlots(static_cast<size_t>(numPoolGroups()), 0);
    auto ssmLcId = mLifeCycles.ssmLifeCycleId();
    int sysBlocks = batch.systemPromptLength / tokensPerBlock;

    for (auto const& [lcIdx, lc] : mLifeCycles)
    {
        PoolGroupIndex pgIdx = mLifeCycleGrouping[lcIdx];
        if (ssmLcId.has_value() && lcIdx == *ssmLcId)
        {
            // SSM: always 1 dedicated block per request, never shared.
            numSlots[static_cast<size_t>(pgIdx)] += static_cast<int>(batch.kvCaches.size());
            continue;
        }
        // Shared sys blocks (counted once): union of non-stale sys blocks across all requests.
        HalfOpenRange sysRange{0, sysBlocks};
        HalfOpenRange staleIntersection = sysRange;
        for (auto const& kv : batch.kvCaches)
        {
            auto stale = getStaleRange(lc, kv.historyLength, tokensPerBlock);
            staleIntersection = intersect(staleIntersection, stale);
        }
        numSlots[static_cast<size_t>(pgIdx)] += sysBlocks - staleIntersection.length();

        // Per-request unique blocks (excluding shared sys blocks already counted above).
        for (auto const& kv : batch.kvCaches)
        {
            int totalBlocks = divUp(kv.capacity, tokensPerBlock);
            auto stale = getStaleRange(lc, kv.historyLength, tokensPerBlock);
            int nonStale = totalBlocks - stale.length();
            int nonStaleSys = sysBlocks - intersect(stale, sysRange).length();
            numSlots[static_cast<size_t>(pgIdx)] += std::max(0, nonStale - nonStaleSys);
        }
    }
    return numSlots;
}

// ---------------------------------------------------------------------------
// slotsToBytes
// ---------------------------------------------------------------------------

std::vector<size_t> StorageManager::slotsToBytes(std::vector<int> const& numSlots, int granularity) const
{
    std::vector<size_t> numBytes(static_cast<size_t>(numPoolGroups()), 0);
    for (int pg = 0; pg < numPoolGroups(); ++pg)
    {
        for (auto poolSize : slotSize(static_cast<PoolGroupIndex>(pg)))
        {
            numBytes[pg] += roundUp(
                static_cast<size_t>(numSlots[pg]) * static_cast<size_t>(poolSize), static_cast<size_t>(granularity));
        }
    }
    return numBytes;
}

// ---------------------------------------------------------------------------
// computeSlotCountForLevel
// ---------------------------------------------------------------------------

std::vector<int> StorageManager::computeSlotCountForLevel(CacheTierConfig const& tierConfig,
    std::vector<std::vector<int>> const& slotSizeLists, std::vector<float> const& ratio) const
{
    CacheTier tier = cacheTierOf(tierConfig);
    size_t quota = cacheTierQuota(tierConfig);
    int granularity = CacheLevelManager::cacheTierGranularity(tier, quota);
    quota = std::max(minQuotaForLevel(slotSizeLists, granularity), roundUp(quota, static_cast<size_t>(granularity)));
    return CacheLevelStorage::ratioToSlotCountList(quota, slotSizeLists, ratio, granularity, mMinSlots);
}

// ---------------------------------------------------------------------------
// minQuotaForLevel
// ---------------------------------------------------------------------------

size_t StorageManager::minQuotaForLevel(std::vector<std::vector<int>> const& slotSizeLists, int granularity) const
{
    size_t total = 0;
    for (size_t pg = 0; pg < slotSizeLists.size(); ++pg)
    {
        for (auto slotSize : slotSizeLists[pg])
        {
            total += roundUp(
                static_cast<size_t>(mMinSlots[pg]) * static_cast<size_t>(slotSize), static_cast<size_t>(granularity));
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// constrainRatio
// ---------------------------------------------------------------------------

std::vector<float> StorageManager::constrainRatio(std::vector<float> const& ratio) const
{
    auto& gpuStorage = *mLevels[0].storage;
    int granularity = gpuStorage.poolSizeGranularity();
    auto slotCountList = gpuStorage.computeSlotCountList(ratio, mMinSlots);
    auto numBytes = slotsToBytes(slotCountList, granularity);
    return normalizeToRatio(numBytes);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
