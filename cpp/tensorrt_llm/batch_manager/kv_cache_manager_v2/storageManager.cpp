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
#include "tensorrt_llm/common/logger.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <set>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// CacheLevelManager
// ---------------------------------------------------------------------------

CacheLevelManager::CacheLevelManager(TypedVec<LifeCycleId, PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cl,
    CacheTierConfig const& tierConfig, StorageConfig const& storageConfig,
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList)
    : cacheLevel(cl)
    , cacheTier(CacheTier(tierConfig.index()))
    , controller(lifeCycleGrouping, cl)
{
    storage = createCacheLevelStorage(tierConfig, storageConfig, slotCountList);
}

size_t CacheLevelManager::cacheTierGranularity(CacheTier tier, size_t quota)
{
    switch (tier)
    {
    case CacheTier::GPU_MEM:
    {
        constexpr size_t kPageSize = 2ULL << 20;
        return kPageSize << std::min(4, std::max(0, static_cast<int>(std::log2(quota / (kPageSize * 512)))));
    }
    case CacheTier::HOST_MEM: return HostMem::kAlignment; // 4 KiB
    case CacheTier::DISK: return size_t{2} << 20;         // DiskCacheLevelStorage::POOL_SIZE_GRANULARITY
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
TypedVec<LifeCycleId, TypedVec<PoolIndex, int>> computeSlotToPageIndices(StorageConfig const& config)
{
    LifeCycleId numLc = config.numLifeCycles();
    TypedVec<LifeCycleId, TypedVec<PoolIndex, int>> result(numLc);

    auto const& slotDescList = config.slotDescList;
    auto const& grouping = config.lifeCycleGrouping();

    for (LifeCycleId lcId{0}; lcId < result.size(); ++lcId)
    {
        PoolGroupIndex pgIdx = grouping[lcId];
        SlotDesc const& sd = slotDescList.at(pgIdx);
        // Find the variant that corresponds to this lifecycle.
        for (auto const& variant : sd.variants)
        {
            if (variant.lifeCycleId == lcId)
            {
                // Each coalesced buffer contributes its numBuffers as the scale.
                result[lcId].reserve(variant.coalescedBuffers.size());
                for (auto const& cb : variant.coalescedBuffers)
                    result[lcId].push_back(cb.numBuffers());
                break;
            }
        }
        if (result[lcId].empty())
            result[lcId].push_back(1); // fallback
    }
    return result;
}

} // namespace

// ---------------------------------------------------------------------------
// StorageManager
// ---------------------------------------------------------------------------

StorageManager::StorageManager(LifeCycleRegistry const& lifeCycles, StorageConfig const& config, int tokensPerBlock,
    std::optional<SwaScratchReuseConfig> swaScratchReuse, std::optional<BatchDesc> const& typicalBatch,
    std::vector<BatchDesc> const& constraints)
    : mLifeCycles(lifeCycles)
    , mStorageConfig(config)
    , mSwaScratchReuse(std::move(swaScratchReuse))
{
    mLifeCycleGrouping = config.lifeCycleGrouping();
    mLayerToLifeCycleIds = config.layerToLifeCycleIds();
    mSlotToPageIndices = computeSlotToPageIndices(config);
    mBufferAttr = config.bufferAttributes();
    mSlotDescList = config.slotDescList;

    // Compute layer attributes and slot utilization fractions for scratch support.
    mLayerAttributes = config.layerAttributes();
    mSlotUtilFracMax.resize(lifeCycles.size(), Rational{0, 1});
    for (auto const& [layerId, layerAttr] : mLayerAttributes)
    {
        LifeCycleId const lcIdx = layerAttr.lifeCycleId;
        if (layerAttr.slotUtilFracMax > mSlotUtilFracMax[lcIdx])
        {
            mSlotUtilFracMax[lcIdx] = layerAttr.slotUtilFracMax;
        }
    }

    TLLM_CHECK_DEBUG(std::all_of(mLifeCycleGrouping.begin(), mLifeCycleGrouping.end(),
        [this](PoolGroupIndex pg) { return pg < numPoolGroups(); }));
    TLLM_CHECK_DEBUG(numPoolGroups()
        == PoolGroupIndex{
            static_cast<int>(std::set<PoolGroupIndex>(mLifeCycleGrouping.begin(), mLifeCycleGrouping.end()).size())});

    // Build one CacheLevelManager per tier.
    TLLM_CHECK_DEBUG(!config.cacheTiers.empty());
    TLLM_CHECK_DEBUG_WITH_INFO(
        std::holds_alternative<GpuCacheTierConfig>(config.cacheTiers[kGpuLevel]), "First cache tier must be GPU");

    // Compute slot size lists for all pool groups.
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> slotSizeLists;
    slotSizeLists.reserve(mSlotDescList.size());
    for (auto const& sd : mSlotDescList)
    {
        slotSizeLists.push_back(sd.slotSizeList());
    }

    size_t gpuQuota = cacheTierQuota(config.cacheTiers[kGpuLevel]);
    size_t gpuGranularity = CacheLevelManager::cacheTierGranularity(CacheTier::GPU_MEM, gpuQuota);

    // Compute min_slots from constraints.
    mMinSlots = computeMinSlotsFromConstraints(constraints, tokensPerBlock, mSwaScratchReuse);

    // Compute init_ratio from typical_batch, constraints, or fallback.
    TypedVec<PoolGroupIndex, float> initRatio;
    if (typicalBatch.has_value())
    {
        initRatio = ratioFromBatch(*typicalBatch, tokensPerBlock, mSwaScratchReuse, gpuGranularity);
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
        initRatio = ratioFromBatch(fallback, tokensPerBlock, mSwaScratchReuse, gpuGranularity);
    }

    mLevels.reserve(config.cacheTiers.size());
    for (CacheLevel level{0}; level < config.cacheTiers.size(); ++level)
    {
        auto slotCountList = computeSlotCountForLevel(config.cacheTiers[level], slotSizeLists, initRatio);
        mLevels.emplace_back(mLifeCycleGrouping, level, config.cacheTiers[level], config, slotCountList);
    }

    TLLM_CHECK_DEBUG(mLevels.empty()
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
    {
        TLLM_CHECK_DEBUG(lvl.storage);
        lvl.storage->destroy();
    }
    mLevels.clear();
}

// ---------------------------------------------------------------------------
// newSlots
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, std::vector<Slot>> StorageManager::newSlots(
    CacheLevel level, TypedVec<LifeCycleId, SlotCount> const& numSlotsPerLc)
{
    TLLM_CHECK_DEBUG(numSlotsPerLc.size() == numLifeCycles());
    auto& storage = *mLevels.at(level).storage;

    // Aggregate by pool group.
    TypedVec<PoolGroupIndex, SlotCount> pgNumSlots(numPoolGroups(), 0);
    for (LifeCycleId lcId{0}; lcId < numSlotsPerLc.size(); ++lcId)
    {
        SlotCount const numSlots = numSlotsPerLc[lcId];
        if (numSlots < 0)
        {
            throw LogicError("StorageManager::newSlots: slot count must be non-negative");
        }
        pgNumSlots[mLifeCycleGrouping[lcId]] += numSlots;
    }

    // Prepare free slots if needed.
    bool needMore = false;
    for (PoolGroupIndex pgIdx{0}; pgIdx < pgNumSlots.size(); ++pgIdx)
    {
        if (pgNumSlots[pgIdx] > storage.numFreeSlots(pgIdx))
        {
            needMore = true;
            break;
        }
    }

    if (needMore)
    {
        prepareFreeSlots(level, pgNumSlots);
    }

    // A14: post-condition — free-slot counts satisfy requirements.
    for (PoolGroupIndex pgIdx{0}; pgIdx < pgNumSlots.size(); ++pgIdx)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(pgNumSlots[pgIdx] <= storage.numFreeSlots(pgIdx),
            "Free slot count does not satisfy requirement after prepareFreeSlots");
    }

    // Allocate.
    TypedVec<LifeCycleId, std::vector<Slot>> ret(numLifeCycles());
    try
    {
        for (LifeCycleId lcId{0}; lcId < ret.size(); ++lcId)
        {
            PoolGroupIndex pg = mLifeCycleGrouping[lcId];
            ret[lcId] = storage.allocateMultiple(pg, numSlotsPerLc[lcId]);
        }
    }
    catch (...)
    {
        for (LifeCycleId lcId{0}; lcId < ret.size(); ++lcId)
        {
            PoolGroupIndex pg = mLifeCycleGrouping[lcId];
            for (auto& s : ret[lcId])
                storage.release(pg, std::move(s));
        }
        throw;
    }
    return ret;
}

TypedVec<LifeCycleId, std::vector<Slot>> StorageManager::newGpuSlots(
    TypedVec<LifeCycleId, SlotCount> const& numSlotsPerLc)
{
    return newSlots(kGpuLevel, numSlotsPerLc);
}

std::vector<Slot> StorageManager::newSlotsForPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, SlotCount numSlots)
{
    if (numSlots < 0)
    {
        throw LogicError("StorageManager::newSlotsForPoolGroup: numSlots must be non-negative");
    }
    auto& storage = *mLevels.at(level).storage;
    if (numSlots > storage.numFreeSlots(pgIdx))
    {
        TypedVec<PoolGroupIndex, SlotCount> requirements(numPoolGroups(), 0);
        requirements.at(pgIdx) = numSlots;
        prepareFreeSlots(level, requirements);
    }
    TLLM_CHECK_DEBUG(numSlots <= storage.numFreeSlots(pgIdx));
    return storage.allocateMultiple(pgIdx, numSlots);
}

Address StorageManager::slotAddress(CacheLevel level, PoolGroupIndex pgIdx, SlotId slotId, PoolIndex poolIdx) const
{
    return mLevels.at(level).storage->slotAddress(pgIdx, slotId).at(poolIdx);
}

CacheTier StorageManager::cacheTier(CacheLevel level) const
{
    return mLevels.at(level).cacheTier;
}

void StorageManager::releaseSlot(LifeCycleId lc, CacheLevel level, Slot slot)
{
    PoolGroupIndex pg = mLifeCycleGrouping.at(lc);
    mLevels.at(level).storage->release(pg, std::move(slot));
}

// ---------------------------------------------------------------------------
// isEvictable
// ---------------------------------------------------------------------------

bool StorageManager::isEvictable(Page const& page, std::optional<CacheLevel> level) const noexcept
{
    PageStatus s = page.status();
    CacheLevel lvl = level.value_or(page.cacheLevel);
    return (s == PageStatus::DROPPABLE && page.isCommitted()) || (s == PageStatus::HELD && lvl < numCacheLevels() - 1);
}

// ---------------------------------------------------------------------------
// scheduleForEviction / excludeFromEviction
// ---------------------------------------------------------------------------

void StorageManager::scheduleForEviction(Page& page)
{
    if (isEvictable(page))
        mLevels.at(page.cacheLevel).controller.scheduleForEviction(page);
}

void StorageManager::excludeFromEviction(Page& page)
{
    TLLM_CHECK_DEBUG(page.nodeRef.has_value());
    mLevels.at(page.cacheLevel).controller.remove(*page.nodeRef);
}

// ---------------------------------------------------------------------------
// prepareFreeSlots
// ---------------------------------------------------------------------------

void StorageManager::prepareFreeSlots(CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& requirements)
{
    TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>> goals(numCacheLevels());
    for (CacheLevel lvl{0}; lvl < goals.size(); ++lvl)
    {
        goals[lvl].resize(numPoolGroups(), 0);
    }
    for (PoolGroupIndex pgIdx{0}; pgIdx < requirements.size(); ++pgIdx)
    {
        goals.at(level).at(pgIdx) = requirements.at(pgIdx);
    }

    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> fallenPages(numPoolGroups());
    _prepareFreeSlots(goals, level, fallenPages);
}

void StorageManager::forceEvict(CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& minNumPages)
{
    auto evicted = mLevels.at(level).controller.evict(minNumPages);

    if (isLastLevel(level))
    {
        // Last level: all evicted pages must be DROPPABLE (they get dropped, not migrated).
        for (auto const& pages : evicted)
        {
            for (auto const& page : pages)
            {
                TLLM_CHECK_DEBUG_WITH_INFO(page->status() == PageStatus::DROPPABLE, "Corrupted eviction controller");
            }
        }
        return;
    }

    TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>> goals(numCacheLevels());
    for (CacheLevel lvl{0}; lvl < goals.size(); ++lvl)
    {
        goals[lvl].resize(numPoolGroups(), 0);
    }
    CacheLevel nextLvl = level + 1;

    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> fallen(numPoolGroups());
    for (PoolGroupIndex pgIdx{0}; pgIdx < fallen.size(); ++pgIdx)
    {
        for (auto& sp : evicted.at(pgIdx))
            fallen.at(pgIdx).push_back(sp);
    }
    _prepareFreeSlots(goals, nextLvl, fallen);
}

// ---------------------------------------------------------------------------
// _prepareFreeSlots (recursive)
// ---------------------------------------------------------------------------

void StorageManager::_prepareFreeSlots(TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>>& goals,
    CacheLevel lvlId, TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>>& fallenPages)
{
    // A7: goals dimensions must match [numCacheLevels][numPoolGroups].
    if (TLLM_UNLIKELY(gDebug))
    {
        TLLM_CHECK_WITH_INFO(goals.size() == numCacheLevels(), "goals.rows must equal numCacheLevels");
        TLLM_CHECK_DEBUG_WITH_INFO(
            std::all_of(goals.begin(), goals.end(), [this](auto const& row) { return row.size() == numPoolGroups(); }),
            "goals.cols must equal numPoolGroups");
    }

    // A8: all fallen pages must come from upper cache levels (cache_level < lvlId).
    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(fallenPages.begin(), fallenPages.end(),
                                   [lvlId](auto const& pages) {
                                       return std::all_of(pages.begin(), pages.end(),
                                           [lvlId](auto const& p) { return p->cacheLevel < lvlId; });
                                   }),
        "Fallen pages must come from upper cache levels");

    auto& lvl = mLevels.at(lvlId);
    auto& storage = *lvl.storage;
    auto& ctrl = lvl.controller;
    bool isLast = isLastLevel(lvlId);

    TypedVec<PoolGroupIndex, SlotCount> numToEvict(numPoolGroups(), 0);
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> heldPages(numPoolGroups());

    for (PoolGroupIndex pgIdx{0}; pgIdx < numToEvict.size(); ++pgIdx)
    {
        SlotCount const goal = goals.at(lvlId).at(pgIdx);
        SlotCount const fallen = slotCountValueFromSize(fallenPages.at(pgIdx).size());
        SlotCount const oldFree = storage.numFreeSlots(pgIdx);
        SlotCount const evictableCount = ctrl.numEvictablePages(pgIdx);
        SlotCount const required = goal + fallen;
        SlotCount const shortage = required > oldFree ? required - oldFree : 0;
        numToEvict.at(pgIdx) = std::min(shortage, evictableCount);

        SlotCount fallenHeld = 0;
        if (isLast)
        {
            // Separate held pages from fallen_pages (mirrors Python's remove_if).
            auto& fp = fallenPages.at(pgIdx);
            heldPages.at(pgIdx) = stealIf(fp, [](SharedPtr<Page> const& p) { return p->status() == PageStatus::HELD; });
            fallenHeld = slotCountValueFromSize(heldPages.at(pgIdx).size());

            if (fallenHeld > oldFree + evictableCount)
                throw OutOfPagesError(
                    "Too many held pages falling to last-level cache for group " + std::to_string(pgIdx.value()));
        }

        if (oldFree + evictableCount < fallenHeld + goal)
            throw OutOfPagesError("Impossible to meet free-slot goal " + std::to_string(goal) + " for group "
                + std::to_string(pgIdx.value()));
    }

    auto evicted = ctrl.evict(numToEvict);
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> acceptedPages(numPoolGroups());

    if (isLast)
    {
        for (PoolGroupIndex pgIdx{0}; pgIdx < evicted.size(); ++pgIdx)
        {
            auto& ev = evicted.at(pgIdx);
            SlotCount const oldFree = storage.numFreeSlots(pgIdx);
            SlotCount const numEvicted = slotCountValueFromSize(ev.size());
            // A9: all evicted pages at last level must be DROPPABLE.
            TLLM_CHECK_DEBUG_WITH_INFO(
                std::all_of(ev.begin(), ev.end(), [](auto const& p) { return p->status() == PageStatus::DROPPABLE; }),
                "Evicted page at last level must be DROPPABLE");
            // Drop droppable evicted pages (GC).
            ev.clear();
            SlotCount const newFree = storage.numFreeSlots(pgIdx);
            TLLM_CHECK_DEBUG(newFree >= numEvicted + oldFree);

            // A10: held_pages count must not exceed new_free.
            TLLM_CHECK_DEBUG_WITH_INFO(slotCountValueFromSize(heldPages.at(pgIdx).size()) <= newFree,
                "held_pages count exceeds new free slot count");

            // Add held pages from upper levels.
            auto& hp = heldPages.at(pgIdx);
            auto& fp = fallenPages.at(pgIdx);
            fp.insert(fp.end(), hp.begin(), hp.end());
            hp.clear();

            SlotCount const goal = goals.at(lvlId).at(pgIdx);
            SlotCount const freeAfterGoal = newFree > goal ? newFree - goal : 0;
            SlotCount const numAccepted = std::min(freeAfterGoal, slotCountValueFromSize(fp.size()));
            if (numAccepted > 0)
            {
                acceptedPages.at(pgIdx).assign(fp.end() - static_cast<std::ptrdiff_t>(numAccepted), fp.end());
            }
            fp.clear();
        }
    }
    else
    {
        // A12: no held pages at non-last level.
        TLLM_CHECK_DEBUG_WITH_INFO(
            std::all_of(heldPages.begin(), heldPages.end(), [](auto const& hp) { return hp.empty(); }),
            "held_pages must be empty at non-last level");

        CacheLevel nextLvl = lvlId + 1;
        for (PoolGroupIndex pgIdx{0}; pgIdx < evicted.size(); ++pgIdx)
        {
            auto& ev = evicted.at(pgIdx);
            SlotCount const oldFree = storage.numFreeSlots(pgIdx);
            SlotCount const numEvicted = slotCountValueFromSize(ev.size());
            auto& fp = fallenPages.at(pgIdx);
            fp.insert(fp.begin(), ev.begin(), ev.end()); // prepend evicted to fallen (preserving order)
            ev.clear();

            SlotCount const goal = goals.at(lvlId).at(pgIdx);
            SlotCount const availableAfterGoal = oldFree + numEvicted > goal ? oldFree + numEvicted - goal : 0;
            SlotCount const numAccepted = std::min(availableAfterGoal, slotCountValueFromSize(fp.size()));
            if (numAccepted > 0)
            {
                acceptedPages.at(pgIdx).assign(fp.end() - static_cast<std::ptrdiff_t>(numAccepted), fp.end());
                fp.erase(fp.end() - static_cast<std::ptrdiff_t>(numAccepted), fp.end());
            }
        }
        _prepareFreeSlots(goals, nextLvl, fallenPages);
    }

    // A13: all fallen pages must have been consumed.
    TLLM_CHECK_DEBUG_WITH_INFO(
        std::all_of(fallenPages.begin(), fallenPages.end(), [](auto const& fp) { return fp.empty(); }),
        "All fallen pages must be consumed after level loop");

    // Migrate accepted pages into lvlId.
    for (PoolGroupIndex pgIdx{0}; pgIdx < acceptedPages.size(); ++pgIdx)
    {
        // Group by source level (mirrors Python's partition()).
        auto bySrcLevel = partition(acceptedPages.at(pgIdx), [](SharedPtr<Page> const& p) { return p->cacheLevel; });

        for (auto& [srcLvl, pages] : bySrcLevel)
        {
            _batchedMigrate(pgIdx, lvlId, srcLvl, pages, /*updateSrc=*/true);
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
    std::vector<SharedPtr<Page>> const& srcPages, bool updateSrc, bool defrag)
{
    TLLM_CHECK_DEBUG(defrag || dstLevel != srcLevel);
    SlotCount const numSlots = slotCountValueFromSize(srcPages.size());

    auto& srcPoolGroup = poolGroup(srcLevel, pgIdx);
    auto& dstPoolGroup = poolGroup(dstLevel, pgIdx);

    if (dstPoolGroup.numFreeSlots() < numSlots)
        throw OutOfPagesError("Not enough free slots for migration");

    auto dstSlots = dstPoolGroup.allocateMultiple(numSlots);
    // A15: allocated slot count must match the request.
    TLLM_CHECK_DEBUG_WITH_INFO(slotCountValueFromSize(dstSlots.size()) == numSlots, "dst_slots size mismatch");
    try
    {
        CacheTier dstTier = mLevels.at(dstLevel).cacheTier;
        CacheTier srcTier = mLevels.at(srcLevel).cacheTier;

        PoolIndex numPools = mNumPools(pgIdx);

        // Build copy tasks per pool.
        TypedVec<PoolIndex, std::vector<CopyTask>> tasksPerPool(numPools);
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            auto const& src = srcPages.at(i);
            auto const& dst = dstSlots.at(i);
            // Fix #8: assert non-defrag migrations only accept pages not scheduled for eviction.
            TLLM_CHECK_DEBUG(defrag || !src->scheduledForEviction());
            for (PoolIndex poolIdx{0}; poolIdx < tasksPerPool.size(); ++poolIdx)
            {
                Address dstAddr = dstPoolGroup.slotAddress(dst.slotId()).at(poolIdx);
                Address srcAddr = srcPoolGroup.slotAddress(src->slotId()).at(poolIdx);
                tasksPerPool.at(poolIdx).push_back({dstAddr, srcAddr});
            }
        }

        // Collect prior events (src + dst ready events) — mirrors Python's prior_events set.
        std::vector<CachedCudaEvent const*> priorEvents;
        priorEvents.reserve(2 * srcPages.size());
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            priorEvents.push_back(&srcPages.at(i)->readyEvent);
            priorEvents.push_back(&dstSlots.at(i).readyEvent);
        }

        // Create a temporary CUDA stream that waits for all prior events before copying.
        TemporaryCudaStream tempStream(priorEvents);
        {
            auto scope = tempStream.enter();
            CUstream stream = tempStream.get();
            auto slotSizes = slotSize(pgIdx);
            for (PoolIndex poolIdx{0}; poolIdx < numPools; ++poolIdx)
            {
                batchedCopy(dstTier, srcTier, slotSizes.at(poolIdx), tasksPerPool.at(poolIdx), stream);
            }
        } // ~Scope records finish event

        CachedCudaEvent finishEvent = tempStream.takeFinishEvent();
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            dstSlots.at(i).readyEvent = finishEvent;
            // Fix #6: set src.ready_event unconditionally — compulsory for the next owner
            // getting this slot from the pool. Mirrors Python: `src.ready_event = finish_event`.
            srcPages.at(i)->readyEvent = finishEvent;
            if (updateSrc)
            {
                bool wasScheduled = srcPages.at(i)->scheduledForEviction();
                if (wasScheduled)
                    excludeFromEviction(*srcPages.at(i));
                // Extract source slot from the page and release it back to the pool.
                Slot srcSlot;
                srcSlot.setSlotId(srcPages.at(i)->slotId()); // asserts valid
                srcSlot.readyEvent = finishEvent;
                srcPages.at(i)->resetSlot();
                srcPoolGroup.release(std::move(srcSlot));
                // Transfer dst slot ownership to the page.
                srcPages.at(i)->setSlot(dstSlots.at(i));
                srcPages.at(i)->cacheLevel = dstLevel;
                if (wasScheduled)
                    scheduleForEviction(*srcPages.at(i));
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
    std::map<std::pair<CacheLevel, PoolGroupIndex>, std::vector<SharedPtr<Page>>> groups;
    for (auto const& t : targets)
    {
        if (t.page->cacheLevel == kGpuLevel)
            continue;
        PoolGroupIndex pg = mLifeCycleGrouping.at(t.lifeCycle);
        groups[{t.page->cacheLevel, pg}].push_back(t.page);
    }
    for (auto& [key, pages] : groups)
        _batchedMigrate(key.second, kGpuLevel, key.first, pages, /*updateSrc=*/true);
}

void StorageManager::prefetch(
    CacheLevel dstLevel, TypedVec<PoolGroupIndex, TypedVec<CacheLevel, std::vector<SharedPtr<Page>>>> const& pages)
{
    TypedVec<PoolGroupIndex, SlotCount> numSlotsToMigrate(numPoolGroups(), 0);
    std::vector<SharedPtr<Page>> scheduled;

    struct ReschedulePagesGuard
    {
        StorageManager& storageManager;
        std::vector<SharedPtr<Page>>& scheduled;

        ~ReschedulePagesGuard()
        {
            for (auto const& page : scheduled)
            {
                storageManager.scheduleForEviction(*page);
            }
            scheduled.clear();
        }
    } reschedulePagesGuard{*this, scheduled};

    for (PoolGroupIndex pgIndex{0}; pgIndex < pages.size(); ++pgIndex)
    {
        auto const& poolGroupPages = pages.at(pgIndex);
        for (CacheLevel level{0}; level < poolGroupPages.size(); ++level)
        {
            auto const& levelPages = poolGroupPages.at(level);
            TLLM_CHECK_DEBUG(level >= dstLevel || levelPages.empty());
            for (auto const& page : levelPages)
            {
                if (page->scheduledForEviction())
                {
                    excludeFromEviction(*page);
                    scheduled.push_back(page);
                }
                else if (isEvictable(*page, dstLevel))
                {
                    scheduled.push_back(page);
                }
                TLLM_CHECK_DEBUG(level >= dstLevel);
                if (level == dstLevel)
                {
                    continue;
                }
                numSlotsToMigrate.at(pgIndex) += 1;
            }
        }
    }

    prepareFreeSlots(dstLevel, numSlotsToMigrate);
    for (PoolGroupIndex pgIndex{0}; pgIndex < pages.size(); ++pgIndex)
    {
        auto const& poolGroupPages = pages.at(pgIndex);
        for (CacheLevel lvl = dstLevel + 1; lvl < numCacheLevels(); ++lvl)
        {
            _batchedMigrate(pgIndex, dstLevel, lvl, poolGroupPages.at(lvl), /*updateSrc=*/true);
        }
    }
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

LifeCycle const& StorageManager::getLifeCycle(LifeCycleId lc) const
{
    return mLifeCycles[lc];
}

PoolGroupIndex StorageManager::getPoolGroupIndex(LifeCycleId lc) const
{
    return mLifeCycleGrouping.at(lc);
}

PoolIndex StorageManager::mNumPools(PoolGroupIndex pgIdx) const
{
    TLLM_CHECK_DEBUG(!mLevels.empty());
    return getUniformAttribute(mLevels, [pgIdx](auto const& lvl) { return lvl.storage->numPools(pgIdx); });
}

PoolIndex StorageManager::numPools(PoolGroupIndex pgIdx) const
{
    return mNumPools(pgIdx);
}

TypedVec<PoolIndex, size_t> StorageManager::slotSize(PoolGroupIndex pgIdx) const
{
    return mSlotDescList.at(pgIdx).slotSizeList();
}

PoolGroupBase& StorageManager::poolGroup(CacheLevel lvl, PoolGroupIndex pgIdx)
{
    return mLevels.at(lvl).storage->poolGroup(pgIdx);
}

MemAddress StorageManager::getMemPoolBaseAddress(LayerId layerId, DataRole role) const
{
    auto it = mBufferAttr.find(BufferId{layerId, role});
    if (it == mBufferAttr.end())
        throw std::out_of_range("Unknown BufferId");
    auto const& attr = it->second;
    PoolGroupIndex pgIdx = mLifeCycleGrouping.at(attr.lifeCycleId);
    return mLevels[kGpuLevel].storage->getBaseAddress(pgIdx, attr.poolIndex, SlotId{0}) + attr.offset;
}

MemAddress StorageManager::getMemPoolBaseAddress(PoolGroupIndex pgIdx, PoolIndex poolIdx) const
{
    return mLevels[kGpuLevel].storage->getBaseAddress(pgIdx, poolIdx, SlotId{0});
}

LayerAttr const& StorageManager::getLayerAttr(LayerId layerId) const
{
    auto it = mLayerAttributes.find(layerId);
    if (it == mLayerAttributes.end())
        throw std::out_of_range("Unknown LayerId for LayerAttr");
    return it->second;
}

SlotCount StorageManager::numSlots(PoolGroupIndex pgIdx, CacheLevel level) const
{
    return mLevels.at(level).storage->numSlots(pgIdx);
}

StorageStatistics StorageManager::getStatistics(CacheLevel level, PoolGroupIndex pgIdx) const
{
    auto const& lvl = mLevels.at(level);
    SlotCount freeSlots = lvl.storage->numFreeSlots(pgIdx);
    SlotCount totalSlots = lvl.storage->numSlots(pgIdx);
    SlotCount evictable = lvl.controller.numEvictablePages(pgIdx);
    auto sizes = lvl.storage->slotSize(pgIdx);
    return StorageStatistics{sizes, totalSlots, freeSlots, evictable};
}

TypedVec<PoolGroupIndex, float> StorageManager::getUtilization(CacheLevel level) const
{
    TypedVec<PoolGroupIndex, float> result;
    result.reserve(numPoolGroups());
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(); ++pgIdx)
    {
        auto const s = getStatistics(level, pgIdx);
        TLLM_CHECK_DEBUG(s.total > 0);
        result.push_back(static_cast<float>(s.unavailable()) / static_cast<float>(s.total));
    }
    return result;
}

float StorageManager::getOverallUtilization(CacheLevel level) const
{
    float num = 0.f, den = 0.f;
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(); ++pgIdx)
    {
        auto s = getStatistics(level, pgIdx);
        float sz = 0.f;
        for (auto v : s.slotSizes)
            sz += static_cast<float>(v);
        num += sz * static_cast<float>(s.unavailable());
        den += sz * static_cast<float>(s.total);
    }
    TLLM_CHECK_DEBUG(den > 0.f);
    return num / den;
}

// ---------------------------------------------------------------------------
// expandPoolGroup
// ---------------------------------------------------------------------------

void StorageManager::expandPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, SlotCount newNumSlots)
{
    auto& pg = poolGroup(level, pgIdx);
    TLLM_CHECK_DEBUG(newNumSlots > pg.numSlots());
    pg.resizePools(newNumSlots);
    pg.slotAllocator().expand(newNumSlots);
}

// ---------------------------------------------------------------------------
// shrinkPoolGroup — mirrors Python _storage_manager.py::shrink_pool_group
// ---------------------------------------------------------------------------

void StorageManager::shrinkPoolGroup(
    CacheLevel level, PoolGroupIndex pgIdx, SlotCount newNumSlots, std::vector<SharedPtr<Page>> const& persistentPages)
{
    auto& pg = poolGroup(level, pgIdx);
    auto& allocator = pg.slotAllocator();
    auto& ctrl = mLevels.at(level).controller;
    TLLM_CHECK_DEBUG(newNumSlots < pg.numSlots());

    // A16: persistent_pages preconditions.
    TLLM_CHECK_DEBUG_WITH_INFO(
        persistentPages.size() <= slotCountToSizeT(newNumSlots), "Not enough slots to hold all persistent pages");
    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(persistentPages.begin(), persistentPages.end(),
                                   [this, level, pgIdx](auto const& p)
                                   { return p->cacheLevel == level && mLifeCycleGrouping.at(p->lifeCycle) == pgIdx; }),
        "Persistent page cache level or pool group mismatch");

    // Fast path: when no slot id has ever been issued in the to-be-removed
    // range [newNumSlots, capacity), there is nothing to migrate.
    // numActiveSlots() is a monotone high-water mark of issued ids.
    if (allocator.numActiveSlots() <= newNumSlots)
    {
        allocator.prepareForShrink(newNumSlots);
        allocator.finishShrink();
        pg.resizePools(newNumSlots);
        return;
    }

    // Find overflow pages: scheduled pages with slot_id >= newNumSlots.
    auto gen = ctrl.pageGenerator(pgIdx);
    std::deque<std::pair<SlotCount, SharedPtr<Page>>> overflowSlots;
    {
        SlotCount idx = 0;
        while (auto const* page = gen())
        {
            if ((*page)->slotId() >= newNumSlots)
                overflowSlots.emplace_back(idx, *page);
            ++idx;
        }
    }

    // Persistent pages in overflow range.
    std::vector<SharedPtr<Page>> overflowPersistent;
    for (auto const& p : persistentPages)
    {
        if (p->slotId() >= newNumSlots)
            overflowPersistent.push_back(p);
    }
    SlotCount numOverflowPersistent = slotCountValueFromSize(overflowPersistent.size());

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
    SlotCount minNumEvicted = 0;
    SlotCount numEvictedOverflowSlots = 0;
    while (!overflowSlots.empty()
        && slotCountValueFromSize(overflowSlots.size()) + numOverflowPersistent
            > std::min(newNumSlots, overflowSlots.front().first + allocator.numFreeSlots() - numEvictedOverflowSlots))
    {
        minNumEvicted = overflowSlots.front().first + 1;
        overflowSlots.pop_front();
        ++numEvictedOverflowSlots;
    }

    // Force-evict the required pages.
    TypedVec<PoolGroupIndex, SlotCount> evictReqs(numPoolGroups(), 0);
    evictReqs[pgIdx] = minNumEvicted;
    forceEvict(level, evictReqs);

    // Remaining overflow pages to defragment.
    std::vector<SharedPtr<Page>> overflowPages;
    overflowPages.reserve(overflowSlots.size() + overflowPersistent.size());
    for (auto& [idx, p] : overflowSlots)
        overflowPages.push_back(p);
    for (auto& p : overflowPersistent)
        overflowPages.push_back(p);

    // Ensure free slots for the overflow pages.
    TypedVec<PoolGroupIndex, SlotCount> reqs(numPoolGroups(), 0);
    reqs[pgIdx] = slotCountValueFromSize(overflowPages.size());
    prepareFreeSlots(level, reqs);

    // A17: all overflow pages must be at the expected cache level.
    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(overflowPages.begin(), overflowPages.end(),
                                   [level](auto const& p) { return p->cacheLevel == level; }),
        "Overflow page cache level mismatch");

    // Defragment: migrate overflow pages to free slots within the same level.
    _batchedMigrate(pgIdx, level, level, overflowPages, /*updateSrc=*/true, /*defrag=*/true);

    // A18: post-defrag overflow assertion — overflow slot count matches expectations.
    TLLM_CHECK_DEBUG_WITH_INFO(allocator.numOverflowSlots() == allocator.numActiveSlots() - allocator.targetCapacity(),
        "Post-defrag overflow slot count mismatch");

    // Finalize shrink and resize pools.
    allocator.finishShrink();
    pg.resizePools(newNumSlots);
}

// ---------------------------------------------------------------------------
// adjustCacheLevel — mirrors Python _storage_manager.py::adjust_cache_level
// ---------------------------------------------------------------------------

void StorageManager::adjustCacheLevel(CacheLevel level, std::optional<size_t> newQuota,
    TypedVec<PoolGroupIndex, float> const& ratioList,
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* persistentPages)
{
    auto& lvlStorage = *mLevels.at(level).storage;
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
        TLLM_CHECK_DEBUG(persistentPages == nullptr);

    // Shrink first.
    for (PoolGroupIndex pgIdx{0}; pgIdx < newNumSlots.size(); ++pgIdx)
    {
        if (newNumSlots[pgIdx] >= oldNumSlots[pgIdx])
            continue;
        std::vector<SharedPtr<Page>> pages;
        if (persistentPages)
            pages = (*persistentPages)[pgIdx];
        shrinkPoolGroup(level, pgIdx, newNumSlots[pgIdx], pages);
    }
    // Then expand.
    for (PoolGroupIndex pgIdx{0}; pgIdx < newNumSlots.size(); ++pgIdx)
    {
        if (newNumSlots[pgIdx] <= oldNumSlots[pgIdx])
            continue;
        expandPoolGroup(level, pgIdx, newNumSlots[pgIdx]);
    }
    lvlStorage.postResize();
}

TypedVec<PoolGroupIndex, float> StorageManager::getRatioList(CacheLevel level) const
{
    return mLevels.at(level).storage->ratioList();
}

TypedVec<PoolGroupIndex, float> StorageManager::ratioFromLength(
    int tokensPerBlock, int historyLength, int capacity) const
{
    if (capacity < historyLength)
    {
        TLLM_LOG_WARNING("Bad sampling for capacity and history_length");
        capacity = historyLength;
    }
    int numBlocks = divUp(capacity, tokensPerBlock);
    TypedVec<PoolGroupIndex, size_t> numBytes(numPoolGroups(), 0);
    auto ssmLcId = mLifeCycles.ssmLifeCycleId();
    auto const& lifecycles = mLifeCycles.getAll();
    for (LifeCycleId lcId{0}; lcId < lifecycles.size(); ++lcId)
    {
        PoolGroupIndex pgIdx = mLifeCycleGrouping[lcId];
        auto ss = slotSize(pgIdx);
        size_t slotSizeSum = 0;
        for (auto s : ss)
            slotSizeSum += s;
        int numRequiredBlocks;
        if (ssmLcId.has_value() && lcId == *ssmLcId)
        {
            numRequiredBlocks = 1;
        }
        else
        {
            auto stale = getStaleRange(lifecycles[lcId], historyLength, tokensPerBlock);
            numRequiredBlocks = std::max(numBlocks - stale.length(), 1);
        }
        numBytes[pgIdx] += static_cast<size_t>(numRequiredBlocks) * slotSizeSum;
    }
    return normalizeToRatio(numBytes);
}

// ---------------------------------------------------------------------------
// ratioFromBatch
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, float> StorageManager::ratioFromBatch(BatchDesc const& batch, int tokensPerBlock,
    std::optional<SwaScratchReuseConfig> const& swaScratchReuse, size_t granularity) const
{
    auto numSlots = computeSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
    auto numBytes = slotsToBytes(numSlots, granularity);
    return normalizeToRatio(numBytes);
}

// ---------------------------------------------------------------------------
// computeMinSlotsFromConstraints
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::computeMinSlotsFromConstraints(
    std::vector<BatchDesc> const& constraints, int tokensPerBlock,
    std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const
{
    // Default floor: 1 slot per life cycle in each pool group.
    TypedVec<PoolGroupIndex, SlotCount> maxSlots(numPoolGroups(), 0);
    for (auto pgIdx : mLifeCycleGrouping)
        maxSlots[pgIdx] += 1;
    for (auto const& batch : constraints)
    {
        auto slots = computeSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
        for (PoolGroupIndex pgIdx{0}; pgIdx < slots.size(); ++pgIdx)
        {
            maxSlots[pgIdx] = std::max(maxSlots[pgIdx], slots[pgIdx]);
        }
    }
    return maxSlots;
}

// ---------------------------------------------------------------------------
// computeSlotsForBatch
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::computeSlotsForBatch(
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const
{
    TypedVec<PoolGroupIndex, SlotCount> numSlots(numPoolGroups(), 0);
    auto ssmLcId = mLifeCycles.ssmLifeCycleId();
    int sysBlocks = batch.systemPromptLength / tokensPerBlock;

    for (auto const& [lcIdx, lc] : mLifeCycles)
    {
        PoolGroupIndex pgIdx = mLifeCycleGrouping[lcIdx];
        if (ssmLcId.has_value() && lcIdx == *ssmLcId)
        {
            // SSM: always 1 dedicated block per request, never shared.
            numSlots[pgIdx] += slotCountValueFromSize(batch.kvCaches.size());
            continue;
        }
        // Shared sys blocks (counted once): union of non-stale sys blocks across all requests.
        HalfOpenRange<BlockOrdinal> sysRange{0, sysBlocks};
        HalfOpenRange<BlockOrdinal> staleIntersection = sysRange;
        for (auto const& kv : batch.kvCaches)
        {
            auto stale = getStaleRange(lc, kv.historyLength, tokensPerBlock);
            staleIntersection = intersect(staleIntersection, stale);
        }
        numSlots[pgIdx] += sysBlocks - staleIntersection.length();

        // Per-request unique blocks (excluding shared sys blocks already counted above).
        for (auto const& kv : batch.kvCaches)
        {
            int totalBlocks = divUp(kv.capacity, tokensPerBlock);
            auto stale = getStaleRange(lc, kv.historyLength, tokensPerBlock);
            int nonStale = totalBlocks - stale.length();
            int nonStaleSys = sysBlocks - intersect(stale, sysRange).length();
            int uniqueNonStale = std::max(0, nonStale - nonStaleSys);
            if (swaScratchReuse.has_value())
            {
                auto scratch = computeScratchRange(
                    lc, kv.historyLength, kv.capacity, tokensPerBlock, swaScratchReuse->maxRewindLen);
                int numScratch = scratch.length();
                // Scratch blocks share coalesced slots: actual slots = ceil(numScratch * fracMax).
                numSlots[pgIdx] += (uniqueNonStale - numScratch) + mSlotUtilFracMax[lcIdx].ceilMul(numScratch);
            }
            else
            {
                numSlots[pgIdx] += uniqueNonStale;
            }
        }
    }
    return numSlots;
}

// ---------------------------------------------------------------------------
// slotsToBytes
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, size_t> StorageManager::slotsToBytes(
    TypedVec<PoolGroupIndex, SlotCount> const& numSlots, size_t granularity) const
{
    TypedVec<PoolGroupIndex, size_t> numBytes(numPoolGroups(), 0);
    for (PoolGroupIndex pgIdx{0}; pgIdx < numSlots.size(); ++pgIdx)
    {
        for (auto poolSize : slotSize(pgIdx))
        {
            numBytes[pgIdx] += roundUp(slotCountToSizeT(numSlots[pgIdx]) * poolSize, granularity);
        }
    }
    return numBytes;
}

// ---------------------------------------------------------------------------
// computeSlotCountForLevel
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::computeSlotCountForLevel(CacheTierConfig const& tierConfig,
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    TypedVec<PoolGroupIndex, float> const& ratio) const
{
    CacheTier tier = cacheTierOf(tierConfig);
    size_t quota = cacheTierQuota(tierConfig);
    size_t granularity = CacheLevelManager::cacheTierGranularity(tier, quota);
    quota = std::max(minQuotaForLevel(slotSizeLists, granularity), roundUp(quota, granularity));
    return CacheLevelStorage::ratioToSlotCountList(quota, slotSizeLists, ratio, granularity, mMinSlots);
}

// ---------------------------------------------------------------------------
// minQuotaForLevel
// ---------------------------------------------------------------------------

size_t StorageManager::minQuotaForLevel(
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists, size_t granularity) const
{
    size_t total = 0;
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotSizeLists.size(); ++pgIdx)
    {
        for (auto slotSize : slotSizeLists[pgIdx])
        {
            total += roundUp(slotCountToSizeT(mMinSlots[pgIdx]) * slotSize, granularity);
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// constrainRatio
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, float> StorageManager::constrainRatio(TypedVec<PoolGroupIndex, float> const& ratio) const
{
    auto& gpuStorage = *mLevels[kGpuLevel].storage;
    size_t granularity = gpuStorage.poolSizeGranularity();
    auto slotCountList = gpuStorage.computeSlotCountList(ratio, mMinSlots);
    auto numBytes = slotsToBytes(slotCountList, granularity);
    return normalizeToRatio(numBytes);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
