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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/evictionController.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/storage/core.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declarations.
class Page;
class KvCache;
struct BatchedLockTarget;

// ---------------------------------------------------------------------------
// StorageStatistics — per-pool-group slot counts.
// ---------------------------------------------------------------------------
struct StorageStatistics
{
    std::vector<int> slotSizes; // per-pool slot size in bytes
    int total;                  // total slots
    int free;                   // free (unallocated) slots
    int evictable;              // scheduled for eviction

    int available() const noexcept
    {
        return free + evictable;
    }

    int unavailable() const noexcept
    {
        return total - available();
    }
};

// ---------------------------------------------------------------------------
// CacheLevelManager — one storage level (GPU/Host/Disk) + its eviction controller.
// ---------------------------------------------------------------------------
class CacheLevelManager
{
public:
    CacheLevelManager(std::vector<PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cacheLevel,
        CacheTierConfig const& tierConfig, StorageConfig const& storageConfig, std::vector<int> const& slotCountList);

    // Compute pool size granularity for a given cache tier and quota.
    static int cacheTierGranularity(CacheTier tier, size_t quota);

    CacheLevel cacheLevel;
    CacheTier cacheTier;
    std::unique_ptr<CacheLevelStorage> storage;
    PerLevelEvictionController controller;

    PoolGroupIndex numPoolGroups() const noexcept
    {
        assert(storage->numPoolGroups() == controller.numPoolGroups()
            && "Storage and controller disagree on numPoolGroups");
        return controller.numPoolGroups();
    }
};

// ---------------------------------------------------------------------------
// StorageManager — manages all cache levels and the eviction pipeline.
// Mirrors Python's StorageManager.
// ---------------------------------------------------------------------------
class StorageManager : public std::enable_shared_from_this<StorageManager>
{
public:
    StorageManager(LifeCycleRegistry const& lifeCycles, StorageConfig const& config, int tokensPerBlock,
        std::optional<SwaScratchReuseConfig> swaScratchReuse = std::nullopt,
        std::optional<BatchDesc> const& typicalBatch = std::nullopt, std::vector<BatchDesc> const& constraints = {});
    ~StorageManager();

    StorageManager(StorageManager const&) = delete;
    StorageManager& operator=(StorageManager const&) = delete;

    void destroy();

    // ---- Allocation -------------------------------------------------------

    // Allocate slots for all life cycles at the given cache level.
    // numSlotsPerLc[lcId] = how many slots to allocate for that life cycle.
    // Returns a vector indexed by lcId.
    std::vector<std::vector<Slot>> newSlots(CacheLevel level, std::vector<int> const& numSlotsPerLc);

    std::vector<std::vector<Slot>> newGpuSlots(std::vector<int> const& numSlotsPerLc);

    // Allocate slots for a single pool group at the given cache level.
    // Returns numSlots Slot objects. Throws OutOfPagesError if allocation fails.
    std::vector<Slot> newSlotsForPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, int numSlots);

    // Release a slot back to its pool.
    void releaseSlot(LifeCycleId lc, CacheLevel level, Slot slot);

    // ---- Eviction ----------------------------------------------------------

    // Schedule a page for eviction (if evictable).
    void scheduleForEviction(Page& page);

    // Remove a page from the eviction queue.
    void excludeFromEviction(Page& page);

    // Check if a page is evictable (optionally at a target level).
    bool isEvictable(Page const& page, std::optional<CacheLevel> level = std::nullopt) const noexcept;

    // Ensure numFreeSlots[pgIdx] free GPU slots exist (evicting pages as needed).
    void prepareFreeSlots(CacheLevel level, std::vector<int> const& requirements);

    // Force-evict pages from a level to free space.
    void forceEvict(CacheLevel level, std::vector<int> const& minNumPages);

    // Dynamic cache level resizing.
    void adjustCacheLevel(CacheLevel level, std::optional<size_t> newQuota, std::vector<float> const& ratioList,
        std::vector<std::vector<SharedPtr<Page>>> const* persistentPages);
    void shrinkPoolGroup(
        CacheLevel level, PoolGroupIndex pgIdx, int newNumSlots, std::vector<SharedPtr<Page>> const& persistentPages);
    void expandPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, int newNumSlots);

    // ---- Migration ---------------------------------------------------------

    // Migrate a batch of pages to GPU (used by batchedLockToGpu).
    void batchedMigrateToGpu(std::vector<BatchedLockTarget> const& targets, KvCache& kvCache);

    // ---- Query helpers -----------------------------------------------------

    LifeCycleRegistry const& lifeCycles() const noexcept
    {
        return mLifeCycles;
    }

    LifeCycle const& getLifeCycle(LifeCycleId lc) const;

    int numLifeCycles() const noexcept
    {
        return static_cast<int>(mLifeCycleGrouping.size());
    }

    int numPoolGroups() const noexcept
    {
        return static_cast<int>(mSlotDescList.size());
    }

    int numCacheLevels() const noexcept
    {
        return static_cast<int>(mLevels.size());
    }

    bool isLastLevel(CacheLevel lvl) const noexcept
    {
        return static_cast<int>(lvl) == numCacheLevels() - 1;
    }

    int getPoolGroupIndex(LifeCycleId lc) const;
    int numPools(PoolGroupIndex pgIdx) const;

    // Return the byte size of each pool in a pool group.
    std::vector<int> slotSize(PoolGroupIndex pgIdx) const;

    // Current ratio list for a cache level (proportional to byte usage per pool group).
    std::vector<float> getRatioList(CacheLevel level) const;

    // Compute init ratio from an assumed average history length and capacity.
    std::vector<float> ratioFromLength(int tokensPerBlock, int historyLength, int capacity) const;

    // Compute ratio from a BatchDesc.
    std::vector<float> ratioFromBatch(BatchDesc const& batch, int tokensPerBlock,
        std::optional<SwaScratchReuseConfig> const& swaScratchReuse, int granularity) const;

    // Apply stored min_slots constraint to a ratio list for GPU level.
    std::vector<float> constrainRatio(std::vector<float> const& ratio) const;

    // Byte address of a slot's buffer in GPU memory (per-layer, with offset).
    MemAddress getMemPoolBaseAddress(LayerId layerId, DataRole role) const;

    // Pool group base address without per-layer offset.
    MemAddress getMemPoolBaseAddress(PoolGroupIndex pgIdx, PoolIndex poolIdx) const;

    // Per-layer storage attributes.
    LayerAttr const& getLayerAttr(LayerId layerId) const;

    // Address of a slot's buffer in a specific pool at a cache level.
    Address slotAddress(CacheLevel level, PoolGroupIndex pgIdx, SlotId slotId, int poolIdx) const;

    // Cache tier for a given level.
    CacheTier cacheTier(CacheLevel level) const;

    // NOTE: Python's get_statistics(level) returns a list over all pool groups.
    // C++ takes a single pgIdx for flexibility; the nanobind wrapper loops over
    // all pool groups to match Python's signature.
    StorageStatistics getStatistics(CacheLevel level = kGpuLevel, PoolGroupIndex pgIdx = 0) const;
    std::vector<float> getUtilization(CacheLevel level = kGpuLevel) const;
    float getOverallUtilization(CacheLevel level = kGpuLevel) const;

    // Pool-group slot count (number of pages).
    int numSlots(PoolGroupIndex pgIdx, CacheLevel level = kGpuLevel) const;

    // Layer-to-lifecycle mapping (for KvCacheManager queries).
    std::unordered_map<LayerId, LifeCycleId> const& layerToLifeCycleIds() const noexcept
    {
        return mLayerToLifeCycleIds;
    }

    // Per-lifecycle max slot utilization fraction (for scratch slot computation).
    Rational const& slotUtilFracMax(LifeCycleId lcId) const
    {
        return mSlotUtilFracMax.at(static_cast<size_t>(lcId));
    }

    friend class KvCacheManager;

private:
    // Constraint-based partitioning helpers.
    std::vector<int> computeMinSlotsFromConstraints(std::vector<BatchDesc> const& constraints, int tokensPerBlock,
        std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const;
    std::vector<int> computeSlotsForBatch(
        BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const;
    std::vector<size_t> slotsToBytes(std::vector<int> const& numSlots, int granularity) const;
    std::vector<int> computeSlotCountForLevel(CacheTierConfig const& tierConfig,
        std::vector<std::vector<int>> const& slotSizeLists, std::vector<float> const& ratio) const;
    size_t minQuotaForLevel(std::vector<std::vector<int>> const& slotSizeLists, int granularity) const;

    int mNumPools(PoolGroupIndex pgIdx) const;

    // Internal helpers.
    void _prepareFreeSlots(std::vector<std::vector<int>>& goals, // [level][pgIdx]
        CacheLevel lvlId,
        std::vector<std::vector<SharedPtr<Page>>>& fallenPages); // [pgIdx]

    void _batchedMigrate(PoolGroupIndex pgIdx, CacheLevel dstLevel, CacheLevel srcLevel,
        std::vector<SharedPtr<Page>> const& srcPages, bool updateSrc, bool defrag = false);

    PoolGroupBase& poolGroup(CacheLevel lvl, PoolGroupIndex pgIdx);

    LifeCycleRegistry const& mLifeCycles;
    std::vector<PoolGroupIndex> mLifeCycleGrouping; // lcId → pgIdx
    std::unordered_map<LayerId, LifeCycleId> mLayerToLifeCycleIds;
    StorageConfig mStorageConfig;

    // slot-to-page-index scale factors: [lcId][poolIdx]
    std::vector<std::vector<int>> mSlotToPageIndices;

    // Per-layer storage attributes for scratch slot management.
    std::map<LayerId, LayerAttr> mLayerAttributes;

    // Max slot utilization fraction per lifecycle (across all layers in that lifecycle).
    std::vector<Rational> mSlotUtilFracMax;

    // Whether SWA scratch reuse is enabled.
    std::optional<SwaScratchReuseConfig> mSwaScratchReuse;

    // Get buffer attributes for a (LayerId, DataRole) pair. Throws std::out_of_range if not found.
    // Mirrors Python's get_buffer_attr().
    BufferAttr const& getBufferAttr(LayerId layerId, DataRole role) const
    {
        auto it = mBufferAttr.find(BufferId{layerId, role});
        if (it == mBufferAttr.end())
            throw std::out_of_range("Unknown buffer id");
        return it->second;
    }

    // Buffer attributes keyed by BufferId.
    std::map<BufferId, BufferAttr> mBufferAttr;

    std::vector<SlotDesc> mSlotDescList; // indexed by PoolGroupIndex
    std::vector<int> mMinSlots;          // per pool group minimum slot count from constraints
    std::vector<CacheLevelManager> mLevels;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
