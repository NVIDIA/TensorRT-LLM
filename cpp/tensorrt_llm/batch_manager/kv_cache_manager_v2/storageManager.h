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

#include "kv_cache_manager_v2/coldPageCodec.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/eventSink.h"
#include "kv_cache_manager_v2/evictionController.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/storage/core.h"
#include "tensorrt_llm/common/assert.h"

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declarations.
class Page;
class KvCache;
class CopyEngine;
class StagingBufferManager;
struct BatchedLockTarget;

using MigrationRecorder
    = std::function<void(std::vector<SharedPtr<Page>> const&, std::vector<Slot> const&, CacheLevel, CacheLevel)>;
using DropRecorder = std::function<void(std::vector<SharedPtr<Page>> const&, CacheLevel)>;

// Immutable bidirectional mapping between lifecycles and pool groups.
class LifeCyclePoolGroupMapping
{
public:
    LifeCyclePoolGroupMapping() = default;
    explicit LifeCyclePoolGroupMapping(TypedVec<LifeCycleId, PoolGroupIndex> forward);

    [[nodiscard]] PoolGroupIndex poolGroup(LifeCycleId lifeCycle) const;
    [[nodiscard]] Span<LifeCycleId const> lifeCycles(PoolGroupIndex poolGroup) const;
    [[nodiscard]] PoolGroupIndex numPoolGroups() const noexcept;

    [[nodiscard]] TypedVec<LifeCycleId, PoolGroupIndex> const& forward() const noexcept
    {
        return mForward;
    }

private:
    TypedVec<LifeCycleId, PoolGroupIndex> mForward;
    std::vector<LifeCycleId> mInverse;
    std::vector<size_t> mPoolGroupOffsets;
};

// ---------------------------------------------------------------------------
// StorageStatistics — per-pool-group slot counts.
// ---------------------------------------------------------------------------
struct StorageStatistics
{
    TypedVec<PoolIndex, size_t> slotSizes;
    SlotCount total;     // total slots
    SlotCount free;      // free (unallocated) slots
    SlotCount evictable; // scheduled for eviction

    SlotCount available() const noexcept
    {
        return free + evictable;
    }

    SlotCount unavailable() const noexcept
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
    CacheLevelManager(TypedVec<LifeCycleId, PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cacheLevel,
        CacheTierConfig const& tierConfig, TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
        TypedVec<PoolGroupIndex, SlotCount> const& slotCountList,
        PooledPhysMemAllocator* gpuPhysMemAllocator = nullptr);

    // Compute pool size granularity for a given cache tier and quota.
    static size_t cacheTierGranularity(CacheTier tier, size_t quota);

    CacheLevel cacheLevel;
    CacheTier cacheTier;
    std::unique_ptr<CacheLevelStorage> storage;
    PerLevelEvictionController controller;

    PoolGroupIndex numPoolGroups() const noexcept
    {
        TLLM_CHECK_DEBUG_WITH_INFO(
            storage->numPoolGroups() == controller.numPoolGroups(), "Storage and controller disagree on numPoolGroups");
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
    // Takes ownership of coldPageCodec immediately; nullptr selects the default lossless codec.
    StorageManager(LifeCycleRegistry const& lifeCycles, StorageConfig const& config, int tokensPerBlock,
        std::unique_ptr<IKvCacheColdPageCodec> coldPageCodec = nullptr,
        std::optional<SwaScratchReuseConfig> swaScratchReuse = std::nullopt,
        std::optional<BatchDesc> const& typicalBatch = std::nullopt, std::vector<BatchDesc> const& constraints = {},
        std::optional<std::vector<float>> const& initialPoolRatio = std::nullopt,
        std::shared_ptr<EventSink> eventSink = nullptr, float maxUtilForResume = 1.0f);
    ~StorageManager();

    StorageManager(StorageManager const&) = delete;
    StorageManager& operator=(StorageManager const&) = delete;

    void destroy();

    // ---- Allocation -------------------------------------------------------

    // Allocate slots for all life cycles at the given cache level.
    // numSlotsPerLc[lcId] = how many slots to allocate for that life cycle.
    // Returns a vector indexed by lcId.
    TypedVec<LifeCycleId, std::vector<Slot>> newSlots(CacheLevel level,
        TypedVec<LifeCycleId, SlotCount> const& numSlotsPerLc, MigrationRecorder const& migrationRecorder = {},
        DropRecorder const& dropRecorder = {});

    TypedVec<LifeCycleId, std::vector<Slot>> newGpuSlots(TypedVec<LifeCycleId, SlotCount> const& numSlotsPerLc,
        MigrationRecorder const& migrationRecorder = {}, DropRecorder const& dropRecorder = {});

    // Allocate slots for a single pool group at the given cache level.
    // Returns numSlots Slot objects. Throws OutOfPagesError if allocation fails.
    std::vector<Slot> newSlotsForPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, SlotCount numSlots,
        MigrationRecorder const& migrationRecorder = {}, DropRecorder const& dropRecorder = {});

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
    void prepareFreeSlots(CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& requirements,
        MigrationRecorder const& migrationRecorder = {}, DropRecorder const& dropRecorder = {});

    // Force-evict pages from a level to free space.
    void forceEvict(CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& minNumPages,
        DropRecorder const& dropRecorder = {});

    // Dynamic cache level resizing.
    void adjustCacheLevel(CacheLevel level, std::optional<size_t> newQuota,
        TypedVec<PoolGroupIndex, float> const& ratioList,
        TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* persistentPages);
    void shrinkPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, SlotCount newNumSlots,
        std::vector<SharedPtr<Page>> const& persistentPages);
    void expandPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, SlotCount newNumSlots);

    // ---- Migration ---------------------------------------------------------

    // Migrate a batch of pages to GPU (used by batchedLockToGpu).
    void batchedMigrateToGpu(
        std::vector<BatchedLockTarget> const& targets, KvCache& kvCache, MigrationRecorder const& migrationRecorder);

    // Best-effort migration of grouped pages to a destination cache level.
    void prefetch(
        CacheLevel dstLevel, TypedVec<LifeCycleId, TypedVec<CacheLevel, std::vector<SharedPtr<Page>>>> const& pages);

    // ---- Query helpers -----------------------------------------------------

    LifeCycleRegistry const& lifeCycles() const noexcept
    {
        return mLifeCycles;
    }

    LifeCycle const& getLifeCycle(LifeCycleId lc) const;

    LifeCycleId numLifeCycles() const noexcept
    {
        return mHotPoolGroupMapping.forward().size();
    }

    PoolGroupIndex numPoolGroups(CacheLevel level) const
    {
        return mSlotDescLists.at(level).size();
    }

    PoolGroupIndex numPoolGroups() const
    {
        return numPoolGroups(kHotLevel);
    }

    TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList(CacheLevel level) const
    {
        return mSlotDescLists.at(level);
    }

    TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList() const
    {
        return slotDescList(kHotLevel);
    }

    TypedVec<LifeCycleId, PoolGroupIndex> const& lifeCycleGrouping(CacheLevel level) const
    {
        return poolGroupMapping(level).forward();
    }

    CacheLevel numCacheLevels() const noexcept
    {
        return mLevels.size();
    }

    bool isLastLevel(CacheLevel lvl) const noexcept
    {
        return lvl == numCacheLevels() - 1;
    }

    PoolGroupIndex getPoolGroupIndex(CacheLevel level, LifeCycleId lc) const;
    PoolGroupIndex getPoolGroupIndex(LifeCycleId lc) const;
    PoolIndex numPools(CacheLevel level, PoolGroupIndex pgIdx) const;
    PoolIndex numPools(PoolGroupIndex pgIdx) const;

    // Describe every hot-tier pool group: slot count, slot descriptor and per-pool base
    // addresses. Used to configure the cold-page codec and exposed through KvCacheManager.
    TypedVec<PoolGroupIndex, PoolGroupDesc> poolGroupDescs() const;

    // Return the byte size of each pool in a pool group.
    // Precomputed at construction and returned by reference: slot sizes derive from the
    // immutable slot descriptors, so this needs no invalidation and allocates nothing. It is
    // called once per migrated/dropped page by the statistics recorders.
    TypedVec<PoolIndex, size_t> const& slotSize(CacheLevel level, PoolGroupIndex pgIdx) const
    {
        return mSlotSizes.at(level).at(pgIdx);
    }

    TypedVec<PoolIndex, size_t> const& slotSize(PoolGroupIndex pgIdx) const
    {
        return slotSize(kHotLevel, pgIdx);
    }

    // Current ratio list for a cache level (proportional to byte usage per pool group).
    TypedVec<PoolGroupIndex, float> getRatioList(CacheLevel level) const;

    // Compute lifecycle allocation weights, then project them onto a cache level's pool grouping.
    TypedVec<LifeCycleId, float> ratioFromLength(
        CacheLevel level, int tokensPerBlock, int historyLength, int capacity) const;
    TypedVec<PoolGroupIndex, float> toPoolGroupRatio(
        CacheLevel level, TypedVec<LifeCycleId, float> const& lifeCycleRatio) const;

    // Apply stored min_slots constraint to a ratio list for GPU level.
    TypedVec<PoolGroupIndex, float> constrainPoolGroupRatio(TypedVec<PoolGroupIndex, float> const& ratio) const;

    // Byte address of a slot's buffer in GPU memory (per-layer, with offset).
    MemAddress getMemPoolBaseAddress(LayerId layerId, DataRole role) const;

    void copySlotData(LifeCycleId lifeCycle, CacheLevel dstLevel, CacheLevel srcLevel, SlotId dstSlotId,
        SlotId srcSlotId, CUstream stream);
    // Pool group base address without per-layer offset.
    MemAddress getMemPoolBaseAddress(PoolGroupIndex pgIdx, PoolIndex poolIdx) const;

    // Per-layer storage attributes.
    LayerAttr const& getLayerAttr(LayerId layerId) const;

    // Address of a slot's buffer in a specific pool at a cache level.
    Address slotAddress(CacheLevel level, PoolGroupIndex pgIdx, SlotId slotId, PoolIndex poolIdx) const;

    // Cache tier for a given level.
    CacheTier cacheTier(CacheLevel level) const;

    // NOTE: Python's get_statistics(level) returns a list over all pool groups.
    // C++ takes a single pgIdx for flexibility; the nanobind wrapper loops over
    // all pool groups to match Python's signature.
    StorageStatistics getStatistics(CacheLevel level = kHotLevel, PoolGroupIndex pgIdx = PoolGroupIndex{0}) const;
    TypedVec<PoolGroupIndex, float> getUtilization(CacheLevel level = kHotLevel) const;
    float getOverallUtilization(CacheLevel level = kHotLevel) const;

    // Pool-group slot count (number of pages).
    SlotCount numSlots(PoolGroupIndex pgIdx, CacheLevel level = kHotLevel) const;

    // Layer-to-lifecycle mapping (for KvCacheManager queries).
    std::unordered_map<LayerId, LifeCycleId> const& layerToLifeCycleIds() const noexcept
    {
        return mLayerToLifeCycleIds;
    }

    // Per-lifecycle max slot utilization fraction (for scratch slot computation).
    Rational const& slotUtilFracMax(LifeCycleId lcId) const
    {
        return mSlotUtilFracMax.at(lcId);
    }

    friend class KvCacheManager;
    // White-box test introspection reaches computePoolGroupSlotsForBatch() via friendship
    // (mirrors Python's _compute_slots_for_batch).
    friend class KvCacheIntrospection;

private:
    using PageQueue = std::deque<SharedPtr<Page>>;
    using PagesByLifeCycle = TypedVec<LifeCycleId, PageQueue>;
    using MigrationBatchKey = std::pair<CacheLevel, LayerGroupId>;

    // Minimum per-pool-group slot counts to support a BatchDesc.
    TypedVec<PoolGroupIndex, SlotCount> computePoolGroupSlotsForBatch(
        BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const;

    TypedVec<LifeCycleId, SlotCount> computeSlotsForBatch(
        BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const;

    // Constraint floors apply only to the hot level. Lifecycle demands remain available only as ratio weights.
    TypedVec<PoolGroupIndex, SlotCount> computePoolGroupMinSlotsFromConstraints(
        std::vector<BatchDesc> const& constraints, int tokensPerBlock,
        std::optional<SwaScratchReuseConfig> const& swaScratchReuse, float maxUtilForResume = 1.0f) const;
    TypedVec<LifeCycleId, SlotCount> computeSlotsFromConstraints(std::vector<BatchDesc> const& constraints,
        int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse,
        float maxUtilForResume = 1.0f) const;
    TypedVec<LifeCycleId, float> ratioFromBatch(BatchDesc const& batch, int tokensPerBlock,
        std::optional<SwaScratchReuseConfig> const& swaScratchReuse, size_t granularity) const;
    // Project source-level lifecycle byte weights onto destination pool groups while preserving the implied
    // lifecycle slot-count proportions.
    TypedVec<PoolGroupIndex, float> projectPoolGroupRatio(
        CacheLevel srcLevel, CacheLevel dstLevel, TypedVec<LifeCycleId, float> const& srcLifeCycleRatio) const;
    TypedVec<LifeCycleId, size_t> slotsToBytes(
        TypedVec<LifeCycleId, SlotCount> const& numSlots, size_t granularity) const;
    TypedVec<PoolGroupIndex, size_t> slotsToBytes(
        TypedVec<PoolGroupIndex, SlotCount> const& numSlots, size_t granularity) const;
    TypedVec<PoolGroupIndex, SlotCount> computeSlotCountForLevel(CacheTierConfig const& tierConfig,
        TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
        TypedVec<PoolGroupIndex, float> const& ratio, TypedVec<PoolGroupIndex, SlotCount> const& minSlots) const;
    size_t minQuotaForLevel(TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
        size_t granularity, TypedVec<PoolGroupIndex, SlotCount> const& minSlots) const;

    // Internal helpers.
    // Register the slot descriptors for the next cache level and derive its slot-size table.
    // This is the only mutator of mSlotDescLists/mSlotSizes, so the two cannot go out of sync.
    // Levels must be appended in order -- the hot tier first, then the cold tiers once the codec
    // has been queried -- and the returned level lets callers assert that ordering.
    CacheLevel appendLevelSlotDescList(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescs);

    [[nodiscard]] auto makeEvictionRollbackGuard(TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const& evicted);

    void _prepareFreeSlots(TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>>& goals, CacheLevel lvlId,
        PagesByLifeCycle& fallenPages, MigrationRecorder const& migrationRecorder = {},
        DropRecorder const& dropRecorder = {});

    std::optional<std::vector<Slot>> _batchedMigrate(CacheLevel dstLevel, CacheLevel srcLevel,
        std::vector<SharedPtr<Page>> const& srcPages, bool updateSrc, MigrationRecorder const& migrationRecorder = {},
        bool defrag = false);
    [[nodiscard]] LayerGroupId getMigrationBatchingLayerGroupId(
        CacheLevel dstLevel, CacheLevel srcLevel, LifeCycleId lifeCycle) const;
    void submitMigrationBatch(CacheLevel dstLevel, CacheLevel srcLevel, LayerGroupId batchingLayerGroupId,
        PageIndexPair const* pageIndices, size_t numPages, CUstream stream);

    template <typename Submit>
    bool submitColdPageCodec(PageIndexLocation location, PageIndexPair const* pageIndices, size_t numPages,
        CUstream stream, Submit&& submit);

    PoolGroupBase& poolGroup(CacheLevel lvl, PoolGroupIndex pgIdx);

    LifeCyclePoolGroupMapping const& poolGroupMapping(CacheLevel level) const
    {
        TLLM_CHECK(level >= CacheLevel{0} && level < mSlotDescLists.size());
        // Every cold level uses the same encoded page layout and therefore shares one pool-group mapping.
        return level == kHotLevel ? mHotPoolGroupMapping : mColdPoolGroupMapping;
    }

    LifeCycleRegistry const& mLifeCycles;
    std::shared_ptr<EventSink> mEventSink;
    LifeCyclePoolGroupMapping mHotPoolGroupMapping;
    LifeCyclePoolGroupMapping mColdPoolGroupMapping;
    // Codec batching representative for each lifecycle.
    TypedVec<LifeCycleId, LayerGroupId> mBatchingLayerGroupIds;
    // Codec-selected PageIndexPair memory location for each lifecycle.
    TypedVec<LifeCycleId, PageIndexLocation> mPageIndexLocations;
    std::unordered_map<LayerId, LifeCycleId> mLayerToLifeCycleIds;
    StorageConfig mStorageConfig;

    // slot-to-page-index scale factors: [lcId][poolIdx]
    TypedVec<LifeCycleId, TypedVec<PoolIndex, int>> mSlotToPageIndices;

    // Per-layer storage attributes for scratch slot management.
    std::map<LayerId, LayerAttr> mLayerAttributes;

    // Max slot utilization fraction per lifecycle (across all layers in that lifecycle).
    TypedVec<LifeCycleId, Rational> mSlotUtilFracMax;

    // Whether SWA scratch reuse is enabled.
    std::optional<SwaScratchReuseConfig> mSwaScratchReuse;

    // Owns the configured codec and outlives the storage and staging resources declared below.
    std::unique_ptr<IKvCacheColdPageCodec> mColdPageCodec;

    // Optional model-sized page staging is owned here and borrowed by CopyEngine.
    std::unique_ptr<StagingBufferManager> mPageStagingManager;
    std::unique_ptr<CopyEngine> mCopyEngine;

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

    TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotDesc>> mSlotDescLists;
    // Slot sizes per (level, pool group), built from mSlotDescLists.
    TypedVec<CacheLevel, TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>>> mSlotSizes;
    TypedVec<PoolGroupIndex, SlotCount> mMinSlots;
    // All GPU cache levels borrow this allocator. It must outlive mLevels.
    std::unique_ptr<PooledPhysMemAllocator> mGpuPhysMemAllocator;
    TypedVec<CacheLevel, CacheLevelManager> mLevels;

    static constexpr size_t kDefaultPageStagingBytes = 64u << 20u;
    static constexpr size_t kPageStagingDepth = 3;
    static constexpr size_t kMaxIndexBatchBytes = 64u << 10u;
    static constexpr size_t kIndexStagingBytes = 256u << 10u;
    // Allocated only when a codec requests device indices; protected through codec completion.
    std::unique_ptr<StagingBufferManager> mIndexStagingManager;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
