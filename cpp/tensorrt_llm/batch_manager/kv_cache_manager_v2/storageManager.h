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
#include "kv_cache_manager_v2/storeLevel.h"
#include "tensorrt_llm/common/assert.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Forward declarations.
class Page;
class KvCache;
class CopyEngine;
class StagingBufferManager;

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

    // Change a page's eviction priority. A page scheduled for eviction is re-queued at the
    // most-recently-used end of its new priority's queue at the same cache level. Eviction
    // order only: the page's data, status, slot and level are unchanged. Caller holds the
    // manager's exclusive API lock.
    void setPriority(Page& page, Priority priority);

    // Check if a page is evictable (optionally at a target level).
    bool isEvictable(Page const& page, std::optional<CacheLevel> level = std::nullopt) const noexcept;

    // ---- Demotion (evict-first marks) ---------------------------------------
    //
    // A demoted page is scheduled under kPriorityDemoted instead of its own priority (see
    // Page::evictionPriority()), so eviction at its level picks it before every other page. The mark
    // follows the page when it falls to a lower level. All members require the exclusive API lock.

    //! Mark a scheduled, DROPPABLE committed page evict-first and re-queue it at the back of the demoted
    //! list (FIFO by demotion time). Returns false, changing nothing, for any other page or when the page
    //! is already demoted.
    bool demote(Page& page);

    //! Clear the mark. A scheduled page is re-queued at the back of its own priority list; an unscheduled
    //! page only loses the mark. No-op when the page is not demoted.
    void undemote(Page& page);

    //! Called by ~Page for a page destroyed while demoted.
    void noteDemotedPageDestroyed() noexcept
    {
        ++mNumDestroyedDemotedPages;
    }

    struct DemotionStats
    {
        uint64_t destroyedDemotedPages = 0; // demoted pages destroyed (evicted, cascaded or cleared)
        uint64_t undemotedPages = 0;        // demoted pages rescued by a hold or a re-commit
    };

    [[nodiscard]] DemotionStats demotionStats() const noexcept
    {
        return {mNumDestroyedDemotedPages, mNumUndemotedPages};
    }

    //! The remote host level, if one is configured (always the last level).
    [[nodiscard]] std::optional<CacheLevel> remoteHostLevel() const noexcept
    {
        return mRemoteHostLevel;
    }

    [[nodiscard]] bool isRemoteHostLevel(CacheLevel level) const noexcept
    {
        return mRemoteHostLevel.has_value() && level == *mRemoteHostLevel;
    }

    //! Counters of the remote host level. All zero, with level -1, when none is configured; the host-to-host batch
    //! counts are kept either way. Lock-free; safe after destroy().
    struct RemoteHostStats
    {
        int level = -1;
        uint64_t lentBytes = 0;              // imported capacity
        uint64_t numChunks = 0;              // imported chunks
        uint64_t demotedPages = 0;           // pages copied into the remote level
        uint64_t demotedBytes = 0;           // bytes copied into the remote level
        uint64_t promotedPages = 0;          // pages copied out of the remote level
        uint64_t promotedBytes = 0;          // bytes copied out of the remote level
        uint64_t gpuKernelHostCopies = 0;    // host-to-host batches on the GPU kernel (remote routes)
        uint64_t hostFunctionHostCopies = 0; // host-to-host batches on cuLaunchHostFunc (local host only)
    };

    [[nodiscard]] RemoteHostStats remoteHostStats() const noexcept;

    //! Reserved address ranges (base, bytes) of the remote host level's pools, as built; empty without one.
    [[nodiscard]] std::vector<std::pair<MemAddress, size_t>> remoteHostAddressRanges() const;

    // ---- Store level ---------------------------------------------------------

    //! The store level, if one is configured (always the last level, below the local host level).
    [[nodiscard]] std::optional<CacheLevel> storeLevel() const noexcept
    {
        return mStoreLevel;
    }

    [[nodiscard]] bool isStoreLevel(CacheLevel level) const noexcept
    {
        return mStoreLevel.has_value() && level == *mStoreLevel;
    }

    //! The IO ordering of the store level. Requires one (see storeLevel()).
    [[nodiscard]] StoreLevel& storeIo() const
    {
        TLLM_CHECK_WITH_INFO(mStoreIo != nullptr, "no store level is configured");
        return *mStoreIo;
    }

    //! Counters of the store level (level -1 when none is configured). Lock-free; safe after destroy().
    [[nodiscard]] StoreLevelStats storeLevelStats() const noexcept;

    //! Whether a page may be reused by a match: false only for a store page the degrade fault mode holds unreadable
    //! (StoreLevel::slotReadable()), which is then counted. Requires the manager's lock (shared or exclusive).
    [[nodiscard]] bool storePageReadable(Page const& page) const;

    //! Degrade fault mode: folds the completed store ops in (StoreLevel::reapCompletedOps()). No-op otherwise.
    //! Requires the exclusive API lock.
    void reapStoreOps();

    //! The pinned page staging buffer as (base, bytes); (0, 0) without one.
    [[nodiscard]] std::pair<MemAddress, size_t> pageStagingRange() const noexcept;

    //! Evict pages of `level` one level down until each pool group has at least its share of `bytes` free (shares
    //! follow the level's current pool-group byte ratio). Capped by the evictable pages and, when the next level is
    //! the last, by the room there (free plus droppable slots): never throws OutOfPagesError (a refusal still raised
    //! below is counted as a reserve failure and evicts nothing). Returns the slots evicted. Requires the exclusive
    //! API lock.
    SlotCount keepFreeBytes(CacheLevel level, size_t bytes);

    // Ensure the requested free slots exist at this level (evicting pages as needed).
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

    // Migrate pages excluded from eviction to one destination level. Locked pages cannot move. With a store level,
    // store pages bound for the GPU first move to the host level (stageStorePagesOnHost()); dropRecorder sees the
    // pages that makes room for dropped.
    void batchedMigrate(CacheLevel dstLevel, std::vector<SharedPtr<Page>> const& pages,
        MigrationRecorder const& migrationRecorder, DropRecorder const& dropRecorder = {});

    //! Demote complete locked sparse pages on the requesting owner's stream, updating every live owner.
    //! Caller holds the manager's exclusive lock. Allocation/copy failure preserves GPU ownership.
    void offloadSparsePages(KvCache& requestingCache, std::vector<SharedPtr<Page>> const& pages,
        MigrationRecorder const& migrationRecorder = {}, DropRecorder const& dropRecorder = {});

    //! Promote locked sparse host pages into one shared GPU allocation per page, updating all owners.
    //! Caller holds the manager's exclusive lock. Allocation/copy failure preserves host ownership.
    void promoteSparsePages(CUstream stream, std::vector<SharedPtr<Page>> const& pages,
        MigrationRecorder const& migrationRecorder = {}, DropRecorder const& dropRecorder = {});

    // Best-effort migration of grouped pages to a destination cache level. Returns how many pages
    // it moved off the disk tier. If a later batch fails, earlier batches may already have migrated.
    int64_t prefetch(
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
    // `beamWidth` / `promptLength` describe the typical request the weights are derived from; only
    // the blocks from the prompt tail onward are replicated per beam (see KVCacheDesc).
    TypedVec<LifeCycleId, float> ratioFromLength(CacheLevel level, int tokensPerBlock, int historyLength, int capacity,
        int beamWidth = 1, int promptLength = 0) const;
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
    void _migrateLockedSparsePages(CUstream stream, CacheLevel dstLevel, std::vector<SharedPtr<Page>> const& pages,
        KvCache const* offloadingCache, MigrationRecorder const& migrationRecorder, DropRecorder const& dropRecorder);
    [[nodiscard]] LayerGroupId getMigrationBatchingLayerGroupId(
        CacheLevel dstLevel, CacheLevel srcLevel, LifeCycleId lifeCycle) const;
    void submitMigrationBatch(CacheLevel dstLevel, CacheLevel srcLevel, LayerGroupId batchingLayerGroupId,
        PageIndexPair const* pageIndices, size_t numPages, CUstream stream);

    template <typename Submit>
    bool submitColdPageCodec(PageIndexLocation location, PageIndexPair const* pageIndices, size_t numPages,
        CUstream stream, Submit&& submit);

    //! Encode hot pages into cold pages at dstBasePtr (indices: dst = cold page ordinal, src = hot slot).
    bool encodeColdBatch(LayerGroupId batchingLayerGroupId, void* dstBasePtr, PageIndexPair const* pageIndices,
        size_t numPages, CUstream stream);
    //! Decode cold pages at srcBasePtr into hot pages (indices: dst = hot slot, src = cold page ordinal).
    bool decodeColdBatch(LayerGroupId batchingLayerGroupId, void const* srcBasePtr, PageIndexPair const* pageIndices,
        size_t numPages, CUstream stream);

    //! Moves `count` cold pages between the page staging buffer (page i at stagingBase + i * page bytes) and a cold
    //! level that the GPU cannot reach: toCold writes pageIndices[i].dst from staging, otherwise pageIndices[i].src
    //! is read into staging.
    using ColdHop = std::function<void(
        bool toCold, MemAddress stagingBase, PageIndexPair const* pageIndices, size_t count, CUstream stream)>;

    //! GPU <-> a cold level the GPU cannot reach, through the page staging buffer: the codec runs between GPU and
    //! staging, `hop` between staging and the cold level.
    void submitStagedColdBatch(CacheLevel dstLevel, CacheLevel srcLevel, LayerGroupId batchingLayerGroupId,
        PageIndexPair const* pageIndices, size_t numPages, CUstream stream, ColdHop const& hop);

    //! Moves the pages of `pages` that are in the store level to the host level above it (freeing host slots first),
    //! so that a following migration to the GPU is the plain host-to-GPU copy: the store writes straight into
    //! registered host slots, many rows per op, instead of passing page by page through the staging ring. Returns
    //! the pages moved: none when the host level cannot make room, and the store pages then take the direct route.
    //! The caller keeps the pages out of eviction.
    size_t stageStorePagesOnHost(std::vector<SharedPtr<Page>> const& pages, MigrationRecorder const& migrationRecorder,
        DropRecorder const& dropRecorder);

    //! Every route into or out of the store level, as store IO ops on `stream` (see StoreLevel).
    void submitStoreBatch(CacheLevel dstLevel, CacheLevel srcLevel, LayerGroupId batchingLayerGroupId,
        PageIndexPair const* pageIndices, size_t numPages, CUstream stream);

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

    // slot-to-page-index scale factors: [lcId][poolIdx]
    TypedVec<LifeCycleId, TypedVec<PoolIndex, int>> mSlotToPageIndices;

    // Per-layer storage attributes for scratch slot management.
    std::map<LayerId, LayerAttr> mLayerAttributes;

    // Max slot utilization fraction per lifecycle (across all layers in that lifecycle).
    TypedVec<LifeCycleId, Rational> mSlotUtilFracMax;

    // Whether SWA scratch reuse is enabled.
    std::optional<SwaScratchReuseConfig> mSwaScratchReuse;

    // Whether the caller supplied a codec (declared before mColdPageCodec, which takes it).
    bool mCustomColdPageCodec = false;
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

    // Demotion counters. Plain integers: mutated only under the exclusive API lock.
    uint64_t mNumDestroyedDemotedPages = 0;
    uint64_t mNumUndemotedPages = 0;

    // Remote host level and its traffic counters (relaxed atomics; read by the stats line without the lock). The
    // lent size, chunk count and address ranges are fixed when the level is built.
    std::optional<CacheLevel> mRemoteHostLevel;
    size_t mRemoteLentBytes = 0;
    size_t mRemoteNumChunks = 0;
    std::vector<std::pair<MemAddress, size_t>> mRemoteAddressRanges;
    // Host-to-host copy batches by route: the GPU kernel (a remote level on either side) or cuLaunchHostFunc.
    std::atomic<uint64_t> mGpuKernelHostCopies{0};
    std::atomic<uint64_t> mHostFunctionHostCopies{0};
    std::atomic<uint64_t> mRemoteDemotedPages{0};
    std::atomic<uint64_t> mRemoteDemotedBytes{0};
    std::atomic<uint64_t> mRemotePromotedPages{0};
    std::atomic<uint64_t> mRemotePromotedBytes{0};

    // Store level, its IO ordering (never reset, so the stats stay readable after destroy()) and its traffic
    // counters (relaxed atomics, read by the stats line without the lock). Quota and slot count are fixed at build.
    std::optional<CacheLevel> mStoreLevel;
    std::shared_ptr<StoreLevel> mStoreIo;
    uint64_t mStoreQuotaBytes = 0;
    uint64_t mStoreNumSlots = 0;
    std::atomic<uint64_t> mStoreDemotedPages{0};
    std::atomic<uint64_t> mStoreDemotedBytes{0};
    std::atomic<uint64_t> mStorePromotedPages{0};
    std::atomic<uint64_t> mStorePromotedBytes{0};
    std::atomic<uint64_t> mStorePromotedToGpuPages{0};
    std::atomic<uint64_t> mStorePrefetchedPages{0};
    std::atomic<uint64_t> mStoreReserveCalls{0};
    std::atomic<uint64_t> mStoreReserveEvictedSlots{0};
    std::atomic<uint64_t> mStoreStagingFallbacks{0};
    std::atomic<uint64_t> mStoreReserveFailures{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
