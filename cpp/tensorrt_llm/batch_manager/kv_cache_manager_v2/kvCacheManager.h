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
#include "kv_cache_manager_v2/coldPageCodec.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/eventSink.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/movingAverage.h"
#include "kv_cache_manager_v2/sparseCache.h"
#include "kv_cache_manager_v2/stats.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/poison.h"
#include "kv_cache_manager_v2/utils/reentrantSharedMutex.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// ExpandedBuffer / AggregatedPageDesc — returned by getAggregatedPages().
// ---------------------------------------------------------------------------
struct ExpandedBuffer
{
    BufferId id;
    int expansion; // expansion factor (tokens_per_block / tokens_per_block_override)
};

struct AggregatedPageDesc
{
    MemAddress base;                     // pool base address + buffer offset
    size_t size;                         // byte span of this aggregated buffer group
    size_t stride;                       // slot size (bytes per slot in the pool group)
    LifeCycleId layerGroupId;            // pool group / life-cycle id
    std::vector<ExpandedBuffer> buffers; // constituent buffers in offset order
};

// ---------------------------------------------------------------------------
// PageIndexConverter — convert base page index → kernel page indices.
// ---------------------------------------------------------------------------
struct PageIndexConverter
{
    int scale;
    int expansion;
    int layerOffset = 0;          // sub-page offset within coalesced slot
    int scratchPagesPerBlock = 1; // sub-pages per block for scratch allocation

    // Convert a sequence of base page indices to per-layer page indices.
    // indexMode: SHARED (default) or PER_LAYER. When scratch is active, must be PER_LAYER.
    // scratchRange: block ordinals served from scratch slots, as KvCache::getSwaScratchRange()
    //               reports them; empty means none are. scratchSlots are their page indices,
    //               from KvCache::getScratchSlots().
    // Mirrors _kv_cache_manager.py::PageIndexConverter.__call__.
    std::vector<int> operator()(std::vector<int> const& baseIndices,
        std::optional<PageIndexMode> indexMode = std::nullopt, HalfOpenRange<BlockOrdinal> scratchRange = {},
        Span<int const> scratchSlots = {}) const;

    // Backward-compatible single-index overload.
    std::vector<int> operator()(int baseIndex) const;
};

//! What one life cycle's sparse fetch path is shaped like, as _computeSparseCacheShape
//! derives it from the configuration.
struct SparseCacheShape
{
    //! Cells a request caches per buffer, counting history and everything in flight.
    int maxUnits = 0;
    //! Sub-slots one page divides into. One when the life cycle has no fetch path, so the
    //! page and the unit coincide.
    int unitsPerPage = 1;
    //! Sparse buffers this life cycle has.
    int buffers = 0;
    //! Units a step selects per buffer.
    int selection = 0;
    //! Layers of this life cycle that carry a sparse cache.
    int layers = 0;
    //! Selections those layers resolve between them. One where each selects for itself,
    //! fewer where several share one, and what the metadata's first dimension counts.
    int groups = 0;
};

//! One sparse buffer of a life cycle, with everything the cache derives from it.
//!
//! Ordered layers-outermost, then by pool within a layer, so that both a layer's buffers and
//! a (layer, pool) run are consecutive. The fetch needs the first, because it resolves one
//! layer; the copy needs the second, because a pool fixes a cell's size, the slot stride it
//! is read from, and the unit layout, so one launch serves one pool.
struct SparseBufferPlan
{
    BufferId id;
    PoolIndex poolIndex; //!< Which hot pool holds its bytes; pools partition by size.
    size_t size;         //!< One cell.
    uint32_t pageOffset; //!< Its offset within its hot slot.
    SelectionUnitLayout layout;
    int32_t layerRow;    //!< Metadata row: the dense index of its layer in the group.
    int32_t slotInPool;  //!< Its index among the life cycle's buffers sharing its pool.
};

class Batch;

// ---------------------------------------------------------------------------
// KvCacheManager — top-level KV cache manager.
// Mirrors Python's KVCacheManager.
// ---------------------------------------------------------------------------
class KvCacheManager : public std::enable_shared_from_this<KvCacheManager>
{
public:
    // coldPageCodec is consumed when construction is invoked, including when construction throws; nullptr selects
    // the default lossless codec.
    explicit KvCacheManager(KVCacheManagerConfig const& config, std::shared_ptr<EventSink> eventSink = nullptr,
        std::unique_ptr<IKvCacheColdPageCodec> coldPageCodec = nullptr);
    ~KvCacheManager();

    KvCacheManager(KvCacheManager const&) = delete;
    KvCacheManager& operator=(KvCacheManager const&) = delete;

    // ---- Lifecycle --------------------------------------------------------

    void shutdown();

    // Number of not-yet-destroyed managers in this process. A manager counts from the start of its
    // construction, so one whose constructor throws is counted for the duration of that attempt.
    [[nodiscard]] static uint32_t numLiveManagers() noexcept;

    // Clear all reusable (committed) blocks from the radix tree.
    void clearReusableBlocks();

    // ---- KvCache creation -------------------------------------------------

    // Create a new KvCache. Returned cache is SUSPENDED; call resume() with a stream.
    // input_tokens:         optional sequence to match against existing cached blocks.
    // priorityCb:           optional priority override per block.
    // expectedPromptLength: full prompt token count marking the prefill->generation boundary.
    //                       Beam expansion shares blocks entirely before this boundary and
    //                       copies the writable tail into each additional beam. For beam search,
    //                       pass the actual prompt length if inputTokens is absent or shortened
    //                       for reuse matching. Defaults to non-empty inputTokens.size(); without
    //                       either value, beam expansion uses a zero shared-prefix boundary.
    //                       Once historyLength reaches it, later capacity growth is also recorded
    //                       as generation-phase allocation stats.
    // textOnly:             per-sequence override of the text-only (digest-free) guarantee;
    //                       nullopt inherits the manager config default.
    // enableRequestStats:   collect request-local allocation and reuse statistics even when
    //                       manager-level statistics are disabled.
    // inputTokens is a non-owning view; the caller must keep the underlying buffer alive for the
    // duration of the call (matching reads it but never stores it).
    std::shared_ptr<KvCache> createKvCache(ReuseScope reuseScope = {}, TokenSpan inputTokens = {},
        std::optional<RequestIdType> id = std::nullopt, KvCache::PriorityCb priorityCb = {},
        std::optional<int> expectedPromptLength = std::nullopt, std::optional<bool> textOnly = std::nullopt,
        bool enableRequestStats = false,
        std::optional<executor::KvCacheRetentionConfig> retentionConfig = std::nullopt);

    //! Restore expired, unreferenced pages to the default retention priority.
    void refreshRetention();

    // Move reuse-matched pages without claiming or extending their request retention policy.
    bool prefetchReuse(ReuseScope reuseScope, TokenSpan inputTokens, CacheLevel target, bool knownNoDigest);

    // knownNoDigest: from external text_only knowledge, never a scan (see Hasher::update).
    // Defaults false (safe: the scanning path is taken).
    BlockRadixTree::ReuseMatch matchReuse(
        ReuseScope const& reuseScope, TokenSpan inputTokens, bool knownNoDigest = false) const;
    int probeReuse(ReuseScope reuseScope = {}, TokenSpan inputTokens = {}, bool knownNoDigest = false) const;

    // Read-only, advisory key of the first full block past the reusable prefix.
    // Uses the same fresh, window-aware match as probeReuse without acquiring pages.
    std::optional<BlockKey> probeFirstNewBlockKey(
        ReuseScope reuseScope = {}, TokenSpan inputTokens = {}, bool knownNoDigest = false) const;

    // ---- Memory pool queries -----------------------------------------------

    // Base address of the memory pool. When indexMode is PER_LAYER, returns pool group base
    // (without per-layer offset). When SHARED, returns per-layer base (with offset baked in).
    MemAddress getMemPoolBaseAddress(
        LayerId layerId, DataRole role, std::optional<PageIndexMode> indexMode = std::nullopt) const;

    int getPageStride(LayerId layerId, DataRole role) const;
    size_t getPageIndexUpperBound(LayerId layerId, DataRole role) const;

    //! Whether this buffer was declared sparse, i.e. whether its pages are host memory
    //! read through the sparse GPU cache rather than device memory read directly.
    [[nodiscard]] bool isSparse(LayerId layerId, DataRole role) const;

    //! Bytes `level` holds in pools whose pages are `medium`. A device level's own tier and
    //! its sparse backing medium are accounted separately, the quota paying only for the
    //! former, so a caller counting what it pins asks for each in turn.
    [[nodiscard]] size_t getNumBytes(CacheLevel level, CacheTier medium) const;

    //! The medium a sparse buffer's pages are -- host memory the GPU reads through a device
    //! mapping -- or nullopt when no life cycle is sparse. Pass it to
    //! getNumBytes(kHotLevel, ...) to count what sparse buffers hold.
    [[nodiscard]] std::optional<CacheTier> sparseBackingMedium() const;

    // Scale factor: base_page_index * scale → kernel page index.
    int getPageIndexScale(LayerId layerId, DataRole role) const;

    // Composite converter (scale + expansion).
    PageIndexConverter getPageIndexConverter(LayerId layerId, DataRole role) const;

    // Group a set of BufferIds into contiguous AggregatedPageDesc descriptors.
    // Mirrors Python's KVCacheManager.get_aggregated_pages().
    std::vector<AggregatedPageDesc> getAggregatedPages(std::vector<BufferId> const& buffers) const;

    TypedVec<PoolGroupIndex, PoolGroupDesc> poolGroupDescs() const;

    // ---- Query / info ------------------------------------------------------

    int tokensPerBlock() const noexcept;
    bool enablePartialMatch() const noexcept;

    // Partial commit is independent of partial matching so beam search can
    // retain full-block reuse without publishing a writable prompt tail.
    bool enablePartialCommit() const noexcept;

    bool commitMinSnapshot() const noexcept
    {
        return mConfig.commitMinSnapshot;
    }

    // Deployment-level text-only guarantee (see KVCacheManagerConfig::textOnly).
    bool textOnly() const noexcept
    {
        return mConfig.textOnly;
    }

    bool isSwaScratchReuseEnabled() const noexcept
    {
        return mConfig.enableSwaScratchReuse();
    }

    //! The most scratch slots one request can need, over every life cycle and buffer,
    //! given the widest page table a row will get. Zero where nothing can hold scratch.
    //!
    //! Sizes scratch staging; a caller that stages slots itself asks rather than re-deriving
    //! which blocks can be stale.
    [[nodiscard]] int maxScratchSlots(int maxBlocksPerSeq) const;

    //! Whether managed KV caches support the given page index mode for one life cycle,
    //! which is the granularity a Batch decides at. True means every request supports it;
    //! nullopt means it depends on the individual KvCache, which must then be asked. Never
    //! false, so an engaged optional means support.
    //!
    //! Only scratch-slot substitution can make one layer's converted indices differ from
    //! another's in the same life cycle, so SHARED is definitive for a life cycle that can
    //! never hold scratch -- anything computeScratchRange() returns empty for, which is
    //! every life cycle without a sliding window.
    //!
    //! Configuration only: the answer never depends on what a request is currently doing,
    //! which is what lets a Batch fix its mode at construction.
    std::optional<bool> supportsIndexMode(PageIndexMode mode, LifeCycleId lcId) const;

    bool allowSeqRebasing() const noexcept
    {
        return true;
    }

    int numLayers() const noexcept;

    std::vector<LayerId> layerIds() const;

    //! Deprecated: use getBufferGroupId(). Returns the shared group of a uniform layer.
    //! Throws common::TllmException if the queried layer spans several lifecycle groups.
    LayerGroupId getLayerGroupId(LayerId layerId) const;

    //! Cache group of one buffer; throws std::out_of_range for an unknown buffer.
    CacheGroupId getBufferGroupId(LayerId layerId, DataRole const& role) const;

    //! Deprecated: use bufferGrouping(). Partitions uniform layers by lifecycle.
    //! Throws common::TllmException if any layer spans several groups. Member order
    //! is unspecified; use poolGroupDescs() to inspect physical buffer ordering.
    TypedVec<LayerGroupId, std::vector<LayerId>> layerGrouping() const;

    //! Buffer membership indexed by CacheGroupId. Each buffer appears once.
    //! Member order is unspecified and does not describe physical memory ordering.
    TypedVec<CacheGroupId, std::vector<BufferId>> bufferGrouping() const;

    // Iterator over all buffer identifiers. Mirrors Python's all_buffer_ids property.
    std::vector<BufferId> allBufferIds() const;

    // Sorted by CacheLevel from warm to cold. Mirrors Python's cache_tier_list property.
    TypedVec<CacheLevel, CacheTier> cacheTierList() const;

    // Get the max possible sequence length limited by GPU memory pools.
    // Mirrors Python's clamp_max_seq_len_for_mem().
    int clampMaxSeqLenForMem(int batchSize, int tokenNumUpperBound) const;

    // ---- Resize -----------------------------------------------------------

    bool resize(CacheLevel level, size_t quota, bool bestEfforts = false);
    size_t getQuota(CacheLevel level) const;

    // ---- Statistics -------------------------------------------------------

    // Independent per-pool values sampled together under the shared API lock.
    TypedVec<PoolGroupIndex, StorageStatistics> getStorageStatistics(CacheLevel cacheLevel = kHotLevel) const;
    // Pool-group numbering is level-specific; cold grouping can differ from the hot layout.
    TypedVec<LifeCycleId, PoolGroupIndex> getLifeCyclePoolGroupIndices(CacheLevel cacheLevel = kHotLevel) const;

    // Internal commit* and recordDiskPrefetchBlocks helpers require the caller's exclusive API lock.
    void commitStats(KVCacheStatsDelta const& stats, IterationStatsByLifeCycle const& iterationStatsByLifeCycle = {});
    KVCacheStatsDelta getCommittedStats() const;
    IterationStatsByLifeCycle getAndResetIterationStats();
    PeakBlockStatsByPoolGroup getAndResetIterationPeakBlockStats(CacheLevel cacheLevel);
    // Drain every level at once. The peaks are already tracked as one per-level record, so a
    // caller that wants all of them should not take that record apart one level at a time.
    PeakBlockStatsByCacheLevel getAndResetIterationPeakBlockStatsByLevel();

    void commitSsmSnapshotIterationStats(SsmSnapshotIterationStatsByLifeCycle const& statsByLifeCycle);
    SsmSnapshotIterationStatsByLifeCycle getAndResetSsmSnapshotIterationStats();

    // Per-cache-level split of the reuse block counts, committed alongside the scalar
    // iteration stats so both views cover exactly the same requests.
    void commitReusedBlocksByLevel(ReusedBlocksByLevelByLifeCycle const& byLifeCycle);
    ReusedBlocksByLevelByLifeCycle getAndResetIterationReusedBlocksByLevel();

    // The KV-connector-loaded share of the reused blocks, committed alongside the scalar iteration
    // stats that already include it.
    void commitConnectorReusedBlocks(ConnectorReusedBlocksByLifeCycle const& byLifeCycle);
    ConnectorReusedBlocksByLifeCycle getAndResetIterationConnectorReusedBlocks();

    // Count one ACTIVE->SUSPENDED transition for the current iteration window.
    void recordRequestSuspended();
    // Count one preemption recovery for the current iteration window. Only a
    // previously-ACTIVE cache that was suspended and then successfully resumed
    // counts; a freshly-created cache is activated by its first resume(), but
    // that is an admission, not a recovery, and is not counted.
    void recordRequestResumed();
    // Add the blocks a prefetch call actually migrated off disk in the current iteration window.
    // Independent of reuse-hit attribution, which asks where matched tokens lived rather than
    // what a prefetch moved.
    void recordDiskPrefetchBlocks(int64_t numBlocks);
    // Return the number of disk-prefetched blocks since the last drain and reset it.
    int64_t getAndResetIterationDiskPrefetchBlocks();
    // Accumulate a request's initial current-residency cached-token attribution, indexed by cache
    // level, into the current iteration window. Committed alongside the scalar iteration stats so
    // both views cover exactly the same requests.
    void commitCachedTokensByLevel(CountsByLevel const& counts);
    // Return the per-cache-level cached-token counts accumulated since the last drain and reset
    // them.
    CountsByLevel getAndResetIterationCachedTokensByLevel();
    // The cached tokens a KV connector loaded from an external store, which sit on no cache level
    // and so are not in the per-level counts. Committed alongside them.
    void commitConnectorCachedTokens(int64_t numTokens);
    int64_t getAndResetIterationConnectorCachedTokens();
    // Return {suspended, resumed} counts since the last drain and reset them.
    // Both counters track the same population, so the running
    // (suspended - resumed) total is the number of requests still parked in
    // the SUSPENDED state.
    std::pair<int64_t, int64_t> getAndResetIterationSuspendResumeStats();

    void markStatsDirty(std::optional<RequestIdType> kvCacheId);
    void clearStatsDirty(std::optional<RequestIdType> kvCacheId);
    std::unordered_set<RequestIdType> getDirtyStatsKvCacheIds() const;
    void markStatsExcluded(std::optional<RequestIdType> kvCacheId);
    void clearStatsExcluded(std::optional<RequestIdType> kvCacheId);
    bool isStatsExcluded(std::optional<RequestIdType> kvCacheId) const;

    // Mirrors Python's need_adjustment property and adjust() method.
    // All KvCaches must be suspended before calling adjust().
    bool needAdjustment() const;
    void adjust();

    // ---- Internals used by KvCache ----------------------------------------

    StorageManager& storage() noexcept
    {
        return *mStorage;
    }

    KVCacheManagerConfig const& config() const noexcept
    {
        return mConfig;
    }

    //! The shape of a life cycle's fetch path, or a default one for a life cycle with none.
    [[nodiscard]] SparseCacheShape const& _sparseCacheShape(LifeCycleId lifeCycle) const noexcept
    {
        static constexpr SparseCacheShape kNoFetchPath{};
        return lifeCycle < mSparseCacheShape.size() ? mSparseCacheShape[lifeCycle] : kNoFetchPath;
    }

    //! Cells a request's sparse cache holds per buffer of this life cycle, counting both
    //! populations: the history it caches and everything still in flight. Zero when the life
    //! cycle is not sparse or its fetch path is unconfigured.
    //!
    //! The reservation follows from this alone, so it is a property of the manager. What it
    //! leaves above the selection width is the headroom KvCache::maxInputLength() reports.
    [[nodiscard]] int sparseCacheMaxUnits(LifeCycleId lifeCycle) const noexcept
    {
        return _sparseCacheShape(lifeCycle).maxUnits;
    }

    //! Whether this life cycle fetches sparsely, which is what decides if a step carries any
    //! sparse work for it. A configured path has a positive selection width and at least one
    //! unit per page, so the reservation alone answers it.
    [[nodiscard]] bool hasSparseFetchPath(LifeCycleId lifeCycle) const noexcept
    {
        return sparseCacheMaxUnits(lifeCycle) > 0;
    }

    //! Whether any life cycle has a configured sparse fetch path. A property of the
    //! configuration, so it is the same for every cache this manager holds, and it decides
    //! whether max_input_length is required.
    [[nodiscard]] bool hasSparseCache() const noexcept
    {
        return std::ranges::any_of(mSparseCacheShape, [](SparseCacheShape const& shape) { return shape.maxUnits > 0; });
    }

    //! Selection units carried by one buffer of this life cycle, i.e. tokens_per_block
    //! divided by the unit's token count. One when the unit is a whole block.
    [[nodiscard]] int sparseCacheUnitsPerPage(LifeCycleId lifeCycle) const noexcept
    {
        return _sparseCacheShape(lifeCycle).unitsPerPage;
    }

    //! Sparse buffers this life cycle has, one per (layer, role). Cells and cold-page
    //! offsets are per buffer, so this is what the copy walks.
    //!
    //! Because a life cycle's buffers of one size coalesce into that pool's slot, this is
    //! also the cells a reserved slot yields, which is what makes the cell count and the
    //! cache count the same number.
    [[nodiscard]] int sparseCacheBufferCount(LifeCycleId lifeCycle) const noexcept
    {
        return _sparseCacheShape(lifeCycle).buffers;
    }

    //! The index table specs every sparse life cycle needs, one per (layer, pool) run.
    //!
    //! A Batch must still be given these -- it refuses a sparse buffer it was told nothing
    //! about, since a fetch with no converted indices would read the wrong pages rather than
    //! fail -- but their shape follows from the pool layout, so a caller need not re-derive it.
    [[nodiscard]] std::vector<IndexTableSpec> sparseIndexTableSpecs() const;

    LifeCycleRegistry const& lifeCycles() const noexcept
    {
        return mLifeCycles;
    }

    //! Life cycles the manager was built with; the registry never grows after construction.
    LifeCycleId numLifeCycles() const noexcept
    {
        return mLifeCycles.size();
    }

    BlockRadixTree& radixTree() noexcept
    {
        return *mRadixTree;
    }

    std::shared_ptr<EventSink> const& eventSink() const noexcept
    {
        return mEventSink;
    }

    // Called by KvCache constructor/destructor.
    void registerKvCache(KvCache* kvc);
    void unregisterKvCache(KvCache* kvc);

    // Moving-average updates from closed KvCaches.
    void updateAvgReusedLength(double v)
    {
        mAvgReusedLength.update(v);
    }

    void updateAvgSqrCapacity(double v)
    {
        mAvgSqrCapacity.update(v);
    }

    void updateAvgSqrHistoryLength(double v)
    {
        mAvgSqrHistoryLength.update(v);
    }

    void updateAvgBeamWidth(double v)
    {
        mAvgBeamWidth.update(v);
    }

    void updateAvgPromptLength(double v)
    {
        mAvgPromptLength.update(v);
    }

    void incrementNumSampledKvCaches()
    {
        ++mNumSampledKvCaches;
    }

    // Try to rebalance memory pool ratios based on usage statistics.
    void tryUpdateTargetRatios();

    // ---- Thread safety ----------------------------------------------------
    //
    // One ReentrantSharedMutex guards all mutable state reachable from this manager: the radix
    // tree, the storage manager and the living-KvCache set. Mutating APIs take it exclusively,
    // read-only queries take it shared.
    //
    // SCOPE: it protects state shared *between* KvCaches, not the fields of one KvCache. Each
    // KvCache is driven by its owning thread; calling into the same KvCache from two threads is
    // unsupported. So a KvCache method locks only if it reaches the manager, storage or the tree.
    //
    // See AGENTS.md ("Concurrency model") for the rationale, the granularity trade-off and the
    // rules for adding new APIs, and utils/reentrantSharedMutex.h for the lock's semantics.

    //! Exclusive lock for mutating APIs. Held by KvCache too, via its owning manager.
    [[nodiscard]] ReentrantSharedMutex::Guard lockExclusive() const
    {
        return mApiMutex.lockExclusive();
    }

    //! Shared lock for read-only queries.
    [[nodiscard]] ReentrantSharedMutex::Guard lockShared() const
    {
        return mApiMutex.lockShared();
    }

    // White-box introspection (incl. test-only auto-tuner state mutation) reaches
    // private members directly rather than widening the public API.
    friend class KvCacheIntrospection;
    //! Reaches _sparseLayerSpan to resolve a layer's buffers for its own step.
    friend class Batch;

private:
    //! Units a step selects per buffer of this life cycle. The cache must hold a whole
    //! selection on top of everything in flight, so this is what the in-flight headroom is
    //! measured against.
    [[nodiscard]] int _sparseCacheSelectionWidth(LifeCycleId lifeCycle) const noexcept
    {
        return _sparseCacheShape(lifeCycle).selection;
    }

    //! Layers this life cycle has that carry a sparse cache.
    //!
    //! The cache metadata is per layer rather than per buffer, because every sparse buffer
    //! of a layer selects the same units: one resident set and one recency order serve them
    //! all, and the entry index resolved from it addresses each buffer's own cell.
    [[nodiscard]] int _sparseCacheLayerCount(LifeCycleId lifeCycle) const noexcept
    {
        return _sparseCacheShape(lifeCycle).layers;
    }

    //! Selections this life cycle's layers resolve between them. The metadata is one set
    //! per selection, not per layer: layers handed the same units resolve to the same
    //! residency, so keeping a set each would be keeping copies of one answer.
    [[nodiscard]] int _sparseCacheGroupCount(LifeCycleId lifeCycle) const noexcept
    {
        return _sparseCacheShape(lifeCycle).groups;
    }

    //! Where this layer's selection sits in its life cycle's metadata. Layers sharing a
    //! selection_id share the slot, which is the whole of what sharing means here.
    [[nodiscard]] int _selectionSlotOf(LayerId layerId) const;

    //! [layers of this life cycle] each layer's metadata row, by its dense layer index --
    //! the arena's `layerSelection`, and what the write-back's per-layer scan reads.
    [[nodiscard]] std::vector<int32_t> const& _sparseLayerSelection(LifeCycleId lifeCycle) const;

    //! Layers resolving within one slot. One unless a selection_id is shared, which is what
    //! the fused fetch entry point refuses to serve.
    [[nodiscard]] int _selectionLayerCount(LifeCycleId lifeCycle, int slot) const;

    //! Sub-slots a request's sparse cache reserves per buffer of this life cycle, i.e. the
    //! cell count divided by the expansion. A property of configuration, not of any one
    //! request, which is what lets a batch size its fetch scratch before any member exists.
    [[nodiscard]] int _sparseCacheSubSlotsPerBuffer(LifeCycleId lifeCycle) const noexcept;

    //! Bound on how far capacity can lead history, in units: whatever the cache holds
    //! beyond a full selection. One definition, because the resolve's in-flight array and
    //! the write-back's flush list are bounded by the same headroom.
    [[nodiscard]] int _sparseCacheMaxInFlightUnits(LifeCycleId lifeCycle) const noexcept;

    //! This life cycle's sparse buffers, in the order the cache numbers them.
    [[nodiscard]] std::vector<SparseBufferPlan> _sparseBufferPlan(LifeCycleId lifeCycle) const;

    //! Just their ids, in the same order.
    [[nodiscard]] std::vector<BufferId> _sparseBufferIds(LifeCycleId lifeCycle) const;

    //! Multiprocessors of the device this manager runs on, for sizing a copy's grid.
    [[nodiscard]] int32_t _smCount() const noexcept
    {
        return mSmCount;
    }

    //! Behind a driver call, and the sparse copy wants it per launch, so it is read once at
    //! construction.
    int32_t mSmCount = 0;

    //! Storage attributes of one buffer: its pool, its offset within that pool's slot, its
    //! expanded size and its expansion.
    [[nodiscard]] BufferAttr const& _bufferAttr(LayerId layerId, DataRole const& role) const
    {
        return mStorage->getBufferAttr(layerId, role);
    }

    // First member, so the registration covers the whole lifetime: it is taken before any state
    // this manager could leave behind exists, and dropped after ~KvCacheManager has run.
    PoisonHold mPoisonHold;

    //! Guards all mutable state reachable from this manager. See the scope note above.
    mutable ReentrantSharedMutex mApiMutex;

    //! One layer's run within its life cycle's sparse buffer list, as the sparse entry
    //! points name it: they take a layer's buffers and address them as a range.
    struct SparseLayerSpan
    {
        LifeCycleId lifeCycle{0};
        int32_t bufferBase = 0;
        int32_t numBuffers = 0;
        int32_t layerInGroup = 0;  //!< The dense index of the layer within its life cycle.
        int32_t selectionSlot = 0; //!< Its metadata row, which layers sharing a selection share.
    };

    [[nodiscard]] SparseLayerSpan _sparseLayerSpan(std::vector<BufferId> const& buffers, char const* api) const;

    //! The most blocks of one request that can take scratch slots in this life cycle,
    //! given the widest page table a batch will give a row and, where the caller knows it,
    //! the most tokens one step may ingest.
    //!
    //! Blocks still in the window or held as sinks are never stale and so never scratch,
    //! which is what makes this smaller than the page table.
    //!
    //! `maxTotalInputTokens` usually bounds it far harder. Scratch is the blocks a step
    //! *adds* that are *already* stale at capacity, so a step ingesting T tokens can make
    //! at most `(T - window - rewind + 1) / tpb` of them -- the window and rewind come off
    //! in tokens, before the division. Under chunked prefill that is the chunk rather than
    //! the context, and it is zero whenever a chunk cannot outrun the window.
    //!
    //! Pass it only where the caller truly bounds a step; the Batch's decode-shaped
    //! default would under-count a batch that then runs prefill.
    [[nodiscard]] int _maxScratchBlocks(
        LifeCycleId lcId, int maxBlocksPerSeq, std::optional<int> maxTotalInputTokens = std::nullopt) const;

    //! Slots a life cycle's worst request needs: its scratch blocks at the widest
    //! sub-pages-per-slot ratio any of its layers reaches. Zero where it can hold none.
    [[nodiscard]] int _scratchSlotsFor(
        LifeCycleId lcId, int maxBlocksPerSeq, std::optional<int> maxTotalInputTokens = std::nullopt) const;

    // Throw unless every KvCache has been closed. `api` names the caller so the message
    // points at the mistake rather than at whatever breaks later.
    void _checkNoLivingKvCaches(char const* api) const;

    void _adjustLevel(CacheLevel level, size_t quota);
    bool _needAdjustment(CacheLevel level) const;
    TypedVec<PoolGroupIndex, float> const& _getTargetRatioList(CacheLevel level) const;
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> _gatherLastLevelPersistentPages() const;

    PeakBlockStatsByCacheLevel _currentBlockStatsByCacheLevel() const;
    void _resetIterationPeakNumBlocks(std::optional<CacheLevel> cacheLevel = std::nullopt);

    // Fill the sparse cache shape tables from the configured buffers. Called once at construction.
    void _computeSparseCacheShape();
    void _updateIterationPeakNumBlocks();

    // Current per-pool-group utilization ratios for the hot and cold representations.
    TypedVec<PoolGroupIndex, float> _currentHotRatio() const;
    TypedVec<PoolGroupIndex, float> _currentColdRatios() const;

    KVCacheManagerConfig mConfig;
    LifeCycleRegistry mLifeCycles;

    // All indexed by life cycle; see the sparseCache* accessors.
    TypedVec<LifeCycleId, SparseCacheShape> mSparseCacheShape;

    //! Each sparse layer's slot in its life cycle's selection metadata, dense from zero.
    std::unordered_map<LayerId, int> mSelectionSlot;

    //! The same, by life cycle and dense layer row, which is how a kernel indexes it.
    TypedVec<LifeCycleId, std::vector<int32_t>> mSparseLayerSelection;
    std::shared_ptr<EventSink> mEventSink;
    std::shared_ptr<StorageManager> mStorage;
    std::shared_ptr<BlockRadixTree> mRadixTree;

    // Weak references to all living KvCaches.
    std::set<KvCache*> mLivingKvCaches;

    // Moving averages used for ratio rebalancing.
    MovingAverage mAvgReusedLength;
    MovingAverage mAvgSqrCapacity;
    MovingAverage mAvgSqrHistoryLength;
    // Beam search replicates only the blocks past the prompt tail, so the tuner
    // needs both the typical beam width and where that tail sits.
    MovingAverage mAvgBeamWidth;
    MovingAverage mAvgPromptLength;

    TypedVec<PoolGroupIndex, float> mTargetRatioListHot;
    TypedVec<PoolGroupIndex, float> mTargetRatioListCold;

    int mNumCreatedKvCaches{0};
    int mNumSampledKvCaches{0};
    double mLastAdjustmentTime{0.0};
    int mLastUpdateNumSampledKvCaches{0};

    KVCacheStatsDelta mCommittedStats;
    IterationStatsByLifeCycle mIterationStatsByLifeCycle;
    SsmSnapshotIterationStatsByLifeCycle mSsmSnapshotIterationStatsByLifeCycle;
    ReusedBlocksByLevelByLifeCycle mIterReusedBlocksByLevel;
    ConnectorReusedBlocksByLifeCycle mIterConnectorReusedBlocks;
    PeakBlockStatsByCacheLevel mIterationPeakNumBlocksByCacheLevel;
    std::unordered_set<RequestIdType> mDirtyStatsKvCacheIds;
    std::unordered_set<RequestIdType> mStatsExcludedKvCacheIds;
    int64_t mIterSuspendedRequests{0};
    int64_t mIterResumedRequests{0};
    int64_t mIterDiskPrefetchBlocks{0};
    CountsByLevel mIterCachedTokensByLevel;
    int64_t mIterConnectorCachedTokens{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
