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
#include "kv_cache_manager_v2/stats.h"
#include "kv_cache_manager_v2/storageManager.h"

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
// ScratchDesc — scratch metadata for one layer group of one sequence.
// Scratch blocks store ephemeral KV data using shared coalesced slots.
// Mirrors _kv_cache_manager.py::ScratchDesc.
// ---------------------------------------------------------------------------
struct ScratchDesc
{
    HalfOpenRange<BlockOrdinal> range; // block ordinal range [beg, end)
    std::vector<int> slotIds;          // scratch slot IDs, length = ceil(numScratchBlocks / scale)

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(range);
    }
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
    // scratch: optional scratch descriptor from KvCache::getScratchDesc().
    // Mirrors _kv_cache_manager.py::PageIndexConverter.__call__.
    std::vector<int> operator()(std::vector<int> const& baseIndices,
        std::optional<PageIndexMode> indexMode = std::nullopt, ScratchDesc const* scratch = nullptr) const;

    // Backward-compatible single-index overload.
    std::vector<int> operator()(int baseIndex) const;
};

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

    // Clear all reusable (committed) blocks from the radix tree.
    void clearReusableBlocks();

    // ---- KvCache creation -------------------------------------------------

    // Create a new KvCache. Returned cache is SUSPENDED; call activate() with a stream.
    // input_tokens:         optional sequence to match against existing cached blocks.
    // priorityCb:           optional priority override per block.
    // expectedPromptLength: token count marking the prefill->generation boundary; once
    //                       historyLength reaches it, later capacity growth is recorded as
    //                       generation-phase allocation stats (defaults to inputTokens.size()).
    //                       Stats-only: no effect on allocation, reuse, or correctness.
    // textOnly:             per-sequence override of the text-only (digest-free) guarantee;
    //                       nullopt inherits the manager config default.
    // enableRequestStats:   collect request-local allocation and reuse statistics even when
    //                       manager-level statistics are disabled.
    // inputTokens is a non-owning view; the caller must keep the underlying buffer alive for the
    // duration of the call (matching reads it but never stores it).
    std::shared_ptr<KvCache> createKvCache(ReuseScope reuseScope = {}, TokenSpan inputTokens = {},
        std::optional<RequestIdType> id = std::nullopt, KvCache::PriorityCb priorityCb = {},
        std::optional<int> expectedPromptLength = std::nullopt, std::optional<bool> textOnly = std::nullopt,
        bool enableRequestStats = false);

    // knownNoDigest: from external text_only knowledge, never a scan (see Hasher::update).
    // Defaults false (safe: the scanning path is taken).
    BlockRadixTree::ReuseMatch matchReuse(
        ReuseScope const& reuseScope, TokenSpan inputTokens, bool knownNoDigest = false) const;
    int probeReuse(ReuseScope reuseScope = {}, TokenSpan inputTokens = {}, bool knownNoDigest = false) const;

    // ---- Memory pool queries -----------------------------------------------

    // Base address of the memory pool. When indexMode is PER_LAYER, returns pool group base
    // (without per-layer offset). When SHARED, returns per-layer base (with offset baked in).
    MemAddress getMemPoolBaseAddress(
        LayerId layerId, DataRole role, std::optional<PageIndexMode> indexMode = std::nullopt) const;

    int getPageStride(LayerId layerId, DataRole role) const;
    size_t getPageIndexUpperBound(LayerId layerId, DataRole role) const;

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

    // Whether managed KV caches support the given page index mode.
    // Returns true/false for a definitive answer, nullopt for per-instance check.
    std::optional<bool> supportsIndexMode(PageIndexMode mode) const;

    bool allowSeqRebasing() const noexcept
    {
        return true;
    }

    int numLayers() const noexcept;

    std::vector<LayerId> layerIds() const;
    LayerGroupId getLayerGroupId(LayerId layerId) const;

    // Layer grouping: layers with the same lifecycle share pool allocation.
    // NOTE: the iteration order of the layer lists (and of the groups) is NOT
    // part of the API contract and may differ across backends/runs. Do not rely
    // on it to infer buffer/pool memory order — query poolGroupDescs()
    // (PoolGroupDesc::pools[i].baseAddress + coalescedBuffers) for that.
    TypedVec<LayerGroupId, std::vector<LayerId>> layerGrouping() const;

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

    void commitStats(KVCacheStatsDelta const& stats, IterationStatsByLifeCycle const& iterationStatsByLifeCycle = {});
    KVCacheStatsDelta getCommittedStats() const;
    IterationStatsByLifeCycle getAndResetIterationStats();
    PeakBlockStatsByPoolGroup getAndResetIterationPeakBlockStats(CacheLevel cacheLevel);

    void commitSsmSnapshotIterationStats(SsmSnapshotIterationStatsByLifeCycle const& statsByLifeCycle);
    SsmSnapshotIterationStatsByLifeCycle getAndResetSsmSnapshotIterationStats();

    // Count one ACTIVE->SUSPENDED transition for the current iteration window.
    void recordRequestSuspended();
    // Count one preemption recovery for the current iteration window. Only a
    // previously-ACTIVE cache that was suspended and then successfully resumed
    // counts; a freshly-created cache is activated by its first resume(), but
    // that is an admission, not a recovery, and is not counted.
    void recordRequestResumed();
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

    LifeCycleRegistry const& lifeCycles() const noexcept
    {
        return mLifeCycles;
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

    void incrementNumSampledKvCaches()
    {
        ++mNumSampledKvCaches;
    }

    // Try to rebalance memory pool ratios based on usage statistics.
    void tryUpdateTargetRatios();

    // White-box introspection (incl. test-only auto-tuner state mutation) reaches
    // private members directly rather than widening the public API.
    friend class KvCacheIntrospection;

private:
    // Throw unless every KvCache has been closed. `api` names the caller so the message
    // points at the mistake rather than at whatever breaks later.
    void _checkNoLivingKvCaches(char const* api) const;

    void _adjustLevel(CacheLevel level, size_t quota);
    bool _needAdjustment(CacheLevel level) const;
    TypedVec<PoolGroupIndex, float> const& _getTargetRatioList(CacheLevel level) const;
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> _gatherLastLevelPersistentPages() const;

    PeakBlockStatsByCacheLevel _currentBlockStatsByCacheLevel() const;
    void _resetIterationPeakNumBlocks(std::optional<CacheLevel> cacheLevel = std::nullopt);
    void _updateIterationPeakNumBlocks();

    // Current per-pool-group utilization ratios for the hot and cold representations.
    TypedVec<PoolGroupIndex, float> _currentHotRatio() const;
    TypedVec<PoolGroupIndex, float> _currentColdRatios() const;

    KVCacheManagerConfig mConfig;
    LifeCycleRegistry mLifeCycles;
    std::shared_ptr<EventSink> mEventSink;
    std::shared_ptr<StorageManager> mStorage;
    std::shared_ptr<BlockRadixTree> mRadixTree;

    // Weak references to all living KvCaches.
    std::set<KvCache*> mLivingKvCaches;

    // Moving averages used for ratio rebalancing.
    MovingAverage mAvgReusedLength;
    MovingAverage mAvgSqrCapacity;
    MovingAverage mAvgSqrHistoryLength;

    TypedVec<PoolGroupIndex, float> mTargetRatioListHot;
    TypedVec<PoolGroupIndex, float> mTargetRatioListCold;

    int mNumCreatedKvCaches{0};
    int mNumSampledKvCaches{0};
    double mLastAdjustmentTime{0.0};
    int mLastUpdateNumSampledKvCaches{0};

    KVCacheStatsDelta mCommittedStats;
    IterationStatsByLifeCycle mIterationStatsByLifeCycle;
    SsmSnapshotIterationStatsByLifeCycle mSsmSnapshotIterationStatsByLifeCycle;
    PeakBlockStatsByCacheLevel mIterationPeakNumBlocksByCacheLevel;
    std::unordered_set<RequestIdType> mDirtyStatsKvCacheIds;
    std::unordered_set<RequestIdType> mStatsExcludedKvCacheIds;
    int64_t mIterSuspendedRequests{0};
    int64_t mIterResumedRequests{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
