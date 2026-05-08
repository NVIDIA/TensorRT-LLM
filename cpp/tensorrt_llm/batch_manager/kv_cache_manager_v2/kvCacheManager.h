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
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/movingAverage.h"
#include "kv_cache_manager_v2/storageManager.h"

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// MemoryPoolDesc / MemoryPoolGroupDesc — describe GPU memory pool layout.
// ---------------------------------------------------------------------------
struct MemoryPoolDesc
{
    MemAddress base;
    int pageSize;
};

struct MemoryPoolGroupDesc
{
    int numPages;
    std::vector<MemoryPoolDesc> pools; // indexed by PoolIndex
};

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
    int size;                            // byte span of this aggregated buffer group
    int stride;                          // slot size (bytes per slot in the pool group)
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

    // Convert one base index to `expansion` kernel indices.
    std::vector<int> operator()(int baseIndex) const;
};

// ---------------------------------------------------------------------------
// KvCacheManager — top-level KV cache manager.
// Mirrors Python's KVCacheManager.
// ---------------------------------------------------------------------------
class KvCacheManager : public std::enable_shared_from_this<KvCacheManager>
{
public:
    explicit KvCacheManager(KVCacheManagerConfig const& config);
    ~KvCacheManager();

    KvCacheManager(KvCacheManager const&) = delete;
    KvCacheManager& operator=(KvCacheManager const&) = delete;

    // ---- Lifecycle --------------------------------------------------------

    void shutdown();

    // Clear all reusable (committed) blocks from the radix tree.
    void clearReusableBlocks();

    // ---- KvCache creation -------------------------------------------------

    // Create a new KvCache. Returned cache is SUSPENDED; call activate() with a stream.
    // input_tokens: optional sequence to match against existing cached blocks.
    // priorityCb:   optional priority override per block.
    std::shared_ptr<KvCache> createKvCache(std::optional<int64_t> loraTaskId = std::nullopt,
        std::vector<TokenIdExt> const& inputTokens = {}, std::optional<int64_t> id = std::nullopt,
        KvCache::PriorityCb priorityCb = {});

    // ---- Memory pool queries -----------------------------------------------

    MemAddress getMemPoolBaseAddress(LayerId layerId, DataRole role) const;
    int getPageStride(LayerId layerId, DataRole role) const;
    int getPageIndexUpperBound(LayerId layerId, DataRole role) const;

    // Scale factor: base_page_index * scale → kernel page index.
    int getPageIndexScale(LayerId layerId, DataRole role) const;

    // Composite converter (scale + expansion).
    PageIndexConverter getPageIndexConverter(LayerId layerId, DataRole role) const;

    // Group a set of BufferIds into contiguous AggregatedPageDesc descriptors.
    // Mirrors Python's KVCacheManager.get_aggregated_pages().
    std::vector<AggregatedPageDesc> getAggregatedPages(std::vector<BufferId> const& buffers) const;

    // ---- Query / info ------------------------------------------------------

    int tokensPerBlock() const noexcept;
    bool enablePartialMatch() const noexcept;

    int ssmReuseInterval() const noexcept
    {
        return mConfig.ssmReuseInterval;
    }

    bool allowSeqRebasing() const noexcept
    {
        return true;
    }

    int numLayers() const noexcept;

    std::vector<LayerId> layerIds() const;
    LayerGroupId getLayerGroupId(LayerId layerId) const;

    // Layer grouping: layers with the same lifecycle share pool allocation.
    std::vector<std::vector<LayerId>> layerGrouping() const;

    // Sorted by CacheLevel from warm to cold. Mirrors Python's cache_tier_list property.
    std::vector<CacheTier> cacheTierList() const;

    // Iterator over all buffer identifiers. Mirrors Python's all_buffer_ids property.
    std::vector<BufferId> allBufferIds() const;

    // Get the max possible sequence length limited by GPU memory pools.
    // Mirrors Python's clamp_max_seq_len_for_mem().
    int clampMaxSeqLenForMem(int batchSize, int tokenNumUpperBound) const;

    // ---- Resize -----------------------------------------------------------

    bool resize(CacheLevel level, size_t quota, bool bestEfforts = false);
    size_t getQuota(CacheLevel level) const;

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

    // Try to rebalance memory pool ratios based on usage statistics.
    void tryUpdateTargetRatios();

private:
    void _adjustLevel(CacheLevel level, size_t quota);
    bool _needAdjustment(CacheLevel level) const;
    std::vector<float> const& _getTargetRatioList(CacheLevel level) const;
    std::vector<std::vector<std::shared_ptr<Page>>> _gatherPersistentPages() const;

    // Current per-pool-group GPU utilization ratios.
    std::vector<float> _currentGpuRatio() const;
    std::vector<float> _currentOtherRatios() const;

    KVCacheManagerConfig mConfig;
    LifeCycleRegistry mLifeCycles;
    std::shared_ptr<BlockRadixTree> mRadixTree;
    std::shared_ptr<StorageManager> mStorage;

    // Weak references to all living KvCaches.
    std::set<KvCache*> mLivingKvCaches;

    // Moving averages used for ratio rebalancing.
    MovingAverage mAvgReusedLength;
    MovingAverage mAvgSqrCapacity;
    MovingAverage mAvgSqrHistoryLength;

    std::vector<float> mTargetRatioListGpu;
    std::vector<float> mTargetRatioListOther;

    int mNumCreatedKvCaches{0};
    int mNumClosedKvCaches{0};
    double mLastAdjustmentTime{0.0};
    int mLastUpdateNumClosedRequests{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
