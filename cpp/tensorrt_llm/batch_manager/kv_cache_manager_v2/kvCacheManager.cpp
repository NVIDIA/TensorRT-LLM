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

#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/batch.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/storage/core.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

static double nowSeconds()
{
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
                                   .count())
        / 1e6;
}

// ---------------------------------------------------------------------------
// PageIndexConverter
// ---------------------------------------------------------------------------

std::vector<int> PageIndexConverter::operator()(std::vector<int> const& baseIndices,
    std::optional<PageIndexMode> indexMode, HalfOpenRange<BlockOrdinal> scratchRange,
    Span<int const> scratchSlots) const
{
    if (!indexMode.has_value())
    {
        if (!scratchRange.empty())
        {
            throw std::invalid_argument("index_mode must be provided when scratch is active");
        }
        indexMode = PageIndexMode::SHARED;
    }

    int appliedLayerOffset = (*indexMode == PageIndexMode::PER_LAYER) ? layerOffset : 0;
    int scratchPages = scratchPagesPerBlock;

    std::vector<int> result;
    result.reserve(baseIndices.size() * static_cast<size_t>(expansion));

    for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{static_cast<int>(baseIndices.size())}; ++ordinal)
    {
        int index;
        if (scratchRange.contains(ordinal))
        {
            // Scratch block: its slot comes from the request's own scratch slots, not from
            // base_indices.
            int blockPos = ordinal - scratchRange.beg;
            int totalOffset = blockPos * scratchPages;
            bool const crossesSlots = scale % scratchPages != 0;
            if (crossesSlots)
            {
                // Unequal layer shares can put a block across two slots. Pack
                // the layer's consecutive roles before resolving the slot ID,
                // so a trailing role cannot wrap over an earlier block's data.
                totalOffset += appliedLayerOffset % scratchPages;
            }
            int slotIdx = totalOffset / scale;
            TLLM_CHECK_DEBUG_WITH_INFO(slotIdx < scratchSlots.size(), "Scratch slot %d past the %d the range describes",
                slotIdx, scratchSlots.size());
            int slotId = scratchSlots[slotIdx];
            int offset = totalOffset % scale;
            index = slotId * scale + (crossesSlots ? offset : (offset + appliedLayerOffset) % scale);
        }
        else if (baseIndices[toSizeT(ordinal)] == kBadPageIndex.value())
        {
            index = kBadPageIndex.value();
        }
        else
        {
            index = baseIndices[toSizeT(ordinal)] * scale + appliedLayerOffset;
        }
        for (int i = 0; i < expansion; ++i)
        {
            result.push_back(index != kBadPageIndex.value() ? index * expansion + i : kBadPageIndex.value());
        }
    }
    return result;
}

std::vector<int> PageIndexConverter::operator()(int baseIndex) const
{
    return operator()(std::vector<int>{baseIndex});
}

// ---------------------------------------------------------------------------
// KvCacheManager
// ---------------------------------------------------------------------------

uint32_t KvCacheManager::numLiveManagers() noexcept
{
    return PoisonHold::count();
}

KvCacheManager::KvCacheManager(KVCacheManagerConfig const& config, std::shared_ptr<EventSink> eventSink,
    std::unique_ptr<IKvCacheColdPageCodec> coldPageCodec)
    : mConfig(config)
    , mLifeCycles(config)
    , mEventSink(std::move(eventSink))
    , mAvgReusedLength(config.poolRebalance.movingAverageDecay)
    , mAvgSqrCapacity(config.poolRebalance.movingAverageDecay)
    , mAvgSqrHistoryLength(config.poolRebalance.movingAverageDecay)
    , mAvgBeamWidth(config.poolRebalance.movingAverageDecay)
    , mAvgPromptLength(config.poolRebalance.movingAverageDecay)
{
    mConfig.validate();

    mRadixTree = std::make_shared<BlockRadixTree>(mLifeCycles, mConfig.tokensPerBlock, mEventSink);

    StorageConfig storageConfig = createStorageConfig(mConfig);
    mStorage = std::make_shared<StorageManager>(mLifeCycles, storageConfig, mConfig.tokensPerBlock,
        std::move(coldPageCodec), mConfig.swaScratchReuse, mConfig.typicalStep, mConfig.constraints,
        mConfig.initialPoolRatio, mEventSink, mConfig.maxUtilForResume, mConfig.coldTierOffloadMinPriority);

    _computeSparseCacheShape();

    // The storage above has a device context up, which this needs.
    CUdevice device{};
    cuCheck(cuCtxGetDevice(&device));
    mSmCount = sparseCopyMultiprocessors(device);

    mTargetRatioListHot = _currentHotRatio();
    mTargetRatioListCold = _currentColdRatios();
    _resetIterationPeakNumBlocks();

    mLastAdjustmentTime = nowSeconds();
}

KvCacheManager::~KvCacheManager()
{
    KVCM2_POISON_ON_EXCEPT([this]() { shutdown(); });
}

void KvCacheManager::_computeSparseCacheShape()
{
    LifeCycleId const numLc = mStorage->numLifeCycles();
    mSparseCacheShape = TypedVec<LifeCycleId, SparseCacheShape>(numLc, SparseCacheShape{});
    mSelectionSlot.clear();
    mSparseLayerSelection = TypedVec<LifeCycleId, std::vector<int32_t>>(numLc);

    // Selection ids are the caller's, so they are dense only by luck; these give each one
    // a slot in its life cycle's metadata, and catch an id used across two of them.
    std::unordered_map<int, LifeCycleId> selectionLifeCycle;
    std::map<LifeCycleId, std::map<int, int>> slotsOf;

    for (auto const& layer : mConfig.layers)
    {
        auto const* attn = std::get_if<AttentionLayerConfig>(&layer);
        if (attn == nullptr)
        {
            continue;
        }
        if (!attn->sparseCache.has_value())
        {
            continue;
        }
        LifeCycleId const lcId = getLayerGroupId(attn->layerId);
        auto const& cache = *attn->sparseCache;
        int const unitsPerPage = cache.unitsPerPage(mConfig.tokensPerBlock);

        // Layers of one life cycle must currently declare the same sparse_cache. The arena
        // is per life cycle and its metadata is one stride: a selection's rows are found at
        // `residentOrdinal + slot * entriesPerSelection`, so a single entry count serves
        // them all, and cellsPerBuffer follows from maxCachedUnits and unitsPerPage alike.
        // See SPARSE_ATTENTION_DESIGN.md.
        auto& shape = mSparseCacheShape[lcId];
        if (shape.layers > 0)
        {
            TLLM_CHECK_WITH_INFO(shape.maxUnits == cache.maxCachedUnits && shape.selection == cache.numSelectedBlocks
                    && shape.unitsPerPage == unitsPerPage,
                "Layers sharing a life cycle must declare the same sparse_cache; they share one arena");
        }
        shape.maxUnits = cache.maxCachedUnits;
        shape.selection = cache.numSelectedBlocks;
        shape.unitsPerPage = unitsPerPage;

        bool layerCounted = false;
        for (auto const& buf : attn->buffers)
        {
            // A zero-size buffer has no layout and never reaches storage, so the plan below
            // has no entry for it. Counting it here would size this metadata for a buffer
            // nothing writes.
            if (!buf.isSparse || buf.size == 0)
            {
                continue;
            }
            ++shape.buffers;
            if (!layerCounted)
            {
                ++shape.layers;
                layerCounted = true;
            }
        }
        if (!layerCounted)
        {
            continue;
        }

        // A selection is a life cycle's, because the metadata it indexes is. Layers that
        // name one from either side of a life cycle boundary are naming two things.
        int const selectionId = cache.effectiveSelectionId(attn->layerId);
        auto const [owner, firstHere] = selectionLifeCycle.try_emplace(selectionId, lcId);
        TLLM_CHECK_WITH_INFO(firstHere || owner->second == lcId,
            "Layers sharing a selection_id must share a life cycle; selection %d spans two", selectionId);

        auto& slots = slotsOf[lcId];
        auto const [slot, fresh] = slots.try_emplace(selectionId, static_cast<int>(slots.size()));
        mSelectionSlot[attn->layerId] = slot->second;
        // Indexed by the layer's dense row, which is what the arena and the write-back's
        // grid address it by; _sparseBufferPlan numbers it from this same walk.
        mSparseLayerSelection[lcId].push_back(static_cast<int32_t>(slot->second));
        if (fresh)
        {
            ++shape.groups;
        }
    }
}

int KvCacheManager::_selectionSlotOf(LayerId layerId) const
{
    auto const it = mSelectionSlot.find(layerId);
    TLLM_CHECK_WITH_INFO(it != mSelectionSlot.end(), "This layer carries no sparse selection");
    return it->second;
}

std::vector<int32_t> const& KvCacheManager::_sparseLayerSelection(LifeCycleId lifeCycle) const
{
    return mSparseLayerSelection[lifeCycle];
}

int KvCacheManager::_selectionLayerCount(LifeCycleId lifeCycle, int slot) const
{
    auto const& selections = mSparseLayerSelection[lifeCycle];
    return static_cast<int>(std::count(selections.begin(), selections.end(), slot));
}

int KvCacheManager::_sparseCacheSubSlotsPerBuffer(LifeCycleId lifeCycle) const noexcept
{
    int const maxUnits = sparseCacheMaxUnits(lifeCycle);
    if (maxUnits <= 0)
    {
        return 0;
    }
    return divUp(maxUnits, sparseCacheUnitsPerPage(lifeCycle));
}

int KvCacheManager::_sparseCacheMaxInFlightUnits(LifeCycleId lifeCycle) const noexcept
{
    int const cells = _sparseCacheSubSlotsPerBuffer(lifeCycle) * sparseCacheUnitsPerPage(lifeCycle);
    if (cells <= 0)
    {
        return 0;
    }
    // A step advances capacity one token past history, and one token straddles at most one
    // unit boundary, so at most one unit is ever in flight. That is the whole of the
    // headroom: everything else the reservation holds is available to the selection.
    return 1;
}

std::vector<SparseBufferPlan> KvCacheManager::_sparseBufferPlan(LifeCycleId lifeCycle) const
{
    // Layers outermost, as _computeSparseCacheShape walks them.
    std::vector<SparseBufferPlan> plan;
    std::optional<LayerId> previousLayer;
    int32_t layerRow = -1;
    for (auto const& layer : mConfig.layers)
    {
        auto const* attn = std::get_if<AttentionLayerConfig>(&layer);
        if (attn == nullptr || !attn->sparseCache.has_value() || getLayerGroupId(attn->layerId) != lifeCycle)
        {
            continue;
        }
        auto const firstOfLayer = plan.size();
        for (auto const& buf : attn->buffers)
        {
            // A sparse buffer in a layer with no sparse_cache has storage and locking but
            // no fetch path, so it contributes no cells and no layout.
            // Zero-size as in _computeSparseCacheShape: no layout to derive, and
            // sparseLayout() rejects one rather than inventing it.
            if (!buf.isSparse || buf.size == 0 || !attn->sparseCache.has_value())
            {
                continue;
            }
            if (!previousLayer.has_value() || attn->layerId != *previousLayer)
            {
                previousLayer = attn->layerId;
                ++layerRow;
            }
            auto const attr = _bufferAttr(attn->layerId, buf.role);
            plan.push_back(SparseBufferPlan{BufferId{attn->layerId, buf.role}, attr.poolIndex, attr.size,
                static_cast<uint32_t>(attr.offset),
                buf.sparseLayout(attn->sparseCache->unitsPerPage(mConfig.tokensPerBlock)), layerRow, 0});
        }
        // Group a layer's buffers by pool so a (layer, pool) run is consecutive too; the copy
        // addresses one pool per launch. Stable, so declaration order survives within a pool.
        std::stable_sort(plan.begin() + static_cast<ptrdiff_t>(firstOfLayer), plan.end(),
            [](SparseBufferPlan const& a, SparseBufferPlan const& b) { return a.poolIndex < b.poolIndex; });
    }

    std::map<PoolIndex, int32_t> nextSlot;
    for (auto& entry : plan)
    {
        entry.slotInPool = nextSlot[entry.poolIndex]++;
    }
    return plan;
}

std::vector<BufferId> KvCacheManager::_sparseBufferIds(LifeCycleId lifeCycle) const
{
    std::vector<BufferId> ids;
    for (auto const& entry : _sparseBufferPlan(lifeCycle))
    {
        ids.push_back(entry.id);
    }
    return ids;
}

std::vector<IndexTableSpec> KvCacheManager::sparseIndexTableSpecs() const
{
    std::vector<IndexTableSpec> specs;
    for (LifeCycleId lcId{0}; lcId < mStorage->numLifeCycles(); ++lcId)
    {
        auto const plan = _sparseBufferPlan(lcId);
        for (size_t begin = 0; begin < plan.size();)
        {
            size_t end = begin;
            while (end < plan.size() && plan[end].poolIndex == plan[begin].poolIndex
                && plan[end].layerRow == plan[begin].layerRow)
            {
                ++end;
            }
            IndexTableSpec spec;
            for (size_t i = begin; i < end; ++i)
            {
                spec.buffers.push_back(plan[i].id);
            }
            specs.push_back(std::move(spec));
            begin = end;
        }
    }
    return specs;
}

KvCacheManager::SparseLayerSpan KvCacheManager::_sparseLayerSpan(
    std::vector<BufferId> const& buffers, char const* api) const
{
    TLLM_CHECK_WITH_INFO(!buffers.empty(), std::string(api) + ": no buffers");

    LayerId const layerId = buffers.front().layerId;
    for (auto const& id : buffers)
    {
        TLLM_CHECK_WITH_INFO(id.layerId == layerId, std::string(api) + ": every buffer must belong to one layer");
    }

    SparseLayerSpan span;
    span.lifeCycle = getLayerGroupId(layerId);
    auto const all = _sparseBufferIds(span.lifeCycle);
    // Buffers are grouped by layer, so this layer's are a contiguous run.
    auto const first
        = std::find_if(all.begin(), all.end(), [layerId](BufferId const& id) { return id.layerId == layerId; });
    auto const last = std::find_if(first, all.end(), [layerId](BufferId const& id) { return id.layerId != layerId; });
    TLLM_CHECK_WITH_INFO(std::equal(first, last, buffers.begin(), buffers.end()),
        std::string(api) + ": buffers must be exactly this layer's sparse buffers, in the cache's order");

    span.bufferBase = static_cast<int32_t>(std::distance(all.begin(), first));
    span.numBuffers = static_cast<int32_t>(std::distance(first, last));
    // The metadata row is the layer's index among those carrying a cache, which is the
    // number of distinct layers before it in the same walk.
    for (auto it = all.begin(); it != first; ++it)
    {
        if (it == all.begin() || it->layerId != std::prev(it)->layerId)
        {
            ++span.layerInGroup;
        }
    }
    span.selectionSlot = _selectionSlotOf(layerId);
    return span;
}

void KvCacheManager::_checkNoLivingKvCaches(char const* api) const
{
    TLLM_CHECK_WITH_INFO(mLivingKvCaches.empty(),
        "%s with %zu KV cache(s) still open; close them (or drain the engine) first", api, mLivingKvCaches.size());
}

void KvCacheManager::shutdown()
{
    // Disposal, not work: a caller that detects corruption still has to be able to tear the
    // manager down, so this succeeds while poisoned instead of rejecting. The cleanup below is
    // skipped for the same reason every poisoned destructor skips its own -- the structures it
    // would walk are known to be inconsistent -- so the manager's memory is leaked deliberately.
    //
    // The latch is read under the lock because it is normally set by a thread holding it: a read
    // taken before blocking would only say the cache was intact before this call started waiting.
    auto const apiLock = lockExclusive();
    if (Poison::poisoned())
    {
        return;
    }
    _checkNoLivingKvCaches("shutdown()");
    clearReusableBlocks();
    TLLM_CHECK_DEBUG(mStorage);

    // Post-condition: after clearing all reusable blocks and with no active KvCaches,
    // no evictable pages should remain.
    for (auto const& lvl : mStorage->mLevels)
    {
        for (PoolGroupIndex pgIdx{0}; pgIdx < lvl.numPoolGroups(); ++pgIdx)
        {
            TLLM_CHECK_DEBUG(lvl.controller.numEvictablePages(pgIdx) == 0);
        }
    }

    mStorage->destroy();
}

void KvCacheManager::clearReusableBlocks()
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    _checkNoLivingKvCaches("clear_reusable_blocks()");
    TLLM_CHECK_DEBUG(mRadixTree);
    mRadixTree->clear();
}

std::shared_ptr<KvCache> KvCacheManager::createKvCache(ReuseScope reuseScope, TokenSpan inputTokens,
    std::optional<RequestIdType> id, KvCache::PriorityCb priorityCb, std::optional<int> expectedPromptLength,
    std::optional<bool> textOnly, bool enableRequestStats,
    std::optional<executor::KvCacheRetentionConfig> retentionConfig)
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    if (!priorityCb)
    {
        priorityCb = [](BlockOrdinal, LifeCycleId) { return kPriorityDefault; };
    }

    if (!expectedPromptLength.has_value() && inputTokens.size() != 0)
    {
        expectedPromptLength = static_cast<int>(inputTokens.size());
    }

    // Compute the reuse match once here (shared conceptually with probeReuse) and
    // hand it to the KvCache, rather than having the cache re-walk the radix tree.
    // Mirrors Python KVCacheManager.allocate() passing a ReuseMatch into _KVCache.
    std::optional<BlockRadixTree::ReuseMatch> reuseMatch;
    if (inputTokens.size() != 0)
    {
        reuseMatch = matchReuse(reuseScope, inputTokens, textOnly.value_or(this->textOnly()));
    }

    return std::make_shared<KvCache>(*this, std::move(reuseScope), std::move(reuseMatch), std::move(id),
        std::move(priorityCb), expectedPromptLength, textOnly, enableRequestStats, std::move(retentionConfig));
}

void KvCacheManager::refreshRetention()
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    mStorage->refreshRetention();
}

bool KvCacheManager::prefetchReuse(ReuseScope reuseScope, TokenSpan inputTokens, CacheLevel target, bool knownNoDigest)
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    if (target < kHotLevel || target >= mStorage->numCacheLevels())
        throw std::out_of_range("Prefetch target is outside the configured cache tiers");
    auto reuseMatch = matchReuse(reuseScope, inputTokens, knownNoDigest);
    auto cache = std::make_shared<KvCache>(*this, std::move(reuseScope), std::move(reuseMatch), std::nullopt,
        KvCache::PriorityCb{}, std::nullopt, knownNoDigest, false, std::nullopt, /*isPrefetch=*/true);
    auto close = FuncGuard([&cache]() { cache->close(); });
    return cache->prefetch(target);
}

BlockRadixTree::ReuseMatch KvCacheManager::matchReuse(
    ReuseScope const& reuseScope, TokenSpan inputTokens, bool knownNoDigest) const
{
    KVCM2_REJECT_IF_POISONED();
    return mRadixTree->match(reuseScope, inputTokens, knownNoDigest, enablePartialMatch(), mConfig.reuseMatchBackoff);
}

int KvCacheManager::probeReuse(ReuseScope reuseScope, TokenSpan inputTokens, bool knownNoDigest) const
{
    KVCM2_REJECT_IF_POISONED();
    // Shared: the traversal is read-only and the matched Block* never escape this call.
    auto const apiLock = lockShared();
    return matchReuse(reuseScope, inputTokens, knownNoDigest).numTokens;
}

std::optional<BlockKey> KvCacheManager::probeFirstNewBlockKey(
    ReuseScope reuseScope, TokenSpan inputTokens, bool knownNoDigest) const
{
    // The matched Block* remain protected until the key has been copied out.
    // Call matchReuse directly: probeReuse would nest shared-lock acquisitions.
    auto const apiLock = lockShared();
    auto const match = matchReuse(reuseScope, inputTokens, knownNoDigest);
    int const blockSize = tokensPerBlock();
    int const blockIndex = match.numTokens / blockSize;
    size_t const begin = static_cast<size_t>(blockIndex) * static_cast<size_t>(blockSize);
    if (begin + static_cast<size_t>(blockSize) > static_cast<size_t>(inputTokens.size()))
    {
        return std::nullopt;
    }

    // Derive the boundary from the final, pruned match, not the raw token path.
    // A partial match can end in a block with a different suffix, so only the
    // preceding fully matched block supplies the query's exact prefix key.
    BlockKey const previousKey
        = blockIndex == 0 ? RootBlock::makeKey(reuseScope) : match.blocks[BlockOrdinal{blockIndex - 1}]->key;
    return Block::makeKey(previousKey, inputTokens.begin() + begin, static_cast<size_t>(blockSize), knownNoDigest);
}

// ---- Memory pool queries --------------------------------------------------

MemAddress KvCacheManager::getMemPoolBaseAddress(
    LayerId layerId, DataRole role, std::optional<PageIndexMode> indexMode) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);

    if (!indexMode.has_value())
    {
        if (mConfig.enableSwaScratchReuse())
        {
            throw std::invalid_argument("index_mode must be provided when SWA scratch reuse is enabled");
        }
        indexMode = PageIndexMode::SHARED;
    }

    PoolGroupIndex pgIdx = mStorage->getPoolGroupIndex(attr.lifeCycleId);
    MemAddress addr = mStorage->getMemPoolBaseAddress(pgIdx, attr.poolIndex);
    if (*indexMode == PageIndexMode::SHARED)
    {
        addr = MemAddress(addr + attr.offset);
    }
    return addr;
}

bool KvCacheManager::isSparse(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    auto const pgIdx = mStorage->getPoolGroupIndex(kHotLevel, attr.lifeCycleId);
    return mStorage->poolMedium(kHotLevel, pgIdx, attr.poolIndex) == CacheTier::HOST_MEM;
}

size_t KvCacheManager::getNumBytes(CacheLevel level, CacheTier medium) const
{
    return mStorage->getNumBytes(level, medium);
}

std::optional<CacheTier> KvCacheManager::sparseBackingMedium() const
{
    return mStorage->sparseBackingMedium();
}

int KvCacheManager::getPageStride(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    return exactDiv(static_cast<int>(attr.size), attr.expansion);
}

size_t KvCacheManager::getPageIndexUpperBound(LayerId layerId, DataRole role) const
{
    auto const apiLock = lockShared();
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    LifeCycleId lc = attr.lifeCycleId;
    PoolGroupIndex pg = mStorage->getPoolGroupIndex(lc);
    SlotCount const numSlots = mStorage->numSlots(pg, kHotLevel);
    auto slotSizes = mStorage->slotSize(pg);
    size_t slotSize = slotSizes.at(attr.poolIndex);
    return (exactDiv(slotSize, attr.size) * slotCountToSizeT(numSlots) - exactDiv(attr.offset, attr.size))
        * static_cast<size_t>(attr.expansion);
}

int KvCacheManager::getPageIndexScale(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    return mStorage->mSlotToPageIndices.at(attr.lifeCycleId).at(attr.poolIndex);
}

PageIndexConverter KvCacheManager::getPageIndexConverter(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    int scale = mStorage->mSlotToPageIndices.at(attr.lifeCycleId).at(attr.poolIndex);
    int offset = exactDiv(static_cast<int>(attr.offset), static_cast<int>(attr.size));
    return PageIndexConverter{scale, attr.expansion, offset, attr.scratchPagesPerBlock};
}

int KvCacheManager::_scratchSlotsFor(
    LifeCycleId lcId, int maxBlocksPerSeq, std::optional<int> maxTotalInputTokens) const
{
    auto const blocks = _maxScratchBlocks(lcId, maxBlocksPerSeq, maxTotalInputTokens);
    if (blocks <= 0)
    {
        return 0;
    }
    // The same fraction KvCache::_takeExcessScratchSlots converts with, so what a request
    // takes and what the batch stages for it cannot disagree. It is the widest ratio any
    // of the life cycle's (layer, pool) pairs reaches, and a narrower one would leave the
    // widest layer's slots past the end of the row.
    return mStorage->slotUtilFracMax(lcId).ceilMul(blocks);
}

int KvCacheManager::maxScratchSlots(int maxBlocksPerSeq) const
{
    int result = 0;
    for (LifeCycleId lcId{0}; lcId < numLifeCycles(); ++lcId)
    {
        result = std::max(result, _scratchSlotsFor(lcId, maxBlocksPerSeq));
    }
    return result;
}

int KvCacheManager::_maxScratchBlocks(
    LifeCycleId lcId, int maxBlocksPerSeq, std::optional<int> maxTotalInputTokens) const
{
    if (!mConfig.enableSwaScratchReuse() || maxBlocksPerSeq <= 0)
    {
        return 0;
    }
    auto const* attn = std::get_if<AttnLifeCycle>(&lifeCycles().getLifeCycle(lcId));
    if (attn == nullptr || !attn->windowSize.has_value())
    {
        return 0;
    }
    auto const tpb = tokensPerBlock();
    auto const rewind = mConfig.swaScratchReuse->maxRewindLen;

    // Capacity-limited: a request whose history is empty and whose capacity fills the page
    // table. Deferred to getStaleRange rather than restated, since which blocks are stale
    // turns on the sink count and on the in-flight token the window always keeps.
    auto const capacity = static_cast<int64_t>(maxBlocksPerSeq) * tpb;
    auto const stale = attn->getStaleRange(static_cast<int>(std::max<int64_t>(0, capacity - rewind)), tpb);
    auto bound = static_cast<int64_t>(stale.end.value() - stale.beg.value());

    if (maxTotalInputTokens.has_value())
    {
        // The subtraction is in tokens, before the division: converting to blocks first and
        // subtracting after would lose the alignment this is exact at. The bound is tight --
        // the count is largest when the stale end falls on a block boundary, which is the
        // case this quotient gives.
        auto const reach = static_cast<int64_t>(*maxTotalInputTokens) - *attn->windowSize - rewind + 1;
        bound = std::min(bound, reach / tpb);
    }
    return static_cast<int>(std::max<int64_t>(0, bound));
}

std::optional<bool> KvCacheManager::supportsIndexMode(PageIndexMode mode, LifeCycleId lcId) const
{
    switch (mode)
    {
    case PageIndexMode::PER_LAYER: return true;
    case PageIndexMode::SHARED:
    {
        if (!mConfig.enableSwaScratchReuse())
        {
            return true;
        }
        // A life cycle with no sliding window yields an empty scratch range whatever the
        // watermarks, so no request can ever hold scratch slots for it.
        auto const* attn = std::get_if<AttnLifeCycle>(&lifeCycles().getLifeCycle(lcId));
        bool const mayHoldScratch = attn != nullptr && attn->windowSize.has_value();
        return mayHoldScratch ? std::optional<bool>(std::nullopt) : true;
    }
    }
    return std::nullopt;
}

// ---- getAggregatedPages ---------------------------------------------------

std::vector<AggregatedPageDesc> KvCacheManager::getAggregatedPages(std::vector<BufferId> const& buffers) const
{
    auto const apiLock = lockShared();
    using Key = std::pair<LifeCycleId, PoolIndex>;

    struct Entry
    {
        size_t start;
        size_t end;
        ExpandedBuffer eb;
    };

    std::map<Key, std::vector<Entry>> groups;

    for (auto const& bufferId : buffers)
    {
        auto it = mStorage->mBufferAttr.find(bufferId);
        if (it == mStorage->mBufferAttr.end())
            throw std::out_of_range("getAggregatedPages: unknown buffer id");

        auto const& attr = it->second;
        size_t start = attr.offset;
        size_t end = attr.offset + attr.size;
        Key key{attr.lifeCycleId, attr.poolIndex};
        groups[key].push_back({start, end, ExpandedBuffer{bufferId, attr.expansion}});
    }

    std::vector<AggregatedPageDesc> result;
    for (auto& [key, entries] : groups)
    {
        auto [lifeCycleId, poolIdx] = key;
        auto pgIdx = mStorage->getPoolGroupIndex(lifeCycleId);

        std::sort(entries.begin(), entries.end(), [](Entry const& a, Entry const& b) { return a.start < b.start; });

        auto const poolBase = mStorage->getMemPoolBaseAddress(pgIdx, poolIdx);
        size_t stride = mStorage->slotSize(pgIdx).at(poolIdx);

        auto flush
            = [&, lifeCycleId = lifeCycleId](size_t start, size_t end, std::vector<ExpandedBuffer>& buffersInRange)
        {
            result.push_back(AggregatedPageDesc{
                MemAddress(poolBase + start), end - start, stride, lifeCycleId, std::move(buffersInRange)});
        };

        size_t currentStart = entries.front().start;
        size_t currentEnd = entries.front().end;
        std::vector<ExpandedBuffer> currentBuffers{entries.front().eb};
        for (size_t i = 1; i < entries.size(); ++i)
        {
            if (entries[i].start == currentEnd)
            {
                currentEnd = entries[i].end;
                currentBuffers.push_back(entries[i].eb);
                continue;
            }

            flush(currentStart, currentEnd, currentBuffers);
            currentStart = entries[i].start;
            currentEnd = entries[i].end;
            currentBuffers = {entries[i].eb};
        }
        flush(currentStart, currentEnd, currentBuffers);
    }

    return result;
}

// ---- Pool group layout -----------------------------------------------------

TypedVec<PoolGroupIndex, PoolGroupDesc> KvCacheManager::poolGroupDescs() const
{
    // Shared: PoolGroupDesc carries numSlots, which _adjustLevel changes on a pool resize.
    auto const apiLock = lockShared();
    return mStorage->poolGroupDescs();
}

// ---- Query / info ---------------------------------------------------------

int KvCacheManager::tokensPerBlock() const noexcept
{
    return mRadixTree->tokensPerBlock();
}

bool KvCacheManager::enablePartialMatch() const noexcept
{
    return mConfig.enablePartialReuse;
}

bool KvCacheManager::enablePartialCommit() const noexcept
{
    return mConfig.enablePartialCommit;
}

int KvCacheManager::numLayers() const noexcept
{
    return static_cast<int>(mStorage->layerToLifeCycleIds().size());
}

std::vector<LayerId> KvCacheManager::layerIds() const
{
    std::vector<LayerId> ids;
    for (auto const& [lid, lc] : mStorage->layerToLifeCycleIds())
        ids.push_back(lid);
    return ids;
}

LayerGroupId KvCacheManager::getLayerGroupId(LayerId layerId) const
{
    auto const& lifeCycles = mStorage->layerToLifeCycleIds().at(layerId);
    TLLM_CHECK_WITH_INFO(lifeCycles.size() == 1,
        "Layer %d has buffers in %zu layer groups because its buffers use different reuse policies; "
        "look up its layer group per buffer with get_buffer_group_id(layer_id, role) / getBufferGroupId(layerId, role)",
        layerId, lifeCycles.size());
    return lifeCycles.front();
}

CacheGroupId KvCacheManager::getBufferGroupId(LayerId layerId, DataRole const& role) const
{
    return mStorage->getBufferAttr(layerId, role).lifeCycleId;
}

TypedVec<LayerGroupId, std::vector<LayerId>> KvCacheManager::layerGrouping() const
{
    LifeCycleId numLc = mLifeCycles.size();
    TypedVec<LayerGroupId, std::vector<LayerId>> result(numLc);
    for (auto const& [lid, lifeCycles] : mStorage->layerToLifeCycleIds())
    {
        result.at(getLayerGroupId(lid)).push_back(lid);
    }
    return result;
}

TypedVec<CacheGroupId, std::vector<BufferId>> KvCacheManager::bufferGrouping() const
{
    TypedVec<CacheGroupId, std::vector<BufferId>> result(mLifeCycles.size());
    std::unordered_map<LayerId, std::vector<BufferId>> buffersByLayer;
    for (auto const& buffer : allBufferIds())
    {
        buffersByLayer[buffer.layerId].push_back(buffer);
    }
    // Keep layer order consistent with layerGrouping() for consumers that pair
    // group membership with a declared list of global model layer IDs.
    for (auto const& [layer, lifeCycles] : mStorage->layerToLifeCycleIds())
    {
        for (auto const& buffer : buffersByLayer.at(layer))
        {
            result.at(getBufferGroupId(buffer.layerId, buffer.role)).push_back(buffer);
        }
    }
    return result;
}

// ---- Resize ---------------------------------------------------------------

bool KvCacheManager::resize(CacheLevel level, size_t quota, bool bestEfforts)
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    // Same precondition as adjust(): _adjustLevel may defragment, invalidating any page index an
    // ACTIVE cache holds.
    for (KvCache* kvc : mLivingKvCaches)
        TLLM_CHECK_WITH_INFO(kvc->status() == KvCache::Status::SUSPENDED,
            "level adjustment requires every KvCache to be SUSPENDED: _adjustLevel may "
            "defragment, which relocates pages and invalidates indices an ACTIVE cache holds");
    if (bestEfforts)
        throw std::runtime_error("best_efforts resize not implemented");
    try
    {
        _adjustLevel(level, quota);
        return true;
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("Failed to resize cache level %d to %zu: %s", level.value(), quota, e.what());
        return false;
    }
}

size_t KvCacheManager::getQuota(CacheLevel level) const
{
    auto const apiLock = lockShared();
    auto const& storage = *mStorage->mLevels.at(level).storage;
    return storage.roundedNumBytes(storage.cacheTier());
}

// ---- Statistics ----------------------------------------------------------

void KvCacheManager::commitStats(
    KVCacheStatsDelta const& stats, IterationStatsByLifeCycle const& iterationStatsByLifeCycle)
{
    if (!mConfig.enableStats)
    {
        return;
    }

    _updateIterationPeakNumBlocks();
    mCommittedStats.add(stats);
    for (auto const& [lifeCycle, iterationStats] : iterationStatsByLifeCycle)
    {
        if (!iterationStats.empty())
        {
            mIterationStatsByLifeCycle[lifeCycle].add(iterationStats);
        }
    }
}

KVCacheStatsDelta KvCacheManager::getCommittedStats() const
{
    auto const apiLock = lockShared();
    return mCommittedStats.copy();
}

IterationStatsByLifeCycle KvCacheManager::getAndResetIterationStats()
{
    auto const apiLock = lockExclusive();
    IterationStatsByLifeCycle stats;
    for (auto const& [lifeCycle, delta] : mIterationStatsByLifeCycle)
    {
        if (!delta.empty())
        {
            stats.emplace(lifeCycle, delta.copy());
        }
    }
    mIterationStatsByLifeCycle.clear();
    return stats;
}

void KvCacheManager::commitSsmSnapshotIterationStats(SsmSnapshotIterationStatsByLifeCycle const& statsByLifeCycle)
{
    if (!mConfig.enableStats)
    {
        return;
    }
    for (auto const& [lifeCycle, iterationStats] : statsByLifeCycle)
    {
        if (!iterationStats.empty())
        {
            mSsmSnapshotIterationStatsByLifeCycle[lifeCycle].add(iterationStats);
        }
    }
}

SsmSnapshotIterationStatsByLifeCycle KvCacheManager::getAndResetSsmSnapshotIterationStats()
{
    auto const lock = lockExclusive();
    SsmSnapshotIterationStatsByLifeCycle stats;
    for (auto const& [lifeCycle, delta] : mSsmSnapshotIterationStatsByLifeCycle)
    {
        if (!delta.empty())
        {
            stats.emplace(lifeCycle, delta.copy());
        }
    }
    mSsmSnapshotIterationStatsByLifeCycle.clear();
    return stats;
}

TypedVec<PoolGroupIndex, StorageStatistics> KvCacheManager::getStorageStatistics(CacheLevel cacheLevel) const
{
    auto const apiLock = lockShared();
    TypedVec<PoolGroupIndex, StorageStatistics> result;
    PoolGroupIndex const numPoolGroups = mStorage->numPoolGroups(cacheLevel);
    result.reserve(numPoolGroups);
    for (PoolGroupIndex poolGroup{0}; poolGroup < numPoolGroups; ++poolGroup)
    {
        result.push_back(mStorage->getStatistics(cacheLevel, poolGroup));
    }
    return result;
}

TypedVec<LifeCycleId, PoolGroupIndex> KvCacheManager::getLifeCyclePoolGroupIndices(CacheLevel cacheLevel) const
{
    TypedVec<LifeCycleId, PoolGroupIndex> result;
    result.reserve(mLifeCycles.size());
    for (LifeCycleId lifeCycle{0}; lifeCycle < mLifeCycles.size(); ++lifeCycle)
    {
        result.push_back(mStorage->getPoolGroupIndex(cacheLevel, lifeCycle));
    }
    return result;
}

void KvCacheManager::commitReusedBlocksByLevel(ReusedBlocksByLevelByLifeCycle const& byLifeCycle)
{
    for (auto const& [lifeCycle, byLevel] : byLifeCycle)
    {
        if (!byLevel.empty())
        {
            mIterReusedBlocksByLevel[lifeCycle].add(byLevel);
        }
    }
}

ReusedBlocksByLevelByLifeCycle KvCacheManager::getAndResetIterationReusedBlocksByLevel()
{
    auto const apiLock = lockExclusive();
    ReusedBlocksByLevelByLifeCycle byLifeCycle;
    byLifeCycle.swap(mIterReusedBlocksByLevel);
    return byLifeCycle;
}

void KvCacheManager::commitConnectorReusedBlocks(ConnectorReusedBlocksByLifeCycle const& byLifeCycle)
{
    for (auto const& [lifeCycle, numBlocks] : byLifeCycle)
    {
        TLLM_CHECK_DEBUG(numBlocks >= 0);
        if (numBlocks != 0)
        {
            mIterConnectorReusedBlocks[lifeCycle] += numBlocks;
        }
    }
}

ConnectorReusedBlocksByLifeCycle KvCacheManager::getAndResetIterationConnectorReusedBlocks()
{
    auto const apiLock = lockExclusive();
    ConnectorReusedBlocksByLifeCycle byLifeCycle;
    byLifeCycle.swap(mIterConnectorReusedBlocks);
    return byLifeCycle;
}

void KvCacheManager::recordRequestSuspended()
{
    auto const apiLock = lockExclusive();
    if (!mConfig.enableStats)
    {
        return;
    }
    ++mIterSuspendedRequests;
}

void KvCacheManager::recordRequestResumed()
{
    auto const apiLock = lockExclusive();
    if (!mConfig.enableStats)
    {
        return;
    }
    ++mIterResumedRequests;
}

void KvCacheManager::recordDiskPrefetchBlocks(int64_t numBlocks)
{
    TLLM_CHECK_DEBUG(numBlocks >= 0);
    if (!mConfig.enableStats)
    {
        return;
    }
    mIterDiskPrefetchBlocks += numBlocks;
}

int64_t KvCacheManager::getAndResetIterationDiskPrefetchBlocks()
{
    auto const apiLock = lockExclusive();
    auto const numBlocks = mIterDiskPrefetchBlocks;
    mIterDiskPrefetchBlocks = 0;
    return numBlocks;
}

void KvCacheManager::commitCachedTokensByLevel(CountsByLevel const& counts)
{
    TLLM_CHECK_DEBUG(std::all_of(counts.begin(), counts.end(), [](int64_t count) { return count >= 0; }));
    if (!mConfig.enableStats)
    {
        return;
    }
    addCountsByLevel(mIterCachedTokensByLevel, counts);
}

CountsByLevel KvCacheManager::getAndResetIterationCachedTokensByLevel()
{
    auto const apiLock = lockExclusive();
    CountsByLevel counts;
    std::swap(counts, mIterCachedTokensByLevel);
    return counts;
}

void KvCacheManager::commitConnectorCachedTokens(int64_t numTokens)
{
    TLLM_CHECK_DEBUG(numTokens >= 0);
    if (!mConfig.enableStats)
    {
        return;
    }
    mIterConnectorCachedTokens += numTokens;
}

int64_t KvCacheManager::getAndResetIterationConnectorCachedTokens()
{
    auto const apiLock = lockExclusive();
    auto const numTokens = mIterConnectorCachedTokens;
    mIterConnectorCachedTokens = 0;
    return numTokens;
}

std::pair<int64_t, int64_t> KvCacheManager::getAndResetIterationSuspendResumeStats()
{
    auto const apiLock = lockExclusive();
    // Suspend/resume is a per-request, manager-level event (not per-pool-group), so it
    // is drained alongside getAndResetIterationStats once per iteration-stats fetch.
    auto const suspended = mIterSuspendedRequests;
    auto const resumed = mIterResumedRequests;
    mIterSuspendedRequests = 0;
    mIterResumedRequests = 0;
    return {suspended, resumed};
}

PeakBlockStatsByCacheLevel KvCacheManager::_currentBlockStatsByCacheLevel() const
{
    PeakBlockStatsByCacheLevel result(mStorage->numCacheLevels());
    for (CacheLevel cacheLevel{0}; cacheLevel < mStorage->numCacheLevels(); ++cacheLevel)
    {
        auto& levelStats = result[cacheLevel];
        levelStats.resize(mStorage->numPoolGroups(cacheLevel));
        for (PoolGroupIndex poolGroup{0}; poolGroup < mStorage->numPoolGroups(cacheLevel); ++poolGroup)
        {
            auto const stats = mStorage->getStatistics(cacheLevel, poolGroup);
            levelStats[poolGroup] = {stats.available(), stats.unavailable(), stats.evictable};
        }
    }
    return result;
}

void KvCacheManager::_resetIterationPeakNumBlocks(std::optional<CacheLevel> cacheLevel)
{
    if (!cacheLevel.has_value())
    {
        mIterationPeakNumBlocksByCacheLevel = _currentBlockStatsByCacheLevel();
        return;
    }

    PeakBlockStatsByPoolGroup levelStats(mStorage->numPoolGroups(*cacheLevel));
    for (PoolGroupIndex poolGroup{0}; poolGroup < mStorage->numPoolGroups(*cacheLevel); ++poolGroup)
    {
        auto const stats = mStorage->getStatistics(*cacheLevel, poolGroup);
        levelStats[poolGroup] = {stats.available(), stats.unavailable(), stats.evictable};
    }
    mIterationPeakNumBlocksByCacheLevel[*cacheLevel] = std::move(levelStats);
}

void KvCacheManager::_updateIterationPeakNumBlocks()
{
    auto const current = _currentBlockStatsByCacheLevel();
    for (CacheLevel cacheLevel{0}; cacheLevel < current.size(); ++cacheLevel)
    {
        auto& peakLevel = mIterationPeakNumBlocksByCacheLevel[cacheLevel];
        for (PoolGroupIndex poolGroup{0}; poolGroup < current[cacheLevel].size(); ++poolGroup)
        {
            auto& peak = peakLevel[poolGroup];
            auto const& value = current[cacheLevel][poolGroup];
            peak.available = std::max(peak.available, value.available);
            peak.unavailable = std::max(peak.unavailable, value.unavailable);
            peak.evictable = std::max(peak.evictable, value.evictable);
        }
    }
}

PeakBlockStatsByPoolGroup KvCacheManager::getAndResetIterationPeakBlockStats(CacheLevel cacheLevel)
{
    // Exclusive: samples live storage, then resets the accumulator.
    auto const apiLock = lockExclusive();
    _updateIterationPeakNumBlocks();
    PeakBlockStatsByPoolGroup peak = mIterationPeakNumBlocksByCacheLevel.at(cacheLevel);
    _resetIterationPeakNumBlocks(cacheLevel);
    return peak;
}

PeakBlockStatsByCacheLevel KvCacheManager::getAndResetIterationPeakBlockStatsByLevel()
{
    auto const apiLock = lockExclusive();
    _updateIterationPeakNumBlocks();
    PeakBlockStatsByCacheLevel peak = mIterationPeakNumBlocksByCacheLevel;
    _resetIterationPeakNumBlocks();
    return peak;
}

void KvCacheManager::markStatsDirty(std::optional<RequestIdType> kvCacheId)
{
    auto const apiLock = lockExclusive();
    if (kvCacheId.has_value())
    {
        mDirtyStatsKvCacheIds.insert(*kvCacheId);
    }
}

void KvCacheManager::clearStatsDirty(std::optional<RequestIdType> kvCacheId)
{
    auto const apiLock = lockExclusive();
    if (kvCacheId.has_value())
    {
        mDirtyStatsKvCacheIds.erase(*kvCacheId);
    }
}

std::unordered_set<RequestIdType> KvCacheManager::getDirtyStatsKvCacheIds() const
{
    auto const apiLock = lockShared();
    return mDirtyStatsKvCacheIds;
}

void KvCacheManager::markStatsExcluded(std::optional<RequestIdType> kvCacheId)
{
    auto const apiLock = lockExclusive();
    if (kvCacheId.has_value())
    {
        mStatsExcludedKvCacheIds.insert(*kvCacheId);
        clearStatsDirty(kvCacheId);
    }
}

void KvCacheManager::clearStatsExcluded(std::optional<RequestIdType> kvCacheId)
{
    auto const apiLock = lockExclusive();
    if (kvCacheId.has_value())
    {
        mStatsExcludedKvCacheIds.erase(*kvCacheId);
    }
}

bool KvCacheManager::isStatsExcluded(std::optional<RequestIdType> kvCacheId) const
{
    auto const apiLock = lockShared();
    return kvCacheId.has_value() && mStatsExcludedKvCacheIds.find(*kvCacheId) != mStatsExcludedKvCacheIds.end();
}

TypedVec<CacheLevel, CacheTier> KvCacheManager::cacheTierList() const
{
    TypedVec<CacheLevel, CacheTier> result;
    result.reserve(mStorage->mLevels.size());
    for (auto const& lvl : mStorage->mLevels)
    {
        result.push_back(lvl.cacheTier);
    }
    return result;
}

std::vector<BufferId> KvCacheManager::allBufferIds() const
{
    std::vector<BufferId> result;
    result.reserve(mStorage->mBufferAttr.size());
    for (auto const& item : mStorage->mBufferAttr)
        result.push_back(item.first);
    return result;
}

int KvCacheManager::clampMaxSeqLenForMem(int batchSize, int tokenNumUpperBound) const
{
    auto const apiLock = lockShared();
    TLLM_CHECK_DEBUG(batchSize > 0);
    int tokPerBlock = tokensPerBlock();
    PoolGroupIndex numPg = mStorage->numPoolGroups();
    auto const& lcs = mLifeCycles;
    auto const& lcGrouping = mStorage->lifeCycleGrouping(kHotLevel);

    // Remaining slot counts per pool group.
    TypedVec<PoolGroupIndex, SlotCount> remainingSlots(numPg);
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        remainingSlots[pgIdx] = mStorage->numSlots(pgIdx);
    }

    // Compute required slot counts per pool group for a given seq_len.
    auto getNumSlots = [&](int seqLen) -> TypedVec<PoolGroupIndex, SlotCount>
    {
        TypedVec<PoolGroupIndex, SlotCount> ret(numPg, 0);
        for (LifeCycleId lifeCycleId{0}; lifeCycleId < lcs.size(); ++lifeCycleId)
        {
            auto staleRange = getStaleRange(lcs[lifeCycleId], seqLen, tokPerBlock);
            int numStaleBlocks = staleRange.end - staleRange.beg;
            int numSlots = divUp(seqLen, tokPerBlock) - numStaleBlocks;
            auto pgIdx = lcGrouping[lifeCycleId];
            ret[pgIdx] += numSlots;
        }
        return ret;
    };

    // Reserve slots for (batch_size - 1) minimal sequences.
    auto minSlots = getNumSlots(1);
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        TLLM_CHECK_DEBUG(minSlots[pgIdx] >= 0);
        SlotCount const reservedSlots = minSlots[pgIdx] * (batchSize - 1);
        remainingSlots[pgIdx] -= reservedSlots;
        if (remainingSlots[pgIdx] < 0)
        {
            return 0;
        }
    }

    auto isEnough = [&](int numBlocks) -> bool
    {
        auto needed = getNumSlots(numBlocks * tokPerBlock);
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            TLLM_CHECK_DEBUG(needed[pgIdx] >= 0);
            if (needed[pgIdx] > remainingSlots[pgIdx])
            {
                return false;
            }
        }
        return true;
    };

    if (!isEnough(1))
    {
        return 0;
    }
    int lb = 1;
    int ub = divUp(tokenNumUpperBound, tokPerBlock);
    if (isEnough(ub))
    {
        return tokenNumUpperBound;
    }
    while (lb < ub - 1)
    {
        int mid = (lb + ub) / 2;
        if (isEnough(mid))
        {
            lb = mid;
        }
        else
        {
            ub = mid;
        }
    }
    return std::min(lb * tokPerBlock, tokenNumUpperBound);
}

void KvCacheManager::_adjustLevel(CacheLevel level, size_t quota)
{
    auto const& ratioList = _getTargetRatioList(level);
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* persistent = nullptr;
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> persistentPages;
    if (mStorage->isLastLevel(level))
    {
        persistentPages = _gatherLastLevelPersistentPages();
        persistent = &persistentPages;
    }
    mStorage->adjustCacheLevel(level, quota, ratioList, persistent);
}

TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> KvCacheManager::_gatherLastLevelPersistentPages() const
{
    CacheLevel lastLevel = mStorage->numCacheLevels() - 1;
    PoolGroupIndex numPg = mStorage->numPoolGroups(lastLevel);
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> result(numPg);

    for (KvCache* kvc : mLivingKvCaches)
    {
        TLLM_CHECK_DEBUG(kvc->status() == KvCache::Status::SUSPENDED);
        for (auto const& sb : kvc->blocks())
        {
            for (auto const& beamPages : sb.pages)
            {
                for (LifeCycleId lc{0}; lc < beamPages.size(); ++lc)
                {
                    // Mirrors Python: holder must be _PageHolder type (suspended state).
                    if (blockPageIsNull(beamPages[lc]))
                    {
                        continue;
                    }
                    TLLM_CHECK_DEBUG_WITH_INFO(std::holds_alternative<SharedPtr<PageHolder>>(beamPages[lc]),
                        "Non-null holder must be PageHolder in suspended state");
                    auto const& pg = blockPageGetPage(beamPages[lc]);
                    if (!pg)
                    {
                        continue;
                    }
                    // Mirrors Python assertions for invariant checking.
                    TLLM_CHECK_DEBUG_WITH_INFO(
                        pg->status() == PageStatus::HELD, "Page in suspended KvCache must be HELD");
                    TLLM_CHECK_DEBUG_WITH_INFO((pg->scheduledForEviction() == (pg->cacheLevel != lastLevel)),
                        "Eviction scheduling invariant violated");
                    if (pg->scheduledForEviction())
                    {
                        continue;
                    }
                    PoolGroupIndex pgIdx = mStorage->getPoolGroupIndex(lastLevel, lc);
                    result[pgIdx].push_back(pg);
                }
            }
        }
    }
    return result;
}

// ---- KvCache registry -----------------------------------------------------

void KvCacheManager::registerKvCache(KvCache* kvc)
{
    auto const apiLock = lockExclusive();
    mLivingKvCaches.insert(kvc);
    ++mNumCreatedKvCaches;
}

void KvCacheManager::unregisterKvCache(KvCache* kvc)
{
    auto const apiLock = lockExclusive();
    mLivingKvCaches.erase(kvc);
}

void KvCacheManager::tryUpdateTargetRatios()
{
    if (mNumSampledKvCaches - mLastUpdateNumSampledKvCaches < mConfig.poolRebalance.targetRatioUpdateInterval)
        return;
    mLastUpdateNumSampledKvCaches = mNumSampledKvCaches;

    int tokensPerBlock = mConfig.tokensPerBlock;
    int avgReusedLength = static_cast<int>(std::round(mAvgReusedLength.value()));
    int avgCapacity = static_cast<int>(std::round(std::sqrt(mAvgSqrCapacity.value())));
    int avgHistoryLength = static_cast<int>(std::round(std::sqrt(mAvgSqrHistoryLength.value())));
    int avgBeamWidth = std::max(1, static_cast<int>(std::round(mAvgBeamWidth.value())));
    int avgPromptLength = std::max(0, static_cast<int>(std::round(mAvgPromptLength.value())));
    if (avgCapacity > 0)
    {
        auto const lifeCycleRatio = mStorage->ratioFromLength(
            kHotLevel, tokensPerBlock, avgHistoryLength, avgCapacity, avgBeamWidth, avgPromptLength);
        mTargetRatioListHot = mStorage->constrainPoolGroupRatio(mStorage->toPoolGroupRatio(kHotLevel, lifeCycleRatio));
    }
    if (avgReusedLength > 0)
    {
        CacheLevel const coldLevel = mStorage->numCacheLevels() > CacheLevel{1} ? CacheLevel{1} : kHotLevel;
        // The cold tiers hold committed pages, which are canonicalized to beam 0
        // (_commitBlock() truncates the other beams away), so they never carry a
        // beam factor no matter how wide the requests that produced them were.
        auto const lifeCycleRatio
            = mStorage->ratioFromLength(coldLevel, tokensPerBlock, avgReusedLength, avgReusedLength);
        mTargetRatioListCold = mStorage->toPoolGroupRatio(coldLevel, lifeCycleRatio);
    }
}

TypedVec<PoolGroupIndex, float> KvCacheManager::_currentHotRatio() const
{
    return mStorage->getRatioList(kHotLevel);
}

TypedVec<PoolGroupIndex, float> KvCacheManager::_currentColdRatios() const
{
    CacheLevel numLevels = mStorage->numCacheLevels();
    if (numLevels == CacheLevel{1})
    {
        return _currentHotRatio();
    }
    PoolGroupIndex numPg = mStorage->numPoolGroups(CacheLevel{1});
    TypedVec<PoolGroupIndex, float> result(numPg, 0.f);
    for (CacheLevel lvl{1}; lvl < numLevels; ++lvl)
    {
        auto ratios = mStorage->getRatioList(lvl);
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            result[pgIdx] += ratios[pgIdx];
        }
    }
    float denom = static_cast<float>((numLevels - 1).value());
    for (auto& r : result)
    {
        r /= denom;
    }
    return result;
}

// ---- needAdjustment / adjust -----------------------------------------------

TypedVec<PoolGroupIndex, float> const& KvCacheManager::_getTargetRatioList(CacheLevel level) const
{
    return (level == kHotLevel) ? mTargetRatioListHot : mTargetRatioListCold;
}

bool KvCacheManager::_needAdjustment(CacheLevel level) const
{
    auto const& target = _getTargetRatioList(level);
    auto current = (level == kHotLevel) ? _currentHotRatio() : _currentColdRatios();
    float const tolerance = mConfig.poolRebalance.ratioTolerance;
    for (PoolGroupIndex pgIdx{0}; pgIdx < target.size() && pgIdx < current.size(); ++pgIdx)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(current[pgIdx] > 0.f && target[pgIdx] > 0.f, "ratios must not be zero");
        float ratio = target[pgIdx] / current[pgIdx];
        // Symmetric in target vs current: compare the larger of the two ratios, so the test
        // does not depend on which side has drifted.
        if (std::max(ratio, 1.f / ratio) > 1.f + tolerance)
            return true;
    }
    return false;
}

bool KvCacheManager::needAdjustment() const
{
    auto const apiLock = lockShared();
    if (mNumSampledKvCaches < mConfig.poolRebalance.minSampledKvCaches)
        return false;
    double now = nowSeconds();
    if (now - mLastAdjustmentTime < mConfig.poolRebalance.cooldownSecs)
        return false;
    CacheLevel lastLevel = mStorage->numCacheLevels() - 1;
    return _needAdjustment(kHotLevel) || _needAdjustment(lastLevel);
}

void KvCacheManager::adjust()
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    for (KvCache* kvc : mLivingKvCaches)
        TLLM_CHECK_WITH_INFO(kvc->status() == KvCache::Status::SUSPENDED,
            "level adjustment requires every KvCache to be SUSPENDED: _adjustLevel may "
            "defragment, which relocates pages and invalidates indices an ACTIVE cache holds");

    CacheLevel numLevels = mStorage->numCacheLevels();
    for (CacheLevel level{0}; level < numLevels; ++level)
    {
        if (_needAdjustment(level))
            _adjustLevel(level, getQuota(level));
    }
    mLastAdjustmentTime = nowSeconds();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
