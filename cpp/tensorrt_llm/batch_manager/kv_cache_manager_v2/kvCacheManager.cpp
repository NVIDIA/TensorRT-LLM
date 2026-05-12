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
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/storage/core.h"
#include "kv_cache_manager_v2/utils/math.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

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

std::vector<int> PageIndexConverter::operator()(
    std::vector<int> const& baseIndices, std::optional<PageIndexMode> indexMode, ScratchDesc const* scratch) const
{
    if (!indexMode.has_value())
    {
        if (scratch)
        {
            throw std::invalid_argument("index_mode must be provided when scratch is active");
        }
        indexMode = PageIndexMode::SHARED;
    }

    int appliedLayerOffset = (*indexMode == PageIndexMode::PER_LAYER) ? layerOffset : 0;
    int scratchPages = scratchPagesPerBlock;

    std::vector<int> result;
    result.reserve(baseIndices.size() * static_cast<size_t>(expansion));

    for (int ordinal = 0; ordinal < static_cast<int>(baseIndices.size()); ++ordinal)
    {
        int index;
        if (scratch && scratch->range.contains(ordinal))
        {
            // Scratch block: slot IDs come from ScratchDesc, not base_indices.
            int blockPos = ordinal - scratch->range.beg;
            int totalOffset = blockPos * scratchPages;
            int slotIdx = totalOffset / scale;
            int slotId = scratch->slotIds[static_cast<size_t>(slotIdx)];
            int offset = totalOffset % scale;
            index = slotId * scale + (offset + appliedLayerOffset) % scale;
        }
        else if (baseIndices[static_cast<size_t>(ordinal)] == kBadPageIndex)
        {
            index = kBadPageIndex;
        }
        else
        {
            index = baseIndices[static_cast<size_t>(ordinal)] * scale + appliedLayerOffset;
        }
        for (int i = 0; i < expansion; ++i)
        {
            result.push_back(index != kBadPageIndex ? index * expansion + i : kBadPageIndex);
        }
    }
    return result;
}

std::vector<int> PageIndexConverter::operator()(int baseIndex) const
{
    return operator()(std::vector<int>{baseIndex}, std::nullopt, nullptr);
}

// ---------------------------------------------------------------------------
// KvCacheManager
// ---------------------------------------------------------------------------

KvCacheManager::KvCacheManager(KVCacheManagerConfig const& config)
    : mConfig(config)
    , mLifeCycles(config)
    , mAvgReusedLength(0.9999)
    , mAvgSqrCapacity(0.9999)
    , mAvgSqrHistoryLength(0.9999)
{
    mConfig.validate();

    mRadixTree = std::make_shared<BlockRadixTree>(mLifeCycles, mConfig.tokensPerBlock);

    StorageConfig storageConfig = createStorageConfig(mConfig);
    mStorage = std::make_shared<StorageManager>(mLifeCycles, storageConfig, mConfig.tokensPerBlock,
        mConfig.enableSwaScratchReuse, mConfig.typicalStep, mConfig.constraints);

    mTargetRatioListGpu = _currentGpuRatio();
    mTargetRatioListOther = _currentOtherRatios();

    mLastAdjustmentTime = nowSeconds();
}

KvCacheManager::~KvCacheManager()
{
    shutdown();
}

void KvCacheManager::shutdown()
{
    clearReusableBlocks();
    assert(mStorage);

    // Post-condition: after clearing all reusable blocks and with no active KvCaches,
    // no evictable pages should remain.
    for (auto const& lvl : mStorage->mLevels)
    {
        for (int pg = 0; pg < static_cast<int>(lvl.numPoolGroups()); ++pg)
            assert(lvl.controller.numEvictablePages(static_cast<PoolGroupIndex>(pg)) == 0);
    }

    mStorage->destroy();
}

void KvCacheManager::clearReusableBlocks()
{
    assert(mRadixTree);
    mRadixTree->clear();
}

std::shared_ptr<KvCache> KvCacheManager::createKvCache(std::optional<int64_t> loraTaskId,
    std::vector<TokenIdExt> const& inputTokens, std::optional<int64_t> id, KvCache::PriorityCb priorityCb)
{
    if (!priorityCb)
        priorityCb = [](BlockOrdinal, LifeCycleId) { return kPriorityDefault; };

    return std::make_shared<KvCache>(*this, loraTaskId, inputTokens, std::move(id), std::move(priorityCb));
}

// ---- Memory pool queries --------------------------------------------------

MemAddress KvCacheManager::getMemPoolBaseAddress(
    LayerId layerId, DataRole role, std::optional<PageIndexMode> indexMode) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);

    if (!indexMode.has_value())
    {
        if (mConfig.enableSwaScratchReuse)
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

int KvCacheManager::getPageStride(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    return exactDiv(static_cast<int>(attr.size), attr.expansion);
}

int KvCacheManager::getPageIndexUpperBound(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    LifeCycleId lc = attr.lifeCycleId;
    PoolGroupIndex pg = static_cast<PoolGroupIndex>(mStorage->getPoolGroupIndex(lc));
    int numSlots = mStorage->numSlots(pg, kGpuLevel);
    auto slotSizes = mStorage->slotSize(pg);
    int slotSize = slotSizes.at(static_cast<size_t>(attr.poolIndex));
    return (exactDiv(slotSize, static_cast<int>(attr.size)) * numSlots
               - exactDiv(static_cast<int>(attr.offset), static_cast<int>(attr.size)))
        * attr.expansion;
}

int KvCacheManager::getPageIndexScale(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    return mStorage->mSlotToPageIndices.at(static_cast<size_t>(attr.lifeCycleId))
        .at(static_cast<size_t>(attr.poolIndex));
}

PageIndexConverter KvCacheManager::getPageIndexConverter(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    auto const& layerAttr = mStorage->getLayerAttr(layerId);
    int scale = mStorage->mSlotToPageIndices.at(static_cast<size_t>(attr.lifeCycleId))
                    .at(static_cast<size_t>(attr.poolIndex));
    int offset = exactDiv(static_cast<int>(attr.offset), static_cast<int>(attr.size));
    int scratchPages = layerAttr.slotUtil.at(static_cast<size_t>(attr.poolIndex));
    return PageIndexConverter{scale, attr.expansion, offset, scratchPages};
}

std::optional<bool> KvCacheManager::supportsIndexMode(PageIndexMode mode) const
{
    switch (mode)
    {
    case PageIndexMode::PER_LAYER: return true;
    case PageIndexMode::SHARED: return mConfig.enableSwaScratchReuse ? std::optional<bool>(std::nullopt) : true;
    }
    return std::nullopt;
}

// ---- getAggregatedPages ---------------------------------------------------

std::vector<AggregatedPageDesc> KvCacheManager::getAggregatedPages(std::vector<BufferId> const& buffers) const
{
    // Group buffers by (lifeCycleId, poolIndex), sorted by byte offset within slot.
    using Key = std::pair<LifeCycleId, PoolIndex>;

    struct Entry
    {
        size_t start;
        size_t end;
        ExpandedBuffer eb;
    };

    std::map<Key, std::vector<Entry>> groups;

    for (auto const& bid : buffers)
    {
        auto it = mStorage->mBufferAttr.find(bid);
        if (it == mStorage->mBufferAttr.end())
            throw std::out_of_range("getAggregatedPages: unknown buffer id");
        auto const& attr = it->second;
        size_t start = attr.offset;
        size_t end = attr.offset + attr.size;
        Key key{attr.lifeCycleId, attr.poolIndex};
        groups[key].push_back({start, end, ExpandedBuffer{bid, attr.expansion}});
    }

    std::vector<AggregatedPageDesc> result;
    auto& gpuStorage = *mStorage->mLevels[0].storage; // GPU level

    for (auto& [key, entries] : groups)
    {
        auto [lc, poolIdx] = key;
        PoolGroupIndex pgIdx = static_cast<PoolGroupIndex>(mStorage->getPoolGroupIndex(lc));

        // Sort by start offset.
        std::sort(entries.begin(), entries.end(), [](Entry const& a, Entry const& b) { return a.start < b.start; });

        // Pool base = address of (pgIdx, poolIdx, slot 0).
        MemAddress poolBase = gpuStorage.getBaseAddress(pgIdx, poolIdx, SlotId(0));
        int stride = mStorage->slotSize(pgIdx).at(static_cast<size_t>(poolIdx));

        // Merge contiguous ranges.
        auto flush = [&](size_t cStart, size_t cEnd, std::vector<ExpandedBuffer>& bufs)
        {
            result.push_back(AggregatedPageDesc{
                MemAddress(poolBase + cStart), static_cast<int>(cEnd - cStart), stride, lc, std::move(bufs)});
        };

        size_t curStart = entries[0].start, curEnd = entries[0].end;
        std::vector<ExpandedBuffer> curBufs{entries[0].eb};
        for (size_t i = 1; i < entries.size(); ++i)
        {
            if (entries[i].start == curEnd)
            {
                curEnd = entries[i].end;
                curBufs.push_back(entries[i].eb);
            }
            else
            {
                flush(curStart, curEnd, curBufs);
                curStart = entries[i].start;
                curEnd = entries[i].end;
                curBufs = {entries[i].eb};
            }
        }
        flush(curStart, curEnd, curBufs);
    }

    return result;
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
    return mStorage->layerToLifeCycleIds().at(layerId);
}

std::vector<std::vector<LayerId>> KvCacheManager::layerGrouping() const
{
    int numLc = mLifeCycles.size();
    std::vector<std::vector<LayerId>> result(static_cast<size_t>(numLc));
    for (auto const& [lid, lc] : mStorage->layerToLifeCycleIds())
        result.at(static_cast<size_t>(lc)).push_back(lid);
    return result;
}

// ---- Resize ---------------------------------------------------------------

bool KvCacheManager::resize(CacheLevel level, size_t quota, bool bestEfforts)
{
    if (bestEfforts)
        throw std::runtime_error("best_efforts resize not implemented");
    try
    {
        _adjustLevel(level, quota);
        return true;
    }
    catch (std::exception const& e)
    {
        return false;
    }
}

size_t KvCacheManager::getQuota(CacheLevel level) const
{
    return mStorage->mLevels.at(static_cast<size_t>(level)).storage->totalQuota();
}

std::vector<CacheTier> KvCacheManager::cacheTierList() const
{
    std::vector<CacheTier> result;
    result.reserve(static_cast<size_t>(mStorage->numCacheLevels()));
    for (auto const& lvl : mStorage->mLevels)
        result.push_back(lvl.cacheTier);
    return result;
}

std::vector<BufferId> KvCacheManager::allBufferIds() const
{
    std::vector<BufferId> result;
    result.reserve(mStorage->mBufferAttr.size());
    for (auto const& [id, attr] : mStorage->mBufferAttr)
        result.push_back(id);
    return result;
}

int KvCacheManager::clampMaxSeqLenForMem(int batchSize, int tokenNumUpperBound) const
{
    assert(batchSize > 0);
    int tokPerBlock = tokensPerBlock();
    int numPg = mStorage->numPoolGroups();
    auto const& lcs = mLifeCycles;
    auto const& lcGrouping = mStorage->mLifeCycleGrouping;

    // Remaining slots per pool group.
    std::vector<int> remainingSlots(static_cast<size_t>(numPg));
    for (int pg = 0; pg < numPg; ++pg)
        remainingSlots[static_cast<size_t>(pg)] = mStorage->numSlots(static_cast<PoolGroupIndex>(pg));

    // Compute required slots per pool group for a given seq_len.
    auto getNumSlots = [&](int seqLen) -> std::vector<int>
    {
        std::vector<int> ret(static_cast<size_t>(numPg), 0);
        for (int lcId = 0; lcId < lcs.size(); ++lcId)
        {
            auto staleRange = getStaleRange(lcs[lcId], seqLen, tokPerBlock);
            int numStaleBlocks = staleRange.end - staleRange.beg;
            int numSlots = divUp(seqLen, tokPerBlock) - numStaleBlocks;
            auto pgIdx = lcGrouping[static_cast<size_t>(lcId)];
            ret[static_cast<size_t>(pgIdx)] += numSlots;
        }
        return ret;
    };

    // Reserve slots for (batch_size - 1) minimal sequences.
    auto minSlots = getNumSlots(1);
    for (int pg = 0; pg < numPg; ++pg)
    {
        remainingSlots[static_cast<size_t>(pg)] -= minSlots[static_cast<size_t>(pg)] * (batchSize - 1);
        assert(remainingSlots[static_cast<size_t>(pg)] >= 0);
    }

    auto isEnough = [&](int numBlocks) -> bool
    {
        auto needed = getNumSlots(numBlocks * tokPerBlock);
        for (int pg = 0; pg < numPg; ++pg)
        {
            if (needed[static_cast<size_t>(pg)] > remainingSlots[static_cast<size_t>(pg)])
                return false;
        }
        return true;
    };

    assert(isEnough(1));
    int lb = 1;
    int ub = divUp(tokenNumUpperBound, tokPerBlock);
    if (isEnough(ub))
        return tokenNumUpperBound;
    while (lb < ub - 1)
    {
        int mid = (lb + ub) / 2;
        if (isEnough(mid))
            lb = mid;
        else
            ub = mid;
    }
    return std::min(lb * tokPerBlock, tokenNumUpperBound);
}

void KvCacheManager::_adjustLevel(CacheLevel level, size_t quota)
{
    auto const& ratioList = _getTargetRatioList(level);
    std::vector<std::vector<std::shared_ptr<Page>>> const* persistent = nullptr;
    std::vector<std::vector<std::shared_ptr<Page>>> persistentPages;
    if (mStorage->isLastLevel(level))
    {
        persistentPages = _gatherPersistentPages();
        persistent = &persistentPages;
    }
    mStorage->adjustCacheLevel(level, quota, ratioList, persistent);
}

std::vector<std::vector<std::shared_ptr<Page>>> KvCacheManager::_gatherPersistentPages() const
{
    CacheLevel lastLevel = static_cast<CacheLevel>(mStorage->numCacheLevels() - 1);
    int numPg = mStorage->numPoolGroups();
    std::vector<std::vector<std::shared_ptr<Page>>> result(static_cast<size_t>(numPg));

    for (KvCache* kvc : mLivingKvCaches)
    {
        assert(kvc->status() == KvCache::Status::SUSPENDED);
        for (auto const& sb : kvc->blocks())
        {
            for (auto const& beamPages : sb.pages)
            {
                for (size_t lc = 0; lc < beamPages.size(); ++lc)
                {
                    // Mirrors Python: holder must be _PageHolder type (suspended state).
                    if (blockPageIsNull(beamPages[lc]))
                        continue;
                    assert(std::holds_alternative<std::shared_ptr<PageHolder>>(beamPages[lc])
                        && "Non-null holder must be PageHolder in suspended state");
                    auto const& pg = blockPageGetPage(beamPages[lc]);
                    if (!pg)
                        continue;
                    // Mirrors Python assertions for invariant checking.
                    assert(pg->status() == PageStatus::HELD && "Page in suspended KvCache must be HELD");
                    assert((pg->scheduledForEviction() == (pg->cacheLevel != lastLevel))
                        && "Eviction scheduling invariant violated");
                    if (pg->scheduledForEviction())
                        continue;
                    PoolGroupIndex pgIdx = mStorage->getPoolGroupIndex(static_cast<LifeCycleId>(lc));
                    result[static_cast<size_t>(pgIdx)].push_back(pg);
                }
            }
        }
    }
    return result;
}

// ---- KvCache registry -----------------------------------------------------

void KvCacheManager::registerKvCache(KvCache* kvc)
{
    mLivingKvCaches.insert(kvc);
    ++mNumCreatedKvCaches;
}

void KvCacheManager::unregisterKvCache(KvCache* kvc)
{
    mLivingKvCaches.erase(kvc);
}

void KvCacheManager::tryUpdateTargetRatios()
{
    if (mNumSampledKvCaches - mLastUpdateNumSampledKvCaches < 100)
        return;
    mLastUpdateNumSampledKvCaches = mNumSampledKvCaches;

    int tokensPerBlock = mConfig.tokensPerBlock;
    int avgReusedLength = static_cast<int>(std::round(mAvgReusedLength.value()));
    int avgCapacity = static_cast<int>(std::round(std::sqrt(mAvgSqrCapacity.value())));
    int avgHistoryLength = static_cast<int>(std::round(std::sqrt(mAvgSqrHistoryLength.value())));
    if (avgCapacity > 0)
        mTargetRatioListGpu
            = mStorage->constrainRatio(mStorage->ratioFromLength(tokensPerBlock, avgHistoryLength, avgCapacity));
    if (avgReusedLength > 0)
        mTargetRatioListOther = mStorage->ratioFromLength(tokensPerBlock, avgReusedLength, avgReusedLength);
}

std::vector<float> KvCacheManager::_currentGpuRatio() const
{
    return mStorage->getRatioList(kGpuLevel);
}

std::vector<float> KvCacheManager::_currentOtherRatios() const
{
    int numLevels = mStorage->numCacheLevels();
    if (numLevels == 1)
        return _currentGpuRatio();
    int numPg = mStorage->numPoolGroups();
    std::vector<float> result(static_cast<size_t>(numPg), 0.f);
    for (int lvl = 1; lvl < numLevels; ++lvl)
    {
        auto ratios = mStorage->getRatioList(static_cast<CacheLevel>(lvl));
        for (int j = 0; j < numPg; ++j)
            result[j] += ratios[j];
    }
    float denom = static_cast<float>(numLevels - 1);
    for (auto& r : result)
        r /= denom;
    return result;
}

// ---- needAdjustment / adjust -----------------------------------------------

std::vector<float> const& KvCacheManager::_getTargetRatioList(CacheLevel level) const
{
    return (level == kGpuLevel) ? mTargetRatioListGpu : mTargetRatioListOther;
}

bool KvCacheManager::_needAdjustment(CacheLevel level) const
{
    auto const& target = _getTargetRatioList(level);
    auto current = (level == kGpuLevel) ? _currentGpuRatio() : _currentOtherRatios();
    constexpr float kThreshold = 1.25f;
    for (size_t i = 0; i < target.size() && i < current.size(); ++i)
    {
        assert(current[i] > 0.f && target[i] > 0.f && "ratios must not be zero");
        float ratio = target[i] / current[i];
        if (ratio < 1.f / kThreshold || ratio > kThreshold)
            return true;
    }
    return false;
}

bool KvCacheManager::needAdjustment() const
{
    if (mNumSampledKvCaches < 2000)
        return false;
    double now = nowSeconds();
    if (now - mLastAdjustmentTime < 120.0)
        return false;
    CacheLevel lastLevel = static_cast<CacheLevel>(mStorage->numCacheLevels() - 1);
    return _needAdjustment(kGpuLevel) || _needAdjustment(lastLevel);
}

void KvCacheManager::adjust()
{
    for (KvCache* kvc : mLivingKvCaches)
        assert(kvc->status() == KvCache::Status::SUSPENDED);

    int numLevels = mStorage->numCacheLevels();
    for (int lvl = 0; lvl < numLevels; ++lvl)
    {
        CacheLevel level = static_cast<CacheLevel>(lvl);
        if (_needAdjustment(level))
            _adjustLevel(level, getQuota(level));
    }
    mLastAdjustmentTime = nowSeconds();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
