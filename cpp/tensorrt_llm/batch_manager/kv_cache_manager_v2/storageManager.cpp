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
#include "kv_cache_manager_v2/coldPageCopy.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/copyEngine.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/hostMem.h"
#include "kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/common/logger.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// LifeCyclePoolGroupMapping
// ---------------------------------------------------------------------------

LifeCyclePoolGroupMapping::LifeCyclePoolGroupMapping(TypedVec<LifeCycleId, PoolGroupIndex> forward)
    : mForward(std::move(forward))
{
    size_t numPoolGroups = 0;
    for (PoolGroupIndex poolGroup : mForward)
    {
        TLLM_CHECK_WITH_INFO(poolGroup >= PoolGroupIndex{0}, "Pool group index must be non-negative");
        numPoolGroups = std::max(numPoolGroups, static_cast<size_t>(poolGroup.value()) + 1);
    }
    TLLM_CHECK_WITH_INFO(numPoolGroups <= static_cast<size_t>(std::numeric_limits<int>::max()), "Too many pool groups");
    TLLM_CHECK_WITH_INFO(
        mForward.stdSize() <= static_cast<size_t>(std::numeric_limits<int>::max()), "Too many lifecycles");

    mPoolGroupOffsets.assign(numPoolGroups + 1, 0);
    for (PoolGroupIndex poolGroup : mForward)
    {
        ++mPoolGroupOffsets.at(static_cast<size_t>(poolGroup.value()) + 1);
    }
    for (size_t poolGroup = 0; poolGroup < numPoolGroups; ++poolGroup)
    {
        TLLM_CHECK_WITH_INFO(mPoolGroupOffsets[poolGroup + 1] > 0, "Pool group indices must be canonical");
    }
    std::partial_sum(mPoolGroupOffsets.begin(), mPoolGroupOffsets.end(), mPoolGroupOffsets.begin());

    mInverse.resize(mForward.stdSize());
    auto next = mPoolGroupOffsets;
    for (LifeCycleId lifeCycle{0}; lifeCycle < mForward.size(); ++lifeCycle)
    {
        size_t const poolGroup = static_cast<size_t>(mForward[lifeCycle].value());
        mInverse[next[poolGroup]++] = lifeCycle;
    }
}

PoolGroupIndex LifeCyclePoolGroupMapping::poolGroup(LifeCycleId lifeCycle) const
{
    return mForward.at(lifeCycle);
}

Span<LifeCycleId const> LifeCyclePoolGroupMapping::lifeCycles(PoolGroupIndex poolGroup) const
{
    TLLM_CHECK_WITH_INFO(poolGroup >= PoolGroupIndex{0}, "Pool group index must be non-negative");
    size_t const index = static_cast<size_t>(poolGroup.value());
    size_t const begin = mPoolGroupOffsets.at(index);
    size_t const end = mPoolGroupOffsets.at(index + 1);
    return Span<LifeCycleId const>{mInverse.data() + begin, static_cast<int>(end - begin)};
}

PoolGroupIndex LifeCyclePoolGroupMapping::numPoolGroups() const noexcept
{
    return PoolGroupIndex{static_cast<int>(mPoolGroupOffsets.empty() ? 0 : mPoolGroupOffsets.size() - 1)};
}

// ---------------------------------------------------------------------------
// CacheLevelManager
// ---------------------------------------------------------------------------

CacheLevelManager::CacheLevelManager(TypedVec<LifeCycleId, PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cl,
    CacheTierConfig const& tierConfig, TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, PooledPhysMemAllocator* gpuPhysMemAllocator)
    : cacheLevel(cl)
    , cacheTier(CacheTier(tierConfig.index()))
    , controller(lifeCycleGrouping, cl)
{
    TLLM_CHECK((cacheTier == CacheTier::GPU_MEM) == (gpuPhysMemAllocator != nullptr));
    storage = createCacheLevelStorage(tierConfig, slotDescList, slotCountList, gpuPhysMemAllocator);
}

size_t CacheLevelManager::cacheTierGranularity(CacheTier tier, size_t quota)
{
    switch (tier)
    {
    case CacheTier::GPU_MEM:
    {
        constexpr size_t kPageSize = 2ULL << 20;
        size_t const ratio = quota / (kPageSize * 512);
        int const exponent = ratio == 0 ? 0 : std::min(4, static_cast<int>(std::log2(ratio)));
        return kPageSize << exponent;
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

bool isGpuAccessibleMemory(CacheTier tier) noexcept
{
    return tier == CacheTier::GPU_MEM || tier == CacheTier::HOST_MEM;
}

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

void sortFallenPagesByPriority(TypedVec<LifeCycleId, std::deque<SharedPtr<Page>>>& fallenPages)
{
    for (auto& pages : fallenPages)
    {
        std::stable_sort(
            pages.begin(), pages.end(), [](auto const& lhs, auto const& rhs) { return lhs->priority < rhs->priority; });
    }
}

} // namespace

template <typename Submit>
bool StorageManager::submitColdPageCodec(
    PageIndexLocation location, PageIndexPair const* pageIndices, size_t numPages, CUstream stream, Submit&& submit)
{
    if (location == PageIndexLocation::kHost || numPages == 0)
    {
        return submit(pageIndices, numPages, stream);
    }
    if (location != PageIndexLocation::kDevice || pageIndices == nullptr || !mIndexStagingManager)
    {
        return false;
    }
    TLLM_CHECK_WITH_INFO(
        numPages <= std::numeric_limits<size_t>::max() / sizeof(PageIndexPair), "Codec index array is too large");

    size_t offset = 0;
    while (offset < numPages)
    {
        size_t const remainingBytes = (numPages - offset) * sizeof(PageIndexPair);
        auto device = mIndexStagingManager->acquire(sizeof(PageIndexPair),
            std::min(remainingBytes, kMaxIndexBatchBytes), sizeof(PageIndexPair), alignof(PageIndexPair), stream);
        size_t const chunkPages = std::min(numPages - offset, device.size() / sizeof(PageIndexPair));
        TLLM_CHECK_DEBUG(chunkPages > 0);
        // The index vector is ephemeral. The helper captures it before returning while the device update remains
        // asynchronous.
        detail::copyPageIndicesToDevice(
            static_cast<CUdeviceptr>(device.address()), pageIndices + offset, chunkPages, stream);

        if (!submit(reinterpret_cast<PageIndexPair const*>(device.address()), chunkPages, stream))
        {
            return false;
        }
        offset += chunkPages;
    }
    return true;
}

// ---------------------------------------------------------------------------
// StorageManager
// ---------------------------------------------------------------------------

StorageManager::StorageManager(LifeCycleRegistry const& lifeCycles, StorageConfig const& config, int tokensPerBlock,
    std::unique_ptr<IKvCacheColdPageCodec> coldPageCodec, std::optional<SwaScratchReuseConfig> swaScratchReuse,
    std::optional<BatchDesc> const& typicalBatch, std::vector<BatchDesc> const& constraints,
    std::optional<std::vector<float>> const& initialPoolRatio, std::shared_ptr<EventSink> eventSink,
    float maxUtilForResume)
    : mLifeCycles(lifeCycles)
    , mEventSink(std::move(eventSink))
    , mHotPoolGroupMapping(config.lifeCycleGrouping())
    , mStorageConfig(config)
    , mSwaScratchReuse(std::move(swaScratchReuse))
    , mColdPageCodec(coldPageCodec ? std::move(coldPageCodec) : createDefaultKvCacheColdPageCodec())
{
    IKvCacheColdPageCodec& codec = *mColdPageCodec;
    mLayerToLifeCycleIds = config.layerToLifeCycleIds();
    mSlotToPageIndices = computeSlotToPageIndices(config);
    mBufferAttr = config.bufferAttributes();
    mSlotDescLists.resize(config.cacheTiers.size(), config.slotDescList);

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

    auto const& hotLifeCycleGrouping = lifeCycleGrouping(kHotLevel);
    auto const& hotSlotDescList = slotDescList(kHotLevel);
    TLLM_CHECK_DEBUG(std::all_of(hotLifeCycleGrouping.begin(), hotLifeCycleGrouping.end(),
        [this](PoolGroupIndex pg) { return pg < numPoolGroups(); }));
    TLLM_CHECK_DEBUG(numPoolGroups()
        == PoolGroupIndex{static_cast<int>(
            std::set<PoolGroupIndex>(hotLifeCycleGrouping.begin(), hotLifeCycleGrouping.end()).size())});

    // Build one CacheLevelManager per tier.
    TLLM_CHECK_DEBUG(!config.cacheTiers.empty());
    bool const needsPageStaging = config.cacheTiers.size() > CacheLevel{1}
        && std::any_of(config.cacheTiers.begin() + 1, config.cacheTiers.end(),
            [](CacheTierConfig const& tierConfig) { return !isGpuAccessibleMemory(cacheTierOf(tierConfig)); });
    TLLM_CHECK_DEBUG_WITH_INFO(
        std::holds_alternative<GpuCacheTierConfig>(config.cacheTiers[kHotLevel]), "First cache tier must be GPU");

    // Compute slot size lists for all pool groups.
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> slotSizeLists;
    slotSizeLists.reserve(hotSlotDescList.size());
    for (auto const& sd : hotSlotDescList)
    {
        slotSizeLists.push_back(sd.slotSizeList());
    }

    size_t const gpuQuota = cacheTierQuota(config.cacheTiers[kHotLevel]);
    mGpuPhysMemAllocator = std::make_unique<PooledPhysMemAllocator>(
        CacheLevelManager::cacheTierGranularity(CacheTier::GPU_MEM, gpuQuota));
    size_t const gpuGranularity = mGpuPhysMemAllocator->physMemSize();

    // Constraints are hot-level feasibility floors. A pool group whose life cycles can grow is scaled by
    // 1/maxUtilForResume because KvCache::resume rejects such a pool group above that utilization; a pool group
    // with no growth to reserve for uses its full capacity. Other levels need only the structural one-slot floor.
    mMinSlots
        = computePoolGroupMinSlotsFromConstraints(constraints, tokensPerBlock, mSwaScratchReuse, maxUtilForResume);

    // Derive hot-tier lifecycle byte weights. Cold initialization preserves the slot-count proportions implied by
    // those weights while accounting for the cold representation's page sizes.
    TypedVec<LifeCycleId, float> lifeCycleRatio;
    if (initialPoolRatio.has_value())
    {
        if (initialPoolRatio->size() != toSizeT(numLifeCycles()))
        {
            throw std::invalid_argument("initial_pool_ratio length must match number of layer groups");
        }
        if (std::any_of(initialPoolRatio->begin(), initialPoolRatio->end(), [](float ratio) { return ratio <= 0.0F; }))
        {
            throw std::invalid_argument("initial_pool_ratio values must be positive");
        }

        constexpr double kExpectedRatioSum = 1.0;
        constexpr double kRatioSumTolerance = 1e-6;
        double const ratioSum = std::accumulate(initialPoolRatio->begin(), initialPoolRatio->end(), 0.0);
        if (!std::isfinite(ratioSum) || std::abs(ratioSum - kExpectedRatioSum) > kRatioSumTolerance)
        {
            throw std::invalid_argument("initial_pool_ratio values must sum to 1.0");
        }
        lifeCycleRatio = TypedVec<LifeCycleId, float>(*initialPoolRatio);
    }
    else if (typicalBatch.has_value())
    {
        lifeCycleRatio = ratioFromBatch(*typicalBatch, tokensPerBlock, mSwaScratchReuse, gpuGranularity);
    }
    else if (!constraints.empty())
    {
        auto lifeCycleSlots
            = computeSlotsFromConstraints(constraints, tokensPerBlock, mSwaScratchReuse, maxUtilForResume);
        lifeCycleRatio = normalizeToRatio(slotsToBytes(lifeCycleSlots, gpuGranularity));
    }
    else
    {
        // Fallback: average history length 2048.
        BatchDesc fallback;
        fallback.kvCaches.push_back(KVCacheDesc{2049, 2048});
        lifeCycleRatio = ratioFromBatch(fallback, tokensPerBlock, mSwaScratchReuse, gpuGranularity);
    }

    auto const hotRatio = toPoolGroupRatio(kHotLevel, lifeCycleRatio);

    mLevels.reserve(config.cacheTiers.size());

    auto gpuSlotCounts = computeSlotCountForLevel(config.cacheTiers[kHotLevel], slotSizeLists, hotRatio, mMinSlots);
    if (initialPoolRatio.has_value())
    {
        auto const ssmLifeCycle = mLifeCycles.ssmLifeCycleId();
        if (ssmLifeCycle.has_value())
        {
            auto const poolGroup = getPoolGroupIndex(*ssmLifeCycle);
            if (!poolGroupNeedsHeadroomForGrowth(poolGroup))
            {
                size_t const allocatedQuota = std::max(
                    minQuotaForLevel(slotSizeLists, gpuGranularity, mMinSlots), roundUp(gpuQuota, gpuGranularity));
                size_t const requestedGrains = static_cast<size_t>(
                    std::nearbyint(static_cast<double>(allocatedQuota / gpuGranularity) * hotRatio[poolGroup]));
                size_t const floorGrains
                    = CacheLevelStorage::grainsForSlots(mMinSlots[poolGroup], slotSizeLists[poolGroup], gpuGranularity);
                if (requestedGrains <= floorGrains)
                {
                    gpuSlotCounts[poolGroup] = mMinSlots[poolGroup];
                }
            }
        }
    }
    mLevels.emplace_back(lifeCycleGrouping(kHotLevel), kHotLevel, config.cacheTiers[kHotLevel], slotDescList(kHotLevel),
        gpuSlotCounts, mGpuPhysMemAllocator.get());

    auto& gpuStorage = *mLevels[kHotLevel].storage;
    TypedVec<PoolGroupIndex, PoolGroupDesc> gpuDescs;
    gpuDescs.reserve(numPoolGroups(kHotLevel));
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(kHotLevel); ++pgIdx)
    {
        TypedVec<PoolIndex, PoolDesc> pools;
        auto const poolSizes = slotSize(kHotLevel, pgIdx);
        pools.reserve(poolSizes.size());
        for (PoolIndex poolIdx{0}; poolIdx < poolSizes.size(); ++poolIdx)
        {
            pools.push_back(
                PoolDesc{poolIdx, gpuStorage.getBaseAddress(pgIdx, poolIdx, SlotId{0}), poolSizes.at(poolIdx)});
        }
        gpuDescs.push_back(
            PoolGroupDesc{pgIdx, gpuStorage.numSlots(pgIdx), slotDescList(kHotLevel).at(pgIdx), std::move(pools)});
    }
    TLLM_CHECK_WITH_INFO(
        codec.configure(gpuDescs.raw().data(), gpuDescs.size()), "Cold-page codec configuration failed");

    TypedVec<LifeCycleId, size_t> coldPageBytesByLifeCycle(numLifeCycles());
    size_t maxColdPageBytes = 0;
    mBatchingLayerGroupIds.resize(numLifeCycles());
    mPageIndexLocations.resize(numLifeCycles());
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        size_t const coldPageBytes = codec.queryColdPageBytes(lifeCycle);
        TLLM_CHECK_WITH_INFO(coldPageBytes > 0, "Cold-page codec returned an invalid page size");
        coldPageBytesByLifeCycle[lifeCycle] = coldPageBytes;
        maxColdPageBytes = std::max(maxColdPageBytes, coldPageBytes);

        LayerGroupId const batchingLayerGroupId = codec.getBatchingLayerGroupId(lifeCycle);
        TLLM_CHECK_WITH_INFO(batchingLayerGroupId.value() >= 0 && batchingLayerGroupId < numLifeCycles()
                && batchingLayerGroupId <= lifeCycle,
            "Cold-page codec returned an invalid batching layer-group ID");
        TLLM_CHECK_WITH_INFO(
            getPoolGroupIndex(kHotLevel, batchingLayerGroupId) == getPoolGroupIndex(kHotLevel, lifeCycle),
            "Cold-page codec batching class spans hot pool groups");
        mBatchingLayerGroupIds[lifeCycle] = batchingLayerGroupId;

        PageIndexLocation const pageIndexLocation = codec.queryPageIndexLocation(lifeCycle);
        TLLM_CHECK_WITH_INFO(
            pageIndexLocation == PageIndexLocation::kHost || pageIndexLocation == PageIndexLocation::kDevice,
            "Cold-page codec returned an invalid page-index location");
        mPageIndexLocations[lifeCycle] = pageIndexLocation;
    }

    size_t pageStagingBytes = 0;
    if (needsPageStaging)
    {
        TLLM_CHECK_WITH_INFO(maxColdPageBytes <= std::numeric_limits<size_t>::max() / kPageStagingDepth,
            "Cold-page staging size overflow");
        pageStagingBytes = std::max(kDefaultPageStagingBytes, kPageStagingDepth * maxColdPageBytes);
    }

    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        LayerGroupId const batchingLayerGroupId = mBatchingLayerGroupIds[lifeCycle];
        TLLM_CHECK_WITH_INFO(mBatchingLayerGroupIds[batchingLayerGroupId] == batchingLayerGroupId
                && coldPageBytesByLifeCycle[batchingLayerGroupId] == coldPageBytesByLifeCycle[lifeCycle]
                && mPageIndexLocations[batchingLayerGroupId] == mPageIndexLocations[lifeCycle],
            "Cold-page codec batching class is inconsistent");
    }

    bool const needsIndexStaging = std::any_of(mPageIndexLocations.begin(), mPageIndexLocations.end(),
        [](PageIndexLocation location) { return location == PageIndexLocation::kDevice; });
    if (needsIndexStaging)
    {
        mIndexStagingManager = std::make_unique<StagingBufferManager>(kIndexStagingBytes, StagingBufferMemory::kDevice);
    }

    TypedVec<LifeCycleId, PoolGroupIndex> coldGrouping(numLifeCycles());
    TypedVec<PoolGroupIndex, SlotDesc> coldSlotDescList;
    std::map<size_t, PoolGroupIndex> coldGroupByPageBytes;
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        size_t const coldPageBytes = coldPageBytesByLifeCycle[lifeCycle];
        auto [it, inserted] = coldGroupByPageBytes.emplace(
            coldPageBytes, PoolGroupIndex{static_cast<int>(coldSlotDescList.size().value())});
        PoolGroupIndex const coldPgIdx = it->second;
        if (inserted)
        {
            coldSlotDescList.push_back(SlotDesc{});
        }
        coldGrouping[lifeCycle] = coldPgIdx;

        SlotDescVariant variant;
        variant.lifeCycleId = lifeCycle;
        variant.coalescedBuffers.push_back(
            CoalescedBuffer{coldPageBytes, std::vector<BufferId>{BufferId{-1, "__cold_page__"}}});
        coldSlotDescList[coldPgIdx].variants.push_back(std::move(variant));
    }

    TypedVec<PoolGroupIndex, SlotCount> coldMinSlots(coldSlotDescList.size(), 1);

    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> coldSlotSizeLists;
    coldSlotSizeLists.reserve(coldSlotDescList.size());
    for (auto const& desc : coldSlotDescList)
    {
        auto sizes = desc.slotSizeList();
        TLLM_CHECK_WITH_INFO(sizes.size() == PoolIndex{1}, "Cold pool groups must contain exactly one pool");
        coldSlotSizeLists.push_back(std::move(sizes));
    }

    mColdPoolGroupMapping = LifeCyclePoolGroupMapping(std::move(coldGrouping));

    for (CacheLevel level{1}; level < config.cacheTiers.size(); ++level)
    {
        mSlotDescLists[level] = coldSlotDescList;
        auto const coldRatio = projectPoolGroupRatio(kHotLevel, level, lifeCycleRatio);
        auto slotCounts
            = computeSlotCountForLevel(config.cacheTiers[level], coldSlotSizeLists, coldRatio, coldMinSlots);
        auto* gpuPhysMemAllocator
            = cacheTierOf(config.cacheTiers[level]) == CacheTier::GPU_MEM ? mGpuPhysMemAllocator.get() : nullptr;
        mLevels.emplace_back(lifeCycleGrouping(level), level, config.cacheTiers[level], slotDescList(level), slotCounts,
            gpuPhysMemAllocator);
    }

    for (CacheLevel level{0}; level < mLevels.size(); ++level)
    {
        TLLM_CHECK_DEBUG(numPoolGroups(level) == mLevels[level].storage->numPoolGroups());
        TLLM_CHECK_DEBUG(numPoolGroups(level) == poolGroupMapping(level).numPoolGroups());
    }

    if (needsPageStaging)
    {
        mPageStagingManager
            = std::make_unique<StagingBufferManager>(pageStagingBytes, StagingBufferMemory::kPinnedHost);
    }

    // cuMemcpyBatchAsync cannot copy across adjacent HostMem registrations in one batch entry. The default codec needs
    // the owning HostMem objects to split copies at registration boundaries on linux kernels that require chunked
    // pinning.
    if (detail::needsHostMemRegistration(codec))
    {
        for (CacheLevel level{0}; level < mLevels.size(); ++level)
        {
            if (cacheTier(level) != CacheTier::HOST_MEM)
            {
                continue;
            }
            for (PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numPoolGroups(level); ++poolGroupIndex)
            {
                auto const& hostPoolGroup = static_cast<HostPoolGroup const&>(poolGroup(level, poolGroupIndex));
                for (PoolIndex poolIndex{0}; poolIndex < numPools(level, poolGroupIndex); ++poolIndex)
                {
                    detail::registerHostMem(codec, hostPoolGroup.hostMem(poolIndex));
                }
            }
        }
        if (mPageStagingManager)
        {
            detail::registerHostMem(codec, mPageStagingManager->hostMem());
        }
    }
    mCopyEngine = std::make_unique<CopyEngine>(mPageStagingManager.get());
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
    mGpuPhysMemAllocator.reset();

    mIndexStagingManager.reset();
    mCopyEngine.reset();
    mPageStagingManager.reset();
}

// ---------------------------------------------------------------------------
// newSlots
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, std::vector<Slot>> StorageManager::newSlots(CacheLevel level,
    TypedVec<LifeCycleId, SlotCount> const& numSlotsPerLc, MigrationRecorder const& migrationRecorder,
    DropRecorder const& dropRecorder)
{
    auto const& grouping = lifeCycleGrouping(level);
    TLLM_CHECK_DEBUG(numSlotsPerLc.size() == numLifeCycles());
    auto& storage = *mLevels.at(level).storage;

    // Aggregate by pool group.
    TypedVec<PoolGroupIndex, SlotCount> pgNumSlots(numPoolGroups(level), 0);
    for (LifeCycleId lcId{0}; lcId < numSlotsPerLc.size(); ++lcId)
    {
        SlotCount const numSlots = numSlotsPerLc[lcId];
        if (numSlots < 0)
        {
            throw LogicError("StorageManager::newSlots: slot count must be non-negative");
        }
        pgNumSlots[grouping[lcId]] += numSlots;
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
        prepareFreeSlots(level, pgNumSlots, migrationRecorder, dropRecorder);
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
            PoolGroupIndex pg = grouping[lcId];
            ret[lcId] = storage.allocateMultiple(pg, numSlotsPerLc[lcId]);
        }
    }
    catch (...)
    {
        for (LifeCycleId lcId{0}; lcId < ret.size(); ++lcId)
        {
            PoolGroupIndex pg = grouping[lcId];
            for (auto& s : ret[lcId])
                storage.release(pg, std::move(s));
        }
        throw;
    }
    return ret;
}

TypedVec<LifeCycleId, std::vector<Slot>> StorageManager::newGpuSlots(
    TypedVec<LifeCycleId, SlotCount> const& numSlotsPerLc, MigrationRecorder const& migrationRecorder,
    DropRecorder const& dropRecorder)
{
    return newSlots(kHotLevel, numSlotsPerLc, migrationRecorder, dropRecorder);
}

std::vector<Slot> StorageManager::newSlotsForPoolGroup(CacheLevel level, PoolGroupIndex pgIdx, SlotCount numSlots,
    MigrationRecorder const& migrationRecorder, DropRecorder const& dropRecorder)
{
    if (numSlots < 0)
    {
        throw LogicError("StorageManager::newSlotsForPoolGroup: numSlots must be non-negative");
    }
    auto& storage = *mLevels.at(level).storage;
    if (numSlots > storage.numFreeSlots(pgIdx))
    {
        TypedVec<PoolGroupIndex, SlotCount> requirements(numPoolGroups(level), 0);
        requirements.at(pgIdx) = numSlots;
        prepareFreeSlots(level, requirements, migrationRecorder, dropRecorder);
    }
    TLLM_CHECK_DEBUG(numSlots <= storage.numFreeSlots(pgIdx));
    return storage.allocateMultiple(pgIdx, numSlots);
}

Address StorageManager::slotAddress(CacheLevel level, PoolGroupIndex pgIdx, SlotId slotId, PoolIndex poolIdx) const
{
    return mLevels.at(level).storage->slotAddress(pgIdx, slotId).at(poolIdx);
}

void StorageManager::submitMigrationBatch(CacheLevel dstLevel, CacheLevel srcLevel, LayerGroupId batchingLayerGroupId,
    PageIndexPair const* pageIndices, size_t numPages, CUstream stream)
{
    TLLM_CHECK_DEBUG(pageIndices != nullptr && numPages > 0);
    PoolGroupIndex const srcPgIdx = getPoolGroupIndex(srcLevel, batchingLayerGroupId);
    PoolGroupIndex const dstPgIdx = getPoolGroupIndex(dstLevel, batchingLayerGroupId);
    CacheTier const srcTier = cacheTier(srcLevel);
    CacheTier const dstTier = cacheTier(dstLevel);
    bool const srcIsHot = srcLevel == kHotLevel;
    bool const dstIsHot = dstLevel == kHotLevel;
    auto& srcPoolGroup = poolGroup(srcLevel, srcPgIdx);
    auto& dstPoolGroup = poolGroup(dstLevel, dstPgIdx);

    auto toSlotId = [](int32_t pageIndex)
    {
        TLLM_CHECK_DEBUG(pageIndex >= 0);
        return SlotId{pageIndex};
    };

    if (srcIsHot == dstIsHot)
    {
        auto const srcSizes = slotSize(srcLevel, srcPgIdx);
        TLLM_CHECK_DEBUG(srcSizes == slotSize(dstLevel, dstPgIdx));
        for (PoolIndex poolIdx{0}; poolIdx < srcSizes.size(); ++poolIdx)
        {
            std::vector<CopyTask> tasks;
            tasks.reserve(numPages);
            for (size_t index = 0; index < numPages; ++index)
            {
                tasks.push_back({dstPoolGroup.slotAddress(toSlotId(pageIndices[index].dst)).at(poolIdx),
                    srcPoolGroup.slotAddress(toSlotId(pageIndices[index].src)).at(poolIdx)});
            }
            mCopyEngine->transfer(dstTier, srcTier, srcSizes.at(poolIdx), tasks, stream);
        }
        return;
    }

    CacheLevel const coldLevel = srcIsHot ? dstLevel : srcLevel;
    PoolGroupIndex const coldPgIdx = srcIsHot ? dstPgIdx : srcPgIdx;
    CacheTier const coldTier = srcIsHot ? dstTier : srcTier;
    size_t const coldPageBytes = slotSize(coldLevel, coldPgIdx).at(PoolIndex{0});
    PageIndexLocation const pageIndexLocation = mPageIndexLocations.at(batchingLayerGroupId);
    auto encodeBatch = [this, batchingLayerGroupId, pageIndexLocation](
                           void* dstBasePtr, PageIndexPair const* indices, size_t count, CUstream batchStream)
    {
        return submitColdPageCodec(pageIndexLocation, indices, count, batchStream,
            [this, batchingLayerGroupId, dstBasePtr](
                PageIndexPair const* submittedIndices, size_t submittedPages, CUstream submittedStream)
            {
                return mColdPageCodec->encode(batchingLayerGroupId, dstBasePtr, submittedIndices, submittedPages,
                    reinterpret_cast<cudaStream_t>(submittedStream));
            });
    };
    auto decodeBatch = [this, batchingLayerGroupId, pageIndexLocation](
                           void const* srcBasePtr, PageIndexPair const* indices, size_t count, CUstream batchStream)
    {
        return submitColdPageCodec(pageIndexLocation, indices, count, batchStream,
            [this, batchingLayerGroupId, srcBasePtr](
                PageIndexPair const* submittedIndices, size_t submittedPages, CUstream submittedStream)
            {
                return mColdPageCodec->decode(batchingLayerGroupId, srcBasePtr, submittedIndices, submittedPages,
                    reinterpret_cast<cudaStream_t>(submittedStream));
            });
    };

    if (isGpuAccessibleMemory(coldTier))
    {
        MemAddress const coldBase = mLevels.at(coldLevel).storage->getBaseAddress(coldPgIdx, PoolIndex{0}, SlotId{0});
        bool const submitted = srcIsHot
            ? encodeBatch(reinterpret_cast<void*>(coldBase), pageIndices, numPages, stream)
            : decodeBatch(reinterpret_cast<void const*>(coldBase), pageIndices, numPages, stream);
        TLLM_CHECK_WITH_INFO(submitted, "Cold-page codec rejected a migration batch");
        return;
    }

    TLLM_CHECK_WITH_INFO(coldTier == CacheTier::DISK, "Unsupported cold cache tier");
    thread_local std::vector<PageIndexPair> stagingPageIndices;
    size_t remaining = numPages;
    size_t offset = 0;
    while (remaining > 0)
    {
        size_t const maxStagingBytes = remaining > std::numeric_limits<size_t>::max() / coldPageBytes
            ? std::numeric_limits<size_t>::max()
            : coldPageBytes * remaining;
        auto staging = mPageStagingManager->acquire(coldPageBytes, maxStagingBytes, coldPageBytes, 1, stream);
        size_t const batchSize = std::min(remaining, staging.size() / coldPageBytes);
        stagingPageIndices.resize(batchSize);

        if (srcIsHot)
        {
            for (size_t index = 0; index < batchSize; ++index)
            {
                stagingPageIndices[index] = PageIndexPair{static_cast<int32_t>(index), pageIndices[offset + index].src};
            }
            TLLM_CHECK_WITH_INFO(
                encodeBatch(reinterpret_cast<void*>(staging.address()), stagingPageIndices.data(), batchSize, stream),
                "Cold-page codec rejected a migration batch");

            std::vector<CopyTask> tasks;
            tasks.reserve(batchSize);
            for (size_t index = 0; index < batchSize; ++index)
            {
                tasks.push_back({dstPoolGroup.slotAddress(toSlotId(pageIndices[offset + index].dst)).at(PoolIndex{0}),
                    Address{std::in_place_type<MemAddress>, staging.address() + index * coldPageBytes}});
            }
            mCopyEngine->transfer(CacheTier::DISK, CacheTier::HOST_MEM, coldPageBytes, tasks, stream);
        }
        else
        {
            std::vector<CopyTask> tasks;
            tasks.reserve(batchSize);
            for (size_t index = 0; index < batchSize; ++index)
            {
                tasks.push_back({Address{std::in_place_type<MemAddress>, staging.address() + index * coldPageBytes},
                    srcPoolGroup.slotAddress(toSlotId(pageIndices[offset + index].src)).at(PoolIndex{0})});
            }
            mCopyEngine->transfer(CacheTier::HOST_MEM, CacheTier::DISK, coldPageBytes, tasks, stream);
            for (size_t index = 0; index < batchSize; ++index)
            {
                stagingPageIndices[index] = PageIndexPair{pageIndices[offset + index].dst, static_cast<int32_t>(index)};
            }
            TLLM_CHECK_WITH_INFO(decodeBatch(reinterpret_cast<void const*>(staging.address()),
                                     stagingPageIndices.data(), batchSize, stream),
                "Cold-page codec rejected a migration batch");
        }

        offset += batchSize;
        remaining -= batchSize;
    }

    constexpr size_t kMaxRetainedPageIndexPairs = (1u << 20u) / sizeof(PageIndexPair);
    if (stagingPageIndices.capacity() > kMaxRetainedPageIndexPairs)
    {
        std::vector<PageIndexPair>().swap(stagingPageIndices);
    }
}

void StorageManager::copySlotData(LifeCycleId lifeCycle, CacheLevel dstLevel, CacheLevel srcLevel, SlotId dstSlotId,
    SlotId srcSlotId, CUstream stream)
{
    LayerGroupId const batchingLayerGroupId = getMigrationBatchingLayerGroupId(dstLevel, srcLevel, lifeCycle);
    PageIndexPair const pageIndex{slotIdToPageIndexValue(dstSlotId), slotIdToPageIndexValue(srcSlotId)};
    submitMigrationBatch(dstLevel, srcLevel, batchingLayerGroupId, &pageIndex, 1, stream);
}

CacheTier StorageManager::cacheTier(CacheLevel level) const
{
    return mLevels.at(level).cacheTier;
}

void StorageManager::releaseSlot(LifeCycleId lc, CacheLevel level, Slot slot)
{
    PoolGroupIndex pg = getPoolGroupIndex(level, lc);
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

[[nodiscard]] auto StorageManager::makeEvictionRollbackGuard(
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const& evicted)
{
    std::vector<WeakPtr<Page>> tracker;
    tracker.reserve(std::accumulate(
        evicted.begin(), evicted.end(), size_t{0}, [](size_t acc, auto const& vec) { return acc + vec.size(); }));
    for (auto const& pages : evicted)
    {
        tracker.insert(tracker.end(), pages.begin(), pages.end());
    }
    return FuncGuard(
        [this, tracker = std::move(tracker)]()
        {
            for (auto iter = tracker.rbegin(); iter != tracker.rend(); ++iter)
            {
                auto page = iter->lock();
                if (page && page->hasValidSlot() && !page->scheduledForEviction() && isEvictable(*page))
                {
                    mLevels.at(page->cacheLevel).controller.scheduleForEviction(*page, /*evictFirst=*/true);
                }
            }
        });
}

// ---------------------------------------------------------------------------
// prepareFreeSlots
// ---------------------------------------------------------------------------

void StorageManager::prepareFreeSlots(CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& requirements,
    MigrationRecorder const& migrationRecorder, DropRecorder const& dropRecorder)
{
    TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>> goals(numCacheLevels());
    for (CacheLevel lvl{0}; lvl < goals.size(); ++lvl)
    {
        goals[lvl].resize(numPoolGroups(lvl), 0);
    }
    for (PoolGroupIndex pgIdx{0}; pgIdx < requirements.size(); ++pgIdx)
    {
        goals.at(level).at(pgIdx) = requirements.at(pgIdx);
    }

    PagesByLifeCycle fallenPages(numLifeCycles());
    _prepareFreeSlots(goals, level, fallenPages, migrationRecorder, dropRecorder);
}

void StorageManager::forceEvict(
    CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& minNumPages, DropRecorder const& dropRecorder)
{
    auto evicted = mLevels.at(level).controller.evict(minNumPages);
    auto rescheduleEvictedPagesOnFailure = makeEvictionRollbackGuard(evicted);

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
        if (dropRecorder)
        {
            for (auto const& pages : evicted)
            {
                if (!pages.empty())
                {
                    dropRecorder(pages, level);
                }
            }
        }
        rescheduleEvictedPagesOnFailure.cancel();
        return;
    }

    TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>> goals(numCacheLevels());
    for (CacheLevel lvl{0}; lvl < goals.size(); ++lvl)
    {
        goals[lvl].resize(numPoolGroups(lvl), 0);
    }
    CacheLevel nextLvl = level + 1;

    PagesByLifeCycle fallen(numLifeCycles());
    for (PoolGroupIndex pgIdx{0}; pgIdx < evicted.size(); ++pgIdx)
    {
        for (auto& sp : evicted.at(pgIdx))
        {
            LifeCycleId const lifeCycle = sp->lifeCycle;
            fallen.at(lifeCycle).push_back(std::move(sp));
        }
    }
    sortFallenPagesByPriority(fallen);
    _prepareFreeSlots(goals, nextLvl, fallen, MigrationRecorder{}, dropRecorder);
    rescheduleEvictedPagesOnFailure.cancel();
}

// ---------------------------------------------------------------------------
// _prepareFreeSlots (recursive)
// ---------------------------------------------------------------------------

void StorageManager::_prepareFreeSlots(TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>>& goals,
    CacheLevel lvlId, PagesByLifeCycle& fallenPages, MigrationRecorder const& migrationRecorder,
    DropRecorder const& dropRecorder)
{
    if (TLLM_UNLIKELY(gDebug))
    {
        TLLM_CHECK_WITH_INFO(goals.size() == numCacheLevels(), "goals.rows must equal numCacheLevels");
        for (CacheLevel level{0}; level < goals.size(); ++level)
        {
            TLLM_CHECK_DEBUG_WITH_INFO(
                goals[level].size() == numPoolGroups(level), "goals row must match the level's pool groups");
        }
        TLLM_CHECK_DEBUG_WITH_INFO(fallenPages.size() == numLifeCycles(), "fallenPages must be lifecycle-keyed");
        for (LifeCycleId lifeCycle{0}; lifeCycle < fallenPages.size(); ++lifeCycle)
        {
            TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(fallenPages[lifeCycle].begin(), fallenPages[lifeCycle].end(),
                                           [lifeCycle](auto const& page) { return page->lifeCycle == lifeCycle; }),
                "Fallen page stored under the wrong lifecycle");
        }
    }

    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(fallenPages.begin(), fallenPages.end(),
                                   [lvlId](auto const& pages) {
                                       return std::all_of(pages.begin(), pages.end(),
                                           [lvlId](auto const& p) { return p->cacheLevel < lvlId; });
                                   }),
        "Fallen pages must come from upper cache levels");

    auto& lvl = mLevels.at(lvlId);
    auto& storage = *lvl.storage;
    auto& ctrl = lvl.controller;
    bool const isLast = isLastLevel(lvlId);
    auto const& grouping = poolGroupMapping(lvlId);

    auto countPages = [&](PoolGroupIndex pgIdx, PagesByLifeCycle const& pagesByLifeCycle)
    {
        auto const lifeCycles = grouping.lifeCycles(pgIdx);
        return std::accumulate(lifeCycles.begin(), lifeCycles.end(), SlotCount{0},
            [&](SlotCount count, LifeCycleId lifeCycle)
            { return count + slotCountValueFromSize(pagesByLifeCycle.at(lifeCycle).size()); });
    };

    TypedVec<PoolGroupIndex, SlotCount> numToEvict(numPoolGroups(lvlId), 0);
    PagesByLifeCycle heldPages(numLifeCycles());

    for (PoolGroupIndex pgIdx{0}; pgIdx < numToEvict.size(); ++pgIdx)
    {
        SlotCount const goal = goals.at(lvlId).at(pgIdx);
        SlotCount const fallen = countPages(pgIdx, fallenPages);
        SlotCount const oldFree = storage.numFreeSlots(pgIdx);
        SlotCount const evictableCount = ctrl.numEvictablePages(pgIdx);
        SlotCount const required = goal + fallen;
        SlotCount const shortage = required > oldFree ? required - oldFree : 0;
        numToEvict.at(pgIdx) = std::min(shortage, evictableCount);

        SlotCount fallenHeld = 0;
        if (isLast)
        {
            for (LifeCycleId lifeCycle : grouping.lifeCycles(pgIdx))
            {
                auto& pages = fallenPages.at(lifeCycle);
                auto& held = heldPages.at(lifeCycle);
                PageQueue remaining;
                while (!pages.empty())
                {
                    auto page = std::move(pages.front());
                    pages.pop_front();
                    (page->status() == PageStatus::HELD ? held : remaining).push_back(std::move(page));
                }
                pages = std::move(remaining);
            }
            fallenHeld = countPages(pgIdx, heldPages);

            if (fallenHeld > oldFree + evictableCount)
            {
                throw OutOfPagesError(
                    "Too many held pages falling to last-level cache for group " + std::to_string(pgIdx.value()));
            }
        }

        if (oldFree + evictableCount < fallenHeld + goal)
        {
            throw OutOfPagesError("Impossible to meet free-slot goal " + std::to_string(goal) + " for group "
                + std::to_string(pgIdx.value()));
        }
    }

    auto evicted = ctrl.evict(numToEvict);
    auto rescheduleEvictedPagesOnFailure = makeEvictionRollbackGuard(evicted);
    TypedVec<LifeCycleId, std::vector<SharedPtr<Page>>> acceptedPages(numLifeCycles());

    auto acceptFallenPages = [&](PoolGroupIndex pgIdx, SlotCount count)
    {
        auto const lifeCycles = grouping.lifeCycles(pgIdx);
        auto backPriority = [&](LifeCycleId lifeCycle) -> std::optional<Priority>
        {
            auto const& fallen = fallenPages.at(lifeCycle);
            return fallen.empty() ? std::nullopt : std::optional<Priority>{fallen.back()->priority};
        };
        while (count > 0)
        {
            auto const bestLifeCycle = std::max_element(lifeCycles.begin(), lifeCycles.end(),
                [&](LifeCycleId lhs, LifeCycleId rhs) { return backPriority(lhs) < backPriority(rhs); });
            TLLM_CHECK_DEBUG(bestLifeCycle != lifeCycles.end() && backPriority(*bestLifeCycle).has_value());
            Priority const bestPriority = *backPriority(*bestLifeCycle);
            for (LifeCycleId lifeCycle : lifeCycles)
            {
                auto& fallen = fallenPages.at(lifeCycle);
                auto matchingBegin = std::lower_bound(fallen.begin(), fallen.end(), bestPriority,
                    [](auto const& page, Priority priority) { return page->priority < priority; });
                SlotCount const numAccepted
                    = std::min(count, slotCountValueFromSize(std::distance(matchingBegin, fallen.end())));
                matchingBegin = fallen.end() - numAccepted;
                auto& accepted = acceptedPages.at(lifeCycle);
                accepted.insert(
                    accepted.end(), std::make_move_iterator(matchingBegin), std::make_move_iterator(fallen.end()));
                fallen.erase(matchingBegin, fallen.end());
                count -= numAccepted;
                if (count == 0)
                {
                    break;
                }
            }
        }
    };

    for (PoolGroupIndex pgIdx{0}; pgIdx < evicted.size(); ++pgIdx)
    {
        auto& ev = evicted.at(pgIdx);
        SlotCount const oldFree = storage.numFreeSlots(pgIdx);
        SlotCount const numEvicted = slotCountValueFromSize(ev.size());
        SlotCount freeAfterEviction{};

        if (isLast)
        {
            TLLM_CHECK_DEBUG_WITH_INFO(
                std::all_of(ev.begin(), ev.end(), [](auto const& p) { return p->status() == PageStatus::DROPPABLE; }),
                "Evicted page at last level must be DROPPABLE");
            if (dropRecorder && !ev.empty())
            {
                dropRecorder(ev, lvlId);
            }
            ev.clear();

            freeAfterEviction = storage.numFreeSlots(pgIdx);
            TLLM_CHECK_DEBUG(freeAfterEviction >= numEvicted + oldFree);
            TLLM_CHECK_DEBUG_WITH_INFO(
                countPages(pgIdx, heldPages) <= freeAfterEviction, "held_pages count exceeds new free slot count");
        }
        else
        {
            freeAfterEviction = oldFree + numEvicted;
        }

        SlotCount const goal = goals.at(lvlId).at(pgIdx);
        SlotCount const availableAfterGoal = freeAfterEviction > goal ? freeAfterEviction - goal : 0;
        SlotCount const numHeld = isLast ? countPages(pgIdx, heldPages) : SlotCount{0};
        TLLM_CHECK_DEBUG(numHeld <= availableAfterGoal);
        SlotCount const numAccepted = std::min(availableAfterGoal - numHeld, countPages(pgIdx, fallenPages));
        acceptFallenPages(pgIdx, numAccepted);

        if (isLast)
        {
            for (LifeCycleId lifeCycle : grouping.lifeCycles(pgIdx))
            {
                auto& accepted = acceptedPages.at(lifeCycle);
                auto& held = heldPages.at(lifeCycle);
                accepted.insert(
                    accepted.end(), std::make_move_iterator(held.begin()), std::make_move_iterator(held.end()));
                held.clear();
                fallenPages.at(lifeCycle).clear();
            }
        }
        else
        {
            // Pages evicted from this level fall before rejected incoming pages so that the latter are accepted first.
            for (auto iter = ev.rbegin(); iter != ev.rend(); ++iter)
            {
                LifeCycleId const lifeCycle = (*iter)->lifeCycle;
                fallenPages.at(lifeCycle).push_front(std::move(*iter));
            }
            ev.clear();
        }
    }

    if (!isLast)
    {
        sortFallenPagesByPriority(fallenPages);
        _prepareFreeSlots(goals, lvlId + 1, fallenPages, migrationRecorder, dropRecorder);
    }

    TLLM_CHECK_DEBUG_WITH_INFO(
        std::all_of(fallenPages.begin(), fallenPages.end(), [](auto const& fp) { return fp.empty(); }),
        "All fallen pages must be consumed after level loop");

    for (PoolGroupIndex pgIdx{0}; pgIdx < grouping.numPoolGroups(); ++pgIdx)
    {
        std::map<MigrationBatchKey, std::vector<SharedPtr<Page>>> migrationBatches;
        for (LifeCycleId lifeCycle : grouping.lifeCycles(pgIdx))
        {
            auto& pages = acceptedPages.at(lifeCycle);
            for (auto& page : pages)
            {
                TLLM_CHECK_DEBUG(page->lifeCycle == lifeCycle);
                LayerGroupId const batchingLayerGroupId
                    = getMigrationBatchingLayerGroupId(lvlId, page->cacheLevel, lifeCycle);
                migrationBatches[{page->cacheLevel, batchingLayerGroupId}].push_back(std::move(page));
            }
            pages.clear();
        }
        for (auto& [batchKey, pages] : migrationBatches)
        {
            CacheLevel const srcLevel = batchKey.first;
            _batchedMigrate(lvlId, srcLevel, pages, /*updateSrc=*/true, migrationRecorder);
            for (auto const& page : pages)
            {
                if (!isLast || page->status() != PageStatus::HELD)
                {
                    lvl.controller.scheduleForEviction(*page);
                }
            }
        }
    }
    rescheduleEvictedPagesOnFailure.cancel();
}

// ---------------------------------------------------------------------------
// _batchedMigrate
// ---------------------------------------------------------------------------

LayerGroupId StorageManager::getMigrationBatchingLayerGroupId(
    CacheLevel dstLevel, CacheLevel srcLevel, LifeCycleId lifeCycle) const
{
    bool const srcIsHot = srcLevel == kHotLevel;
    bool const dstIsHot = dstLevel == kHotLevel;
    if (srcIsHot != dstIsHot)
    {
        return mBatchingLayerGroupIds.at(lifeCycle);
    }

    auto const& srcGrouping = poolGroupMapping(srcLevel);
    auto const lifeCycles = srcGrouping.lifeCycles(srcGrouping.poolGroup(lifeCycle));
    TLLM_CHECK_DEBUG(lifeCycles.size() > 0);
    LayerGroupId const batchingLayerGroupId = lifeCycles[0];
    TLLM_CHECK_DEBUG(getPoolGroupIndex(dstLevel, batchingLayerGroupId) == getPoolGroupIndex(dstLevel, lifeCycle));
    return batchingLayerGroupId;
}

std::optional<std::vector<Slot>> StorageManager::_batchedMigrate(CacheLevel dstLevel, CacheLevel srcLevel,
    std::vector<SharedPtr<Page>> const& srcPages, bool updateSrc, MigrationRecorder const& migrationRecorder,
    bool defrag)
{
    TLLM_CHECK_DEBUG(defrag || dstLevel != srcLevel);
    if (srcPages.empty())
    {
        return updateSrc ? std::nullopt : std::optional<std::vector<Slot>>{std::in_place};
    }

    SlotCount const numSlots = slotCountValueFromSize(srcPages.size());
    LifeCycleId const firstLifeCycle = srcPages.front()->lifeCycle;
    LayerGroupId const batchingLayerGroupId = getMigrationBatchingLayerGroupId(dstLevel, srcLevel, firstLifeCycle);
    PoolGroupIndex const srcPgIdx = getPoolGroupIndex(srcLevel, batchingLayerGroupId);
    PoolGroupIndex const dstPgIdx = getPoolGroupIndex(dstLevel, batchingLayerGroupId);
    TLLM_CHECK_DEBUG(std::all_of(srcPages.begin(), srcPages.end(),
        [this, srcLevel, dstLevel, srcPgIdx, dstPgIdx, batchingLayerGroupId](auto const& page)
        {
            return getPoolGroupIndex(srcLevel, page->lifeCycle) == srcPgIdx
                && getPoolGroupIndex(dstLevel, page->lifeCycle) == dstPgIdx
                && getMigrationBatchingLayerGroupId(dstLevel, srcLevel, page->lifeCycle) == batchingLayerGroupId;
        }));
    auto& srcPoolGroup = poolGroup(srcLevel, srcPgIdx);
    auto& dstPoolGroup = poolGroup(dstLevel, dstPgIdx);

    if (dstPoolGroup.numFreeSlots() < numSlots)
        throw OutOfPagesError("Not enough free slots for migration");

    auto dstSlots = dstPoolGroup.allocateMultiple(numSlots);
    // A15: allocated slot count must match the request.
    TLLM_CHECK_DEBUG_WITH_INFO(slotCountValueFromSize(dstSlots.size()) == numSlots, "dst_slots size mismatch");
    try
    {
        thread_local std::vector<PageIndexPair> pageIndices;
        pageIndices.clear();
        pageIndices.reserve(srcPages.size());
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            TLLM_CHECK_DEBUG(defrag || !srcPages.at(i)->scheduledForEviction());
            pageIndices.push_back(PageIndexPair{
                slotIdToPageIndexValue(dstSlots.at(i).slotId()), slotIdToPageIndexValue(srcPages.at(i)->slotId())});
        }

        std::vector<CachedCudaEvent const*> priorEvents;
        priorEvents.reserve(2 * srcPages.size());
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            priorEvents.push_back(&srcPages.at(i)->readyEvent);
            priorEvents.push_back(&dstSlots.at(i).readyEvent);
        }

        TemporaryCudaStream tempStream(priorEvents);
        auto updateReadyEvents = FuncGuard(
            [&]()
            {
                auto const finishEvent = tempStream.takeFinishEvent();
                for (std::size_t i = 0; i < srcPages.size(); ++i)
                {
                    dstSlots.at(i).readyEvent = finishEvent;
                    srcPages.at(i)->readyEvent = finishEvent;
                }
            });
        {
            auto scope = tempStream.enter();
            CUstream const stream = tempStream.get();
            submitMigrationBatch(
                dstLevel, srcLevel, batchingLayerGroupId, pageIndices.data(), pageIndices.size(), stream);
        }
        updateReadyEvents.run();

        constexpr size_t kMaxRetainedPageIndexPairs = (1u << 20u) / sizeof(PageIndexPair);
        if (pageIndices.capacity() > kMaxRetainedPageIndexPairs)
        {
            std::vector<PageIndexPair>().swap(pageIndices);
        }
        if (migrationRecorder && !defrag)
        {
            migrationRecorder(srcPages, dstSlots, srcLevel, dstLevel);
        }
        std::set<std::pair<std::string, int>> emittedCacheLevelUpdates;
        bool const emitCacheLevelUpdates
            = updateSrc && !defrag && srcLevel != dstLevel && static_cast<bool>(mEventSink);
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            if (updateSrc)
            {
                bool wasScheduled = srcPages.at(i)->scheduledForEviction();
                if (wasScheduled)
                    excludeFromEviction(*srcPages.at(i));
                // Replace the page's source slot with its destination and release the source slot back to the pool.
                Slot srcSlot = srcPages.at(i)->exchangeSlot(std::move(dstSlots.at(i)));
                srcPoolGroup.release(std::move(srcSlot));
                srcPages.at(i)->cacheLevel = dstLevel;
                if (emitCacheLevelUpdates && srcPages.at(i)->isCommitted())
                {
                    auto const& page = static_cast<CommittedPage const&>(*srcPages.at(i));
                    Block const* block = page.block;
                    std::string const blockKey = block
                        ? std::string(reinterpret_cast<char const*>(block->key.data()), block->key.size())
                        : std::string{};
                    if (block && !block->isOrphan()
                        && emittedCacheLevelUpdates.insert({blockKey, page.lifeCycle.value()}).second)
                    {
                        mEventSink->addCacheLevelUpdated(block->key, srcLevel, dstLevel, page.lifeCycle);
                    }
                }
                if (wasScheduled)
                    scheduleForEviction(*srcPages.at(i));
            }
        }
        return updateSrc ? std::nullopt : std::optional<std::vector<Slot>>{std::move(dstSlots)};
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

void StorageManager::batchedMigrateToGpu(
    std::vector<BatchedLockTarget> const& targets, KvCache& /*kvCache*/, MigrationRecorder const& migrationRecorder)
{
    std::map<MigrationBatchKey, std::vector<SharedPtr<Page>>> groups;
    for (auto const& t : targets)
    {
        if (t.page->cacheLevel == kHotLevel)
        {
            continue;
        }
        CacheLevel const srcLevel = t.page->cacheLevel;
        groups[{srcLevel, getMigrationBatchingLayerGroupId(kHotLevel, srcLevel, t.lifeCycle)}].push_back(t.page);
    }
    for (auto& [key, pages] : groups)
    {
        _batchedMigrate(kHotLevel, key.first, pages, /*updateSrc=*/true, migrationRecorder);
    }
}

void StorageManager::prefetch(
    CacheLevel dstLevel, TypedVec<LifeCycleId, TypedVec<CacheLevel, std::vector<SharedPtr<Page>>>> const& pages)
{
    TypedVec<PoolGroupIndex, SlotCount> numSlotsToMigrate(numPoolGroups(dstLevel), 0);
    std::vector<SharedPtr<Page>> scheduled;

    auto reschedulePagesGuard = FuncGuard(
        [this, &scheduled]()
        {
            for (auto const& page : scheduled)
            {
                scheduleForEviction(*page);
            }
            scheduled.clear();
        });

    for (LifeCycleId lifeCycle{0}; lifeCycle < pages.size(); ++lifeCycle)
    {
        auto const& lifeCyclePages = pages.at(lifeCycle);
        for (CacheLevel level{0}; level < lifeCyclePages.size(); ++level)
        {
            auto const& levelPages = lifeCyclePages.at(level);
            TLLM_CHECK_DEBUG(level >= dstLevel || levelPages.empty());
            TLLM_CHECK_DEBUG(std::all_of(levelPages.begin(), levelPages.end(),
                [lifeCycle](auto const& page) { return page->lifeCycle == lifeCycle; }));
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
                if (level != dstLevel)
                {
                    auto const dstPgIdx = getPoolGroupIndex(dstLevel, lifeCycle);
                    ++numSlotsToMigrate.at(dstPgIdx);
                }
            }
        }
    }

    prepareFreeSlots(dstLevel, numSlotsToMigrate);
    std::map<MigrationBatchKey, std::vector<SharedPtr<Page>>> migrationGroups;
    for (LifeCycleId lifeCycle{0}; lifeCycle < pages.size(); ++lifeCycle)
    {
        auto const& lifeCyclePages = pages.at(lifeCycle);
        for (CacheLevel level = dstLevel + 1; level < numCacheLevels(); ++level)
        {
            auto const& levelPages = lifeCyclePages.at(level);
            if (levelPages.empty())
            {
                continue;
            }
            auto& group = migrationGroups[{level, getMigrationBatchingLayerGroupId(dstLevel, level, lifeCycle)}];
            group.insert(group.end(), levelPages.begin(), levelPages.end());
        }
    }
    for (auto& [migrationPath, migrationPages] : migrationGroups)
    {
        _batchedMigrate(dstLevel, migrationPath.first, migrationPages, /*updateSrc=*/true);
    }
    reschedulePagesGuard.run();
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

LifeCycle const& StorageManager::getLifeCycle(LifeCycleId lc) const
{
    return mLifeCycles[lc];
}

PoolGroupIndex StorageManager::getPoolGroupIndex(CacheLevel level, LifeCycleId lc) const
{
    return poolGroupMapping(level).poolGroup(lc);
}

PoolGroupIndex StorageManager::getPoolGroupIndex(LifeCycleId lc) const
{
    return getPoolGroupIndex(kHotLevel, lc);
}

bool StorageManager::poolGroupNeedsHeadroomForGrowth(PoolGroupIndex pgIdx) const
{
    auto const poolGroupLifeCycles = poolGroupMapping(kHotLevel).lifeCycles(pgIdx);
    return std::any_of(poolGroupLifeCycles.begin(), poolGroupLifeCycles.end(),
        [this](LifeCycleId lifeCycle) { return !hasConstStateSize(mLifeCycles[lifeCycle]); });
}

PoolIndex StorageManager::numPools(CacheLevel level, PoolGroupIndex pgIdx) const
{
    return mLevels.at(level).storage->numPools(pgIdx);
}

PoolIndex StorageManager::numPools(PoolGroupIndex pgIdx) const
{
    return numPools(kHotLevel, pgIdx);
}

TypedVec<PoolIndex, size_t> StorageManager::slotSize(CacheLevel level, PoolGroupIndex pgIdx) const
{
    return slotDescList(level).at(pgIdx).slotSizeList();
}

TypedVec<PoolIndex, size_t> StorageManager::slotSize(PoolGroupIndex pgIdx) const
{
    return slotSize(kHotLevel, pgIdx);
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
    PoolGroupIndex pgIdx = getPoolGroupIndex(kHotLevel, attr.lifeCycleId);
    return mLevels[kHotLevel].storage->getBaseAddress(pgIdx, attr.poolIndex, SlotId{0}) + attr.offset;
}

MemAddress StorageManager::getMemPoolBaseAddress(PoolGroupIndex pgIdx, PoolIndex poolIdx) const
{
    return mLevels[kHotLevel].storage->getBaseAddress(pgIdx, poolIdx, SlotId{0});
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
    result.reserve(numPoolGroups(level));
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(level); ++pgIdx)
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
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(level); ++pgIdx)
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
                                   [this, level, pgIdx](auto const& p) {
                                       return p->cacheLevel == level && getPoolGroupIndex(level, p->lifeCycle) == pgIdx;
                                   }),
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
    TypedVec<PoolGroupIndex, SlotCount> evictReqs(numPoolGroups(level), 0);
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
    TypedVec<PoolGroupIndex, SlotCount> reqs(numPoolGroups(level), 0);
    reqs[pgIdx] = slotCountValueFromSize(overflowPages.size());
    prepareFreeSlots(level, reqs);

    // A17: all overflow pages must be at the expected cache level.
    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(overflowPages.begin(), overflowPages.end(),
                                   [level](auto const& p) { return p->cacheLevel == level; }),
        "Overflow page cache level mismatch");

    // Defragment: migrate overflow pages to free slots within the same level.
    _batchedMigrate(level, level, overflowPages, /*updateSrc=*/true, MigrationRecorder{}, /*defrag=*/true);

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
    auto const minSlots
        = level == kHotLevel ? mMinSlots : TypedVec<PoolGroupIndex, SlotCount>(numPoolGroups(level), SlotCount{1});
    size_t minQuota = minQuotaForLevel(lvlStorage.slotSizeLists(), lvlStorage.poolSizeGranularity(), minSlots);
    if (quota < minQuota)
    {
        throw std::invalid_argument("Quota " + std::to_string(quota)
            + " is insufficient for min_slots constraints (requires at least " + std::to_string(minQuota) + ")");
    }
    auto newNumSlots = lvlStorage.computeSlotCountList(ratioList, minSlots, quota);

    TLLM_CHECK_DEBUG(isLastLevel(level) || persistentPages == nullptr);

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
    if (lvlStorage.cacheTier() == CacheTier::GPU_MEM)
    {
        TLLM_CHECK_DEBUG(mGpuPhysMemAllocator);
        mGpuPhysMemAllocator->clear();
    }
}

TypedVec<PoolGroupIndex, float> StorageManager::getRatioList(CacheLevel level) const
{
    return mLevels.at(level).storage->ratioList();
}

TypedVec<LifeCycleId, float> StorageManager::ratioFromLength(
    CacheLevel level, int tokensPerBlock, int historyLength, int capacity) const
{
    if (capacity < historyLength)
    {
        TLLM_LOG_WARNING("Bad sampling for capacity and history_length");
        capacity = historyLength;
    }

    int const numBlocks = divUp(capacity, tokensPerBlock);
    TypedVec<LifeCycleId, size_t> numBytes(numLifeCycles(), 0);
    auto const ssmLcId = mLifeCycles.ssmLifeCycleId();
    auto const& lifeCycles = mLifeCycles.getAll();
    for (LifeCycleId lifeCycle{0}; lifeCycle < lifeCycles.size(); ++lifeCycle)
    {
        auto const poolSizes = slotSize(level, getPoolGroupIndex(level, lifeCycle));
        size_t const slotBytes = std::accumulate(poolSizes.begin(), poolSizes.end(), size_t{0});
        int numRequiredBlocks;
        if (ssmLcId.has_value() && lifeCycle == *ssmLcId)
        {
            numRequiredBlocks = 1;
        }
        else
        {
            auto const stale = getStaleRange(lifeCycles[lifeCycle], historyLength, tokensPerBlock);
            numRequiredBlocks = std::max(numBlocks - stale.length(), 1);
        }
        numBytes[lifeCycle] = static_cast<size_t>(numRequiredBlocks) * slotBytes;
    }
    return normalizeToRatio(numBytes);
}

TypedVec<PoolGroupIndex, float> StorageManager::toPoolGroupRatio(
    CacheLevel level, TypedVec<LifeCycleId, float> const& lifeCycleRatio) const
{
    TLLM_CHECK_WITH_INFO(
        lifeCycleRatio.size() == numLifeCycles(), "Lifecycle ratio length must match the number of lifecycles");
    TypedVec<PoolGroupIndex, float> poolGroupRatio(numPoolGroups(level), 0.0F);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        poolGroupRatio[getPoolGroupIndex(level, lifeCycle)] += lifeCycleRatio[lifeCycle];
    }
    return normalizeToRatio(poolGroupRatio);
}

TypedVec<PoolGroupIndex, float> StorageManager::projectPoolGroupRatio(
    CacheLevel srcLevel, CacheLevel dstLevel, TypedVec<LifeCycleId, float> const& srcLifeCycleRatio) const
{
    TLLM_CHECK_WITH_INFO(
        srcLifeCycleRatio.size() == numLifeCycles(), "Lifecycle ratio length must match the number of lifecycles");
    TypedVec<PoolGroupIndex, double> dstPoolGroupWeights(numPoolGroups(dstLevel), 0.0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        auto bytesPerSlot = [&](CacheLevel level)
        {
            auto const sizes = slotSize(level, getPoolGroupIndex(level, lifeCycle));
            return std::accumulate(sizes.begin(), sizes.end(), size_t{0});
        };
        size_t const srcBytesPerSlot = bytesPerSlot(srcLevel);
        size_t const dstBytesPerSlot = bytesPerSlot(dstLevel);
        TLLM_CHECK_WITH_INFO(srcBytesPerSlot > 0 && dstBytesPerSlot > 0, "Cache slot size must be positive");
        dstPoolGroupWeights[getPoolGroupIndex(dstLevel, lifeCycle)] += static_cast<double>(srcLifeCycleRatio[lifeCycle])
            * static_cast<double>(dstBytesPerSlot) / static_cast<double>(srcBytesPerSlot);
    }
    return normalizeToRatio(dstPoolGroupWeights);
}

TypedVec<LifeCycleId, float> StorageManager::ratioFromBatch(BatchDesc const& batch, int tokensPerBlock,
    std::optional<SwaScratchReuseConfig> const& swaScratchReuse, size_t granularity) const
{
    auto const numSlots = computeSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
    return normalizeToRatio(slotsToBytes(numSlots, granularity));
}

// ---------------------------------------------------------------------------
// computeSlotsFromConstraints
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, SlotCount> StorageManager::computeSlotsFromConstraints(std::vector<BatchDesc> const& constraints,
    int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse, float maxUtilForResume) const
{
    TLLM_CHECK_WITH_INFO(maxUtilForResume > 0.0F && maxUtilForResume <= 1.0F, "max_util_for_resume must be in (0, 1]");
    // All returned elements are positive. Constraint-derived floors include headroom for the utilization gate
    // checked by KvCache::resume, but only for life cycles whose hot pool group is subject to that gate. A pool
    // group that cannot grow is not, so it is sized to its exact floor.
    TypedVec<LifeCycleId, SlotCount> maxSlots(numLifeCycles(), 0);

    auto swaFloorBlocks = [tokensPerBlock](AttnLifeCycle const& lc) -> int
    {
        int window = *lc.windowSize;
        // Handle oscillation of slot count required by SWA while the window slides.
        return lc.numSinkBlocks + (window + tokensPerBlock - 2) / tokensPerBlock + 1;
    };

    // Full-attention lifecycles share the largest SWA floor: all attention
    // lifecycles see the same seq_len, so this is a valid lower bound.
    int floorNumBlocks = 1;
    for (auto const& [lcId, attn] : mLifeCycles.attentionLifeCycles())
    {
        if (attn->windowSize.has_value())
            floorNumBlocks = std::max(floorNumBlocks, swaFloorBlocks(*attn));
    }
    for (auto const& [lcIdx, lc] : mLifeCycles)
    {
        auto const* attn = std::get_if<AttnLifeCycle>(&lc);
        if (attn == nullptr)
        {
            // SSM / non-attention: 1 slot floor per life cycle.
            maxSlots[lcIdx] = 1;
        }
        else if (attn->windowSize.has_value())
        {
            maxSlots[lcIdx] = swaFloorBlocks(*attn);
        }
        else
        {
            maxSlots[lcIdx] = floorNumBlocks;
        }
    }
    for (auto const& batch : constraints)
    {
        auto slots = computeSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
        for (LifeCycleId lifeCycle{0}; lifeCycle < slots.size(); ++lifeCycle)
        {
            auto const poolGroup = getPoolGroupIndex(kHotLevel, lifeCycle);
            double const utilizationLimit = poolGroupNeedsHeadroomForGrowth(poolGroup) ? maxUtilForResume : 1.0;
            auto const scaledSlots
                = static_cast<SlotCount>(std::ceil(static_cast<double>(slots[lifeCycle]) / utilizationLimit));
            maxSlots[lifeCycle] = std::max(maxSlots[lifeCycle], scaledSlots);
        }
    }
    return maxSlots;
}

TypedVec<PoolGroupIndex, SlotCount> StorageManager::computePoolGroupMinSlotsFromConstraints(
    std::vector<BatchDesc> const& constraints, int tokensPerBlock,
    std::optional<SwaScratchReuseConfig> const& swaScratchReuse, float maxUtilForResume) const
{
    auto const lifeCycleFloors = computeSlotsFromConstraints({}, tokensPerBlock, swaScratchReuse, maxUtilForResume);
    TypedVec<PoolGroupIndex, SlotCount> maxSlots(numPoolGroups(kHotLevel), 0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        maxSlots[getPoolGroupIndex(kHotLevel, lifeCycle)] += lifeCycleFloors[lifeCycle];
    }

    for (auto const& batch : constraints)
    {
        auto const slots = computePoolGroupSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
        for (PoolGroupIndex poolGroup{0}; poolGroup < slots.size(); ++poolGroup)
        {
            double const utilizationLimit = poolGroupNeedsHeadroomForGrowth(poolGroup) ? maxUtilForResume : 1.0;
            auto const scaledSlots
                = static_cast<SlotCount>(std::ceil(static_cast<double>(slots[poolGroup]) / utilizationLimit));
            maxSlots[poolGroup] = std::max(maxSlots[poolGroup], scaledSlots);
        }
    }
    return maxSlots;
}

// ---------------------------------------------------------------------------
// computeSlotsForBatch
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, SlotCount> StorageManager::computeSlotsForBatch(
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const
{
    TypedVec<LifeCycleId, SlotCount> numSlots(numLifeCycles(), 0);
    auto ssmLcId = mLifeCycles.ssmLifeCycleId();
    int sysBlocks = batch.systemPromptLength / tokensPerBlock;

    for (auto const& [lcIdx, lc] : mLifeCycles)
    {
        if (ssmLcId.has_value() && lcIdx == *ssmLcId)
        {
            // SSM: always 1 dedicated block per request, never shared.
            numSlots[lcIdx] += slotCountValueFromSize(batch.kvCaches.size());
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
        numSlots[lcIdx] += sysBlocks - staleIntersection.length();

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
                numSlots[lcIdx] += (uniqueNonStale - numScratch) + mSlotUtilFracMax[lcIdx].ceilMul(numScratch);
            }
            else
            {
                numSlots[lcIdx] += uniqueNonStale;
            }
        }
    }
    return numSlots;
}

// ---------------------------------------------------------------------------
// computePoolGroupSlotsForBatch
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::computePoolGroupSlotsForBatch(
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const
{
    auto const slotsByLifeCycle = computeSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
    TypedVec<PoolGroupIndex, SlotCount> numSlots(numPoolGroups(kHotLevel), 0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < slotsByLifeCycle.size(); ++lifeCycle)
    {
        numSlots[getPoolGroupIndex(kHotLevel, lifeCycle)] += slotsByLifeCycle[lifeCycle];
    }
    return numSlots;
}

// ---------------------------------------------------------------------------
// slotsToBytes
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, size_t> StorageManager::slotsToBytes(
    TypedVec<LifeCycleId, SlotCount> const& numSlots, size_t granularity) const
{
    TypedVec<LifeCycleId, size_t> numBytes(numLifeCycles(), 0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numSlots.size(); ++lifeCycle)
    {
        auto const poolGroup = getPoolGroupIndex(kHotLevel, lifeCycle);
        for (auto const poolSize : slotSize(kHotLevel, poolGroup))
        {
            numBytes[lifeCycle] += roundUp(slotCountToSizeT(numSlots[lifeCycle]) * poolSize, granularity);
        }
    }
    return numBytes;
}

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
    TypedVec<PoolGroupIndex, float> const& ratio, TypedVec<PoolGroupIndex, SlotCount> const& minSlots) const
{
    CacheTier tier = cacheTierOf(tierConfig);
    size_t quota = cacheTierQuota(tierConfig);
    size_t granularity = tier == CacheTier::GPU_MEM ? mGpuPhysMemAllocator->physMemSize()
                                                    : CacheLevelManager::cacheTierGranularity(tier, quota);
    quota = std::max(minQuotaForLevel(slotSizeLists, granularity, minSlots), roundUp(quota, granularity));
    return CacheLevelStorage::ratioToSlotCountList(quota, slotSizeLists, ratio, granularity, minSlots);
}

// ---------------------------------------------------------------------------
// minQuotaForLevel
// ---------------------------------------------------------------------------

size_t StorageManager::minQuotaForLevel(TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    size_t granularity, TypedVec<PoolGroupIndex, SlotCount> const& minSlots) const
{
    size_t total = 0;
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotSizeLists.size(); ++pgIdx)
    {
        for (auto slotSize : slotSizeLists[pgIdx])
        {
            total += roundUp(slotCountToSizeT(minSlots[pgIdx]) * slotSize, granularity);
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// constrainPoolGroupRatio
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, float> StorageManager::constrainPoolGroupRatio(
    TypedVec<PoolGroupIndex, float> const& ratio) const
{
    auto& gpuStorage = *mLevels[kHotLevel].storage;
    size_t granularity = gpuStorage.poolSizeGranularity();
    auto slotCountList = gpuStorage.computeSlotCountList(ratio, mMinSlots);
    auto numBytes = slotsToBytes(slotCountList, granularity);
    return normalizeToRatio(numBytes);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
