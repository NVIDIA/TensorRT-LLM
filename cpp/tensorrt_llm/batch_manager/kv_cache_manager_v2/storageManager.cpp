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
#include "kv_cache_manager_v2/batchedPageCopy.h"
#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/copyEngine.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/storage/remoteMem.h"
#include "kv_cache_manager_v2/utils/copyUtils.h"
#include "kv_cache_manager_v2/utils/funcGuard.h"
#include "kv_cache_manager_v2/utils/hostMem.h"
#include "kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/common/logger.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>

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
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, PooledPhysMemAllocator* gpuPhysMemAllocator,
    HostMemBackingOptions const& options, std::optional<size_t> sparseReservation,
    SharedPtr<RemoteMemClient> remoteClient)
    : cacheLevel(cl)
    , cacheTier(cacheTierOf(tierConfig))
    , controller(lifeCycleGrouping, cl)
{
    TLLM_CHECK((cacheTier == CacheTier::GPU_MEM) == (gpuPhysMemAllocator != nullptr));
    storage = createCacheLevelStorage(tierConfig, slotDescList, slotCountList, gpuPhysMemAllocator, options,
        sparseReservation, std::move(remoteClient));
}

CacheLevelManager::CacheLevelManager(TypedVec<LifeCycleId, PoolGroupIndex> const& lifeCycleGrouping, CacheLevel cl,
    std::unique_ptr<CacheLevelStorage> levelStorage)
    : cacheLevel(cl)
    , cacheTier(levelStorage->cacheTier())
    , storage(std::move(levelStorage))
    , controller(lifeCycleGrouping, cl)
{
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

bool isRemoteTier(CacheTierConfig const& tier) noexcept
{
    return std::holds_alternative<RemoteCacheTierConfig>(tier);
}

//! A remote level's own split of its quota across layer groups (RemoteCacheTierConfig::poolRatio), normalized;
//! nullopt where the level follows the hot level's ratio. Throws when the entry count is not one per layer group.
std::optional<TypedVec<LifeCycleId, float>> remotePoolRatioOf(CacheTierConfig const& tier, LifeCycleId numLifeCycles)
{
    auto const* remote = std::get_if<RemoteCacheTierConfig>(&tier);
    if (remote == nullptr || remote->poolRatio.empty())
    {
        return std::nullopt;
    }
    auto const& weights = remote->poolRatio;
    TLLM_CHECK_WITH_INFO(weights.size() == toSizeT(numLifeCycles),
        "remote pool ratio: %zu entries for %zu layer groups", weights.size(), toSizeT(numLifeCycles));
    double const sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    TypedVec<LifeCycleId, float> ratio(numLifeCycles, 0.F);
    for (LifeCycleId lc{0}; lc < numLifeCycles; ++lc)
    {
        ratio[lc] = static_cast<float>(weights[toSizeT(lc)] / sum);
    }
    return ratio;
}

// The host sparse pool quota an unset configuration gets: 80% of the host memory the host
// tiers' quotas leave. Host tiers configured past physical memory leave none.
size_t defaultHostSparsePoolQuota(TypedVec<CacheLevel, CacheTierConfig> const& cacheTiers)
{
    size_t hostTierQuota = 0;
    for (auto const& tier : cacheTiers)
    {
        if (cacheTierOf(tier) == CacheTier::HOST_MEM)
        {
            hostTierQuota += cacheTierQuota(tier);
        }
    }
    size_t const total = hostTotalMemory();
    size_t const remaining = total > hostTierQuota ? total - hostTierQuota : 0;
    return remaining / 5 * 4;
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
        TLLM_CHECK_WITH_INFO(!result[lcId].empty(), "Lifecycle %d has no SlotDescVariant", lcId.value());
    }
    return result;
}

SlotId toSlotId(int32_t pageIndex)
{
    TLLM_CHECK_DEBUG(pageIndex >= 0);
    return SlotId{pageIndex};
}

//! Drop a page-index buffer kept across calls (thread_local) once a large batch has grown it past 1 MiB.
void trimRetainedPageIndices(std::vector<PageIndexPair>& pageIndices)
{
    constexpr size_t kMaxRetainedPageIndexPairs = (1u << 20u) / sizeof(PageIndexPair);
    if (pageIndices.capacity() > kMaxRetainedPageIndexPairs)
    {
        std::vector<PageIndexPair>().swap(pageIndices);
    }
}

void sortFallenPagesByPriority(
    TypedVec<LifeCycleId, std::deque<SharedPtr<Page>, PoolAllocator<SharedPtr<Page>>>>& fallenPages)
{
    for (auto& pages : fallenPages)
    {
        std::stable_sort(pages.begin(), pages.end(),
            [](auto const& lhs, auto const& rhs) { return lhs->priority() < rhs->priority(); });
    }
}

//! Whether a page's bytes can no longer change, so a disk copy of it stays valid. A committed page is final only
//! once its block is full: a request that commits a partial tail keeps decoding into the same slot.
bool hasFinalContent(Page const& page) noexcept
{
    if (!page.isCommitted())
    {
        return false;
    }
    auto const& committed = static_cast<CommittedPage const&>(page);
    return committed.block != nullptr && committed.numTokensInBlock >= committed.block->tokensPerBlock();
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
        copyToDeviceSyncSource(static_cast<CUdeviceptr>(device.address()), pageIndices + offset, chunkPages, stream);

        if (!submit(reinterpret_cast<PageIndexPair const*>(device.address()), chunkPages, stream))
        {
            return false;
        }
        offset += chunkPages;
    }
    return true;
}

bool StorageManager::submitBoundPoolColdPageCodec(bool encode, LayerGroupId batchingLayerGroupId,
    PoolGroupIndex hotPgIdx, void const* coldBase, PageIndexPair const* pageIndices, size_t numPages, CUstream stream)
{
    auto& hot = poolGroup(kHotLevel, hotPgIdx);
    size_t const numPools = static_cast<size_t>(hot.numPools().value());
    thread_local std::vector<std::vector<PageIndexPair>> perPool;
    perPool.resize(numPools);
    for (PoolIndex poolIdx{0}; poolIdx < hot.numPools(); ++poolIdx)
    {
        auto& indices = perPool[static_cast<size_t>(poolIdx.value())];
        indices.assign(pageIndices, pageIndices + numPages);
        if (!hot.isBoundPool(poolIdx))
            continue;
        auto const& pool = hot.arenaPool(poolIdx);
        for (auto& pair : indices)
        {
            int32_t& hotIndex = encode ? pair.src : pair.dst;
            auto const local = pool.localId(SlotId{hotIndex});
            TLLM_CHECK_WITH_INFO(local.has_value(), "Bound pool %d has no binding for group slot id %d",
                static_cast<int>(poolIdx.value()), hotIndex);
            hotIndex = slotIdToPageIndexValue(*local);
        }
    }
    std::vector<PageIndexPair const*> arrays(numPools);
    auto submit = [&](size_t count)
    {
        auto const cudaStream = reinterpret_cast<cudaStream_t>(stream);
        return encode
            ? mColdPageCodec->encodePools(
                batchingLayerGroupId, const_cast<void*>(coldBase), arrays.data(), numPools, count, cudaStream)
            : mColdPageCodec->decodePools(batchingLayerGroupId, coldBase, arrays.data(), numPools, count, cudaStream);
    };
    if (mPageIndexLocations.at(batchingLayerGroupId) == PageIndexLocation::kHost || numPages == 0)
    {
        for (size_t p = 0; p < numPools; ++p)
            arrays[p] = perPool[p].data();
        return submit(numPages);
    }
    TLLM_CHECK_WITH_INFO(mIndexStagingManager, "Device page indices need index staging");
    // Device page indices: stage every pool's chunk in one slice, pool after pool.
    size_t const rowBytes = numPools * sizeof(PageIndexPair);
    size_t offset = 0;
    while (offset < numPages)
    {
        size_t const maxBytes
            = std::max(rowBytes, roundDown(std::min((numPages - offset) * rowBytes, kMaxIndexBatchBytes), rowBytes));
        auto device = mIndexStagingManager->acquire(rowBytes, maxBytes, rowBytes, alignof(PageIndexPair), stream);
        size_t const chunkPages = std::min(numPages - offset, device.size() / rowBytes);
        TLLM_CHECK_DEBUG(chunkPages > 0);
        for (size_t p = 0; p < numPools; ++p)
        {
            MemAddress const address = device.address() + p * chunkPages * sizeof(PageIndexPair);
            copyToDeviceSyncSource(static_cast<CUdeviceptr>(address), perPool[p].data() + offset, chunkPages, stream);
            arrays[p] = reinterpret_cast<PageIndexPair const*>(address);
        }
        if (!submit(chunkPages))
            return false;
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
    float maxUtilForResume, Priority coldTierOffloadMinPriority, std::optional<BatchDesc> const& coldTypicalBatch)
    : mLifeCycles(lifeCycles)
    , mEventSink(std::move(eventSink))
    , mColdTierOffloadMinPriority(coldTierOffloadMinPriority)
    , mHotPoolGroupMapping(config.lifeCycleGrouping())
    , mSwaScratchReuse(std::move(swaScratchReuse))
    , mColdPageCodec(coldPageCodec ? std::move(coldPageCodec) : createDefaultKvCacheColdPageCodec())
{
    IKvCacheColdPageCodec& codec = *mColdPageCodec;
    mLayerToLifeCycleIds = config.layerToLifeCycleIds();
    mSlotToPageIndices = computeSlotToPageIndices(config);
    mBufferAttr = config.bufferAttributes();
    // The hot tier is registered up front; the cold tiers follow once the codec has told us
    // their page sizes. Nothing may read a cold level before then.
    TLLM_CHECK(appendLevelSlotDescList(config.slotDescList) == kHotLevel);

    // A scratch slot serves one layer at a time. Reserve the largest layer share
    // across the coalesced pools of each lifecycle, retaining exact fractions.
    mSlotUtilFracMax.resize(lifeCycles.size(), Rational{0, 1});
    for (auto const& slotDesc : config.slotDescList)
    {
        for (auto const& variant : slotDesc.variants)
        {
            for (auto const& buffers : variant.coalescedBuffers)
            {
                std::unordered_map<LayerId, int> counts;
                for (auto const& buffer : buffers.bufferIds)
                {
                    ++counts[buffer.layerId];
                }
                for (auto const& [layer, count] : counts)
                {
                    Rational const fraction{count, buffers.numBuffers()};
                    auto& maximum = mSlotUtilFracMax[variant.lifeCycleId];
                    if (fraction > maximum)
                    {
                        maximum = fraction;
                    }
                }
            }
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

    // Every pool's size, and which of them are sparse. One array of sizes, so the two budgets
    // cannot disagree about what a pool costs; the selector only says who pays.
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> slotSizeLists;
    TypedVec<PoolGroupIndex, std::vector<PoolIndex>> sparsePools;
    slotSizeLists.reserve(hotSlotDescList.size());
    sparsePools.reserve(hotSlotDescList.size());
    for (auto const& sd : hotSlotDescList)
    {
        auto const all = sd.slotSizeList();
        std::vector<PoolIndex> sparse;
        size_t dense = 0;
        for (PoolIndex p{0}; p < all.size(); ++p)
        {
            if (sd.isSparse(p))
            {
                sparse.push_back(p);
            }
            else
            {
                ++dense;
            }
        }
        // Only the GPU tier needs this: sparsity exists at the hot level alone, so a cold
        // level's groups size entirely from their own quota.
        TLLM_CHECK_WITH_INFO(dense > 0,
            "A pool group must hold at least one non-sparse pool; a wholly sparse group has no device "
            "bytes from which to size its slots");
        slotSizeLists.push_back(all);
        sparsePools.push_back(std::move(sparse));
    }

    size_t const gpuQuota = cacheTierQuota(config.cacheTiers[kHotLevel]);
    mGpuPhysMemAllocator
        = std::make_unique<PooledPhysMemAllocator>(tierAllocationGranularity(CacheTier::GPU_MEM, gpuQuota));
    size_t const gpuGranularity = mGpuPhysMemAllocator->physMemSize();

    // Constraints are hot-level feasibility floors. They are scaled by
    // 1/maxUtilForResume because KvCache::resume rejects a pool group above that
    // utilization. Other levels need only the structural one-slot floor.
    mHotMinSlots
        = computePoolGroupMinSlotsFromConstraints(constraints, tokensPerBlock, mSwaScratchReuse, maxUtilForResume);
    mMaxUtilForResume = maxUtilForResume;

    // A fixed-capacity group's floor is its fixed size, so every quota check reserves it. Like the
    // elastic floors, it raises a GPU quota that is too small to hold all floors.
    mFixedSlots = config.fixedSlotCounts;
    if (mFixedSlots.empty())
    {
        mFixedSlots = FixedSlotCounts(numPoolGroups(), std::nullopt);
    }
    TLLM_CHECK_WITH_INFO(mFixedSlots.size() == numPoolGroups(), "Fixed slot counts must cover every pool group");
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(); ++pgIdx)
    {
        if (auto const fixed = mFixedSlots[pgIdx])
        {
            mHotMinSlots[pgIdx] = *fixed;
        }
    }

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
        lifeCycleRatio = ratioFromBatch(*typicalBatch, tokensPerBlock, mSwaScratchReuse, gpuGranularity, sparsePools);
    }
    else if (!constraints.empty())
    {
        auto lifeCycleSlots
            = computeSlotsFromConstraints(constraints, tokensPerBlock, mSwaScratchReuse, maxUtilForResume);
        lifeCycleRatio = normalizeToRatio(slotsToBytes(lifeCycleSlots, gpuGranularity, sparsePools));
    }
    else
    {
        // Fallback: average history length 2048.
        BatchDesc fallback;
        fallback.kvCaches.push_back(KVCacheDesc{2049, 2048});
        lifeCycleRatio = ratioFromBatch(fallback, tokensPerBlock, mSwaScratchReuse, gpuGranularity, sparsePools);
    }

    auto const hotRatio = toPoolGroupRatio(kHotLevel, lifeCycleRatio);

    // Checked before any level exists, so a malformed ratio fails construction before remote memory is claimed.
    mHasRemoteLevel = std::any_of(config.cacheTiers.begin(), config.cacheTiers.end(), isRemoteTier);
    TypedVec<CacheLevel, std::optional<TypedVec<LifeCycleId, float>>> remotePoolRatios;
    for (auto const& tier : config.cacheTiers)
    {
        remotePoolRatios.push_back(remotePoolRatioOf(tier, numLifeCycles()));
    }

    mLevels.reserve(config.cacheTiers.size());

    // A sparse pool's hot pages are host memory. Its bytes are excluded from the GPU quota
    // rather than sized from it: a slot spans every pool of its group, so a sparse pool's
    // row count follows from the group's device-backed pools, which the quota does size.
    // Those pages still need host memory, so an unset quota is filled in here and kept: it
    // bounds the slot counts and each sparse pool's address-space reservation alike.
    mHostSparsePoolQuota = config.hostSparsePoolQuota.has_value() ? *config.hostSparsePoolQuota
                                                                  : defaultHostSparsePoolQuota(config.cacheTiers);
    // From the descriptors: the level's storage does not exist yet.
    mGpuArena = config.gpuArena;
    mDropUnreferencedOnRebalance = config.dropUnreferencedOnRebalance;
    mInPlaceRelocation = config.inPlaceRelocation;
    if (mGpuArena && mInPlaceRelocation)
    {
        // In-place relocation copies on a pooled TemporaryCudaStream. The pool creates its 128 streams on first
        // use, which otherwise lands inside the first rebalance that relocates a live request (~6.4 ms measured
        // there with DeepSeek-V4-Flash). Create them now, at startup.
        static_cast<void>(CudaStreamPool::instance());
    }
    if (mGpuArena)
    {
        // Copy-based rebalancing moves pool windows inside one device arena, which has no room for host-backed
        // (sparse) pools: they hold offloaded sparse attention history, whose pinned host memory is planned once
        // (CoReba doc 38, phase 1). A group holding one is therefore fixed at the size the VMM allocator gives it
        // now -- the same call, so the host plan sees the same counts -- and lives beside the arena.
        auto const secondBudget = _secondaryBudget(sparsePools, hostMemSizeGranularity(), mHostSparsePoolQuota);
        bool const sparseToFix
            = std::any_of(sparsePools.begin(), sparsePools.end(), [&](auto const& pools) { return !pools.empty(); });
        if (sparseToFix)
        {
            auto const initial = computeSlotCountForLevel(config.cacheTiers[kHotLevel], slotSizeLists, hotRatio,
                mHotMinSlots, gpuGranularity, secondBudget, mFixedSlots);
            for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(); ++pgIdx)
            {
                if (sparsePools[pgIdx].empty() || mFixedSlots[pgIdx].has_value())
                    continue;
                mFixedSlots[pgIdx] = initial[pgIdx];
                mHotMinSlots[pgIdx] = initial[pgIdx];
                TLLM_LOG_INFO(
                    "KVCacheManagerV2: copy-based rebalancing holds GPU pool group %d (host-backed sparse pools) at "
                    "%lld slots",
                    pgIdx.value(), static_cast<long long>(initial[pgIdx]));
            }
        }
        if (std::all_of(mFixedSlots.begin(), mFixedSlots.end(), [](auto const& fixed) { return fixed.has_value(); }))
        {
            // Nothing left to move: the VMM storage holds fixed groups as they are, and no rebalance changes them.
            TLLM_LOG_INFO(
                "KVCacheManagerV2: every GPU pool group has a fixed size; copy-based rebalancing has "
                "nothing to rebalance and is not used");
            mGpuArena = false;
            mInPlaceRelocation = false;
            mDropUnreferencedOnRebalance = false;
        }
    }
    if (mGpuArena)
    {
        // Fixed groups (configured, or holding sparse pools) never move, so they are allocated as the VMM path
        // allocates them, beside the arena; the arena holds the other ("elastic") groups.
        size_t fixedDeviceBytes = 0;
        TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> elasticSizes;
        TypedVec<PoolGroupIndex, SlotCount> elasticFloors;
        auto const hotDescs = slotDescList(kHotLevel);
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(); ++pgIdx)
        {
            if (auto const fixed = mFixedSlots[pgIdx])
            {
                for (PoolIndex p{0}; p < slotSizeLists[pgIdx].size(); ++p)
                {
                    if (!hotDescs[pgIdx].isSparse(p))
                        fixedDeviceBytes += roundUp(slotCountToSizeT(*fixed) * slotSizeLists[pgIdx][p], gpuGranularity);
                }
                continue;
            }
            elasticSizes.push_back(slotSizeLists[pgIdx]);
            elasticFloors.push_back(mHotMinSlots[pgIdx]);
        }
        // The rest of the GPU quota becomes one arena, rounded down to whole physical chunks. Like the VMM path
        // (computeSlotCountForLevel), a quota below what minSlots needs is raised to it: the executor's
        // estimation pass sizes the first manager from free memory, which can be tiny (one GPU holding
        // most of DeepSeek-V4's weights leaves under 1 GiB).
        size_t const elasticQuota = gpuQuota > fixedDeviceBytes ? gpuQuota - fixedDeviceBytes : 0;
        size_t const arenaBytes = std::max(roundDown(elasticQuota, gpuGranularity),
            roundUp(ArenaGpuCacheLevelStorage::minArenaBytes(elasticSizes, elasticFloors), gpuGranularity));
        auto arena = std::make_unique<ArenaGpuCacheLevelStorage>(hotDescs, arenaBytes, hotRatio, mHotMinSlots,
            *mGpuPhysMemAllocator, mFixedSlots, HostMemBackingOptions{}, mHostSparsePoolQuota);
        // Migration to and from lower tiers: a bound pool has its own slot ids, so the cold-page codec gets one
        // page-index array per pool (submitBoundPoolColdPageCodec), which it must accept.
        TLLM_CHECK_WITH_INFO(
            !arena->hasBoundPools() || config.cacheTiers.size() == CacheLevel{1} || codec.supportsPerPoolPageIndices(),
            "Copy-based rebalancing with several pools in a pool group needs a cold-page codec that supports "
            "per-pool page indices (supportsPerPoolPageIndices) for host or disk cache tiers");
        mLevels.emplace_back(lifeCycleGrouping(kHotLevel), kHotLevel, std::move(arena));
    }
    else
    {
        auto gpuSlotCounts = computeSlotCountForLevel(config.cacheTiers[kHotLevel], slotSizeLists, hotRatio,
            mHotMinSlots, gpuGranularity, _secondaryBudget(sparsePools, hostMemSizeGranularity(), mHostSparsePoolQuota),
            mFixedSlots);
        mLevels.emplace_back(lifeCycleGrouping(kHotLevel), kHotLevel, config.cacheTiers[kHotLevel],
            slotDescList(kHotLevel), gpuSlotCounts, mGpuPhysMemAllocator.get(), HostMemBackingOptions{},
            mHostSparsePoolQuota);
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(); ++pgIdx)
        {
            if (mFixedSlots[pgIdx])
            {
                TLLM_LOG_INFO(
                    "KVCacheManagerV2: GPU pool group %d (region layer group %d) has a fixed capacity of %lld slots",
                    pgIdx.value(), mHotPoolGroupMapping.lifeCycles(pgIdx)[0].value(),
                    static_cast<long long>(gpuSlotCounts[pgIdx]));
            }
        }
    }

    auto const gpuDescs = poolGroupDescs();
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
    std::map<std::pair<ReusePolicy, size_t>, PoolGroupIndex> coldGroupByPageBytes;
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        size_t const coldPageBytes = coldPageBytesByLifeCycle[lifeCycle];
        auto const policy = effectivePoolReusePolicy(mLifeCycles[lifeCycle]);
        auto [it, inserted] = coldGroupByPageBytes.emplace(
            std::make_pair(policy, coldPageBytes), PoolGroupIndex{static_cast<int>(coldSlotDescList.size().value())});
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

    // Every pool group needs an allocation unit of its own, so a level whose configured maximum is
    // below that total cannot be built at any size: the quota is raised to the same minimum below,
    // and would then sit above the maximum from the outset. Checked for all cold levels before any
    // of them is built, so a rejection does not leave earlier levels already allocated.
    for (CacheLevel level{1}; level < config.cacheTiers.size(); ++level)
    {
        auto const& tierConfig = config.cacheTiers[level];
        auto const maxQuota = cacheTierMaxQuota(tierConfig);
        if (!maxQuota.has_value())
        {
            continue;
        }
        size_t const granularity = tierCommitUnit(tierConfig, gpuGranularity);
        if (size_t const minQuota = minQuotaForLevel(coldSlotSizeLists, granularity, coldMinSlots);
            *maxQuota < minQuota)
        {
            throw std::invalid_argument("max_quota " + std::to_string(*maxQuota) + " for cache level "
                + std::to_string(level.value()) + " is below the " + std::to_string(minQuota)
                + " bytes it needs at minimum, one allocation unit of " + std::to_string(granularity) + " per pool");
        }
    }

    // A remote level with its own pool ratio takes that share of its own bytes per layer group (layer groups that
    // share a cold pool group add up); it is a ratio of this level's bytes, so it is not projected from the hot level.
    auto const levelShareRatio = [this](CacheLevel level, TypedVec<LifeCycleId, float> const& shares)
    {
        TypedVec<PoolGroupIndex, double> weights(numPoolGroups(level), 0.0);
        for (LifeCycleId lc{0}; lc < numLifeCycles(); ++lc)
        {
            weights[getPoolGroupIndex(level, lc)] += static_cast<double>(shares[lc]);
        }
        return normalizeToRatio(weights);
    };

    for (CacheLevel level{1}; level < config.cacheTiers.size(); ++level)
    {
        TLLM_CHECK(appendLevelSlotDescList(coldSlotDescList) == level);
        // An explicit remote pool ratio wins, then the cold typical step, then the hot level's projection.
        auto const& remotePoolRatio = remotePoolRatios[level];
        auto const coldRatio = remotePoolRatio.has_value() ? levelShareRatio(level, *remotePoolRatio)
            : coldTypicalBatch.has_value() ? coldPoolGroupRatioFromBatch(level, *coldTypicalBatch, tokensPerBlock)
                                           : projectPoolGroupRatio(kHotLevel, level, lifeCycleRatio);
        // No second budget: a cold group is a single pool holding whatever the codec packs
        // into it, dense and sparse buffers alike, so its quota pays for all of it.
        auto slotCounts = computeSlotCountForLevel(
            config.cacheTiers[level], coldSlotSizeLists, coldRatio, coldMinSlots, gpuGranularity);
        if (mGpuArena && cacheTierOf(config.cacheTiers[level]) == CacheTier::HOST_MEM)
        {
            // Copy-based rebalancing: the host quota becomes one arena too, committed once and never resized.
            auto const& tierConfig = config.cacheTiers[level];
            size_t const unit = ArenaHostCacheLevelStorage::arenaUnitBytes(tierCommitUnit(tierConfig, gpuGranularity));
            size_t const arenaBytes = std::max(roundDown(cacheTierQuota(tierConfig), unit),
                roundUp(ArenaGpuCacheLevelStorage::minArenaBytes(coldSlotSizeLists, coldMinSlots), unit));
            auto arena = std::make_unique<ArenaHostCacheLevelStorage>(slotDescList(level), arenaBytes, unit, coldRatio,
                coldMinSlots, tierBackingOptions(tierConfig, gpuGranularity, config.hostMemBacking));
            mLevels.emplace_back(lifeCycleGrouping(level), level, std::move(arena));
            continue;
        }
        auto* gpuPhysMemAllocator
            = cacheTierOf(config.cacheTiers[level]) == CacheTier::GPU_MEM ? mGpuPhysMemAllocator.get() : nullptr;
        mLevels.emplace_back(lifeCycleGrouping(level), level, config.cacheTiers[level], slotDescList(level), slotCounts,
            gpuPhysMemAllocator, tierBackingOptions(config.cacheTiers[level], gpuGranularity, config.hostMemBacking),
            std::nullopt, remoteClientFor(config.cacheTiers[level]));
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

    mCopyEngine = std::make_unique<CopyEngine>(mPageStagingManager.get());

    // A remote level moves its pages between the page staging buffer and remote memory, so the staging buffer is
    // registered with each client once, for the life of this manager.
    for (auto const& client : mRemoteClients)
    {
        TLLM_CHECK_WITH_INFO(mPageStagingManager != nullptr, "A remote cache level needs page staging");
        client->registerLocal(mPageStagingManager->baseAddress(), mPageStagingManager->totalSize());
    }
    mLastLiveLevel = numCacheLevels() - 1;

    mNumRetainedCopies.resize(numCacheLevels());
    for (CacheLevel level{0}; level < mLevels.size(); ++level)
    {
        mNumRetainedCopies[level].resize(numPoolGroups(level), 0);
    }
    // The options are per disk tier: a tier that does not set them gets neither retained copies nor write-through.
    mKeepCopyAfterRestoreAt = TypedVec<CacheLevel, bool>(numCacheLevels(), false);
    mProtectedCopyFractionAt = TypedVec<CacheLevel, float>(numCacheLevels(), 1.0F);
    mWriteThroughAt = TypedVec<CacheLevel, bool>(numCacheLevels(), false);
    for (CacheLevel level{0}; level < config.cacheTiers.size(); ++level)
    {
        if (auto const* disk = std::get_if<DiskCacheTierConfig>(&config.cacheTiers[level]))
        {
            mKeepCopyAfterRestoreAt[level] = disk->keepCopyAfterRestore;
            mProtectedCopyFractionAt[level] = disk->protectedCopyFraction;
            // Write-through copies pages as they are evicted into the host tier right above this disk tier. Without
            // one (no host cache, or a KV connector that takes the host tier's place) it would never run.
            bool const hostAbove
                = level.value() >= 1 && cacheTier(CacheLevel{level.value() - 1}) == CacheTier::HOST_MEM;
            if (disk->writeThrough && !hostAbove)
            {
                TLLM_LOG_WARNING(
                    "[kvCacheManagerV2] disk tier %d: write_through needs a host tier right above it; disabled",
                    static_cast<int>(level.value()));
            }
            mWriteThroughAt[level] = disk->writeThrough && hostAbove;
            mDiskWriteThrough = mDiskWriteThrough || mWriteThroughAt[level];
            configureAsyncDiskIo(static_cast<size_t>(disk->numIoThreads)); // validate() rejects negative values
        }
    }
    if (mDiskWriteThrough && !asyncDiskIoEnabled())
    {
        // Write-through copies are background writes; without the I/O pool they would run on the stream's host
        // function, ahead of restores.
        TLLM_LOG_WARNING("[kvCacheManagerV2] disk write_through needs async disk I/O (num_io_threads > 0); disabled");
        mDiskWriteThrough = false;
        mWriteThroughAt = TypedVec<CacheLevel, bool>(numCacheLevels(), false);
    }
    for (CacheLevel level{0}; level < numCacheLevels(); ++level)
    {
        if (mKeepCopyAfterRestoreAt[level] || mWriteThroughAt[level])
        {
            TLLM_LOG_INFO(
                "[kvCacheManagerV2] disk tier %d: keep_copy_after_restore=%d write_through=%d "
                "protected_copy_fraction=%g",
                static_cast<int>(level.value()), static_cast<int>(mKeepCopyAfterRestoreAt[level]),
                static_cast<int>(mWriteThroughAt[level]), static_cast<double>(mProtectedCopyFractionAt[level]));
        }
    }
}

SharedPtr<RemoteMemClient> StorageManager::remoteClientFor(CacheTierConfig const& tier)
{
    auto const* remote = std::get_if<RemoteCacheTierConfig>(&tier);
    if (remote == nullptr)
    {
        return nullptr;
    }
    RemoteMemOptions options{remote->path, remote->waitTimeoutMs, remote->expectedDonors};
    for (auto const& client : mRemoteClients)
    {
        if (client->directory() != remote->path)
        {
            continue;
        }
        TLLM_CHECK_WITH_INFO(client->options() == options && client->transferConfig() == remote->transfer,
            "remote cache levels sharing rendezvous directory '%s' must use the same timeouts, transport, devices and "
            "expected donors",
            remote->path.c_str());
        return client;
    }
    return mRemoteClients.emplace_back(makeRemoteMemClient(std::move(options), remote->transfer));
}

RemoteCacheLevelStorage* StorageManager::remoteLevel(CacheLevel level) const
{
    auto const& lvl = mLevels.at(level);
    return lvl.cacheTier == CacheTier::REMOTE ? static_cast<RemoteCacheLevelStorage*>(lvl.storage.get()) : nullptr;
}

bool StorageManager::anyRemoteLevelFailed() const noexcept
{
    return std::any_of(
        mRemoteClients.begin(), mRemoteClients.end(), [](auto const& client) { return client->failed(); });
}

size_t StorageManager::numFailedRemoteClients() const noexcept
{
    return static_cast<size_t>(std::count_if(
        mRemoteClients.begin(), mRemoteClients.end(), [](auto const& client) { return client->failed(); }));
}

bool StorageManager::remoteLevelFailed(CacheLevel level) const
{
    // One flag per client while none has failed: this runs whenever a page is scheduled for eviction.
    if (std::none_of(mRemoteClients.begin(), mRemoteClients.end(), [](auto const& client) { return client->failed(); }))
    {
        return false;
    }
    auto const* const remote = remoteLevel(level);
    return remote != nullptr && remote->client().failed();
}

std::vector<SharedPtr<Page>> StorageManager::queuedPages(CacheLevel level, PageStatus status)
{
    auto& controller = mLevels.at(level).controller;
    std::vector<SharedPtr<Page>> pages;
    for (PoolGroupIndex pgIdx{0}; pgIdx < controller.numPoolGroups(); ++pgIdx)
    {
        auto next = controller.pageGenerator(pgIdx);
        while (auto const* page = next())
        {
            if ((*page)->status() == status)
            {
                pages.push_back(*page);
            }
        }
    }
    return pages;
}

void StorageManager::dropPagesOfFailedRemoteLevels()
{
    if (!mUnquiescedRemoteTransfers.empty())
    {
        reapQuiescedRemoteTransfers();
    }
    for (CacheLevel level{1}; level < numCacheLevels(); ++level)
    {
        if (!remoteLevelFailed(level))
        {
            continue;
        }
        auto const dropped = queuedPages(level, PageStatus::DROPPABLE);
        for (auto const& page : dropped)
        {
            excludeFromEviction(*page);
        }
        if (!dropped.empty())
        {
            TLLM_LOG_WARNING(
                "KVCM2 remote level %d failed: dropped its %zu cached page(s)", level.value(), dropped.size());
        }
        // `dropped` holds the last references: the pages leave the radix tree and release their slots here.
    }
    // Trailing failed levels take no pages, so held pages above them stop being evictable: they would only fall
    // into a level that cannot hold them.
    CacheLevel lastLive = numCacheLevels() - 1;
    while (lastLive > kHotLevel && remoteLevelFailed(lastLive))
    {
        --lastLive;
    }
    if (lastLive < mLastLiveLevel)
    {
        mLastLiveLevel = lastLive;
        for (auto const& page : queuedPages(lastLive, PageStatus::HELD))
        {
            excludeFromEviction(*page);
        }
    }
}

StorageManager::~StorageManager()
{
    KVCM2_POISON_ON_EXCEPT([this]() { destroy(); });
}

void StorageManager::destroy()
{
    // A transfer that missed its deadline gets one more transfer timeout to quiesce. One that still has not may yet
    // access its staging lease, its chunks and its client's registrations: all of them are leaked rather than reused.
    bool leakStaging = false;
    for (auto& transfer : mUnquiescedRemoteTransfers)
    {
        auto const deadline = remoteDeadlineAfter(transfer.client->transferConfig().timeoutMs);
        while (!transfer.attempt->quiesced() && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!transfer.attempt->quiesced())
        {
            TLLM_LOG_ERROR(
                "KVCM2 remote level %s: a transfer never quiesced; its staging, claims and transport are leaked",
                transfer.client->directory().c_str());
            transfer.client->abandon();
            static_cast<void>(transfer.attempt.release());
            static_cast<void>(transfer.staging.release());
            leakStaging = true;
        }
    }
    mUnquiescedRemoteTransfers.clear();
    // Before the levels go: the last level holding a client destroys it, and the staging must be deregistered first,
    // after every stream-ordered transfer that uses it has finished.
    for (auto const& client : mRemoteClients)
    {
        client->drain();
        client->deregisterLocal(mPageStagingManager->baseAddress(), mPageStagingManager->totalSize());
    }
    mRemoteClients.clear();
    releaseAllRetainedCopies(/*level=*/std::nullopt);
    for (auto& lvl : mLevels)
    {
        TLLM_CHECK_DEBUG(lvl.storage);
        lvl.storage->destroy();
    }
    mLevels.clear();
    mGpuPhysMemAllocator.reset();

    mIndexStagingManager.reset();
    mCopyEngine.reset();
    if (leakStaging)
    {
        static_cast<void>(mPageStagingManager.release());
    }
    mPageStagingManager.reset();
}

size_t StorageManager::reapQuiescedRemoteTransfers() noexcept
{
    auto const quiesced = [](UnquiescedRemoteTransfer& transfer) { return transfer.attempt->quiesced(); };
    mUnquiescedRemoteTransfers.erase(
        std::remove_if(mUnquiescedRemoteTransfers.begin(), mUnquiescedRemoteTransfers.end(), quiesced),
        mUnquiescedRemoteTransfers.end());
    return mUnquiescedRemoteTransfers.size();
}

void StorageManager::runRemoteTransfer(RemoteMemClient& client, bool write, std::vector<RemoteIo> const& ios,
    size_t bytes, CUstream stream, std::unique_ptr<StagingBuffer>& staging)
{
    auto attempt = client.submit(write, ios, bytes, stream);
    if (client.streamOrdered())
    {
        // Concluded at once; the staging lease, retired on the stream, is reused only after the copies have run.
        auto const outcome = attempt->poll();
        if (!outcome.has_value() || !outcome->ok())
        {
            client.fail(outcome.has_value() ? outcome->fault : "KVCM2 remote level: a transfer did not conclude");
        }
        return;
    }
    int64_t const timeoutMs = client.transferConfig().timeoutMs;
    auto const outcome = attempt->wait(timeoutMs);
    if (!attempt->quiesced())
    {
        mUnquiescedRemoteTransfers.push_back({std::move(attempt), std::move(staging), &client});
    }
    if (!outcome.has_value())
    {
        client.fail("KVCM2 remote level: a " + std::string(write ? "WRITE" : "READ") + " of "
            + std::to_string(ios.size()) + " page(s) x " + std::to_string(bytes) + " B timed out after "
            + std::to_string(timeoutMs) + " ms");
    }
    if (!outcome->ok())
    {
        client.fail(outcome->fault);
    }
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
    return mLevels.at(level).storage->poolGroup(pgIdx).slotAddress(slotId, poolIdx);
}

bool StorageManager::encodeColdPages(
    LayerGroupId layerGroup, void* dstBase, PageIndexPair const* pageIndices, size_t numPages, CUstream stream)
{
    // A hot pool group with bound pools (copy-based rebalancing) translates its slot ids per pool.
    PoolGroupIndex const hotPgIdx = getPoolGroupIndex(kHotLevel, layerGroup);
    if (poolGroup(kHotLevel, hotPgIdx).hasBoundPools())
    {
        return submitBoundPoolColdPageCodec(
            /*encode=*/true, layerGroup, hotPgIdx, dstBase, pageIndices, numPages, stream);
    }
    return submitColdPageCodec(mPageIndexLocations.at(layerGroup), pageIndices, numPages, stream,
        [this, layerGroup, dstBase](PageIndexPair const* submitted, size_t submittedPages, CUstream submittedStream)
        {
            return mColdPageCodec->encode(
                layerGroup, dstBase, submitted, submittedPages, reinterpret_cast<cudaStream_t>(submittedStream));
        });
}

bool StorageManager::decodeColdPages(
    LayerGroupId layerGroup, void const* srcBase, PageIndexPair const* pageIndices, size_t numPages, CUstream stream)
{
    PoolGroupIndex const hotPgIdx = getPoolGroupIndex(kHotLevel, layerGroup);
    if (poolGroup(kHotLevel, hotPgIdx).hasBoundPools())
    {
        return submitBoundPoolColdPageCodec(
            /*encode=*/false, layerGroup, hotPgIdx, srcBase, pageIndices, numPages, stream);
    }
    return submitColdPageCodec(mPageIndexLocations.at(layerGroup), pageIndices, numPages, stream,
        [this, layerGroup, srcBase](PageIndexPair const* submitted, size_t submittedPages, CUstream submittedStream)
        {
            return mColdPageCodec->decode(
                layerGroup, srcBase, submitted, submittedPages, reinterpret_cast<cudaStream_t>(submittedStream));
        });
}

void StorageManager::submitMigrationBatch(CacheLevel dstLevel, CacheLevel srcLevel, LayerGroupId batchingLayerGroupId,
    PageIndexPair const* pageIndices, size_t numPages, CUstream stream, bool move)
{
    TLLM_CHECK_DEBUG(pageIndices != nullptr && numPages > 0);
    PoolGroupIndex const srcPgIdx = getPoolGroupIndex(srcLevel, batchingLayerGroupId);
    PoolGroupIndex const dstPgIdx = getPoolGroupIndex(dstLevel, batchingLayerGroupId);
    bool const srcIsHot = srcLevel == kHotLevel;
    bool const dstIsHot = dstLevel == kHotLevel;
    auto& srcPoolGroup = poolGroup(srcLevel, srcPgIdx);
    auto& dstPoolGroup = poolGroup(dstLevel, dstPgIdx);

    if (srcIsHot == dstIsHot)
    {
        if (!srcIsHot && (remoteLevel(srcLevel) != nullptr || remoteLevel(dstLevel) != nullptr))
        {
            submitRemoteColdMigration(dstLevel, srcLevel, dstPgIdx, srcPgIdx, pageIndices, numPages, stream);
            return;
        }
        auto const srcSizes = slotSize(srcLevel, srcPgIdx);
        TLLM_CHECK_DEBUG(srcSizes == slotSize(dstLevel, dstPgIdx));
        // A block moving within an arena group leaves its bound pools' data where it is: the
        // destination slot id takes over the source's bindings.
        bool const keepBoundPools = move && &srcPoolGroup == &dstPoolGroup && srcPoolGroup.hasBoundPools();
        if (keepBoundPools)
        {
            for (size_t index = 0; index < numPages; ++index)
            {
                srcPoolGroup.swapBoundPoolBindings(toSlotId(pageIndices[index].dst), toSlotId(pageIndices[index].src));
            }
        }
        for (PoolIndex poolIdx{0}; poolIdx < srcSizes.size(); ++poolIdx)
        {
            if (keepBoundPools && srcPoolGroup.isBoundPool(poolIdx))
                continue;
            std::vector<CopyTask> tasks;
            tasks.reserve(numPages);
            for (size_t index = 0; index < numPages; ++index)
            {
                tasks.push_back({dstPoolGroup.slotAddress(toSlotId(pageIndices[index].dst), poolIdx),
                    srcPoolGroup.slotAddress(toSlotId(pageIndices[index].src), poolIdx)});
            }
            // Per pool, not per group: a GPU-tier group holds device-backed pools beside
            // host-backed sparse ones, and the copy engine picks its implementation from
            // the medium. Taking the group's tier would run a device-to-device kernel over
            // host pointers -- legal, since the host memory is registered, but tuned for a
            // bandwidth it does not have.
            mCopyEngine->transfer(poolMedium(dstLevel, dstPgIdx, poolIdx), poolMedium(srcLevel, srcPgIdx, poolIdx),
                srcSizes.at(poolIdx), tasks, stream);
        }
        return;
    }

    if (remoteLevel(srcIsHot ? dstLevel : srcLevel) != nullptr)
    {
        submitRemoteHotMigration(dstLevel, srcLevel, batchingLayerGroupId, pageIndices, numPages, stream);
        return;
    }

    CacheLevel const coldLevel = srcIsHot ? dstLevel : srcLevel;
    PoolGroupIndex const coldPgIdx = srcIsHot ? dstPgIdx : srcPgIdx;
    // Only a GPU-tier pool is ever host-backed, so for a cold level cacheTier() and
    // poolMedium() agree.
    CacheTier const coldTier = cacheTier(coldLevel);
    size_t const coldPageBytes = slotSize(coldLevel, coldPgIdx).at(PoolIndex{0});
    auto encodeBatch = [this, batchingLayerGroupId](
                           void* dstBasePtr, PageIndexPair const* indices, size_t count, CUstream batchStream)
    { return encodeColdPages(batchingLayerGroupId, dstBasePtr, indices, count, batchStream); };
    auto decodeBatch = [this, batchingLayerGroupId](
                           void const* srcBasePtr, PageIndexPair const* indices, size_t count, CUstream batchStream)
    { return decodeColdPages(batchingLayerGroupId, srcBasePtr, indices, count, batchStream); };

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
        // 16-byte alignment: the cold-page codec's copy path uses 16-byte vector accesses and
        // requires its base pointer to be 16-aligned. Nothing here needs finer granularity, so
        // this is a floor rather than a computed requirement.
        auto const staging = mPageStagingManager->acquire(coldPageBytes, maxStagingBytes, coldPageBytes, 16, stream);
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
                tasks.push_back({dstPoolGroup.slotAddress(toSlotId(pageIndices[offset + index].dst), PoolIndex{0}),
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
                    srcPoolGroup.slotAddress(toSlotId(pageIndices[offset + index].src), PoolIndex{0})});
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

    trimRetainedPageIndices(stagingPageIndices);
}

void StorageManager::submitRemoteHotMigration(CacheLevel dstLevel, CacheLevel srcLevel,
    LayerGroupId batchingLayerGroupId, PageIndexPair const* pageIndices, size_t numPages, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(mPageStagingManager != nullptr, "A remote cache level needs page staging");
    bool const srcIsHot = srcLevel == kHotLevel;
    CacheLevel const coldLevel = srcIsHot ? dstLevel : srcLevel;
    PoolGroupIndex const coldPgIdx = getPoolGroupIndex(coldLevel, batchingLayerGroupId);
    size_t const coldPageBytes = slotSize(coldLevel, coldPgIdx).at(PoolIndex{0});
    auto encodeBatch = [this, batchingLayerGroupId](
                           void* dstBasePtr, PageIndexPair const* indices, size_t count, CUstream batchStream)
    { return encodeColdPages(batchingLayerGroupId, dstBasePtr, indices, count, batchStream); };
    auto decodeBatch = [this, batchingLayerGroupId](
                           void const* srcBasePtr, PageIndexPair const* indices, size_t count, CUstream batchStream)
    { return decodeColdPages(batchingLayerGroupId, srcBasePtr, indices, count, batchStream); };

    // Kept across calls, as the disk path's: a migration batch reuses the previous one's allocation.
    thread_local std::vector<PageIndexPair> stagingPageIndices;
    size_t remaining = numPages;
    size_t offset = 0;
    // Remote level: the codec runs against the page staging buffer, and the client moves the bytes between
    // staging and the remote slots, on the stream or on the host (a failure then throws before _batchedMigrate
    // points any page at its new slot).
    RemoteRange& range = remoteLevel(coldLevel)->remoteRange(coldPgIdx, PoolIndex{0});
    RemoteMemClient& client = range.client();
    // Before any staging lease: a failed level's unquiesced transfer may hold all of it, and the lease would then fail
    // with a plain exception instead of the RemoteMemError that costs only this request.
    client.checkUsable();
    bool const streamOrdered = client.streamOrdered();
    std::vector<RemoteIo> ios;
    while (remaining > 0)
    {
        size_t const maxStagingBytes = remaining > std::numeric_limits<size_t>::max() / coldPageBytes
            ? std::numeric_limits<size_t>::max()
            : coldPageBytes * remaining;
        ios.clear();
        if (srcIsHot)
        {
            // Hot -> remote: encode into staging on the stream, then WRITE once the encode is done.
            auto staging = std::make_unique<StagingBuffer>(*mPageStagingManager, coldPageBytes, maxStagingBytes,
                coldPageBytes, 16, std::optional<CUstream>{stream});
            MemAddress const base = staging->address();
            size_t const batchSize = std::min(remaining, staging->size() / coldPageBytes);
            stagingPageIndices.resize(batchSize);
            for (size_t index = 0; index < batchSize; ++index)
            {
                stagingPageIndices[index] = PageIndexPair{static_cast<int32_t>(index), pageIndices[offset + index].src};
            }
            TLLM_CHECK_WITH_INFO(
                encodeBatch(reinterpret_cast<void*>(base), stagingPageIndices.data(), batchSize, stream),
                "Cold-page codec rejected a migration batch");
            for (size_t index = 0; index < batchSize; ++index)
            {
                auto const [donor, remoteAddr] = range.locate(toSlotId(pageIndices[offset + index].dst));
                ios.push_back(RemoteIo{base + index * coldPageBytes, donor, remoteAddr});
            }
            // Stream-ordered, the WRITE starts when the stream has encoded the batch and the lease retires after
            // it; on the host, it waits for the encode.
            if (!streamOrdered)
            {
                cuCheck(cuStreamSynchronize(stream));
            }
            runRemoteTransfer(client, /*write=*/true, ios, coldPageBytes, stream, staging);
            offset += batchSize;
            remaining -= batchSize;
        }
        else
        {
            // Remote -> hot: READ into staging, then decode on the stream. Stream-ordered, the lease is the
            // stream's from the start and the READ begins when the stream reaches it; on the host, the lease is
            // host-synchronous and moves to the stream only after the READ has finished.
            auto staging = std::make_unique<StagingBuffer>(*mPageStagingManager, coldPageBytes, maxStagingBytes,
                coldPageBytes, 16, streamOrdered ? std::optional<CUstream>{stream} : std::nullopt);
            MemAddress const base = staging->address();
            size_t const batchSize = std::min(remaining, staging->size() / coldPageBytes);
            for (size_t index = 0; index < batchSize; ++index)
            {
                auto const [donor, remoteAddr] = range.locate(toSlotId(pageIndices[offset + index].src));
                ios.push_back(RemoteIo{base + index * coldPageBytes, donor, remoteAddr});
            }
            runRemoteTransfer(client, /*write=*/false, ios, coldPageBytes, stream, staging);
            if (!streamOrdered)
            {
                staging->setStream(stream);
            }
            stagingPageIndices.resize(batchSize);
            for (size_t index = 0; index < batchSize; ++index)
            {
                stagingPageIndices[index] = PageIndexPair{pageIndices[offset + index].dst, static_cast<int32_t>(index)};
            }
            TLLM_CHECK_WITH_INFO(
                decodeBatch(reinterpret_cast<void const*>(base), stagingPageIndices.data(), batchSize, stream),
                "Cold-page codec rejected a migration batch");
            offset += batchSize;
            remaining -= batchSize;
        }
    }
    trimRetainedPageIndices(stagingPageIndices);
}

void StorageManager::submitRemoteColdMigration(CacheLevel dstLevel, CacheLevel srcLevel, PoolGroupIndex dstPgIdx,
    PoolGroupIndex srcPgIdx, PageIndexPair const* pageIndices, size_t numPages, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(mPageStagingManager != nullptr, "A remote cache level needs page staging");
    auto* const srcRemote = remoteLevel(srcLevel);
    auto* const dstRemote = remoteLevel(dstLevel);
    auto const sizes = slotSize(srcLevel, srcPgIdx);
    TLLM_CHECK_DEBUG(sizes == slotSize(dstLevel, dstPgIdx));
    auto& srcGroup = poolGroup(srcLevel, srcPgIdx);
    auto& dstGroup = poolGroup(dstLevel, dstPgIdx);
    // Cold levels hold one pool per group (the codec's single cold page), but nothing here depends on it.
    for (PoolIndex poolIdx{0}; poolIdx < sizes.size(); ++poolIdx)
    {
        size_t const pageBytes = sizes.at(poolIdx);
        RemoteRange* const srcRange = srcRemote != nullptr ? &srcRemote->remoteRange(srcPgIdx, poolIdx) : nullptr;
        RemoteRange* const dstRange = dstRemote != nullptr ? &dstRemote->remoteRange(dstPgIdx, poolIdx) : nullptr;
        // The local side is copied to or from staging by the GPU's batched copy kernel, which needs memory the GPU
        // can address (the host level's pinned pages are).
        TLLM_CHECK_WITH_INFO(srcRange != nullptr || isGpuAccessibleMemory(poolMedium(srcLevel, srcPgIdx, poolIdx)),
            "Remote level: unsupported source medium");
        TLLM_CHECK_WITH_INFO(dstRange != nullptr || isGpuAccessibleMemory(poolMedium(dstLevel, dstPgIdx, poolIdx)),
            "Remote level: unsupported destination medium");
        // Stream-ordered only if every remote side's transfers are: the legs (and the local copies) then follow one
        // another on the stream.
        bool const streamOrdered = (srcRange == nullptr || srcRange->client().streamOrdered())
            && (dstRange == nullptr || dstRange->client().streamOrdered());
        // Before any staging lease, as in submitRemoteHotMigration.
        for (RemoteRange const* remoteRange : {srcRange, dstRange})
        {
            if (remoteRange != nullptr)
            {
                remoteRange->client().checkUsable();
            }
        }
        std::vector<RemoteIo> ios;
        std::vector<Task<MemAddress, MemAddress>> tasks;
        size_t remaining = numPages;
        size_t offset = 0;
        while (remaining > 0)
        {
            size_t const maxStagingBytes = remaining > std::numeric_limits<size_t>::max() / pageBytes
                ? std::numeric_limits<size_t>::max()
                : pageBytes * remaining;
            // Stream-ordered: the stream's lease, so every leg below is ordered after the range's previous users.
            // On the host: a host-synchronous lease, nothing else is using this range of staging once acquire returns.
            auto staging = std::make_unique<StagingBuffer>(*mPageStagingManager, pageBytes, maxStagingBytes, pageBytes,
                16, streamOrdered ? std::optional<CUstream>{stream} : std::nullopt);
            size_t const batchSize = std::min(remaining, staging->size() / pageBytes);
            MemAddress const base = staging->address();

            // 1. Source pages into staging.
            if (srcRange != nullptr)
            {
                ios.clear();
                for (size_t index = 0; index < batchSize; ++index)
                {
                    auto const [donor, remoteAddr] = srcRange->locate(toSlotId(pageIndices[offset + index].src));
                    ios.push_back(RemoteIo{base + index * pageBytes, donor, remoteAddr});
                }
                runRemoteTransfer(srcRange->client(), /*write=*/false, ios, pageBytes, stream, staging);
            }
            else
            {
                tasks.clear();
                for (size_t index = 0; index < batchSize; ++index)
                {
                    tasks.push_back({base + index * pageBytes,
                        std::get<MemAddress>(
                            srcGroup.slotAddress(toSlotId(pageIndices[offset + index].src)).at(poolIdx))});
                }
                // The stream already waits for the source pages' writers; the copy kernel reads host memory over C2C.
                staging->setStream(stream);
                cuCheck(copyHostToDevice(tasks, static_cast<ssize_t>(pageBytes), stream));
                if (!streamOrdered)
                {
                    cuCheck(cuStreamSynchronize(stream));
                }
            }

            // 2. Staging into the destination pages.
            if (dstRange != nullptr)
            {
                ios.clear();
                for (size_t index = 0; index < batchSize; ++index)
                {
                    auto const [donor, remoteAddr] = dstRange->locate(toSlotId(pageIndices[offset + index].dst));
                    ios.push_back(RemoteIo{base + index * pageBytes, donor, remoteAddr});
                }
                // Stream-ordered, starts when the stream has the source bytes in staging; on the host, they are there.
                runRemoteTransfer(dstRange->client(), /*write=*/true, ios, pageBytes, stream, staging);
            }
            else
            {
                tasks.clear();
                for (size_t index = 0; index < batchSize; ++index)
                {
                    tasks.push_back({std::get<MemAddress>(
                                         dstGroup.slotAddress(toSlotId(pageIndices[offset + index].dst)).at(poolIdx)),
                        base + index * pageBytes});
                }
                // On the host: the READ above has finished, so handing the lease to the stream needs no
                // synchronization. Stream-ordered: the lease is the stream's already, and the copy waits for the READ.
                staging->setStream(stream);
                cuCheck(copyHostToDevice(tasks, static_cast<ssize_t>(pageBytes), stream));
            }
            offset += batchSize;
            remaining -= batchSize;
        }
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

CacheTier StorageManager::poolMedium(CacheLevel level, PoolGroupIndex pgIdx, PoolIndex poolIdx) const
{
    return mLevels.at(level).storage->poolGroup(pgIdx).poolMedium(poolIdx);
}

size_t StorageManager::getNumBytes(CacheLevel level, CacheTier medium) const
{
    return mLevels.at(level).storage->roundedNumBytes(medium);
}

std::optional<CacheTier> StorageManager::sparseBackingMedium() const
{
    return mLevels.at(kHotLevel).storage->secondaryMedium();
}

void StorageManager::releaseSlot(LifeCycleId lc, CacheLevel level, Slot slot)
{
    PoolGroupIndex pg = getPoolGroupIndex(level, lc);
    mLevels.at(level).storage->release(pg, std::move(slot));
}

StorageManager::DiskCopyCounters StorageManager::diskCopyCounters() const noexcept
{
    DiskCopyCounters counters = mDiskCopyCounters;
    counters.retained = mRetainedCopies.size();
    return counters;
}

SlotCount StorageManager::numRetainedCopies(CacheLevel level, std::optional<PoolGroupIndex> pgIdx) const
{
    auto const& counts = mNumRetainedCopies.at(level);
    if (pgIdx.has_value())
    {
        return counts.at(*pgIdx);
    }
    return std::accumulate(counts.begin(), counts.end(), SlotCount{0});
}

SlotCount StorageManager::protectedCopyShare(CacheLevel level, PoolGroupIndex pgIdx) const
{
    float const fraction = mProtectedCopyFractionAt.at(level);
    SlotCount const slots = numSlots(pgIdx, level);
    return fraction >= 1.0F ? slots
                            : static_cast<SlotCount>(static_cast<double>(fraction) * static_cast<double>(slots));
}

void StorageManager::addRetainedCopy(
    Page& page, CacheLevel level, PoolGroupIndex pgIdx, Slot slot, std::shared_ptr<BackgroundDiskWrites const> write)
{
    // With several disk levels a page may still retain a copy at another level; keep only the newest.
    if (auto it = mRetainedByPage.find(&page); it != mRetainedByPage.end())
    {
        eraseRetainedCopy(it->second);
        ++mDiskCopyCounters.reclaimed;
    }
    mRetainedCopies.push_back(RetainedCopy{&page, level, pgIdx, std::move(slot), false, std::move(write)});
    try
    {
        mRetainedByPage.emplace(&page, std::prev(mRetainedCopies.end()));
    }
    catch (...)
    {
        // Without its map entry the node would outlive the page; give the slot back instead.
        mLevels.at(level).storage->release(pgIdx, std::move(mRetainedCopies.back().slot));
        mRetainedCopies.pop_back();
        throw;
    }
    ++mNumRetainedCopies.at(level).at(pgIdx);
    ++mDiskCopyCounters.created;
}

bool StorageManager::validateRetainedCopy(Page const& page, CacheLevel level)
{
    auto it = mRetainedByPage.find(&page);
    if (it == mRetainedByPage.end() || it->second->level != level)
    {
        return false;
    }
    auto& copy = *it->second;
    // A copy whose write is still running can stand in for the page: whoever takes the slot back waits on its ready
    // event, as for any slot. Once the write has finished, the copy is usable only if it did not fail. A failure is
    // recorded before the transfer publishes completion, so a finished event means the record is in place.
    bool const finished = copy.slot.readyEvent.isClosed() || copy.slot.readyEvent.queryComplete();
    bool usable = true;
    if (finished)
    {
        copy.write.reset();
        Address const address = slotAddress(copy.level, copy.pgIdx, copy.slot.slotId(), PoolIndex{0});
        if (auto const* disk = std::get_if<DiskAddress>(&address); disk && consumeFailedDiskWrite(disk->fd, disk->pos))
        {
            TLLM_LOG_WARNING("[kvCacheManagerV2] discarding a disk copy whose write failed; the page is written again");
            usable = false;
        }
    }
    if (!usable)
    {
        eraseRetainedCopy(it->second);
        ++mDiskCopyCounters.reclaimed;
    }
    return usable;
}

std::optional<Slot> StorageManager::takeRetainedCopy(Page const& page, CacheLevel level)
{
    auto it = mRetainedByPage.find(&page);
    if (it == mRetainedByPage.end() || it->second->level != level)
    {
        return std::nullopt;
    }
    // Check again even if reserved: the write may have finished, and failed, since. A failed copy's slot goes back to
    // the pool, so the caller's room check still holds.
    if (!validateRetainedCopy(page, level))
    {
        return std::nullopt;
    }
    it = mRetainedByPage.find(&page);
    auto entry = it->second;
    // The eviction now depends on this write, so it must not wait behind other background writes.
    if (promoteIfPending(*entry))
    {
        ++mDiskCopyCounters.reusedPending;
    }
    Slot slot = std::move(entry->slot);
    --mNumRetainedCopies.at(level).at(entry->pgIdx);
    mRetainedByPage.erase(it);
    mRetainedCopies.erase(entry);
    return slot;
}

bool StorageManager::reserveRetainedCopy(Page const& page, CacheLevel level, bool reserve) noexcept
{
    auto it = mRetainedByPage.find(&page);
    if (it == mRetainedByPage.end() || it->second->level != level)
    {
        return false;
    }
    it->second->reserved = reserve;
    return true;
}

bool StorageManager::promoteIfPending(RetainedCopy const& copy) const
{
    if (copy.slot.readyEvent.isClosed() || copy.slot.readyEvent.queryComplete())
    {
        return false;
    }
    if (copy.write)
    {
        promoteBackgroundDiskWrites(*copy.write);
    }
    return true;
}

void StorageManager::eraseRetainedCopy(RetainedCopyList::iterator iter)
{
    // Whoever reuses the released disk slot (or the page's host slot, below) waits for a write still in flight, so it
    // must not wait behind other background writes.
    promoteIfPending(*iter);
    // A write-through may still be reading the page's host slot, and its completion is tracked only here. Move it
    // onto the page so whoever later releases the host slot waits for it.
    if (mDiskWriteThrough)
    {
        Page& page = *iter->page;
        if (page.cacheLevel != kHotLevel && cacheTier(page.cacheLevel) == CacheTier::HOST_MEM)
        {
            if (auto pending = pendingDiskWrite(page); !pending.isClosed())
            {
                std::vector<CachedCudaEvent> events{page.readyEvent, std::move(pending)};
                page.readyEvent = mergeEvents(events);
            }
        }
    }
    mLevels.at(iter->level).storage->release(iter->pgIdx, std::move(iter->slot));
    --mNumRetainedCopies.at(iter->level).at(iter->pgIdx);
    mRetainedByPage.erase(iter->page);
    mRetainedCopies.erase(iter);
}

void StorageManager::releaseRetainedCopies(CacheLevel level, PoolGroupIndex pgIdx, SlotCount count)
{
    for (auto it = mRetainedCopies.begin(); it != mRetainedCopies.end() && count > 0;)
    {
        auto next = std::next(it);
        if (it->level == level && it->pgIdx == pgIdx && !it->reserved)
        {
            eraseRetainedCopy(it);
            ++mDiskCopyCounters.reclaimed;
            --count;
        }
        it = next;
    }
}

void StorageManager::releaseAllRetainedCopies(std::optional<CacheLevel> level)
{
    for (auto it = mRetainedCopies.begin(); it != mRetainedCopies.end();)
    {
        auto next = std::next(it);
        if (!level.has_value() || it->level == *level)
        {
            eraseRetainedCopy(it);
            ++mDiskCopyCounters.reclaimed;
        }
        it = next;
    }
}

void StorageManager::onPageDestroyed(Page& page) noexcept
{
    if (mRetainedByPage.empty())
    {
        return;
    }
    KVCM2_POISON_ON_EXCEPT(
        [this, &page]()
        {
            auto it = mRetainedByPage.find(&page);
            if (it != mRetainedByPage.end())
            {
                eraseRetainedCopy(it->second);
                ++mDiskCopyCounters.dropped;
            }
        });
}

CachedCudaEvent StorageManager::pendingDiskWrite(Page const& page) const
{
    auto it = mRetainedByPage.find(&page);
    if (it != mRetainedByPage.end() && promoteIfPending(*it->second))
    {
        return it->second->slot.readyEvent;
    }
    return CachedCudaEvent::makeNull();
}

void StorageManager::writeThroughToDisk(CacheLevel hostLevel, std::vector<SharedPtr<Page>> const& pages)
{
    CacheLevel const diskLevel = hostLevel + 1;
    if (diskLevel >= numCacheLevels() || cacheTier(hostLevel) != CacheTier::HOST_MEM
        || cacheTier(diskLevel) != CacheTier::DISK || !mWriteThroughAt.at(diskLevel))
    {
        return;
    }
    std::map<LayerGroupId, std::vector<SharedPtr<Page>>> batches;
    for (auto const& page : pages)
    {
        if (page->cacheLevel == hostLevel && hasFinalContent(*page)
            && mRetainedByPage.find(page.get()) == mRetainedByPage.end())
        {
            batches[getMigrationBatchingLayerGroupId(diskLevel, hostLevel, page->lifeCycle)].push_back(page);
        }
    }
    for (auto& [batchingLayerGroupId, batch] : batches)
    {
        PoolGroupIndex const pgIdx = getPoolGroupIndex(diskLevel, batchingLayerGroupId);
        // A write-through copy only saves a later write, so it never makes room: it takes free disk slots only,
        // stays within the background write budget so it cannot build up a backlog, and stays within the tier's
        // protected copy share, since a copy beyond it would be the first to go when the tier runs short.
        SlotCount const free = poolGroup(diskLevel, pgIdx).numFreeSlots();
        SlotCount const share = protectedCopyShare(diskLevel, pgIdx);
        SlotCount const copies = numRetainedCopies(diskLevel, pgIdx);
        size_t const pageBytes = slotSize(diskLevel, pgIdx).at(PoolIndex{0});
        size_t const take = std::min({batch.size(), static_cast<size_t>(std::max<SlotCount>(free, 0)),
            static_cast<size_t>(share > copies ? share - copies : SlotCount{0}),
            pageBytes == 0 ? batch.size() : backgroundDiskWriteBudget() / pageBytes});
        mDiskCopyCounters.writeThroughSkipped += batch.size() - take;
        if (take == 0)
        {
            continue;
        }
        batch.resize(take);
        // Copy without moving. The page's ready event is left alone, so host -> GPU restores don't wait for this
        // write; its completion lives on the retained slot, and releases of the host slot merge it in
        // (pendingDiskWrite).
        // The required GPU -> host eviction has already succeeded, so a failure here only costs the early copy:
        // give back the disk slots not yet retained and let these pages be written on their host -> disk eviction.
        std::optional<std::vector<Slot>> slots;
        size_t added = 0;
        try
        {
            std::shared_ptr<BackgroundDiskWrites const> writes;
            {
                BackgroundDiskWriteScope background;
                slots = _batchedMigrate(diskLevel, hostLevel, batch, /*updateSrc=*/false, MigrationRecorder{});
                writes = background.takeWrites();
            }
            TLLM_CHECK(slots.has_value() && slots->size() == take);
            for (; added < take; ++added)
            {
                addRetainedCopy(*batch[added], diskLevel, pgIdx, std::move(slots->at(added)), writes);
            }
        }
        catch (std::exception const& e)
        {
            if (slots.has_value())
            {
                for (auto& slot : *slots)
                {
                    if (slot.hasValidSlot())
                    {
                        poolGroup(diskLevel, pgIdx).release(std::move(slot));
                    }
                }
            }
            mDiskCopyCounters.writeThroughSkipped += take - added;
            TLLM_LOG_WARNING("[KVCacheManagerV2] Disk write-through of %zu pages skipped: %s", take - added, e.what());
        }
        mDiskCopyCounters.writeThroughPages += added;
    }
}

// ---------------------------------------------------------------------------
// isEvictable
// ---------------------------------------------------------------------------

bool StorageManager::isEvictable(Page const& page, std::optional<CacheLevel> level) const noexcept
{
    PageStatus s = page.status();
    CacheLevel lvl = level.value_or(page.cacheLevel);
    if (s == PageStatus::DROPPABLE && page.isCommitted())
    {
        // A migration batch can outlive checkpoint invalidation. Do not put
        // detached bundle members back in an LRU that would retain their slots.
        auto const& committed = static_cast<CommittedPage const&>(page);
        return committed.block && !committed.block->isOrphan() && committed.block->holdsPage(committed);
    }
    // A held page cannot drop, so it is evictable only with a level below it that still takes pages.
    return s == PageStatus::HELD && lvl < mLastLiveLevel;
}

// ---------------------------------------------------------------------------
// scheduleForEviction / excludeFromEviction
// ---------------------------------------------------------------------------

void StorageManager::scheduleForEviction(Page& page)
{
    // A droppable page on a failed remote level is left unscheduled, so it drops with its last reference.
    if (isEvictable(page) && !(page.status() == PageStatus::DROPPABLE && remoteLevelFailed(page.cacheLevel)))
    {
        mLevels.at(page.cacheLevel).controller.scheduleForEviction(page);
        scheduleRetentionExpiry(page);
    }
}

void StorageManager::updatePriority(Page& page, Priority priority)
{
    TLLM_CHECK_WITH_INFO(
        kPriorityMin <= priority && priority <= kPriorityMax, "Retention priority must be between 0 and 100");
    if (priority == page.mPriority)
    {
        return;
    }
    // Keep ownership across removal: a droppable page may be owned only by its queue.
    auto const keepAlive = page.sharedFromThis();
    bool const queued = page.scheduledForEviction();
    if (queued)
    {
        excludeFromEviction(page);
    }
    auto const oldPriority = page.mPriority;
    page.mPriority = priority;
    if (queued)
    {
        mLevels.at(page.cacheLevel).controller.scheduleForEviction(page);
    }
    if (mEventSink && page.isCommitted())
    {
        auto const& committed = static_cast<CommittedPage const&>(page);
        if (committed.block != nullptr && committed.block->holdsPage(committed))
        {
            mEventSink->addPriorityUpdated(committed.block->key, oldPriority, priority, page.lifeCycle);
        }
    }
}

void StorageManager::cancelRetentionExpiry(Page& page)
{
    if (page.mRetentionExpiry.has_value())
    {
        mRetentionExpiries.erase(*page.mRetentionExpiry);
        page.mRetentionExpiry.reset();
    }
}

void StorageManager::scheduleRetentionExpiry(Page& page)
{
    if (page.status() == PageStatus::DROPPABLE && page.mRetentionDuration.has_value()
        && page.priority() != kPriorityDefault && !page.mRetentionExpiry.has_value())
    {
        page.mRetentionExpiry = mRetentionExpiries.emplace(
            std::chrono::steady_clock::now() + *page.mRetentionDuration, page.sharedFromThis());
    }
}

void StorageManager::refreshRetention()
{
    auto const now = std::chrono::steady_clock::now();
    while (!mRetentionExpiries.empty() && mRetentionExpiries.begin()->first <= now)
    {
        auto const page = mRetentionExpiries.begin()->second.lock();
        mRetentionExpiries.erase(mRetentionExpiries.begin());
        if (page)
        {
            page->mRetentionExpiry.reset();
            // A prefetch may hold storage without claiming or extending its retention interval.
            updatePriority(*page, kPriorityDefault);
        }
    }
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
    refreshRetention();
    dropPagesOfFailedRemoteLevels();
    for (auto failedBefore = numFailedRemoteClients();; failedBefore = numFailedRemoteClients())
    {
        try
        {
            _prepareFreeSlots(level, requirements, migrationRecorder, dropRecorder);
            return;
        }
        catch (RemoteMemError const&)
        {
            // An eviction WRITE failed the level before pointing any page at it, and the rollback guards put the
            // evicted pages back (kept alive through the unwinding by _prepareFreeSlots' keepAlive): retire the level
            // and evict past it, instead of failing whoever needed the room.
            if (numFailedRemoteClients() == failedBefore)
            {
                throw;
            }
            dropPagesOfFailedRemoteLevels();
        }
    }
}

std::vector<CacheLevel> StorageManager::takeNewlyFailedRemoteLevels()
{
    std::vector<CacheLevel> levels;
    if (std::none_of(mRemoteClients.begin(), mRemoteClients.end(), [](auto const& client) { return client->failed(); }))
    {
        return levels;
    }
    dropPagesOfFailedRemoteLevels();
    for (CacheLevel level{1}; level < numCacheLevels(); ++level)
    {
        if (remoteLevelFailed(level)
            && std::find(mRetiredRemoteLevels.begin(), mRetiredRemoteLevels.end(), level) == mRetiredRemoteLevels.end())
        {
            mRetiredRemoteLevels.push_back(level);
            levels.push_back(level);
        }
    }
    return levels;
}

void StorageManager::_prepareFreeSlots(CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& requirements,
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
    std::vector<SharedPtr<Page>> keepAlive;
    _prepareFreeSlots(goals, level, fallenPages, keepAlive, migrationRecorder, dropRecorder);
}

void StorageManager::forceEvict(
    CacheLevel level, TypedVec<PoolGroupIndex, SlotCount> const& minNumPages, DropRecorder const& dropRecorder)
{
    // Declared before the rollback guard, so that the pages they hold outlive it on failure.
    PagesByLifeCycle fallen(numLifeCycles());
    std::vector<SharedPtr<Page>> keepAlive;
    auto evicted = mLevels.at(level).controller.evict(minNumPages);
    auto rescheduleEvictedPagesOnFailure = makeEvictionRollbackGuard(evicted);

    if (isLastLiveLevel(level))
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

    for (PoolGroupIndex pgIdx{0}; pgIdx < evicted.size(); ++pgIdx)
    {
        for (auto& sp : evicted.at(pgIdx))
        {
            LifeCycleId const lifeCycle = sp->lifeCycle;
            fallen.at(lifeCycle).push_back(std::move(sp));
        }
    }
    sortFallenPagesByPriority(fallen);
    _prepareFreeSlots(goals, nextLvl, fallen, keepAlive, MigrationRecorder{}, dropRecorder);
    rescheduleEvictedPagesOnFailure.cancel();
}

// ---------------------------------------------------------------------------
// _prepareFreeSlots (recursive)
// ---------------------------------------------------------------------------

void StorageManager::_prepareFreeSlots(TypedVec<CacheLevel, TypedVec<PoolGroupIndex, SlotCount>>& goals,
    CacheLevel lvlId, PagesByLifeCycle& fallenPages, std::vector<SharedPtr<Page>>& keepAlive,
    MigrationRecorder const& migrationRecorder, DropRecorder const& dropRecorder)
{
    TypedVec<LifeCycleId, std::vector<SharedPtr<Page>>> belowOffloadThreshold(numLifeCycles());
    auto restoreBelowThreshold = FuncGuard(
        [&]()
        {
            for (auto const& pages : belowOffloadThreshold)
            {
                for (auto iter = pages.rbegin(); iter != pages.rend(); ++iter)
                {
                    auto const& page = *iter;
                    if (page->hasValidSlot() && !page->scheduledForEviction() && isEvictable(*page))
                        mLevels.at(page->cacheLevel).controller.scheduleForEviction(*page, /*evictFirst=*/true);
                }
            }
        });
    for (auto& pages : fallenPages)
    {
        for (auto iter = pages.begin(); iter != pages.end();)
        {
            auto const& page = *iter;
            if (page->cacheLevel == kHotLevel && page->status() == PageStatus::DROPPABLE
                && page->priority() < mColdTierOffloadMinPriority)
            {
                belowOffloadThreshold.at(page->lifeCycle).push_back(std::move(*iter));
                iter = pages.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }
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
    // Pages stop at the last live level: below it are only failed remote levels, which take none. The held pages
    // falling to it are reserved there, as at the last level; nothing recurses into the failed levels.
    bool const isLast = isLastLiveLevel(lvlId);
    // A failed remote level counts as full with nothing to evict: pages fall past it, or drop after the last level.
    bool const isFailed = remoteLevelFailed(lvlId);
    auto const& grouping = poolGroupMapping(lvlId);

    auto countPages = [&](PoolGroupIndex pgIdx, PagesByLifeCycle const& pagesByLifeCycle)
    {
        auto const lifeCycles = grouping.lifeCycles(pgIdx);
        return std::accumulate(lifeCycles.begin(), lifeCycles.end(), SlotCount{0},
            [&](SlotCount count, LifeCycleId lifeCycle)
            { return count + slotCountValueFromSize(pagesByLifeCycle.at(lifeCycle).size()); });
    };

    // Keep-disk-copy: a fallen page whose copy is retained at this level takes that slot back, so it needs no free
    // slot here. Reserve those copies, then reclaim other retained copies only for what eviction cannot cover.
    PagesByLifeCycle keptPages(numLifeCycles());
    // Kept pages came out of the caller's fallenPages, which may hold their only references. If this frame fails
    // before accepting them, hand them back so the caller's rollback reschedules them instead of dropping them.
    bool keptPagesAccepted = false;
    auto unreserveKeptPages = FuncGuard(
        [&]()
        {
            for (LifeCycleId lifeCycle{0}; lifeCycle < keptPages.size(); ++lifeCycle)
            {
                for (auto& page : keptPages.at(lifeCycle))
                {
                    reserveRetainedCopy(*page, lvlId, /*reserve=*/false);
                    if (!keptPagesAccepted)
                    {
                        try
                        {
                            fallenPages.at(lifeCycle).push_back(std::move(page));
                        }
                        catch (std::exception const&)
                        {
                            // Out of memory while unwinding: the page is dropped, as it would be without the guard.
                        }
                    }
                }
            }
        });
    if (numRetainedCopies(lvlId) > 0)
    {
        for (LifeCycleId lifeCycle{0}; lifeCycle < fallenPages.size(); ++lifeCycle)
        {
            auto& pages = fallenPages.at(lifeCycle);
            PageQueue remaining;
            while (!pages.empty())
            {
                auto page = std::move(pages.front());
                pages.pop_front();
                if (validateRetainedCopy(*page, lvlId) && reserveRetainedCopy(*page, lvlId, /*reserve=*/true))
                {
                    keptPages.at(lifeCycle).push_back(std::move(page));
                }
                else
                {
                    remaining.push_back(std::move(page));
                }
            }
            pages = std::move(remaining);
        }
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(lvlId); ++pgIdx)
        {
            SlotCount const required = goals.at(lvlId).at(pgIdx) + countPages(pgIdx, fallenPages);
            // Copies beyond the tier's protected share give way first, before any page whose only copy is here.
            if (SlotCount const free = storage.numFreeSlots(pgIdx); required > free)
            {
                SlotCount const copies = numRetainedCopies(lvlId, pgIdx);
                if (SlotCount const share = protectedCopyShare(lvlId, pgIdx); copies > share)
                {
                    releaseRetainedCopies(lvlId, pgIdx, std::min(copies - share, required - free));
                }
            }
            // Within the share: a retained copy belongs to a page that still lives in a hotter level, so it is more
            // recently used than any page whose only copy is here. At the last level, evicting such a page drops it
            // without any I/O, so evict those first (LRU) and reclaim retained copies only for the shortfall. At
            // other levels evicting a page means migrating it down, which costs more than reclaiming a duplicate.
            SlotCount const available
                = storage.numFreeSlots(pgIdx) + (isLast ? ctrl.numEvictablePages(pgIdx) : SlotCount{0});
            if (required > available)
            {
                releaseRetainedCopies(lvlId, pgIdx, required - available);
            }
        }
        if (isLast)
        {
            // Held pages must fit at the last level. A reserved copy of a droppable page only saves that page a
            // write, so when held pages need the room, give the copy up: the page then falls like any other
            // droppable page and is placed only if a slot is left.
            bool movedKeptPages = false;
            for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(lvlId); ++pgIdx)
            {
                SlotCount heldNeed = goals.at(lvlId).at(pgIdx);
                for (LifeCycleId lifeCycle : grouping.lifeCycles(pgIdx))
                {
                    for (auto const& page : fallenPages.at(lifeCycle))
                    {
                        if (page->status() == PageStatus::HELD)
                        {
                            ++heldNeed;
                        }
                    }
                }
                SlotCount room = storage.numFreeSlots(pgIdx) + ctrl.numEvictablePages(pgIdx);
                for (LifeCycleId lifeCycle : grouping.lifeCycles(pgIdx))
                {
                    auto& kept = keptPages.at(lifeCycle);
                    for (auto it = kept.begin(); it != kept.end() && room < heldNeed;)
                    {
                        if ((*it)->status() == PageStatus::HELD)
                        {
                            ++it;
                            continue;
                        }
                        if (auto found = mRetainedByPage.find(it->get()); found != mRetainedByPage.end())
                        {
                            eraseRetainedCopy(found->second);
                            ++mDiskCopyCounters.reclaimed;
                            ++room;
                        }
                        fallenPages.at(lifeCycle).push_back(std::move(*it));
                        it = kept.erase(it);
                        movedKeptPages = true;
                    }
                }
            }
            if (movedKeptPages)
            {
                sortFallenPagesByPriority(fallenPages);
            }
        }
    }

    TypedVec<PoolGroupIndex, SlotCount> numToEvict(numPoolGroups(lvlId), 0);
    PagesByLifeCycle heldPages(numLifeCycles());

    for (PoolGroupIndex pgIdx{0}; pgIdx < numToEvict.size(); ++pgIdx)
    {
        SlotCount const goal = goals.at(lvlId).at(pgIdx);
        SlotCount const fallen = countPages(pgIdx, fallenPages);
        SlotCount const oldFree = isFailed ? 0 : storage.numFreeSlots(pgIdx);
        SlotCount const evictableCount = isFailed ? 0 : ctrl.numEvictablePages(pgIdx);
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
            return fallen.empty() ? std::nullopt : std::optional<Priority>{fallen.back()->priority()};
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
                    [](auto const& page, Priority priority) { return page->priority() < priority; });
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
        SlotCount const oldFree = isFailed ? 0 : storage.numFreeSlots(pgIdx);
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

            freeAfterEviction = isFailed ? 0 : storage.numFreeSlots(pgIdx);
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
                auto& rejected = fallenPages.at(lifeCycle);
                if (dropRecorder && !rejected.empty() && !isLastLevel(lvlId))
                {
                    // Above failed trailing levels these droppable pages would have been migrated down: count them
                    // as dropped from the level they leave. At the last level, rejected pages are cleared below
                    // without a drop record.
                    std::map<CacheLevel, std::vector<SharedPtr<Page>>> byLevel;
                    for (auto const& page : rejected)
                    {
                        byLevel[page->cacheLevel].push_back(page);
                    }
                    for (auto const& [fromLevel, pages] : byLevel)
                    {
                        dropRecorder(pages, fromLevel);
                    }
                }
                rejected.clear();
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

    // The accepted pages live only in this frame from here on (the eviction lists let go of them): if a migration
    // below or here throws, they would die before the rollback guards of this frame and the ones above run.
    for (auto const& pages : acceptedPages)
    {
        keepAlive.insert(keepAlive.end(), pages.begin(), pages.end());
    }
    if (!isLast)
    {
        sortFallenPagesByPriority(fallenPages);
        _prepareFreeSlots(goals, lvlId + 1, fallenPages, keepAlive, migrationRecorder, dropRecorder);
    }

    TLLM_CHECK_DEBUG_WITH_INFO(
        std::all_of(fallenPages.begin(), fallenPages.end(), [](auto const& fp) { return fp.empty(); }),
        "All fallen pages must be consumed after level loop");

    for (LifeCycleId lifeCycle{0}; lifeCycle < keptPages.size(); ++lifeCycle)
    {
        auto& accepted = acceptedPages.at(lifeCycle);
        accepted.insert(accepted.end(), keptPages.at(lifeCycle).begin(), keptPages.at(lifeCycle).end());
    }
    keptPagesAccepted = true;

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
            if (mDiskWriteThrough && srcLevel == kHotLevel)
            {
                writeThroughToDisk(lvlId, pages);
            }
            for (auto const& page : pages)
            {
                if (srcLevel == kHotLevel && page->status() == PageStatus::DROPPABLE)
                {
                    // Offloading an unreferenced page ends its released retention interval.
                    page->claimRetention(std::nullopt, std::nullopt);
                }
                scheduleForEviction(*page);
            }
        }
    }
    rescheduleEvictedPagesOnFailure.cancel();
    if (dropRecorder)
        for (auto const& pages : belowOffloadThreshold)
            if (!pages.empty())
                dropRecorder(pages, kHotLevel);
    restoreBelowThreshold.cancel();
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
        if (migrationRecorder && !defrag)
        {
            migrationRecorder(srcPages, {}, srcLevel, dstLevel);
        }
        return updateSrc ? std::nullopt : std::optional<std::vector<Slot>>{std::in_place};
    }

    if (updateSrc && !defrag
        && std::any_of(
            srcPages.begin(), srcPages.end(), [](auto const& page) { return page->status() == PageStatus::LOCKED; }))
        throw LogicError("Cannot migrate a locked page between storage levels");

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

    // Keep-disk-copy: pages with a retained copy at dstLevel take that slot back and are not copied.
    std::vector<Slot> dstSlots(srcPages.size());
    std::vector<char> reusedCopy(srcPages.size(), 0);
    SlotCount numNewSlots = numSlots;
    try
    {
        if (updateSrc && !defrag && !mRetainedByPage.empty())
        {
            // Check for room before taking any copy: a failure after taking them would have to put them back.
            SlotCount numUsable = 0;
            for (auto const& page : srcPages)
            {
                auto it = mRetainedByPage.find(page.get());
                if (it != mRetainedByPage.end() && it->second->level == dstLevel
                    && (it->second->reserved || validateRetainedCopy(*page, dstLevel)))
                {
                    ++numUsable;
                }
            }
            if (dstPoolGroup.numFreeSlots() < numSlots - numUsable)
            {
                throw OutOfPagesError("Not enough free slots for migration");
            }
            for (std::size_t i = 0; i < srcPages.size(); ++i)
            {
                if (auto kept = takeRetainedCopy(*srcPages.at(i), dstLevel))
                {
                    dstSlots.at(i) = std::move(*kept);
                    reusedCopy.at(i) = 1;
                    --numNewSlots;
                }
            }
        }
        if (dstPoolGroup.numFreeSlots() < numNewSlots)
        {
            throw OutOfPagesError("Not enough free slots for migration");
        }
        auto freshSlots = dstPoolGroup.allocateMultiple(numNewSlots);
        // A15: allocated slot count must match the request.
        TLLM_CHECK_DEBUG_WITH_INFO(slotCountValueFromSize(freshSlots.size()) == numNewSlots, "dst_slots size mismatch");
        std::size_t nextFresh = 0;
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            if (!reusedCopy.at(i))
            {
                dstSlots.at(i) = std::move(freshSlots.at(nextFresh++));
            }
        }
        bool const retainSrcCopy = updateSrc && !defrag && dstLevel < srcLevel && cacheTier(srcLevel) == CacheTier::DISK
            && mKeepCopyAfterRestoreAt.at(srcLevel);
        thread_local std::vector<PageIndexPair> pageIndices;
        pageIndices.clear();
        pageIndices.reserve(srcPages.size());
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            TLLM_CHECK_DEBUG(defrag || !srcPages.at(i)->scheduledForEviction());
            if (reusedCopy.at(i))
            {
                continue;
            }
            pageIndices.push_back(PageIndexPair{
                slotIdToPageIndexValue(dstSlots.at(i).slotId()), slotIdToPageIndexValue(srcPages.at(i)->slotId())});
        }

        std::vector<CUevent> priorEvents;
        priorEvents.reserve(2 * srcPages.size());
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            priorEvents.push_back(srcPages.at(i)->readyEvent.handle());
            priorEvents.push_back(dstSlots.at(i).readyEvent.handle());
        }

        TemporaryCudaStream tempStream(std::move(priorEvents));
        auto updateReadyEvents = FuncGuard(
            [&]()
            {
                auto const finishEvent = tempStream.takeFinishEvent();
                for (std::size_t i = 0; i < srcPages.size(); ++i)
                {
                    dstSlots.at(i).readyEvent = finishEvent;
                    // A copy that leaves the page in place (write-through) must not hold up the page; its completion
                    // lives on the destination slot.
                    if (updateSrc)
                    {
                        srcPages.at(i)->readyEvent = finishEvent;
                    }
                }
            });
        {
            auto scope = tempStream.enter();
            CUstream const stream = tempStream.get();
            if (!pageIndices.empty())
            {
                submitMigrationBatch(dstLevel, srcLevel, batchingLayerGroupId, pageIndices.data(), pageIndices.size(),
                    stream, /*move=*/defrag);
            }
        }
        updateReadyEvents.run();

        trimRetainedPageIndices(pageIndices);
        if (migrationRecorder && !defrag)
        {
            migrationRecorder(srcPages, dstSlots, srcLevel, dstLevel);
        }
        std::set<std::pair<std::string, int>> emittedCacheLevelUpdates;
        bool const emitCacheLevelUpdates
            = updateSrc && !defrag && srcLevel != dstLevel && static_cast<bool>(mEventSink);
        // Source slot event merged with a pending write-through, per (source event, pending write).
        std::vector<std::pair<std::pair<CUevent, CUevent>, CachedCudaEvent>> mergedPendingWrites;
        for (std::size_t i = 0; i < srcPages.size(); ++i)
        {
            if (updateSrc)
            {
                bool wasScheduled = srcPages.at(i)->scheduledForEviction();
                if (wasScheduled)
                    excludeFromEviction(*srcPages.at(i));
                // A write-through may still be reading the source host slot, so whoever reuses that slot must wait
                // for it too. Merge before exchangeSlot, so nothing that can throw runs between the exchange and the
                // level update. The pages of a batch share one finish event, so this merges once per pending write.
                std::optional<CachedCudaEvent> srcSlotReady;
                if (mDiskWriteThrough && cacheTier(srcLevel) == CacheTier::HOST_MEM)
                {
                    if (auto pending = pendingDiskWrite(*srcPages.at(i)); !pending.isClosed())
                    {
                        auto const key = std::make_pair(srcPages.at(i)->readyEvent.handle(), pending.handle());
                        auto merged = std::find_if(mergedPendingWrites.begin(), mergedPendingWrites.end(),
                            [&key](auto const& entry) { return entry.first == key; });
                        if (merged == mergedPendingWrites.end())
                        {
                            std::vector<CachedCudaEvent> events{srcPages.at(i)->readyEvent, std::move(pending)};
                            merged = mergedPendingWrites.emplace(mergedPendingWrites.end(), key, mergeEvents(events));
                        }
                        srcSlotReady = merged->second;
                    }
                }
                // Replace the page's source slot with its destination. The source slot is released, or kept as a
                // retained disk copy.
                Slot srcSlot = srcPages.at(i)->exchangeSlot(std::move(dstSlots.at(i)));
                srcPages.at(i)->cacheLevel = dstLevel;
                if (srcSlotReady.has_value())
                {
                    srcSlot.readyEvent = std::move(*srcSlotReady);
                }
                if (retainSrcCopy && hasFinalContent(*srcPages.at(i)))
                {
                    // The page's bytes are final, so the disk data stays valid while the page lives above.
                    addRetainedCopy(*srcPages.at(i), srcLevel, srcPgIdx, std::move(srcSlot));
                }
                else
                {
                    srcPoolGroup.release(std::move(srcSlot));
                }
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
        mDiskCopyCounters.reused += static_cast<uint64_t>(numSlots - numNewSlots);
        return updateSrc ? std::nullopt : std::optional<std::vector<Slot>>{std::move(dstSlots)};
    }
    catch (...)
    {
        for (auto& s : dstSlots)
        {
            if (s.hasValidSlot())
            {
                dstPoolGroup.release(std::move(s));
            }
        }
        throw;
    }
}

// ---------------------------------------------------------------------------
// batchedMigrate
// ---------------------------------------------------------------------------

void StorageManager::batchedMigrate(
    CacheLevel dstLevel, std::vector<SharedPtr<Page>> const& pages, MigrationRecorder const& migrationRecorder)
{
    std::map<MigrationBatchKey, std::vector<SharedPtr<Page>>> groups;
    uint64_t const visit = nextVisitStamp();
    for (auto const& page : pages)
    {
        if (page->cacheLevel == dstLevel || page->visitStamp == visit)
        {
            continue;
        }
        page->visitStamp = visit;
        CacheLevel const srcLevel = page->cacheLevel;
        groups[{srcLevel, getMigrationBatchingLayerGroupId(dstLevel, srcLevel, page->lifeCycle)}].push_back(page);
    }
    for (auto& [key, group] : groups)
    {
        _batchedMigrate(dstLevel, key.first, group, /*updateSrc=*/true, migrationRecorder);
    }
}

int64_t StorageManager::prefetch(
    CacheLevel dstLevel, TypedVec<LifeCycleId, TypedVec<CacheLevel, std::vector<SharedPtr<Page>>>> const& pages)
{
    TypedVec<PoolGroupIndex, SlotCount> numSlotsToMigrate(numPoolGroups(dstLevel), 0);
    int64_t diskBlocksMigrated = 0;
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
        CacheLevel const srcLevel = migrationPath.first;
        _batchedMigrate(dstLevel, srcLevel, migrationPages, /*updateSrc=*/true);
        // Per batch, after it landed: the pages are already grouped by source level, so this costs
        // nothing per page and never credits a batch that did not run.
        if (cacheTier(srcLevel) == CacheTier::DISK)
        {
            diskBlocksMigrated += static_cast<int64_t>(migrationPages.size());
        }
    }
    reschedulePagesGuard.run();
    return diskBlocksMigrated;
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

PoolIndex StorageManager::numPools(CacheLevel level, PoolGroupIndex pgIdx) const
{
    return mLevels.at(level).storage->numPools(pgIdx);
}

PoolIndex StorageManager::numPools(PoolGroupIndex pgIdx) const
{
    return numPools(kHotLevel, pgIdx);
}

CacheLevel StorageManager::appendLevelSlotDescList(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescs)
{
    CacheLevel const level = mSlotDescLists.size();
    mSlotDescLists.push_back(slotDescs);

    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> sizes;
    sizes.reserve(slotDescs.size());
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotDescs.size(); ++pgIdx)
    {
        sizes.push_back(slotDescs.at(pgIdx).slotSizeList());
    }
    mSlotSizes.push_back(std::move(sizes));

    TLLM_CHECK_DEBUG(mSlotSizes.size() == mSlotDescLists.size());
    return level;
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

TypedVec<PoolGroupIndex, PoolGroupDesc> StorageManager::poolGroupDescs() const
{
    auto const& descList = slotDescList(kHotLevel);
    TypedVec<PoolGroupIndex, PoolGroupDesc> result;
    result.reserve(descList.size());
    for (PoolGroupIndex pgIdx{0}; pgIdx < descList.size(); ++pgIdx)
    {
        auto const& slotSizeList = slotSize(kHotLevel, pgIdx);
        TypedVec<PoolIndex, PoolDesc> pools;
        pools.reserve(slotSizeList.size());
        for (PoolIndex poolIdx{0}; poolIdx < slotSizeList.size(); ++poolIdx)
        {
            pools.push_back(PoolDesc{poolIdx, getMemPoolBaseAddress(pgIdx, poolIdx), slotSizeList.at(poolIdx),
                poolMedium(kHotLevel, pgIdx, poolIdx)});
        }
        result.push_back(PoolGroupDesc{pgIdx, numSlots(pgIdx, kHotLevel), descList.at(pgIdx), std::move(pools)});
    }
    return result;
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
    // Retained disk copies are duplicates the level releases when it needs the room. They are reported apart from
    // evictable pages; available() counts both.
    SlotCount retainedCopies = mNumRetainedCopies.at(level).at(pgIdx);
    auto sizes = lvl.storage->slotSize(pgIdx);
    return StorageStatistics{sizes, totalSlots, freeSlots, evictable, retainedCopies};
}

TypedVec<PoolGroupIndex, float> StorageManager::getUtilization(CacheLevel level) const
{
    TypedVec<PoolGroupIndex, float> result;
    result.reserve(numPoolGroups(level));
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPoolGroups(level); ++pgIdx)
    {
        auto const s = getStatistics(level, pgIdx);
        TLLM_CHECK_WITH_INFO(s.total > 0, "getUtilization: pool group %d at level %d has zero capacity",
            static_cast<int>(pgIdx.value()), static_cast<int>(level.value()));
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
    auto const window = poolGroup(level, pgIdx).slotAllocator().window();
    TLLM_CHECK_DEBUG(newNumSlots < window.size());
    shrinkPoolGroupToWindow(level, pgIdx, SlotWindow{window.begin, window.begin + newNumSlots}, persistentPages);
}

void StorageManager::shrinkPoolGroupToWindow(CacheLevel level, PoolGroupIndex pgIdx, SlotWindow target,
    std::vector<SharedPtr<Page>> const& persistentPages, std::vector<SharedPtr<Page>> const& lockedPages,
    ScratchSlotLockList const& scratchSlots)
{
    refreshRetention();
    auto& pg = poolGroup(level, pgIdx);
    auto& allocator = pg.slotAllocator();
    auto& ctrl = mLevels.at(level).controller;
    TLLM_CHECK_DEBUG(allocator.window().contains(target) && allocator.window() != target);
    SlotCount const newNumSlots = target.size();

    // A16: persistent_pages preconditions.
    TLLM_CHECK_DEBUG_WITH_INFO(
        persistentPages.size() <= slotCountToSizeT(newNumSlots), "Not enough slots to hold all persistent pages");
    TLLM_CHECK_WITH_INFO(std::all_of(persistentPages.begin(), persistentPages.end(),
                             [this, level, pgIdx](auto const& p)
                             { return p->cacheLevel == level && getPoolGroupIndex(level, p->lifeCycle) == pgIdx; }),
        "Persistent page cache level or pool group mismatch");

    // Fast path: when no slot id has ever been issued in the part of the window being given up,
    // there is nothing to migrate.
    if (allocator.numIssuedOutside(target) == 0)
    {
        allocator.prepareForShrinkWindow(target);
        allocator.finishShrink();
        pg.setPoolWindows(target);
        return;
    }

    // Copy-based rebalancing, step 7: pages that would have to move and that nothing references
    // can be dropped instead, when the server is configured to.
    if (mDropUnreferencedOnRebalance && isArenaLevel(level))
    {
        dropUnreferencedPagesOutside(level, pgIdx, target);
    }

    // Find overflow pages: scheduled pages whose slot lies outside the target.
    auto gen = ctrl.pageGenerator(pgIdx);
    std::deque<std::pair<SlotCount, SharedPtr<Page>>> overflowSlots;
    {
        SlotCount idx = 0;
        while (auto const* page = gen())
        {
            if (!target.contains((*page)->slotId()))
                overflowSlots.emplace_back(idx, *page);
            ++idx;
        }
    }

    // Persistent pages in overflow range.
    std::vector<SharedPtr<Page>> overflowPersistent;
    for (auto const& p : persistentPages)
    {
        if (!target.contains(p->slotId()))
            overflowPersistent.push_back(p);
    }
    SlotCount numOverflowPersistent = slotCountValueFromSize(overflowPersistent.size());

    // Locked pages (in-place relocation) in overflow range. Like persistent pages they cannot be
    // evicted; unlike them they are relocated with their owners' page indices republished.
    TLLM_CHECK_WITH_INFO(std::all_of(lockedPages.begin(), lockedPages.end(),
                             [this, level, pgIdx](auto const& p)
                             { return p->cacheLevel == level && getPoolGroupIndex(level, p->lifeCycle) == pgIdx; }),
        "Locked page cache level or pool group mismatch");
    std::vector<SharedPtr<Page>> overflowLocked;
    for (auto const& p : lockedPages)
    {
        if (!target.contains(p->slotId()))
            overflowLocked.push_back(p);
    }
    // Scratch slots (in-place relocation with SWA scratch reuse) in overflow range: held by ACTIVE
    // caches outside the page machinery; they move like locked pages.
    TLLM_CHECK_WITH_INFO(std::all_of(scratchSlots.begin(), scratchSlots.end(),
                             [this, level, pgIdx](ScratchSlotLock const* s)
                             { return level == kHotLevel && getPoolGroupIndex(level, s->lifeCycle()) == pgIdx; }),
        "Scratch slot cache level or pool group mismatch");
    ScratchSlotLockList overflowScratch;
    for (ScratchSlotLock* s : scratchSlots)
    {
        if (!target.contains(s->slot().slotId()))
            overflowScratch.push_back(s);
    }
    // The callers gather these by iterating KvCache pointers (a std::set ordered by address), so their order differs
    // between processes. Relocation hands out free slots in list order, so sort by slot id: TP ranks must end up
    // with the same slot layout, because later shrinks evict and migrate by slot position.
    auto const bySlotId = [](auto const& a, auto const& b) { return a->slotId() < b->slotId(); };
    std::sort(overflowPersistent.begin(), overflowPersistent.end(), bySlotId);
    std::sort(overflowLocked.begin(), overflowLocked.end(), bySlotId);
    std::sort(overflowScratch.begin(), overflowScratch.end(),
        [](ScratchSlotLock const* a, ScratchSlotLock const* b) { return a->slot().slotId() < b->slot().slotId(); });
    // Pages that must move and cannot be evicted.
    SlotCount const numMustMove
        = numOverflowPersistent + slotCountValueFromSize(overflowLocked.size() + overflowScratch.size());

    // A2: RUNTIME check — persistent and locked overflow pages must fit in the new capacity.
    if (numMustMove > newNumSlots)
    {
        throw OutOfPagesError("Not enough slots to hold all persistent pages");
    }

    // Mark the allocator for shrink.
    allocator.prepareForShrinkWindow(target);

    // Calculate minimum number of lowest-priority pages to evict.
    // Need numEvictedOverflowSlots because evicted overflow pages won't become free,
    // because only free non-overflow slots can be used for defragmentation.
    SlotCount minNumEvicted = 0;
    SlotCount numEvictedOverflowSlots = 0;
    while (!overflowSlots.empty()
        && slotCountValueFromSize(overflowSlots.size()) + numMustMove
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
    reqs[pgIdx] = slotCountValueFromSize(overflowPages.size() + overflowLocked.size() + overflowScratch.size());
    // The eviction order captured above must remain stable through defragmentation.
    _prepareFreeSlots(level, reqs, MigrationRecorder{}, DropRecorder{});

    // A17: all overflow pages must be at the expected cache level.
    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(overflowPages.begin(), overflowPages.end(),
                                   [level](auto const& p) { return p->cacheLevel == level; }),
        "Overflow page cache level mismatch");

    // Defragment: migrate overflow pages to free slots within the same level.
    _batchedMigrate(level, level, overflowPages, /*updateSrc=*/true, MigrationRecorder{}, /*defrag=*/true);
    _relocateLockedPages(level, pgIdx, overflowLocked, overflowScratch);

    // A18: post-defrag overflow assertion — overflow slot count matches expectations.
    TLLM_CHECK_DEBUG_WITH_INFO(
        allocator.numOverflowSlots() == allocator.numIssuedOutsideTarget(), "Post-defrag overflow slot count mismatch");

    // Finalize shrink and resize pools. In an arena adjust the copies still reading the vacated region are not waited
    // for here: whoever takes the region over waits for them (moveArenaWindow).
    allocator.finishShrink(mDeferArenaShrinkSync ? &mArenaVacatedCopies : nullptr);
    pg.setPoolWindows(target);
}

SlotCount StorageManager::dropUnreferencedPagesOutside(CacheLevel level, PoolGroupIndex pgIdx, SlotWindow target)
{
    auto& ctrl = mLevels.at(level).controller;
    // Collect first: removing a page from its eviction queue invalidates the generator.
    std::vector<SharedPtr<Page>> victims;
    auto gen = ctrl.pageGenerator(pgIdx);
    while (auto const* page = gen())
    {
        if (!target.contains((*page)->slotId()) && (*page)->status() == PageStatus::DROPPABLE)
            victims.push_back(*page);
    }
    for (auto const& page : victims)
    {
        excludeFromEviction(*page);
    }
    auto const numDropped = slotCountValueFromSize(victims.size());
    // The eviction queue was these pages' only owner, so releasing the last references here
    // unlinks them from the radix tree and returns their slots (~CommittedPage, ~Page).
    victims.clear();
    return numDropped;
}

void StorageManager::_relocateLockedPages(CacheLevel level, PoolGroupIndex pgIdx,
    std::vector<SharedPtr<Page>> const& pages, ScratchSlotLockList const& scratchSlots)
{
    if (pages.empty() && scratchSlots.empty())
    {
        return;
    }
    auto& pool = poolGroup(level, pgIdx);
    std::vector<SharedPtr<UniqPageLock>> locks;
    locks.reserve(pages.size());
    std::set<CUstream> ownerStreams;
    for (auto const& page : pages)
    {
        auto holder = page->holder.lock();
        auto lock = holder ? holder->uniqLock.lock() : nullptr;
        TLLM_CHECK_WITH_INFO(lock && page->cacheLevel == level, "In-place relocation requires a page locked here");
        for (auto const& owner : lock->owners())
        {
            TLLM_CHECK_WITH_INFO(owner.kvCache->isActive(), "In-place relocation requires active owners");
            ownerStreams.insert(owner.kvCache->cudaStream());
        }
        locks.push_back(std::move(lock));
    }
    for (ScratchSlotLock const* scratch : scratchSlots)
    {
        TLLM_CHECK_WITH_INFO(scratch->owner().isActive(), "In-place relocation requires active owners");
        ownerStreams.insert(scratch->owner().cudaStream());
    }

    // Destinations: one per page, then one per scratch slot.
    auto dstSlots = pool.allocateMultiple(slotCountValueFromSize(pages.size() + scratchSlots.size()));
    auto releaseDestinations = FuncGuard(
        [&]()
        {
            for (auto& slot : dstSlots)
            {
                if (slot.hasValidSlot())
                    pool.release(std::move(slot));
            }
        });

    // The copy waits for each owner's prior work, each page's readiness and readers, and each
    // destination's previous users.
    std::vector<CachedCudaEvent> ownerEvents;
    ownerEvents.reserve(ownerStreams.size());
    std::vector<CUevent> priorEvents;
    for (auto const ownerStream : ownerStreams)
    {
        ownerEvents.emplace_back(reinterpret_cast<CudaStream>(ownerStream));
        priorEvents.push_back(ownerEvents.back().handle());
    }
    std::map<LayerGroupId, std::vector<PageIndexPair>> batches;
    for (size_t i = 0; i < pages.size(); ++i)
    {
        priorEvents.push_back(pages[i]->readyEvent.handle());
        priorEvents.push_back(dstSlots[i].readyEvent.handle());
        for (auto const& event : locks[i]->finishEvents)
            priorEvents.push_back(event.handle());
        batches[getMigrationBatchingLayerGroupId(level, level, pages[i]->lifeCycle)].push_back(
            PageIndexPair{slotIdToPageIndexValue(dstSlots[i].slotId()), slotIdToPageIndexValue(pages[i]->slotId())});
    }
    // A scratch slot has a single user, its owner, whose prior work is already waited for.
    for (size_t j = 0; j < scratchSlots.size(); ++j)
    {
        Slot const& dst = dstSlots[pages.size() + j];
        Slot const& src = scratchSlots[j]->slot();
        priorEvents.push_back(src.readyEvent.handle());
        priorEvents.push_back(dst.readyEvent.handle());
        batches[getMigrationBatchingLayerGroupId(level, level, scratchSlots[j]->lifeCycle())].push_back(
            PageIndexPair{slotIdToPageIndexValue(dst.slotId()), slotIdToPageIndexValue(src.slotId())});
    }

    TemporaryCudaStream tempStream(std::move(priorEvents));
    {
        auto scope = tempStream.enter();
        CUstream const stream = tempStream.get();
        for (auto const& [layerGroup, indices] : batches)
        {
            submitMigrationBatch(level, level, layerGroup, indices.data(), indices.size(), stream, /*move=*/true);
        }
    }
    CachedCudaEvent const completion = tempStream.takeFinishEvent();
    for (size_t i = 0; i < pages.size(); ++i)
    {
        dstSlots[i].readyEvent = completion;
        locks[i]->recordMigrationEvent(completion);
    }
    // Every owner's next reader observes the copied data.
    for (auto const ownerStream : ownerStreams)
    {
        completion.waitInStream(reinterpret_cast<CudaStream>(ownerStream));
    }
    for (size_t i = 0; i < pages.size(); ++i)
    {
        // The returned source slot carries the copy's completion as its ready event.
        pool.release(locks[i]->relocate(std::move(dstSlots[i])));
    }
    for (size_t j = 0; j < scratchSlots.size(); ++j)
    {
        Slot& dst = dstSlots[pages.size() + j];
        dst.readyEvent = completion;
        Slot source = scratchSlots[j]->relocate(std::move(dst));
        // Its next user must not overwrite it before the copy has read it.
        source.readyEvent = completion;
        pool.release(std::move(source));
    }
    releaseDestinations.cancel();
}

bool StorageManager::isHotArenaGroup(PoolGroupIndex pgIdx) const
{
    return mGpuArena && static_cast<ArenaGpuCacheLevelStorage const&>(*mLevels.at(kHotLevel).storage).inArena(pgIdx);
}

bool StorageManager::isArenaLevel(CacheLevel level) const
{
    if (level == kHotLevel)
        return mGpuArena;
    return dynamic_cast<ArenaHostCacheLevelStorage const*>(mLevels.at(level).storage.get()) != nullptr;
}

void StorageManager::moveArenaWindow(CacheLevel level, PoolGroupIndex pgIdx, SlotWindow from, SlotWindow to,
    std::vector<SharedPtr<Page>> const& persistentPages, std::vector<SharedPtr<Page>> const& lockedPages,
    ScratchSlotLockList const& scratchSlots)
{
    auto& pg = poolGroup(level, pgIdx);
    TLLM_CHECK_DEBUG(pg.slotAllocator().window() == from);
    // Growing first lets the blocks this group gives up be copied into the space it gains.
    SlotWindow const hull = from.hull(to);
    if (hull != from)
    {
        pg.setPoolWindows(hull);
        pg.slotAllocator().expandWindow(hull);
        if (mDeferArenaShrinkSync)
        {
            // The space gained was given up by a group that moved first in this adjust; copies out of it may still
            // run. (Ids of `from` that were never issued wait too: harmless.)
            pg.slotAllocator().setFreshIdReadyEvent(hull, mArenaVacatedCopies);
        }
    }
    if (hull != to)
    {
        shrinkPoolGroupToWindow(level, pgIdx, to, persistentPages, lockedPages, scratchSlots);
    }
}

void StorageManager::adjustArenaLevel(CacheLevel level, std::optional<size_t> newQuota,
    TypedVec<PoolGroupIndex, float> const& ratioList,
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* persistentPages,
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* lockedPages,
    TypedVec<PoolGroupIndex, ScratchSlotLockList> const* scratchSlots)
{
    auto& storage = *mLevels.at(level).storage;
    auto* const gpuArena = level == kHotLevel ? &static_cast<ArenaGpuCacheLevelStorage&>(storage) : nullptr;
    auto* const hostArena = gpuArena ? nullptr : &static_cast<ArenaHostCacheLevelStorage&>(storage);
    size_t const arenaQuota = storage.roundedNumBytes(storage.cacheTier());
    if (newQuota.has_value() && roundUp(*newQuota, storage.poolSizeGranularity(storage.cacheTier())) != arenaQuota)
    {
        throw std::invalid_argument("Copy-based rebalancing uses a fixed arena of " + std::to_string(arenaQuota)
            + " bytes at cache level " + std::to_string(level.value()) + "; its quota cannot be changed to "
            + std::to_string(*newQuota));
    }
    auto const oldWindows = gpuArena ? gpuArena->windows() : hostArena->windows();
    // Target clamping: a group cannot shrink below the pages it is unable to give up (occupied but
    // not evictable, i.e. held by requests when this is the last level), scaled by 1/maxUtilForResume
    // so that KvCache::resume does not then refuse every suspended request. Computing every window
    // before touching any keeps an impossible target from leaving the arena half-adjusted.
    auto minSlots
        = level == kHotLevel ? mHotMinSlots : TypedVec<PoolGroupIndex, SlotCount>(numPoolGroups(level), SlotCount{1});
    auto const& ctrl = mLevels.at(level).controller;
    for (PoolGroupIndex pgIdx{0}; pgIdx < minSlots.size(); ++pgIdx)
    {
        SlotCount const unevictable
            = storage.poolGroup(pgIdx).slotAllocator().numOccupiedSlots() - ctrl.numEvictablePages(pgIdx);
        auto const resumable = static_cast<SlotCount>(
            std::ceil(static_cast<double>(unevictable) / static_cast<double>(mMaxUtilForResume)));
        minSlots[pgIdx] = std::max(minSlots[pgIdx], resumable);
    }
    TypedVec<PoolGroupIndex, SlotWindow> newWindows;
    if (gpuArena)
    {
        auto const newPoolWindows = gpuArena->computePoolWindows(ratioList, minSlots);
        if (gpuArena->hasBoundPools())
        {
            adjustArenaPools(newPoolWindows, persistentPages, lockedPages, scratchSlots);
            return;
        }
        for (auto const& windows : newPoolWindows)
            newWindows.push_back(windows[PoolIndex{0}]);
    }
    else
    {
        newWindows = hostArena->computeWindows(ratioList, minSlots);
    }
    PoolGroupIndex const numPg = oldWindows.size();
    auto byteRange = [this, level](PoolGroupIndex pgIdx, SlotWindow w)
    {
        size_t const s = slotSize(level, pgIdx)[PoolIndex{0}];
        return std::make_pair(toSizeT(w.begin) * s, toSizeT(w.end) * s);
    };
    // Group a gains from group b if a's new range overlaps b's old range. Such space must be vacated
    // by b before a may use it, so b moves first. Along a line of groups this order always exists:
    // a gaining from b means a's new range reaches past the boundary between them, so b cannot also
    // gain from a.
    // A fixed group lives outside the arena (its window is in ids of its own allocation), so it neither moves
    // nor overlaps anything.
    auto inArena = [&](PoolGroupIndex pgIdx) { return !gpuArena || gpuArena->inArena(pgIdx); };
    auto gainsFrom = [&](PoolGroupIndex a, PoolGroupIndex b)
    {
        if (!inArena(a) || !inArena(b))
            return false;
        auto const [aBegin, aEnd] = byteRange(a, newWindows[a]);
        auto const [bBegin, bEnd] = byteRange(b, oldWindows[b]);
        return a != b && aBegin < bEnd && bBegin < aEnd;
    };
    enum class State : std::uint8_t
    {
        kPending,
        kMoving,
        kDone
    };
    TypedVec<PoolGroupIndex, State> state(numPg, State::kPending);
    // Groups that shrink hand their pending copies to the groups that grow into the space (see moveArenaWindow) rather
    // than make the host wait for them.
    mDeferArenaShrinkSync = true;
    mArenaVacatedCopies = CachedCudaEvent::makeNull();
    auto endDeferredSync = FuncGuard(
        [this]()
        {
            mDeferArenaShrinkSync = false;
            mArenaVacatedCopies = CachedCudaEvent::makeNull();
        });
    std::function<void(PoolGroupIndex)> move = [&](PoolGroupIndex pgIdx)
    {
        if (state[pgIdx] == State::kDone)
            return;
        TLLM_CHECK_WITH_INFO(state[pgIdx] == State::kPending, "Cyclic arena window dependency");
        state[pgIdx] = State::kMoving;
        for (PoolGroupIndex other{0}; other < numPg; ++other)
        {
            if (gainsFrom(pgIdx, other))
                move(other);
        }
        if (newWindows[pgIdx] != oldWindows[pgIdx])
        {
            static std::vector<SharedPtr<Page>> const kNoPages;
            static ScratchSlotLockList const kNoScratch;
            moveArenaWindow(level, pgIdx, oldWindows[pgIdx], newWindows[pgIdx],
                persistentPages ? (*persistentPages)[pgIdx] : kNoPages, lockedPages ? (*lockedPages)[pgIdx] : kNoPages,
                scratchSlots ? (*scratchSlots)[pgIdx] : kNoScratch);
        }
        state[pgIdx] = State::kDone;
    };
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        move(pgIdx);
    }
}

void StorageManager::adjustArenaPools(TypedVec<PoolGroupIndex, TypedVec<PoolIndex, SlotWindow>> const& newWindows,
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* persistentPages,
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* lockedPages,
    TypedVec<PoolGroupIndex, ScratchSlotLockList> const* scratchSlots)
{
    auto& arena = static_cast<ArenaGpuCacheLevelStorage&>(*mLevels.at(kHotLevel).storage);
    auto const oldWindows = arena.poolWindows();
    PoolGroupIndex const numPg = oldWindows.size();

    // 1. A group that shrinks first gives up the blocks that will not fit, so that each of its pools
    //    can then move on its own, in any order.
    TypedVec<PoolGroupIndex, SlotCount> giveUp(numPg, 0);
    bool anyShrink = false;
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        SlotCount const oldCount = oldWindows[pgIdx][PoolIndex{0}].size();
        SlotCount const newCount = newWindows[pgIdx][PoolIndex{0}].size();
        if (newCount < oldCount)
        {
            giveUp[pgIdx] = oldCount - newCount;
            anyShrink = true;
        }
    }
    if (anyShrink)
    {
        prepareFreeSlots(kHotLevel, giveUp);
    }

    // 2. Move every pool. One whose new bytes overlap another's old bytes waits for that one: pools
    //    keep their order in the arena, so (as for whole groups) no two can wait for each other.
    struct Entry
    {
        PoolGroupIndex pg;
        PoolIndex pool;
    };

    std::vector<Entry> entries;
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        if (!arena.inArena(pgIdx))
            continue; // fixed: outside the arena, never moves
        for (PoolIndex poolIdx{0}; poolIdx < oldWindows[pgIdx].size(); ++poolIdx)
            entries.push_back(Entry{pgIdx, poolIdx});
    }
    auto byteRange = [&](Entry const& e, SlotWindow w)
    {
        size_t const s = slotSize(kHotLevel, e.pg)[e.pool];
        return std::make_pair(toSizeT(w.begin) * s, toSizeT(w.end) * s);
    };
    auto gainsFrom = [&](size_t a, size_t b)
    {
        auto const [aBegin, aEnd] = byteRange(entries[a], newWindows[entries[a].pg][entries[a].pool]);
        auto const [bBegin, bEnd] = byteRange(entries[b], oldWindows[entries[b].pg][entries[b].pool]);
        return a != b && aBegin < bEnd && bBegin < aEnd;
    };
    enum class State : std::uint8_t
    {
        kPending,
        kMoving,
        kDone
    };
    std::vector<State> state(entries.size(), State::kPending);
    static std::vector<SharedPtr<Page>> const kNoPages;
    static ScratchSlotLockList const kNoScratch;
    std::function<void(size_t)> move = [&](size_t i)
    {
        if (state[i] == State::kDone)
            return;
        TLLM_CHECK_WITH_INFO(state[i] == State::kPending, "Cyclic arena window dependency");
        state[i] = State::kMoving;
        for (size_t other = 0; other < entries.size(); ++other)
        {
            if (gainsFrom(i, other))
                move(other);
        }
        auto const [pgIdx, poolIdx] = entries[i];
        SlotWindow const from = oldWindows[pgIdx][poolIdx];
        SlotWindow const to = newWindows[pgIdx][poolIdx];
        if (from != to)
        {
            if (poolIdx == PoolIndex{0})
            {
                // Bound pools need nothing for scratch slots: remapArenaPool moves every group id
                // in use, scratch slots included.
                moveArenaWindow(kHotLevel, pgIdx, from, to, persistentPages ? (*persistentPages)[pgIdx] : kNoPages,
                    lockedPages ? (*lockedPages)[pgIdx] : kNoPages, scratchSlots ? (*scratchSlots)[pgIdx] : kNoScratch);
            }
            else
            {
                remapArenaPool(pgIdx, poolIdx, to);
            }
        }
        state[i] = State::kDone;
    };
    for (size_t i = 0; i < entries.size(); ++i)
    {
        move(i);
    }

    // 3. Every slot id in a group's window is bound again, ready to be allocated.
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        if (arena.inArena(pgIdx))
            arena.poolGroup(pgIdx).completeBindings();
    }
    arena.bumpBindingVersion();
}

void StorageManager::remapArenaPool(PoolGroupIndex pgIdx, PoolIndex poolIdx, SlotWindow target)
{
    auto& group = poolGroup(kHotLevel, pgIdx);
    auto& pool = group.arenaPool(poolIdx);
    TLLM_CHECK_WITH_INFO(
        pool.isBound(), "remapArenaPool: pool %d of group %d is not bound", poolIdx.value(), pgIdx.value());
    auto const inUse = group.slotAllocator().occupiedSlotIds();
    TLLM_CHECK_WITH_INFO(slotCountValueFromSize(inUse.size()) <= target.size(),
        "remapArenaPool: %zu blocks in use do not fit the new window of %ld slots", inUse.size(),
        static_cast<long>(target.size()));

    // Free group slot ids give up their bindings, which frees their ids for the blocks in use.
    {
        std::vector<std::uint8_t> used(pool.bindings().size(), 0);
        for (SlotId const slot : inUse)
            used.at(toSizeT(slot)) = 1;
        for (size_t id = 0; id < used.size(); ++id)
        {
            if (!used[id])
                pool.unbind(SlotId{static_cast<SlotCount>(id)});
        }
    }

    // Blocks in use whose data lies outside the target move to free ids inside it.
    pool.setWindow(pool.window().hull(target));
    std::vector<CopyTask> tasks;
    SlotId next = target.begin;
    for (SlotId const slot : inUse)
    {
        auto const local = pool.localId(slot);
        TLLM_CHECK_WITH_INFO(local.has_value(), "remapArenaPool: group slot id %ld is in use but not bound",
            static_cast<long>(slot.value()));
        if (target.contains(*local))
            continue;
        while (pool.groupId(next).has_value())
            ++next;
        TLLM_CHECK_DEBUG(target.contains(next));
        tasks.push_back(CopyTask{Address{std::in_place_type<MemAddress>, pool.localAddress(next)},
            Address{std::in_place_type<MemAddress>, pool.localAddress(*local)}});
        pool.unbind(slot);
        pool.bind(slot, next);
    }
    if (!tasks.empty())
    {
        // The rebalance runs between iterations. Every reader and writer of the blocks, and every
        // earlier user of the space they move into, has finished once the context is idle.
        cuCheck(cuCtxSynchronize());
        TemporaryCudaStream tempStream(std::vector<CUevent>{});
        {
            auto scope = tempStream.enter();
            mCopyEngine->transfer(CacheTier::GPU_MEM, CacheTier::GPU_MEM, pool.slotSize(), tasks, tempStream.get());
        }
        // The ids just vacated may be bound to a new block as soon as this returns.
        tempStream.takeFinishEvent().synchronize();
    }
    pool.setWindow(target);
}

SlotCount StorageManager::slotIdUpperBound(PoolGroupIndex pgIdx, PoolIndex poolIdx) const
{
    if (!isHotArenaGroup(pgIdx))
    {
        // VMM pools share the group's slot ids, and a pool may hold more slots than the group uses (the group's
        // count is the minimum over its pools): views over any pool of the group cover the group's slots.
        return numSlots(pgIdx, kHotLevel);
    }
    // Arena pools have arena-relative ids of their own; the bound is the pool's (the arena end).
    return mLevels.at(kHotLevel).storage->poolGroup(pgIdx).slotIdUpperBound(poolIdx);
}

TypedVec<PoolGroupIndex, TypedVec<PoolIndex, SlotWindow>> StorageManager::hotPoolWindows() const
{
    auto const& storage = *mLevels.at(kHotLevel).storage;
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, SlotWindow>> result;
    for (PoolGroupIndex pgIdx{0}; pgIdx < storage.numPoolGroups(); ++pgIdx)
    {
        auto const& pg = storage.poolGroup(pgIdx);
        TypedVec<PoolIndex, SlotWindow> pools;
        for (PoolIndex poolIdx{0}; poolIdx < pg.numPools(); ++poolIdx)
        {
            pools.push_back(isHotArenaGroup(pgIdx) ? pg.arenaPool(poolIdx).window() : pg.slotAllocator().window());
        }
        result.push_back(std::move(pools));
    }
    return result;
}

std::optional<std::vector<int32_t>> StorageManager::hotPoolBindings(PoolGroupIndex pgIdx, PoolIndex poolIdx) const
{
    auto const& pg = mLevels.at(kHotLevel).storage->poolGroup(pgIdx);
    if (!pg.isBoundPool(poolIdx))
        return std::nullopt;
    return pg.arenaPool(poolIdx).bindings();
}

uint64_t StorageManager::hotPoolBindingVersion() const noexcept
{
    if (!mGpuArena)
        return 0;
    return static_cast<ArenaGpuCacheLevelStorage const&>(*mLevels.at(kHotLevel).storage).bindingVersion();
}

TypedVec<PoolGroupIndex, SlotWindow> StorageManager::hotWindows() const
{
    return levelWindows(kHotLevel);
}

TypedVec<PoolGroupIndex, SlotWindow> StorageManager::levelWindows(CacheLevel level) const
{
    auto const& storage = *mLevels.at(level).storage;
    TypedVec<PoolGroupIndex, SlotWindow> result;
    result.reserve(storage.numPoolGroups());
    for (PoolGroupIndex pgIdx{0}; pgIdx < storage.numPoolGroups(); ++pgIdx)
    {
        result.push_back(storage.poolGroup(pgIdx).slotAllocator().window());
    }
    return result;
}

// ---------------------------------------------------------------------------
// adjustCacheLevel — mirrors Python _storage_manager.py::adjust_cache_level
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::planCacheLevel(
    CacheLevel level, std::optional<size_t> newQuota, TypedVec<PoolGroupIndex, float> const& ratioList) const
{
    // Arena levels compute windows, not counts from a quota (adjustArenaLevel), and are never remapped, so no
    // caller needs a plan for them.
    TLLM_CHECK_WITH_INFO(!isArenaLevel(level), "planCacheLevel is not defined for an arena level");
    auto const& lvlStorage = *mLevels.at(level).storage;
    size_t quota = newQuota.has_value()
        ? roundUp(newQuota.value(), lvlStorage.poolSizeGranularity(lvlStorage.cacheTier()))
        : lvlStorage.roundedNumBytes(lvlStorage.cacheTier());
    // Chargeable, not total: a rebalance that charged host bytes to the device budget would
    // under-size the level and evict live pages to reach it.
    size_t minQuota = minQuotaForLevel(level);
    if (quota < minQuota)
    {
        throw std::invalid_argument("Quota " + std::to_string(quota)
            + " is insufficient for min_slots constraints (requires at least " + std::to_string(minQuota) + ")");
    }
    // Rejected here rather than deeper down: a level whose address space was
    // reserved up front cannot grow past it without moving the base address,
    // and by the time a pool discovers that, some groups may already be resized.
    if (auto const maxQuota = lvlStorage.maxQuota(); maxQuota.has_value() && quota > *maxQuota)
    {
        throw std::invalid_argument("Quota " + std::to_string(quota) + " exceeds the reserved maximum of "
            + std::to_string(*maxQuota) + " for this cache level (raise the tier's max_quota)");
    }
    // Fixed hot-tier groups keep their size; only elastic groups are resized.
    return slotCountsForLevel(level, quota, ratioList);
}

void StorageManager::adjustCacheLevel(CacheLevel level, std::optional<size_t> newQuota,
    TypedVec<PoolGroupIndex, float> const& ratioList,
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* persistentPages,
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* lockedPages,
    TypedVec<PoolGroupIndex, ScratchSlotLockList> const* scratchSlots)
{
    if (isArenaLevel(level))
    {
        adjustArenaLevel(level, newQuota, ratioList, persistentPages, lockedPages, scratchSlots);
        return;
    }
    TLLM_CHECK_WITH_INFO(lockedPages == nullptr && scratchSlots == nullptr,
        "Locked pages and scratch slots can only be relocated in an arena");
    auto& lvlStorage = *mLevels.at(level).storage;
    auto oldNumSlots = lvlStorage.slotCountList();
    // One computation with planCacheLevel, so a caller can act on the plan (e.g. deregister a shrinking
    // NIXL-registered tail) before the resize and rely on it matching.
    auto newNumSlots = planCacheLevel(level, newQuota, ratioList);
    for (PoolGroupIndex pgIdx{0}; level == kHotLevel && pgIdx < newNumSlots.size(); ++pgIdx)
    {
        TLLM_CHECK_DEBUG(!mFixedSlots[pgIdx] || newNumSlots[pgIdx] == oldNumSlots[pgIdx]);
    }

    TLLM_CHECK_DEBUG(isLastLevel(level) || persistentPages == nullptr);

    // Shrink first.
    for (PoolGroupIndex pgIdx{0}; pgIdx < newNumSlots.size(); ++pgIdx)
    {
        if (newNumSlots[pgIdx] >= oldNumSlots[pgIdx])
            continue;
        // Shrinking moves or frees this group's slots. Retained copies are duplicates, so drop them rather than
        // relocate them; groups that keep or grow their size keep their copies.
        releaseRetainedCopies(level, pgIdx, numRetainedCopies(level, pgIdx));
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
    CacheLevel level, int tokensPerBlock, int historyLength, int capacity, int beamWidth, int promptLength) const
{
    if (capacity < historyLength)
    {
        TLLM_LOG_WARNING("Bad sampling for capacity and history_length");
        capacity = historyLength;
    }

    int const numBlocks = divUp(capacity, tokensPerBlock);
    int const effBeamWidth = std::max(1, beamWidth);
    // Floors, so the block holding the prompt tail lands on the per-beam side —
    // same boundary KvCache::_appendBeams() uses.
    HalfOpenRange<BlockOrdinal> const sharedRange{0, std::min(std::max(promptLength, 0) / tokensPerBlock, numBlocks)};
    TypedVec<LifeCycleId, size_t> numBytes(numLifeCycles(), 0);
    auto const& lifeCycles = mLifeCycles.getAll();
    for (LifeCycleId lifeCycle{0}; lifeCycle < lifeCycles.size(); ++lifeCycle)
    {
        auto const poolSizes = slotSize(level, getPoolGroupIndex(level, lifeCycle));
        size_t const slotBytes = std::accumulate(poolSizes.begin(), poolSizes.end(), size_t{0});
        int numRequiredBlocks;
        if (mLifeCycles.isSsm(lifeCycle))
        {
            // One recurrent-state block per beam.
            numRequiredBlocks = effBeamWidth;
        }
        else
        {
            auto const stale = getStaleRange(lifeCycles[lifeCycle], historyLength, tokensPerBlock);
            int const nonStale = std::max(numBlocks - stale.length(), 1);
            int const nonStaleShared
                = std::min(sharedRange.length() - intersect(stale, sharedRange).length(), nonStale);
            int const nonStaleBeam = nonStale - nonStaleShared;
            numRequiredBlocks = nonStaleShared + effBeamWidth * nonStaleBeam;
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

TypedVec<PoolGroupIndex, float> StorageManager::coldPoolGroupRatioFromBatch(
    CacheLevel level, BatchDesc const& batch, int tokensPerBlock) const
{
    auto const numSlots = typicalSlotsForBatch(batch, tokensPerBlock, mSwaScratchReuse);
    TypedVec<PoolGroupIndex, double> weights(numPoolGroups(level), 0.0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numLifeCycles(); ++lifeCycle)
    {
        PoolGroupIndex const pgIdx = getPoolGroupIndex(level, lifeCycle);
        auto const sizes = slotSize(level, pgIdx);
        // At least one slot each, so every group gets the positive share ratioToSlotCountList requires.
        weights[pgIdx] += static_cast<double>(std::max(numSlots[lifeCycle], SlotCount{1}))
            * static_cast<double>(std::accumulate(sizes.begin(), sizes.end(), size_t{0}));
    }
    return normalizeToRatio(weights);
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
    std::optional<SwaScratchReuseConfig> const& swaScratchReuse, size_t deviceGranularity,
    TypedVec<PoolGroupIndex, std::vector<PoolIndex>> const& sparsePools) const
{
    return normalizeToRatio(
        slotsToBytes(typicalSlotsForBatch(batch, tokensPerBlock, swaScratchReuse), deviceGranularity, sparsePools));
}

TypedVec<LifeCycleId, SlotCount> StorageManager::typicalSlotsForBatch(
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse) const
{
    auto numSlots = computeSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
    // A soft preference, not a feasibility floor: keep one optional endpoint
    // per typical request in addition to its private working pages.
    for (auto const& [lc, lifeCycle] : mLifeCycles)
    {
        if (reusePolicy(lifeCycle) == ReusePolicy::OPTIONAL)
        {
            numSlots[lc] += slotCountValueFromSize(batch.kvCaches.size());
        }
    }
    return numSlots;
}

// ---------------------------------------------------------------------------
// computeSlotsFromConstraints
// ---------------------------------------------------------------------------

TypedVec<LifeCycleId, SlotCount> StorageManager::computeSlotsFromConstraints(std::vector<BatchDesc> const& constraints,
    int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse, float maxUtilForResume) const
{
    TLLM_CHECK_WITH_INFO(maxUtilForResume > 0.0F && maxUtilForResume <= 1.0F, "max_util_for_resume must be in (0, 1]");
    // All returned elements are positive. Constraint-derived floors include headroom
    // for the utilization gate checked by KvCache::resume.
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
            auto const scaledSlots = static_cast<SlotCount>(
                std::ceil(static_cast<double>(slots[lifeCycle]) / static_cast<double>(maxUtilForResume)));
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
            auto const scaledSlots = static_cast<SlotCount>(
                std::ceil(static_cast<double>(slots[poolGroup]) / static_cast<double>(maxUtilForResume)));
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
    int sysBlocks = batch.systemPromptLength / tokensPerBlock;

    for (auto const& [lcIdx, lc] : mLifeCycles)
    {
        if (mLifeCycles.isSsm(lcIdx))
        {
            // SSM: one dedicated block per request per beam, never shared.
            // _appendBeams() gives every beam its own recurrent-state slot.
            size_t ssmBlocks = 0;
            for (auto const& kv : batch.kvCaches)
            {
                ssmBlocks += static_cast<size_t>(std::max(1, kv.beamWidth));
            }
            numSlots[lcIdx] += slotCountValueFromSize(ssmBlocks);
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

            // Split into the prefix the beams share and the tail each beam owns.
            // The block holding the prompt tail is itself per-beam, so the
            // boundary floors rather than rounds up — this matches
            // _appendBeams()'s firstGenerationBlock exactly.
            int const beamWidth = std::max(1, kv.beamWidth);
            HalfOpenRange<BlockOrdinal> const sharedRange{0, std::min(kv.promptLength / tokensPerBlock, totalBlocks)};
            int const nonStaleShared = sharedRange.length() - intersect(stale, sharedRange).length();
            int const uniqueShared = std::min(std::max(0, nonStaleShared - nonStaleSys), uniqueNonStale);
            int const uniqueBeam = uniqueNonStale - uniqueShared;

            if (swaScratchReuse.has_value())
            {
                auto scratch = computeScratchRange(
                    lc, kv.historyLength, kv.capacity, tokensPerBlock, swaScratchReuse->maxRewindLen);
                int numScratch = scratch.length();
                // Scratch blocks are input blocks, so they sit in the per-beam
                // tail; only a conservative remainder can reach the shared prefix.
                int const scratchBeam = std::min(numScratch, uniqueBeam);
                int const scratchShared = std::min(numScratch - scratchBeam, uniqueShared);
                // Scratch blocks share coalesced slots: actual slots = ceil(numScratch * fracMax).
                numSlots[lcIdx] += (uniqueShared - scratchShared) + mSlotUtilFracMax[lcIdx].ceilMul(scratchShared)
                    + beamWidth * ((uniqueBeam - scratchBeam) + mSlotUtilFracMax[lcIdx].ceilMul(scratchBeam));
            }
            else
            {
                numSlots[lcIdx] += uniqueShared + beamWidth * uniqueBeam;
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

TypedVec<LifeCycleId, size_t> StorageManager::slotsToBytes(TypedVec<LifeCycleId, SlotCount> const& numSlots,
    size_t deviceGranularity, TypedVec<PoolGroupIndex, std::vector<PoolIndex>> const& sparsePools) const
{
    TypedVec<LifeCycleId, size_t> numBytes(numLifeCycles(), 0);
    for (LifeCycleId lifeCycle{0}; lifeCycle < numSlots.size(); ++lifeCycle)
    {
        auto const poolGroup = getPoolGroupIndex(kHotLevel, lifeCycle);
        auto const& sparse = sparsePools.at(poolGroup);
        auto const& sizes = slotSize(kHotLevel, poolGroup);
        for (PoolIndex pool{0}; pool < sizes.size(); ++pool)
        {
            auto const medium
                = std::ranges::find(sparse, pool) == sparse.end() ? CacheTier::GPU_MEM : CacheTier::HOST_MEM;
            numBytes[lifeCycle] += roundUp(
                slotCountToSizeT(numSlots[lifeCycle]) * sizes[pool], hotPoolSizeGranularity(medium, deviceGranularity));
        }
    }
    return numBytes;
}

// ---------------------------------------------------------------------------
// computeSlotCountForLevel
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::computeSlotCountForLevel(CacheTierConfig const& tierConfig,
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    TypedVec<PoolGroupIndex, float> const& ratio, TypedVec<PoolGroupIndex, SlotCount> const& minSlots,
    size_t gpuPhysMemSize, std::optional<SecondaryPoolBudget> const& secondBudget, FixedSlotCounts const& fixedSlots)
{
    size_t quota = cacheTierQuota(tierConfig);
    size_t const granularity = tierCommitUnit(tierConfig, gpuPhysMemSize);
    quota = std::max(minQuotaForLevel(slotSizeLists, granularity, minSlots, secondBudget), roundUp(quota, granularity));
    return CacheLevelStorage::ratioToSlotCountList(
        quota, slotSizeLists, ratio, granularity, minSlots, secondBudget, fixedSlots);
}

//! Host-backed pools round to HostMem::sizeGranularity(), not to the device granularity, so the
//! budget is counted in that grain: charging them in device grains would over-count every
//! pool of every group by up to one device grain.
std::optional<SecondaryPoolBudget> StorageManager::_secondaryBudget(CacheLevelStorage const& level) const
{
    auto const medium = level.secondaryMedium();
    if (!medium.has_value())
    {
        // Nothing here sits outside what this level's quota pays for.
        return std::nullopt;
    }
    return _secondaryBudget(level.poolsOfMedium(*medium), level.poolSizeGranularity(*medium), mHostSparsePoolQuota);
}

std::optional<SecondaryPoolBudget> StorageManager::_secondaryBudget(
    TypedVec<PoolGroupIndex, std::vector<PoolIndex>> secondaryPools, size_t granularity, std::optional<size_t> quota)
{
    // Returned whenever the level has such a pool, bounded or not: the allocator needs to
    // know which pools this level's quota does not pay for before it needs to know whether
    // anything limits them. Only `quota` carries the latter.
    if (std::ranges::none_of(secondaryPools, [](auto const& pools) { return !pools.empty(); }))
    {
        return std::nullopt;
    }
    SecondaryPoolBudget budget;
    budget.quota = quota;
    budget.granularity = granularity;
    budget.pools = std::move(secondaryPools);
    return budget;
}

size_t StorageManager::tierCommitUnit(CacheTierConfig const& tierConfig, size_t gpuPhysMemSize)
{
    // Every GPU level shares one allocator, whose unit was derived from the hot
    // level's quota. A cold GPU level must report that unit rather than one
    // recomputed from its own quota, which is smaller and would not match the
    // chunks the allocator actually hands out.
    return cacheTierOf(tierConfig) == CacheTier::GPU_MEM
        ? gpuPhysMemSize
        : tierAllocationGranularity(cacheTierOf(tierConfig), cacheTierQuota(tierConfig));
}

HostMemBackingOptions StorageManager::tierBackingOptions(
    CacheTierConfig const& tierConfig, size_t gpuPhysMemSize, HostMemBackingConfig const& backing)
{
    HostMemBackingOptions options;
    if (auto const* host = std::get_if<HostCacheTierConfig>(&tierConfig))
    {
        // The tier's deprecated flag can only make the placement stricter.
        options.allowRemoteNumaFallback = backing.allowRemoteNumaFallback && host->allowRemoteNumaFallback;
        options.numaNode = backing.numaNode;
    }
    return options;
}

// ---------------------------------------------------------------------------
// minQuotaForLevel
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> StorageManager::raiseHotMinSlots(BatchDesc const& batch, int tokensPerBlock)
{
    auto const needed
        = computePoolGroupMinSlotsFromConstraints({batch}, tokensPerBlock, mSwaScratchReuse, mMaxUtilForResume);
    for (PoolGroupIndex pgIdx{0}; pgIdx < mHotMinSlots.size() && pgIdx < needed.size(); ++pgIdx)
    {
        SlotCount const capped = std::min(needed[pgIdx], numSlots(pgIdx, kHotLevel));
        mHotMinSlots[pgIdx] = std::max(mHotMinSlots[pgIdx], capped);
    }
    return mHotMinSlots;
}

TypedVec<PoolGroupIndex, SlotCount> StorageManager::_minSlotsFor(CacheLevel level) const
{
    return level == kHotLevel ? mHotMinSlots : TypedVec<PoolGroupIndex, SlotCount>(numPoolGroups(level), SlotCount{1});
}

size_t StorageManager::minQuotaForLevel(CacheLevel level) const
{
    auto const& storage = *mLevels.at(level).storage;
    return minQuotaForLevel(storage.slotSizeLists(), storage.poolSizeGranularity(storage.cacheTier()),
        _minSlotsFor(level), _secondaryBudget(storage));
}

TypedVec<PoolGroupIndex, SlotCount> StorageManager::slotCountsForLevel(
    CacheLevel level, size_t quota, TypedVec<PoolGroupIndex, float> const& ratio) const
{
    auto const& storage = *mLevels.at(level).storage;
    return CacheLevelStorage::ratioToSlotCountList(quota, storage.slotSizeLists(), ratio,
        storage.poolSizeGranularity(storage.cacheTier()), _minSlotsFor(level), _secondaryBudget(storage),
        level == kHotLevel ? mFixedSlots : FixedSlotCounts{});
}

size_t StorageManager::minQuotaForLevel(TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    size_t granularity, TypedVec<PoolGroupIndex, SlotCount> const& minSlots,
    std::optional<SecondaryPoolBudget> const& secondBudget)
{
    auto const pools = CacheLevelStorage::poolsPaidByQuota(slotSizeLists, secondBudget);
    size_t total = 0;
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotSizeLists.size(); ++pgIdx)
    {
        // Only the pools this quota pays for: a sparse pool's floor is charged to the host
        // budget instead, and counting it here would inflate the device quota's own.
        for (PoolIndex poolIdx : pools[pgIdx])
        {
            total += roundUp(slotCountToSizeT(minSlots[pgIdx]) * slotSizeLists[pgIdx][poolIdx], granularity);
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
    // Fixed groups enter at their fixed slot count, so their target matches their current share and never
    // requests an adjustment.
    auto slotCountList = slotCountsForLevel(kHotLevel, gpuStorage.roundedNumBytes(gpuStorage.cacheTier()), ratio);
    // All-bytes, each medium in its own grain -- the same measure ratioList() reports, so
    // the target this derives and the layout a rebalance then achieves are comparable. They
    // have to be: _needAdjustment treats this as a prediction of the allocator's own output
    // and re-fires until the two agree.
    return normalizeToRatio(gpuStorage.allMediaBytes(slotCountList));
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
