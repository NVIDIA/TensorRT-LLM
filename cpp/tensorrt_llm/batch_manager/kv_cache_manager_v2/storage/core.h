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
#include "kv_cache_manager_v2/cudaVirtMem.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/hostMem.h"
#include "kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/common/assert.h"

#include <deque>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Slot — represents ownership of one allocated slot in a pool group.
// Mirrors _storage/_core.py::Slot.
//
// ready_event: completes when the slot's data is safe to use.
//   - Newly allocated: completes when previous users are done.
//   - After migration: completes when the copy is done.
//   - Passed to release(): completes when current users are done.
// ---------------------------------------------------------------------------
struct Slot
{
    CachedCudaEvent readyEvent = CachedCudaEvent::makeNull();

    Slot() = default;

    ~Slot()
    {
        KVCM2_CHECK_FATAL_DEBUG_WITH_INFO(
            !hasValidSlot(), "Slot %zu destroyed without being released", toSizeT(mSlotId.value()));
    }

    // A slot id has exactly one owner: transferring a Slot empties the source, so it cannot be
    // released twice. Copying is not available for the same reason.
    Slot(Slot&& other) noexcept
        : readyEvent(std::move(other.readyEvent))
        , mSlotId(std::exchange(other.mSlotId, std::nullopt))
        , mLocalityDomainId(std::exchange(other.mLocalityDomainId, std::nullopt))
    {
    }

    Slot& operator=(Slot&& other) noexcept
    {
        if (this != &other)
        {
            readyEvent = std::move(other.readyEvent);
            mSlotId = std::exchange(other.mSlotId, std::nullopt);
            mLocalityDomainId = std::exchange(other.mLocalityDomainId, std::nullopt);
        }
        return *this;
    }

    Slot(Slot const&) = delete;
    Slot& operator=(Slot const&) = delete;

    // Mirrors Python @property slot_id: asserts valid, returns unwrapped value.
    [[nodiscard]] SlotId slotId() const
    {
        return mSlotId.value();
    }

    [[nodiscard]] bool hasValidSlot() const noexcept
    {
        return mSlotId.has_value();
    }

    void setSlotId(SlotId id) noexcept
    {
        mSlotId = id;
    }

    // Locality domain whose physical memory backs this slot; nullopt for non-localized storage.
    [[nodiscard]] std::optional<int> localityDomainId() const noexcept
    {
        return mLocalityDomainId;
    }

    void setLocalityDomainId(std::optional<int> id) noexcept
    {
        mLocalityDomainId = id;
    }

    void resetSlot() noexcept
    {
        mSlotId.reset();
        mLocalityDomainId.reset();
    }

    bool queryReady()
    {
        return readyEvent.queryComplete();
    }

    // Transfer slot ownership into an empty slot. Overwriting an occupied slot would drop its
    // id without releasing it, so that is rejected rather than silently allowed by the move.
    void setSlot(Slot& src)
    {
        if (hasValidSlot())
        {
            throw LogicError("Slot::setSlot: already has a valid slot");
        }
        *this = std::move(src);
    }

    // Replace this slot, invalidate replacement, and return the previous slot.
    [[nodiscard]] Slot exchangeSlot(Slot&& replacement)
    {
        TLLM_CHECK(hasValidSlot() && replacement.hasValidSlot());
        Slot previous;
        auto replacementId = std::exchange(replacement.mSlotId, std::nullopt);
        previous.mSlotId = std::exchange(mSlotId, replacementId);
        auto replacementDomain = std::exchange(replacement.mLocalityDomainId, std::nullopt);
        previous.mLocalityDomainId = std::exchange(mLocalityDomainId, replacementDomain);
        previous.readyEvent = std::exchange(readyEvent, std::move(replacement.readyEvent));
        return previous;
    }

private:
    std::optional<SlotId> mSlotId;
    std::optional<int> mLocalityDomainId;
};

// ---------------------------------------------------------------------------
// SlotIdMap — maps an allocator's local slot indices onto chunks of a slot-id space shared by
// several allocators (one per locality domain), so each hands out disjoint global ids.
// Mirrors _storage/_core.py::_SlotIdMap.
// ---------------------------------------------------------------------------
class SlotIdMap
{
public:
    explicit SlotIdMap(SlotCount chunkSize)
        : mChunkSize(chunkSize)
    {
    }

    [[nodiscard]] SlotCount chunkSize() const noexcept
    {
        return mChunkSize;
    }

    [[nodiscard]] std::vector<size_t> const& chunks() const noexcept
    {
        return mChunks;
    }

    void append(size_t chunk);
    size_t pop();

    [[nodiscard]] SlotId slotId(SlotCount localIdx) const noexcept
    {
        return SlotId{static_cast<SlotCount>(mChunks[slotCountToSizeT(localIdx / mChunkSize)]) * mChunkSize
            + localIdx % mChunkSize};
    }

    // nullopt when slotId lies in a chunk this map does not own.
    [[nodiscard]] std::optional<SlotCount> localIndex(SlotId slotId) const noexcept;

private:
    SlotCount mChunkSize;
    std::vector<size_t> mChunks;
    std::unordered_map<size_t, size_t> mChunkIndices;
};

// ---------------------------------------------------------------------------
// SlotAllocator — manages a fixed-capacity array of slot ids.
// Mirrors _storage/_core.py::SlotAllocator.
//
// Without a SlotIdMap the slot id is the local index. With one, capacity and shrink accounting
// stay local while the ids handed out come from the map's chunks.
// ---------------------------------------------------------------------------
class SlotAllocator
{
public:
    explicit SlotAllocator(
        SlotCount capacity, SlotIdMap const* slotMap = nullptr, std::optional<int> localityDomainId = std::nullopt);
    ~SlotAllocator();

    [[nodiscard]] SlotCount numFreeSlots() const noexcept;
    [[nodiscard]] SlotCount numOccupiedSlots() const noexcept;

    [[nodiscard]] SlotCount numSlots() const noexcept
    {
        return mCapacity;
    }

    [[nodiscard]] std::optional<int> localityDomainId() const noexcept
    {
        return mLocalityDomainId;
    }

    // Local index of a slot this allocator issued; nullopt for a foreign slot id.
    [[nodiscard]] std::optional<SlotCount> localIndexOf(SlotId slot) const noexcept
    {
        return localIndex(slot);
    }

    // SlotId is signed, so a bare `slot < numSlots()` admits negatives. Both ends are checked
    // here so no caller has to remember the lower one.
    [[nodiscard]] bool isValidSlotId(SlotId slot) const noexcept
    {
        auto const local = localIndex(slot);
        return local.has_value() && *local >= 0 && *local < numSlots();
    }

    Slot allocate();
    std::vector<Slot> allocateMultiple(SlotCount numSlots);
    void release(Slot slot);

    void expand(SlotCount newNumSlots);
    void prepareForShrink(SlotCount newNumSlots);
    bool finishShrink();

    [[nodiscard]] bool shrinkInProgress() const noexcept
    {
        return mTargetCapacity < mCapacity;
    }

    [[nodiscard]] std::vector<SlotId> getSlotsBlockingShrink() const;

    // Read-only accessors for debug assertions (mirrors Python's direct attribute access).
    [[nodiscard]] SlotCount numOverflowSlots() const noexcept
    {
        return slotCountValueFromSize(mOverflowSlots.size());
    }

    [[nodiscard]] SlotCount numActiveSlots() const noexcept
    {
        return mNumActiveSlots;
    }

    [[nodiscard]] SlotCount targetCapacity() const noexcept
    {
        return mTargetCapacity;
    }

    void synchronize();

private:
    void scrubEvents();
    [[nodiscard]] bool check() const noexcept;

    [[nodiscard]] std::optional<SlotCount> localIndex(SlotId slot) const noexcept
    {
        return mSlotMap == nullptr ? std::optional<SlotCount>{slot.value()} : mSlotMap->localIndex(slot);
    }

    [[nodiscard]] SlotId slotId(SlotCount localIdx) const noexcept
    {
        return mSlotMap == nullptr ? SlotId{localIdx} : mSlotMap->slotId(localIdx);
    }

    SlotCount mCapacity;
    SlotCount mTargetCapacity;
    SlotCount mNumActiveSlots;
    SlotCount mNumReadyRecycledSlots;
    std::deque<Slot> mRecycledSlots;
    std::vector<Slot> mOverflowSlots;
    DynamicBitset mOccupiedMask; // indexed by local index
    SlotIdMap const* mSlotMap;
    std::optional<int> mLocalityDomainId;
};

// ---------------------------------------------------------------------------
// SlotPoolBase — abstract base for a single memory pool.
// Mirrors _storage/_core.py::SlotPoolBase.
// ---------------------------------------------------------------------------
class SlotPoolBase
{
public:
    explicit SlotPoolBase(size_t slotSize)
        : mSlotSize(slotSize)
    {
    }

    virtual ~SlotPoolBase() = default;

    size_t slotSize() const noexcept
    {
        return mSlotSize;
    }

    virtual SlotCount numSlots() const noexcept = 0;

    // SlotId is signed, so a bare `slot < numSlots()` admits negatives, which then convert to a
    // huge size_t offset. Both ends are checked here so no caller has to remember the lower one.
    [[nodiscard]] bool isValidSlotId(SlotId slot) const noexcept
    {
        return slot >= SlotId{0} && slot < numSlots();
    }

    size_t numBytes() const noexcept
    {
        return mSlotSize * slotCountToSizeT(numSlots());
    }

    virtual void destroy() = 0;
    virtual void resize(SlotCount newNumSlots) = 0;
    virtual Address slotAddress(SlotId slot) const = 0;

protected:
    size_t mSlotSize;
};

// ---------------------------------------------------------------------------
// GpuSlotPool — GPU virtual memory pool.
// ---------------------------------------------------------------------------
class GpuSlotPool : public SlotPoolBase
{
public:
    // slotMaps: one per locality domain when the group places each domain's backing at
    // chunk-aligned offsets of a shared slot-id space; nullptr for a single contiguous mapping.
    GpuSlotPool(size_t slotSize, size_t vmSize, PooledPhysMemAllocator& physMemAllocator, SlotCount numSlots,
        std::vector<SlotIdMap> const* slotMaps = nullptr);

    // Slots backed across all locality domains.
    SlotCount numSlots() const noexcept override;
    SlotCount numSlots(int localityDomainId) const noexcept;
    void destroy() override;
    void resize(SlotCount newNumSlots) override;
    void resize(SlotCount newNumSlots, int localityDomainId);
    Address slotAddress(SlotId slot) const override;

    [[nodiscard]] int numLocalityDomains() const noexcept
    {
        return mVirtMem.numLocalityDomains();
    }

    // Merged mapped byte ranges relative to the pool base.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> mappedRanges() const
    {
        return mVirtMem.mappedRanges();
    }

    // Extend by exactly one physical memory chunk; returns new numSlots.
    SlotCount extendByOnePhysMem();

    static size_t computeNumPhysMem(size_t slotSize, SlotCount numSlots, size_t physMemSize) noexcept;
    static SlotCount computeNumSlots(size_t slotSize, size_t numPhysMem, size_t physMemSize) noexcept;

private:
    VirtMem mVirtMem;
    std::vector<SlotIdMap> const* mSlotMaps;
    // Requested slots per domain; the mapped count can exceed it by the tail of a chunk.
    std::vector<SlotCount> mRequestedNumSlots;
};

// ---------------------------------------------------------------------------
// HostSlotPool — pinned host memory pool.
// ---------------------------------------------------------------------------
class HostSlotPool : public SlotPoolBase
{
public:
    HostSlotPool(size_t slotSize, SlotCount numSlots);

    SlotCount numSlots() const noexcept override;
    void destroy() override;
    void resize(SlotCount newNumSlots) override;
    Address slotAddress(SlotId slot) const override;

    size_t alignedSize(SlotCount numSlots) const noexcept;

    [[nodiscard]] HostMem const* hostMem() const noexcept
    {
        return &mHostMem;
    }

private:
    HostMem mHostMem;
};

// ---------------------------------------------------------------------------
// DiskSlotPool — temp-file backed disk pool.
// ---------------------------------------------------------------------------
class DiskSlotPool : public SlotPoolBase
{
public:
    // directory: path under which to create the temp file.
    DiskSlotPool(std::string const& directory, size_t slotSize, SlotCount numSlots);
    ~DiskSlotPool() override;

    SlotCount numSlots() const noexcept override;
    void destroy() override;
    void resize(SlotCount newNumSlots) override;
    Address slotAddress(SlotId slot) const override;

    int fd() const noexcept
    {
        return mFd;
    }

private:
    int mFd = kBadFileDescriptor;
    // Tracks the file size, which only resize() changes: the file is created unlinked and owned
    // solely by this pool, so nothing else can grow or shrink it behind us. Held here because
    // slotAddress() bounds-checks against it on every call, and querying the file would make that
    // an lseek per page of a migration.
    SlotCount mNumSlots{0};
};

// ---------------------------------------------------------------------------
// PoolGroupBase — manages multiple pools with mirrored slot allocation.
// Mirrors _storage/_core.py::PoolGroupBase.
// ---------------------------------------------------------------------------
class PoolGroupBase
{
public:
    explicit PoolGroupBase(SlotCount numSlots);
    virtual ~PoolGroupBase();

    PoolIndex numPools() const noexcept
    {
        return mPools.size();
    }

    int numLocalityDomains() const noexcept
    {
        return static_cast<int>(mSlotAllocators.size());
    }

    SlotCount numSlots(int localityDomainId = 0) const noexcept;

    SlotCount numFreeSlots(int localityDomainId = 0) const noexcept
    {
        return mSlotAllocators[localityDomainId]->numFreeSlots();
    }

    SlotAllocator& slotAllocator(int localityDomainId = 0) noexcept
    {
        return *mSlotAllocators[localityDomainId];
    }

    SlotAllocator const& slotAllocator(int localityDomainId = 0) const noexcept
    {
        return *mSlotAllocators[localityDomainId];
    }

    Slot allocate(int localityDomainId = 0);
    std::vector<Slot> allocateMultiple(SlotCount numSlots, int localityDomainId = 0);
    // Returns the slot to the allocator of the domain that issued it.
    void release(Slot slot);
    void destroy();
    virtual void resizePools(std::optional<SlotCount> newNumSlots = std::nullopt, int localityDomainId = 0);

    // Usable global slot ranges [begin, end), merged and sorted, excluding unmapped holes.
    virtual std::vector<std::pair<SlotCount, SlotCount>> slotRanges() const;

    // One past the highest usable slot id; equals numSlots() for a contiguous group.
    SlotCount slotIndexUpperBound() const
    {
        SlotCount upperBound = 0;
        for (auto const& [begin, end] : slotRanges())
        {
            upperBound = std::max(upperBound, end);
        }
        return upperBound;
    }

    // Addresses for all pools at a given slot id.
    TypedVec<PoolIndex, Address> slotAddress(SlotId slotId) const;
    // Sizes per pool.
    TypedVec<PoolIndex, size_t> slotSize() const;

    // Total bytes across all pools for one slot.
    size_t numBytes() const noexcept
    {
        size_t total = 0;
        for (auto const& pool : mPools)
            total += pool->numBytes();
        return total;
    }

    // Total bytes across all pools, rounding each pool's bytes up to granularity.
    // Mirrors Python total_quota: sum(round_up(p.num_bytes, granularity) for p in pg._pools).
    size_t roundedNumBytes(size_t granularity) const noexcept
    {
        size_t total = 0;
        for (auto const& pool : mPools)
            total += roundUp(pool->numBytes(), granularity);
        return total;
    }

protected:
    // Mirrors Python _get_num_slots_from_pools: min(p.num_slots for p in pools).
    virtual SlotCount getNumSlotsFromPools(int localityDomainId = 0) const noexcept;

    std::vector<std::unique_ptr<SlotAllocator>> mSlotAllocators;
    TypedVec<PoolIndex, std::unique_ptr<SlotPoolBase>> mPools;
    bool mDestroyed = false;
};

// ---------------------------------------------------------------------------
// GpuPoolGroup / HostPoolGroup / DiskPoolGroup
// ---------------------------------------------------------------------------

// With several locality domains the pools share one slot-id space: ids are handed out in chunks
// whose byte extent is physically aligned in every pool, each chunk backed by one domain's memory.
class GpuPoolGroup : public PoolGroupBase
{
public:
    // numSlots: initial slot count per locality domain (size numLocalityDomains()).
    GpuPoolGroup(std::vector<SlotCount> const& numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList,
        PooledPhysMemAllocator& physMemAllocator);

    void resizePools(std::optional<SlotCount> newNumSlots = std::nullopt, int localityDomainId = 0) override;
    std::vector<std::pair<SlotCount, SlotCount>> slotRanges() const override;

    [[nodiscard]] std::vector<std::pair<size_t, size_t>> mappedRanges(PoolIndex poolIdx) const
    {
        return static_cast<GpuSlotPool const&>(*mPools.at(poolIdx)).mappedRanges();
    }

protected:
    SlotCount getNumSlotsFromPools(int localityDomainId = 0) const noexcept override;

private:
    // Grow or shrink a domain's chunk list to cover numSlots.
    void resizeSlotMap(SlotCount numSlots, int localityDomainId);

    std::vector<SlotIdMap> mSlotMaps; // empty with one locality domain
    SlotCount mChunkSize{0};
    size_t mMaxChunks{0};
    size_t mNumChunks{0};
    std::priority_queue<size_t, std::vector<size_t>, std::greater<>> mFreeChunks;
};

class HostPoolGroup : public PoolGroupBase
{
public:
    HostPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList);

    [[nodiscard]] HostMem const* hostMem(PoolIndex poolIndex) const;
};

class DiskPoolGroup : public PoolGroupBase
{
public:
    DiskPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, std::string const& directory);
};

// ---------------------------------------------------------------------------
// CacheLevelStorage — manages all pool groups for one cache tier.
// Mirrors _storage/_core.py::CacheLevelStorage.
// ---------------------------------------------------------------------------
class CacheLevelStorage
{
public:
    virtual ~CacheLevelStorage() = default;

    virtual CacheTier cacheTier() const noexcept = 0;

    PoolGroupIndex numPoolGroups() const noexcept
    {
        return mPoolGroups.size();
    }

    std::vector<Slot> allocateMultiple(PoolGroupIndex pgIdx, SlotCount numSlots, int localityDomainId = 0);
    void release(PoolGroupIndex pgIdx, Slot slot);
    SlotCount numFreeSlots(PoolGroupIndex pgIdx, int localityDomainId = 0) const;
    TypedVec<PoolIndex, Address> slotAddress(PoolGroupIndex pgIdx, SlotId slotId) const;

    // 1 unless this is a localized GPU level.
    virtual int numLocalityDomains() const noexcept
    {
        return 1;
    }

    // Additional accessors used by StorageManager and KvCacheManager.
    virtual void destroy()
    {
        for (auto& pg : mPoolGroups)
            pg->destroy();
    }

    PoolIndex numPools(PoolGroupIndex pgIdx) const
    {
        return mPoolGroups.at(pgIdx)->numPools();
    }

    SlotCount numSlots(PoolGroupIndex pgIdx, int localityDomainId = 0) const
    {
        return mPoolGroups.at(pgIdx)->numSlots(localityDomainId);
    }

    SlotCount slotIndexUpperBound(PoolGroupIndex pgIdx) const
    {
        return mPoolGroups.at(pgIdx)->slotIndexUpperBound();
    }

    std::vector<std::pair<SlotCount, SlotCount>> slotRanges(PoolGroupIndex pgIdx) const
    {
        return mPoolGroups.at(pgIdx)->slotRanges();
    }

    TypedVec<PoolIndex, size_t> slotSize(PoolGroupIndex pgIdx) const
    {
        auto szl = mPoolGroups.at(pgIdx)->slotSize();
        TypedVec<PoolIndex, size_t> ret;
        ret.reserve(szl.size());
        for (auto sz : szl)
            ret.push_back(sz);
        return ret;
    }

    PoolGroupBase& poolGroup(PoolGroupIndex pgIdx)
    {
        return *mPoolGroups.at(pgIdx);
    }

    MemAddress getBaseAddress(PoolGroupIndex pgIdx, PoolIndex poolIdx, SlotId slotId) const
    {
        return std::get<MemAddress>(mPoolGroups.at(pgIdx)->slotAddress(slotId).at(poolIdx));
    }

    size_t totalQuota() const noexcept
    {
        size_t granularity = poolSizeGranularity();
        size_t total = 0;
        for (auto const& pg : mPoolGroups)
            total += pg->roundedNumBytes(granularity);
        return total;
    }

    // Returns numSlots(localityDomainId) per pool group.
    TypedVec<PoolGroupIndex, SlotCount> slotCountList(int localityDomainId = 0) const
    {
        TypedVec<PoolGroupIndex, SlotCount> ret;
        ret.reserve(mPoolGroups.size());
        for (auto const& pg : mPoolGroups)
            ret.push_back(pg->numSlots(localityDomainId));
        return ret;
    }

    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> slotSizeLists() const
    {
        TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> ret;
        ret.reserve(mPoolGroups.size());
        for (auto const& pg : mPoolGroups)
        {
            auto szl = pg->slotSize();
            TypedVec<PoolIndex, size_t> sizes;
            sizes.reserve(szl.size());
            for (auto sz : szl)
                sizes.push_back(sz);
            ret.push_back(std::move(sizes));
        }
        return ret;
    }

    // Current ratio list: proportion of bytes per pool group.
    TypedVec<PoolGroupIndex, float> ratioList() const
    {
        TypedVec<PoolGroupIndex, float> ret(mPoolGroups.size(), 0.f);
        float total = 0.f;
        for (PoolGroupIndex pgIdx{0}; pgIdx < mPoolGroups.size(); ++pgIdx)
        {
            auto sz = static_cast<float>(mPoolGroups[pgIdx]->numBytes());
            ret[pgIdx] = sz;
            total += sz;
        }
        TLLM_CHECK_DEBUG(total > 0.f);
        for (auto& r : ret)
            r /= total;
        return ret;
    }

    virtual size_t poolSizeGranularity() const noexcept
    {
        return size_t{2} << 20;
    }

    // Compute slot counts per pool group for a given ratio, min_slots, and optional quota.
    // Instance convenience method — delegates to the static version below.
    TypedVec<PoolGroupIndex, SlotCount> computeSlotCountList(TypedVec<PoolGroupIndex, float> const& ratioList,
        TypedVec<PoolGroupIndex, SlotCount> const& minSlots, std::optional<size_t> quota = std::nullopt) const;

    // Static version: compute slot counts from quota, slot size lists, ratio, granularity, and min_slots.
    // Mirrors Python CacheLevelStorage.ratio_to_slot_count_list.
    static TypedVec<PoolGroupIndex, SlotCount> ratioToSlotCountList(size_t quota,
        TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
        TypedVec<PoolGroupIndex, float> const& ratioList, size_t granularity,
        TypedVec<PoolGroupIndex, SlotCount> const& minSlots);

    // Distribute grains among pools within a pool group.
    // Returns {num_slots, grains_consumed}.
    static std::pair<SlotCount, size_t> grainsToSlots(
        size_t pgGrains, TypedVec<PoolIndex, size_t> const& slotSizeList, size_t granularity);

    // Compute minimum grains needed for numSlots in a pool group.
    static size_t grainsForSlots(
        SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, size_t granularity);

protected:
    TypedVec<PoolGroupIndex, std::unique_ptr<PoolGroupBase>> mPoolGroups;
};

class GpuCacheLevelStorage : public CacheLevelStorage
{
public:
    GpuCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
        TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, PooledPhysMemAllocator& physMemAllocator);

    CacheTier cacheTier() const noexcept override
    {
        return CacheTier::GPU_MEM;
    }

    size_t poolSizeGranularity() const noexcept override
    {
        return mPhysMemAllocator.physMemSize();
    }

    int numLocalityDomains() const noexcept override
    {
        return mPhysMemAllocator.numLocalityDomains();
    }

    std::vector<std::pair<size_t, size_t>> mappedRanges(PoolGroupIndex pgIdx, PoolIndex poolIdx) const
    {
        return static_cast<GpuPoolGroup const&>(*mPoolGroups.at(pgIdx)).mappedRanges(poolIdx);
    }

private:
    PooledPhysMemAllocator& mPhysMemAllocator;
};

class HostCacheLevelStorage : public CacheLevelStorage
{
public:
    HostCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
        TypedVec<PoolGroupIndex, SlotCount> const& slotCountList);

    CacheTier cacheTier() const noexcept override
    {
        return CacheTier::HOST_MEM;
    }

    size_t poolSizeGranularity() const noexcept override
    {
        return HostMem::kAlignment;
    }
};

class DiskCacheLevelStorage : public CacheLevelStorage
{
public:
    DiskCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
        TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, std::string directory);

    CacheTier cacheTier() const noexcept override
    {
        return CacheTier::DISK;
    }

    std::string const& directory() const noexcept
    {
        return mDirectory;
    }

private:
    std::string mDirectory;
};

// Factory: create appropriate CacheLevelStorage for a given tier config.
std::unique_ptr<CacheLevelStorage> createCacheLevelStorage(CacheTierConfig const& tierCfg,
    TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList,
    PooledPhysMemAllocator* gpuPhysMemAllocator = nullptr);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
