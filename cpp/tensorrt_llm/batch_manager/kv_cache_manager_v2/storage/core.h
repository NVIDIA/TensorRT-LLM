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

    void resetSlot() noexcept
    {
        mSlotId.reset();
    }

    bool queryReady()
    {
        return readyEvent.queryComplete();
    }

    // Transfer slot ownership: moves slotId and readyEvent from src to this.
    void setSlot(Slot& src)
    {
        if (hasValidSlot())
        {
            throw LogicError("Slot::setSlot: already has a valid slot");
        }
        mSlotId = src.mSlotId;
        readyEvent = std::move(src.readyEvent);
        src.mSlotId.reset();
    }

private:
    std::optional<SlotId> mSlotId;
};

// ---------------------------------------------------------------------------
// SlotAllocator — manages a fixed-capacity array of slot ids.
// Mirrors _storage/_core.py::SlotAllocator.
// ---------------------------------------------------------------------------
class SlotAllocator
{
public:
    explicit SlotAllocator(SlotCount capacity);
    ~SlotAllocator();

    [[nodiscard]] SlotCount numFreeSlots() const noexcept;
    [[nodiscard]] SlotCount numOccupiedSlots() const noexcept;

    [[nodiscard]] SlotCount numSlots() const noexcept
    {
        return mCapacity;
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

    SlotCount mCapacity;
    SlotCount mTargetCapacity;
    SlotCount mNumActiveSlots;
    SlotCount mNumReadyRecycledSlots;
    std::deque<Slot> mRecycledSlots;
    std::vector<Slot> mOverflowSlots;
    DynamicBitset mOccupiedMask;
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
    GpuSlotPool(size_t slotSize, size_t vmSize, PooledPhysMemAllocator& physMemAllocator, SlotCount numSlots);

    SlotCount numSlots() const noexcept override;
    void destroy() override;
    void resize(SlotCount newNumSlots) override;
    Address slotAddress(SlotId slot) const override;

    // Extend by exactly one physical memory chunk; returns new numSlots.
    SlotCount extendByOnePhysMem();

    static size_t computeNumPhysMem(size_t slotSize, SlotCount numSlots, size_t physMemSize) noexcept;
    static SlotCount computeNumSlots(size_t slotSize, size_t numPhysMem, size_t physMemSize) noexcept;

private:
    VirtMem mVirtMem;
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

    SlotCount numSlots() const noexcept;

    SlotCount numFreeSlots() const noexcept
    {
        return mSlotAllocator.numFreeSlots();
    }

    SlotAllocator& slotAllocator() noexcept
    {
        return mSlotAllocator;
    }

    SlotAllocator const& slotAllocator() const noexcept
    {
        return mSlotAllocator;
    }

    Slot allocate();
    std::vector<Slot> allocateMultiple(SlotCount numSlots);
    void release(Slot slot);
    void destroy();
    void resizePools(std::optional<SlotCount> newNumSlots = std::nullopt);

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
    SlotCount getNumSlotsFromPools() const noexcept;

    SlotAllocator mSlotAllocator;
    TypedVec<PoolIndex, std::unique_ptr<SlotPoolBase>> mPools;
    bool mDestroyed = false;
};

// ---------------------------------------------------------------------------
// GpuPoolGroup / HostPoolGroup / DiskPoolGroup
// ---------------------------------------------------------------------------
class GpuPoolGroup : public PoolGroupBase
{
public:
    GpuPoolGroup(
        SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, PooledPhysMemAllocator& physMemAllocator);
};

class HostPoolGroup : public PoolGroupBase
{
public:
    HostPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList);
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

    std::vector<Slot> allocateMultiple(PoolGroupIndex pgIdx, SlotCount numSlots);
    void release(PoolGroupIndex pgIdx, Slot slot);
    SlotCount numFreeSlots(PoolGroupIndex pgIdx) const;
    TypedVec<PoolIndex, Address> slotAddress(PoolGroupIndex pgIdx, SlotId slotId) const;

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

    SlotCount numSlots(PoolGroupIndex pgIdx) const
    {
        return mPoolGroups.at(pgIdx)->numSlots();
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

    // Returns numSlots() per pool group.
    TypedVec<PoolGroupIndex, SlotCount> slotCountList() const
    {
        TypedVec<PoolGroupIndex, SlotCount> ret;
        ret.reserve(mPoolGroups.size());
        for (auto const& pg : mPoolGroups)
            ret.push_back(pg->numSlots());
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

    virtual void postResize() {}

protected:
    TypedVec<PoolGroupIndex, std::unique_ptr<PoolGroupBase>> mPoolGroups;
};

class GpuCacheLevelStorage : public CacheLevelStorage
{
public:
    GpuCacheLevelStorage(
        StorageConfig const& storageCfg, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, size_t physMemSize);

    CacheTier cacheTier() const noexcept override
    {
        return CacheTier::GPU_MEM;
    }

    size_t poolSizeGranularity() const noexcept override
    {
        return mPhysMemAllocator->physMemSize();
    }

    void postResize() override
    {
        CacheLevelStorage::postResize();
        mPhysMemAllocator->clear();
    }

    void destroy() override
    {
        CacheLevelStorage::destroy();
        mPhysMemAllocator->clear();
    }

private:
    std::unique_ptr<PooledPhysMemAllocator> mPhysMemAllocator;
};

class HostCacheLevelStorage : public CacheLevelStorage
{
public:
    HostCacheLevelStorage(StorageConfig const& storageCfg, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList);

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
    DiskCacheLevelStorage(StorageConfig const& storageCfg, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList,
        std::string directory);

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
    StorageConfig const& storageCfg, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
