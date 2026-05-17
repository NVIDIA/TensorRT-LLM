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
    SlotId slotId() const
    {
        return mSlotId.value();
    }

    bool hasValidSlot() const noexcept
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
    explicit SlotAllocator(int capacity);
    ~SlotAllocator();

    int numFreeSlots() const noexcept;
    int numOccupiedSlots() const noexcept;

    int numSlots() const noexcept
    {
        return mCapacity;
    }

    Slot allocate();
    std::vector<Slot> allocateMultiple(int numSlots);
    void release(Slot slot);

    void expand(int newNumSlots);
    void prepareForShrink(int newNumSlots);
    bool finishShrink();

    bool shrinkInProgress() const noexcept
    {
        return mTargetCapacity < mCapacity;
    }

    std::vector<SlotId> getSlotsBlockingShrink() const;

    // Read-only accessors for debug assertions (mirrors Python's direct attribute access).
    int numOverflowSlots() const noexcept
    {
        return static_cast<int>(mOverflowSlots.size());
    }

    int numActiveSlots() const noexcept
    {
        return mNumActiveSlots;
    }

    int targetCapacity() const noexcept
    {
        return mTargetCapacity;
    }

    void synchronize();

private:
    void scrubEvents();
    bool check() const noexcept;

    int mCapacity;
    int mTargetCapacity;
    int mNumActiveSlots;
    int mNumReadyRecycledSlots;
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

    virtual int numSlots() const noexcept = 0;

    size_t numBytes() const noexcept
    {
        return mSlotSize * static_cast<size_t>(numSlots());
    }

    virtual void destroy() = 0;
    virtual void resize(int newNumSlots) = 0;
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
    GpuSlotPool(size_t slotSize, size_t vmSize, PooledPhysMemAllocator& physMemAllocator, int numSlots);

    int numSlots() const noexcept override;
    void destroy() override;
    void resize(int newNumSlots) override;
    Address slotAddress(SlotId slot) const override;

    // Extend by exactly one physical memory chunk; returns new numSlots.
    int extendByOnePhysMem();

    static int computeNumPhysMem(size_t slotSize, int numSlots, size_t physMemSize) noexcept;
    static int computeNumSlots(size_t slotSize, int numPhysMem, size_t physMemSize) noexcept;

private:
    VirtMem mVirtMem;
};

// ---------------------------------------------------------------------------
// HostSlotPool — pinned host memory pool.
// ---------------------------------------------------------------------------
class HostSlotPool : public SlotPoolBase
{
public:
    HostSlotPool(size_t slotSize, int numSlots);

    int numSlots() const noexcept override;
    void destroy() override;
    void resize(int newNumSlots) override;
    Address slotAddress(SlotId slot) const override;

    size_t alignedSize(int numSlots) const noexcept;

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
    DiskSlotPool(std::string const& directory, size_t slotSize, int numSlots);
    ~DiskSlotPool() override;

    int numSlots() const noexcept override;
    void destroy() override;
    void resize(int newNumSlots) override;
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
    explicit PoolGroupBase(int numSlots);
    virtual ~PoolGroupBase();

    int numPools() const noexcept
    {
        return static_cast<int>(mPools.size());
    }

    int numSlots() const noexcept;

    int numFreeSlots() const noexcept
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
    std::vector<Slot> allocateMultiple(int numSlots);
    void release(Slot slot);
    void destroy();
    void resizePools(std::optional<int> newNumSlots = std::nullopt);

    // Addresses for all pools at a given slot id.
    std::vector<Address> slotAddress(SlotId slotId) const;
    // Sizes per pool.
    std::vector<size_t> slotSize() const;

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
    size_t roundedNumBytes(int granularity) const noexcept
    {
        size_t total = 0;
        for (auto const& pool : mPools)
            total += roundUp(pool->numBytes(), static_cast<size_t>(granularity));
        return total;
    }

protected:
    // Mirrors Python _get_num_slots_from_pools: min(p.num_slots for p in pools).
    int getNumSlotsFromPools() const noexcept;

    SlotAllocator mSlotAllocator;
    std::vector<std::unique_ptr<SlotPoolBase>> mPools;
    bool mDestroyed = false;
};

// ---------------------------------------------------------------------------
// GpuPoolGroup / HostPoolGroup / DiskPoolGroup
// ---------------------------------------------------------------------------
class GpuPoolGroup : public PoolGroupBase
{
public:
    GpuPoolGroup(int numSlots, std::vector<size_t> const& slotSizeList, PooledPhysMemAllocator& physMemAllocator);
};

class HostPoolGroup : public PoolGroupBase
{
public:
    HostPoolGroup(int numSlots, std::vector<size_t> const& slotSizeList);
};

class DiskPoolGroup : public PoolGroupBase
{
public:
    DiskPoolGroup(int numSlots, std::vector<size_t> const& slotSizeList, std::string const& directory);
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

    int numPoolGroups() const noexcept
    {
        return static_cast<int>(mPoolGroups.size());
    }

    std::vector<Slot> allocateMultiple(PoolGroupIndex pgIdx, int numSlots);
    void release(PoolGroupIndex pgIdx, Slot slot);
    int numFreeSlots(PoolGroupIndex pgIdx) const;
    std::vector<Address> slotAddress(PoolGroupIndex pgIdx, SlotId slotId) const;

    // Additional accessors used by StorageManager and KvCacheManager.
    virtual void destroy()
    {
        for (auto& pg : mPoolGroups)
            pg->destroy();
    }

    int numPools(PoolGroupIndex pgIdx) const
    {
        return mPoolGroups.at(static_cast<size_t>(pgIdx))->numPools();
    }

    int numSlots(PoolGroupIndex pgIdx) const
    {
        return mPoolGroups.at(static_cast<size_t>(pgIdx))->numSlots();
    }

    std::vector<int> slotSize(PoolGroupIndex pgIdx) const
    {
        auto szl = mPoolGroups.at(static_cast<size_t>(pgIdx))->slotSize();
        return std::vector<int>(szl.begin(), szl.end());
    }

    PoolGroupBase& poolGroup(PoolGroupIndex pgIdx)
    {
        return *mPoolGroups.at(static_cast<size_t>(pgIdx));
    }

    MemAddress getBaseAddress(PoolGroupIndex pgIdx, PoolIndex poolIdx, SlotId slotId) const
    {
        return std::get<MemAddress>(
            mPoolGroups.at(static_cast<size_t>(pgIdx))->slotAddress(slotId).at(static_cast<size_t>(poolIdx)));
    }

    size_t totalQuota() const noexcept
    {
        int granularity = poolSizeGranularity();
        size_t total = 0;
        for (auto const& pg : mPoolGroups)
            total += pg->roundedNumBytes(granularity);
        return total;
    }

    // Returns numSlots() per pool group.
    std::vector<int> slotCountList() const
    {
        std::vector<int> ret;
        ret.reserve(mPoolGroups.size());
        for (auto const& pg : mPoolGroups)
            ret.push_back(pg->numSlots());
        return ret;
    }

    // Returns slotSize lists per pool group (vector of vectors).
    std::vector<std::vector<int>> slotSizeLists() const
    {
        std::vector<std::vector<int>> ret;
        ret.reserve(mPoolGroups.size());
        for (auto const& pg : mPoolGroups)
        {
            auto szl = pg->slotSize();
            ret.emplace_back(szl.begin(), szl.end());
        }
        return ret;
    }

    // Current ratio list: proportion of bytes per pool group.
    std::vector<float> ratioList() const
    {
        int numPg = static_cast<int>(mPoolGroups.size());
        std::vector<float> ret(static_cast<size_t>(numPg), 0.f);
        float total = 0.f;
        for (int i = 0; i < numPg; ++i)
        {
            auto sz = static_cast<float>(mPoolGroups[i]->numBytes());
            ret[i] = sz;
            total += sz;
        }
        assert(total > 0.f);
        for (auto& r : ret)
            r /= total;
        return ret;
    }

    virtual int poolSizeGranularity() const noexcept
    {
        return 2 << 20;
    }

    // Compute slot counts per pool group for a given ratio, min_slots, and optional quota.
    // Instance convenience method — delegates to the static version below.
    std::vector<int> computeSlotCountList(std::vector<float> const& ratioList, std::vector<int> const& minSlots,
        std::optional<size_t> quota = std::nullopt) const;

    // Static version: compute slot counts from quota, slot size lists, ratio, granularity, and min_slots.
    // Mirrors Python CacheLevelStorage.ratio_to_slot_count_list.
    static std::vector<int> ratioToSlotCountList(size_t quota, std::vector<std::vector<int>> const& slotSizeLists,
        std::vector<float> const& ratioList, int granularity, std::vector<int> const& minSlots);

    virtual void postResize() {}

protected:
    std::vector<std::unique_ptr<PoolGroupBase>> mPoolGroups;

private:
    // Distribute grains among pools within a pool group.
    // Returns {num_slots, grains_consumed}.
    static std::pair<int, int64_t> grainsToSlots(
        int64_t pgGrains, std::vector<int> const& slotSizeList, int granularity);

    // Compute minimum grains needed for numSlots in a pool group.
    static int64_t grainsForSlots(int numSlots, std::vector<int> const& slotSizeList, int granularity);
};

class GpuCacheLevelStorage : public CacheLevelStorage
{
public:
    GpuCacheLevelStorage(StorageConfig const& storageCfg, std::vector<int> const& slotCountList, size_t physMemSize);

    CacheTier cacheTier() const noexcept override
    {
        return CacheTier::GPU_MEM;
    }

    int poolSizeGranularity() const noexcept override
    {
        return static_cast<int>(mPhysMemAllocator->physMemSize());
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
    HostCacheLevelStorage(StorageConfig const& storageCfg, std::vector<int> const& slotCountList);

    CacheTier cacheTier() const noexcept override
    {
        return CacheTier::HOST_MEM;
    }

    int poolSizeGranularity() const noexcept override
    {
        return static_cast<int>(HostMem::kAlignment);
    }
};

class DiskCacheLevelStorage : public CacheLevelStorage
{
public:
    DiskCacheLevelStorage(
        StorageConfig const& storageCfg, std::vector<int> const& slotCountList, std::string directory);

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
std::unique_ptr<CacheLevelStorage> createCacheLevelStorage(
    CacheTierConfig const& tierCfg, StorageConfig const& storageCfg, std::vector<int> const& slotCountList);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
