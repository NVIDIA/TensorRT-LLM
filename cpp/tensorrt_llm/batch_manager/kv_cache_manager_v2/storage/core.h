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
#include <ranges>
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
    {
    }

    Slot& operator=(Slot&& other) noexcept
    {
        if (this != &other)
        {
            readyEvent = std::move(other.readyEvent);
            mSlotId = std::exchange(other.mSlotId, std::nullopt);
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

    void resetSlot() noexcept
    {
        mSlotId.reset();
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
        previous.readyEvent = std::exchange(readyEvent, std::move(replacement.readyEvent));
        return previous;
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

    // SlotId is signed, so a bare `slot < numSlots()` admits negatives. Both ends are checked
    // here so no caller has to remember the lower one.
    [[nodiscard]] bool isValidSlotId(SlotId slot) const noexcept
    {
        return slot >= SlotId{0} && slot < numSlots();
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

//! Granularity a disk pool is sized in. Disk has no allocation unit to match,
//! so this is just a rounding quantum.
inline constexpr size_t kDiskPoolSizeGranularity = size_t{2} << 20;

//! The grain a hot-level pool of `medium` rounds its bytes to, where
//! `deviceGranularity` is the VMM allocation granularity device pools grow in.
//!
//! For the config stage, which sizes pools before any exists to be asked. Once storage is
//! built, ask the group: PoolGroupBase::poolSizeGranularity(CacheTier). The two agree
//! because a host-backed pool commits in hostMemSizeGranularity() whatever level it is on.
inline size_t hotPoolSizeGranularity(CacheTier medium, size_t deviceGranularity) noexcept
{
    return medium == CacheTier::HOST_MEM ? hostMemSizeGranularity() : deviceGranularity;
}

//! Unit a tier of the given size allocates and releases physical memory in,
//! and therefore the quantum its pool sizes are rounded to.
//!
//! For GPU it scales with the tier: 2 MiB for small tiers, rising to 32 MiB, beyond
//! which larger units measure no faster. A resize can give memory back only in whole
//! units, so this trades that precision against the per-allocation cost.
//!
//! This is the policy. What a given StorageManager actually uses for a tier can
//! differ -- see StorageManager::tierCommitUnit -- because every GPU level
//! shares one allocator.
[[nodiscard]] size_t tierAllocationGranularity(CacheTier tier, size_t quota);

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

    //! The unit this pool's backing is allocated in, and so the grain its byte total
    //! rounds to. A property of the allocator behind the pool, not of the level it
    //! sits in: two pools of one group can answer differently.
    [[nodiscard]] virtual size_t sizeGranularity() const noexcept = 0;

    //! What this pool's pages are. A group asks its pools rather than holding the answer,
    //! so a group of one tier may still hold a pool of another.
    [[nodiscard]] virtual CacheTier medium() const noexcept = 0;

    //! Block until the backing for every slot up to `slot` exists.
    //!
    //! Only a progressively filled pool has anything to wait for; the rest are backed when
    //! they are built, so this does nothing for them.
    virtual void waitSlotUsable(SlotId) const {}

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

    //! The VMM allocation granularity, which is what this pool's mapping grows in.
    [[nodiscard]] size_t sizeGranularity() const noexcept override
    {
        return mVirtMem.physMemSize();
    }

    [[nodiscard]] CacheTier medium() const noexcept override
    {
        return CacheTier::GPU_MEM;
    }

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
    //! vmSize is the address space reserved up front, and so the bound on every
    //! later resize. Reserving it all at once is what keeps the base address
    //! fixed, as it does for GpuSlotPool.
    //!
    //! The backing memory is committed on a worker thread, so a slot may exist
    //! before it is backed. Closing that gap is the owning pool group's job: it
    //! calls waitSlotUsable() before returning a slot from allocate().
    HostSlotPool(size_t slotSize, SlotCount numSlots, size_t vmSize, HostMemBackingOptions const& options);

    SlotCount numSlots() const noexcept override;
    void destroy() override;
    void resize(SlotCount newNumSlots) override;
    Address slotAddress(SlotId slot) const override;

    size_t alignedSize(SlotCount numSlots) const noexcept;

    //! The commit unit, which alignedSize() already rounds to.
    [[nodiscard]] size_t sizeGranularity() const noexcept override
    {
        return mCommitUnit;
    }

    [[nodiscard]] CacheTier medium() const noexcept override
    {
        return CacheTier::HOST_MEM;
    }

    //! Blocks until `slot` is backed by physical memory, rethrowing the fill worker's
    //! exception if the fill failed.
    void waitSlotUsable(SlotId slot) const override;

private:
    //! Declared before mHostMem: alignedSize() reads it while building mHostMem.
    size_t mCommitUnit;
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

    [[nodiscard]] size_t sizeGranularity() const noexcept override
    {
        return kDiskPoolSizeGranularity;
    }

    [[nodiscard]] CacheTier medium() const noexcept override
    {
        return CacheTier::DISK;
    }

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

    //! The tier this pool group belongs to, which is what names it to the levels above:
    //! which quota pays for it, which budget it is rebalanced within.
    //!
    //! A property of the pool group rather than of its level, because groups of one level
    //! can differ. This is not necessarily the medium of any given pool's pages -- see
    //! poolMedium().
    [[nodiscard]] virtual CacheTier cacheTier() const noexcept = 0;

    //! What one pool's pages actually are. This is cacheTier() for every pool except a
    //! host-backed pool of a GPU-tier group, where the group's tier would name device
    //! memory for a host address.
    [[nodiscard]] CacheTier poolMedium(PoolIndex poolIdx) const
    {
        return mPools.at(poolIdx)->medium();
    }

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

    //! Allocate, then block until the slots handed back are backed.
    //!
    //! The wait is the whole of a group's fill synchronization, and this is the single
    //! point every later use of a slot passes through. Nothing downstream -- not
    //! slotAddress(), not a device mapping read from it -- waits again. A pool with no
    //! fill to wait on answers at once.
    Slot allocate();
    std::vector<Slot> allocateMultiple(SlotCount numSlots);
    void release(Slot slot);
    void destroy();
    void resizePools(std::optional<SlotCount> newNumSlots = std::nullopt);

    // Addresses for all pools at a given slot id.
    TypedVec<PoolIndex, Address> slotAddress(SlotId slotId) const;
    // Sizes per pool.
    TypedVec<PoolIndex, size_t> slotSize() const;

    //! The grain a pool of `medium` rounds its byte total to. Media of one group differ --
    //! a host-backed pool of a GPU group commits in host units, not in the device's VMM
    //! granularity -- but pools sharing a medium do not: the grain follows the medium's
    //! allocator, not the pool's slot size, which registration bounds absorb instead.
    //!
    //! Returns 0 if no pool of `medium` is here to ask.
    [[nodiscard]] size_t poolSizeGranularity(CacheTier medium) const
    {
        auto ofMedium = std::views::iota(0, mPools.size().value())
            | std::views::filter([this, medium](int pool) { return poolMedium(PoolIndex{pool}) == medium; });
        if (std::ranges::empty(ofMedium))
        {
            return 0;
        }
        return getUniformAttribute(
            ofMedium, [this](int pool) { return mPools.at(PoolIndex{pool})->sizeGranularity(); });
    }

    //! The medium of this group's pools that are not of its tier, if it has any. At most
    //! one: a group's pools are its tier's own or of a single other medium.
    //!
    //! This is the group's half of what SecondaryPoolBudget calls the second medium. Ask
    //! it rather than comparing against cacheTier() at each site: a group of the host tier
    //! has no second medium even though all its pools are host memory, and a test written
    //! as "pools that are HOST_MEM" names every one of them.
    [[nodiscard]] std::optional<CacheTier> secondaryMedium() const
    {
        auto offTier = std::views::iota(0, mPools.size().value())
            | std::views::filter([this](int pool) { return poolMedium(PoolIndex{pool}) != cacheTier(); });
        if (std::ranges::empty(offTier))
        {
            return std::nullopt;
        }
        return getUniformAttribute(offTier, [this](int pool) { return poolMedium(PoolIndex{pool}); });
    }

    //! Bytes this group would hold at `slots`, each pool rounded to its own allocator's
    //! grain. An absent `medium` counts every pool; a named one counts only its own.
    //!
    //! One slot count for every pool, though pools can report different counts: each
    //! rounds its own allocation up to its own grain, so the excess is under one grain
    //! and rounding returns the allocation whichever count is used.
    [[nodiscard]] size_t roundedBytesWith(SlotCount slots, std::optional<CacheTier> medium) const
    {
        size_t total = 0;
        for (auto const& pool : mPools)
        {
            if (medium.has_value() && pool->medium() != *medium)
            {
                continue;
            }
            total += roundUp(slotCountToSizeT(slots) * pool->slotSize(), pool->sizeGranularity());
        }
        return total;
    }

    //! Bytes this group holds now in pools whose pages are `medium`. Pools of any other
    //! medium contribute nothing.
    //!
    //! Bytes of different media are not comparable -- they are paid for from different
    //! budgets and rounded to different grains -- so every caller measuring them names the
    //! one it means. roundedNumBytes(cacheTier()) is what this group's own tier pays for;
    //! roundedNumBytes(HOST_MEM) on a GPU-tier group is what it holds in host pages besides.
    [[nodiscard]] size_t roundedNumBytes(CacheTier medium) const
    {
        return roundedBytesWith(numSlots(), medium);
    }

protected:
    // Mirrors Python _get_num_slots_from_pools: min(p.num_slots for p in pools).
    [[nodiscard]] SlotCount getNumSlotsFromPools() const noexcept;

    //! Block until every pool backs the range ending just past `slot`.
    void _waitSlotUsable(SlotId slot) const;

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
    //! `sparsity`, when given, marks the pools to allocate from host memory instead of
    //! device memory. They hold a sparse buffer's pages: ordinary hot pages that are read
    //! through the sparse GPU cache rather than directly, so they are device-mapped but do
    //! not consume the tier's device budget.
    //!
    //! The group still reports GPU_MEM. Migration never reads the hot side's tier -- the
    //! codec path keeps only the cold side's -- and host memory is device-accessible, so a
    //! mixed group needs no second tier.
    //!
    //! `hostOptions` backs the sparse pools. One object for the whole level, so every
    //! host-backed pool commits in the same unit.
    //!
    //! `sparseReservation`, when given, caps the host address space each sparse pool
    //! reserves; a pool never reserves less than its initial slots. Unset reserves for the
    //! most slots device memory could buy, which exceeds the address space once a sparse
    //! slot is several hundred times the group's device bytes per slot.
    GpuPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList,
        PooledPhysMemAllocator& physMemAllocator, TypedVec<PoolIndex, bool> sparsity = {},
        HostMemBackingOptions const& hostOptions = {}, std::optional<size_t> sparseReservation = std::nullopt);

    [[nodiscard]] CacheTier cacheTier() const noexcept override
    {
        return CacheTier::GPU_MEM;
    }
};

//! Host pools are committed progressively on a background thread, so a slot may
//! be handed out before its memory is backed. PoolGroupBase::allocate waits for it.
//!
//! The wait is cheap in the steady state: fresh slot ids ascend from zero and
//! recycled slots are reused first, so the Nth fresh allocation waits only for
//! the chunk containing it, and once the fill completes every wait is a single
//! relaxed load.
class HostPoolGroup : public PoolGroupBase
{
public:
    //! groupVmSize is the address space each pool here may grow into.
    //!
    //! Pools are sized by slot-size ratio, so with more than one pool the group
    //! can reserve more than groupVmSize in total. Host groups hold exactly one
    //! pool today, which makes the two equal.
    HostPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, size_t groupVmSize,
        HostMemBackingOptions const& options = {});

    [[nodiscard]] CacheTier cacheTier() const noexcept override
    {
        return CacheTier::HOST_MEM;
    }
};

class DiskPoolGroup : public PoolGroupBase
{
public:
    DiskPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, std::string const& directory);

    [[nodiscard]] CacheTier cacheTier() const noexcept override
    {
        return CacheTier::DISK;
    }
};

//! A second medium's budget over a disjoint subset of each pool group's pools.
//!
//! A slot spans every pool of its group, so one slot count drives the bytes taken in every
//! pool -- but the pools of the GPU tier are of two media, each with its own total and its
//! own allocation grain. This names the second one. `pools[pg]` lists exactly the pools
//! `quota` pays for; the rest are what the primary quota pays for, derived by
//! poolsPaidByQuota(). The two are complements: no pool is named by both, and together
//! they are all of the group's pools.
//!
//! `pools[pg]` may be empty, which is how "this group costs the second medium nothing" is
//! said. Such a group takes no share of `quota` and can never be bound by it.
struct SecondaryPoolBudget
{
    //! Bytes of the second medium, for the whole level, or unset to leave it unbounded.
    //! A budget, not a per-group maximum: how it divides between groups is what the
    //! allocator decides.
    //!
    //! `pools` is a property of the pools rather than of any budget, so a level with a
    //! second medium supplies this struct even when nothing bounds it; only `quota` says
    //! whether anything does.
    std::optional<size_t> quota;
    //! The second medium's allocation grain, which is a property of its allocator rather
    //! than of the level. Host-backed pools round to HostMem::sizeGranularity(), not to the device
    //! granularity, so charging them in device grains would over-count every one of them.
    size_t granularity = 0;
    //! The pools `quota` pays for, per pool group.
    TypedVec<PoolGroupIndex, std::vector<PoolIndex>> pools;
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

    //! Bytes this level holds in pools whose pages are `medium`, each rounded to that
    //! medium's grain. Every group of a level is of the level's tier, so the media
    //! reachable here are this tier's own and, on a device level, the host pages its
    //! host-backed pools hold.
    //!
    //! At cacheTier() this is what the level's quota pays for. Host-backed pools of a
    //! device level fall outside it, which is the point: a rebalance given an inflated
    //! figure would hand the level more slots than the device can hold.
    [[nodiscard]] size_t roundedNumBytes(CacheTier medium) const
    {
        size_t total = 0;
        for (auto const& pg : mPoolGroups)
            total += pg->roundedNumBytes(medium);
        return total;
    }

    //! The medium of this level's pools that are not of its own tier, if any group has
    //! one. At most one across the level: every group here is of the level's tier, and a
    //! group has at most one second medium.
    [[nodiscard]] std::optional<CacheTier> secondaryMedium() const
    {
        for (auto const& pg : mPoolGroups)
        {
            if (auto const medium = pg->secondaryMedium())
            {
                return medium;
            }
        }
        return std::nullopt;
    }

    //! The grain pools of `medium` round to at this level, or 0 when it has none. Groups
    //! holding that medium are held to agree, as the pools within one are.
    //!
    //! Read off a pool, for every medium alike. A host pool commits in the unit its level
    //! was configured with and a device pool in its allocator's, so asking the pool gives
    //! the quantum the quota is expressed in as well as the second medium's grain --
    //! poolSizeGranularity(cacheTier()) is the former.
    [[nodiscard]] size_t poolSizeGranularity(CacheTier medium) const
    {
        auto grainOf = [medium](auto const& pg) { return pg->poolSizeGranularity(medium); };
        auto withMedium = mPoolGroups | std::views::filter([&](auto const& pg) { return grainOf(pg) != 0; });
        if (std::ranges::empty(withMedium))
        {
            return 0;
        }
        return getUniformAttribute(withMedium, grainOf);
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

    //! The pools whose pages are `medium`, as indices into slotSizeLists()[pg].
    //!
    //! For "the pools this level's quota does not pay for", pass secondaryMedium() rather
    //! than a literal: on a host level every pool is HOST_MEM, yet none of them is a
    //! second medium.
    //!
    //! The complement is what the quota pays for, which the allocator derives rather than
    //! taking: that way no pool can be charged twice or missed.
    [[nodiscard]] TypedVec<PoolGroupIndex, std::vector<PoolIndex>> poolsOfMedium(CacheTier medium) const
    {
        TypedVec<PoolGroupIndex, std::vector<PoolIndex>> ret;
        ret.reserve(mPoolGroups.size());
        for (auto const& pg : mPoolGroups)
        {
            std::vector<PoolIndex> matching;
            for (PoolIndex pool{0}; pool < pg->slotSize().size(); ++pool)
            {
                if (pg->poolMedium(pool) == medium)
                {
                    matching.push_back(pool);
                }
            }
            ret.push_back(std::move(matching));
        }
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

    //! Bytes each pool group holds at `slotCounts`, every pool in its own allocator's grain.
    //!
    //! This is the measure the pool ratio speaks in, and the one the allocator divides
    //! against. Measuring a host-backed pool in device grains instead -- they differ by up
    //! to 8192x -- would put the two sides of a rebalance's comparison in different units.
    [[nodiscard]] TypedVec<PoolGroupIndex, size_t> allMediaBytes(
        TypedVec<PoolGroupIndex, SlotCount> const& slotCounts) const
    {
        TypedVec<PoolGroupIndex, size_t> ret(mPoolGroups.size(), 0);
        for (PoolGroupIndex pgIdx{0}; pgIdx < mPoolGroups.size(); ++pgIdx)
        {
            ret[pgIdx] = mPoolGroups[pgIdx]->roundedBytesWith(slotCounts[pgIdx], std::nullopt);
        }
        return ret;
    }

    //! Current ratio list: proportion of bytes per pool group.
    //!
    //! Counts every pool, host-backed ones included -- unlike the quota, which pays for
    //! the dense ones alone. The asymmetry is deliberate: this ratio is projected onto cold
    //! levels, where one pool holds dense and sparse buffers together, and it is what a
    //! rebalance compares its target against. Both sides of that comparison have to measure
    //! the same bytes or it never settles; see "The quota excludes sparse pools; the ratio
    //! counts them" in SPARSE_ATTENTION_DESIGN.md.
    [[nodiscard]] TypedVec<PoolGroupIndex, float> ratioList() const
    {
        TypedVec<PoolGroupIndex, SlotCount> counts(mPoolGroups.size(), 0);
        for (PoolGroupIndex pgIdx{0}; pgIdx < mPoolGroups.size(); ++pgIdx)
        {
            counts[pgIdx] = mPoolGroups[pgIdx]->numSlots();
        }
        auto const bytes = allMediaBytes(counts);
        TypedVec<PoolGroupIndex, float> ret(mPoolGroups.size(), 0.f);
        float total = 0.f;
        for (PoolGroupIndex pgIdx{0}; pgIdx < mPoolGroups.size(); ++pgIdx)
        {
            auto sz = static_cast<float>(bytes[pgIdx]);
            ret[pgIdx] = sz;
            total += sz;
        }
        TLLM_CHECK_DEBUG(total > 0.f);
        for (auto& r : ret)
            r /= total;
        return ret;
    }

    //! Largest quota this level may be resized to, when it has one.
    //!
    //! Unset means unbounded here: GPU and disk levels are limited by the
    //! device and the filesystem rather than by a reservation made up front.
    [[nodiscard]] virtual std::optional<size_t> maxQuota() const noexcept
    {
        return std::nullopt;
    }

    //! Divide `quota` between pool groups by `ratioList`, honouring `minSlots` as a floor,
    //! and `secondBudget` as a second medium's total over a disjoint set of the same pools.
    //!
    //! `slotSizeLists` holds every pool's size; the pools `quota` pays for are those
    //! `secondBudget->pools[pg]` does not name. A slot spans all of them, so one count per
    //! group settles both media at once.
    //!
    //! Both budgets and the floors are kept exactly; the ratio is the only soft bound. The
    //! floors come off the top rather than out of anyone's share, so no division of the rest
    //! can break one, and a group bound by either budget releases what it does not use of
    //! the other. A group named by neither budget for a medium costs that medium nothing.
    //!
    //! The two budgets can only conflict at the floors, and that condition names neither the
    //! ratio nor `quota`. It is therefore a property of the configuration: it is rejected the
    //! first time this runs, and a rebalance -- which changes both of those and none of the
    //! terms in the condition -- cannot begin to fail.
    //!
    //! With `secondBudget` unset, `quota` is the only bound and every pool is charged to it.
    //!
    //! @throws std::runtime_error when the floors alone exceed either budget.
    static TypedVec<PoolGroupIndex, SlotCount> ratioToSlotCountList(size_t quota,
        TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
        TypedVec<PoolGroupIndex, float> const& ratioList, size_t granularity,
        TypedVec<PoolGroupIndex, SlotCount> const& minSlots,
        std::optional<SecondaryPoolBudget> const& secondBudget = std::nullopt);

    //! The pools `quota` pays for: every pool the second budget does not name.
    //!
    //! Derived rather than given, so the two budgets partition the group's pools by
    //! construction -- a pool cannot be charged to both, nor left to neither.
    static TypedVec<PoolGroupIndex, std::vector<PoolIndex>> poolsPaidByQuota(
        TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
        std::optional<SecondaryPoolBudget> const& secondBudget);

    // Distribute grains among the named pools of one pool group.
    // Returns {num_slots, grains_consumed}. `which` must not be empty.
    static std::pair<SlotCount, size_t> grainsToSlots(size_t pgGrains, TypedVec<PoolIndex, size_t> const& slotSizeList,
        std::vector<PoolIndex> const& which, size_t granularity);

    // Minimum grains the named pools of one pool group need for numSlots. Zero for an empty
    // `which`, which is what lets a group with no pool of a medium cost that medium nothing.
    static size_t grainsForSlots(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList,
        std::vector<PoolIndex> const& which, size_t granularity);

protected:
    TypedVec<PoolGroupIndex, std::unique_ptr<PoolGroupBase>> mPoolGroups;
};

class GpuCacheLevelStorage : public CacheLevelStorage
{
public:
    //! Each SlotDesc declares per pool which pools are sparse, and so host-allocated.
    //! `sparseReservation` is passed to every GpuPoolGroup.
    GpuCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
        TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, PooledPhysMemAllocator& physMemAllocator,
        HostMemBackingOptions const& hostOptions = {}, std::optional<size_t> sparseReservation = std::nullopt);

    CacheTier cacheTier() const noexcept override
    {
        return CacheTier::GPU_MEM;
    }

private:
    PooledPhysMemAllocator& mPhysMemAllocator;
};

class HostCacheLevelStorage : public CacheLevelStorage
{
public:
    HostCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
        TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, std::optional<size_t> maxQuota = std::nullopt,
        HostMemBackingOptions const& options = {});

    CacheTier cacheTier() const noexcept override
    {
        return CacheTier::HOST_MEM;
    }

    //! Always set: an unset configured maxQuota resolves to the host memory the
    //! OS reports, and the pools reserve against that resolved value. Rounded
    //! up to poolSizeGranularity(cacheTier()), which is the quantum quotas are expressed
    //! in, so the configured bound is reachable.
    [[nodiscard]] std::optional<size_t> maxQuota() const noexcept override
    {
        return mMaxQuota;
    }

private:
    HostMemBackingOptions mOptions;
    size_t mMaxQuota = 0;
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
// `sparseReservation` applies to a GPU tier only; see GpuPoolGroup.
std::unique_ptr<CacheLevelStorage> createCacheLevelStorage(CacheTierConfig const& tierCfg,
    TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList,
    PooledPhysMemAllocator* gpuPhysMemAllocator = nullptr, HostMemBackingOptions const& options = {},
    std::optional<size_t> sparseReservation = std::nullopt);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
