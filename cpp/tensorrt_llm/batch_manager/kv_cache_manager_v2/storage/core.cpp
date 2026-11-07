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

#include "kv_cache_manager_v2/storage/core.h"

#include "kv_cache_manager_v2/utils/funcGuard.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unistd.h>
#include <unordered_map>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

size_t tierAllocationGranularity(CacheTier tier, size_t quota)
{
    switch (tier)
    {
    case CacheTier::GPU_MEM:
    {
        // Device memory is mapped and unmapped in this unit and never released any other
        // way, so a large tier wants a large one: the mappings, not the bytes, are what a
        // bigger quota would otherwise multiply.
        constexpr size_t kPageSize = 2ULL << 20;
        size_t const ratio = quota / (kPageSize * 512);
        int const exponent = ratio == 0 ? 0 : std::min(4, static_cast<int>(std::log2(ratio)));
        return kPageSize << exponent;
    }
    // Host memory has no such constraint: madvise and mprotect take any page-aligned range,
    // so the unit is a fill chunk rather than a quantum and one size serves every tier. The
    // backing raises it to a page should a platform ever ask for more.
    case CacheTier::HOST_MEM: return hostMemSizeGranularity();
    case CacheTier::DISK: return kDiskPoolSizeGranularity;
    default: throw std::invalid_argument("Invalid cache tier");
    }
}

// ---------------------------------------------------------------------------
// SlotAllocator
// ---------------------------------------------------------------------------

SlotAllocator::SlotAllocator(SlotCount capacity)
    : mCapacity(capacity)
    , mTargetCapacity(capacity)
    , mNumActiveSlots(0)
    , mNumReadyRecycledSlots(0)
    , mOccupiedMask(slotCountToSizeT(capacity))
{
}

SlotAllocator::~SlotAllocator()
{
    // Mirrors Python SlotAllocator.__del__ (assert_critical checks).
    if (TLLM_UNLIKELY(gDebug))
    {
        KVCM2_CHECK_FATAL_WITH_INFO(slotCountToSizeT(mNumReadyRecycledSlots) == mRecycledSlots.size(),
            "SlotAllocator destroyed with unfinished events — did you call synchronize()?");
        KVCM2_CHECK_FATAL_WITH_INFO(mTargetCapacity == mCapacity && mOverflowSlots.empty(),
            "SlotAllocator destroyed while resize is in progress");
        KVCM2_CHECK_FATAL_WITH_INFO(
            mOccupiedMask.numSetBits() == 0, "SlotAllocator destroyed with occupied slots still in use");
        KVCM2_CHECK_FATAL_WITH_INFO(mRecycledSlots.size() == slotCountToSizeT(mNumActiveSlots),
            "SlotAllocator destroyed with some slots not recycled");
    }

    // Recycled and overflow slots still carry their ids; the allocator owns them, so clear them
    // here rather than letting ~Slot report each one. A slot that escaped the allocator is not in
    // the recycle lists and is still reported.
    for (auto& slot : mRecycledSlots)
    {
        slot.resetSlot();
    }
    for (auto& slot : mOverflowSlots)
    {
        slot.resetSlot();
    }
}

SlotCount SlotAllocator::numFreeSlots() const noexcept
{
    SlotCount const inactiveSlots = mTargetCapacity > mNumActiveSlots ? mTargetCapacity - mNumActiveSlots : 0;
    return slotCountValueFromSize(mRecycledSlots.size()) + inactiveSlots;
}

SlotCount SlotAllocator::numOccupiedSlots() const noexcept
{
    return slotCountValueFromSize(mOccupiedMask.numSetBits());
}

Slot SlotAllocator::allocate()
{
    if (numFreeSlots() == 0)
    {
        throw OutOfPagesError("SlotAllocator: no free slots");
    }
    scrubEvents();

    Slot slot;
    if (mNumReadyRecycledSlots > 0)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(!mRecycledSlots.empty(), "ready recycled slots > 0 but deque is empty");
        slot = std::move(mRecycledSlots.front());
        mRecycledSlots.pop_front();
        TLLM_CHECK_DEBUG_WITH_INFO(slot.hasValidSlot(), "ready recycled slot has no valid id");
        --mNumReadyRecycledSlots;
        TLLM_CHECK_DEBUG_WITH_INFO(slot.readyEvent.isClosed(), "ready recycled slot has non-null event");
    }
    else if (mNumActiveSlots < std::min(mCapacity, mTargetCapacity))
    {
        slot.setSlotId(SlotId{mNumActiveSlots++});
    }
    else
    {
        slot = std::move(mRecycledSlots.front());
        mRecycledSlots.pop_front();
        TLLM_CHECK_DEBUG_WITH_INFO(slot.hasValidSlot(), "non-ready recycled slot has no valid id");
    }
    mOccupiedMask.set(toSizeT(slot.slotId()));
    return slot;
}

std::vector<Slot> SlotAllocator::allocateMultiple(SlotCount numSlots)
{
    if (numSlots < 0)
    {
        throw LogicError("SlotAllocator::allocateMultiple: slot count must be non-negative");
    }
    if (numFreeSlots() < numSlots)
    {
        throw OutOfPagesError("SlotAllocator: not enough free slots");
    }
    std::vector<Slot> result;
    result.reserve(slotCountToSizeT(numSlots));
    for (SlotCount slotCount{0}; slotCount < numSlots; ++slotCount)
    {
        result.push_back(allocate());
    }
    return result;
}

void SlotAllocator::release(Slot slot)
{
    if (!slot.hasValidSlot())
    {
        throw LogicError("SlotAllocator::release: slot has no valid id");
    }
    SlotId const slotId = slot.slotId();
    if (!isValidSlotId(slotId) || !mOccupiedMask.get(toSizeT(slotId)))
    {
        // The id is not one this allocator has outstanding, so drop it rather than let the
        // rejected slot look like an unreleased one.
        slot.resetSlot();
        throw LogicError("SlotAllocator::release: slot is not occupied");
    }
    mOccupiedMask.clear(toSizeT(slotId));
    if (slotId < mTargetCapacity)
    {
        mRecycledSlots.push_back(std::move(slot));
    }
    else
    {
        mOverflowSlots.push_back(std::move(slot));
    }
    scrubEvents();
    TLLM_CHECK_DEBUG(check());
}

void SlotAllocator::expand(SlotCount newNumSlots)
{
    TLLM_CHECK_DEBUG(check());
    TLLM_CHECK_DEBUG(mTargetCapacity == mCapacity);
    TLLM_CHECK_DEBUG(newNumSlots > mCapacity);
    mOccupiedMask.resize(slotCountToSizeT(newNumSlots));
    mCapacity = newNumSlots;
    mTargetCapacity = newNumSlots;
    TLLM_CHECK_DEBUG(check());
}

void SlotAllocator::prepareForShrink(SlotCount newNumSlots)
{
    TLLM_CHECK_DEBUG(check());
    TLLM_CHECK_DEBUG(mTargetCapacity == mCapacity);
    TLLM_CHECK_DEBUG(newNumSlots < mCapacity);
    std::deque<Slot> newRecycled;
    SlotCount newNumReady = 0;
    SlotCount const oldNumReady = mNumReadyRecycledSlots;
    SlotCount idx = 0;
    for (auto& s : mRecycledSlots)
    {
        if (s.slotId() < newNumSlots)
        {
            if (idx < oldNumReady)
                ++newNumReady;
            newRecycled.push_back(std::move(s));
        }
        else
        {
            mOverflowSlots.push_back(std::move(s));
        }
        ++idx;
    }
    mRecycledSlots = std::move(newRecycled);
    mNumReadyRecycledSlots = newNumReady;
    mTargetCapacity = newNumSlots;
    TLLM_CHECK_DEBUG(check());
}

bool SlotAllocator::finishShrink()
{
    TLLM_CHECK_DEBUG(check());
    // Overflow-range IDs that were ever issued are exactly
    // max(0, _num_active_slots - _target_capacity); the underused case
    // (_num_active_slots <= _target_capacity) collapses to zero.
    SlotCount const expectedOverflow = std::max(SlotCount{0}, mNumActiveSlots - mTargetCapacity);
    if (shrinkInProgress() && slotCountValueFromSize(mOverflowSlots.size()) == expectedOverflow)
    {
        // Validate uniqueness of slot IDs in overflow (debug only).
        if (TLLM_UNLIKELY(gDebug))
        {
            std::set<SlotId> ids;
            for (auto const& s : mOverflowSlots)
            {
                TLLM_CHECK(s.hasValidSlot());
                ids.insert(s.slotId());
            }
            TLLM_CHECK_WITH_INFO(ids.size() == mOverflowSlots.size(), "Duplicate slot IDs in overflow slots");
        }
        // Synchronize overflow events. Slots often share an event; synchronize() closes it, so
        // later slots holding the same event return immediately.
        for (auto& s : mOverflowSlots)
            s.readyEvent.synchronize();
        for (auto& s : mOverflowSlots)
            s.resetSlot();
        mOverflowSlots.clear();
        mCapacity = mTargetCapacity;
        mNumActiveSlots = std::min(mNumActiveSlots, mCapacity);
        scrubEvents();
        TLLM_CHECK_DEBUG(check());
        return true;
    }
    throw std::runtime_error("SlotAllocator::finishShrink: cannot finish shrink yet");
}

std::vector<SlotId> SlotAllocator::getSlotsBlockingShrink() const
{
    std::vector<SlotId> result;
    for (SlotCount id = mTargetCapacity; id < mCapacity; ++id)
    {
        if (mOccupiedMask.get(slotCountToSizeT(id)))
            result.push_back(SlotId{id});
    }
    return result;
}

void SlotAllocator::synchronize()
{
    while (slotCountToSizeT(mNumReadyRecycledSlots) != mRecycledSlots.size())
    {
        scrubEvents();
    }
}

bool SlotAllocator::check() const noexcept
{
    // Mirrors Python SlotAllocator._check().
    if (mNumActiveSlots > mCapacity)
        return false;
    if (mTargetCapacity > mCapacity)
        return false;
    if (!shrinkInProgress() && !mOverflowSlots.empty())
        return false;
    for (auto const& slot : mOverflowSlots)
    {
        if (!slot.hasValidSlot())
            return false;
        SlotCount id = slot.slotId().value();
        if (id < mTargetCapacity || id >= mCapacity)
            return false;
    }
    SlotCount const accountedSlots = slotCountValueFromSize(mRecycledSlots.size())
        + slotCountValueFromSize(mOverflowSlots.size()) + numOccupiedSlots();
    if (accountedSlots != mNumActiveSlots)
        return false;
    return true;
}

void SlotAllocator::scrubEvents()
{
    for (size_t i = slotCountToSizeT(mNumReadyRecycledSlots); i < mRecycledSlots.size(); ++i)
    {
        if (mRecycledSlots[i].queryReady())
        {
            ++mNumReadyRecycledSlots;
        }
        else
        {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// GpuSlotPool
// ---------------------------------------------------------------------------

GpuSlotPool::GpuSlotPool(size_t slotSize, size_t vmSize, PooledPhysMemAllocator& physMemAllocator, SlotCount numSlots)
    : SlotPoolBase(slotSize)
    , mVirtMem(vmSize, physMemAllocator)
{
    TLLM_CHECK_DEBUG_WITH_INFO(
        vmSize % physMemAllocator.physMemSize() == 0, "vm_size must be aligned to phys_mem_size");
    resize(numSlots);
}

size_t GpuSlotPool::computeNumPhysMem(size_t slotSize, SlotCount numSlots, size_t physMemSize) noexcept
{
    return divUp(slotCountToSizeT(numSlots) * slotSize, physMemSize);
}

SlotCount GpuSlotPool::computeNumSlots(size_t slotSize, size_t numPhysMem, size_t physMemSize) noexcept
{
    return slotCountValueFromSize(numPhysMem * physMemSize / slotSize);
}

SlotCount GpuSlotPool::numSlots() const noexcept
{
    return computeNumSlots(mSlotSize, mVirtMem.numPhysMem(), mVirtMem.physMemSize());
}

void GpuSlotPool::destroy()
{
    mVirtMem.destroy();
}

void GpuSlotPool::resize(SlotCount newNumSlots)
{
    size_t physSize = mVirtMem.physMemSize();
    size_t newPhysMem = computeNumPhysMem(mSlotSize, newNumSlots, physSize);
    mVirtMem.realloc(physSize * newPhysMem);
}

Address GpuSlotPool::slotAddress(SlotId slot) const
{
    TLLM_CHECK_DEBUG_WITH_INFO(isValidSlotId(slot), "GpuSlotPool::slotAddress: slot index out of bounds");
    return MemAddress(mVirtMem.address() + mSlotSize * toSizeT(slot));
}

// ---------------------------------------------------------------------------
// HostSlotPool
// ---------------------------------------------------------------------------

HostSlotPool::HostSlotPool(size_t slotSize, SlotCount numSlots, size_t vmSize, HostMemBackingOptions const& options)
    : SlotPoolBase(slotSize)
    , mCommitUnit(hostMemSizeGranularity())
    // registerUnit = slotSize: every copy here is a whole slot and the driver rejects
    // one that crosses a registration, so the backing keeps its registration bounds on
    // slot multiples -- which leaves the commit unit free of the slot size.
    //
    // waitForFill = false: the owning pool group blocks per allocation on only the range
    // it hands out, so serving can start before the whole tier is committed.
    , mHostMem(std::max(vmSize, alignedSize(numSlots)), alignedSize(numSlots),
          HostMemBackingOptions{.registerUnit = slotSize, .allowRemoteNumaFallback = options.allowRemoteNumaFallback},
          /*waitForFill=*/false)
{
}

void HostSlotPool::waitSlotUsable(SlotId slot) const
{
    mHostMem.waitUsable(mSlotSize * (toSizeT(slot) + 1));
}

size_t HostSlotPool::alignedSize(SlotCount numSlots) const noexcept
{
    // Rounded to the unit memory is allocated and released in, which is what
    // every size handed to HostMem must be a multiple of. Deliberately not the
    // fill chunk: the last fill step is simply short.
    return roundUp(slotCountToSizeT(numSlots) * mSlotSize, mCommitUnit);
}

SlotCount HostSlotPool::numSlots() const noexcept
{
    return slotCountValueFromSize(mHostMem.size() / mSlotSize);
}

void HostSlotPool::destroy()
{
    mHostMem.destroy();
}

void HostSlotPool::resize(SlotCount newNumSlots)
{
    mHostMem.resize(alignedSize(newNumSlots));
}

Address HostSlotPool::slotAddress(SlotId slot) const
{
    TLLM_CHECK_DEBUG_WITH_INFO(isValidSlotId(slot), "HostSlotPool::slotAddress: slot index out of bounds");
    return MemAddress(mHostMem.address() + mSlotSize * toSizeT(slot));
}

// ---------------------------------------------------------------------------
// DiskSlotPool
// ---------------------------------------------------------------------------

DiskSlotPool::DiskSlotPool(std::string const& directory, size_t slotSize, SlotCount numSlots)
    : SlotPoolBase(slotSize)
{
    // Try O_TMPFILE first, fall back to mkstemp.
    mFd = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_EXCL, 0664);
    if (mFd < 0)
    {
        if (errno == EOPNOTSUPP)
        {
            char tmpl[PATH_MAX];
            snprintf(tmpl, sizeof(tmpl), "%s/kvXXXXXX", directory.c_str());
            mFd = ::mkstemp(tmpl);
            if (mFd < 0)
            {
                throw DiskOOMError("DiskSlotPool: mkstemp failed: " + std::string(std::strerror(errno)));
            }
            ::unlink(tmpl);
        }
        else
        {
            throw DiskOOMError("DiskSlotPool: open O_TMPFILE failed: " + std::string(std::strerror(errno)));
        }
    }
    resize(numSlots);
}

DiskSlotPool::~DiskSlotPool()
{
    destroy();
}

SlotCount DiskSlotPool::numSlots() const noexcept
{
    return mNumSlots;
}

void DiskSlotPool::destroy()
{
    if (mFd != kBadFileDescriptor)
    {
        ::close(mFd);
        mFd = kBadFileDescriptor;
    }
    mNumSlots = SlotCount{0};
}

void DiskSlotPool::resize(SlotCount newNumSlots)
{
    // Recorded only once the file really changed size: resizeFile throws without resizing.
    resizeFile(mFd, slotCountToSizeT(newNumSlots) * mSlotSize);
    mNumSlots = newNumSlots;
}

Address DiskSlotPool::slotAddress(SlotId slot) const
{
    TLLM_CHECK_WITH_INFO(isValidSlotId(slot), "DiskSlotPool::slotAddress: slot index out of bounds");
    size_t const byteOffset = toSizeT(slot) * mSlotSize;
    TLLM_CHECK_DEBUG_WITH_INFO(byteOffset <= static_cast<size_t>(std::numeric_limits<ssize_t>::max()),
        "DiskSlotPool::slotAddress: byte offset out of range");
    return DiskAddress{mFd, static_cast<ssize_t>(byteOffset)};
}

// ---------------------------------------------------------------------------
// PoolGroupBase
// ---------------------------------------------------------------------------

PoolGroupBase::PoolGroupBase(SlotCount numSlots)
    : mSlotAllocator(numSlots)
{
}

SlotCount PoolGroupBase::getNumSlotsFromPools() const noexcept
{
    if (mPools.empty())
        return 0;
    SlotCount minSlots = mPools.front()->numSlots();
    for (PoolIndex poolIdx{1}; poolIdx < mPools.size(); ++poolIdx)
    {
        minSlots = std::min(minSlots, mPools[poolIdx]->numSlots());
    }
    return minSlots;
}

PoolGroupBase::~PoolGroupBase()
{
    KVCM2_POISON_ON_EXCEPT([this]() { destroy(); });
}

SlotCount PoolGroupBase::numSlots() const noexcept
{
    SlotCount n = mSlotAllocator.numSlots();
    if (TLLM_UNLIKELY(gDebug))
    {
        // Mirrors Python PoolGroupBase.num_slots: assert num_slots <= self._get_num_slots_from_pools()
        [[maybe_unused]] SlotCount poolSlots = getNumSlotsFromPools();
        TLLM_CHECK_WITH_INFO(n <= poolSlots, "SlotAllocator capacity exceeds pool capacity");
    }
    return n;
}

Slot PoolGroupBase::allocate()
{
    Slot slot = mSlotAllocator.allocate();
    // A failed fill surfaces out of _waitSlotUsable, after the slot is already taken, so it
    // is returned to the allocator before the error propagates.
    auto rollback = FuncGuard([&]() { release(std::move(slot)); });
    _waitSlotUsable(slot.slotId());
    rollback.cancel();
    return slot;
}

void PoolGroupBase::_waitSlotUsable(SlotId slot) const
{
    for (auto const& pool : mPools)
    {
        pool->waitSlotUsable(slot);
    }
}

std::vector<Slot> PoolGroupBase::allocateMultiple(SlotCount numSlots)
{
    if (numSlots == SlotCount{0})
    {
        return {};
    }
    std::vector<Slot> slots = mSlotAllocator.allocateMultiple(numSlots);
    auto rollback
        = FuncGuard([&]() { std::ranges::for_each(slots, [this](Slot& slot) { release(std::move(slot)); }); });
    // Slot ids are not returned in order, so the highest one bounds them all.
    _waitSlotUsable(std::ranges::max(slots | std::views::transform(&Slot::slotId)));
    rollback.cancel();
    return slots;
}

void PoolGroupBase::release(Slot slot)
{
    mSlotAllocator.release(std::move(slot));
}

void PoolGroupBase::destroy()
{
    if (mDestroyed)
        return;
    if (mSlotAllocator.numSlots() != 0)
    {
        mSlotAllocator.synchronize();
        mSlotAllocator.prepareForShrink(0);
        mSlotAllocator.finishShrink();
    }
    for (auto& p : mPools)
        p->destroy();
    mDestroyed = true;
}

void PoolGroupBase::resizePools(std::optional<SlotCount> newNumSlots)
{
    SlotCount n = newNumSlots.value_or(mSlotAllocator.numSlots());
    for (auto& p : mPools)
        p->resize(n);
    // Mirrors Python PoolGroupBase.resize_pools: assert NDEBUG or self._check(True)
    // After resize, allocator capacity must not exceed pool capacity (allow mismatch).
    TLLM_CHECK_DEBUG_WITH_INFO(mSlotAllocator.numSlots() <= getNumSlotsFromPools(),
        "After resizePools: allocator capacity exceeds pool capacity");
}

TypedVec<PoolIndex, Address> PoolGroupBase::slotAddress(SlotId slotId) const
{
    TypedVec<PoolIndex, Address> addrs;
    addrs.reserve(mPools.size());
    for (auto const& p : mPools)
        addrs.push_back(p->slotAddress(slotId));
    return addrs;
}

TypedVec<PoolIndex, size_t> PoolGroupBase::slotSize() const
{
    TypedVec<PoolIndex, size_t> sizes;
    sizes.reserve(mPools.size());
    for (auto const& p : mPools)
        sizes.push_back(p->slotSize());
    return sizes;
}

// ---------------------------------------------------------------------------
// GpuPoolGroup
// ---------------------------------------------------------------------------

GpuPoolGroup::GpuPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList,
    PooledPhysMemAllocator& physMemAllocator, TypedVec<PoolIndex, bool> sparsity,
    HostMemBackingOptions const& hostOptions, std::optional<size_t> sparseReservation)
    : PoolGroupBase(numSlots)
{
    if (sparsity.empty())
    {
        sparsity = TypedVec<PoolIndex, bool>(slotSizeList.size(), false);
    }
    TLLM_CHECK_WITH_INFO(
        sparsity.size() == slotSizeList.size(), "GpuPoolGroup: one sparsity flag per pool is required");
    size_t physMemSize = physMemAllocator.physMemSize();
    // Query total GPU memory to size virtual address space (mirrors Python GpuPoolGroup).
    size_t totalGpuMem = 0;
    {
        CUdevice dev{};
        cuCheck(cuCtxGetDevice(&dev));
        cuCheck(cuDeviceTotalMem(&totalGpuMem, dev));
    }
    TLLM_CHECK_WITH_INFO(!slotSizeList.empty(), "GpuPoolGroup: slotSizeList must not be empty");
    // Every pool reserves address space for the most slots this group could ever hold: the
    // count all of device memory would buy. A slot is one row spanning every pool, and only
    // the device-backed pools draw on that memory, so they alone set the count. No quota can
    // exceed device memory, so this reservation is a bound the allocator cannot reach.
    size_t deviceSlotBytes = 0;
    for (PoolIndex p{0}; p < slotSizeList.size(); ++p)
    {
        if (!sparsity[p])
        {
            deviceSlotBytes += slotSizeList.at(p);
        }
    }
    TLLM_CHECK_WITH_INFO(deviceSlotBytes > 0,
        "GpuPoolGroup: a wholly sparse group has no device bytes per slot to size its reservation from");
    size_t const maxSlots = totalGpuMem / deviceSlotBytes;
    for (PoolIndex p{0}; p < slotSizeList.size(); ++p)
    {
        size_t const sz = slotSizeList.at(p);
        size_t const grain = sparsity[p] ? hostMemSizeGranularity() : physMemSize;
        size_t reserved = maxSlots * sz;
        if (sparsity[p] && sparseReservation.has_value())
        {
            reserved = std::min(reserved, *sparseReservation);
        }
        size_t const vmSize = std::max(roundUp(reserved, grain), roundUp(slotCountToSizeT(numSlots) * sz, grain));
        if (sparsity[p])
        {
            // A sparse pool's pages: host memory, device-mapped, so the sparse cache fills
            // from them without a host round trip. The reservation is host address space,
            // not device, but it is as hard a bound: HostMem keeps a fixed base and rejects
            // a resize past it. maxSlots alone would keep it unreachable, but scaled by a
            // sparse slot it can exceed the address space, so `sparseReservation` caps it
            // and growth past the cap is refused.
            mPools.push_back(std::make_unique<HostSlotPool>(sz, numSlots, vmSize, hostOptions));
            continue;
        }
        mPools.push_back(std::make_unique<GpuSlotPool>(sz, vmSize, physMemAllocator, numSlots));
    }
}

// ---------------------------------------------------------------------------
// HostPoolGroup
// ---------------------------------------------------------------------------

HostPoolGroup::HostPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, size_t groupVmSize,
    HostMemBackingOptions const& options)
    : PoolGroupBase(numSlots)
{
    TLLM_CHECK_WITH_INFO(!slotSizeList.empty(), "HostPoolGroup: slotSizeList must not be empty");
    // Each pool reserves address space proportional to its slot size, so that
    // the pools can grow together up to the bound. Mirrors GpuPoolGroup, which
    // divides the device's memory the same way.
    size_t const maxSlotSize = *std::max_element(slotSizeList.begin(), slotSizeList.end());
    for (size_t sz : slotSizeList)
    {
        double const sizeRatio = static_cast<double>(sz) / static_cast<double>(maxSlotSize);
        // Rounded here because the share is scaled by a ratio and so lands anywhere, while
        // HostMem takes only whole granules. Up rather than down, and the shares may then
        // sum past groupVmSize: this is address space, which costs nothing until committed.
        auto const poolVmSize
            = roundUp(static_cast<size_t>(static_cast<double>(groupVmSize) * sizeRatio), hostMemSizeGranularity());
        mPools.push_back(std::make_unique<HostSlotPool>(sz, numSlots, poolVmSize, options));
    }
}

// ---------------------------------------------------------------------------
// DiskPoolGroup
// ---------------------------------------------------------------------------

DiskPoolGroup::DiskPoolGroup(
    SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, std::string const& directory)
    : PoolGroupBase(numSlots)
{
    for (size_t sz : slotSizeList)
    {
        mPools.push_back(std::make_unique<DiskSlotPool>(directory, sz, numSlots));
    }
}

// ---------------------------------------------------------------------------
// CacheLevelStorage
// ---------------------------------------------------------------------------

std::vector<Slot> CacheLevelStorage::allocateMultiple(PoolGroupIndex pgIdx, SlotCount numSlots)
{
    return mPoolGroups.at(pgIdx)->allocateMultiple(numSlots);
}

void CacheLevelStorage::release(PoolGroupIndex pgIdx, Slot slot)
{
    mPoolGroups.at(pgIdx)->release(std::move(slot));
}

SlotCount CacheLevelStorage::numFreeSlots(PoolGroupIndex pgIdx) const
{
    return mPoolGroups.at(pgIdx)->numFreeSlots();
}

TypedVec<PoolIndex, Address> CacheLevelStorage::slotAddress(PoolGroupIndex pgIdx, SlotId slotId) const
{
    return mPoolGroups.at(pgIdx)->slotAddress(slotId);
}

// ---------------------------------------------------------------------------
// GpuCacheLevelStorage
// ---------------------------------------------------------------------------

GpuCacheLevelStorage::GpuCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, PooledPhysMemAllocator& physMemAllocator,
    HostMemBackingOptions const& hostOptions, std::optional<size_t> sparseReservation)
    : mPhysMemAllocator(physMemAllocator)
{
    TLLM_CHECK_WITH_INFO(slotCountList.size() == slotDescList.size(),
        "GpuCacheLevelStorage: slotCountList and slotDescList must have the same length");
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotDescList.size(); ++pgIdx)
    {
        auto const& slotDesc = slotDescList[pgIdx];
        auto const sizes = slotDesc.slotSizeList();
        TypedVec<PoolIndex, bool> sparsity;
        sparsity.reserve(sizes.size());
        for (PoolIndex pool{0}; pool < sizes.size(); ++pool)
        {
            sparsity.push_back(slotDesc.isSparse(pool));
        }
        mPoolGroups.push_back(std::make_unique<GpuPoolGroup>(
            slotCountList[pgIdx], sizes, mPhysMemAllocator, std::move(sparsity), hostOptions, sparseReservation));
    }
}

// ---------------------------------------------------------------------------
// HostCacheLevelStorage
// ---------------------------------------------------------------------------

HostCacheLevelStorage::HostCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, std::optional<size_t> maxQuota,
    HostMemBackingOptions const& options)
    : mOptions(options)
    // Resolved once here so the bound the pools reserve against is the same one
    // a later resize is checked against. Quotas are quantized to the commit
    // unit before they are compared, so the bound is quantized the same way: a
    // bound that sat between two multiples would reject the very quota the
    // pools had already reserved for.
    , mMaxQuota(roundUp(maxQuota.value_or(hostTotalMemory()), hostMemSizeGranularity()))
{
    TLLM_CHECK_WITH_INFO(slotCountList.size() == slotDescList.size(),
        "HostCacheLevelStorage: slotCountList and slotDescList must have the same length");
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotDescList.size(); ++pgIdx)
    {
        mPoolGroups.push_back(std::make_unique<HostPoolGroup>(
            slotCountList[pgIdx], slotDescList[pgIdx].slotSizeList(), mMaxQuota, mOptions));
    }
}

// ---------------------------------------------------------------------------
// DiskCacheLevelStorage
// ---------------------------------------------------------------------------

DiskCacheLevelStorage::DiskCacheLevelStorage(TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList,
    TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, std::string directory)
    : mDirectory(std::move(directory))
{
    TLLM_CHECK_WITH_INFO(slotCountList.size() == slotDescList.size(),
        "DiskCacheLevelStorage: slotCountList and slotDescList must have the same length");
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotDescList.size(); ++pgIdx)
    {
        mPoolGroups.push_back(
            std::make_unique<DiskPoolGroup>(slotCountList[pgIdx], slotDescList[pgIdx].slotSizeList(), mDirectory));
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<CacheLevelStorage> createCacheLevelStorage(CacheTierConfig const& tierCfg,
    TypedVec<PoolGroupIndex, SlotDesc> const& slotDescList, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList,
    PooledPhysMemAllocator* gpuPhysMemAllocator, HostMemBackingOptions const& options,
    std::optional<size_t> sparseReservation)
{
    TLLM_CHECK((cacheTierOf(tierCfg) == CacheTier::GPU_MEM) == (gpuPhysMemAllocator != nullptr));
    return std::visit(
        [&](auto const& cfg) -> std::unique_ptr<CacheLevelStorage>
        {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, GpuCacheTierConfig>)
            {
                return std::make_unique<GpuCacheLevelStorage>(
                    slotDescList, slotCountList, *gpuPhysMemAllocator, options, sparseReservation);
            }
            else if constexpr (std::is_same_v<T, HostCacheTierConfig>)
            {
                return std::make_unique<HostCacheLevelStorage>(slotDescList, slotCountList, cfg.maxQuota, options);
            }
            else
            {
                return std::make_unique<DiskCacheLevelStorage>(slotDescList, slotCountList, cfg.path);
            }
        },
        tierCfg);
}

// ---------------------------------------------------------------------------
// CacheLevelStorage helper: grainsToSlots
// Distribute grains among pools within a pool group.
// Returns {num_slots, grains_consumed}.
// Mirrors Python CacheLevelStorage._grains_to_slots.
// ---------------------------------------------------------------------------

std::pair<SlotCount, size_t> CacheLevelStorage::grainsToSlots(size_t pgGrains,
    TypedVec<PoolIndex, size_t> const& slotSizeList, std::vector<PoolIndex> const& which, size_t granularity)
{
    TLLM_CHECK_WITH_INFO(granularity > 0, "Cache storage granularity must be positive");
    // With nothing to divide the grains between there is no slot count to find, and the
    // search below would run past SlotCount's range looking for one.
    TLLM_CHECK_WITH_INFO(!which.empty(), "grainsToSlots needs at least one pool to size against");
    // Indexed by real pool index; entries outside `which` stay zero and are never read.
    TypedVec<PoolIndex, size_t> minPoolGrains(slotSizeList.size(), 0);
    size_t minTotal = 0;
    for (PoolIndex poolIdx : which)
    {
        TLLM_CHECK_WITH_INFO(slotSizeList[poolIdx] > 0, "Cache slot size must be positive");
        minPoolGrains[poolIdx] = divUp(slotSizeList[poolIdx], granularity);
        minTotal += minPoolGrains[poolIdx];
    }
    if (pgGrains < minTotal)
        return {0, 0};

    SlotCount numSlots{std::numeric_limits<SlotCount>::max()};
    size_t remainingPgGrains = pgGrains;

    // Sort the named pools by slot size ascending. Ties may fall either way: every term
    // below is a function of the sizes alone.
    std::vector<PoolIndex> poolOrder = which;
    std::sort(poolOrder.begin(), poolOrder.end(),
        [&](PoolIndex a, PoolIndex b) { return slotSizeList[a] < slotSizeList[b]; });

    for (size_t j = 0; j < poolOrder.size(); ++j)
    {
        PoolIndex const poolIdx = poolOrder[j];
        size_t slotSz = slotSizeList[poolIdx];
        size_t poolSzSum = 0;
        for (size_t k = j; k < poolOrder.size(); ++k)
            poolSzSum += slotSizeList[poolOrder[k]];
        double poolFrac = (poolSzSum > 0) ? static_cast<double>(slotSz) / static_cast<double>(poolSzSum) : 1.0;
        size_t roundedGrains = static_cast<size_t>(std::nearbyint(static_cast<double>(remainingPgGrains) * poolFrac));
        size_t poolGrains = std::max(minPoolGrains[poolIdx], roundedGrains);
        SlotCount const poolSlots = slotCountValueFromSize(poolGrains * granularity / slotSz);
        numSlots = std::min(numSlots, poolSlots);
        TLLM_CHECK_DEBUG(poolGrains <= remainingPgGrains);
        remainingPgGrains -= poolGrains;
    }
    TLLM_CHECK_DEBUG(remainingPgGrains == 0);
    TLLM_CHECK_DEBUG(numSlots > 0);

    auto slotsToGrains = [&](SlotCount slots) { return grainsForSlots(slots, slotSizeList, which, granularity); };
    SlotCount lo = numSlots;
    SlotCount step = 1;
    SlotCount hi = lo + step;
    while (slotsToGrains(hi) <= pgGrains)
    {
        lo = hi;
        step *= 2;
        hi = lo + step;
    }
    while (lo + 1 < hi)
    {
        SlotCount const mid = lo + ((hi - lo) / 2);
        if (slotsToGrains(mid) <= pgGrains)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    size_t const used = slotsToGrains(lo);
    TLLM_CHECK_DEBUG(used <= pgGrains);
    TLLM_CHECK_DEBUG(slotsToGrains(lo + 1) > pgGrains);
    return {lo, used};
}

// ---------------------------------------------------------------------------
// CacheLevelStorage helper: grainsForSlots
// Compute minimum grains needed for numSlots in a pool group.
// Mirrors Python CacheLevelStorage._grains_for_slots.
// ---------------------------------------------------------------------------

size_t CacheLevelStorage::grainsForSlots(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList,
    std::vector<PoolIndex> const& which, size_t granularity)
{
    size_t total = 0;
    for (PoolIndex poolIdx : which)
        total += divUp(slotCountToSizeT(numSlots) * slotSizeList[poolIdx], granularity);
    return total;
}

// ---------------------------------------------------------------------------
// CacheLevelStorage::ratioToSlotCountList (static)
// Mirrors Python CacheLevelStorage.ratio_to_slot_count_list.
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, std::vector<PoolIndex>> CacheLevelStorage::poolsPaidByQuota(
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    std::optional<SecondaryPoolBudget> const& secondBudget)
{
    TypedVec<PoolGroupIndex, std::vector<PoolIndex>> ret;
    ret.reserve(slotSizeLists.size());
    for (PoolGroupIndex pgIdx{0}; pgIdx < slotSizeLists.size(); ++pgIdx)
    {
        auto const numPools = slotSizeLists[pgIdx].size();
        TypedVec<PoolIndex, bool> taken(numPools, false);
        if (secondBudget.has_value())
        {
            for (PoolIndex poolIdx : secondBudget->pools[pgIdx])
            {
                TLLM_CHECK_WITH_INFO(poolIdx < numPools,
                    "A second budget names pool %d of group %d, which has only %d pools", poolIdx.value(),
                    pgIdx.value(), numPools.value());
                // Naming a pool twice would charge it twice and leave the quota paying for
                // one fewer than it thinks.
                TLLM_CHECK_WITH_INFO(
                    !taken[poolIdx], "A second budget names pool %d of group %d twice", poolIdx.value(), pgIdx.value());
                taken[poolIdx] = true;
            }
        }
        std::vector<PoolIndex> paid;
        paid.reserve(toSizeT(numPools));
        for (PoolIndex poolIdx{0}; poolIdx < numPools; ++poolIdx)
        {
            if (!taken[poolIdx])
            {
                paid.push_back(poolIdx);
            }
        }
        ret.push_back(std::move(paid));
    }
    return ret;
}

namespace
{

//! Bytes one pool group holds at `slots`, each pool rounded to its own grain. This is the
//! quantity the pool ratio speaks in: what the group actually occupies, device and host
//! together.
size_t allBytesForSlots(
    TypedVec<PoolIndex, size_t> const& sizes, TypedVec<PoolIndex, size_t> const& grains, SlotCount slots)
{
    auto const n = slotCountToSizeT(slots);
    size_t total = 0;
    for (PoolIndex poolIdx{0}; poolIdx < sizes.size(); ++poolIdx)
    {
        total += roundUp(n * sizes[poolIdx], grains[poolIdx]);
    }
    return total;
}

//! Steps a group may take to correct itself before only a budget may carry it further.
constexpr int kSelfCorrectionLimit = 8;

//! Two levels within this of each other are the same level. Paired with the float32 ratio:
//! the band sits four orders of magnitude above double's own noise, so no comparison below
//! needs exact arithmetic.
constexpr double kRatioEps = 1.0 / 16777216.0;

//! The largest rung at or below `budget` bytes, which is free-run closed by construction.
//!
//! Bracketed, not galloped: `A(n) >= n * S` with `S` the summed slot size, so no count above
//! `budget / S` can fit and bisection has an upper end without searching for one.
SlotCount rungAtOrBelow(
    TypedVec<PoolIndex, size_t> const& sizes, TypedVec<PoolIndex, size_t> const& grains, size_t budget)
{
    size_t sumSizes = 0;
    for (PoolIndex poolIdx{0}; poolIdx < sizes.size(); ++poolIdx)
    {
        sumSizes += sizes[poolIdx];
    }
    TLLM_CHECK_WITH_INFO(sumSizes > 0, "A pool group must have a positive slot size");
    SlotCount lo{0};
    auto hi = slotCountValueFromSize(budget / sumSizes + 1);
    if (allBytesForSlots(sizes, grains, hi) <= budget)
    {
        return hi;
    }
    while (lo + 1 < hi)
    {
        SlotCount const mid = lo + ((hi - lo) / 2);
        if (allBytesForSlots(sizes, grains, mid) <= budget)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

//! The most slots the grains already bought for `slots` will hold, in every pool at once.
//!
//! The counterpart of PoolGroupBase::getNumSlotsFromPools, and what makes a count
//! self-consistent: growing to it costs nothing in any medium, so it never disturbs a
//! budget.
//!
//! Every pool is in view, including ones the second budget pays for. While that budget is
//! still paying for a pool its grains are as real a cost as the quota's, so a run that held
//! one of the two fixed while the other moved would not be free.
SlotCount freeRun(TypedVec<PoolIndex, size_t> const& sizes, TypedVec<PoolIndex, size_t> const& grains, SlotCount slots)
{
    auto const n = slotCountToSizeT(slots);
    SlotCount best{std::numeric_limits<SlotCount>::max()};
    for (PoolIndex poolIdx{0}; poolIdx < sizes.size(); ++poolIdx)
    {
        auto const g = grains[poolIdx];
        auto const sz = sizes[poolIdx];
        best = std::min(best, slotCountValueFromSize(divUp(n * sz, g) * g / sz));
    }
    return best;
}

//! The most slots the grains the quota has already been charged for will hold.
//!
//! The same run over the pools the quota pays for. Once the second budget's level has
//! apportioned all of it, what is left over buys its pools at no cost anyone is accounting
//! for, so those pools no longer belong in a free run: leaving them in pins the count to
//! the bottom of the range over which the quota's own cost is flat, at a position fixed by
//! their width -- the one thing the quota does not measure -- while every slot up to the
//! top of that range is already bought and sits unused.
SlotCount freeRunOnQuota(
    TypedVec<PoolIndex, size_t> const& sizes, std::vector<PoolIndex> const& paid, size_t granularity, SlotCount slots)
{
    auto const n = slotCountToSizeT(slots);
    SlotCount best{std::numeric_limits<SlotCount>::max()};
    for (PoolIndex poolIdx : paid)
    {
        auto const sz = sizes[poolIdx];
        best = std::min(best, slotCountValueFromSize(divUp(n * sz, granularity) * granularity / sz));
    }
    return best;
}

} // namespace

TypedVec<PoolGroupIndex, SlotCount> CacheLevelStorage::ratioToSlotCountList(size_t quota,
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& slotSizeLists,
    TypedVec<PoolGroupIndex, float> const& ratioList, size_t granularity,
    TypedVec<PoolGroupIndex, SlotCount> const& minSlots, std::optional<SecondaryPoolBudget> const& secondBudget)
{
    PoolGroupIndex const numPg = slotSizeLists.size();
    TLLM_CHECK_DEBUG(ratioList.size() == numPg);
    TLLM_CHECK_WITH_INFO(granularity > 0, "Cache storage granularity must be positive");

    bool const hasSecond = secondBudget.has_value();
    bool const bounded = hasSecond && secondBudget->quota.has_value();
    size_t const grain2 = hasSecond ? secondBudget->granularity : granularity;
    TLLM_CHECK_WITH_INFO(!hasSecond || grain2 > 0, "A second budget needs a positive granularity");

    auto const pools = poolsPaidByQuota(slotSizeLists, secondBudget);
    static std::vector<PoolIndex> const kNoPools;
    auto secondPools = [&](PoolGroupIndex pgIdx) -> std::vector<PoolIndex> const&
    { return hasSecond ? secondBudget->pools[pgIdx] : kNoPools; };

    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        TLLM_CHECK_WITH_INFO(ratioList[pgIdx] > 0.f, "ratioToSlotCountList: all ratios must be positive");
        TLLM_CHECK_WITH_INFO(!pools[pgIdx].empty(),
            "ratioToSlotCountList: pool group %d has no pool the primary quota pays for", pgIdx.value());
    }

    // One grain per pool, settled once: a pool's grain follows its medium, so the budget a
    // pool belongs to decides only which total it lands in, never how it rounds.
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> grains;
    grains.reserve(numPg);
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        TypedVec<PoolIndex, size_t> pgGrains(slotSizeLists[pgIdx].size(), granularity);
        for (PoolIndex poolIdx : secondPools(pgIdx))
        {
            pgGrains[poolIdx] = grain2;
        }
        grains.push_back(std::move(pgGrains));
    }

    auto quotaGrains = [&](PoolGroupIndex pgIdx, SlotCount slots)
    { return grainsForSlots(slots, slotSizeLists[pgIdx], pools[pgIdx], granularity); };
    auto secondGrains = [&](PoolGroupIndex pgIdx, SlotCount slots)
    { return grainsForSlots(slots, slotSizeLists[pgIdx], secondPools(pgIdx), grain2); };
    auto allBytes = [&](PoolGroupIndex pgIdx, SlotCount slots)
    { return allBytesForSlots(slotSizeLists[pgIdx], grains[pgIdx], slots); };

    size_t const budgetGrains = quota / granularity;
    size_t const budget2Grains = bounded ? *secondBudget->quota / grain2 : 0;

    // The floors are not negotiable, so they come off the top before anything is shared.
    // Snapping each to its free run costs no grain in any medium, and starting there is what
    // makes "never decrease" enough to keep them.
    TypedVec<PoolGroupIndex, SlotCount> slotCntList(numPg, 0);
    size_t floorQuota = 0;
    size_t floorSecond = 0;
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        slotCntList[pgIdx] = freeRun(slotSizeLists[pgIdx], grains[pgIdx], std::max(minSlots[pgIdx], SlotCount{0}));
        floorQuota += quotaGrains(pgIdx, slotCntList[pgIdx]);
        floorSecond += secondGrains(pgIdx, slotCntList[pgIdx]);
    }
    if (bounded)
    {
        TLLM_CHECK_WITH_INFO(floorSecond <= budget2Grains,
            "Second memory budget of %zu bytes cannot hold the %zu the batches in `constraints` need. Raise the "
            "budget or lower the constraints.",
            *secondBudget->quota, floorSecond * grain2);
    }
    TLLM_CHECK_WITH_INFO(floorQuota <= budgetGrains, "Insufficient quota to satisfy min_slots constraints");

    size_t remainingQuota = budgetGrains - floorQuota;
    size_t remainingSecond = bounded ? budget2Grains - floorSecond : 0;

    // The floors are where every rule below starts, and none of them may fall back to it.
    TypedVec<PoolGroupIndex, SlotCount> const floorSlots = slotCntList;

    // Normalised once, order-independently: a float sum that depended on the order of the
    // pool groups would carry that order into every comparison below, and the tie band
    // magnifies a last-ulp difference into a different answer.
    TypedVec<PoolGroupIndex, double> ratio(numPg, 0.0);
    {
        std::vector<double> values;
        values.reserve(toSizeT(numPg));
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            values.push_back(static_cast<double>(ratioList[pgIdx]));
        }
        std::sort(values.begin(), values.end());
        long double total = 0.0L;
        for (double v : values)
        {
            total += v;
        }
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            ratio[pgIdx] = static_cast<double>(static_cast<long double>(ratioList[pgIdx]) / total);
        }
    }

    // A level is a byte count: the all-bytes total a group of ratio 1 would be entitled to.
    // Because a group's occupied bytes and its two grain costs are the same quantity in
    // different units, this one scalar drives both budgets.
    auto nearestRung = [&](PoolGroupIndex pgIdx, size_t level)
    {
        double const target = static_cast<double>(level) * ratio[pgIdx];
        SlotCount const nLo = rungAtOrBelow(slotSizeLists[pgIdx], grains[pgIdx], static_cast<size_t>(target));
        if (nLo < floorSlots[pgIdx])
        {
            return floorSlots[pgIdx];
        }
        SlotCount const nHi = freeRun(slotSizeLists[pgIdx], grains[pgIdx], nLo + 1);
        // Of the two rungs bracketing the target take the nearer; an exact midpoint stays low.
        return 2.0 * target > static_cast<double>(allBytes(pgIdx, nLo) + allBytes(pgIdx, nHi)) ? nHi : nLo;
    };

    // Largest level whose spend fits, and whether the search ran out of headroom without
    // the spend ever exceeding the residue -- which is how "this budget cannot bind" is
    // reported. `spend` must be non-decreasing, which `nearestRung` is.
    auto highestLevelFitting = [](auto&& spend, size_t residue)
    {
        if (spend(size_t{0}) > residue)
        {
            return std::pair<size_t, bool>{0, false};
        }
        size_t lo = 0;
        size_t hi = 1;
        bool exceeded = false;
        for (int i = 0; i < 64; ++i)
        {
            if (spend(hi) > residue)
            {
                exceeded = true;
                break;
            }
            lo = hi;
            if (hi > std::numeric_limits<size_t>::max() / 2)
            {
                break;
            }
            hi *= 2;
        }
        if (!exceeded)
        {
            return std::pair<size_t, bool>{hi, true};
        }
        while (lo + 1 < hi)
        {
            size_t const mid = lo + ((hi - lo) / 2);
            if (spend(mid) <= residue)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        return std::pair<size_t, bool>{lo, false};
    };

    // Among the levels paying for the same amount, the lowest. They differ only for a group
    // whose cost in that budget is flat over a range of counts -- a small dense pool beside
    // a large host-backed one -- and the extra level buys such a group nothing the budget
    // pays for, while pushing it off its target ratio.
    auto leastLevelReaching = [](auto&& spend, size_t target, size_t upper)
    {
        size_t lo = 0;
        size_t hi = upper;
        while (lo < hi)
        {
            size_t const mid = lo + ((hi - lo) / 2);
            if (spend(mid) >= target)
            {
                hi = mid;
            }
            else
            {
                lo = mid + 1;
            }
        }
        return lo;
    };

    TypedVec<PoolGroupIndex, bool> onSecond(numPg, false);
    bool anyOnSecond = false;
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        onSecond[pgIdx] = !secondPools(pgIdx).empty();
        anyOnSecond = anyOnSecond || onSecond[pgIdx];
    }

    // One level for the whole host-backed class: equal levels are what an equal ratio means,
    // so holding the class at one level is what keeps the ratio within it.
    size_t level2 = 0;
    bool hasLevel2 = false;
    bool saturated2 = true;
    if (bounded && anyOnSecond)
    {
        auto spendSecond = [&](size_t level)
        {
            size_t total = 0;
            for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
            {
                if (onSecond[pgIdx])
                {
                    total += secondGrains(pgIdx, nearestRung(pgIdx, level)) - secondGrains(pgIdx, floorSlots[pgIdx]);
                }
            }
            return total;
        };
        auto const found = highestLevelFitting(spendSecond, remainingSecond);
        level2 = found.first;
        saturated2 = found.second;
        hasLevel2 = true;
    }

    auto capped = [&](PoolGroupIndex pgIdx, size_t level)
    {
        if (hasLevel2 && onSecond[pgIdx] && level > level2)
        {
            level = level2;
        }
        return nearestRung(pgIdx, level);
    };

    auto spendPrimary = [&](size_t level)
    {
        size_t total = 0;
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            total += quotaGrains(pgIdx, capped(pgIdx, level)) - quotaGrains(pgIdx, floorSlots[pgIdx]);
        }
        return total;
    };

    auto const foundPrimary = highestLevelFitting(spendPrimary, remainingQuota);
    size_t level1 = foundPrimary.first;
    if (!foundPrimary.second)
    {
        level1 = leastLevelReaching(spendPrimary, spendPrimary(level1), level1);
    }

    // The secondary binds only where the primary would have carried the class higher.
    bool const secondaryBinds = hasLevel2 && !saturated2 && level1 > level2;

    auto levelFor
        = [&](PoolGroupIndex pgIdx) { return hasLevel2 && onSecond[pgIdx] && level1 > level2 ? level2 : level1; };

    // The rung at or below the level, not the nearer one: which groups round up is a
    // comparison between groups, and a per-group rounding settles it by whichever crossing
    // the ratio's last bits happen to put first. The promotions below compare the crossings
    // against each other instead, and so can see that two are tied.
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        double const target = static_cast<double>(levelFor(pgIdx)) * ratio[pgIdx];
        SlotCount want = rungAtOrBelow(slotSizeLists[pgIdx], grains[pgIdx], static_cast<size_t>(target));
        want = std::max(want, floorSlots[pgIdx]);
        remainingQuota -= quotaGrains(pgIdx, want) - quotaGrains(pgIdx, slotCntList[pgIdx]);
        if (bounded)
        {
            remainingSecond -= secondGrains(pgIdx, want) - secondGrains(pgIdx, slotCntList[pgIdx]);
        }
        slotCntList[pgIdx] = want;
    }

    // Finish the grains the quota has already been charged for -- see freeRunOnQuota. This
    // runs once the promotions have stopped, because leftover is what the two levels did
    // not apportion, and that is not known until they have stopped spending.
    std::vector<PoolGroupIndex> classA;
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        if (onSecond[pgIdx])
        {
            classA.push_back(pgIdx);
        }
    }

    // Summed order-independently, like the ratio normaliser above: a float sum that
    // depended on the order of the pool groups would carry that order into the level below.
    double classShare = 0.0;
    {
        std::vector<double> values;
        values.reserve(classA.size());
        for (PoolGroupIndex pgIdx : classA)
        {
            values.push_back(ratio[pgIdx]);
        }
        std::sort(values.begin(), values.end());
        long double total = 0.0L;
        for (double v : values)
        {
            total += v;
        }
        classShare = static_cast<double>(total);
    }

    TypedVec<PoolGroupIndex, SlotCount> tops(numPg, 0);
    auto closePaidGrains = [&]()
    {
        bool wanted = false;
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            tops[pgIdx] = freeRunOnQuota(slotSizeLists[pgIdx], pools[pgIdx], granularity, slotCntList[pgIdx]);
            // A group the second budget names no pool of is already at its top: its free run
            // is over the quota's pools alone.
            TLLM_CHECK_DEBUG(onSecond[pgIdx] || tops[pgIdx] == slotCntList[pgIdx]);
            wanted = wanted || tops[pgIdx] > slotCntList[pgIdx];
        }
        if (!wanted)
        {
            return false;
        }
        size_t need = 0;
        for (PoolGroupIndex pgIdx : classA)
        {
            need += secondGrains(pgIdx, tops[pgIdx]) - secondGrains(pgIdx, slotCntList[pgIdx]);
        }
        if (!bounded || need <= remainingSecond)
        {
            if (bounded)
            {
                remainingSecond -= need;
            }
            for (PoolGroupIndex pgIdx : classA)
            {
                slotCntList[pgIdx] = tops[pgIdx];
            }
            return true;
        }

        // More closures are wanted than the leftover can buy. One level for the whole
        // host-backed class, Webster rounded: the rule level2 divided the budget by,
        // applied to what it did not divide, so the ratio inside that class is what decides
        // where the leftover goes. A level is a function of the groups' own data, so which
        // of them it reaches cannot depend on the order they were passed in.
        auto closureTarget = [&](PoolGroupIndex pgIdx, size_t level)
        {
            double const share = static_cast<double>(level) * ratio[pgIdx] / classShare;
            SlotCount const nLo = rungAtOrBelow(slotSizeLists[pgIdx], grains[pgIdx], static_cast<size_t>(share));
            if (nLo >= tops[pgIdx])
            {
                return tops[pgIdx];
            }
            SlotCount const nHi = freeRun(slotSizeLists[pgIdx], grains[pgIdx], nLo + 1);
            SlotCount const want
                = 2.0 * share > static_cast<double>(allBytes(pgIdx, nLo) + allBytes(pgIdx, nHi)) ? nHi : nLo;
            return std::min(tops[pgIdx], std::max(slotCntList[pgIdx], want));
        };
        auto closureSpend = [&](size_t level)
        {
            size_t total = 0;
            for (PoolGroupIndex pgIdx : classA)
            {
                total += secondGrains(pgIdx, closureTarget(pgIdx, level)) - secondGrains(pgIdx, slotCntList[pgIdx]);
            }
            return total;
        };
        size_t const level3 = highestLevelFitting(closureSpend, remainingSecond).first;
        bool moved = false;
        for (PoolGroupIndex pgIdx : classA)
        {
            SlotCount const want = closureTarget(pgIdx, level3);
            if (want > slotCntList[pgIdx])
            {
                remainingSecond -= secondGrains(pgIdx, want) - secondGrains(pgIdx, slotCntList[pgIdx]);
                slotCntList[pgIdx] = want;
                moved = true;
            }
        }
        return moved;
    };

    // Each group's next step becomes the nearer of its two rungs at its own crossing level.
    // Taking the steps in ascending crossing order is rounding to the nearest rung done
    // across the groups rather than one at a time; carrying on past the level is what spends
    // the fraction of a step the level search had to leave behind.
    TypedVec<PoolGroupIndex, int> corrections(numPg, 0);

    //! Everything that distinguishes one pool group from another, in a fixed order.
    //!
    //! Two groups agreeing on all of it are interchangeable, so the index that settles them
    //! last picks between relabelings of one answer rather than between two answers.
    //! Anything that distinguishes two groups and is missing from here is a way for the
    //! answer to depend on the order they were passed in.
    //!
    //! What a slot costs each budget leads: of two groups equally entitled to a step, the
    //! one whose slots are cheaper leaves more of the residue for whoever comes next, which
    //! is the tie-break's first rule carried from one step to the whole ladder. The slot
    //! sizes cannot say this on their own -- two groups can hold the same sizes and differ
    //! in which of them the second budget pays for, and so in what a slot costs the quota.
    struct Identity
    {
        size_t quotaRate{0};
        size_t secondRate{0};
        std::vector<size_t> sizes;
        std::vector<size_t> secondPools;
        SlotCount floor{0};

        bool operator<(Identity const& other) const
        {
            return std::tie(quotaRate, secondRate, sizes, secondPools, floor)
                < std::tie(other.quotaRate, other.secondRate, other.sizes, other.secondPools, other.floor);
        }
    };

    TypedVec<PoolGroupIndex, Identity> identities;
    identities.reserve(numPg);
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        Identity ident;
        for (PoolIndex poolIdx : pools[pgIdx])
        {
            ident.quotaRate += slotSizeLists[pgIdx][poolIdx];
        }
        for (PoolIndex poolIdx : secondPools(pgIdx))
        {
            ident.secondRate += slotSizeLists[pgIdx][poolIdx];
            ident.secondPools.push_back(toSizeT(poolIdx));
        }
        std::sort(ident.secondPools.begin(), ident.secondPools.end());
        ident.sizes.reserve(toSizeT(slotSizeLists[pgIdx].size()));
        for (PoolIndex poolIdx{0}; poolIdx < slotSizeLists[pgIdx].size(); ++poolIdx)
        {
            ident.sizes.push_back(slotSizeLists[pgIdx][poolIdx]);
        }
        ident.floor = minSlots[pgIdx];
        identities.push_back(std::move(ident));
    }

    //! One step a group could take, and everything needed to choose between them.
    struct Candidate
    {
        double crossing;
        PoolGroupIndex pgIdx;
        SlotCount next;
        bool correcting;
        size_t d1;
        size_t d2;
    };

    // A cheaper step leaves more residue for whoever follows; at an equal level the larger
    // ratio is the more entitled; the group's identity is the last canonical discriminator
    // before the index.
    auto cheaperStep = [&](Candidate const& lhs, Candidate const& rhs)
    {
        if (lhs.d1 != rhs.d1)
        {
            return lhs.d1 < rhs.d1;
        }
        if (lhs.d2 != rhs.d2)
        {
            return lhs.d2 < rhs.d2;
        }
        if (ratio[lhs.pgIdx] != ratio[rhs.pgIdx])
        {
            return ratio[lhs.pgIdx] > ratio[rhs.pgIdx];
        }
        if (identities[lhs.pgIdx] < identities[rhs.pgIdx])
        {
            return true;
        }
        if (identities[rhs.pgIdx] < identities[lhs.pgIdx])
        {
            return false;
        }
        return toSizeT(lhs.pgIdx) < toSizeT(rhs.pgIdx);
    };

    std::vector<Candidate> cands;
    cands.reserve(toSizeT(numPg));
    while (true)
    {
        size_t placed = 0;
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            placed += allBytes(pgIdx, slotCntList[pgIdx]);
        }

        cands.clear();
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            SlotCount const next = freeRun(slotSizeLists[pgIdx], grains[pgIdx], slotCntList[pgIdx] + 1);
            size_t const d1 = quotaGrains(pgIdx, next) - quotaGrains(pgIdx, slotCntList[pgIdx]);
            size_t const d2 = secondGrains(pgIdx, next) - secondGrains(pgIdx, slotCntList[pgIdx]);
            if (d1 > remainingQuota || (bounded && d2 > remainingSecond))
            {
                continue;
            }
            double const crossing = static_cast<double>(allBytes(pgIdx, slotCntList[pgIdx]) + allBytes(pgIdx, next))
                / (2.0 * ratio[pgIdx]);

            // The level at which the rest of the allocation is exactly this group's share.
            // Solving B = r * (B + rest) gives rest / (1 - r), a total that excludes the
            // group's own bytes and so does not chase itself upward. A ratio at or above one
            // leaves no rest to be consistent with; zero sends every step to the budget
            // clause, which is the right rule for a lone group and is what stops the loop.
            double const rest = static_cast<double>(placed - allBytes(pgIdx, slotCntList[pgIdx]));
            double cap = ratio[pgIdx] < 1.0 ? rest / (1.0 - ratio[pgIdx]) : 0.0;
            if (secondaryBinds && onSecond[pgIdx])
            {
                cap = std::min(cap, static_cast<double>(level2));
            }

            bool correcting = true;
            if (corrections[pgIdx] >= kSelfCorrectionLimit || crossing * (1.0 + kRatioEps) >= cap)
            {
                // Past its own share, so only the budget that sets this group's level may
                // carry it further, and only out of that budget's own residue: a class-A
                // group may not creep past its class level on device grains alone. A step
                // neither bounded budget pays for moves the group off target for nothing.
                if (secondaryBinds && onSecond[pgIdx])
                {
                    if (d2 < 1)
                    {
                        continue;
                    }
                }
                else if (d1 < 1)
                {
                    continue;
                }
                correcting = false;
            }

            cands.push_back(Candidate{crossing, pgIdx, next, correcting, d1, d2});
        }
        if (cands.empty())
        {
            // The two levels have stopped spending, so whatever the second budget still
            // holds is leftover. Closing a group out of it can make that group's next step
            // cost a quota grain, so the promotions and the closure alternate; two closures
            // never run back to back, since the second has nothing left to reach.
            if (closePaidGrains())
            {
                continue;
            }
            break;
        }

        // Everything within the ratio's own precision of the lowest crossing is tied, and
        // the tie is settled by the groups' data, so no answer depends on where a group
        // happens to sit in the list.
        double lowest = cands.front().crossing;
        for (auto const& cand : cands)
        {
            lowest = std::min(lowest, cand.crossing);
        }
        Candidate const* chosen = nullptr;
        for (auto const& cand : cands)
        {
            if (cand.crossing > lowest * (1.0 + kRatioEps))
            {
                continue;
            }
            if (chosen == nullptr || cheaperStep(cand, *chosen))
            {
                chosen = &cand;
            }
        }

        if (chosen->correcting)
        {
            ++corrections[chosen->pgIdx];
        }
        remainingQuota
            -= quotaGrains(chosen->pgIdx, chosen->next) - quotaGrains(chosen->pgIdx, slotCntList[chosen->pgIdx]);
        if (bounded)
        {
            remainingSecond
                -= secondGrains(chosen->pgIdx, chosen->next) - secondGrains(chosen->pgIdx, slotCntList[chosen->pgIdx]);
        }
        slotCntList[chosen->pgIdx] = chosen->next;
    }

    return slotCntList;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
