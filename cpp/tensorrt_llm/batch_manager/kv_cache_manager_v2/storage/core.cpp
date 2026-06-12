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

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unistd.h>
#include <unordered_map>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

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
        TLLM_CHECK_WITH_INFO(slotCountToSizeT(mNumReadyRecycledSlots) == mRecycledSlots.size(),
            "SlotAllocator destroyed with unfinished events — did you call synchronize()?");
        TLLM_CHECK_WITH_INFO(mTargetCapacity == mCapacity && mOverflowSlots.empty(),
            "SlotAllocator destroyed while resize is in progress");
        TLLM_CHECK_WITH_INFO(
            mOccupiedMask.numSetBits() == 0, "SlotAllocator destroyed with occupied slots still in use");
        TLLM_CHECK_WITH_INFO(mRecycledSlots.size() == slotCountToSizeT(mNumActiveSlots),
            "SlotAllocator destroyed with some slots not recycled");
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
    if (slotId >= numSlots() || !mOccupiedMask.get(toSizeT(slotId)))
    {
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
        // Synchronize overflow events (deduplicated — slots often share events).
        {
            std::vector<CachedCudaEvent*> overflowEvents;
            overflowEvents.reserve(mOverflowSlots.size());
            for (auto& s : mOverflowSlots)
                overflowEvents.push_back(&s.readyEvent);
            synchronizeAll(overflowEvents);
        }
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

SlotCount GpuSlotPool::extendByOnePhysMem()
{
    mVirtMem.extend(1);
    return numSlots();
}

Address GpuSlotPool::slotAddress(SlotId slot) const
{
    TLLM_CHECK_DEBUG_WITH_INFO(slot < numSlots(), "GpuSlotPool::slotAddress: slot index out of bounds");
    return MemAddress(mVirtMem.address() + mSlotSize * toSizeT(slot));
}

// ---------------------------------------------------------------------------
// HostSlotPool
// ---------------------------------------------------------------------------

HostSlotPool::HostSlotPool(size_t slotSize, SlotCount numSlots)
    : SlotPoolBase(slotSize)
    , mHostMem(alignedSize(numSlots))
{
}

size_t HostSlotPool::alignedSize(SlotCount numSlots) const noexcept
{
    return roundUp(slotCountToSizeT(numSlots) * mSlotSize, HostMem::kAlignment);
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
    TLLM_CHECK_DEBUG_WITH_INFO(slot < numSlots(), "HostSlotPool::slotAddress: slot index out of bounds");
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
    TLLM_CHECK_DEBUG(mFd != kBadFileDescriptor);
    off_t sz = ::lseek(mFd, 0, SEEK_END);
    return (sz < 0 || mSlotSize == 0) ? 0 : slotCountValueFromSize(static_cast<size_t>(sz) / mSlotSize);
}

void DiskSlotPool::destroy()
{
    if (mFd != kBadFileDescriptor)
    {
        ::close(mFd);
        mFd = kBadFileDescriptor;
    }
}

void DiskSlotPool::resize(SlotCount newNumSlots)
{
    resizeFile(mFd, slotCountToSizeT(newNumSlots) * mSlotSize);
}

Address DiskSlotPool::slotAddress(SlotId slot) const
{
    TLLM_CHECK_DEBUG_WITH_INFO(slot < numSlots(), "DiskSlotPool::slotAddress: slot index out of bounds");
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
    destroy();
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
    return mSlotAllocator.allocate();
}

std::vector<Slot> PoolGroupBase::allocateMultiple(SlotCount numSlots)
{
    return mSlotAllocator.allocateMultiple(numSlots);
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

GpuPoolGroup::GpuPoolGroup(
    SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, PooledPhysMemAllocator& physMemAllocator)
    : PoolGroupBase(numSlots)
{
    size_t physMemSize = physMemAllocator.physMemSize();
    // Query total GPU memory to size virtual address space (mirrors Python GpuPoolGroup).
    size_t totalGpuMem = 0;
    {
        CUdevice dev{};
        cuCtxGetDevice(&dev);
        cuDeviceTotalMem(&totalGpuMem, dev);
    }
    // @TODO: We should replace maxSlotSize with sum. This should also be updated in Python. Will do it later.
    size_t maxSlotSize = *std::max_element(slotSizeList.begin(), slotSizeList.end());
    for (size_t sz : slotSizeList)
    {
        // VA proportional to GPU memory, scaled by slot size ratio (mirrors Python).
        // Compute ratio as double first to avoid size_t overflow.
        double sizeRatio = static_cast<double>(sz) / static_cast<double>(maxSlotSize);
        size_t vmSize = roundDown(static_cast<size_t>(static_cast<double>(totalGpuMem) * sizeRatio), physMemSize);
        vmSize = std::max(vmSize, roundUp(slotCountToSizeT(numSlots) * sz, physMemSize));
        mPools.push_back(std::make_unique<GpuSlotPool>(sz, vmSize, physMemAllocator, numSlots));
    }
}

// ---------------------------------------------------------------------------
// HostPoolGroup
// ---------------------------------------------------------------------------

HostPoolGroup::HostPoolGroup(SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList)
    : PoolGroupBase(numSlots)
{
    for (size_t sz : slotSizeList)
    {
        mPools.push_back(std::make_unique<HostSlotPool>(sz, numSlots));
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

GpuCacheLevelStorage::GpuCacheLevelStorage(
    StorageConfig const& storageCfg, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, size_t physMemSize)
{
    TLLM_CHECK_DEBUG_WITH_INFO(slotCountList.size() == storageCfg.slotDescList.size(),
        "GpuCacheLevelStorage: slotCountList and slotDescList must have the same length");
    mPhysMemAllocator = std::make_unique<PooledPhysMemAllocator>(physMemSize);

    for (PoolGroupIndex pgIdx{0}; pgIdx < storageCfg.slotDescList.size(); ++pgIdx)
    {
        mPoolGroups.push_back(std::make_unique<GpuPoolGroup>(
            slotCountList[pgIdx], storageCfg.slotDescList[pgIdx].slotSizeList(), *mPhysMemAllocator));
    }
}

// ---------------------------------------------------------------------------
// HostCacheLevelStorage
// ---------------------------------------------------------------------------

HostCacheLevelStorage::HostCacheLevelStorage(
    StorageConfig const& storageCfg, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList)
{
    TLLM_CHECK_DEBUG_WITH_INFO(slotCountList.size() == storageCfg.slotDescList.size(),
        "HostCacheLevelStorage: slotCountList and slotDescList must have the same length");
    for (PoolGroupIndex pgIdx{0}; pgIdx < storageCfg.slotDescList.size(); ++pgIdx)
    {
        mPoolGroups.push_back(
            std::make_unique<HostPoolGroup>(slotCountList[pgIdx], storageCfg.slotDescList[pgIdx].slotSizeList()));
    }
}

// ---------------------------------------------------------------------------
// DiskCacheLevelStorage
// ---------------------------------------------------------------------------

DiskCacheLevelStorage::DiskCacheLevelStorage(
    StorageConfig const& storageCfg, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList, std::string directory)
    : mDirectory(std::move(directory))
{
    TLLM_CHECK_DEBUG_WITH_INFO(slotCountList.size() == storageCfg.slotDescList.size(),
        "DiskCacheLevelStorage: slotCountList and slotDescList must have the same length");
    for (PoolGroupIndex pgIdx{0}; pgIdx < storageCfg.slotDescList.size(); ++pgIdx)
    {
        mPoolGroups.push_back(std::make_unique<DiskPoolGroup>(
            slotCountList[pgIdx], storageCfg.slotDescList[pgIdx].slotSizeList(), mDirectory));
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<CacheLevelStorage> createCacheLevelStorage(CacheTierConfig const& tierCfg,
    StorageConfig const& storageCfg, TypedVec<PoolGroupIndex, SlotCount> const& slotCountList)
{
    return std::visit(
        [&](auto const& cfg) -> std::unique_ptr<CacheLevelStorage>
        {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, GpuCacheTierConfig>)
            {
                // Compute phys mem size (granularity) from quota.
                constexpr size_t kPageSize = 2ULL << 20;
                size_t physMemSize = kPageSize
                    << std::min(4, std::max(0, static_cast<int>(std::log2(cfg.quota / (kPageSize * 512)))));
                return std::make_unique<GpuCacheLevelStorage>(storageCfg, slotCountList, physMemSize);
            }
            else if constexpr (std::is_same_v<T, HostCacheTierConfig>)
            {
                return std::make_unique<HostCacheLevelStorage>(storageCfg, slotCountList);
            }
            else
            {
                return std::make_unique<DiskCacheLevelStorage>(storageCfg, slotCountList, cfg.path);
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

std::pair<SlotCount, size_t> CacheLevelStorage::grainsToSlots(
    size_t pgGrains, TypedVec<PoolIndex, size_t> const& slotSizeList, size_t granularity)
{
    TypedVec<PoolIndex, size_t> minPoolGrains(slotSizeList.size());
    for (PoolIndex poolIdx{0}; poolIdx < slotSizeList.size(); ++poolIdx)
    {
        minPoolGrains[poolIdx] = divUp(slotSizeList[poolIdx], granularity);
    }

    size_t minTotal = 0;
    for (auto g : minPoolGrains)
    {
        minTotal += g;
    }
    if (pgGrains < minTotal)
        return {0, 0};

    SlotCount numSlots{std::numeric_limits<SlotCount>::max()};
    size_t remainingPgGrains = pgGrains;

    // Sort pools by slot size ascending.
    std::vector<PoolIndex> poolOrder;
    poolOrder.reserve(slotSizeList.stdSize());
    for (PoolIndex poolIdx{0}; poolIdx < slotSizeList.size(); ++poolIdx)
    {
        poolOrder.push_back(poolIdx);
    }
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

    auto slotsToGrains = [&](SlotCount slots) { return grainsForSlots(slots, slotSizeList, granularity); };
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

size_t CacheLevelStorage::grainsForSlots(
    SlotCount numSlots, TypedVec<PoolIndex, size_t> const& slotSizeList, size_t granularity)
{
    size_t total = 0;
    for (auto s : slotSizeList)
        total += divUp(slotCountToSizeT(numSlots) * s, granularity);
    return total;
}

// ---------------------------------------------------------------------------
// CacheLevelStorage::ratioToSlotCountList (static)
// Mirrors Python CacheLevelStorage.ratio_to_slot_count_list.
// ---------------------------------------------------------------------------

TypedVec<PoolGroupIndex, SlotCount> CacheLevelStorage::ratioToSlotCountList(size_t quota,
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> const& sizeLists,
    TypedVec<PoolGroupIndex, float> const& ratioList, size_t granularity,
    TypedVec<PoolGroupIndex, SlotCount> const& minSlots)
{
    PoolGroupIndex numPg = sizeLists.size();
    TLLM_CHECK_DEBUG(ratioList.size() == numPg);
    TLLM_CHECK_DEBUG_WITH_INFO(std::all_of(ratioList.begin(), ratioList.end(), [](auto x) { return x > 0; }),
        "ratioToSlotCountList: all ratios must be positive");
    TLLM_CHECK_DEBUG(quota % granularity == 0);
    size_t totalGrains = quota / granularity;
    if (TLLM_UNLIKELY(gDebug))
    {
        [[maybe_unused]] size_t minGrains = 0;
        for (auto const& sizes : sizeLists)
            minGrains += toSizeT(sizes.size());
        TLLM_CHECK_WITH_INFO(totalGrains >= minGrains,
            "ratioToSlotCountList: insufficient total grains for at least 1 slot per pool group");
    }

    TypedVec<PoolGroupIndex, SlotCount> slotCntList(numPg, 0);
    size_t remainingGrains = totalGrains;
    std::vector<PoolGroupIndex> activePgs(toSizeT(numPg));
    std::iota(activePgs.begin(), activePgs.end(), PoolGroupIndex{0});

    // Iteratively peel off constrained PGs until all active PGs are
    // unconstrained:
    //   1. Distribute remaining quota among active PGs by ratio.
    //   2. Any PG with slots <= min_slots is constrained — pin it to
    //      min_slots and subtract its grains from the budget.
    //   3. Repeat with the remaining PGs and re-normalized ratios.
    // Each iteration removes at least one PG, so this terminates.
    while (!activePgs.empty())
    {
        // Distribute remainingGrains among active PGs by ratio.
        size_t nActive = activePgs.size();
        std::vector<float> activeRatio(nActive);
        for (size_t i = 0; i < nActive; ++i)
            activeRatio[i] = ratioList[activePgs[i]];

        std::vector<SlotCount> slotsForActive(nActive, 0);
        std::vector<size_t> grainsForActive(nActive, 0);
        size_t budget = remainingGrains;

        // Sort indices by ratio ascending.
        std::vector<size_t> idxLst(nActive);
        std::iota(idxLst.begin(), idxLst.end(), size_t{0});
        std::sort(idxLst.begin(), idxLst.end(), [&](size_t a, size_t b) { return activeRatio[a] < activeRatio[b]; });

        for (size_t i = 0; i < idxLst.size(); ++i)
        {
            size_t idx = idxLst[i];
            double ratioSum = 0.0;
            for (size_t j = i; j < idxLst.size(); ++j)
                ratioSum += static_cast<double>(activeRatio[idxLst[j]]);
            double pct = (ratioSum > 0.0) ? static_cast<double>(activeRatio[idx]) / ratioSum : 1.0;
            auto [slots, used] = grainsToSlots(static_cast<size_t>(std::nearbyint(static_cast<double>(budget) * pct)),
                sizeLists[activePgs[idx]], granularity);
            slotsForActive[idx] = slots;
            grainsForActive[idx] = used;
            TLLM_CHECK_DEBUG(used <= budget);
            budget -= used;
        }

        // Identify constrained PGs (slots <= min_slots).
        std::vector<size_t> constrained;
        std::vector<size_t> unconstrained;
        for (size_t idx = 0; idx < nActive; ++idx)
        {
            PoolGroupIndex pgIdx = activePgs[idx];
            SlotCount const minSlotCount = minSlots[pgIdx];
            if (slotsForActive[idx] <= minSlotCount)
                constrained.push_back(idx);
            else
                unconstrained.push_back(idx);
        }

        if (constrained.empty())
        {
            // All active PGs are unconstrained — accept their allocations.
            for (size_t idx = 0; idx < nActive; ++idx)
                slotCntList[activePgs[idx]] = slotsForActive[idx];
            break;
        }

        // Pin constrained PGs to min_slots and subtract from budget.
        for (size_t idx : constrained)
        {
            PoolGroupIndex pgIdx = activePgs[idx];
            SlotCount const minSlotCount = minSlots[pgIdx];
            size_t minGrains = grainsForSlots(minSlotCount, sizeLists[pgIdx], granularity);
            auto [slots, used] = grainsToSlots(minGrains, sizeLists[pgIdx], granularity);
            slotCntList[pgIdx] = slots;
            TLLM_CHECK_DEBUG(used <= remainingGrains);
            remainingGrains -= used;
        }

        if (unconstrained.empty())
        {
            // All PGs are constrained — nothing left to redistribute.
            break;
        }

        if (remainingGrains == 0)
            throw std::runtime_error("Insufficient quota to satisfy min_slots constraints");

        // Continue with unconstrained PGs only.
        std::vector<PoolGroupIndex> newActivePgs;
        newActivePgs.reserve(unconstrained.size());
        for (size_t idx : unconstrained)
            newActivePgs.push_back(activePgs[idx]);
        activePgs = std::move(newActivePgs);
    }

    // _g2s may under-count slots due to imperfect grain distribution
    // across pools. Try bumping each PG's slot count while it still fits
    // within the same grain budget.
    for (PoolGroupIndex pgIdx{0}; pgIdx < sizeLists.size(); ++pgIdx)
    {
        size_t grainsNow = grainsForSlots(slotCntList[pgIdx], sizeLists[pgIdx], granularity);
        while (grainsForSlots(slotCntList[pgIdx] + 1, sizeLists[pgIdx], granularity) <= grainsNow)
            slotCntList[pgIdx] += 1;
    }

    return slotCntList;
}

// Instance convenience wrapper.
TypedVec<PoolGroupIndex, SlotCount> CacheLevelStorage::computeSlotCountList(
    TypedVec<PoolGroupIndex, float> const& ratioList, TypedVec<PoolGroupIndex, SlotCount> const& minSlots,
    std::optional<size_t> quota) const
{
    size_t q = quota.value_or(totalQuota());
    TLLM_CHECK_DEBUG(ratioList.size() == mPoolGroups.size());
    return ratioToSlotCountList(q, slotSizeLists(), ratioList, poolSizeGranularity(), minSlots);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
