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

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <fcntl.h>
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

SlotAllocator::SlotAllocator(int capacity)
    : mCapacity(capacity)
    , mTargetCapacity(capacity)
    , mNumActiveSlots(0)
    , mNumReadyRecycledSlots(0)
    , mOccupiedMask(capacity)
{
}

int SlotAllocator::numFreeSlots() const noexcept
{
    return static_cast<int>(mRecycledSlots.size()) + std::max(mTargetCapacity - mNumActiveSlots, 0);
}

int SlotAllocator::numOccupiedSlots() const noexcept
{
    return mOccupiedMask.numSetBits();
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
        slot = std::move(mRecycledSlots.front());
        mRecycledSlots.pop_front();
        --mNumReadyRecycledSlots;
    }
    else if (mNumActiveSlots < std::min(mCapacity, mTargetCapacity))
    {
        slot.setSlotId(mNumActiveSlots++);
    }
    else
    {
        slot = std::move(mRecycledSlots.front());
        mRecycledSlots.pop_front();
    }
    mOccupiedMask.set(slot.slotId());
    return slot;
}

std::vector<Slot> SlotAllocator::allocateMultiple(int numSlots)
{
    if (numFreeSlots() < numSlots)
    {
        throw OutOfPagesError("SlotAllocator: not enough free slots");
    }
    std::vector<Slot> result;
    result.reserve(static_cast<size_t>(numSlots));
    for (int i = 0; i < numSlots; ++i)
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
    if (slot.slotId() >= mCapacity || !mOccupiedMask.get(slot.slotId()))
    {
        throw LogicError("SlotAllocator::release: slot is not occupied");
    }
    mOccupiedMask.clear(slot.slotId());
    if (slot.slotId() < mTargetCapacity)
    {
        mRecycledSlots.push_back(std::move(slot));
    }
    else
    {
        mOverflowSlots.push_back(std::move(slot));
    }
    scrubEvents();
}

void SlotAllocator::expand(int newNumSlots)
{
    assert(mTargetCapacity == mCapacity);
    assert(newNumSlots > mCapacity);
    mOccupiedMask.resize(newNumSlots);
    mCapacity = newNumSlots;
    mTargetCapacity = newNumSlots;
}

void SlotAllocator::prepareForShrink(int newNumSlots)
{
    assert(mTargetCapacity == mCapacity);
    assert(newNumSlots < mCapacity);
    std::deque<Slot> newRecycled;
    std::vector<Slot> newRecycledList;
    int newNumReady = 0;
    int oldNumReady = mNumReadyRecycledSlots;
    int idx = 0;
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
}

bool SlotAllocator::finishShrink()
{
    if (shrinkInProgress() && mTargetCapacity + static_cast<int>(mOverflowSlots.size()) == mNumActiveSlots)
    {
        // Mirrors Python assertions: validate overflow slots before shrinking.
        assert(static_cast<int>(mOverflowSlots.size()) == mNumActiveSlots - mTargetCapacity
            && "Overflow slot count mismatch");
        // Validate uniqueness of slot IDs in overflow (debug only).
        if (!gNdebug)
        {
            std::set<SlotId> ids;
            for (auto const& s : mOverflowSlots)
            {
                assert(s.hasValidSlot());
                ids.insert(s.slotId());
            }
            assert(static_cast<int>(ids.size()) == static_cast<int>(mOverflowSlots.size())
                && "Duplicate slot IDs in overflow slots");
        }
        // Synchronize all overflow events.
        for (auto& s : mOverflowSlots)
        {
            s.readyEvent.synchronize();
            s.resetSlot();
        }
        mOverflowSlots.clear();
        mCapacity = mTargetCapacity;
        mNumActiveSlots = std::min(mNumActiveSlots, mCapacity);
        scrubEvents();
        return true;
    }
    throw std::runtime_error("SlotAllocator::finishShrink: cannot finish shrink yet");
}

std::vector<SlotId> SlotAllocator::getSlotsBlockingShrink() const
{
    std::vector<SlotId> result;
    for (int id = mTargetCapacity; id < mCapacity; ++id)
    {
        if (mOccupiedMask.get(id))
            result.push_back(id);
    }
    return result;
}

void SlotAllocator::synchronize()
{
    while (mNumReadyRecycledSlots != static_cast<int>(mRecycledSlots.size()))
    {
        scrubEvents();
    }
}

void SlotAllocator::scrubEvents()
{
    int num = mNumReadyRecycledSlots;
    for (int i = num; i < static_cast<int>(mRecycledSlots.size()); ++i)
    {
        if (mRecycledSlots[static_cast<size_t>(i)].queryReady())
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

GpuSlotPool::GpuSlotPool(size_t slotSize, size_t vmSize, PooledPhysMemAllocator& physMemAllocator, int numSlots)
    : SlotPoolBase(slotSize)
    , mVirtMem(vmSize, physMemAllocator)
{
    resize(numSlots);
}

int GpuSlotPool::computeNumPhysMem(size_t slotSize, int numSlots, size_t physMemSize) noexcept
{
    return static_cast<int>(divUp(static_cast<size_t>(numSlots) * slotSize, physMemSize));
}

int GpuSlotPool::computeNumSlots(size_t slotSize, int numPhysMem, size_t physMemSize) noexcept
{
    return static_cast<int>(static_cast<size_t>(numPhysMem) * physMemSize / slotSize);
}

int GpuSlotPool::numSlots() const noexcept
{
    return computeNumSlots(mSlotSize, mVirtMem.numPhysMem(), mVirtMem.physMemSize());
}

void GpuSlotPool::destroy()
{
    mVirtMem.destroy();
}

void GpuSlotPool::resize(int newNumSlots)
{
    size_t physSize = mVirtMem.physMemSize();
    int newPhysMem = computeNumPhysMem(mSlotSize, newNumSlots, physSize);
    mVirtMem.realloc(physSize * static_cast<size_t>(newPhysMem));
}

int GpuSlotPool::extendByOnePhysMem()
{
    mVirtMem.extend(1);
    return numSlots();
}

Address GpuSlotPool::slotAddress(SlotId slot) const
{
    return MemAddress(mVirtMem.address() + mSlotSize * static_cast<size_t>(slot));
}

// ---------------------------------------------------------------------------
// HostSlotPool
// ---------------------------------------------------------------------------

HostSlotPool::HostSlotPool(size_t slotSize, int numSlots)
    : SlotPoolBase(slotSize)
    , mHostMem(alignedSize(numSlots))
{
}

size_t HostSlotPool::alignedSize(int numSlots) const noexcept
{
    return roundUp(static_cast<size_t>(numSlots) * mSlotSize, HostMem::kAlignment);
}

int HostSlotPool::numSlots() const noexcept
{
    return static_cast<int>(mHostMem.size() / mSlotSize);
}

void HostSlotPool::destroy()
{
    mHostMem.destroy();
}

void HostSlotPool::resize(int newNumSlots)
{
    mHostMem.resize(alignedSize(newNumSlots));
}

Address HostSlotPool::slotAddress(SlotId slot) const
{
    return MemAddress(mHostMem.address() + mSlotSize * static_cast<size_t>(slot));
}

// ---------------------------------------------------------------------------
// DiskSlotPool
// ---------------------------------------------------------------------------

DiskSlotPool::DiskSlotPool(std::string const& directory, size_t slotSize, int numSlots)
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

int DiskSlotPool::numSlots() const noexcept
{
    if (mFd == kBadFileDescriptor)
        return 0;
    off_t sz = ::lseek(mFd, 0, SEEK_END);
    return (sz < 0 || mSlotSize == 0) ? 0 : static_cast<int>(sz / static_cast<off_t>(mSlotSize));
}

void DiskSlotPool::destroy()
{
    if (mFd != kBadFileDescriptor)
    {
        ::close(mFd);
        mFd = kBadFileDescriptor;
    }
}

void DiskSlotPool::resize(int newNumSlots)
{
    resizeFile(mFd, static_cast<size_t>(newNumSlots) * mSlotSize);
}

Address DiskSlotPool::slotAddress(SlotId slot) const
{
    return DiskAddress{mFd, static_cast<ssize_t>(static_cast<size_t>(slot) * mSlotSize)};
}

// ---------------------------------------------------------------------------
// PoolGroupBase
// ---------------------------------------------------------------------------

PoolGroupBase::PoolGroupBase(int numSlots)
    : mSlotAllocator(numSlots)
{
}

PoolGroupBase::~PoolGroupBase()
{
    destroy();
}

int PoolGroupBase::numSlots() const noexcept
{
    return mSlotAllocator.numSlots();
}

Slot PoolGroupBase::allocate()
{
    return mSlotAllocator.allocate();
}

std::vector<Slot> PoolGroupBase::allocateMultiple(int numSlots)
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

void PoolGroupBase::resizePools(std::optional<int> newNumSlots)
{
    int n = newNumSlots.value_or(mSlotAllocator.numSlots());
    for (auto& p : mPools)
        p->resize(n);
}

std::vector<Address> PoolGroupBase::slotAddress(SlotId slotId) const
{
    std::vector<Address> addrs;
    addrs.reserve(mPools.size());
    for (auto const& p : mPools)
        addrs.push_back(p->slotAddress(slotId));
    return addrs;
}

std::vector<size_t> PoolGroupBase::slotSize() const
{
    std::vector<size_t> sizes;
    sizes.reserve(mPools.size());
    for (auto const& p : mPools)
        sizes.push_back(p->slotSize());
    return sizes;
}

// ---------------------------------------------------------------------------
// GpuPoolGroup
// ---------------------------------------------------------------------------

GpuPoolGroup::GpuPoolGroup(
    int numSlots, std::vector<size_t> const& slotSizeList, PooledPhysMemAllocator& physMemAllocator)
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
        vmSize = std::max(vmSize, roundUp(static_cast<size_t>(numSlots) * sz, physMemSize));
        mPools.push_back(std::make_unique<GpuSlotPool>(sz, vmSize, physMemAllocator, numSlots));
    }
}

// ---------------------------------------------------------------------------
// HostPoolGroup
// ---------------------------------------------------------------------------

HostPoolGroup::HostPoolGroup(int numSlots, std::vector<size_t> const& slotSizeList)
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

DiskPoolGroup::DiskPoolGroup(int numSlots, std::vector<size_t> const& slotSizeList, std::string const& directory)
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

std::vector<Slot> CacheLevelStorage::allocateMultiple(PoolGroupIndex pgIdx, int numSlots)
{
    return mPoolGroups.at(static_cast<size_t>(pgIdx))->allocateMultiple(numSlots);
}

void CacheLevelStorage::release(PoolGroupIndex pgIdx, Slot slot)
{
    mPoolGroups.at(static_cast<size_t>(pgIdx))->release(std::move(slot));
}

int CacheLevelStorage::numFreeSlots(PoolGroupIndex pgIdx) const
{
    return mPoolGroups.at(static_cast<size_t>(pgIdx))->numFreeSlots();
}

std::vector<Address> CacheLevelStorage::slotAddress(PoolGroupIndex pgIdx, SlotId slotId) const
{
    return mPoolGroups.at(static_cast<size_t>(pgIdx))->slotAddress(slotId);
}

// ---------------------------------------------------------------------------
// GpuCacheLevelStorage
// ---------------------------------------------------------------------------

GpuCacheLevelStorage::GpuCacheLevelStorage(
    StorageConfig const& storageCfg, std::vector<int> const& slotCountList, size_t physMemSize)
{
    mPhysMemAllocator = std::make_unique<PooledPhysMemAllocator>(physMemSize);

    for (size_t i = 0; i < storageCfg.slotDescList.size(); ++i)
        mPoolGroups.push_back(std::make_unique<GpuPoolGroup>(
            slotCountList[i], storageCfg.slotDescList[i].slotSizeList(), *mPhysMemAllocator));
}

// ---------------------------------------------------------------------------
// HostCacheLevelStorage
// ---------------------------------------------------------------------------

HostCacheLevelStorage::HostCacheLevelStorage(StorageConfig const& storageCfg, std::vector<int> const& slotCountList)
{
    for (size_t i = 0; i < storageCfg.slotDescList.size(); ++i)
        mPoolGroups.push_back(
            std::make_unique<HostPoolGroup>(slotCountList[i], storageCfg.slotDescList[i].slotSizeList()));
}

// ---------------------------------------------------------------------------
// DiskCacheLevelStorage
// ---------------------------------------------------------------------------

DiskCacheLevelStorage::DiskCacheLevelStorage(
    StorageConfig const& storageCfg, std::vector<int> const& slotCountList, std::string directory)
    : mDirectory(std::move(directory))
{
    for (size_t i = 0; i < storageCfg.slotDescList.size(); ++i)
        mPoolGroups.push_back(
            std::make_unique<DiskPoolGroup>(slotCountList[i], storageCfg.slotDescList[i].slotSizeList(), mDirectory));
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<CacheLevelStorage> createCacheLevelStorage(
    CacheTierConfig const& tierCfg, StorageConfig const& storageCfg, std::vector<int> const& slotCountList)
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

static int divUpLocal(int a, int b)
{
    return (a + b - 1) / b;
}

std::pair<int, int64_t> CacheLevelStorage::grainsToSlots(
    int64_t pgGrains, std::vector<int> const& slotSizeList, int granularity)
{
    int numPools = static_cast<int>(slotSizeList.size());
    std::vector<int64_t> minPoolGrains(static_cast<size_t>(numPools));
    for (int p = 0; p < numPools; ++p)
        minPoolGrains[p] = divUpLocal(slotSizeList[p], granularity);

    int64_t minTotal = 0;
    for (auto g : minPoolGrains)
        minTotal += g;
    pgGrains = std::max(pgGrains, minTotal);

    int numSlots = INT_MAX;
    int64_t remainingPgGrains = pgGrains;

    // Sort pools by slot size ascending.
    std::vector<int> poolOrder(static_cast<size_t>(numPools));
    std::iota(poolOrder.begin(), poolOrder.end(), 0);
    std::sort(poolOrder.begin(), poolOrder.end(), [&](int a, int b) { return slotSizeList[a] < slotSizeList[b]; });

    for (size_t j = 0; j < poolOrder.size(); ++j)
    {
        int pool = poolOrder[j];
        int slotSz = slotSizeList[pool];
        int poolSzSum = 0;
        for (size_t k = j; k < poolOrder.size(); ++k)
            poolSzSum += slotSizeList[poolOrder[k]];
        float poolFrac = (poolSzSum > 0) ? static_cast<float>(slotSz) / static_cast<float>(poolSzSum) : 1.f;
        int64_t poolGrains
            = std::max(minPoolGrains[pool], static_cast<int64_t>(std::round(remainingPgGrains * poolFrac)));
        numSlots = std::min(numSlots, static_cast<int>(poolGrains * granularity / slotSz));
        remainingPgGrains -= poolGrains;
    }
    assert(remainingPgGrains == 0);
    assert(numSlots > 0);
    return {numSlots, pgGrains};
}

// ---------------------------------------------------------------------------
// CacheLevelStorage helper: grainsForSlots
// Compute minimum grains needed for numSlots in a pool group.
// Mirrors Python CacheLevelStorage._grains_for_slots.
// ---------------------------------------------------------------------------

int64_t CacheLevelStorage::grainsForSlots(int numSlots, std::vector<int> const& slotSizeList, int granularity)
{
    int64_t total = 0;
    for (auto s : slotSizeList)
        total += divUpLocal(numSlots * s, granularity);
    return total;
}

// ---------------------------------------------------------------------------
// CacheLevelStorage::ratioToSlotCountList (static)
// Mirrors Python CacheLevelStorage.ratio_to_slot_count_list.
// ---------------------------------------------------------------------------

std::vector<int> CacheLevelStorage::ratioToSlotCountList(size_t quota, std::vector<std::vector<int>> const& sizeLists,
    std::vector<float> const& ratioList, int granularity, std::vector<int> const& minSlots)
{
    int numPg = static_cast<int>(sizeLists.size());
    assert(static_cast<int>(ratioList.size()) == numPg);
    assert(quota % static_cast<size_t>(granularity) == 0);
    int64_t totalGrains = static_cast<int64_t>(quota) / granularity;

    int g = granularity;

    // Step 0: compute unconstrained slot counts from ratio.
    std::vector<int> unconstrained(static_cast<size_t>(numPg), 0);
    std::vector<int64_t> grains(static_cast<size_t>(numPg), 0);
    int64_t remaining = totalGrains;

    // Sort pool groups by ratio ascending.
    std::vector<int> pgOrder(static_cast<size_t>(numPg));
    std::iota(pgOrder.begin(), pgOrder.end(), 0);
    std::sort(pgOrder.begin(), pgOrder.end(), [&](int a, int b) { return ratioList[a] < ratioList[b]; });

    for (size_t i = 0; i < pgOrder.size(); ++i)
    {
        int pg = pgOrder[i];
        float ratioSum = 0.f;
        for (size_t j = i; j < pgOrder.size(); ++j)
            ratioSum += ratioList[pgOrder[j]];
        float pct = (ratioSum > 0.f) ? ratioList[pg] / ratioSum : 1.f;

        auto [slots, used] = grainsToSlots(std::llround(remaining * pct), sizeLists[pg], g);
        unconstrained[pg] = slots;
        grains[pg] = used;
        remaining -= used;
    }
    assert(remaining == 0);

    // Step 1: identify constrained pool groups (unconstrained < min_slots).
    std::set<int> constrainedPgs;
    for (int pg = 0; pg < numPg; ++pg)
    {
        if (unconstrained[pg] < minSlots[pg])
            constrainedPgs.insert(pg);
    }
    if (constrainedPgs.empty())
        return unconstrained;

    // Step 2: floor constrained PGs to min_slots, compute extra grains needed.
    auto slotCntList = unconstrained;
    int64_t extraNeeded = 0;
    for (int pg : constrainedPgs)
    {
        auto [slots, used] = grainsToSlots(grainsForSlots(minSlots[pg], sizeLists[pg], g), sizeLists[pg], g);
        extraNeeded += used - grains[pg];
        slotCntList[pg] = slots;
        grains[pg] = used;
    }

    // Step 3: unconstrained PGs pay the extra cost proportionally by ratio.
    std::vector<int> freePgs;
    for (int pg = 0; pg < numPg; ++pg)
    {
        if (constrainedPgs.find(pg) == constrainedPgs.end())
            freePgs.push_back(pg);
    }
    std::sort(freePgs.begin(), freePgs.end(), [&](int a, int b) { return ratioList[a] < ratioList[b]; });

    while (extraNeeded > 0 && !freePgs.empty())
    {
        std::unordered_map<int, int64_t> spares;
        int64_t totalSpare = 0;
        for (int pg : freePgs)
        {
            int64_t spare = grains[pg] - grainsForSlots(minSlots[pg], sizeLists[pg], g);
            spares[pg] = spare;
            totalSpare += std::max<int64_t>(0, spare);
        }
        assert(totalSpare > 0 && "Insufficient quota to satisfy min_slots constraints");

        float totalRatio = 0.f;
        for (int pg : freePgs)
            totalRatio += ratioList[pg];

        std::vector<int> newFreePgs;
        // Use signed math for still_needed (as Python comment suggests).
        int64_t stillNeeded = extraNeeded;
        for (int pg : freePgs)
        {
            int64_t share = std::llround(extraNeeded * (static_cast<double>(ratioList[pg]) / totalRatio));
            int64_t give = std::min(share, std::max<int64_t>(0, spares[pg]));
            int64_t newGrains = grains[pg] - give;
            auto [slots, used] = grainsToSlots(newGrains, sizeLists[pg], g);
            stillNeeded -= grains[pg] - used;
            slotCntList[pg] = slots;
            grains[pg] = used;
            if (used > grainsForSlots(minSlots[pg], sizeLists[pg], g))
                newFreePgs.push_back(pg);
        }
        extraNeeded = std::max<int64_t>(0, stillNeeded);
        freePgs = std::move(newFreePgs);
    }

    return slotCntList;
}

// Instance convenience wrapper.
std::vector<int> CacheLevelStorage::computeSlotCountList(
    std::vector<float> const& ratioList, std::vector<int> const& minSlots, std::optional<size_t> quota) const
{
    size_t q = quota.value_or(totalQuota());
    assert(static_cast<int>(ratioList.size()) == static_cast<int>(mPoolGroups.size()));
    return ratioToSlotCountList(q, slotSizeLists(), ratioList, poolSizeGranularity(), minSlots);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
