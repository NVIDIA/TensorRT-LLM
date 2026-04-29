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
        SlotId id = slot.slotId();
        if (id < mTargetCapacity || id >= mCapacity)
            return false;
    }
    if (static_cast<int>(mRecycledSlots.size()) + static_cast<int>(mOverflowSlots.size()) + numOccupiedSlots()
        != mNumActiveSlots)
        return false;
    return true;
}

void SlotAllocator::forceFinishShrink() noexcept
{
    // Force-complete a pending shrink without throwing.
    // Used in destroy() paths where we cannot propagate exceptions.
    if (!shrinkInProgress())
        return;
    // Synchronize and discard all overflow slots.
    for (auto& s : mOverflowSlots)
    {
        try
        {
            s.readyEvent.synchronize();
        }
        catch (...)
        {
        }
        s.resetSlot();
    }
    mOverflowSlots.clear();
    mCapacity = mTargetCapacity;
    mNumActiveSlots = std::min(mNumActiveSlots, mCapacity);
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

std::pair<int, int64_t> CacheLevelStorage::grainsToSlots(
    int64_t pgGrains, std::vector<int> const& slotSizeList, int granularity)
{
    int numPools = static_cast<int>(slotSizeList.size());
    std::vector<int64_t> minPoolGrains(static_cast<size_t>(numPools));
    for (int p = 0; p < numPools; ++p)
        minPoolGrains[p] = divUp(slotSizeList[p], granularity);

    int64_t minTotal = 0;
    for (auto g : minPoolGrains)
        minTotal += g;
    if (pgGrains < minTotal)
        return {0, 0};

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
    return {numSlots, grainsForSlots(numSlots, slotSizeList, granularity)};
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
        total += divUp(numSlots * s, granularity);
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

    std::vector<int> slotCntList(static_cast<size_t>(numPg), 0);
    int64_t remainingGrains = totalGrains;
    std::vector<int> activePgs(static_cast<size_t>(numPg));
    std::iota(activePgs.begin(), activePgs.end(), 0);

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

        std::vector<int> slotsForActive(nActive, 0);
        std::vector<int64_t> grainsForActive(nActive, 0);
        int64_t budget = remainingGrains;

        // Sort indices by ratio ascending.
        std::vector<size_t> idxLst(nActive);
        std::iota(idxLst.begin(), idxLst.end(), size_t{0});
        std::sort(idxLst.begin(), idxLst.end(), [&](size_t a, size_t b) { return activeRatio[a] < activeRatio[b]; });

        for (size_t i = 0; i < idxLst.size(); ++i)
        {
            size_t idx = idxLst[i];
            float ratioSum = 0.f;
            for (size_t j = i; j < idxLst.size(); ++j)
                ratioSum += activeRatio[idxLst[j]];
            float pct = (ratioSum > 0.f) ? activeRatio[idx] / ratioSum : 1.f;
            auto [slots, used] = grainsToSlots(std::llround(budget * pct), sizeLists[activePgs[idx]], g);
            slotsForActive[idx] = slots;
            grainsForActive[idx] = used;
            budget -= used;
        }
        assert(budget >= 0);

        // Identify constrained PGs (slots <= min_slots).
        std::vector<size_t> constrained;
        std::vector<size_t> unconstrained;
        for (size_t idx = 0; idx < nActive; ++idx)
        {
            int pg = activePgs[idx];
            if (slotsForActive[idx] <= minSlots[pg])
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
            int pg = activePgs[idx];
            int64_t minGrains = grainsForSlots(minSlots[pg], sizeLists[pg], g);
            auto [slots, used] = grainsToSlots(minGrains, sizeLists[pg], g);
            slotCntList[pg] = slots;
            remainingGrains -= used;
        }

        if (unconstrained.empty())
        {
            // All PGs are constrained — nothing left to redistribute.
            break;
        }

        if (remainingGrains <= 0)
            throw std::runtime_error("Insufficient quota to satisfy min_slots constraints");

        // Continue with unconstrained PGs only.
        std::vector<int> newActivePgs;
        newActivePgs.reserve(unconstrained.size());
        for (size_t idx : unconstrained)
            newActivePgs.push_back(activePgs[idx]);
        activePgs = std::move(newActivePgs);
    }

    // _g2s may under-count slots due to imperfect grain distribution
    // across pools. Try bumping each PG's slot count while it still fits
    // within the same grain budget.
    for (int pg = 0; pg < numPg; ++pg)
    {
        int64_t grainsNow = grainsForSlots(slotCntList[pg], sizeLists[pg], g);
        while (grainsForSlots(slotCntList[pg] + 1, sizeLists[pg], g) <= grainsNow)
            slotCntList[pg] += 1;
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
