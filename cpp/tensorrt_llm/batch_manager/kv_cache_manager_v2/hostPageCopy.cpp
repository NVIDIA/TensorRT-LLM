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

#include "kv_cache_manager_v2/hostPageCopy.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/optionalGilRelease.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
HostPageCopy::HostPageCopy(StorageManager& manager, LifeCycleId lifeCycle, CacheLevel level, Slot& slot)
    : mManager(&manager)
    , mLifeCycle(lifeCycle)
    , mLevel(level)
{
    manager.pinHostCopy(level, manager.getPoolGroupIndex(level, lifeCycle));
    mSlot.setSlot(slot);
}

HostPageCopy::~HostPageCopy()
{
    KVCM2_POISON_ON_EXCEPT(
        [this]()
        {
            TLLM_CHECK(mOpenReaders == 0);
            mSlot.readyEvent = allUses();
            mManager->unpinHostCopy(mLevel, poolGroup(), mSlot.readyEvent);
            mManager->releaseSlot(mLifeCycle, mLevel, std::move(mSlot));
        });
}

bool HostPageCopy::ready()
{
    return mValidTokens > 0 && mBackupReady.queryComplete();
}

PoolGroupIndex HostPageCopy::poolGroup() const
{
    return mManager->getPoolGroupIndex(mLevel, mLifeCycle);
}

MemAddress HostPageCopy::address() const
{
    return std::get<MemAddress>(mManager->slotAddress(mLevel, poolGroup(), slotId(), PoolIndex{0}));
}

MemAddress HostPageCopy::poolBaseAddress() const
{
    return std::get<MemAddress>(mManager->slotAddress(mLevel, poolGroup(), SlotId{0}, PoolIndex{0}));
}

size_t HostPageCopy::pageBytes() const
{
    return mManager->slotSize(mLevel, poolGroup()).at(PoolIndex{0});
}

size_t HostPageCopy::poolBytes() const
{
    return pageBytes() * slotCountToSizeT(mManager->numSlots(poolGroup(), mLevel));
}

bool HostPageCopy::matches(CacheLevel level, SlotId slotId) const
{
    return level == mLevel && slotId == mSlot.slotId();
}

void HostPageCopy::invalidate()
{
    if (mOpenReaders != 0)
        throw LogicError("Close host readers before invalidating a host copy");
    mValidTokens = 0;
}

CachedCudaEvent HostPageCopy::allUses()
{
    mUses.push_back(mSlot.readyEvent);
    auto event = mergeEvents(mUses);
    mUses.clear();
    return event;
}

void HostPageCopy::beginBackup(CUstream stream)
{
    invalidate();
    mSlot.readyEvent = allUses();
    mSlot.readyEvent.waitInStream(reinterpret_cast<CudaStream>(stream));
}

void HostPageCopy::finishBackup(CachedCudaEvent event, int validTokens)
{
    mSlot.readyEvent = event;
    mBackupReady = std::move(event);
    mValidTokens = validTokens;
}

void HostPageCopy::beginRead(CUstream stream)
{
    if (mValidTokens <= 0)
        throw LogicError("Host copy has no valid data");
    mBackupReady.waitInStream(reinterpret_cast<CudaStream>(stream));
    ++mOpenReaders;
    ++mManager->mHostReaders;
}

void HostPageCopy::recordUse(CachedCudaEvent event)
{
    mUses.push_back(std::move(event));
    if (mUses.size() > 32)
    {
        mSlot.readyEvent = allUses();
    }
}

void HostPageCopy::finishRead(CUstream stream)
{
    TLLM_CHECK(mOpenReaders > 0);
    recordUse(CachedCudaEvent(reinterpret_cast<CudaStream>(stream)));
    --mOpenReaders;
    --mManager->mHostReaders;
}

HostPageRead::HostPageRead(std::shared_ptr<KvCacheManager> owner, std::shared_ptr<HostPageCopy> copy, CUstream stream)
    : mOwner(std::move(owner))
    , mCopy(std::move(copy))
    , mStream(stream)
{
    KVCM2_API_GUARD();
    auto const apiLock = mOwner->lockExclusive();
    if (!mCopy)
    {
        throw LogicError("Page has no retained host copy");
    }
    mCopy->beginRead(stream);
}

HostPageRead::~HostPageRead()
{
    KVCM2_POISON_ON_EXCEPT(
        [this]()
        {
            OptionalGilRelease const gilRelease;
            close();
        });
}

void HostPageRead::close()
{
    if (!mOwner)
    {
        return;
    }
    // Keep the manager alive until the API lock has been released.
    auto const owner = mOwner;
    auto const apiLock = owner->lockExclusive();
    if (Poison::poisoned())
    {
        return;
    }
    if (mCopy)
    {
        mCopy->finishRead(mStream);
        mCopy.reset();
        mOwner.reset();
    }
}

bool HostPageRead::ready()
{
    auto const apiLock = lockShared();
    return mCopy->ready();
}

int HostPageRead::completedTokens()
{
    auto const apiLock = lockShared();
    return mCopy->ready() ? mCopy->validTokens() : 0;
}

ReentrantSharedMutex::Guard HostPageRead::lockShared() const
{
    KVCM2_REJECT_IF_POISONED();
    copy(); // Reject a closed reader before accessing its manager.
    return mOwner->lockShared();
}

HostPageCopy const& HostPageRead::copy() const
{
    if (!mCopy)
        throw LogicError("Host reader is closed");
    return *mCopy;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
