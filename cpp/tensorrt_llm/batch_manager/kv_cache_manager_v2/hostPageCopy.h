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

#include "kv_cache_manager_v2/storage/core.h"
#include "kv_cache_manager_v2/utils/reentrantSharedMutex.h"

#include <memory>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
class StorageManager;
class KvCacheManager;
class HostPageRead;

//! A retained host slot, independent of the page's ordinary GPU/host slot.
//! StorageManager owns the pools. Pages own copies; external readers also keep
//! KvCacheManager alive. Callers hold the manager API lock when accessing shared state.
class HostPageCopy
{
public:
    HostPageCopy(StorageManager& manager, LifeCycleId lifeCycle, CacheLevel level, Slot& slot);
    ~HostPageCopy();
    HostPageCopy(HostPageCopy const&) = delete;
    HostPageCopy& operator=(HostPageCopy const&) = delete;

    bool ready();

    //! Covered prefix after the backup event. Use ready() before publishing it globally.
    int validTokens() const noexcept
    {
        return mValidTokens;
    }

    CacheLevel level() const noexcept
    {
        return mLevel;
    }

    SlotId slotId() const
    {
        return mSlot.slotId();
    }

    PoolGroupIndex poolGroup() const;
    MemAddress address() const;
    MemAddress poolBaseAddress() const;
    size_t pageBytes() const;
    size_t poolBytes() const;
    bool matches(CacheLevel level, SlotId slotId) const;

    //! Invalidate before submitting new GPU writes. Open readers must first close.
    void invalidate();

private:
    friend class StorageManager;
    friend class Page;
    friend class HostPageRead;
    void beginBackup(CUstream stream);
    void finishBackup(CachedCudaEvent event, int validTokens);
    void beginRead(CUstream stream);
    void finishRead(CUstream stream);
    void recordUse(CachedCudaEvent event);
    CachedCudaEvent allUses();

    StorageManager* mManager;
    LifeCycleId mLifeCycle;
    CacheLevel mLevel;
    Slot mSlot;
    CachedCudaEvent mBackupReady = CachedCudaEvent::makeNull();
    int mValidTokens = 0;
    int mOpenReaders = 0;
    std::vector<CachedCudaEvent> mUses;
};

//! Keeps host addresses alive through work submitted on one CUDA stream.
//! Acquisition waits on backup in that stream. close() records completion;
//! neither operation waits on the CPU. Close before invalidating or overwriting.
class HostPageRead
{
public:
    HostPageRead(std::shared_ptr<KvCacheManager> owner, std::shared_ptr<HostPageCopy> copy, CUstream stream);
    ~HostPageRead();
    HostPageRead(HostPageRead const&) = delete;
    HostPageRead& operator=(HostPageRead const&) = delete;
    void close();
    //! Internal access: hold lockShared() while reading the returned copy.
    HostPageCopy const& copy() const;
    [[nodiscard]] ReentrantSharedMutex::Guard lockShared() const;
    bool ready();
    int completedTokens();

private:
    std::shared_ptr<KvCacheManager> mOwner;
    std::shared_ptr<HostPageCopy> mCopy;
    CUstream mStream;
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
