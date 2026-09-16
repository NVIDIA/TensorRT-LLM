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
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/hostMem.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
class KvCache;
class KvCacheManager;
class HostPageRead;
class StorageManager;

//! Borrowed metadata only. No KV ownership, model layout, allocation, or transfer.
//! CPU pointers refer to mapped pinned memory; gpuAddress() gives their CUDA aliases.
//! Keep the manager alive, and acquire a HostSourceRead for each GPU use (including graph replay).
struct HostSourceView
{
    int maxRequests;
    int maxBeams;
    int numLifeCycles;
    int maxPages;
    int numLevels;
    int tokensPerBlock;

    // [maxRequests]. Validity is separate so all uint64 request IDs are supported.
    uint64_t const* requestIds;
    uint64_t const* generations;
    int32_t const* requestValid;
    // [maxRequests, maxBeams, numLifeCycles, maxPages]. Invalid: -1, -1, 0.
    int64_t const* slotIds;
    int32_t const* hostLevels;
    int32_t const* completedTokens;
    // [numLevels, numLifeCycles, 4]: GPU-readable base address, slot bytes, pool bytes, pool group.
    // Cold-page codecs currently have one host pool per lifecycle. Non-host tiers are all zero.
    uint64_t const* poolMetadata;

    MemAddress hostBase;
    MemAddress deviceBase;

    MemAddress gpuAddress(void const* pointer) const noexcept
    {
        return deviceBase + (reinterpret_cast<MemAddress>(pointer) - hostBase);
    }

    size_t pageIndex(int request, int beam, int lifeCycle, int page) const noexcept
    {
        return ((static_cast<size_t>(request) * maxBeams + beam) * numLifeCycles + lifeCycle) * maxPages + page;
    }
};

//! Separate from the non-owning view: holds HostPageRead handles through one submitted GPU use.
//! close() records completion. Releasing the last request owner can also run synchronous cleanup.
//! Close before changing table metadata.
class HostSourceRead
{
public:
    ~HostSourceRead();
    HostSourceRead(HostSourceRead const&) = delete;
    HostSourceRead& operator=(HostSourceRead const&) = delete;
    void close();

private:
    friend class HostSourceTable;
    HostSourceRead(std::shared_ptr<KvCacheManager> owner, CUstream stream);
    std::shared_ptr<KvCacheManager> mOwner;
    CUstream mStream;
    // Keep request destructors from changing the table while the scope is open.
    std::vector<std::shared_ptr<KvCache>> mRequests;
    std::vector<std::unique_ptr<HostPageRead>> mPages;
};

//! KVCM-owned, fixed-capacity host-source metadata. All calls require the manager's exclusive lock.
//! Updates happen outside graph capture. They wait for earlier table readers, never for backups.
class HostSourceTable
{
public:
    HostSourceTable(StorageManager& storage, int tokensPerBlock, int maxRequests, int maxPages, int maxBeams);
    ~HostSourceTable();

    HostSourceView view() const noexcept
    {
        return mView;
    }

    void addRequest(KvCache& cache);
    void setRequestId(KvCache& cache, std::optional<RequestIdType> id);
    void removeRequest(KvCache const& cache);
    int requestSlot(KvCache const& cache) const;
    void checkCapacity(KvCache const& cache, int capacity) const;
    void beginUpdate();
    void endUpdate();
    void refresh();
    std::unique_ptr<HostSourceRead> acquire(std::shared_ptr<KvCacheManager> owner, CUstream stream);
    void finishRead(CachedCudaEvent event);
    void waitForReaders();

private:
    void fill();
    void clearSources();
    size_t numEntries() const noexcept;
    StorageManager& mStorage;
    std::unique_ptr<HostMem> mMemory;
    HostSourceView mView{};
    std::vector<KvCache*> mRequests;
    std::vector<CachedCudaEvent> mReadEvents;
    int mOpenReaders = 0;
    int mUpdateDepth = 0;
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
