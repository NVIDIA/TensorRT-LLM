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
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
class KvCache;
class KvCacheManager;
class HostPageRead;
class StorageManager;
class Page;

//! IndexMapper request row and the generation expected by this batch.
using HostSourceRow = std::pair<int, uint64_t>;

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
//! Close before changing the batch's rows. Other request rows remain independent.
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
    std::vector<HostSourceRow> mRows;
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

    //! No row allocator here: the executor supplies its existing IndexMapper row. -1 releases it.
    void bindRequest(KvCache& cache, int row);
    void setRequestId(KvCache& cache, std::optional<RequestIdType> id);
    void removeRequest(KvCache const& cache);
    int requestSlot(KvCache const& cache) const;
    HostSourceRow requestRef(KvCache const& cache) const;
    void checkCapacity(KvCache const& cache, int capacity) const;
    // Page updates include shared users. range limits append updates; other structural changes rebuild this request.
    void beginUpdate(
        KvCache* cache, std::optional<BlockOrdinal> ordinal, std::optional<std::pair<int, int>> range = std::nullopt);
    void endUpdate();
    void refreshForTest();
    std::unique_ptr<HostSourceRead> acquire(
        std::shared_ptr<KvCacheManager> owner, std::vector<HostSourceRow> const& rows, CUstream stream);
    void finishRead(std::vector<HostSourceRow> const& rows, CachedCudaEvent event);
    void waitForReaders();

private:
    struct RequestRow
    {
        KvCache* cache = nullptr;
        int openReaders = 0;
        std::vector<CachedCudaEvent> readEvents;
        std::unordered_set<size_t> entries;
        std::unordered_set<size_t> pending;
        std::unordered_set<size_t> published;
    };

    struct Source
    {
        Page const* key; // Only used to remove the reverse index; never dereferenced.
        WeakPtr<Page> page;
        int coverage;
    };

    void updatePools();
    void rebuildRow(int row);
    void rebuildRange(int row, int begin, int end);
    void eraseEntry(size_t index);
    void publish(size_t index);
    void clearEntry(size_t index);
    void eraseEntries(int row);
    void waitForRow(int row);
    int rowForEntry(size_t index) const noexcept;
    size_t numEntries() const noexcept;
    StorageManager& mStorage;
    std::unique_ptr<HostMem> mMemory;
    HostSourceView mView{};
    std::vector<RequestRow> mRows;
    std::unordered_map<KvCache const*, int> mRequestRows;
    std::unordered_map<size_t, Source> mSources;
    std::unordered_map<Page const*, std::unordered_set<size_t>> mPageEntries;
    std::unordered_set<int> mDirtyRows;
    std::unordered_map<int, std::pair<int, int>> mDirtyRanges;
    std::unordered_set<size_t> mDirtyEntries;
    std::vector<CachedCudaEvent> mEmptyReadEvents;
    bool mPoolsDirty = false;
    int mOpenReaders = 0;
    int mUpdateDepth = 0;
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
