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

#include "kv_cache_manager_v2/hostSourceTable.h"
#include "kv_cache_manager_v2/hostPageCopy.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/utils/optionalGilRelease.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{
size_t checkedProduct(size_t a, size_t b)
{
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
    {
        throw std::invalid_argument("Host source table size overflow");
    }
    return a * b;
}

size_t checkedSum(size_t a, size_t b)
{
    if (a > std::numeric_limits<size_t>::max() - b)
    {
        throw std::invalid_argument("Host source table size overflow");
    }
    return a + b;
}

void checkOutsideCapture(CUstream stream)
{
    CUstreamCaptureStatus status;
    cuCheck(cuStreamIsCapturing(stream, &status));
    if (status != CU_STREAM_CAPTURE_STATUS_NONE)
    {
        throw LogicError("Acquire and close host source reads outside CUDA graph capture");
    }
}

template <typename T>
T* writable(T const* pointer)
{
    return const_cast<T*>(pointer);
}
} // namespace

HostSourceTable::HostSourceTable(
    StorageManager& storage, int tokensPerBlock, int maxRequests, int maxPages, int maxBeams)
    : mStorage(storage)
{
    if (maxRequests <= 0 || maxPages <= 0 || maxBeams <= 0)
    {
        throw std::invalid_argument("Host source table capacities must be positive");
    }
    mView.maxRequests = maxRequests;
    mView.maxPages = maxPages;
    mView.maxBeams = maxBeams;
    mView.numLifeCycles = storage.numLifeCycles().value();
    mView.numLevels = storage.numCacheLevels().value();
    mView.tokensPerBlock = tokensPerBlock;
    size_t const entries
        = checkedProduct(checkedProduct(checkedProduct(maxRequests, maxBeams), mView.numLifeCycles), maxPages);
    size_t const pools = checkedProduct(checkedProduct(mView.numLevels, mView.numLifeCycles), 4);
    // Each array starts on an eight-byte boundary. Reserve a little extra for alignment.
    size_t const requestsBytes = checkedProduct(maxRequests, 24);
    size_t const entriesBytes = checkedProduct(entries, 16);
    size_t const poolsBytes = checkedProduct(pools, sizeof(uint64_t));
    size_t const bytes = checkedSum(checkedSum(requestsBytes, entriesBytes), checkedSum(poolsBytes, 64));
    mMemory = std::make_unique<HostMem>(bytes);
    mView.hostBase = mMemory->address();
    CUdeviceptr device = 0;
    cuCheck(cuMemHostGetDevicePointer(&device, reinterpret_cast<void*>(mView.hostBase), 0));
    mView.deviceBase = device;
    std::memset(reinterpret_cast<void*>(mView.hostBase), 0, mMemory->size());
    size_t offset = 0;
    auto allocate = [&](auto*& pointer, size_t count)
    {
        using T = std::remove_pointer_t<std::decay_t<decltype(pointer)>>;
        offset = (offset + 7) / 8 * 8;
        pointer = reinterpret_cast<T*>(mView.hostBase + offset);
        offset += count * sizeof(T);
    };
    allocate(mView.requestIds, maxRequests);
    allocate(mView.generations, maxRequests);
    allocate(mView.requestValid, maxRequests);
    allocate(mView.slotIds, entries);
    allocate(mView.hostLevels, entries);
    allocate(mView.completedTokens, entries);
    allocate(mView.poolMetadata, pools);
    TLLM_CHECK(offset <= mMemory->size());
    mRequests.resize(maxRequests, nullptr);
    fill();
}

HostSourceTable::~HostSourceTable()
{
    KVCM2_POISON_ON_EXCEPT([this]() { waitForReaders(); });
}

size_t HostSourceTable::numEntries() const noexcept
{
    return static_cast<size_t>(mView.maxRequests) * mView.maxBeams * mView.numLifeCycles * mView.maxPages;
}

int HostSourceTable::requestSlot(KvCache const& cache) const
{
    auto const it = std::find(mRequests.begin(), mRequests.end(), &cache);
    return it == mRequests.end() ? -1 : static_cast<int>(it - mRequests.begin());
}

void HostSourceTable::checkCapacity(KvCache const& cache, int capacity) const
{
    if (cache.id
        && (static_cast<int64_t>(capacity) > static_cast<int64_t>(mView.maxPages) * mView.tokensPerBlock
            || cache.beamWidth().value() > mView.maxBeams))
    {
        throw std::out_of_range("Request exceeds the reserved host source table capacity");
    }
}

void HostSourceTable::addRequest(KvCache& cache)
{
    setRequestId(cache, cache.id);
}

void HostSourceTable::setRequestId(KvCache& cache, std::optional<RequestIdType> id)
{
    int const oldSlot = requestSlot(cache);
    if (id == cache.id && (oldSlot >= 0 || !id))
    {
        return;
    }
    if (!id || cache.isClosed())
    {
        removeRequest(cache);
        cache.id = id;
        return;
    }
    if (static_cast<int64_t>(cache.capacity()) > static_cast<int64_t>(mView.maxPages) * mView.tokensPerBlock
        || cache.beamWidth().value() > mView.maxBeams)
    {
        throw std::out_of_range("Request exceeds the reserved host source table capacity");
    }
    for (auto const* existing : mRequests)
    {
        if (existing && existing != &cache && existing->id == id)
        {
            throw std::invalid_argument("Host source table request IDs must be unique");
        }
    }
    auto const it = oldSlot >= 0 ? mRequests.begin() + oldSlot : std::find(mRequests.begin(), mRequests.end(), nullptr);
    if (it == mRequests.end())
    {
        throw std::out_of_range("No free request slot in the host source table");
    }
    auto const slot = it - mRequests.begin();
    auto& generation = writable(mView.generations)[slot];
    if (generation == std::numeric_limits<uint64_t>::max())
    {
        throw LogicError("Host source request generation exhausted");
    }
    ++generation;
    cache.id = id;
    *it = &cache;
}

void HostSourceTable::removeRequest(KvCache const& cache)
{
    int const slot = requestSlot(cache);
    if (slot >= 0)
    {
        mRequests[slot] = nullptr;
    }
}

void HostSourceTable::waitForReaders()
{
    if (mOpenReaders != 0)
    {
        throw LogicError("Close host source readers before changing the table");
    }
    for (auto const& event : mReadEvents)
    {
        event.synchronize();
    }
    mReadEvents.clear();
}

void HostSourceTable::clearSources()
{
    std::fill_n(writable(mView.slotIds), numEntries(), -1);
    std::fill_n(writable(mView.hostLevels), numEntries(), -1);
    std::fill_n(writable(mView.completedTokens), numEntries(), 0);
}

void HostSourceTable::beginUpdate()
{
    if (mUpdateDepth == 0)
    {
        waitForReaders();
        clearSources();
    }
    ++mUpdateDepth;
}

void HostSourceTable::endUpdate()
{
    TLLM_CHECK(mUpdateDepth > 0);
    if (--mUpdateDepth == 0)
    {
        fill();
    }
}

void HostSourceTable::refresh()
{
    if (mUpdateDepth != 0)
    {
        throw LogicError("Cannot refresh host sources during a lifecycle update");
    }
    waitForReaders();
    fill();
}

void HostSourceTable::fill()
{
    clearSources();
    std::fill_n(writable(mView.requestValid), mView.maxRequests, 0);
    std::fill_n(writable(mView.requestIds), mView.maxRequests, 0);
    std::fill_n(writable(mView.poolMetadata), static_cast<size_t>(mView.numLevels) * mView.numLifeCycles * 4, 0);
    for (CacheLevel level{1}; level < mStorage.numCacheLevels(); ++level)
    {
        if (mStorage.cacheTier(level) != CacheTier::HOST_MEM)
        {
            continue;
        }
        for (LifeCycleId lc{0}; lc < mStorage.numLifeCycles(); ++lc)
        {
            auto const group = mStorage.getPoolGroupIndex(level, lc);
            auto const& sizes = mStorage.slotSize(level, group);
            TLLM_CHECK_WITH_INFO(sizes.size() == PoolIndex{1}, "Host source table requires a single cold-page pool");
            auto* pool = writable(mView.poolMetadata) + (level.value() * mView.numLifeCycles + lc.value()) * 4;
            pool[1] = sizes.at(PoolIndex{0});
            pool[2] = pool[1] * slotCountToSizeT(mStorage.numSlots(group, level));
            pool[3] = group.value();
            if (pool[2] != 0)
            {
                auto const base = std::get<MemAddress>(mStorage.slotAddress(level, group, SlotId{0}, PoolIndex{0}));
                CUdeviceptr device = 0;
                cuCheck(cuMemHostGetDevicePointer(&device, reinterpret_cast<void*>(base), 0));
                pool[0] = device;
            }
        }
    }
    for (int row = 0; row < mView.maxRequests; ++row)
    {
        auto const* cache = mRequests[row];
        if (!cache)
        {
            continue;
        }
        writable(mView.requestIds)[row] = *cache->id;
        writable(mView.requestValid)[row] = 1;
        for (BlockOrdinal ordinal{0}; ordinal < cache->numBlocks(); ++ordinal)
        {
            auto const& beams = cache->blocks()[ordinal].pages;
            for (BeamIndex beam{0}; beam < beams.size(); ++beam)
            {
                for (LifeCycleId lc{0}; lc < mStorage.numLifeCycles(); ++lc)
                {
                    auto const& page = blockPageGetPage(beams[beam][lc]);
                    if (!page || !std::holds_alternative<AttnLifeCycle>(mStorage.getLifeCycle(lc)))
                    {
                        continue;
                    }
                    auto const& copy = page->hostCopy();
                    if (!copy || !copy->ready())
                    {
                        continue;
                    }
                    int coverage = std::clamp(cache->historyLength() - ordinal.value() * mView.tokensPerBlock, 0,
                        std::min(mView.tokensPerBlock, copy->validTokens()));
                    if (page->isCommitted())
                    {
                        coverage = std::min(coverage, static_cast<CommittedPage const&>(*page).numTokensInBlock);
                    }
                    if (coverage == 0)
                    {
                        continue;
                    }
                    auto const index = mView.pageIndex(row, beam.value(), lc.value(), ordinal.value());
                    writable(mView.slotIds)[index] = copy->slotId().value();
                    writable(mView.hostLevels)[index] = copy->level().value();
                    writable(mView.completedTokens)[index] = coverage;
                }
            }
        }
    }
}

std::unique_ptr<HostSourceRead> HostSourceTable::acquire(std::shared_ptr<KvCacheManager> owner, CUstream stream)
{
    checkOutsideCapture(stream);
    if (mOpenReaders == 0)
    {
        refresh();
    }
    auto read = std::unique_ptr<HostSourceRead>(new HostSourceRead(std::move(owner), stream));
    ++mOpenReaders;
    std::unordered_set<HostPageCopy*> seen;
    for (int row = 0; row < mView.maxRequests; ++row)
    {
        auto const* cache = mRequests[row];
        if (!cache)
        {
            continue;
        }
        for (BlockOrdinal ordinal{0}; ordinal < cache->numBlocks(); ++ordinal)
        {
            auto const& beams = cache->blocks()[ordinal].pages;
            for (BeamIndex beam{0}; beam < beams.size(); ++beam)
            {
                for (LifeCycleId lc{0}; lc < mStorage.numLifeCycles(); ++lc)
                {
                    auto const index = mView.pageIndex(row, beam.value(), lc.value(), ordinal.value());
                    if (mView.completedTokens[index] == 0)
                    {
                        continue;
                    }
                    auto const& copy = blockPageGetPage(beams[beam][lc])->hostCopy();
                    if (seen.insert(copy.get()).second)
                    {
                        read->mPages.push_back(std::make_unique<HostPageRead>(read->mOwner, copy, stream));
                    }
                }
            }
        }
    }
    return read;
}

void HostSourceTable::finishRead(CachedCudaEvent event)
{
    TLLM_CHECK(mOpenReaders > 0);
    mReadEvents.push_back(std::move(event));
    --mOpenReaders;
}

HostSourceRead::HostSourceRead(std::shared_ptr<KvCacheManager> owner, CUstream stream)
    : mOwner(std::move(owner))
    , mStream(stream)
{
    for (auto* cache : mOwner->mLivingKvCaches)
    {
        mRequests.push_back(cache->shared_from_this());
    }
}

HostSourceRead::~HostSourceRead()
{
    KVCM2_POISON_ON_EXCEPT(
        [this]()
        {
            OptionalGilRelease const gilRelease;
            close();
        });
}

void HostSourceRead::close()
{
    if (!mOwner)
    {
        return;
    }
    auto const owner = mOwner;
    auto const apiLock = owner->lockExclusive();
    if (Poison::poisoned())
    {
        return;
    }
    checkOutsideCapture(mStream);
    for (auto& page : mPages)
    {
        page->close();
    }
    mPages.clear();
    owner->mHostSources->finishRead(CachedCudaEvent(reinterpret_cast<CudaStream>(mStream)));
    mRequests.clear();
    mOwner.reset();
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
