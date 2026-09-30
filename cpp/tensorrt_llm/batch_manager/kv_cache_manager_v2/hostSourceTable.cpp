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
    mRows.resize(maxRequests);
    std::fill_n(writable(mView.slotIds), numEntries(), -1);
    std::fill_n(writable(mView.hostLevels), numEntries(), -1);
    updatePools();
}

HostSourceTable::~HostSourceTable()
{
    KVCM2_POISON_ON_EXCEPT([this]() { waitForReaders(); });
}

size_t HostSourceTable::numEntries() const noexcept
{
    return static_cast<size_t>(mView.maxRequests) * mView.maxBeams * mView.numLifeCycles * mView.maxPages;
}

int HostSourceTable::rowForEntry(size_t index) const noexcept
{
    return static_cast<int>(index / (static_cast<size_t>(mView.maxBeams) * mView.numLifeCycles * mView.maxPages));
}

int HostSourceTable::requestSlot(KvCache const& cache) const
{
    auto const it = mRequestRows.find(&cache);
    return it == mRequestRows.end() ? -1 : it->second;
}

HostSourceRow HostSourceTable::requestRef(KvCache const& cache) const
{
    int const row = requestSlot(cache);
    return {row, row < 0 ? 0 : mView.generations[row]};
}

void HostSourceTable::checkCapacity(KvCache const& cache, int capacity) const
{
    if (static_cast<int64_t>(capacity) > static_cast<int64_t>(mView.maxPages) * mView.tokensPerBlock
        || cache.beamWidth().value() > mView.maxBeams)
        throw std::out_of_range("Request exceeds the manager's host source limits");
}

void HostSourceTable::bindRequest(KvCache& cache, int row)
{
    if (row == -1)
    {
        removeRequest(cache);
        return;
    }
    if (row < 0 || row >= mView.maxRequests || !cache.id || cache.isClosed())
        throw std::invalid_argument("Bind a live request with an ID to a valid IndexMapper row");
    checkCapacity(cache, cache.capacity());
    if (requestSlot(cache) == row)
        return;
    if (mRows[row].cache)
        throw LogicError("Host source row is still bound to another request");
    waitForRow(row);
    if (mView.generations[row] == std::numeric_limits<uint64_t>::max())
        throw LogicError("Host source generation exhausted");
    removeRequest(cache);
    ++writable(mView.generations)[row];
    mRows[row].cache = &cache;
    mRequestRows.emplace(&cache, row);
    writable(mView.requestIds)[row] = *cache.id;
    writable(mView.requestValid)[row] = 1;
    rebuildRow(row);
}

void HostSourceTable::setRequestId(KvCache& cache, std::optional<RequestIdType> id)
{
    if (cache.id == id)
        return;
    int const row = requestSlot(cache);
    if (!id)
        removeRequest(cache);
    else if (row >= 0)
    {
        waitForRow(row);
        if (mView.generations[row] == std::numeric_limits<uint64_t>::max())
            throw LogicError("Host source generation exhausted");
        ++writable(mView.generations)[row];
        writable(mView.requestIds)[row] = *id;
    }
    cache.id = id;
}

void HostSourceTable::clearEntry(size_t index)
{
    writable(mView.slotIds)[index] = -1;
    writable(mView.hostLevels)[index] = -1;
    writable(mView.completedTokens)[index] = 0;
    mRows[rowForEntry(index)].published.erase(index);
}

void HostSourceTable::eraseEntry(size_t index)
{
    auto const source = mSources.find(index);
    if (source == mSources.end())
        return;
    auto it = mPageEntries.find(source->second.key);
    it->second.erase(index);
    if (it->second.empty())
        mPageEntries.erase(it);
    mSources.erase(source);
    clearEntry(index);
    auto& state = mRows[rowForEntry(index)];
    state.entries.erase(index);
    state.pending.erase(index);
}

void HostSourceTable::eraseEntries(int row)
{
    auto& entries = mRows[row].entries;
    while (!entries.empty())
        eraseEntry(*entries.begin());
}

void HostSourceTable::removeRequest(KvCache const& cache)
{
    int const row = requestSlot(cache);
    if (row < 0)
        return;
    waitForRow(row);
    eraseEntries(row);
    mRows[row].cache = nullptr;
    mRequestRows.erase(&cache);
    writable(mView.requestValid)[row] = 0;
    writable(mView.requestIds)[row] = 0;
    mDirtyRows.erase(row);
    mDirtyRanges.erase(row);
}

void HostSourceTable::waitForRow(int row)
{
    auto& state = mRows[row];
    if (state.openReaders != 0)
        throw LogicError("Close host source readers before changing their request rows");
    for (auto const& event : state.readEvents)
        event.synchronize();
    state.readEvents.clear();
}

void HostSourceTable::waitForReaders()
{
    if (mOpenReaders != 0)
        throw LogicError("Close host source readers before changing pools or shutting down");
    for (int row = 0; row < mView.maxRequests; ++row)
        waitForRow(row);
    for (auto const& event : mEmptyReadEvents)
        event.synchronize();
    mEmptyReadEvents.clear();
}

void HostSourceTable::beginUpdate(
    KvCache* cache, std::optional<BlockOrdinal> ordinal, std::optional<std::pair<int, int>> range)
{
    if (!cache)
    {
        waitForReaders();
        mPoolsDirty = true;
    }
    else
    {
        int const row = requestSlot(*cache);
        if (row >= 0)
            waitForRow(row);
        std::unordered_set<size_t> affected;
        if (ordinal)
        {
            // An unbound request may still back up a prefix shared by bound requests.
            if (ordinal->value() >= 0 && *ordinal < cache->numBlocks())
            {
                for (auto const& beam : cache->blocks()[*ordinal].pages)
                    for (auto const& entry : beam)
                    {
                        auto const& page = blockPageGetPage(entry);
                        if (!page)
                            continue;
                        auto const users = mPageEntries.find(page.get());
                        if (users != mPageEntries.end())
                            affected.insert(users->second.begin(), users->second.end());
                    }
            }
        }
        else if (row >= 0)
        {
            // Request structure changes do not modify another request's immutable prefix.
            if (range)
            {
                for (int page = range->first; page < range->second; ++page)
                    for (int beam = 0; beam < mView.maxBeams; ++beam)
                        for (int lc = 0; lc < mView.numLifeCycles; ++lc)
                        {
                            auto const index = mView.pageIndex(row, beam, lc, page);
                            if (mSources.count(index))
                                affected.insert(index);
                        }
            }
            else
                affected = mRows[row].entries;
        }
        // Validate all shared users before changing any metadata, so a rejected update leaves it intact.
        for (auto const index : affected)
            waitForRow(rowForEntry(index));
        mDirtyEntries.insert(affected.begin(), affected.end());
        for (auto const index : affected)
            clearEntry(index);
        if (row >= 0 && range)
        {
            auto [it, inserted] = mDirtyRanges.emplace(row, *range);
            if (!inserted)
            {
                it->second.first = std::min(it->second.first, range->first);
                it->second.second = std::max(it->second.second, range->second);
            }
        }
        else if (row >= 0 && !ordinal)
            mDirtyRows.insert(row);
    }
    ++mUpdateDepth;
}

void HostSourceTable::endUpdate()
{
    TLLM_CHECK(mUpdateDepth > 0);
    if (--mUpdateDepth != 0)
        return;
    if (mPoolsDirty)
    {
        updatePools();
        mPoolsDirty = false;
    }
    for (auto const row : mDirtyRows)
        rebuildRow(row);
    for (auto const& [row, range] : mDirtyRanges)
        if (!mDirtyRows.count(row))
            rebuildRange(row, range.first, range.second);
    mDirtyRows.clear();
    mDirtyRanges.clear();
    for (auto const index : mDirtyEntries)
    {
        if (mSources.count(index))
            publish(index);
    }
    mDirtyEntries.clear();
}

void HostSourceTable::refreshForTest()
{
    if (mUpdateDepth != 0)
        throw LogicError("Cannot refresh during a lifecycle update");
    waitForReaders();
    // Test-only event polling. Production polls just the pending entries in the acquired batch.
    for (auto& row : mRows)
    {
        auto const pending = row.pending;
        for (auto const index : pending)
            publish(index);
    }
}

void HostSourceTable::updatePools()
{
    std::fill_n(writable(mView.poolMetadata), static_cast<size_t>(mView.numLevels) * mView.numLifeCycles * 4, 0);
    for (CacheLevel level{1}; level < mStorage.numCacheLevels(); ++level)
    {
        if (mStorage.cacheTier(level) != CacheTier::HOST_MEM)
            continue;
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
}

void HostSourceTable::rebuildRow(int row)
{
    eraseEntries(row);
    rebuildRange(row, 0, mRows[row].cache ? mRows[row].cache->numBlocks().value() : 0);
}

void HostSourceTable::rebuildRange(int row, int begin, int end)
{
    // Remove only the changed ordinals, including any pages removed by a capacity decrease.
    for (int page = begin; page < end; ++page)
        for (int beam = 0; beam < mView.maxBeams; ++beam)
            for (int lc = 0; lc < mView.numLifeCycles; ++lc)
                eraseEntry(mView.pageIndex(row, beam, lc, page));
    auto const* cache = mRows[row].cache;
    if (!cache)
        return;
    checkCapacity(*cache, cache->capacity());
    for (BlockOrdinal ordinal{begin}; ordinal < cache->numBlocks() && ordinal.value() < end; ++ordinal)
    {
        auto const& beams = cache->blocks()[ordinal].pages;
        for (BeamIndex beam{0}; beam < beams.size(); ++beam)
        {
            for (LifeCycleId lc{0}; lc < mStorage.numLifeCycles(); ++lc)
            {
                auto const& page = blockPageGetPage(beams[beam][lc]);
                if (!page || !std::holds_alternative<AttnLifeCycle>(mStorage.getLifeCycle(lc)))
                    continue;
                int coverage = std::clamp(
                    cache->historyLength() - ordinal.value() * mView.tokensPerBlock, 0, mView.tokensPerBlock);
                if (page->isCommitted())
                    coverage = std::min(coverage, static_cast<CommittedPage const&>(*page).numTokensInBlock);
                auto const index = mView.pageIndex(row, beam.value(), lc.value(), ordinal.value());
                mSources.emplace(index, Source{page.get(), page, coverage});
                mPageEntries[page.get()].insert(index);
                mRows[row].entries.insert(index);
                publish(index);
            }
        }
    }
}

void HostSourceTable::publish(size_t index)
{
    auto const& source = mSources.at(index);
    auto const page = source.page.lock();
    auto& state = mRows[rowForEntry(index)];
    state.pending.erase(index);
    clearEntry(index);
    if (!page || !page->hostCopy() || source.coverage == 0)
        return;
    auto const& copy = page->hostCopy();
    if (!copy->ready())
    {
        if (copy->validTokens() > 0)
            state.pending.insert(index);
        return;
    }
    writable(mView.slotIds)[index] = copy->slotId().value();
    writable(mView.hostLevels)[index] = copy->level().value();
    writable(mView.completedTokens)[index] = std::min(source.coverage, copy->validTokens());
    state.published.insert(index);
}

std::unique_ptr<HostSourceRead> HostSourceTable::acquire(
    std::shared_ptr<KvCacheManager> owner, std::vector<HostSourceRow> const& rows, CUstream stream)
{
    checkOutsideCapture(stream);
    if (mUpdateDepth != 0)
        throw LogicError("Cannot read during a lifecycle update");
    std::unordered_set<int> seenRows;
    for (auto const& [row, generation] : rows)
    {
        if (row < 0 || row >= mView.maxRequests || !mRows[row].cache || mView.generations[row] != generation
            || !seenRows.insert(row).second)
            throw std::invalid_argument("Host source batch contains a stale, invalid, or duplicate row");
    }
    auto read = std::unique_ptr<HostSourceRead>(new HostSourceRead(std::move(owner), stream));
    ++mOpenReaders;
    std::unordered_set<HostPageCopy*> seen;
    for (auto const& ref : rows)
    {
        auto& state = mRows[ref.first];
        if (state.openReaders == 0)
        {
            waitForRow(ref.first);
            auto const pending = state.pending;
            for (auto const index : pending)
                publish(index);
        }
        read->mRequests.push_back(state.cache->shared_from_this());
        read->mRows.push_back(ref);
        ++state.openReaders;
        for (auto const index : state.published)
        {
            auto const page = mSources.at(index).page.lock();
            TLLM_CHECK(page && page->hostCopy());
            auto const& copy = page->hostCopy();
            if (seen.insert(copy.get()).second)
                read->mPages.push_back(std::make_unique<HostPageRead>(read->mOwner, copy, stream));
        }
    }
    return read;
}

void HostSourceTable::finishRead(std::vector<HostSourceRow> const& rows, CachedCudaEvent event)
{
    TLLM_CHECK(mOpenReaders > 0);
    if (rows.empty())
    {
        // Keep only unfinished uses; empty batches still borrow pool metadata and table storage.
        mEmptyReadEvents.erase(std::remove_if(mEmptyReadEvents.begin(), mEmptyReadEvents.end(),
                                   [](auto const& prior) { return prior.queryComplete(); }),
            mEmptyReadEvents.end());
        mEmptyReadEvents.push_back(event);
    }
    for (auto const& [row, generation] : rows)
    {
        auto& state = mRows[row];
        TLLM_CHECK(state.openReaders > 0 && mView.generations[row] == generation);
        state.readEvents.push_back(event);
        --state.openReaders;
    }
    --mOpenReaders;
}

HostSourceRead::HostSourceRead(std::shared_ptr<KvCacheManager> owner, CUstream stream)
    : mOwner(std::move(owner))
    , mStream(stream)
{
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
    owner->mHostSources->finishRead(mRows, CachedCudaEvent(reinterpret_cast<CudaStream>(mStream)));
    mRows.clear();
    mRequests.clear();
    mOwner.reset();
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
