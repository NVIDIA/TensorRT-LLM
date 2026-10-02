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

#include "kv_cache_manager_v2/batch.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/utils/funcGuard.h"
#include "kv_cache_manager_v2/utils/optionalGilRelease.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

void checkOutsideCapture(CudaStream stream)
{
    CUstreamCaptureStatus status;
    cuCheck(cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &status));
    if (status != CU_STREAM_CAPTURE_STATUS_NONE)
    {
        throw LogicError("Batch publication and reader fences must run outside CUDA graph capture");
    }
}

} // namespace

Batch::Batch(std::shared_ptr<KvCacheManager> manager, int maxRows, int maxBlocks, int maxBeamWidth)
    : mManager(std::move(manager))
    , mMaxRows(maxRows)
    , mMaxBlocks(maxBlocks)
    , mMaxBeamWidth(maxBeamWidth)
{
    KVCM2_API_GUARD();
    if (!mManager || maxRows <= 0 || maxBlocks <= 0 || maxBeamWidth != 1)
    {
        throw std::invalid_argument("Batch requires a manager, positive dimensions and beam width 1");
    }
    auto const apiLock = mManager->lockExclusive();
    mNumLayerGroups = mManager->lifeCycles().size().value();
    size_t const rows = static_cast<size_t>(mNumLayerGroups) * mMaxRows * mMaxBeamWidth;
    size_t const columns = static_cast<size_t>(mMaxBlocks) + 1;
    if (rows == 0 || rows > std::numeric_limits<size_t>::max() / sizeof(int32_t) / columns)
    {
        throw std::invalid_argument("Batch metadata dimensions overflow");
    }
    mTableElements = rows * mMaxBlocks;
    mTotalBytes = rows * columns * sizeof(int32_t);
    mRows.resize(mMaxRows, nullptr);
    mDirty.resize(mMaxRows, 1);
    cuCheck(cuCtxGetDevice(&mDeviceId));
    CUdeviceptr ptr = 0;
    cuCheck(cuMemAlloc(&ptr, mTotalBytes));
    mDeviceMemory.reset(reinterpret_cast<std::byte*>(ptr));
    // Two generations cover the usual publication/consumer overlap without
    // allocating or registering pinned memory in the publication path.
    mUploads.push_back({std::make_unique<HostMem>(mTotalBytes)});
    mUploads.push_back({std::make_unique<HostMem>(mTotalBytes)});
}

Batch::~Batch()
{
    KVCM2_POISON_ON_EXCEPT(
        [this]()
        {
            OptionalGilRelease const gilRelease;
            close();
            auto const apiLock = mManager->lockExclusive();
            mReady.synchronize();
            for (auto const& reader : mReaders)
            {
                reader.synchronize();
            }
            for (auto const& upload : mUploads)
            {
                upload.completion.synchronize();
            }
        });
}

void Batch::checkOpen() const
{
    if (mClosed)
    {
        throw LogicError("Batch is closed");
    }
}

void Batch::checkPublished() const
{
    checkOpen();
    if (!mPublished)
    {
        throw LogicError("Batch metadata changed; publish before consumption");
    }
}

void Batch::markDirty(int row) noexcept
{
    mDirty[row] = 1;
    mPublished = false;
}

int Batch::add(KvCache& cache, std::optional<int> row)
{
    KVCM2_API_GUARD();
    auto const apiLock = mManager->lockExclusive();
    checkOpen();
    if (&cache.manager() != mManager.get() || cache.isClosed())
    {
        throw LogicError("Batch members must be live requests from the same manager");
    }
    if (cache.mPageStorageBatch != nullptr)
    {
        if (cache.mPageStorageBatch == this && (!row || row == cache.mPageStorageRow))
        {
            return *cache.mPageStorageRow;
        }
        throw LogicError("A request can belong to only one batch and row");
    }
    int const index = row.value_or(static_cast<int>(std::find(mRows.begin(), mRows.end(), nullptr) - mRows.begin()));
    if (index < 0 || index >= mMaxRows || mRows[index] != nullptr)
    {
        throw std::invalid_argument("Batch row is unavailable");
    }
    if (cache.numBlocks().value() > mMaxBlocks || cache.beamWidth().value() > mMaxBeamWidth)
    {
        throw std::invalid_argument("Request exceeds batch dimensions");
    }
    mRows[index] = &cache;
    cache.mPageStorageBatch = this;
    cache.mPageStorageRow = index;
    cache.onPageStorageChanged();
    return index;
}

void Batch::remove(KvCache& cache)
{
    auto const apiLock = mManager->lockExclusive();
    if (cache.mPageStorageBatch == nullptr)
    {
        return;
    }
    if (cache.mPageStorageBatch != this)
    {
        throw LogicError("Request belongs to another batch");
    }
    int const row = *cache.mPageStorageRow;
    mRows[row] = nullptr;
    markDirty(row);
    cache.mPageStorageBatch = nullptr;
    cache.mPageStorageRow.reset();
    cache.onPageStorageChanged();
}

void Batch::close()
{
    auto const apiLock = mManager->lockExclusive();
    if (mClosed || Poison::poisoned())
    {
        return;
    }
    for (auto* cache : mRows)
    {
        if (cache != nullptr)
        {
            remove(*cache);
        }
    }
    mClosed = true;
    mPublished = false;
}

size_t Batch::rowOffset(int group, int row) const noexcept
{
    return (static_cast<size_t>(group) * mMaxRows + row) * mMaxBeamWidth * mMaxBlocks;
}

size_t Batch::countOffset(int group, int row) const noexcept
{
    return mTableElements + (static_cast<size_t>(group) * mMaxRows + row) * mMaxBeamWidth;
}

std::vector<int> Batch::dirtyRows() const
{
    auto const apiLock = mManager->lockShared();
    checkOpen();
    std::vector<int> rows;
    for (int row = 0; row < mMaxRows; ++row)
    {
        if (mDirty[row])
        {
            rows.push_back(row);
        }
    }
    return rows;
}

std::vector<int> Batch::publish(CudaStream stream)
{
    KVCM2_API_GUARD();
    auto const apiLock = mManager->lockExclusive();
    checkOpen();
    auto const cudaStream = reinterpret_cast<CUstream>(stream);
    checkOutsideCapture(stream);
    // Owner release can unblock history without changing a request's watermark or metadata version.
    // Retry before collecting dirty rows: one shared-page move can invalidate several rows.
    for (auto* cache : mRows)
    {
        if (cache != nullptr && cache->isActive() && cache->mIsDecoding && cache->mHasDeferredSparseOffload)
        {
            cache->_offloadSparseHistory({0, 0}, cache->mHistoryLength);
        }
    }
    auto rows = dirtyRows();
    if (rows.empty())
    {
        mReady.waitInStream(stream);
        return rows;
    }

    struct RequestSnapshot
    {
        KvCache* cache;
        uint64_t version;
        std::vector<PageStorageSnapshot> groups;
    };

    std::vector<RequestSnapshot> snapshots;
    snapshots.reserve(rows.size());
    for (int row : rows)
    {
        auto* cache = mRows[row];
        RequestSnapshot request{cache, cache ? cache->pageStorageVersion() : 0, {}};
        if (cache != nullptr)
        {
            if (cache->numBlocks().value() > mMaxBlocks || cache->beamWidth().value() != mMaxBeamWidth)
            {
                throw std::invalid_argument("Request exceeds batch dimensions");
            }
            request.groups.reserve(mNumLayerGroups);
            for (int group = 0; group < mNumLayerGroups; ++group)
            {
                request.groups.push_back(cache->getPageStorageSnapshot(LayerGroupId{group}));
            }
        }
        snapshots.push_back(std::move(request));
    }

    // A busy staging buffer cannot be overwritten by the CPU just by queueing a
    // stream wait. Reuse only completed buffers; retain each in-flight generation.
    std::unique_ptr<HostMem> staging;
    for (auto it = mUploads.begin(); it != mUploads.end();)
    {
        if (it->completion.queryComplete())
        {
            staging = std::move(it->staging);
            mUploads.erase(it);
            break;
        }
        else
        {
            ++it;
        }
    }
    if (!staging)
    {
        staging = std::make_unique<HostMem>(mTotalBytes);
    }
    auto* host = reinterpret_cast<int32_t*>(staging->address());
    for (size_t i = 0; i < rows.size(); ++i)
    {
        for (int group = 0; group < mNumLayerGroups; ++group)
        {
            int32_t* indices = host + rowOffset(group, rows[i]);
            std::fill_n(indices, mMaxBlocks, kBadPageIndex.value());
            int32_t& count = host[countOffset(group, rows[i])];
            count = 0;
            if (snapshots[i].cache != nullptr)
            {
                auto const& snapshot = snapshots[i].groups[group];
                std::copy(snapshot.basePageIndices().begin(), snapshot.basePageIndices().end(), indices);
                count = snapshot.eligibleHistoryBlocks();
            }
        }
    }
    mUploads.push_back({std::move(staging)});
    auto& upload = mUploads.back();
    auto fence = FuncGuard(
        [&]()
        {
            mReady = CachedCudaEvent(stream);
            upload.completion = mReady;
        });
    mReady.waitInStream(stream);
    for (auto const& reader : mReaders)
    {
        reader.waitInStream(stream);
    }
    mReaders.clear();
    for (auto const& request : snapshots)
    {
        for (auto const& snapshot : request.groups)
        {
            snapshot.waitReady(stream);
        }
    }
    auto const device = reinterpret_cast<CUdeviceptr>(mDeviceMemory.get());
    for (int row : rows)
    {
        for (int group = 0; group < mNumLayerGroups; ++group)
        {
            size_t const offset = rowOffset(group, row);
            cuCheck(cuMemcpyHtoDAsync(
                device + offset * sizeof(int32_t), host + offset, mMaxBlocks * sizeof(int32_t), cudaStream));
            size_t const count = countOffset(group, row);
            cuCheck(cuMemcpyHtoDAsync(device + count * sizeof(int32_t), host + count, sizeof(int32_t), cudaStream));
        }
    }
    fence.run();
    for (size_t i = 0; i < rows.size(); ++i)
    {
        auto const& request = snapshots[i];
        if (request.cache != nullptr)
        {
            TLLM_CHECK(request.cache->acknowledgePageStorage(request.version));
        }
        mDirty[rows[i]] = 0;
    }
    mPublished = true;
    return rows;
}

void Batch::waitReady(CudaStream stream) const
{
    KVCM2_API_GUARD();
    auto const apiLock = mManager->lockShared();
    checkPublished();
    checkOutsideCapture(stream);
    mReady.waitInStream(stream);
}

void Batch::recordRead(CudaStream stream)
{
    KVCM2_API_GUARD();
    auto const apiLock = mManager->lockExclusive();
    checkOpen();
    checkOutsideCapture(stream);
    std::erase_if(mReaders, [](auto const& event) { return event.queryComplete(); });
    CachedCudaEvent completion(stream);
    mReaders.push_back(completion);
    for (auto* cache : mRows)
    {
        if (cache != nullptr && cache->isActive())
        {
            completion.waitInStream(reinterpret_cast<CudaStream>(cache->cudaStream()));
        }
    }
}

std::vector<std::optional<bool>> Batch::resize(std::vector<std::optional<int>> const& capacities,
    std::vector<std::optional<int>> const& historyLengths, CudaStream stream)
{
    KVCM2_API_GUARD();
    auto const apiLock = mManager->lockExclusive();
    checkOpen();
    checkOutsideCapture(stream);
    if (capacities.size() != mRows.size() || historyLengths.size() != mRows.size())
    {
        throw std::invalid_argument("Batch resize arguments must be indexed by stable row");
    }
    for (int row = 0; row < mMaxRows; ++row)
    {
        if ((capacities[row]
                && (*capacities[row] < 0
                    || static_cast<int64_t>(*capacities[row])
                        > static_cast<int64_t>(mMaxBlocks) * mManager->tokensPerBlock()))
            || (historyLengths[row] && *historyLengths[row] < 0)
            || (mRows[row] == nullptr && (capacities[row] || historyLengths[row])))
        {
            throw std::invalid_argument("Invalid batch resize capacity, history, or empty row");
        }
    }
    std::vector<std::optional<bool>> results(mMaxRows);
    for (int row = 0; row < mMaxRows; ++row)
    {
        if (auto* cache = mRows[row])
        {
            results[row] = cache->resize(capacities[row], historyLengths[row]);
        }
    }
    publish(stream);
    return results;
}

MemAddress Batch::pageTableAddress(LayerGroupId group) const
{
    if (group.value() < 0 || group.value() >= mNumLayerGroups)
    {
        throw std::out_of_range("Invalid batch layer group");
    }
    return reinterpret_cast<MemAddress>(mDeviceMemory.get()) + rowOffset(group.value(), 0) * sizeof(int32_t);
}

MemAddress Batch::numBlocksAddress(LayerGroupId group) const
{
    if (group.value() < 0 || group.value() >= mNumLayerGroups)
    {
        throw std::out_of_range("Invalid batch layer group");
    }
    return reinterpret_cast<MemAddress>(mDeviceMemory.get()) + countOffset(group.value(), 0) * sizeof(int32_t);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
