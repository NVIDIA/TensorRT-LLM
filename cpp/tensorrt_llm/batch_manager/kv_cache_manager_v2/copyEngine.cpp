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

#include "kv_cache_manager_v2/copyEngine.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/math.h"

// Reuse existing copy implementations (no Python round-trip).
#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"

#include <cassert>
#include <stdexcept>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Helpers: convert Address variant to typed pointers for copy functions.
// ---------------------------------------------------------------------------

static MemAddress asMemAddress(Address const& a)
{
    return std::get<MemAddress>(a);
}

static DiskAddress asDiskAddress(Address const& a)
{
    return std::get<DiskAddress>(a);
}

// ---------------------------------------------------------------------------
// Single-hop copy dispatchers — map to existing copy functions.
// ---------------------------------------------------------------------------

static void copyGpuToGpu(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    std::vector<Task<MemAddress, MemAddress>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({asMemAddress(task.dst), asMemAddress(task.src)});
    cuCheck(copyDeviceToDevice(t, static_cast<ssize_t>(numBytes), stream));
}

static void copyHostToHost(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    std::vector<Task<MemAddress, MemAddress>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({asMemAddress(task.dst), asMemAddress(task.src)});
    cuCheck(copyHostToHost(t, static_cast<ssize_t>(numBytes), stream));
}

static void copyDiskToDisk(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    std::vector<Task<DiskAddress, DiskAddress>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({asDiskAddress(task.dst), asDiskAddress(task.src)});
    cuCheck(copyDiskToDisk(t, static_cast<ssize_t>(numBytes), stream));
}

static void copyGpuToHost(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    std::vector<Task<MemAddress, MemAddress>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({asMemAddress(task.dst), asMemAddress(task.src)});
    cuCheck(copyDeviceToHost(t, static_cast<ssize_t>(numBytes), stream));
}

static void copyHostToGpu(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    std::vector<Task<MemAddress, MemAddress>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({asMemAddress(task.dst), asMemAddress(task.src)});
    cuCheck(copyHostToDevice(t, static_cast<ssize_t>(numBytes), stream));
}

static void copyDiskToHost(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    std::vector<Task<MemAddress, DiskAddress>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({asMemAddress(task.dst), asDiskAddress(task.src)});
    cuCheck(copyDiskToHost(t, static_cast<ssize_t>(numBytes), stream));
}

static void copyHostToDisk(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    std::vector<Task<DiskAddress, MemAddress>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({asDiskAddress(task.dst), asMemAddress(task.src)});
    cuCheck(copyHostToDisk(t, static_cast<ssize_t>(numBytes), stream));
}

// ---------------------------------------------------------------------------
// StagingBuffer
// ---------------------------------------------------------------------------

StagingBuffer::StagingBuffer(StagingBufferManager& manager, size_t minSize, size_t maxSize, CUstream stream)
    : mManager(manager)
    , mStream(stream)
{
    if (minSize > manager.totalSize())
        throw std::invalid_argument("StagingBuffer: minSize exceeds total staging buffer size");

    std::unique_lock<std::mutex> lock(mManager.mMutex);

    // Compute how many grains to use.
    size_t available = mManager.suggestNextMaxGrains() * kGranularity;
    size_t actualSize = std::min(maxSize, available);
    actualSize = std::max(actualSize, minSize);
    mNumGrains = divUp(actualSize, kGranularity);
    mSize = mNumGrains * kGranularity;
    mStartGrain = mManager.mNext;
    mManager.mNext += mNumGrains;
    if (mManager.mNext >= mManager.numGrains())
        mManager.mNext = 0;

    mAddress = mManager.baseAddress() + mStartGrain * kGranularity;
    lock.unlock();

    // Lock grains and drain their ready events before we can use the buffer.
    for (size_t i = 0; i < mNumGrains; ++i)
    {
        GrainMetadata& g = mManager.mGrains[mStartGrain + i];
        g.mutex.lock();
        // Wait for the previous user's event before we start writing.
        if (!g.readyEvent.isClosed())
        {
            g.readyEvent.waitInStream(reinterpret_cast<CudaStream>(mStream));
            g.readyEvent.close();
        }
    }
}

StagingBuffer::~StagingBuffer()
{
    // One shared completion event for all grains (all on the same stream → same completion point).
    CachedCudaEvent finishEvent(reinterpret_cast<CudaStream>(mStream));
    for (int i = static_cast<int>(mNumGrains) - 1; i >= 0; --i)
    {
        GrainMetadata& g = mManager.mGrains[mStartGrain + static_cast<size_t>(i)];
        g.readyEvent = finishEvent;
        g.mutex.unlock();
    }
}

// ---------------------------------------------------------------------------
// StagingBufferManager
// ---------------------------------------------------------------------------

StagingBufferManager::StagingBufferManager(size_t size)
    : mBuffer(size)
    , mGrains(size / kGranularity)
{
    assert(size % kGranularity == 0);
}

StagingBuffer StagingBufferManager::acquire(size_t minSize, size_t maxSize, CUstream stream)
{
    return StagingBuffer(*this, minSize, maxSize, stream);
}

// ---------------------------------------------------------------------------
// CopyEngine
// ---------------------------------------------------------------------------

StagingBufferManager& CopyEngine::getStagingManager()
{
    if (!mStagingManager)
        mStagingManager = std::make_unique<StagingBufferManager>(64u << 20u); // 64 MB
    return *mStagingManager;
}

// Two-hop transfer via host staging buffer (e.g., GPU→Disk or Disk→GPU).
// firstHop:  copies src → staging
// secondHop: copies staging → dst
using SingleHopFn = void (*)(std::vector<CopyTask> const&, size_t, CUstream);

static void twoHopTransfer(StagingBufferManager& manager, SingleHopFn firstHop, SingleHopFn secondHop, size_t numBytes,
    std::vector<CopyTask> const& tasks, CUstream stream)
{
    size_t remaining = tasks.size();
    size_t offset = 0;

    while (remaining > 0)
    {
        StagingBuffer buf = manager.acquire(numBytes, numBytes * remaining, stream);
        MemAddress addr = buf.address();
        size_t n = buf.size() / numBytes;
        assert(n > 0 && n <= remaining);

        // First hop: src → staging
        {
            std::vector<CopyTask> hop1;
            hop1.reserve(n);
            for (size_t i = 0; i < n; ++i)
                hop1.push_back({Address{addr + numBytes * i}, tasks[offset + i].src});
            firstHop(hop1, numBytes, buf.stream());
        }

        // Second hop: staging → dst
        {
            std::vector<CopyTask> hop2;
            hop2.reserve(n);
            for (size_t i = 0; i < n; ++i)
                hop2.push_back({tasks[offset + i].dst, Address{addr + numBytes * i}});
            secondHop(hop2, numBytes, buf.stream());
        }

        offset += n;
        remaining -= n;
    }
}

void CopyEngine::transfer(
    CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> tasks, CUstream stream)
{
    // Dispatch table: [dstTier][srcTier]
    // CacheTier values: GPU_MEM=0, HOST_MEM=1, DISK=2
    enum
    {
        GPU = 0,
        HOST = 1,
        DISK = 2
    };

    int dst = static_cast<int>(dstTier);
    int src = static_cast<int>(srcTier);

    if (dst == GPU && src == GPU)
    {
        copyGpuToGpu(tasks, numBytes, stream);
    }
    else if (dst == GPU && src == HOST)
    {
        copyHostToGpu(tasks, numBytes, stream);
    }
    else if (dst == GPU && src == DISK)
    {
        // Two-hop: Disk→Host→GPU
        twoHopTransfer(getStagingManager(), copyDiskToHost, copyHostToGpu, numBytes, tasks, stream);
    }
    else if (dst == HOST && src == GPU)
    {
        copyGpuToHost(tasks, numBytes, stream);
    }
    else if (dst == HOST && src == HOST)
    {
        copyHostToHost(tasks, numBytes, stream);
    }
    else if (dst == HOST && src == DISK)
    {
        copyDiskToHost(tasks, numBytes, stream);
    }
    else if (dst == DISK && src == GPU)
    {
        // Two-hop: GPU→Host→Disk
        twoHopTransfer(getStagingManager(), copyGpuToHost, copyHostToDisk, numBytes, tasks, stream);
    }
    else if (dst == DISK && src == HOST)
    {
        copyHostToDisk(tasks, numBytes, stream);
    }
    else if (dst == DISK && src == DISK)
    {
        copyDiskToDisk(tasks, numBytes, stream);
    }
    else
    {
        throw std::invalid_argument("CopyEngine::transfer: unsupported tier combination");
    }
}

// ---------------------------------------------------------------------------
// Module-level singleton
// ---------------------------------------------------------------------------

CopyEngine& globalCopyEngine()
{
    static CopyEngine engine;
    return engine;
}

void batchedCopy(CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> tasks, CUstream stream)
{
    globalCopyEngine().transfer(dstTier, srcTier, numBytes, std::move(tasks), stream);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
