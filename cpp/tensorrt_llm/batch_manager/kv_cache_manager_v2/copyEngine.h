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

#include "tensorrt_llm/common/assert.h"
#include <cuda.h>
#include <memory>
#include <mutex>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// CopyTask — source and destination address pair for a bulk copy.
// ---------------------------------------------------------------------------
struct CopyTask
{
    Address dst;
    Address src;
};

// ---------------------------------------------------------------------------
// GrainMetadata — per-grain synchronization state inside StagingBufferManager.
// ---------------------------------------------------------------------------
struct GrainMetadata
{
    std::mutex mutex;
    CachedCudaEvent readyEvent; // event protecting this grain

    GrainMetadata()
        : readyEvent(CachedCudaEvent::makeNull())
    {
    }

    GrainMetadata(GrainMetadata const&) = delete;
    GrainMetadata& operator=(GrainMetadata const&) = delete;
    GrainMetadata(GrainMetadata&&) = delete;
    GrainMetadata& operator=(GrainMetadata&&) = delete;
};

class StagingBufferManager;

// ---------------------------------------------------------------------------
// StagingBuffer — RAII handle to a slice of the StagingBufferManager's buffer.
//
// On construction, acquires grain locks and waits for readyEvents.
// On destruction, records a new event and releases grain locks.
// ---------------------------------------------------------------------------
class StagingBuffer
{
public:
    static constexpr size_t kGranularity = 1u << 20; // 1 MB grains

    StagingBuffer(StagingBufferManager& manager, size_t minSize, size_t maxSize, CUstream stream);
    ~StagingBuffer();

    StagingBuffer(StagingBuffer const&) = delete;
    StagingBuffer& operator=(StagingBuffer const&) = delete;
    StagingBuffer(StagingBuffer&&) = delete;
    StagingBuffer& operator=(StagingBuffer&&) = delete;

    MemAddress address() const noexcept
    {
        return mAddress;
    }

    size_t size() const noexcept
    {
        return mSize;
    }

    CUstream stream() const noexcept
    {
        return mStream;
    }

private:
    StagingBufferManager& mManager;
    size_t mStartGrain{0};
    size_t mNumGrains{0};
    size_t mSize{0};
    MemAddress mAddress{0};
    CUstream mStream;
};

// ---------------------------------------------------------------------------
// StagingBufferManager — ring-buffer allocator over a CUDA-registered HostMem.
// Used for two-hop GPU↔Disk transfers.
// ---------------------------------------------------------------------------
class StagingBufferManager
{
public:
    static constexpr size_t kGranularity = StagingBuffer::kGranularity; // 1 MB

    explicit StagingBufferManager(size_t size);

    // Not movable: GrainMetadata has mutexes.
    StagingBufferManager(StagingBufferManager const&) = delete;
    StagingBufferManager& operator=(StagingBufferManager const&) = delete;
    StagingBufferManager(StagingBufferManager&&) = delete;
    StagingBufferManager& operator=(StagingBufferManager&&) = delete;

    // Acquire a staging slice. Thread-safe.
    // minSize: minimum required bytes. maxSize: best-effort upper bound.
    // Returns an RAII StagingBuffer that holds grain locks until destroyed.
    StagingBuffer acquire(size_t minSize, size_t maxSize, CUstream stream);

    size_t totalSize() const noexcept
    {
        TLLM_CHECK_DEBUG_WITH_INFO(
            mGrains.size() * kGranularity == mBuffer.size(), "grain count * granularity must equal buffer size");
        return mBuffer.size();
    }

    size_t numGrains() const noexcept
    {
        return mGrains.size();
    }

    MemAddress baseAddress() const noexcept
    {
        return mBuffer.address();
    }

private:
    friend class StagingBuffer;

    // Caller must hold mMutex.
    size_t suggestNextMaxGrains() const noexcept
    {
        return numGrains() - mNext;
    }

    std::mutex mMutex;
    HostMem mBuffer;
    std::vector<GrainMetadata> mGrains;
    size_t mNext{0};
};

// ---------------------------------------------------------------------------
// CopyEngine — dispatches bulk transfers between cache tiers.
//
// Single-hop pairs call the appropriate copy function from kvCacheManagerV2Utils.
// Two-hop pairs (GPU↔Disk) route through a lazily-allocated StagingBuffer.
// ---------------------------------------------------------------------------
class CopyEngine
{
public:
    CopyEngine() = default;
    ~CopyEngine() = default;

    CopyEngine(CopyEngine const&) = delete;
    CopyEngine& operator=(CopyEngine const&) = delete;

    // Transfer num_bytes per task. tasks must all share the same (dstTier, srcTier).
    // stream: the CUDA stream on which GPU ops are enqueued.
    void transfer(
        CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> const& tasks, CUstream stream);

    void close() noexcept
    {
        mStagingManager.reset();
    }

private:
    StagingBufferManager& getStagingManager();

    std::unique_ptr<StagingBufferManager> mStagingManager;
};

// ---------------------------------------------------------------------------
// Module-level singleton (mirrors Python's _copy_engine global).
// ---------------------------------------------------------------------------
CopyEngine& globalCopyEngine();

// Convenience wrapper used by higher layers.
void batchedCopy(CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> tasks, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
