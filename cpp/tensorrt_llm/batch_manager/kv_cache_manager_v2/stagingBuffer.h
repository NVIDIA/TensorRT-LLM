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
#include <cuda.h>
#include <list>
#include <memory>
#include <optional>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class StagingBufferManager;

enum class StagingBufferMemory
{
    kPinnedHost,
    kDevice,
};

struct DeviceDeleter
{
    void operator()(std::byte* ptr) const noexcept;
};

using CudaUniqPtr = std::unique_ptr<std::byte, DeviceDeleter>;

struct StagingBufferRange;

// ---------------------------------------------------------------------------
// StagingBuffer — RAII handle to a slice of the StagingBufferManager's buffer.
//
// On construction, reserves a size-granular, address-aligned byte range. A stream value makes reused-range
// waits asynchronous on that stream; nullopt synchronizes reused ranges before returning. The stream then identifies
// where asynchronous access to this lease is ordered. nullopt means no asynchronous access is outstanding.
// On destruction, a stream-owned range is retired with an event recorded on that stream; a nullopt-owned range is
// immediately reusable.
// ---------------------------------------------------------------------------
class StagingBuffer
{
public:
    StagingBuffer(StagingBufferManager& manager, size_t minSize, size_t maxSize, size_t sizeGranularity,
        size_t alignment, std::optional<CUstream> stream);
    ~StagingBuffer();

    StagingBuffer(StagingBuffer const&) = delete;
    StagingBuffer& operator=(StagingBuffer const&) = delete;
    StagingBuffer(StagingBuffer&&) = delete;
    StagingBuffer& operator=(StagingBuffer&&) = delete;

    [[nodiscard]] MemAddress address() const noexcept;

    [[nodiscard]] size_t size() const noexcept;

    [[nodiscard]] std::optional<CUstream> stream() const noexcept
    {
        return mStream;
    }

    // Transfer access ordering to stream. Switching between streams inserts an event dependency; switching from a
    // stream to nullopt synchronizes it. Switching from nullopt to a stream inserts no synchronization, so
    // the caller must finish synchronous access before the transition.
    void setStream(std::optional<CUstream> stream);

private:
    StagingBufferManager& mManager;
    StagingBufferRange* mRange{nullptr};
    std::optional<CUstream> mStream;
};

// ---------------------------------------------------------------------------
// StagingBufferManager — ring-buffer allocator over pinned host or device memory.
//
// acquire(..., stream) inserts waits for prior users on stream and returns without host synchronization.
// acquire(..., nullopt) synchronizes prior users before returning. The returned StagingBuffer retains the same
// optional stream and uses it to protect the range when the lease is retired.
// ---------------------------------------------------------------------------
class StagingBufferManager
{
public:
    StagingBufferManager(size_t size, StagingBufferMemory memory);
    ~StagingBufferManager();

    StagingBufferManager(StagingBufferManager const&) = delete;
    StagingBufferManager& operator=(StagingBufferManager const&) = delete;
    StagingBufferManager(StagingBufferManager&&) = delete;
    StagingBufferManager& operator=(StagingBufferManager&&) = delete;

    // Acquire a staging slice. The owning KVCM serializes access.
    // minSize: minimum required bytes. maxSize: best-effort upper bound.
    // sizeGranularity: required positive size multiple in bytes.
    // alignment: required power-of-two address alignment in bytes.
    StagingBuffer acquire(
        size_t minSize, size_t maxSize, size_t sizeGranularity, size_t alignment, std::optional<CUstream> stream);

    [[nodiscard]] size_t totalSize() const noexcept
    {
        return mSize;
    }

    [[nodiscard]] MemAddress baseAddress() const noexcept;

    [[nodiscard]] StagingBufferMemory memory() const noexcept;

private:
    friend class StagingBuffer;

    StagingBufferRange* reserve(
        size_t minSize, size_t maxSize, size_t sizeGranularity, size_t alignment, std::optional<CUstream> stream);
    void retire(StagingBufferRange* range, CachedCudaEvent readyEvent) noexcept;

    size_t mSize;
    std::variant<HostMem, CudaUniqPtr> mMemoryOwner;
    // Full spatial partition; retired fragments retain their previous temporal owner event.
    std::list<StagingBufferRange> mRanges;
    std::list<StagingBufferRange>::iterator mHead;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
