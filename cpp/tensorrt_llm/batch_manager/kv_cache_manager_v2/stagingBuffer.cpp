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

#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

using MemoryOwner = std::variant<HostMem, CudaUniqPtr>;

CudaUniqPtr allocateDeviceMemory(size_t size)
{
    CUdeviceptr ptr = 0;
    cuCheck(cuMemAlloc(&ptr, size));
    return CudaUniqPtr{reinterpret_cast<std::byte*>(ptr)};
}

MemoryOwner makeMemoryOwner(size_t size, StagingBufferMemory memory)
{
    TLLM_CHECK_WITH_INFO(size > 0, "Staging buffer size must be non-zero");
    switch (memory)
    {
    case StagingBufferMemory::kPinnedHost: return MemoryOwner{std::in_place_type<HostMem>, size};
    case StagingBufferMemory::kDevice: return MemoryOwner{std::in_place_type<CudaUniqPtr>, allocateDeviceMemory(size)};
    default: throw std::invalid_argument("StagingBufferManager received an invalid memory kind");
    }
}

} // namespace

void DeviceDeleter::operator()(std::byte* ptr) const noexcept
{
    if (ptr != nullptr)
    {
        CUresult const result = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
        if (result != CUDA_SUCCESS)
        {
            char const* errorString = nullptr;
            cuGetErrorString(result, &errorString);
            TLLM_LOG_ERROR(
                "Failed to free staging-buffer device memory: %s", errorString != nullptr ? errorString : "unknown");
        }
    }
}

struct StagingBufferRange
{
    StagingBufferRange(size_t begin, size_t end, CachedCudaEvent readyEvent, bool retired)
        : begin(begin)
        , end(end)
        , readyEvent(std::move(readyEvent))
        , retired(retired)
    {
    }

    [[nodiscard]] bool contains(size_t offset) const noexcept
    {
        return begin <= offset && offset < end;
    }

    size_t begin;
    size_t end;
    CachedCudaEvent readyEvent;
    bool retired;
};

StagingBuffer::StagingBuffer(StagingBufferManager& manager, size_t minSize, size_t maxSize, size_t sizeGranularity,
    size_t alignment, std::optional<CUstream> stream)
    : mManager(manager)
    , mRange(manager.reserve(minSize, maxSize, sizeGranularity, alignment, stream))
    , mStream(stream)
{
}

StagingBuffer::~StagingBuffer() noexcept
{
    // One completion event protects the entire reserved byte range.
    CachedCudaEvent finishEvent = CachedCudaEvent::makeNull();
    if (mStream.has_value())
    {
        finishEvent = CachedCudaEvent(reinterpret_cast<CudaStream>(*mStream));
    }
    mManager.retire(mRange, std::move(finishEvent));
}

MemAddress StagingBuffer::address() const noexcept
{
    return mManager.baseAddress() + mRange->begin;
}

size_t StagingBuffer::size() const noexcept
{
    return mRange->end - mRange->begin;
}

void StagingBuffer::setStream(std::optional<CUstream> stream)
{
    if (mStream == stream)
    {
        return;
    }
    if (mStream.has_value())
    {
        if (stream.has_value())
        {
            CachedCudaEvent handoff(reinterpret_cast<CudaStream>(*mStream));
            handoff.waitInStream(reinterpret_cast<CudaStream>(*stream));
        }
        else
        {
            TLLM_CU_CHECK(cuStreamSynchronize(*mStream));
        }
    }
    mStream = stream;
}

// ---------------------------------------------------------------------------
// StagingBufferManager
// ---------------------------------------------------------------------------

StagingBufferManager::StagingBufferManager(size_t size, StagingBufferMemory memory)
    : mSize(size)
    , mMemoryOwner(makeMemoryOwner(size, memory))
{
    mRanges.emplace_back(0, mSize, CachedCudaEvent::makeNull(), /*retired=*/true);
    mHead = mRanges.begin();
}

StagingBufferManager::~StagingBufferManager() noexcept
{
    terminateOnException("Failed to destroy staging-buffer manager safely",
        [&]()
        {
            std::vector<CachedCudaEvent*> readyEvents;
            readyEvents.reserve(mRanges.size());
            for (auto& range : mRanges)
            {
                TLLM_CHECK_WITH_INFO(range.retired, "Destroying a staging-buffer manager with a live buffer");
                readyEvents.push_back(&range.readyEvent);
            }
            synchronizeAll(readyEvents);
        });
}

MemAddress StagingBufferManager::baseAddress() const noexcept
{
    if (auto const* host = std::get_if<HostMem>(&mMemoryOwner))
    {
        return host->address();
    }
    return reinterpret_cast<MemAddress>(std::get<CudaUniqPtr>(mMemoryOwner).get());
}

StagingBufferMemory StagingBufferManager::memory() const noexcept
{
    return std::holds_alternative<HostMem>(mMemoryOwner) ? StagingBufferMemory::kPinnedHost
                                                         : StagingBufferMemory::kDevice;
}

StagingBuffer StagingBufferManager::acquire(
    size_t minSize, size_t maxSize, size_t sizeGranularity, size_t alignment, std::optional<CUstream> stream)
{
    return {*this, minSize, maxSize, sizeGranularity, alignment, stream};
}

StagingBufferRange* StagingBufferManager::reserve(
    size_t minSize, size_t maxSize, size_t sizeGranularity, size_t alignment, std::optional<CUstream> stream)
{
    TLLM_CHECK_WITH_INFO(minSize > 0 && minSize <= mSize && minSize <= maxSize && sizeGranularity > 0 && alignment > 0
            && (alignment & (alignment - 1)) == 0 && sizeGranularity % alignment == 0 && minSize % sizeGranularity == 0
            && maxSize % sizeGranularity == 0,
        "StagingBufferManager reserve() received invalid arguments");

    MemAddress const bufferBase = baseAddress();
    auto alignedOffset = [bufferBase, alignment](size_t offset)
    { return static_cast<size_t>(roundUp(bufferBase + offset, static_cast<MemAddress>(alignment)) - bufferBase); };

    TLLM_CHECK_WITH_INFO(
        alignedOffset(0) + minSize <= mSize, "StagingBuffer minimum size does not fit with the requested alignment");

    using RangeIter = std::list<StagingBufferRange>::iterator;
    auto findRange = [this](size_t offset, RangeIter hint) -> RangeIter
    {
        TLLM_CHECK(offset < mSize);
        auto containsOffset = [offset](StagingBufferRange const& range) { return range.contains(offset); };
        if (hint != mRanges.end() && hint->begin <= offset)
        {
            auto range = std::find_if(hint, mRanges.end(), containsOffset);
            TLLM_CHECK_DEBUG(range != mRanges.end());
            return range;
        }

        auto reverseRange = std::find_if(std::make_reverse_iterator(hint), mRanges.rend(), containsOffset);
        TLLM_CHECK_DEBUG(reverseRange != mRanges.rend());
        return std::prev(reverseRange.base());
    };

    auto splitRange = [this, &findRange](RangeIter hint, size_t offset) -> RangeIter
    {
        if (offset == 0)
        {
            return mRanges.begin();
        }
        if (offset == mSize)
        {
            return mRanges.end();
        }
        auto range = findRange(offset, hint);
        if (range->begin == offset)
        {
            return range;
        }
        TLLM_CHECK(range->retired);
        auto const rightEnd = range->end;
        range->end = offset;
        return mRanges.emplace(std::next(range), offset, rightEnd, range->readyEvent, /*retired=*/true);
    };

    auto reserveFrom = [&](RangeIter hint) -> StagingBufferRange*
    {
        auto runBegin = hint;
        while (runBegin != mRanges.end())
        {
            while (runBegin != mRanges.end() && !runBegin->retired)
            {
                ++runBegin;
            }
            if (runBegin == mRanges.end())
            {
                return nullptr;
            }

            size_t const dataBegin = alignedOffset(runBegin->begin);
            auto runEnd = runBegin;
            size_t availableEnd = runBegin->begin;
            auto hasSpace = [&](size_t size) { return dataBegin <= availableEnd && size <= availableEnd - dataBegin; };
            size_t numRunRanges = 0;
            while (runEnd != mRanges.end() && runEnd->retired && !hasSpace(maxSize))
            {
                TLLM_CHECK_DEBUG(runEnd->begin <= availableEnd);
                availableEnd = runEnd->end;
                ++runEnd;
                ++numRunRanges;
            }
            if (hasSpace(minSize))
            {
                size_t const usableBytes = std::min(maxSize, availableEnd - dataBegin);
                size_t const dataSize = usableBytes - usableBytes % sizeGranularity;
                TLLM_CHECK_DEBUG(dataSize >= minSize);
                size_t const end = dataBegin + dataSize;
                auto payloadBegin = splitRange(runBegin, dataBegin);
                auto payloadEnd = splitRange(payloadBegin, end);

                auto collectEvents = [&](auto& readyEvents)
                {
                    readyEvents.reserve(numRunRanges);
                    for (auto range = payloadBegin; range != payloadEnd; ++range)
                    {
                        TLLM_CHECK_DEBUG(range->retired);
                        readyEvents.push_back(&range->readyEvent);
                    }
                };
                if (stream.has_value())
                {
                    std::vector<CachedCudaEvent const*> readyEvents;
                    collectEvents(readyEvents);
                    streamWaitEvents(reinterpret_cast<CudaStream>(*stream), readyEvents);
                }
                else
                {
                    std::vector<CachedCudaEvent*> readyEvents;
                    collectEvents(readyEvents);
                    synchronizeAll(readyEvents);
                }

                mRanges.erase(payloadBegin, payloadEnd);
                auto range
                    = mRanges.emplace(payloadEnd, dataBegin, end, CachedCudaEvent::makeNull(), /*retired=*/false);
                mHead = end == mSize ? mRanges.begin() : payloadEnd;
                return &*range;
            }

            runBegin = runEnd;
        }
        return nullptr;
    };

    if (auto* range = reserveFrom(mHead))
    {
        return range;
    }
    if (mHead != mRanges.begin())
    {
        if (auto* range = reserveFrom(mRanges.begin()))
        {
            return range;
        }
    }

    TLLM_CHECK_WITH_INFO(false,
        "StagingBufferManager has no contiguous retired range satisfying the request: minSize=%zu, maxSize=%zu, "
        "totalSize=%zu",
        minSize, maxSize, mSize);
    return nullptr;
}

void StagingBufferManager::retire(StagingBufferRange* range, CachedCudaEvent readyEvent) noexcept
{
    TLLM_CHECK_DEBUG(range != nullptr);
    TLLM_CHECK_DEBUG(!range->retired);
    range->readyEvent = std::move(readyEvent);
    range->retired = true;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
