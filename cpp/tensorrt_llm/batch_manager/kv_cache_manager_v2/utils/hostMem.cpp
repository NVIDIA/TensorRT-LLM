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

#include "kv_cache_manager_v2/utils/hostMem.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <exception>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

MemAddress hostMmap(size_t size)
{
    // MAP_NORESERVE: the mapping is a reservation sized to an upper bound, and
    // only a prefix of it is ever committed. Without it a strict-overcommit
    // system charges the whole range against swap and refuses the mapping.
    void* ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ptr == MAP_FAILED || ptr == nullptr)
    {
        throw HostOOMError(std::string("mmap failed: ") + std::strerror(errno));
    }
    return reinterpret_cast<MemAddress>(ptr);
}

void hostMunmap(MemAddress ptr, size_t size) noexcept
{
    int ret = ::munmap(reinterpret_cast<void*>(ptr), size);
    if (ret != 0)
    {
        std::fprintf(stderr, "munmap failed with errno %d\n", errno);
    }
}

void resizeFile(int fd, size_t newSize)
{
    off_t oldSize = ::lseek(fd, 0, SEEK_END);
    if (static_cast<size_t>(oldSize) < newSize)
    {
        int ret = ::posix_fallocate(fd, oldSize, static_cast<off_t>(newSize - oldSize));
        if (ret != 0)
        {
            throw DiskOOMError("posix_fallocate failed: " + std::string(std::strerror(ret)));
        }
    }
    else if (static_cast<size_t>(oldSize) > newSize)
    {
        if (::ftruncate(fd, static_cast<off_t>(newSize)) != 0)
        {
            throw DiskOOMError("ftruncate failed: " + std::string(std::strerror(errno)));
        }
    }
}

bool hostUseThp()
{
    char const* value = std::getenv("TLLM_KV_CACHE_MANAGER_V2_THP");
    return value == nullptr || std::string_view(value) == "1";
}

size_t hostTotalMemory()
{
    long const pages = ::sysconf(_SC_PHYS_PAGES);
    long const pageSize = ::sysconf(_SC_PAGE_SIZE);
    TLLM_CHECK_WITH_INFO(pages > 0 && pageSize > 0, "Could not determine host memory size");
    return static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
}

void hostMadvisePageMode(MemAddress ptr, size_t size, bool useThp, HostMadviseFn madviseFn) noexcept
{
    HostMadviseFn const fn = madviseFn != nullptr ? madviseFn : ::madvise;
    int const advice = useThp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE;
    if (fn(reinterpret_cast<void*>(ptr), size, advice) != 0)
    {
        std::fprintf(stderr, "madvise failed with errno %d\n", errno);
    }
}

void hostPrefaultChunk(MemAddress ptr, size_t size, HostMadviseFn madviseFn, HostMemsetFn memsetFn)
{
    HostMadviseFn const advise = madviseFn != nullptr ? madviseFn : ::madvise;
    HostMemsetFn const touch = memsetFn != nullptr ? memsetFn : ::memset;
#ifdef MADV_POPULATE_WRITE
    if (advise(reinterpret_cast<void*>(ptr), size, MADV_POPULATE_WRITE) == 0)
    {
        return;
    }

    int const errorCode = errno;
    if (errorCode == EINVAL || errorCode == ENOSYS)
    {
        touch(reinterpret_cast<void*>(ptr), 0, size);
        return;
    }
    if (errorCode == ENOMEM)
    {
        throw HostOOMError("madvise(MADV_POPULATE_WRITE) failed: " + std::string(std::strerror(errorCode)));
    }
    throw std::system_error(errorCode, std::generic_category(), "madvise(MADV_POPULATE_WRITE) failed");
#else
    // MADV_POPULATE_WRITE requires glibc >= 2.34 / Linux >= 5.14 headers and is not defined in
    // older build environments (e.g. Rocky8 package-sanity images). Fall back to explicitly
    // touching the pages to force population, matching the EINVAL/ENOSYS runtime path above.
    (void) advise;
    touch(reinterpret_cast<void*>(ptr), 0, size);
#endif
}

// ---------------------------------------------------------------------------
// HostMem implementation
// ---------------------------------------------------------------------------

size_t HostMem::chunkCount() const noexcept
{
    return mChunkSize == 0 ? 0 : divUp(mSize, mChunkSize);
}

//! Commits [beginOffset, mSize) in chunk-sized steps, publishing the watermark
//! after each one.
//!
//! `beginOffset` is a byte offset rather than a chunk index because a grow
//! resumes at the old size, which need not fall on a chunk boundary. The step
//! that straddles the boundary is committed as a short range, so no byte is
//! ever committed twice.
void HostMem::commitRange(size_t beginOffset)
{
    size_t offset = beginOffset;
    while (offset < mSize)
    {
        size_t const end = std::min(roundUp(offset + 1, mChunkSize), mSize);
        mBacking->commit(offset, end - offset);
        offset = end;
        // A chunk counts as usable only once it is committed to its end, so a
        // partially committed trailing chunk does not advance the watermark.
        mFilledChunks.store(
            static_cast<uint32_t>(offset == mSize ? chunkCount() : offset / mChunkSize), std::memory_order_release);
        mFilledChunks.notify_all();
    }
}

void HostMem::startFill(size_t beginOffset)
{
    mFillWorker = std::thread(
        [this, beginOffset]
        {
            // A new thread starts on device 0 whatever the creator selected, and
            // the VMM backing maps against the current device.
            try
            {
                TLLM_CUDA_CHECK(cudaSetDevice(mDevice));
                commitRange(beginOffset);
            }
            catch (...)
            {
                // Stored before the sentinel: a waiter that observes the
                // sentinel through an acquire load is guaranteed to see it.
                mFillFailure = std::current_exception();
                mFilledChunks.store(kFillFailed, std::memory_order_release);
                mFilledChunks.notify_all();
            }
        });
}

void HostMem::joinFill() noexcept
{
    if (mFillWorker.joinable())
    {
        mFillWorker.join();
    }
}

void HostMem::waitUsable(size_t endOffset) const
{
    if (endOffset == 0 || mChunkSize == 0)
    {
        return;
    }
    TLLM_CHECK_DEBUG(endOffset <= mSize);
    auto const needed = static_cast<uint32_t>(divUp(endOffset, mChunkSize));

    uint32_t current = mFilledChunks.load(std::memory_order_acquire);
    while (current < needed)
    {
        mFilledChunks.wait(current, std::memory_order_acquire);
        current = mFilledChunks.load(std::memory_order_acquire);
    }
    if (current == kFillFailed)
    {
        std::rethrow_exception(mFillFailure);
    }
}

HostMem::HostMem(size_t maxSize, size_t initialSize, size_t commitUnit, bool waitForFill)
    : mWaitForFill(waitForFill)
{
    TLLM_CUDA_CHECK(cudaGetDevice(&mDevice));
    if (maxSize == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(initialSize <= maxSize, "HostMem initial size %zu exceeds max size %zu", initialSize, maxSize);
    mBacking = createHostMemBacking(commitUnit);
    size_t const granularity = mBacking->commitGranularity();
    TLLM_CHECK_WITH_INFO(initialSize % granularity == 0,
        "HostMem size %zu is not a multiple of the commit granularity %zu", initialSize, granularity);
    mChunkSize = roundUp(kFillChunkSize, granularity);

    try
    {
        // The reservation is rounded up so that the largest accepted resize is
        // itself a valid, aligned size.
        mMaxSize = roundUp(maxSize, granularity);
        mSize = initialSize;
        mAddr = mBacking->reserve(mMaxSize);
        TLLM_CHECK_DEBUG(mAddr % granularity == 0);
        startFill(0);
        if (mWaitForFill)
        {
            waitUsable(mSize);
        }
    }
    catch (...)
    {
        joinFill();
        mBacking->destroy();
        mAddr = 0;
        mSize = 0;
        mMaxSize = 0;
        throw;
    }
}

HostMem::~HostMem()
{
    KVCM2_POISON_ON_EXCEPT([this]() { destroy(); });
}

void HostMem::destroy()
{
    joinFill();
    if (mBacking != nullptr)
    {
        mBacking->destroy();
        mBacking.reset();
    }
    mAddr = 0;
    mSize = 0;
    mMaxSize = 0;
    mChunkSize = 0;
    mFillFailure = nullptr;
    mFilledChunks.store(0, std::memory_order_release);
}

void HostMem::resize(size_t newSize)
{
    TLLM_CHECK_WITH_INFO(mBacking != nullptr, "HostMem::resize called after destroy");

    // The fill must be finished before the mapping changes underneath it, and a
    // failed fill has to surface here rather than be resized over.
    joinFill();
    if (mFilledChunks.load(std::memory_order_acquire) == kFillFailed)
    {
        std::rethrow_exception(mFillFailure);
    }

    if (newSize == mSize)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(newSize % mBacking->commitGranularity() == 0,
        "HostMem size %zu is not a multiple of the commit granularity %zu", newSize, mBacking->commitGranularity());
    TLLM_CHECK_WITH_INFO(newSize <= mMaxSize,
        "HostMem resize to %zu exceeds the reserved maximum of %zu; raise the tier's maxQuota", newSize, mMaxSize);

    if (newSize < mSize)
    {
        // Shrink within the existing reservation: only the tail is released and
        // the base address is unchanged. The release is exact, because newSize
        // is a multiple of the unit the backing allocates in.
        mBacking->decommit(newSize, mSize - newSize);
        mSize = newSize;
        mFilledChunks.store(static_cast<uint32_t>(chunkCount()), std::memory_order_release);
        mFilledChunks.notify_all();
        return;
    }

    // Grow: the address space is already reserved, so this only commits the new
    // tail. Everything below keeps its contents and its address.
    size_t const oldSize = mSize;
    mSize = newSize;
    // The watermark counts whole chunks, and the old size rounded up to one.
    // Growing makes that rounding wrong -- it would claim the new tail is
    // already usable -- so it is rebased to the chunks genuinely complete.
    mFilledChunks.store(static_cast<uint32_t>(oldSize / mChunkSize), std::memory_order_release);
    try
    {
        startFill(oldSize);
    }
    catch (...)
    {
        // mSize already names the larger range, so without this every later
        // wait past the old size would block on a watermark that no worker is
        // left to advance.
        mFillFailure = std::current_exception();
        mFilledChunks.store(kFillFailed, std::memory_order_release);
        mFilledChunks.notify_all();
        throw;
    }
    if (mWaitForFill)
    {
        waitUsable(mSize);
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
