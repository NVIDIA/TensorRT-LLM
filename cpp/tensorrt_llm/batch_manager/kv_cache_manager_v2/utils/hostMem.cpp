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
#include "tensorrt_llm/runtime/moeLoadBalancer/topologyDetector.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
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
#include <utility>
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
    // Off unless asked for. MADV_HUGEPAGE opts a region's faults into direct
    // compaction wherever /sys/kernel/mm/transparent_hugepage/defrag is
    // "madvise" or "always". On a node whose free memory has fragmented below
    // the 2 MiB order that costs far more than the huge pages save, and
    // fragmentation accumulates with uptime, so the penalty grows while the
    // benefit does not.
    char const* value = std::getenv("TLLM_KV_CACHE_MANAGER_V2_THP");
    return value != nullptr && std::string_view(value) == "1";
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

namespace
{

//! Threads that populate the initial fill alongside the one committing it.
//! Populating scales with threads until the kernel's page allocator saturates.
constexpr int kInitialFillHelpers = 7;

} // namespace

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
//!
//! With `helperCount` > 0 the range is opened up front and that many threads
//! populate it chunk by chunk ahead of the commits. Populating in parallel
//! slows everything else in the process, so only the initial fill, which runs
//! before serving, uses helpers. Until it ends, a reader that skips
//! waitUsable() reads zeros past the watermark instead of faulting.
//!
//! A step is populated by whichever thread claims it first, and one a helper
//! has claimed is committed only once that helper is done with it; a helper
//! that fails hands its step back. Where MADV_POPULATE_WRITE is unavailable
//! populating means zeroing, which must not reach a step already handed out.
void HostMem::commitRange(size_t beginOffset, int helperCount)
{
    enum StepState : uint8_t
    {
        kPending,
        kClaimed,
        kPopulated
    };

    bool const useHelpers = helperCount > 0 && mSize - beginOffset > mChunkSize;
    size_t const firstChunk = beginOffset / mChunkSize;
    // One entry per step, indexed by its chunk relative to firstChunk.
    std::vector<std::atomic<uint8_t>> stepStates(useHelpers ? chunkCount() - firstChunk : 0);
    auto const stepRange = [&](size_t step)
    {
        size_t const chunk = firstChunk + step;
        return std::pair{std::max(chunk * mChunkSize, beginOffset), std::min((chunk + 1) * mChunkSize, mSize)};
    };
    std::atomic<size_t> nextHelperStep{0};
    // Declared after what they use: destruction stops and joins them on every
    // way out of this function.
    std::vector<std::jthread> helpers;
    if (useHelpers)
    {
        mBacking.open(beginOffset, mSize - beginOffset);
        try
        {
            for (int i = 0; i < helperCount; ++i)
            {
                helpers.emplace_back(
                    [this, &stepStates, &stepRange, &nextHelperStep](std::stop_token stop)
                    {
                        try
                        {
                            TLLM_CUDA_CHECK(cudaSetDevice(mDevice));
                            runtime::TopologyDetector::getInstance().bindThreadByCurrentGpu();
                            for (size_t step;
                                 !stop.stop_requested() && (step = nextHelperStep.fetch_add(1)) < stepStates.size();)
                            {
                                auto& state = stepStates[step];
                                uint8_t expected = kPending;
                                if (!state.compare_exchange_strong(expected, kClaimed, std::memory_order_relaxed))
                                {
                                    continue;
                                }
                                auto const [begin, end] = stepRange(step);
                                try
                                {
                                    mBacking.populate(begin, end - begin);
                                }
                                catch (...)
                                {
                                    // Handed back for commit() to populate.
                                    state.store(kPending, std::memory_order_release);
                                    state.notify_all();
                                    throw;
                                }
                                state.store(kPopulated, std::memory_order_release);
                                state.notify_all();
                            }
                        }
                        catch (...)
                        {
                            // commit() populates whatever a helper leaves and
                            // reports any failure itself.
                        }
                    });
            }
        }
        catch (std::system_error const&)
        {
            // Out of threads: those running and commit() cover the range.
        }
    }

    // Whether a helper populated the step; if not, the caller now owns it.
    auto const claimOrAwaitStep = [&](size_t step)
    {
        auto& state = stepStates[step];
        for (;;)
        {
            uint8_t seen = kPending;
            if (state.compare_exchange_strong(seen, kClaimed, std::memory_order_acquire))
            {
                return false;
            }
            if (seen == kPopulated)
            {
                return true;
            }
            state.wait(kClaimed, std::memory_order_acquire);
        }
    };

    size_t offset = beginOffset;
    while (offset < mSize)
    {
        size_t const end = std::min(roundUp(offset + 1, mChunkSize), mSize);
        bool const populated = useHelpers && claimOrAwaitStep(offset / mChunkSize - firstChunk);
        mBacking.commit(offset, end - offset, populated);
        offset = end;
        // A chunk counts as usable only once it is committed to its end, so a
        // partially committed trailing chunk does not advance the watermark.
        mFilledChunks.store(
            static_cast<uint32_t>(offset == mSize ? chunkCount() : offset / mChunkSize), std::memory_order_release);
        mFilledChunks.notify_all();
    }
}

void HostMem::startFill(size_t beginOffset, int helperCount)
{
    mFillWorker = std::thread(
        [this, beginOffset, helperCount]
        {
            // A new thread starts on device 0 whatever the creator selected, and
            // registration needs a context on the right one. Selecting it here
            // also gives bindThreadByCurrentGpu the right GPU to bind to, which
            // keeps the prefault writes on the node that holds the pages.
            try
            {
                TLLM_CUDA_CHECK(cudaSetDevice(mDevice));
                // Registering host memory would invalidate a CUDA graph capture
                // running on any other thread; relaxed mode exempts this one.
                // Safe because it only registers above the watermark, which no
                // consumer may touch.
                cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;
                TLLM_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&captureMode));
                runtime::TopologyDetector::getInstance().bindThreadByCurrentGpu();
                commitRange(beginOffset, helperCount);
            }
            catch (...)
            {
                if (helperCount > 0 && mSize - beginOffset > mChunkSize)
                {
                    // Helpers may have populated past the failed commit, which
                    // released only its own step.
                    size_t const committed = mFilledChunks.load(std::memory_order_relaxed) * mChunkSize;
                    mBacking.decommit(committed, mSize - committed);
                }
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

HostMem::HostMem(size_t vmSize, size_t initialSize, HostMemBackingOptions const& options, bool waitForFill)
    : mBacking(options)
    , mWaitForFill(waitForFill)
{
    TLLM_CUDA_CHECK(cudaGetDevice(&mDevice));
    if (vmSize == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(
        initialSize <= vmSize, "HostMem initial size %zu exceeds reserved size %zu", initialSize, vmSize);
    size_t const granularity = mBacking.commitGranularity();
    TLLM_CHECK_WITH_INFO(initialSize % granularity == 0,
        "HostMem size %zu is not a multiple of the commit granularity %zu", initialSize, granularity);
    mChunkSize = roundUp(kFillChunkSize, granularity);

    try
    {
        // The reservation is rounded up so that the largest accepted resize is
        // itself a valid, aligned size.
        mVmSize = roundUp(vmSize, granularity);
        mSize = initialSize;
        mAddr = mBacking.reserve(mVmSize);
        startFill(0, kInitialFillHelpers);
        if (mWaitForFill)
        {
            waitUsable(mSize);
        }
    }
    catch (...)
    {
        joinFill();
        mBacking.destroy();
        mAddr = 0;
        mSize = 0;
        mVmSize = 0;
        throw;
    }
}

HostMem::~HostMem()
{
    // Joined outside the poison guard: once KVCM2 is poisoned the guard skips destroy(), and a
    // joinable mFillWorker would then terminate the process when the member is destroyed.
    joinFill();
    // Guarded so a failure here reaches the latch rather than escaping a destructor. Whether the
    // range itself survives a poisoned teardown is the backing's decision, not this one: its
    // destructor runs with the member whether or not the guard skipped the call above.
    KVCM2_POISON_ON_EXCEPT([this]() { destroy(); });
}

void HostMem::destroy()
{
    joinFill();
    mBacking.destroy();
    mAddr = 0;
    mSize = 0;
    mVmSize = 0;
    mChunkSize = 0;
    mFillFailure = nullptr;
    mFilledChunks.store(0, std::memory_order_release);
}

void HostMem::resize(size_t newSize)
{
    TLLM_CHECK_WITH_INFO(mVmSize != 0, "HostMem::resize called on a destroyed or empty allocation");

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

    TLLM_CHECK_WITH_INFO(newSize % mBacking.commitGranularity() == 0,
        "HostMem size %zu is not a multiple of the commit granularity %zu", newSize, mBacking.commitGranularity());
    TLLM_CHECK_WITH_INFO(newSize <= mVmSize,
        "HostMem resize to %zu exceeds the reserved maximum of %zu; raise the tier's maxQuota", newSize, mVmSize);

    if (newSize < mSize)
    {
        // Shrink within the existing reservation: only the tail is released and
        // the base address is unchanged. The release is exact, because newSize
        // is a multiple of the unit the backing allocates in.
        mBacking.decommit(newSize, mSize - newSize);
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
        startFill(oldSize, 0);
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
