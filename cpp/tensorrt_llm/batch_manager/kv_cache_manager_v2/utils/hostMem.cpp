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
#include "tensorrt_llm/common/logger.h"
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
#include <numeric>
#include <optional>
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
    void* ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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

MemAddress hostMremap(MemAddress ptr, size_t oldSize, size_t newSize)
{
    void* newPtr = ::mremap(reinterpret_cast<void*>(ptr), oldSize, newSize, MREMAP_MAYMOVE);
    if (newPtr == MAP_FAILED || newPtr == nullptr)
    {
        throw HostOOMError(std::string("mremap failed: ") + std::strerror(errno));
    }
    return reinterpret_cast<MemAddress>(newPtr);
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

int hostPrefaultThreads()
{
    char const* value = std::getenv("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS");
    if (value != nullptr)
    {
        return std::stoi(value);
    }
    unsigned int const detectedCpuCount = std::thread::hardware_concurrency();
    unsigned int const cpuCount = detectedCpuCount == 0 ? 32 : detectedCpuCount;
    return static_cast<int>(std::min(64U, cpuCount / 2));
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

bool HostMem::shouldUseChunkedRegistration()
{
    static bool const chunked = []()
    {
        struct utsname u
        {
        };

        if (::uname(&u) != 0)
        {
            return false;
        }
        // Check for Linux kernel 6.11, 6.12, 6.13 prefix.
        std::string_view rel{u.release};
        for (auto prefix : {"6.11", "6.12", "6.13"})
        {
            if (rel.substr(0, 4) == prefix)
            {
                return true;
            }
        }
        return false;
    }();
    return chunked;
}

namespace
{
bool hostDeltaPinEnabled()
{
    char const* value = std::getenv("TLLM_KV_CACHE_MANAGER_V2_HOST_DELTA_PIN");
    return value == nullptr || std::string_view(value) != "0";
}
} // namespace

HostMem::HostMem(size_t size, size_t alignHint, size_t maxSize)
    : mUseThp(hostUseThp())
{
    size_t const regAlign = std::lcm(kAlignment, std::max<size_t>(alignHint, 1));
    if (maxSize != 0 && regAlign <= kMaxExtentSize && hostDeltaPinEnabled())
    {
        mRegAlign = regAlign;
        mMaxExtent = kMaxExtentSize / regAlign * regAlign;
        if (initReserved(size, std::max(maxSize, size)))
        {
            return;
        }
        mRegAlign = kAlignment;
        mMaxExtent = kMaxExtentSize;
    }
    if (size == 0)
    {
        return;
    }
    mAddr = hostMmap(size);
    TLLM_CHECK_DEBUG(mAddr % kAlignment == 0);
    mSize = size;
    try
    {
        madvisePageMode();
        int const prefaultThreads = hostPrefaultThreads();
        if (prefaultThreads > 0)
        {
            parallelPrefault(prefaultThreads);
        }
        registerToCuda();
    }
    catch (...)
    {
        unregisterFromCuda();
        hostMunmap(mAddr, mSize);
        mAddr = 0;
        mSize = 0;
        throw;
    }
}

HostMem::~HostMem()
{
    KVCM2_POISON_ON_EXCEPT([this]() { destroy(); });
}

void HostMem::resize(size_t newSize)
{
    if (reserved())
    {
        resizeReserved(newSize);
        return;
    }
    unregisterFromCuda();
    try
    {
        mAddr = hostMremap(mAddr, mSize, newSize);
        TLLM_CHECK_DEBUG(mAddr % kAlignment == 0);
        mSize = newSize;
        madvisePageMode();
    }
    catch (...)
    {
        registerToCuda();
        throw;
    }
    registerToCuda();
}

void HostMem::destroy()
{
    if (mAddr == 0)
    {
        return;
    }
    if (reserved())
    {
        unregisterAllExtents();
        hostMunmap(mAddr, mReserved);
        mAddr = 0;
        mSize = 0;
        mReserved = 0;
        return;
    }
    unregisterFromCuda();
    hostMunmap(mAddr, mSize);
    mAddr = 0;
    mSize = 0;
}

void HostMem::madvisePageMode()
{
    TLLM_CHECK_DEBUG(mAddr && mSize);
    hostMadvisePageMode(mAddr, mSize, mUseThp);
}

void HostMem::parallelPrefault(int numThreads)
{
    parallelPrefaultRange(0, mSize, numThreads);
}

void HostMem::parallelPrefaultRange(size_t begin, size_t end, int numThreads)
{
    if (end <= begin || numThreads <= 0)
    {
        return;
    }
    size_t const rangeSize = end - begin;
    size_t const numChunks = (rangeSize + kPrefaultChunkSize - 1) / kPrefaultChunkSize;
    int const workerCount = std::min<int>(numThreads, static_cast<int>(numChunks));
    std::atomic_size_t nextChunk{0};
    std::atomic_bool failed{false};
    std::exception_ptr error;
    std::mutex errorMutex;

    auto worker = [&]()
    {
        while (!failed.load())
        {
            size_t const chunkIndex = nextChunk.fetch_add(1);
            if (chunkIndex >= numChunks)
            {
                return;
            }
            size_t const offset = begin + chunkIndex * kPrefaultChunkSize;
            size_t const chunkSize = std::min(kPrefaultChunkSize, end - offset);
            try
            {
                hostPrefaultChunk(mAddr + offset, chunkSize);
            }
            catch (...)
            {
                failed.store(true);
                std::lock_guard<std::mutex> lock(errorMutex);
                if (error == nullptr)
                {
                    error = std::current_exception();
                }
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(workerCount));
    for (int i = 0; i < workerCount; ++i)
    {
        workers.emplace_back(worker);
    }
    for (auto& thread : workers)
    {
        thread.join();
    }
    if (error != nullptr)
    {
        std::rethrow_exception(error);
    }
}

void HostMem::registerToCuda()
{
    TLLM_CHECK_DEBUG(mNumRegisteredChunks == 0);
    static bool chunked = shouldUseChunkedRegistration();

    size_t chunkSize = (chunked && mSize > kChunkSize) ? kChunkSize : mSize;
    for (size_t offset = 0; offset < mSize; offset += chunkSize)
    {
        size_t sz = std::min(chunkSize, mSize - offset);
        CUresult res = cuMemHostRegister(
            reinterpret_cast<void*>(mAddr + offset), sz, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
        cuCheck(res);
        ++mNumRegisteredChunks;
    }
}

void HostMem::unregisterFromCuda()
{
    static bool chunked = shouldUseChunkedRegistration();
    size_t chunkSize = (chunked && mSize > kChunkSize) ? kChunkSize : mSize;
    for (size_t offset = 0; offset < mSize && mNumRegisteredChunks > 0; offset += chunkSize)
    {
        cuMemHostUnregister(reinterpret_cast<void*>(mAddr + offset));
        --mNumRegisteredChunks;
    }
    TLLM_CHECK_DEBUG(mNumRegisteredChunks == 0);
}

// ---------------------------------------------------------------------------
// Reserved-address-space mode: the buffer never moves, so CUDA registrations stay valid across resizes and only
// the changed range is pinned or unpinned. The copy-engine path needs every per-page copy inside one registered
// region; extents are cut only at multiples of mRegAlign (a multiple of the slot size), so no slot straddles two.
// ---------------------------------------------------------------------------

bool HostMem::initReserved(size_t size, size_t maxSize)
{
    size_t const reserve = roundUp(std::max(maxSize, kAlignment), kAlignment);
    void* ptr = ::mmap(nullptr, reserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ptr == MAP_FAILED || ptr == nullptr)
    {
        // Address space can be limited (RLIMIT_AS); the mremap mode needs only the committed size.
        TLLM_LOG_WARNING(
            "HostMem: cannot reserve %zu bytes of address space (%s); resizing will re-register the "
            "whole buffer",
            reserve, std::strerror(errno));
        return false;
    }
    mAddr = reinterpret_cast<MemAddress>(ptr);
    mReserved = reserve;
    mSize = 0;
    try
    {
        resizeReserved(size);
    }
    catch (...)
    {
        unregisterAllExtents();
        hostMunmap(mAddr, mReserved);
        mAddr = 0;
        mSize = 0;
        mReserved = 0;
        throw;
    }
    return true;
}

void HostMem::commitRange(size_t begin, size_t end)
{
    if (end <= begin)
    {
        return;
    }
    void* addr = reinterpret_cast<void*>(mAddr + begin);
    if (::mprotect(addr, end - begin, PROT_READ | PROT_WRITE) != 0)
    {
        throw HostOOMError(std::string("mprotect failed: ") + std::strerror(errno));
    }
    try
    {
        hostMadvisePageMode(mAddr + begin, end - begin, mUseThp);
        parallelPrefaultRange(begin, end, hostPrefaultThreads());
    }
    catch (...)
    {
        releaseRange(begin, end);
        throw;
    }
}

void HostMem::releaseRange(size_t begin, size_t end) noexcept
{
    if (end <= begin)
    {
        return;
    }
    // Replacing the range with a fresh PROT_NONE mapping frees its pages and keeps the address space reserved.
    void* ptr = ::mmap(reinterpret_cast<void*>(mAddr + begin), end - begin, PROT_NONE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
    if (ptr == MAP_FAILED)
    {
        std::fprintf(stderr, "HostMem: releasing [%zu, %zu) failed with errno %d\n", begin, end, errno);
    }
}

void HostMem::registerRange(size_t begin, size_t end)
{
    TLLM_CHECK_DEBUG(begin % mRegAlign == 0);
    size_t const alignedEnd = end / mRegAlign * mRegAlign;
    size_t cur = begin;
    while (cur < end)
    {
        size_t next = std::min(end, cur + mMaxExtent);
        if (next == end && alignedEnd > cur && alignedEnd < end)
        {
            next = alignedEnd; // keep the unaligned tail as its own small extent
        }
        CUresult const res = cuMemHostRegister(reinterpret_cast<void*>(mAddr + cur), next - cur,
            CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
        cuCheck(res);
        mExtents.emplace_back(cur, next);
        cur = next;
    }
}

void HostMem::unregisterAfter(size_t offset)
{
    // Drop every extent that reaches past offset. One that straddles it is re-registered up to offset; because its
    // start is a multiple of mRegAlign that keeps the boundary invariant, and it costs at most one extent.
    std::optional<size_t> reRegisterFrom;
    while (!mExtents.empty() && mExtents.back().second > offset)
    {
        auto const [b, e] = mExtents.back();
        cuCheck(cuMemHostUnregister(reinterpret_cast<void*>(mAddr + b)));
        mExtents.pop_back();
        if (b < offset)
        {
            reRegisterFrom = b;
        }
    }
    if (reRegisterFrom.has_value())
    {
        registerRange(*reRegisterFrom, offset);
    }
}

void HostMem::unregisterAllExtents() noexcept
{
    while (!mExtents.empty())
    {
        cuMemHostUnregister(reinterpret_cast<void*>(mAddr + mExtents.back().first));
        mExtents.pop_back();
    }
}

void HostMem::resizeReserved(size_t newSize)
{
    TLLM_CHECK_DEBUG(newSize % kAlignment == 0);
    size_t const oldSize = mSize;
    if (newSize == oldSize)
    {
        return;
    }
    if (newSize > mReserved)
    {
        // Out of reserved address space: move the committed pages into a larger reservation. mremap moves page
        // tables, not data, but the new address invalidates every registration, so this costs a full re-pin.
        size_t const reserve = roundUp(newSize, kAlignment);
        void* fresh = ::mmap(nullptr, reserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (fresh == MAP_FAILED || fresh == nullptr)
        {
            throw HostOOMError(std::string("mmap reserve failed: ") + std::strerror(errno));
        }
        unregisterAllExtents();
        if (oldSize > 0)
        {
            void* moved
                = ::mremap(reinterpret_cast<void*>(mAddr), oldSize, oldSize, MREMAP_MAYMOVE | MREMAP_FIXED, fresh);
            if (moved == MAP_FAILED)
            {
                int const err = errno;
                hostMunmap(reinterpret_cast<MemAddress>(fresh), reserve);
                registerRange(0, oldSize);
                throw HostOOMError(std::string("mremap failed: ") + std::strerror(err));
            }
        }
        if (mReserved > oldSize)
        {
            hostMunmap(mAddr + oldSize, mReserved - oldSize);
        }
        mAddr = reinterpret_cast<MemAddress>(fresh);
        mReserved = reserve;
        try
        {
            commitRange(oldSize, newSize);
            registerRange(0, newSize);
        }
        catch (...)
        {
            // The data now lives at the new address; restore the old size there.
            unregisterAllExtents();
            releaseRange(oldSize, newSize);
            registerRange(0, oldSize);
            throw;
        }
        mSize = newSize;
        return;
    }
    if (newSize > oldSize)
    {
        commitRange(oldSize, newSize);
        // Re-register only the unaligned tail extent (shorter than mRegAlign) together with the new range.
        size_t regStart = oldSize;
        if (!mExtents.empty() && mExtents.back().second % mRegAlign != 0)
        {
            regStart = mExtents.back().first;
            cuCheck(cuMemHostUnregister(reinterpret_cast<void*>(mAddr + regStart)));
            mExtents.pop_back();
        }
        try
        {
            registerRange(regStart, newSize);
        }
        catch (...)
        {
            unregisterAfter(regStart);
            registerRange(regStart, oldSize);
            releaseRange(oldSize, newSize);
            throw;
        }
        mSize = newSize;
        return;
    }
    unregisterAfter(newSize);
    releaseRange(newSize, oldSize);
    mSize = newSize;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
