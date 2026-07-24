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

#include "tensorrt_llm/common/assert.h"
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
}

HostMem::HostMem(size_t size)
    : mUseThp(hostUseThp())
{
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
    destroy();
}

void HostMem::resize(size_t newSize)
{
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
    size_t const numChunks = (mSize + kPrefaultChunkSize - 1) / kPrefaultChunkSize;
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
            size_t const offset = chunkIndex * kPrefaultChunkSize;
            size_t const chunkSize = std::min(kPrefaultChunkSize, mSize - offset);
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

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
