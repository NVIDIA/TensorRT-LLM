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
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <unistd.h>

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
{
    if (size == 0)
    {
        return;
    }
    mAddr = hostMmap(size);
    TLLM_CHECK_DEBUG(mAddr % kAlignment == 0);
    mSize = size;
    madviseHugepages();
    registerToCuda();
}

HostMem::~HostMem()
{
    destroy();
}

void HostMem::resize(size_t newSize)
{
    unregisterFromCuda();
    mAddr = hostMremap(mAddr, mSize, newSize);
    TLLM_CHECK_DEBUG(mAddr % kAlignment == 0);
    mSize = newSize;
    madviseHugepages();
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

void HostMem::madviseHugepages()
{
    TLLM_CHECK_DEBUG(mAddr && mSize);
    // Advisory — ignore failures.
    ::madvise(reinterpret_cast<void*>(mAddr), mSize, MADV_HUGEPAGE);
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
