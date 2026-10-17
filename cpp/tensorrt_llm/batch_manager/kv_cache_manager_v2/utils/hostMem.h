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

#include <cstddef>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// HostMem — mmap-backed pinned host memory, resizable.
//
// Given a maximum size, HostMem reserves that much address space up front and resizes in place: the buffer never
// moves, CUDA registrations stay valid, and a resize pins or unpins only the changed range. Registrations are kept
// in extents of at most 1 GB whose boundaries are multiples of lcm(4 KB, alignHint), so with alignHint = slot size no
// slot straddles two registrations and per-slot copy-engine descriptors stay inside one registered region.
// TLLM_KV_CACHE_MANAGER_V2_HOST_DELTA_PIN=0 disables this. Without a maximum size (or if the reservation fails),
// resize() mremaps the buffer and re-registers all of it.
//
// Memory is:
//   - Anonymous private mmap
//   - Advised according to TLLM_KV_CACHE_MANAGER_V2_THP
//   - Optionally prefaulted in parallel before CUDA registration
//   - Registered to CUDA as page-locked (CU_MEMHOSTREGISTER_DEVICEMAP)
//
// On kernels 6.11/6.12/6.13, pinning is chunked in 2GB pieces to work around
// a kernel bug that prevents pinning more than 2GB in one call.
// ---------------------------------------------------------------------------
class HostMem
{
public:
    static constexpr size_t kAlignment = 4096;                 // 4 KB
    static constexpr size_t kChunkSize = 2ULL << 30;           // 2 GB
    static constexpr size_t kPrefaultChunkSize = 512ULL << 20; // 512 MB

    //! @param size       Initial size in bytes (multiple of kAlignment).
    //! @param alignHint  Registration extents never split a multiple of this (e.g. a slot size), so a copy of one
    //!                   slot never crosses a registered region.
    //! @param maxSize    When non-zero, reserve this much address space so resize() grows and shrinks in place and
    //!                   pins / unpins only the changed range. Zero keeps the mremap + full re-registration path.
    explicit HostMem(size_t size, size_t alignHint = kAlignment, size_t maxSize = 0);
    ~HostMem();

    HostMem(HostMem const&) = delete;
    HostMem& operator=(HostMem const&) = delete;

    // Resize, preserving data. In place (fixed address, delta registration) when address space was reserved;
    // otherwise mremap with full unregistration and re-registration.
    void resize(size_t newSize);

    // Unregister from CUDA and unmap. Safe to call multiple times.
    void destroy();

    // Detect whether host registration must be chunked on this kernel. Cached after the first query.
    static bool shouldUseChunkedRegistration();

    MemAddress address() const noexcept
    {
        return mAddr;
    }

    size_t size() const noexcept
    {
        return mSize;
    }

private:
    static constexpr size_t kMaxExtentSize = 1ULL << 30; // 1 GB, below the 2 GB kernel-bug limit

    void registerToCuda();
    void unregisterFromCuda();
    void madvisePageMode();
    void parallelPrefault(int numThreads);
    void parallelPrefaultRange(size_t begin, size_t end, int numThreads);

    // Reserved-address-space (delta registration) mode.
    [[nodiscard]] bool reserved() const noexcept
    {
        return mReserved != 0;
    }

    //! Returns false, leaving the object empty, when the address space cannot be reserved.
    bool initReserved(size_t size, size_t maxSize);
    void resizeReserved(size_t newSize);
    void commitRange(size_t begin, size_t end);
    void releaseRange(size_t begin, size_t end) noexcept;
    void registerRange(size_t begin, size_t end);
    void unregisterAfter(size_t offset);
    void unregisterAllExtents() noexcept;

    MemAddress mAddr = 0;
    size_t mSize = 0;
    int mNumRegisteredChunks = 0;
    bool mUseThp = true;

    size_t mReserved = 0;          // reserved address space in bytes; 0 = mremap mode
    size_t mRegAlign = kAlignment; // lcm(kAlignment, alignHint): legal extent boundary
    size_t mMaxExtent = kMaxExtentSize;
    // Registered extents [begin, end) as offsets, ascending and contiguous from 0 to mSize. Every boundary except
    // mSize is a multiple of mRegAlign; a final extent shorter than mRegAlign is the unaligned tail.
    std::vector<std::pair<size_t, size_t>> mExtents;
};

// ---------------------------------------------------------------------------
// Low-level wrappers used internally (also exposed for storage pool use).
// ---------------------------------------------------------------------------
MemAddress hostMmap(size_t size);                                      // throws HostOOMError
void hostMunmap(MemAddress ptr, size_t size) noexcept;
MemAddress hostMremap(MemAddress ptr, size_t oldSize, size_t newSize); // throws HostOOMError
void resizeFile(int fd, size_t newSize);                               // throws DiskOOMError

using HostMadviseFn = int (*)(void*, size_t, int);
using HostMemsetFn = void* (*) (void*, int, size_t);

bool hostUseThp();
int hostPrefaultThreads();
void hostMadvisePageMode(MemAddress ptr, size_t size, bool useThp, HostMadviseFn madviseFn = nullptr) noexcept;
void hostPrefaultChunk(MemAddress ptr, size_t size, HostMadviseFn madviseFn = nullptr, HostMemsetFn memsetFn = nullptr);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
