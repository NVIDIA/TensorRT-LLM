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

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// HostMem — mmap-backed pinned host memory, resizable via mremap.
// Mirrors _utils.py::HostMem.
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

    explicit HostMem(size_t size);
    ~HostMem();

    HostMem(HostMem const&) = delete;
    HostMem& operator=(HostMem const&) = delete;

    // Resize in-place (mremap, preserves data). Unregisters and re-registers with CUDA.
    void resize(size_t newSize);

    // Unregister from CUDA and unmap. Safe to call multiple times.
    void destroy();

    MemAddress address() const noexcept
    {
        return mAddr;
    }

    size_t size() const noexcept
    {
        return mSize;
    }

private:
    void registerToCuda();
    void unregisterFromCuda();
    void madvisePageMode();
    void parallelPrefault(int numThreads);

    MemAddress mAddr = 0;
    size_t mSize = 0;
    int mNumRegisteredChunks = 0;
    bool mUseThp = true;

    // Detect kernel version once at startup.
    static bool shouldUseChunkedRegistration();
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
