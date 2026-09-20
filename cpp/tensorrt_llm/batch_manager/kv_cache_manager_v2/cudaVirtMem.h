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

#include <cuda.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// PhysMemChunk — RAII wrapper for a CUmemGenericAllocationHandle.
// Constructor calls cuMemCreate, destructor calls cuMemRelease.
// ---------------------------------------------------------------------------
class PhysMemWrapper
{
public:
    PhysMemWrapper(size_t size, CUmemAllocationProp const& prop);
    ~PhysMemWrapper();

    PhysMemWrapper(PhysMemWrapper const&) = delete;
    PhysMemWrapper& operator=(PhysMemWrapper const&) = delete;

    [[nodiscard]] CUmemGenericAllocationHandle handle() const noexcept
    {
        return mHandle;
    }

private:
    CUmemGenericAllocationHandle mHandle;
};

// ---------------------------------------------------------------------------
// PooledPhysMemAllocator — creates and pools physical GPU memory chunks.
// Mirrors _cuda_virt_mem.py::PooledPhysMemAllocator.
// ---------------------------------------------------------------------------
class PooledPhysMemAllocator
{
public:
    using PooledPhysMem = SimplePool<PhysMemWrapper>::PoolItem;

    // physMemSize: size of each physical chunk in bytes.
    explicit PooledPhysMemAllocator(size_t physMemSize);
    ~PooledPhysMemAllocator();

    PooledPhysMemAllocator(PooledPhysMemAllocator const&) = delete;
    PooledPhysMemAllocator& operator=(PooledPhysMemAllocator const&) = delete;

    // Borrow a physical memory handle from the pool (or allocate a new one).
    // Dropping the returned PhysMemHandle returns it to the pool.
    [[nodiscard]] PooledPhysMem acquire();

    // Release all cached (unused) physical memory back to the driver.
    // Mirrors Python PooledPhysMemAllocator.clear().
    void clear()
    {
        mPool.clear();
    }

    [[nodiscard]] size_t physMemSize() const noexcept
    {
        return mPhysMemSize;
    }

    int deviceId() const noexcept
    {
        return mDeviceId;
    }

private:
    size_t mPhysMemSize{};
    int mDeviceId{};
    CUmemAllocationProp mProp{};
    SimplePool<PhysMemWrapper> mPool;
};

// ---------------------------------------------------------------------------
// VirtMem — a virtual address range backed by physical GPU memory chunks.
// Physical chunks are mapped/unmapped at the end of the range (stack discipline).
// Mirrors _cuda_virt_mem.py::VirtMem.
//
// Invariant: mappedBytes() == numPhysMem() * physMemSize()
//            mappedBytes() <= virtualBytes()
// ---------------------------------------------------------------------------
class VirtMem
{
public:
    // vmSize: total virtual address space reserved (bytes).
    //         Must be a multiple of physMemSize.
    // physMemAllocator: shared allocator for physical chunks.
    // initNumPhysMem: number of physical chunks to map immediately.
    VirtMem(size_t vmSize, PooledPhysMemAllocator& physMemAllocator, size_t initNumPhysMem = 0);
    ~VirtMem() noexcept;

    VirtMem(VirtMem const&) = delete;
    VirtMem& operator=(VirtMem const&) = delete;

    // Map numPhysMem additional chunks at the top of the address range.
    void extend(size_t numPhysMem);

    // Unmap numPhysMem chunks from the top (synchronizes CUDA first).
    void shrink(size_t numPhysMem);

    // Adjust mapped bytes to exactly numBytes (extend or shrink).
    void realloc(size_t numBytes);

    void destroy();

    [[nodiscard]] MemAddress address() const noexcept
    {
        return static_cast<MemAddress>(mAddr);
    }

    [[nodiscard]] size_t physMemSize() const noexcept
    {
        return mPhysMemAllocator.physMemSize();
    }

    [[nodiscard]] size_t mappedBytes() const noexcept
    {
        return mPhysMemAllocator.physMemSize() * numPhysMem();
    }

    [[nodiscard]] size_t virtualBytes() const noexcept
    {
        return mVmSize;
    }

    [[nodiscard]] size_t numPhysMem() const noexcept
    {
        return mPhysHandles.size();
    }

private:
    void push(PooledPhysMemAllocator::PooledPhysMem handle);
    void pop();

    CUdeviceptr mAddr = 0;
    size_t mVmSize = 0;
    PooledPhysMemAllocator& mPhysMemAllocator;
    std::vector<PooledPhysMemAllocator::PooledPhysMem> mPhysHandles;
    CUmemAccessDesc mAccessDesc{};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
