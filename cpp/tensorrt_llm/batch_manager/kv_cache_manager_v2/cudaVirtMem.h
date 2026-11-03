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
#include <memory>
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
//
// Chunks are physMemSize() bytes, the unit sizes are counted in. With a
// maxChunkSize() above that, the allocator also hands out larger chunks of
// physMemSize() << k bytes (k >= 1) up to maxChunkSize(), which VirtMem maps
// wherever a range is aligned to and long enough for one. Sizes, and so the
// pools built on them, stay multiples of physMemSize(); only the number of
// separate allocations behind them drops.
// ---------------------------------------------------------------------------
class PooledPhysMemAllocator
{
public:
    using PooledPhysMem = SimplePool<PhysMemWrapper>::PoolItem;

    // physMemSize: size of each physical chunk in bytes.
    // maxChunkSize: largest chunk VirtMem may map; values at or below physMemSize disable larger chunks.
    //               Rounded down to physMemSize times a power of two.
    explicit PooledPhysMemAllocator(size_t physMemSize, size_t maxChunkSize = 0);
    ~PooledPhysMemAllocator();

    PooledPhysMemAllocator(PooledPhysMemAllocator const&) = delete;
    PooledPhysMemAllocator& operator=(PooledPhysMemAllocator const&) = delete;

    // Borrow a physical memory handle of physMemSize() bytes from the pool (or allocate a new one).
    // Dropping the returned PhysMemHandle returns it to the pool.
    [[nodiscard]] PooledPhysMem acquire();

    // Same for a chunk of `chunkSize` bytes: physMemSize() << k, at most maxChunkSize().
    [[nodiscard]] PooledPhysMem acquire(size_t chunkSize);

    // Release all cached (unused) physical memory back to the driver.
    // Mirrors Python PooledPhysMemAllocator.clear().
    void clear()
    {
        mPool.clear();
        for (auto& pool : mLargerPools)
        {
            pool->clear();
        }
    }

    [[nodiscard]] size_t physMemSize() const noexcept
    {
        return mPhysMemSize;
    }

    // Largest chunk this allocator hands out; physMemSize() when it has no larger chunks.
    [[nodiscard]] size_t maxChunkSize() const noexcept
    {
        return mPhysMemSize << mLargerPools.size();
    }

    // Chunk to map at byte `offset` of a range that still needs `remaining` bytes (both multiples of
    // physMemSize()): the largest size that `offset` is a multiple of and that fits in `remaining`.
    [[nodiscard]] size_t chunkSizeAt(size_t offset, size_t remaining) const noexcept;

    int deviceId() const noexcept
    {
        return mDeviceId;
    }

private:
    size_t mPhysMemSize{};
    int mDeviceId{};
    CUmemAllocationProp mProp{};
    SimplePool<PhysMemWrapper> mPool;
    // mLargerPools[k] hands out chunks of mPhysMemSize << (k + 1).
    std::vector<std::unique_ptr<SimplePool<PhysMemWrapper>>> mLargerPools;
};

// ---------------------------------------------------------------------------
// VirtMem — a virtual address range backed by physical GPU memory chunks.
// Physical chunks are mapped/unmapped at the end of the range (stack discipline).
// Mirrors _cuda_virt_mem.py::VirtMem.
//
// With an allocator that has larger chunks, a range is mapped from the largest
// chunks its offsets allow (see PooledPhysMemAllocator::chunkSizeAt), so the
// chunks need not all be the same size. A shrink unmaps only whole chunks
// above the new size: when the new end falls inside a larger chunk, that chunk
// stays mapped and usableBytes() drops below mappedBytes() until a later grow
// uses it again.
//
// Invariant: usableBytes() <= mappedBytes() <= virtualBytes(), and without
//            larger chunks mappedBytes() == usableBytes() == numPhysMem() * physMemSize()
// ---------------------------------------------------------------------------
class VirtMem
{
public:
    // vmSize: total virtual address space reserved (bytes).
    //         Must be a multiple of physMemSize.
    // physMemAllocator: shared allocator for physical chunks.
    // initNumPhysMem: number of physMemSize units to map immediately.
    VirtMem(size_t vmSize, PooledPhysMemAllocator& physMemAllocator, size_t initNumPhysMem = 0);
    ~VirtMem() noexcept;

    VirtMem(VirtMem const&) = delete;
    VirtMem& operator=(VirtMem const&) = delete;

    // Grow the usable range by numPhysMem units of physMemSize at the top of the address range.
    void extend(size_t numPhysMem);

    // Shrink the usable range by numPhysMem units (synchronizes CUDA first when it unmaps).
    void shrink(size_t numPhysMem);

    // Adjust the usable range to numBytes rounded up to physMemSize (extend or shrink).
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

    // Bytes backed by physical memory.
    [[nodiscard]] size_t mappedBytes() const noexcept
    {
        return mMappedBytes;
    }

    // Bytes the owner asked for (a multiple of physMemSize): what it may address.
    [[nodiscard]] size_t usableBytes() const noexcept
    {
        return mUsableBytes;
    }

    [[nodiscard]] size_t virtualBytes() const noexcept
    {
        return mVmSize;
    }

    // Number of mapped chunks (separate physical allocations).
    [[nodiscard]] size_t numPhysMem() const noexcept
    {
        return mPhysHandles.size();
    }

private:
    void push(PooledPhysMemAllocator::PooledPhysMem handle, size_t size);
    void pop();

    CUdeviceptr mAddr = 0;
    size_t mVmSize = 0;
    size_t mMappedBytes = 0;
    size_t mUsableBytes = 0;
    PooledPhysMemAllocator& mPhysMemAllocator;
    std::vector<PooledPhysMemAllocator::PooledPhysMem> mPhysHandles;
    std::vector<size_t> mChunkSizes;
    CUmemAccessDesc mAccessDesc{};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
