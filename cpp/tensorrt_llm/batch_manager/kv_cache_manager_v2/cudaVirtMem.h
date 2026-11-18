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
#include "kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/runtime/locality_domain/locality_domain_utils.h"

#include <cuda.h>
#include <map>
#include <memory>
#include <optional>
#include <utility>
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

    // Adopts a handle created elsewhere (locality-domain allocations).
    explicit PhysMemWrapper(CUmemGenericAllocationHandle handle) noexcept
        : mHandle(handle)
    {
    }

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
// With more than one locality domain, each domain has its own pool of chunks located in that
// domain's memory (or ordinary device memory in mock mode) and physMemSize() is rounded up to a
// multiple of the localized allocation granularity.
//
// Chunks are physMemSize() bytes, the unit sizes are counted in. With one
// locality domain and a maxChunkSize() above that, the allocator also hands out
// larger chunks of physMemSize() << k bytes (k >= 1) up to maxChunkSize(), which
// VirtMem maps wherever a range is aligned to and long enough for one. Sizes,
// and so the pools built on them, stay multiples of physMemSize(); only the
// number of separate allocations behind them drops.
// ---------------------------------------------------------------------------
class PooledPhysMemAllocator
{
public:
    using PooledPhysMem = SimplePool<PhysMemWrapper>::PoolItem;

    // physMemSize: size of each physical chunk in bytes.
    // maxChunkSize: largest chunk VirtMem may map; values at or below physMemSize disable larger chunks.
    //               Rounded down to physMemSize times a power of two. Ignored with several locality domains.
    explicit PooledPhysMemAllocator(size_t physMemSize, int numLocalityDomains = 1, size_t maxChunkSize = 0);
    ~PooledPhysMemAllocator();

    PooledPhysMemAllocator(PooledPhysMemAllocator const&) = delete;
    PooledPhysMemAllocator& operator=(PooledPhysMemAllocator const&) = delete;

    // Borrow a physical memory handle of physMemSize() bytes from the pool (or allocate a new one).
    // Dropping the returned PhysMemHandle returns it to the pool.
    [[nodiscard]] PooledPhysMem acquire(LocalityDomainId localityDomainId = LocalityDomainId{0});

    // Same for a chunk of `chunkSize` bytes: physMemSize() << k, at most maxChunkSize().
    [[nodiscard]] PooledPhysMem acquireChunk(size_t chunkSize);

    // Release all cached (unused) physical memory back to the driver.
    // Mirrors Python PooledPhysMemAllocator.clear().
    void clear();

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

    int numLocalityDomains() const noexcept
    {
        return mPools.size().value();
    }

private:
    void selectDefaultProp();
    void selectLocalizedProp(int numLocalityDomains);
    PhysMemWrapper* createChunk(LocalityDomainId localityDomainId);

    size_t mPhysMemSize{};
    int mDeviceId{};
    CUmemAllocationProp mProp{};
    // Present only when chunks are really placed per domain; mock mode leaves it empty.
    std::optional<locality_domain::LocalizationHandle> mLocalization;
    TypedVec<LocalityDomainId, std::unique_ptr<SimplePool<PhysMemWrapper>>> mPools;
    // mLargerPools[k] hands out chunks of mPhysMemSize << (k + 1); empty with several locality domains.
    std::vector<std::unique_ptr<SimplePool<PhysMemWrapper>>> mLargerPools;
};

// ---------------------------------------------------------------------------
// VirtMem — a virtual address range backed by physical GPU memory chunks.
// Mirrors _cuda_virt_mem.py::VirtMem.
//
// Each locality domain keeps its own stack of mappings. With one domain chunks are mapped and
// unmapped at the end of the range; with several the caller places every chunk at an explicit
// byte offset so the pools sharing a slot-id space agree on which domain backs which slots.
//
// With one domain and an allocator that has larger chunks, the range is mapped
// from the largest chunks its offsets allow (see
// PooledPhysMemAllocator::chunkSizeAt), so the chunks need not all be the same
// size. A shrink unmaps only whole chunks above the new size: when the new end
// falls inside a larger chunk, that chunk stays mapped and usableBytes() drops
// below mappedBytes() until a later grow uses it again.
//
// Invariant: usableBytes(d) <= mappedBytes(d) for every domain d, and without
//            larger chunks mappedBytes(d) == usableBytes(d) == numPhysMem(d) * physMemSize()
//            sum of mappedBytes(d) <= virtualBytes()
// ---------------------------------------------------------------------------
class VirtMem
{
public:
    // vmSize: total virtual address space reserved (bytes).
    //         Must be a multiple of physMemSize.
    // physMemAllocator: shared allocator for physical chunks.
    // initNumPhysMem: number of physical chunks to map immediately (domain 0).
    VirtMem(size_t vmSize, PooledPhysMemAllocator& physMemAllocator, size_t initNumPhysMem = 0);
    ~VirtMem() noexcept;

    VirtMem(VirtMem const&) = delete;
    VirtMem& operator=(VirtMem const&) = delete;

    // Grow the usable range of localityDomainId by numPhysMem units of physMemSize, mapped starting at
    // byte offset `offset`. offset defaults to the end of the usable range and must be given with
    // several domains. With one domain, growing at the end of the usable range may map larger chunks.
    void extend(size_t numPhysMem, LocalityDomainId localityDomainId = LocalityDomainId{0},
        std::optional<size_t> offset = std::nullopt);

    // Shrink the usable range of localityDomainId by numPhysMem units, unmapping the most recently
    // mapped chunks (synchronizes CUDA first when it unmaps).
    void shrink(size_t numPhysMem, LocalityDomainId localityDomainId = LocalityDomainId{0});

    // Adjust the usable range of localityDomainId to numBytes rounded up to physMemSize (extend or shrink).
    void realloc(size_t numBytes, LocalityDomainId localityDomainId = LocalityDomainId{0});

    void destroy();

    [[nodiscard]] MemAddress address() const noexcept
    {
        return static_cast<MemAddress>(mAddr);
    }

    [[nodiscard]] size_t physMemSize() const noexcept
    {
        return mPhysMemAllocator.physMemSize();
    }

    // Bytes of localityDomainId backed by physical memory.
    [[nodiscard]] size_t mappedBytes(LocalityDomainId localityDomainId = LocalityDomainId{0}) const noexcept
    {
        return mMappedBytes[localityDomainId];
    }

    // Bytes of localityDomainId the owner asked for (a multiple of physMemSize): what it may address.
    [[nodiscard]] size_t usableBytes(LocalityDomainId localityDomainId = LocalityDomainId{0}) const noexcept
    {
        return mUsableBytes[localityDomainId];
    }

    [[nodiscard]] size_t virtualBytes() const noexcept
    {
        return mVmSize;
    }

    // Number of mapped chunks (separate physical allocations) of localityDomainId.
    [[nodiscard]] size_t numPhysMem(LocalityDomainId localityDomainId = LocalityDomainId{0}) const noexcept
    {
        return mMappings[localityDomainId].size();
    }

    [[nodiscard]] int numLocalityDomains() const noexcept
    {
        return mMappings.size().value();
    }

    // Merged mapped byte ranges [begin, end) relative to address(), across all domains.
    [[nodiscard]] std::vector<HalfOpenRange<size_t>> mappedRanges() const;

private:
    struct Mapping
    {
        size_t offset;
        size_t size;
        PooledPhysMemAllocator::PooledPhysMem handle;
    };

    void push(PooledPhysMemAllocator::PooledPhysMem handle, size_t offset, size_t size,
        LocalityDomainId localityDomainId);
    void pop(LocalityDomainId localityDomainId);
    // One domain: map or unmap chunks at the end so the usable range becomes `target` bytes.
    void resizeAtEnd(size_t target);

    CUdeviceptr mAddr = 0;
    size_t mVmSize = 0;
    PooledPhysMemAllocator& mPhysMemAllocator;
    TypedVec<LocalityDomainId, std::vector<Mapping>> mMappings;
    TypedVec<LocalityDomainId, size_t> mMappedBytes;
    TypedVec<LocalityDomainId, size_t> mUsableBytes;
    // Offset -> size of every mapped chunk, across all domains.
    std::map<size_t, size_t> mMappedOffsets;
    CUmemAccessDesc mAccessDesc{};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
