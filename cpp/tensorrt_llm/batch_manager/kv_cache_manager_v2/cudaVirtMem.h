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
#include "tensorrt_llm/runtime/locality_domain/locality_domain_utils.h"

#include <cuda.h>
#include <memory>
#include <optional>
#include <set>
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
// ---------------------------------------------------------------------------
class PooledPhysMemAllocator
{
public:
    using PooledPhysMem = SimplePool<PhysMemWrapper>::PoolItem;

    // physMemSize: size of each physical chunk in bytes.
    explicit PooledPhysMemAllocator(size_t physMemSize, int numLocalityDomains = 1);
    ~PooledPhysMemAllocator();

    PooledPhysMemAllocator(PooledPhysMemAllocator const&) = delete;
    PooledPhysMemAllocator& operator=(PooledPhysMemAllocator const&) = delete;

    // Borrow a physical memory handle from the pool (or allocate a new one).
    // Dropping the returned PhysMemHandle returns it to the pool.
    [[nodiscard]] PooledPhysMem acquire(LocalityDomainId localityDomainId = LocalityDomainId{0});

    // Release all cached (unused) physical memory back to the driver.
    // Mirrors Python PooledPhysMemAllocator.clear().
    void clear();

    [[nodiscard]] size_t physMemSize() const noexcept
    {
        return mPhysMemSize;
    }

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
};

// ---------------------------------------------------------------------------
// VirtMem — a virtual address range backed by physical GPU memory chunks.
// Mirrors _cuda_virt_mem.py::VirtMem.
//
// Each locality domain keeps its own stack of mappings. With one domain chunks are mapped and
// unmapped at the end of the range; with several the caller places every chunk at an explicit
// byte offset so the pools sharing a slot-id space agree on which domain backs which slots.
//
// Invariant: mappedBytes(d) == numPhysMem(d) * physMemSize() for every domain d
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

    // Map numPhysMem additional chunks of localityDomainId starting at byte offset `offset`.
    // offset defaults to the end of the mapped range and must be given with several domains.
    void extend(size_t numPhysMem, LocalityDomainId localityDomainId = LocalityDomainId{0},
        std::optional<size_t> offset = std::nullopt);

    // Unmap the most recently mapped numPhysMem chunks of localityDomainId (synchronizes CUDA first).
    void shrink(size_t numPhysMem, LocalityDomainId localityDomainId = LocalityDomainId{0});

    // Adjust mapped bytes of localityDomainId to exactly numBytes (extend or shrink).
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

    [[nodiscard]] size_t mappedBytes(LocalityDomainId localityDomainId = LocalityDomainId{0}) const noexcept
    {
        return mPhysMemAllocator.physMemSize() * numPhysMem(localityDomainId);
    }

    [[nodiscard]] size_t virtualBytes() const noexcept
    {
        return mVmSize;
    }

    [[nodiscard]] size_t numPhysMem(LocalityDomainId localityDomainId = LocalityDomainId{0}) const noexcept
    {
        return mMappings[localityDomainId].size();
    }

    [[nodiscard]] int numLocalityDomains() const noexcept
    {
        return mMappings.size().value();
    }

    // Merged mapped byte ranges [begin, end) relative to address(), across all domains.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> mappedRanges() const;

private:
    struct Mapping
    {
        size_t offset;
        PooledPhysMemAllocator::PooledPhysMem handle;
    };

    void push(PooledPhysMemAllocator::PooledPhysMem handle, size_t offset, LocalityDomainId localityDomainId);
    void pop(LocalityDomainId localityDomainId);

    CUdeviceptr mAddr = 0;
    size_t mVmSize = 0;
    PooledPhysMemAllocator& mPhysMemAllocator;
    TypedVec<LocalityDomainId, std::vector<Mapping>> mMappings;
    std::set<size_t> mMappedOffsets;
    CUmemAccessDesc mAccessDesc{};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
