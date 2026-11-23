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

#include "kv_cache_manager_v2/cudaVirtMem.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <cstring>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// PooledPhysMemAllocator
// ---------------------------------------------------------------------------

static bool isPropSupported(CUmemAllocationProp const& prop)
{
    CUmemGenericAllocationHandle handle;
    CUresult err = cuMemCreate(&handle, 2ULL << 20, &prop, 0);
    if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED || err == CUDA_ERROR_INVALID_DEVICE
        || err == CUDA_ERROR_INVALID_VALUE)
    {
        return false;
    }
    if (err == CUDA_SUCCESS)
    {
        cuMemRelease(handle);
        return true;
    }
    throw CuError(err);
}

// ---------------------------------------------------------------------------
// PhysMemChunk
// ---------------------------------------------------------------------------

PhysMemWrapper::PhysMemWrapper(size_t size, CUmemAllocationProp const& prop)
{
    cuCheck(cuMemCreate(&mHandle, size, &prop, 0));
}

PhysMemWrapper::PhysMemWrapper(CUmemGenericAllocationHandle adopted) noexcept
    : mHandle(adopted)
{
}

PhysMemWrapper::~PhysMemWrapper()
{
    if (mOwned)
    {
        cuMemRelease(mHandle);
    }
}

// ---------------------------------------------------------------------------
// PooledPhysMemAllocator
// ---------------------------------------------------------------------------

PooledPhysMemAllocator::PooledPhysMemAllocator(size_t physMemSize)
    : mPhysMemSize(physMemSize)
    , mPool([this]() -> PhysMemWrapper* { return new PhysMemWrapper(mPhysMemSize, mProp); },
          [](PhysMemWrapper* chunk) { delete chunk; })
{
    // Get current device.
    cuCheck(cuCtxGetDevice(&mDeviceId));

    // Build the best allocation property. Prefer shareable handle types so
    // UCX cuda_ipc can export the memory for intra-node zero-copy transfers:
    // FABRIC (MNNVL) > POSIX_FILE_DESCRIPTOR (pidfd-based, UCX >= 1.22) > NONE,
    // and within each handle type prefer gpuDirectRDMACapable over not.
    mProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    mProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    mProp.location.id = mDeviceId;

    constexpr CUmemAllocationHandleType kHandleTypePreference[]
        = {CU_MEM_HANDLE_TYPE_FABRIC, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, CU_MEM_HANDLE_TYPE_NONE};
    bool supported = false;
    for (auto const handleType : kHandleTypePreference)
    {
        for (bool const rdmaCapable : {true, false})
        {
            mProp.requestedHandleTypes = handleType;
            mProp.allocFlags.gpuDirectRDMACapable = rdmaCapable ? 1 : 0;
            if (isPropSupported(mProp))
            {
                supported = true;
                break;
            }
        }
        if (supported)
        {
            break;
        }
    }
    if (!supported)
    {
        throw std::runtime_error("PooledPhysMemAllocator: no supported physical memory allocation property");
    }
    if (!mProp.allocFlags.gpuDirectRDMACapable)
    {
        // Otherwise this surfaces only later, as an opaque RDMA registration failure.
        TLLM_LOG_WARNING(
            "PooledPhysMemAllocator: device %d refused gpuDirectRDMACapable for KV cache memory with handle type %d, "
            "so the pools are allocated without it. nvidia_peermem (ibv_reg_mr) cannot register this memory for "
            "GPUDirect RDMA; dma-buf registration still can.",
            static_cast<int>(mDeviceId), static_cast<int>(mProp.requestedHandleTypes));
    }
}

PooledPhysMemAllocator::PooledPhysMemAllocator(ImportTag, size_t chunkBytes)
    : mPhysMemSize(chunkBytes)
    , mImported(true)
    , mPool(
          [this]() -> PhysMemWrapper*
          {
              // Runs under the pool's mutex, as does the destroy function below; nothing else touches the list.
              if (mFreeImported.empty())
              {
                  throw OutOfMemoryError("Remote host tier: every lent chunk is in use");
              }
              auto* chunk = new PhysMemWrapper(mFreeImported.back());
              mFreeImported.pop_back();
              return chunk;
          },
          [this](PhysMemWrapper* chunk)
          {
              mFreeImported.push_back(chunk->release());
              delete chunk;
          })
{
    cuCheck(cuCtxGetDevice(&mDeviceId));
}

namespace
{

//! Reads the first 8 bytes of an imported chunk through a temporary device-only mapping.
uint64_t readChunkHead(CUmemGenericAllocationHandle handle, size_t chunkBytes, int deviceId)
{
    CUdeviceptr va = 0;
    // Aligned to the chunk, which is a multiple of the imported allocation's granularity (checked by the caller).
    cuCheck(cuMemAddressReserve(&va, chunkBytes, chunkBytes, 0, 0));
    bool mapped = false;
    uint64_t head = 0;
    try
    {
        cuCheck(cuMemMap(va, chunkBytes, 0, handle, 0));
        mapped = true;
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = deviceId;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuCheck(cuMemSetAccess(va, chunkBytes, &access, 1));
        cuCheck(cuMemcpyDtoH(&head, va, sizeof(head)));
    }
    catch (...)
    {
        if (mapped)
        {
            cuMemUnmap(va, chunkBytes);
        }
        cuMemAddressFree(va, chunkBytes);
        throw;
    }
    cuCheck(cuMemUnmap(va, chunkBytes));
    cuCheck(cuMemAddressFree(va, chunkBytes));
    return head;
}

} // namespace

std::unique_ptr<PooledPhysMemAllocator> PooledPhysMemAllocator::importFabric(
    size_t chunkBytes, std::vector<std::string> const& fabricHandles, uint64_t verifyNonce)
{
    TLLM_CHECK_WITH_INFO(chunkBytes > 0 && !fabricHandles.empty(), "Remote host tier: no chunks to import");
    std::unique_ptr<PooledPhysMemAllocator> allocator(new PooledPhysMemAllocator(ImportTag{}, chunkBytes));
    auto& imported = allocator->mFreeImported;
    imported.reserve(fabricHandles.size());
    for (size_t index = 0; index < fabricHandles.size(); ++index)
    {
        auto const& bytes = fabricHandles[index];
        CUmemFabricHandle fabric{};
        TLLM_CHECK_WITH_INFO(
            bytes.size() == sizeof(fabric.data), "Remote host tier: bad fabric handle size %zu", bytes.size());
        std::memcpy(fabric.data, bytes.data(), sizeof(fabric.data));
        CUmemGenericAllocationHandle handle{};
        cuCheck(cuMemImportFromShareableHandle(&handle, &fabric, CU_MEM_HANDLE_TYPE_FABRIC));
        // From here the destructor releases it, also when a check below throws.
        imported.push_back(handle);

        CUmemAllocationProp prop{};
        cuCheck(cuMemGetAllocationPropertiesFromHandle(&prop, handle));
        // Only the location type is checked. An imported handle can report HOST_NUMA id 0 whatever node the lender
        // allocated on, so the NUMA id is not validated.
        TLLM_CHECK_WITH_INFO(
            prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA || prop.location.type == CU_MEM_LOCATION_TYPE_HOST,
            "Remote host tier: chunk %zu is not host memory (location type %d)", index,
            static_cast<int>(prop.location.type));
        size_t granularity = 0;
        cuCheck(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        TLLM_CHECK_WITH_INFO(granularity > 0 && chunkBytes % granularity == 0,
            "Remote host tier: chunk size %zu is not a multiple of the granularity %zu", chunkBytes, granularity);
        allocator->mImportGranularity = std::max(allocator->mImportGranularity, granularity);

        uint64_t const head = readChunkHead(handle, chunkBytes, allocator->mDeviceId);
        uint64_t const expected = verifyNonce ^ static_cast<uint64_t>(index);
        TLLM_CHECK_WITH_INFO(head == expected,
            "Remote host tier: chunk %zu reads 0x%016llx, expected 0x%016llx (stale or foreign handles)", index,
            static_cast<unsigned long long>(head), static_cast<unsigned long long>(expected));
    }
    allocator->mNumImported = imported.size();
    // acquire() pops from the back; reverse so the first chunks are handed out first.
    std::reverse(imported.begin(), imported.end());
    return allocator;
}

PooledPhysMemAllocator::~PooledPhysMemAllocator()
{
    mPool.clear();
    for (auto handle : mFreeImported)
    {
        cuMemRelease(handle);
    }
    mFreeImported.clear();
}

PooledPhysMemAllocator::PooledPhysMem PooledPhysMemAllocator::acquire()
{
    return mPool.get();
}

// ---------------------------------------------------------------------------
// VirtMem
// ---------------------------------------------------------------------------

VirtMem::VirtMem(size_t vmSize, PooledPhysMemAllocator& physMemAllocator, size_t initNumPhysMem)
    : mVmSize(vmSize)
    , mPhysMemAllocator(physMemAllocator)
{
    TLLM_CHECK_DEBUG(vmSize % physMemAllocator.physMemSize() == 0);

    // Imported chunks are mapped at chunk offsets, so the range is aligned to the chunk size, a multiple of the
    // imported allocation's granularity. Local allocators keep the driver's default alignment.
    size_t const alignment = physMemAllocator.isImported() ? physMemAllocator.physMemSize() : 0;
    cuCheck(cuMemAddressReserve(&mAddr, vmSize, alignment, 0, 0));
    if (alignment != 0 && (mAddr % alignment != 0 || mAddr % physMemAllocator.importGranularity() != 0))
    {
        cuMemAddressFree(mAddr, vmSize);
        TLLM_THROW("VirtMem: reserved address 0x%llx is not aligned to the imported chunk size %zu",
            static_cast<unsigned long long>(mAddr), alignment);
    }

    mAccessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    mAccessDesc.location.id = physMemAllocator.deviceId();
    mAccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    try
    {
        extend(initNumPhysMem);
    }
    catch (...)
    {
        // A constructor that throws never runs ~VirtMem, so the reservation above has to be
        // released here or the address range stays held for the life of the process. extend()
        // has already unmapped whatever it managed to map before failing.
        KVCM2_POISON_ON_EXCEPT([this]() { destroy(); });
        throw;
    }
}

VirtMem::~VirtMem() noexcept
{
    KVCM2_POISON_ON_EXCEPT([this]() { destroy(); });
}

void VirtMem::push(PooledPhysMemAllocator::PooledPhysMem handle)
{
    size_t physSize = mPhysMemAllocator.physMemSize();
    CUdeviceptr offset = mAddr + physSize * static_cast<size_t>(mPhysHandles.size());
    TLLM_CHECK_DEBUG(physSize * (mPhysHandles.size() + 1) <= mVmSize);

    cuCheck(cuMemMap(offset, physSize, 0, handle->handle(), 0));
    if (!mPhysMemAllocator.isImported())
    {
        cuCheck(cuMemSetAccess(offset, physSize, &mAccessDesc, 1));
    }
    else
    {
        try
        {
            cuCheck(cuMemSetAccess(offset, physSize, &mAccessDesc, 1));
        }
        catch (...)
        {
            // An imported chunk goes back to the fixed set of lent chunks when `handle` is dropped, and a later
            // push may map it again; it must not stay mapped here.
            cuMemUnmap(offset, physSize);
            throw;
        }
    }
    mPhysHandles.push_back(std::move(handle));
}

void VirtMem::pop()
{
    TLLM_CHECK_DEBUG(!mPhysHandles.empty());
    size_t physSize = mPhysMemAllocator.physMemSize();
    CUdeviceptr offset = mAddr + physSize * (mPhysHandles.size() - 1);
    cuCheck(cuMemUnmap(offset, physSize));
    mPhysHandles.pop_back(); // PhysMemHandle destructor returns to pool
}

void VirtMem::extend(size_t numToAdd)
{
    size_t old = numPhysMem();
    try
    {
        for (size_t i = 0; i < numToAdd; ++i)
        {
            push(mPhysMemAllocator.acquire());
        }
    }
    catch (...)
    {
        // Rollback: remove any newly-mapped chunks.
        while (numPhysMem() > old)
        {
            pop();
        }
        throw;
    }
}

void VirtMem::shrink(size_t numToRemove)
{
    cuCheck(cuCtxSynchronize());
    for (size_t i = 0; i < numToRemove; ++i)
    {
        pop();
    }
}

void VirtMem::realloc(size_t numBytes)
{
    size_t physSize = mPhysMemAllocator.physMemSize();
    size_t required = divUp(numBytes, physSize);
    size_t current = numPhysMem();
    if (required > current)
    {
        extend(required - current);
    }
    else if (required < current)
    {
        shrink(current - required);
    }
}

void VirtMem::destroy()
{
    if (mVmSize == 0)
    {
        return;
    }
    cuCheck(cuCtxSynchronize());
    while (!mPhysHandles.empty())
    {
        pop();
    }
    cuCheck(cuMemAddressFree(mAddr, mVmSize));
    mAddr = 0;
    mVmSize = 0;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
