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

PhysMemWrapper::~PhysMemWrapper()
{
    cuMemRelease(mHandle);
}

// ---------------------------------------------------------------------------
// PooledPhysMemAllocator
// ---------------------------------------------------------------------------

PooledPhysMemAllocator::PooledPhysMemAllocator(size_t physMemSize, size_t maxChunkSize)
    : mPhysMemSize(physMemSize)
    , mPool([this]() -> PhysMemWrapper* { return new PhysMemWrapper(mPhysMemSize, mProp); },
          [](PhysMemWrapper* chunk) { delete chunk; })
{
    TLLM_CHECK_WITH_INFO(physMemSize > 0, "PooledPhysMemAllocator: physMemSize must be positive");
    for (size_t size = physMemSize << 1; size > physMemSize && size <= maxChunkSize; size <<= 1)
    {
        auto create = [this, size]() -> PhysMemWrapper* { return new PhysMemWrapper(size, mProp); };
        auto destroy = [](PhysMemWrapper* chunk) { delete chunk; };
        mLargerPools.push_back(std::make_unique<SimplePool<PhysMemWrapper>>(create, destroy));
    }

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

PooledPhysMemAllocator::~PooledPhysMemAllocator()
{
    clear();
}

PooledPhysMemAllocator::PooledPhysMem PooledPhysMemAllocator::acquire()
{
    return mPool.get();
}

PooledPhysMemAllocator::PooledPhysMem PooledPhysMemAllocator::acquire(size_t chunkSize)
{
    if (chunkSize == mPhysMemSize)
    {
        return mPool.get();
    }
    for (size_t k = 0; k < mLargerPools.size(); ++k)
    {
        if (chunkSize == mPhysMemSize << (k + 1))
        {
            return mLargerPools[k]->get();
        }
    }
    TLLM_THROW(
        "PooledPhysMemAllocator: no %zu-byte chunks (unit %zu, max %zu)", chunkSize, mPhysMemSize, maxChunkSize());
}

size_t PooledPhysMemAllocator::chunkSizeAt(size_t offset, size_t remaining) const noexcept
{
    size_t size = maxChunkSize();
    while (size > mPhysMemSize && (offset % size != 0 || size > remaining))
    {
        size >>= 1;
    }
    return size;
}

// ---------------------------------------------------------------------------
// VirtMem
// ---------------------------------------------------------------------------

VirtMem::VirtMem(size_t vmSize, PooledPhysMemAllocator& physMemAllocator, size_t initNumPhysMem)
    : mVmSize(vmSize)
    , mPhysMemAllocator(physMemAllocator)
{
    TLLM_CHECK_DEBUG(vmSize % physMemAllocator.physMemSize() == 0);

    // With larger chunks, align the range to the largest so the chunks land on aligned addresses.
    size_t const alignment
        = physMemAllocator.maxChunkSize() > physMemAllocator.physMemSize() ? physMemAllocator.maxChunkSize() : 0;
    cuCheck(cuMemAddressReserve(&mAddr, vmSize, alignment, 0, 0));

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

void VirtMem::push(PooledPhysMemAllocator::PooledPhysMem handle, size_t size)
{
    CUdeviceptr offset = mAddr + mMappedBytes;
    TLLM_CHECK_DEBUG(mMappedBytes + size <= mVmSize);

    cuCheck(cuMemMap(offset, size, 0, handle->handle(), 0));
    cuCheck(cuMemSetAccess(offset, size, &mAccessDesc, 1));
    mPhysHandles.push_back(std::move(handle));
    mChunkSizes.push_back(size);
    mMappedBytes += size;
}

void VirtMem::pop()
{
    TLLM_CHECK_DEBUG(!mPhysHandles.empty());
    size_t const size = mChunkSizes.back();
    CUdeviceptr offset = mAddr + mMappedBytes - size;
    cuCheck(cuMemUnmap(offset, size));
    mPhysHandles.pop_back(); // PhysMemHandle destructor returns to pool
    mChunkSizes.pop_back();
    mMappedBytes -= size;
}

void VirtMem::extend(size_t numToAdd)
{
    realloc(mUsableBytes + numToAdd * physMemSize());
}

void VirtMem::shrink(size_t numToRemove)
{
    size_t const bytes = numToRemove * physMemSize();
    TLLM_CHECK_DEBUG(bytes <= mUsableBytes);
    realloc(mUsableBytes - bytes);
}

void VirtMem::realloc(size_t numBytes)
{
    size_t const unit = physMemSize();
    size_t const target = divUp(numBytes, unit) * unit;
    if (target > mMappedBytes)
    {
        size_t const old = numPhysMem();
        try
        {
            while (mMappedBytes < target)
            {
                size_t const size = mPhysMemAllocator.chunkSizeAt(mMappedBytes, target - mMappedBytes);
                push(mPhysMemAllocator.acquire(size), size);
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
    else if (!mPhysHandles.empty() && mMappedBytes - mChunkSizes.back() >= target)
    {
        // Unmap only whole chunks above the target: one that straddles it holds live bytes below it.
        cuCheck(cuCtxSynchronize());
        while (!mPhysHandles.empty() && mMappedBytes - mChunkSizes.back() >= target)
        {
            pop();
        }
    }
    mUsableBytes = target;
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
    mUsableBytes = 0;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
