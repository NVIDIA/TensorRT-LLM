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

PooledPhysMemAllocator::PooledPhysMemAllocator(size_t physMemSize)
    : mPhysMemSize(physMemSize)
    , mPool([this]() -> PhysMemWrapper* { return new PhysMemWrapper(mPhysMemSize, mProp); },
          [](PhysMemWrapper* chunk) { delete chunk; })
{
    // Get current device.
    cuCheck(cuCtxGetDevice(&mDeviceId));

    // Build the best allocation property.
    mProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    mProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    mProp.location.id = mDeviceId;
    mProp.allocFlags.gpuDirectRDMACapable = 1;
    mProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    if (!isPropSupported(mProp))
    {
        mProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
        if (!isPropSupported(mProp))
        {
            mProp.allocFlags.gpuDirectRDMACapable = 0;
            if (!isPropSupported(mProp))
            {
                throw std::runtime_error("PooledPhysMemAllocator: no supported physical memory allocation property");
            }
        }
    }
}

PooledPhysMemAllocator::~PooledPhysMemAllocator()
{
    mPool.clear();
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

    cuCheck(cuMemAddressReserve(&mAddr, vmSize, 0, 0, 0));

    mAccessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    mAccessDesc.location.id = physMemAllocator.deviceId();
    mAccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    extend(initNumPhysMem);
}

VirtMem::~VirtMem() noexcept
{
    try
    {
        destroy();
    }
    catch (...)
    {
        // Destructors cannot surface CUDA cleanup failures. Explicit destroy()
        // still reports them to match Python VirtMem.destroy().
    }
}

void VirtMem::push(PooledPhysMemAllocator::PooledPhysMem handle)
{
    size_t physSize = mPhysMemAllocator.physMemSize();
    CUdeviceptr offset = mAddr + physSize * static_cast<size_t>(mPhysHandles.size());
    TLLM_CHECK_DEBUG(physSize * (mPhysHandles.size() + 1) <= mVmSize);

    cuCheck(cuMemMap(offset, physSize, 0, handle->handle(), 0));
    cuCheck(cuMemSetAccess(offset, physSize, &mAccessDesc, 1));
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
