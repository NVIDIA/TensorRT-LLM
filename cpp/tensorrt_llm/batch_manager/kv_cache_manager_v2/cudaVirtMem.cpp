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
#include "kv_cache_manager_v2/localityDomain.h"
#include "kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <numeric>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// PooledPhysMemAllocator
// ---------------------------------------------------------------------------

static bool isUnsupportedAllocationError(CUresult err) noexcept
{
    return err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED || err == CUDA_ERROR_INVALID_DEVICE
        || err == CUDA_ERROR_INVALID_VALUE;
}

static bool isPropSupported(CUmemAllocationProp const& prop)
{
    CUmemGenericAllocationHandle handle;
    CUresult err = cuMemCreate(&handle, 2ULL << 20, &prop, 0);
    if (isUnsupportedAllocationError(err))
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

PooledPhysMemAllocator::PooledPhysMemAllocator(size_t physMemSize, int numLocalityDomains)
    : mPhysMemSize(physMemSize)
{
    TLLM_CHECK_WITH_INFO(numLocalityDomains >= 1, "PooledPhysMemAllocator: need at least one locality domain");

    // Get current device.
    cuCheck(cuCtxGetDevice(&mDeviceId));

    mProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    mProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    mProp.location.id = mDeviceId;

    if (numLocalityDomains > 1 && localizationMode() != LocalizationMode::MOCK)
    {
        selectLocalizedProp(numLocalityDomains);
    }
    else
    {
        selectDefaultProp();
    }

    mPools.reserve(LocalityDomainId{numLocalityDomains});
    for (LocalityDomainId localityDomainId{0}; localityDomainId < numLocalityDomains; ++localityDomainId)
    {
        mPools.push_back(std::make_unique<SimplePool<PhysMemWrapper>>([this, localityDomainId]() -> PhysMemWrapper*
            { return createChunk(localityDomainId); },
            [](PhysMemWrapper* chunk) { delete chunk; }));
    }
}

void PooledPhysMemAllocator::selectDefaultProp()
{
    // Build the best allocation property. Prefer shareable handle types so
    // UCX cuda_ipc can export the memory for intra-node zero-copy transfers:
    // FABRIC (MNNVL) > POSIX_FILE_DESCRIPTOR (pidfd-based, UCX >= 1.22) > NONE,
    // and within each handle type prefer gpuDirectRDMACapable over not.
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

// Same preference order as the default property, but every domain has to accept the candidate and
// the chunk size is rounded up to a multiple of every domain's granularity.
void PooledPhysMemAllocator::selectLocalizedProp(int numLocalityDomains)
{
    mLocalization.emplace();
    if (!mLocalization->supportsMemoryLocalization())
    {
        throw std::runtime_error("PooledPhysMemAllocator: locality domain memory localization is not supported");
    }

    constexpr CUmemAllocationHandleType kHandleTypePreference[]
        = {CU_MEM_HANDLE_TYPE_FABRIC, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, CU_MEM_HANDLE_TYPE_NONE};
    for (auto const handleType : kHandleTypePreference)
    {
        for (bool const rdmaCapable : {true, false})
        {
            mProp.requestedHandleTypes = handleType;
            mProp.allocFlags.gpuDirectRDMACapable = rdmaCapable ? 1 : 0;

            size_t granularity = 1;
            bool supported = true;
            for (LocalityDomainId localityDomainId{0}; localityDomainId < numLocalityDomains && supported;
                 ++localityDomainId)
            {
                size_t domainGranularity = 0;
                CUresult const err = mLocalization->tryGetLocalizedAllocationGranularity(
                    &domainGranularity, localityDomainId.value(), handleType, rdmaCapable, mProp.allocFlags.usage);
                if (isUnsupportedAllocationError(err))
                {
                    supported = false;
                    break;
                }
                cuCheck(err);
                TLLM_CHECK_WITH_INFO(domainGranularity > 0, "CUDA returned an invalid allocation granularity");
                granularity = std::lcm(granularity, domainGranularity);
            }
            if (!supported)
            {
                continue;
            }

            size_t const probeSize = roundUp(size_t{2} << 20, granularity);
            for (LocalityDomainId localityDomainId{0}; localityDomainId < numLocalityDomains && supported;
                 ++localityDomainId)
            {
                CUmemGenericAllocationHandle handle{};
                CUresult const err = mLocalization->tryCreateLocalizedAllocationHandle(
                    &handle, probeSize, localityDomainId.value(), handleType, rdmaCapable, mProp.allocFlags.usage);
                if (isUnsupportedAllocationError(err))
                {
                    supported = false;
                    break;
                }
                cuCheck(err);
                cuMemRelease(handle);
            }
            if (supported)
            {
                mPhysMemSize = roundUp(mPhysMemSize, granularity);
                return;
            }
        }
    }
    throw std::runtime_error(
        "PooledPhysMemAllocator: no physical memory allocation property is supported by every locality domain");
}

PhysMemWrapper* PooledPhysMemAllocator::createChunk(LocalityDomainId localityDomainId)
{
    if (!mLocalization.has_value())
    {
        return new PhysMemWrapper(mPhysMemSize, mProp);
    }
    CUmemGenericAllocationHandle handle{};
    cuCheck(mLocalization->tryCreateLocalizedAllocationHandle(&handle, mPhysMemSize, localityDomainId.value(),
        mProp.requestedHandleTypes, mProp.allocFlags.gpuDirectRDMACapable != 0, mProp.allocFlags.usage));
    return new PhysMemWrapper(handle);
}

PooledPhysMemAllocator::~PooledPhysMemAllocator()
{
    clear();
}

PooledPhysMemAllocator::PooledPhysMem PooledPhysMemAllocator::acquire(LocalityDomainId localityDomainId)
{
    return mPools.at(localityDomainId)->get();
}

void PooledPhysMemAllocator::clear()
{
    for (auto& pool : mPools)
    {
        pool->clear();
    }
}

// ---------------------------------------------------------------------------
// VirtMem
// ---------------------------------------------------------------------------

VirtMem::VirtMem(size_t vmSize, PooledPhysMemAllocator& physMemAllocator, size_t initNumPhysMem)
    : mVmSize(vmSize)
    , mPhysMemAllocator(physMemAllocator)
    , mMappings(LocalityDomainId{physMemAllocator.numLocalityDomains()})
{
    TLLM_CHECK_DEBUG(vmSize % physMemAllocator.physMemSize() == 0);

    cuCheck(cuMemAddressReserve(&mAddr, vmSize, 0, 0, 0));

    mAccessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    mAccessDesc.location.id = physMemAllocator.deviceId();
    mAccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    try
    {
        extend(initNumPhysMem, LocalityDomainId{0}, size_t{0});
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

void VirtMem::push(PooledPhysMemAllocator::PooledPhysMem handle, size_t offset, LocalityDomainId localityDomainId)
{
    size_t physSize = mPhysMemAllocator.physMemSize();
    CUdeviceptr ptr = mAddr + offset;
    TLLM_CHECK_DEBUG(offset + physSize <= mVmSize);

    cuCheck(cuMemMap(ptr, physSize, 0, handle->handle(), 0));
    cuCheck(cuMemSetAccess(ptr, physSize, &mAccessDesc, 1));
    mMappings[localityDomainId].push_back(Mapping{offset, std::move(handle)});
    mMappedOffsets.insert(offset);
}

void VirtMem::pop(LocalityDomainId localityDomainId)
{
    auto& mappings = mMappings[localityDomainId];
    TLLM_CHECK_DEBUG(!mappings.empty());
    size_t physSize = mPhysMemAllocator.physMemSize();
    size_t const offset = mappings.back().offset;
    cuCheck(cuMemUnmap(mAddr + offset, physSize));
    mMappedOffsets.erase(offset);
    mappings.pop_back(); // PhysMemHandle destructor returns to pool
}

void VirtMem::extend(size_t numToAdd, LocalityDomainId localityDomainId, std::optional<size_t> offset)
{
    TLLM_CHECK_WITH_INFO(localityDomainId >= 0 && localityDomainId < numLocalityDomains(),
        "VirtMem::extend: locality domain %d out of range", localityDomainId.value());
    size_t const physSize = mPhysMemAllocator.physMemSize();
    if (!offset.has_value())
    {
        TLLM_CHECK_WITH_INFO(numLocalityDomains() == 1, "VirtMem::extend: offset is required with locality domains");
        offset = mappedBytes();
    }
    size_t const begin = *offset;
    size_t const end = begin + numToAdd * physSize;
    if (begin % physSize != 0 || end > mVmSize)
    {
        throw LogicError("VirtMem::extend: mapping is outside or misaligned with the reservation");
    }
    for (size_t pos = begin; pos < end; pos += physSize)
    {
        if (mMappedOffsets.count(pos) != 0)
        {
            throw LogicError("VirtMem::extend: mapping overlaps an existing mapping");
        }
    }

    size_t const old = numPhysMem(localityDomainId);
    try
    {
        for (size_t pos = begin; pos < end; pos += physSize)
        {
            push(mPhysMemAllocator.acquire(localityDomainId), pos, localityDomainId);
        }
    }
    catch (...)
    {
        // Rollback: remove any newly-mapped chunks.
        while (numPhysMem(localityDomainId) > old)
        {
            pop(localityDomainId);
        }
        throw;
    }
}

void VirtMem::shrink(size_t numToRemove, LocalityDomainId localityDomainId)
{
    cuCheck(cuCtxSynchronize());
    for (size_t i = 0; i < numToRemove; ++i)
    {
        pop(localityDomainId);
    }
}

void VirtMem::realloc(size_t numBytes, LocalityDomainId localityDomainId)
{
    size_t physSize = mPhysMemAllocator.physMemSize();
    size_t required = divUp(numBytes, physSize);
    size_t current = numPhysMem(localityDomainId);
    if (required > current)
    {
        extend(required - current, localityDomainId);
    }
    else if (required < current)
    {
        shrink(current - required, localityDomainId);
    }
}

std::vector<std::pair<size_t, size_t>> VirtMem::mappedRanges() const
{
    size_t const physSize = mPhysMemAllocator.physMemSize();
    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t const offset : mMappedOffsets)
    {
        if (!ranges.empty() && ranges.back().second == offset)
        {
            ranges.back().second = offset + physSize;
        }
        else
        {
            ranges.emplace_back(offset, offset + physSize);
        }
    }
    return ranges;
}

void VirtMem::destroy()
{
    if (mVmSize == 0)
    {
        return;
    }
    cuCheck(cuCtxSynchronize());
    for (LocalityDomainId localityDomainId{0}; localityDomainId < numLocalityDomains(); ++localityDomainId)
    {
        while (numPhysMem(localityDomainId) != 0)
        {
            pop(localityDomainId);
        }
    }
    cuCheck(cuMemAddressFree(mAddr, mVmSize));
    mAddr = 0;
    mVmSize = 0;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
