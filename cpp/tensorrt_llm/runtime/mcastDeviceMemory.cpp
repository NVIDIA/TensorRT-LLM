/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mcastDeviceMemory.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/mcastDevMemUtils.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <exception>
#include <set>
#include <vector>

namespace tensorrt_llm::runtime
{

namespace
{

constexpr size_t kSignalPadAlignment = 16UL;

bool isPowerOfTwo(size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

size_t roundUp(size_t value, size_t granularity)
{
    return (value + granularity - 1) & ~(granularity - 1);
}

void checkDriverCleanup(CUresult result, char const* operation) noexcept
{
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED)
    {
        TLLM_LOG_WARNING("[McastDeviceMemory] CUDA driver cleanup operation %s failed with error %d", operation,
            static_cast<int>(result));
    }
}

} // namespace

McastDeviceMemory::McastDeviceMemory(
    size_t bufSize, uint32_t groupSize, uint32_t groupRank, int deviceIdx, bool mnNvlink, int64_t mpiCommFortranHandle)
    : mIsMNNvlink(mnNvlink)
    , mDeviceIdx(deviceIdx)
    , mGroupSize(groupSize)
    , mGroupRank(groupRank)
    , mBufSize(bufSize)
#if ENABLE_MULTI_DEVICE
    , mGroupComm(MPI_Comm_f2c(mpiCommFortranHandle), false)
#else
    , mGroupComm(nullptr, false)
#endif
{
    try
    {
        size_t const requestedBufferSize = mBufSize;
        mCommSize = mGroupComm.getSize();
        mCommRank = mGroupComm.getRank();
        mWorldRank = tensorrt_llm::mpi::MpiComm::session().getRank();
        TLLM_CHECK_WITH_INFO(mBufSize > 0, "[McastDeviceMemory] Buffer size must be positive");
        TLLM_CHECK_WITH_INFO(mCommSize > 0, "[McastDeviceMemory] Communicator must contain at least one rank");
        TLLM_CHECK_WITH_INFO(mGroupSize == static_cast<uint32_t>(mCommSize),
            "[McastDeviceMemory] Supplied group size %u does not match communicator size %d", mGroupSize, mCommSize);
        TLLM_CHECK_WITH_INFO(mGroupRank < mGroupSize, "[McastDeviceMemory] Group rank %u is outside group size %u",
            mGroupRank, mGroupSize);
        TLLM_CHECK_WITH_INFO(mGroupRank == static_cast<uint32_t>(mCommRank),
            "[McastDeviceMemory] Supplied group rank %u does not match communicator rank %d", mGroupRank, mCommRank);

        uint64_t const localRequest = static_cast<uint64_t>(requestedBufferSize);
        std::vector<uint64_t> requests(mGroupSize);
        mGroupComm.allgather(&localRequest, requests.data(), 1, mpi::MpiType::kUINT64);
        for (uint64_t const request : requests)
        {
            TLLM_CHECK_WITH_INFO(request == localRequest,
                "[McastDeviceMemory] All ranks must allocate the same logical size; local=%zu, peer=%zu",
                static_cast<size_t>(localRequest), static_cast<size_t>(request));
        }

        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceIdx));
        int multicastSupported{0};
        TLLM_CU_CHECK(cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, mDeviceIdx));
        TLLM_CHECK_WITH_INFO(
            multicastSupported != 0, "[McastDeviceMemory] Device %d does not support multicast memory", mDeviceIdx);

        if (mIsMNNvlink)
        {
            int fabricHandleSupported{0};
            TLLM_CU_CHECK(cuDeviceGetAttribute(
                &fabricHandleSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, mDeviceIdx));
            TLLM_CHECK_WITH_INFO(fabricHandleSupported != 0,
                "[McastDeviceMemory] Device %d does not support fabric handles", mDeviceIdx);
        }

        mSignalPadOffset = roundUp(requestedBufferSize, kSignalPadAlignment);

        if (mIsMNNvlink)
        {
            allocMnMcastMem(requestedBufferSize);
        }
        else
        {
            mAllocationSize = mSignalPadOffset + kSIGNAL_PAD_SIZE;
            allocNvlsMcastMem(mAllocationSize);
        }

        TLLM_LOG_DEBUG(
            "[McastDeviceMemory] World rank: %d, group rank: %u, group size: %u, isMultiNode: %d, "
            "device index: %d, requested buffer size: %zu, usable buffer size: %zu, allocation size: %zu, "
            "signal pad offset: %zu",
            mWorldRank, mGroupRank, mGroupSize, mIsMNNvlink, mDeviceIdx, requestedBufferSize, mBufSize, mAllocationSize,
            mSignalPadOffset);

        initializePointerArrays();
    }
    catch (...)
    {
        cleanup();
        throw;
    }
}

McastDeviceMemory::~McastDeviceMemory() noexcept
{
    cleanup();
}

void* McastDeviceMemory::getUnicastPtr(uint32_t rank) const
{
    TLLM_CHECK_WITH_INFO(
        rank < mGroupSize, "[McastDeviceMemory] Unicast rank %u is outside group size %u", rank, mGroupSize);
    return reinterpret_cast<void*>(mUcPtrs[rank]);
}

void* McastDeviceMemory::getMulticastPtr() const
{
    return reinterpret_cast<void*>(mMcPtr);
}

void McastDeviceMemory::allocMnMcastMem(size_t bufSize)
{
    TLLM_CHECK_WITH_INFO(bufSize == mBufSize,
        "[McastDeviceMemory] Internal logical size mismatch: expected %zu, got %zu", mBufSize, bufSize);

    CUmemAllocationHandleType constexpr kHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop{};
    prop.requestedHandleTypes = kHandleType;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = mDeviceIdx;

    CUmulticastObjectProp mcProp{};
    int gpuDirectRdmaSupported{0};
    TLLM_CU_CHECK(cuDeviceGetAttribute(
        &gpuDirectRdmaSupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, mDeviceIdx));
    prop.allocFlags.gpuDirectRDMACapable = gpuDirectRdmaSupported != 0;

    TLLM_CU_CHECK(cuMemGetAllocationGranularity(&mAllocationGranularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    TLLM_CHECK_WITH_INFO(isPowerOfTwo(mAllocationGranularity),
        "[McastDeviceMemory] CUDA allocation granularity must be a nonzero power of two, got %zu",
        mAllocationGranularity);
    size_t const requestedPhysicalSize = mSignalPadOffset + kSIGNAL_PAD_SIZE;
    mAllocationSize = roundUp(requestedPhysicalSize, mAllocationGranularity);

    mcProp.numDevices = mGroupSize;
    mcProp.size = mAllocationSize;
    mcProp.handleTypes = kHandleType;
    TLLM_CU_CHECK(
        cuMulticastGetGranularity(&mMulticastRecommendedGranularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    TLLM_CHECK_WITH_INFO(isPowerOfTwo(mMulticastRecommendedGranularity),
        "[McastDeviceMemory] Multicast recommended granularity must be a nonzero power of two, got %zu",
        mMulticastRecommendedGranularity);
    mAllocationSize = roundUp(mAllocationSize, mMulticastRecommendedGranularity);
    mcProp.size = mAllocationSize;

    // Fabric allocations are rounded to the queried multicast granularity (512 MiB on GB200). Expose the
    // alignment slack as usable workspace and keep the signal pad at the end, so Python does not need a separate
    // workspace growth heuristic.
    mBufSize = mAllocationSize - kSIGNAL_PAD_SIZE;
    mSignalPadOffset = mBufSize;

    std::vector<uint64_t> alignedProperties(mGroupSize * 3);
    uint64_t const localAlignedProperties[3] = {static_cast<uint64_t>(mAllocationSize),
        static_cast<uint64_t>(mAllocationGranularity), static_cast<uint64_t>(mMulticastRecommendedGranularity)};
    mGroupComm.allgather(localAlignedProperties, alignedProperties.data(), 3, mpi::MpiType::kUINT64);
    for (size_t index = 0; index < alignedProperties.size(); ++index)
    {
        TLLM_CHECK_WITH_INFO(alignedProperties[index] == localAlignedProperties[index % 3],
            "[McastDeviceMemory] Ranks computed inconsistent aligned allocation properties");
    }

    mUcHandles.resize(mGroupSize);
    mUcHandleAcquired.resize(mGroupSize, 0);
    mUcMapped.resize(mGroupSize, 0);

    CUmemFabricHandle localFabricHandle{};
    TLLM_CU_CHECK(cuMemCreate(&mUcHandles[mGroupRank], mAllocationSize, &prop, 0));
    mUcHandleAcquired[mGroupRank] = 1;
    TLLM_CU_CHECK(
        cuMemExportToShareableHandle(&localFabricHandle, mUcHandles[mGroupRank], CU_MEM_HANDLE_TYPE_FABRIC, 0));

    std::vector<CUmemFabricHandle> fabricHandles(mGroupSize);
    mGroupComm.allgather(&localFabricHandle, fabricHandles.data(), sizeof(localFabricHandle), mpi::MpiType::kCHAR);
    for (uint32_t rank = 0; rank < mGroupSize; ++rank)
    {
        if (rank != mGroupRank)
        {
            TLLM_CU_CHECK(cuMemImportFromShareableHandle(
                &mUcHandles[rank], static_cast<void*>(&fabricHandles[rank]), CU_MEM_HANDLE_TYPE_FABRIC));
            mUcHandleAcquired[rank] = 1;
        }
    }

    CUmemFabricHandle multicastFabricHandle{};
    if (mGroupRank == 0)
    {
        TLLM_CU_CHECK(cuMulticastCreate(&mMcHandle, &mcProp));
        mMcHandleAcquired = true;
        TLLM_CU_CHECK(cuMemExportToShareableHandle(&multicastFabricHandle, mMcHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    mGroupComm.bcast(&multicastFabricHandle, sizeof(multicastFabricHandle), mpi::MpiType::kCHAR, 0);

    if (mGroupRank != 0)
    {
        TLLM_CU_CHECK(cuMemImportFromShareableHandle(
            &mMcHandle, static_cast<void*>(&multicastFabricHandle), CU_MEM_HANDLE_TYPE_FABRIC));
        mMcHandleAcquired = true;
    }
    TLLM_CU_CHECK(cuMulticastAddDevice(mMcHandle, mDeviceIdx));

    CUmemAccessDesc accessDesc{};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = mDeviceIdx;

    mUcReservationSize = mAllocationSize * mGroupSize;
    CUdeviceptr unicastBase{0};
    TLLM_CU_CHECK(cuMemAddressReserve(&unicastBase, mUcReservationSize, mMulticastRecommendedGranularity, 0ULL, 0));
    mUcPtrBase = unicastBase;
    mUcPtrs.resize(mGroupSize);
    for (uint32_t rank = 0; rank < mGroupSize; ++rank)
    {
        CUdeviceptr const rankPtr = mUcPtrBase + mAllocationSize * rank;
        TLLM_CU_CHECK(cuMemMap(rankPtr, mAllocationSize, 0, mUcHandles[rank], 0));
        mUcMapped[rank] = 1;
        mUcPtrs[rank] = rankPtr;
    }
    TLLM_CU_CHECK(cuMemSetAccess(mUcPtrBase, mUcReservationSize, &accessDesc, 1));

    CUdeviceptr multicastPtr{0};
    TLLM_CU_CHECK(cuMemAddressReserve(&multicastPtr, mAllocationSize, mMulticastRecommendedGranularity, 0ULL, 0));
    mMcPtr = multicastPtr;
    mMcAddressReserved = true;
    TLLM_CU_CHECK(cuMemMap(mMcPtr, mAllocationSize, 0, mMcHandle, 0));
    mMcMapped = true;
    TLLM_CU_CHECK(cuMemSetAccess(mMcPtr, mAllocationSize, &accessDesc, 1));

    TLLM_CU_CHECK(cuMulticastBindMem(mMcHandle, 0, mUcHandles[mGroupRank], 0, mAllocationSize, 0));
    mMcBound = true;
}

void McastDeviceMemory::allocNvlsMcastMem(size_t bufSize)
{
    auto const ranksVector = tensorrt_llm::mpi::getWorldRanks(mGroupComm);
    std::set<int> const ranks(ranksVector.begin(), ranksVector.end());
    mNvlsHandle = tensorrt_llm::runtime::ipcNvlsAllocate(bufSize, ranks);
    TLLM_CHECK_WITH_INFO(mNvlsHandle != nullptr, "[McastDeviceMemory] ipcNvlsAllocate returned a null handle");
    mMcHandle = mNvlsHandle->mc_handle;
    mMcPtr = mNvlsHandle->mc_va;
    mUcPtrs = mNvlsHandle->ipc_uc_vas;
    mUcHandles = mNvlsHandle->ipc_uc_handles;
}

void McastDeviceMemory::initializePointerArrays()
{
    TLLM_CHECK_WITH_INFO(mUcPtrs.size() == mGroupSize, "[McastDeviceMemory] Expected %u unicast pointers, got %zu",
        mGroupSize, mUcPtrs.size());
    mSignalPads.resize(mGroupSize);
    for (size_t rank = 0; rank < mGroupSize; ++rank)
    {
        mSignalPads[rank] = mUcPtrs[rank] + mSignalPadOffset;
        if (rank == mGroupRank)
        {
            TLLM_CU_CHECK(cuMemsetD8(mSignalPads[rank], 0, kSIGNAL_PAD_SIZE));
        }
    }

    size_t const pointerArraySize = mGroupSize * sizeof(CUdeviceptr);
    TLLM_CUDA_CHECK(cudaMalloc(&mSignalPadsDev, pointerArraySize));
    TLLM_CUDA_CHECK(cudaMalloc(&mUcPtrsDev, pointerArraySize));
    TLLM_CUDA_CHECK(cudaMemcpy(mSignalPadsDev, mSignalPads.data(), pointerArraySize, cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(mUcPtrsDev, mUcPtrs.data(), pointerArraySize, cudaMemcpyHostToDevice));
}

// Invoked from the destructor and from constructor rollback. Must not throw: a throw
// during destruction calls std::terminate. Use warn-only CUDA checks and swallow errors.
void McastDeviceMemory::cleanup() noexcept
{
    int previousDevice{-1};
    bool const restoreDevice = cudaGetDevice(&previousDevice) == cudaSuccess;
    TLLM_CUDA_CHECK_WARN(cudaSetDevice(mDeviceIdx));

    try
    {
        tensorrt_llm::common::unregisterMcastDevMemBuffer(this);
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_WARNING(
            "[McastDeviceMemory] Failed to unregister multicast pointers during cleanup: %s", error.what());
    }
    catch (...)
    {
        TLLM_LOG_WARNING("[McastDeviceMemory] Failed to unregister multicast pointers during cleanup");
    }

    if (mSignalPadsDev != nullptr || mUcPtrsDev != nullptr || mMcPtr != 0 || mUcPtrBase != 0 || mNvlsHandle != nullptr)
    {
        TLLM_CUDA_CHECK_WARN(cudaDeviceSynchronize());
    }

    if (mSignalPadsDev != nullptr)
    {
        TLLM_CUDA_CHECK_WARN(cudaFree(mSignalPadsDev));
        mSignalPadsDev = nullptr;
    }
    if (mUcPtrsDev != nullptr)
    {
        TLLM_CUDA_CHECK_WARN(cudaFree(mUcPtrsDev));
        mUcPtrsDev = nullptr;
    }

    if (mIsMNNvlink)
    {
        if (mMcMapped)
        {
            checkDriverCleanup(cuMemUnmap(mMcPtr, mAllocationSize), "cuMemUnmap(multicast)");
            mMcMapped = false;
        }
        if (mMcBound && mMcHandleAcquired)
        {
            checkDriverCleanup(cuMulticastUnbind(mMcHandle, mDeviceIdx, 0, mAllocationSize), "cuMulticastUnbind");
            mMcBound = false;
        }
        if (mMcAddressReserved)
        {
            checkDriverCleanup(cuMemAddressFree(mMcPtr, mAllocationSize), "cuMemAddressFree(multicast)");
            mMcAddressReserved = false;
            mMcPtr = 0;
        }
        if (mMcHandleAcquired)
        {
            checkDriverCleanup(cuMemRelease(mMcHandle), "cuMemRelease(multicast)");
            mMcHandleAcquired = false;
            mMcHandle = 0;
        }

        for (size_t rank = mUcMapped.size(); rank > 0; --rank)
        {
            size_t const index = rank - 1;
            if (mUcMapped[index] != 0)
            {
                checkDriverCleanup(cuMemUnmap(mUcPtrs[index], mAllocationSize), "cuMemUnmap(unicast)");
                mUcMapped[index] = 0;
            }
        }
        for (size_t rank = mUcHandleAcquired.size(); rank > 0; --rank)
        {
            size_t const index = rank - 1;
            if (mUcHandleAcquired[index] != 0)
            {
                checkDriverCleanup(cuMemRelease(mUcHandles[index]), "cuMemRelease(unicast)");
                mUcHandleAcquired[index] = 0;
            }
        }
        if (mUcPtrBase != 0)
        {
            checkDriverCleanup(cuMemAddressFree(mUcPtrBase, mUcReservationSize), "cuMemAddressFree(unicast)");
            mUcPtrBase = 0;
            mUcReservationSize = 0;
        }
    }
    else if (mNvlsHandle != nullptr)
    {
        try
        {
            tensorrt_llm::runtime::ipcNvlsFree(mNvlsHandle);
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_WARNING("[McastDeviceMemory] ipcNvlsFree failed during cleanup: %s", error.what());
        }
        catch (...)
        {
            TLLM_LOG_WARNING("[McastDeviceMemory] ipcNvlsFree failed during cleanup");
        }
        mNvlsHandle = nullptr;
    }

    if (restoreDevice)
    {
        TLLM_CUDA_CHECK_WARN(cudaSetDevice(previousDevice));
    }
}

} // namespace tensorrt_llm::runtime
