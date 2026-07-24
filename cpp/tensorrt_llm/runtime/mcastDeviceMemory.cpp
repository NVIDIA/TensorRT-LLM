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
#include <cuda.h>

// Rest of includes
#include "mcastDeviceMemory.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <exception>

namespace tensorrt_llm::runtime
{

namespace
{
// An efficient implementation assuming gran is a power of 2
inline size_t roundUp(size_t val, size_t gran)
{
    return (val + gran - 1) & ~(gran - 1);
}

template <typename Cleanup>
void cleanupNoThrow(char const* resource, Cleanup cleanup) noexcept
{
    try
    {
        cleanup();
    }
    catch (std::exception const& error)
    {
        try
        {
            TLLM_LOG_WARNING("Failed to release %s during McastDeviceMemory destruction: %s", resource, error.what());
        }
        catch (...)
        {
            // Logging must not make destruction terminate during static teardown.
        }
    }
    catch (...)
    {
        try
        {
            TLLM_LOG_WARNING("Failed to release %s during McastDeviceMemory destruction", resource);
        }
        catch (...)
        {
            // Logging must not make destruction terminate during static teardown.
        }
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
    , mSignalPadOffset(0)
    , mAllocationSize(0)
    , mMcPtr(0)
    , mUcBasePtr(0)
    , mMcHandle(0)
#if ENABLE_MULTI_DEVICE
    , mGroupComm(MPI_Comm_f2c(mpiCommFortranHandle), false)
#else
    , mGroupComm(nullptr, false)
#endif
    , mUcPtrsDev(nullptr)
    , mSignalPadsDev(nullptr)
    , mMapped(false)
    , mNvlsHandle(nullptr)
{

    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceIdx));
    // Check if the device support multicasting
    int multicast_supported{0};
    TLLM_CU_CHECK(cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, mDeviceIdx));
    if (multicast_supported == 0)
    {
        TLLM_THROW("[McastDeviceMemory] Device does not support multicasting.");
    }

    // From pytorch implementation for alignment
    constexpr size_t kSignalPadAlignment = 16UL;
    mSignalPadOffset = roundUp(mBufSize, kSignalPadAlignment);
    int const world_rank{tensorrt_llm::mpi::MpiComm::session().getRank()};

    TLLM_LOG_DEBUG(
        "[McastDeviceMemory] World Rank: %u, Group Rank: %u, Group size: %u, isMultiNode: %d, "
        "device_idx: %d, Signal pad offset: %zu",
        world_rank, mGroupRank, mGroupSize, mIsMNNvlink, mDeviceIdx, mSignalPadOffset);

    if (mIsMNNvlink)
    {
        // For multi-node, we also need to check if fabric handle is supported
        int fabric_handle_supported{0};
        TLLM_CU_CHECK(cuDeviceGetAttribute(
            &fabric_handle_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, mDeviceIdx));
        if (fabric_handle_supported == 0)
        {
            TLLM_THROW("[McastDeviceMemory] Device does not support fabric handle.");
        }
        allocMnMcastMem(mBufSize);
    }
    else
    {
        allocNvlsMcastMem(mSignalPadOffset + kSIGNAL_PAD_SIZE);
    }
    initializePointerTables();
}

void McastDeviceMemory::initializePointerTables()
{
    // Initialize signal pads and rebuild graph-independent peer pointer tables.
    mSignalPads.resize(mGroupSize);
    for (size_t i = 0; i < mGroupSize; i++)
    {
        mSignalPads[i] = mUcPtrs[i] + mSignalPadOffset;
        if (i == mGroupRank)
        {
            TLLM_CU_CHECK(cuMemsetD8(mSignalPads[i], 0, kSIGNAL_PAD_SIZE));
        }
    }
    if (mSignalPadsDev == nullptr)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&mSignalPadsDev, mGroupSize * sizeof(CUdeviceptr)));
    }
    if (mUcPtrsDev == nullptr)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&mUcPtrsDev, mGroupSize * sizeof(CUdeviceptr)));
    }
    TLLM_CUDA_CHECK(
        cudaMemcpy(mSignalPadsDev, mSignalPads.data(), mGroupSize * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(mUcPtrsDev, mUcPtrs.data(), mGroupSize * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));
}

McastDeviceMemory::~McastDeviceMemory() noexcept
{
    cleanupNoThrow(
        "multicast buffer registration", [this]() { tensorrt_llm::common::unregisterMcastDevMemBuffer(this); });
    if (mIsMNNvlink)
    {
        if (mMapped)
        {
            cleanupNoThrow("MNNVL mappings", [this]() { releaseMnMcastMem(); });
        }
        if (mUcBasePtr != 0)
        {
            cleanupNoThrow("unicast virtual address reservation",
                [this]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(mUcBasePtr, mAllocationSize * mGroupSize)); });
        }
        if (mMcPtr != 0)
        {
            cleanupNoThrow("multicast virtual address reservation",
                [this]() { TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(mMcPtr, mAllocationSize)); });
        }
    }
    else
    {
        // The nvlsfree function will free the handle pointer as well
        cleanupNoThrow("NVLS multicast memory", [this]() { tensorrt_llm::runtime::ipcNvlsFree(mNvlsHandle); });
    }

    if (mSignalPadsDev != nullptr)
    {
        cleanupNoThrow("signal-pad pointer table", [this]() { TLLM_CUDA_CHECK(cudaFree(mSignalPadsDev)); });
    }
    if (mUcPtrsDev != nullptr)
    {
        cleanupNoThrow("unicast pointer table", [this]() { TLLM_CUDA_CHECK(cudaFree(mUcPtrsDev)); });
    }
}

void McastDeviceMemory::allocMnMcastMem(size_t bufSize)
{
    CUmemAllocationHandleType const handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = handle_type;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = mDeviceIdx;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t alloc_granularity{0}, mc_granularity{0};
    TLLM_CU_CHECK(cuMemGetAllocationGranularity(&alloc_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // Round up the buffer size for grnularity
    size_t allocationSize = roundUp(bufSize + kSIGNAL_PAD_SIZE, alloc_granularity);
    CUmulticastObjectProp mcProp = {.numDevices = mGroupSize, .size = allocationSize, .handleTypes = handle_type};
    TLLM_CU_CHECK(cuMulticastGetGranularity(&mc_granularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    allocationSize = roundUp(allocationSize, mc_granularity);
    mcProp.size = allocationSize;
    if (mAllocationSize == 0)
    {
        mAllocationSize = allocationSize;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(mAllocationSize == allocationSize,
            "[McastDeviceMemory] Restored allocation layout differs from the retained virtual-address layout.");
    }
    mUcHandles.assign(mGroupSize, 0);
    // Allocates local gpu memory
    TLLM_CU_CHECK(cuMemCreate(&(mUcHandles[mGroupRank]), mAllocationSize, &prop, 0));

    // Connect peer UC handles
    CUmemFabricHandle* exphndl{nullptr};
    CUmemFabricHandle myhndl;
    TLLM_CU_CHECK(cuMemExportToShareableHandle(&myhndl, mUcHandles[mGroupRank], CU_MEM_HANDLE_TYPE_FABRIC, 0));
    // All gather
    TLLM_CUDA_CHECK(cudaMallocHost(&exphndl, mGroupSize * sizeof(CUmemFabricHandle)));
    std::memcpy(exphndl + mGroupRank, &myhndl, sizeof(CUmemFabricHandle));
    mGroupComm.allgather(exphndl + mGroupRank, exphndl, sizeof(CUmemFabricHandle), mpi::MpiType::kCHAR);
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());

    for (uint32_t p = 0; p < mGroupSize; p++)
    {
        if (p != mGroupRank)
        {
            TLLM_CU_CHECK(cuMemImportFromShareableHandle(
                &mUcHandles[p], reinterpret_cast<void*>(&exphndl[p]), CU_MEM_HANDLE_TYPE_FABRIC));
        }
    }
    TLLM_CUDA_CHECK(cudaFreeHost(exphndl));

    // Initialize multicasting
    CUmemFabricHandle* fabric_handle;
    TLLM_CUDA_CHECK(cudaMallocHost(&fabric_handle, sizeof(CUmemFabricHandle)));
    if (mGroupRank == 0)
    {
        TLLM_CU_CHECK(cuMulticastCreate(&mMcHandle, &mcProp));
        TLLM_CU_CHECK(cuMemExportToShareableHandle(fabric_handle, mMcHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    // Broadcast
    mGroupComm.bcast(fabric_handle, sizeof(CUmemFabricHandle), mpi::MpiType::kCHAR, 0);
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    if (mGroupRank != 0)
    {
        TLLM_CU_CHECK(cuMemImportFromShareableHandle(&mMcHandle, fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));
    }
    TLLM_CU_CHECK(cuMulticastAddDevice(mMcHandle, mDeviceIdx));
    TLLM_CUDA_CHECK(cudaFreeHost(fabric_handle));

    // Bind memory addresses
    mUcPtrs.resize(mGroupSize);
    if (mUcBasePtr == 0)
    {
        TLLM_CU_CHECK(cuMemAddressReserve(&mUcBasePtr, mAllocationSize * mGroupSize, mc_granularity, 0ULL, 0));
    }
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = mDeviceIdx;

    for (uint32_t i = 0; i < mGroupSize; i++)
    {
        mUcPtrs[i] = mUcBasePtr + (mAllocationSize * i);
        TLLM_CU_CHECK(cuMemMap(mUcPtrs[i], mAllocationSize, 0, mUcHandles[i], 0));
    }
    TLLM_CU_CHECK(cuMemSetAccess(mUcBasePtr, mAllocationSize * mGroupSize, &accessDesc, 1));

    // Bind MC Pointers
    if (mMcPtr == 0)
    {
        TLLM_CU_CHECK(cuMemAddressReserve(&mMcPtr, mAllocationSize, mc_granularity, 0ULL, 0));
    }
    TLLM_CU_CHECK(cuMemMap(mMcPtr, mAllocationSize, 0, mMcHandle, 0));
    TLLM_CU_CHECK(cuMemSetAccess(mMcPtr, mAllocationSize, &accessDesc, 1));

    TLLM_CU_CHECK(cuMulticastBindMem(mMcHandle, 0, mUcHandles[mGroupRank], 0 /*memOffset*/, mAllocationSize, 0));
    mMapped = true;
}

void McastDeviceMemory::releaseMnMcastMem()
{
    TLLM_CU_CHECK(cuMulticastUnbind(mMcHandle, mDeviceIdx, 0, mAllocationSize));
    TLLM_CU_CHECK(cuMemUnmap(mMcPtr, mAllocationSize));
    for (uint32_t rank = 0; rank < mGroupSize; rank++)
    {
        TLLM_CU_CHECK(cuMemUnmap(mUcPtrs[rank], mAllocationSize));
    }
    TLLM_CU_CHECK(cuMemRelease(mMcHandle));
    mMcHandle = 0;
    for (auto& handle : mUcHandles)
    {
        TLLM_CU_CHECK(cuMemRelease(handle));
        handle = 0;
    }
    mMapped = false;
}

void McastDeviceMemory::checkpointPrepare()
{
    TLLM_CHECK_WITH_INFO(
        mIsMNNvlink, "[McastDeviceMemory] Stable-VA checkpointing is only supported for fabric-backed MNNVL memory.");
    if (!mMapped)
    {
        return;
    }
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceIdx));
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    mGroupComm.barrier();
    releaseMnMcastMem();
    mGroupComm.barrier();
}

void McastDeviceMemory::checkpointRestore(int64_t mpiCommFortranHandle)
{
    TLLM_CHECK_WITH_INFO(
        mIsMNNvlink, "[McastDeviceMemory] Stable-VA checkpointing is only supported for fabric-backed MNNVL memory.");
    if (mMapped)
    {
        return;
    }
#if ENABLE_MULTI_DEVICE
    mGroupComm = tensorrt_llm::mpi::MpiComm(MPI_Comm_f2c(mpiCommFortranHandle), false);
#else
    mGroupComm = tensorrt_llm::mpi::MpiComm(nullptr, false);
#endif
    TLLM_CHECK_WITH_INFO(
        mGroupComm.getRank() == static_cast<int>(mGroupRank) && mGroupComm.getSize() == static_cast<int>(mGroupSize),
        "[McastDeviceMemory] Restore communicator does not match the original rank or group size.");
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceIdx));
    allocMnMcastMem(mBufSize);
    initializePointerTables();
    mGroupComm.barrier();
}

void McastDeviceMemory::allocNvlsMcastMem(size_t bufSize)
{
    // Get the world ranks for ranks in this group
    auto ranks_ = tensorrt_llm::mpi::getWorldRanks(mGroupComm);
    std::set<int> ranks(ranks_.begin(), ranks_.end());
    // Reuse existing implementation
    mNvlsHandle = tensorrt_llm::runtime::ipcNvlsAllocate(bufSize, ranks);
    mMcHandle = mNvlsHandle->mc_handle;
    mMcPtr = mNvlsHandle->mc_va;
    mUcPtrs = mNvlsHandle->ipc_uc_vas;
    mUcHandles = mNvlsHandle->ipc_uc_handles;
    mMapped = true;
}

} // namespace tensorrt_llm::runtime
