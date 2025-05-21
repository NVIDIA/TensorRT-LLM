/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace tensorrt_llm::runtime
{

namespace
{
#define CUCHECK(cmd)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult retval = cmd;                                                                                         \
        if (retval != CUDA_SUCCESS)                                                                                    \
        {                                                                                                              \
            const char* error_string;                                                                                  \
            cuGetErrorString(retval, &error_string);                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, error_string);                               \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// An efficient implementation assuming gran is a power of 2
inline size_t roundUp(size_t val, size_t gran)
{
    return (val + gran - 1) & ~(gran - 1);
}
} // namespace

McastDeviceMemory::McastDeviceMemory(
    size_t bufSize, uint32_t groupSize, uint32_t groupRank, int deviceIdx, bool mnNvlink)
    : mIsMNNvlink(mnNvlink)
    , mDeviceIdx(deviceIdx)
    , mGroupSize(groupSize)
    , mGroupRank(groupRank)
    , mBufSize(bufSize)
    , mSignalPadOffset(0)
    , mAllocationSize(0)
    , mMcPtr(0)
    , mMcHandle(0)
{

    cudaSetDevice(mDeviceIdx);
    // Check if the device support multicasting
    int multicast_supported{0};
    CUCHECK(cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, mDeviceIdx));
    if (multicast_supported == 0)
    {
        TLLM_THROW("[McastDeviceMemory] Device does not support multicasting.");
    }

    // From pytorch implementation for alignment
    constexpr size_t kSignalPadAlignment = 16UL;
    mSignalPadOffset = roundUp(mBufSize, kSignalPadAlignment);
    TLLM_LOG_DEBUG(
        "[McastDeviceMemory] Rank: %u, Group size: %u, isMultiNode: %d, device_idx: %d, Signal pad offset: %zu",
        mGroupRank, mGroupSize, mIsMNNvlink, mDeviceIdx, mSignalPadOffset);

    if (mIsMNNvlink)
    {
        // For multi-node, we also need to check if fabric handle is supported
        int fabric_handle_supported{0};
        CUCHECK(cuDeviceGetAttribute(
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
    mSignalPadsDev.resize(mGroupSize);
    for (size_t i = 0; i < mGroupSize; i++)
    {
        mSignalPadsDev[i] = mUcPtrs[i] + mSignalPadOffset;
        if (i == mGroupRank)
        {
            cuMemsetD8(mSignalPadsDev[i], 0, kSIGNAL_PAD_SIZE);
        }
    }
}

McastDeviceMemory::~McastDeviceMemory()
{
    tensorrt_llm::common::unregisterMcastDevMemBuffer(this);
    if (mIsMNNvlink)
    {
        for (uint32_t rank = 0; rank < mGroupSize; rank++)
        {
            if (rank == mGroupRank)
            {
                cuMemRelease(mUcHandles[rank]);
            }
            else
            {
                mUcHandles[rank] = 0;
            }
        }
        cuMemRelease(mMcHandle);
    }
    else
    {
        // The nvlsfree function will free the handle pointer as well
        tensorrt_llm::runtime::ipcNvlsFree(mNvlsHandle);
    }
}

void McastDeviceMemory::allocMnMcastMem(size_t bufSize)
{

    auto const& mpi_comm = tensorrt_llm::mpi::MpiComm::world();

    CUmemAllocationHandleType const handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = handle_type;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = mDeviceIdx;

    size_t granularity{0};
    TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // Round up the buffer size for grnularity
    mAllocationSize = roundUp(bufSize + kSIGNAL_PAD_SIZE, granularity);

    mUcHandles.resize(mGroupSize);
    // Allocates local gpu memory
    TLLM_CU_CHECK(cuMemCreate(&(mUcHandles[mGroupRank]), mAllocationSize, &prop, 0));

    // Connect peer UC handles
    CUmemFabricHandle* exphndl{nullptr};
    CUmemFabricHandle myhndl;
    TLLM_CU_CHECK(cuMemExportToShareableHandle(&myhndl, mUcHandles[mGroupRank], CU_MEM_HANDLE_TYPE_FABRIC, 0));
    // All gather
    cudaMallocHost(&exphndl, mGroupSize * sizeof(CUmemFabricHandle));
    memcpy(exphndl + mGroupRank * sizeof(CUmemFabricHandle), &myhndl, sizeof(CUmemFabricHandle));
    mpi_comm.allgather(
        exphndl + mGroupRank * sizeof(CUmemFabricHandle), exphndl, sizeof(CUmemFabricHandle), mpi::MpiType::kCHAR);
    cudaDeviceSynchronize();

    for (uint32_t p = 0; p < mGroupSize; p++)
        if (p != mGroupRank)
            TLLM_CU_CHECK(cuMemImportFromShareableHandle(
                &mUcHandles[p], reinterpret_cast<void*>(&exphndl[p]), CU_MEM_HANDLE_TYPE_FABRIC));
    cudaFreeHost(exphndl);

    // Initialize multicasting
    CUmulticastObjectProp mcProp
        = {.numDevices = mGroupSize, .size = mAllocationSize, .handleTypes = CU_MEM_HANDLE_TYPE_FABRIC};
    CUmemFabricHandle* fabric_handle;
    cudaMallocHost(&fabric_handle, sizeof(CUmemFabricHandle));
    if (mGroupRank == 0)
    {
        TLLM_CU_CHECK(cuMulticastCreate(&mMcHandle, &mcProp));
        TLLM_CU_CHECK(cuMemExportToShareableHandle((void*) fabric_handle, mMcHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    // Broadcast
    mpi_comm.bcast(fabric_handle, sizeof(CUmemFabricHandle), mpi::MpiType::kCHAR, 0);
    cudaDeviceSynchronize();
    if (mGroupRank != 0)
    {
        TLLM_CU_CHECK(cuMemImportFromShareableHandle(&mMcHandle, (void*) fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));
    }
    TLLM_CU_CHECK(cuMulticastAddDevice(mMcHandle, mDeviceIdx));
    cudaFreeHost(fabric_handle);

    // Bind memory addresses
    mUcPtrs.resize(mGroupSize);
    CUdeviceptr ptr;
    TLLM_CU_CHECK(cuMemAddressReserve(&ptr, mAllocationSize * mGroupSize, 0, 0, 0));
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = mDeviceIdx;

    for (uint32_t i = 0; i < mGroupSize; i++)
    {
        TLLM_CU_CHECK(cuMemMap(ptr + (mAllocationSize * i), mAllocationSize, 0, mUcHandles[i], 0));
        mUcPtrs[i] = (ptr + (mAllocationSize * i));
    }
    TLLM_CU_CHECK(cuMemSetAccess(ptr, mAllocationSize * mGroupSize, &accessDesc, 1));

    // Bind MC Pointers
    TLLM_CU_CHECK(cuMemAddressReserve(&mMcPtr, mAllocationSize, 0, 0U, 0));
    TLLM_CU_CHECK(cuMemMap(mMcPtr, mAllocationSize, 0, mMcHandle, 0));
    TLLM_CU_CHECK(cuMemSetAccess(mMcPtr, mAllocationSize, &accessDesc, 1));

    TLLM_CU_CHECK(cuMulticastBindMem(mMcHandle, 0, mUcHandles[mGroupRank], 0 /*memOffset*/, mAllocationSize, 0));
}

void McastDeviceMemory::allocNvlsMcastMem(size_t bufSize)
{
    // Create a std::set to include all ranks in range (0, group_size)
    std::set<int> ranks;
    for (uint32_t i = 0; i < mGroupSize; ++i)
    {
        ranks.insert(i);
    }
    // Reuse existing implementation
    mNvlsHandle = tensorrt_llm::runtime::ipcNvlsAllocate(bufSize, ranks);
    mMcHandle = mNvlsHandle->mc_handle;
    mMcPtr = mNvlsHandle->mc_va;
    mUcPtrs = mNvlsHandle->ipc_uc_vas;
    mUcHandles = mNvlsHandle->ipc_uc_handles;
}

} // namespace tensorrt_llm::runtime
