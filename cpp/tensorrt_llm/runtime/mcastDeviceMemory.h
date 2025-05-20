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
#pragma once

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <vector>

namespace tensorrt_llm::runtime
{

//! \brief A class that manages multicast device memory for efficient communication between GPUs.
//!
//! This class uses IPC-based allocation if mnNvlink is true, otherwise it uses fabric allocation.
//! The fabric allocation can also be used for single-node/intra-node-only communication, but the machine
//! must properly configure IMEX services. See:
//! https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/gettingstarted.html
//!
//! The class manages both unicast pointers (one per rank) and a single multicast pointer,
//! along with signal pads used for synchronization between devices.
class McastDeviceMemory
{
public:
    // Disallow copy construction
    McastDeviceMemory(McastDeviceMemory const&) = delete;
    McastDeviceMemory& operator=(McastDeviceMemory const&) = delete;

    McastDeviceMemory(size_t bufSize, uint32_t groupSize, uint32_t groupRank, int deviceIdx, bool mnNvlink);

    //! Get the raw array of signal pad pointers to all ranks (including self)
    void** getSignalPadPtrsDev()
    {
        return reinterpret_cast<void**>(mSignalPadsDev.data());
    }

    //! Get the raw array of unicast pointers to all ranks (including self)
    void** getBufferPtrsDev()
    {
        return reinterpret_cast<void**>(mUcPtrs.data());
    }

    //! Get the raw unicast pointer to a given rank
    void* getUnicastPtr(uint32_t rank)
    {
        auto* data_ptr = reinterpret_cast<void*>(mUcPtrs[rank]);
        tensorrt_llm::common::registerMcastDevMemBuffer(data_ptr, this);
        return data_ptr;
    }

    //! Get the raw multicast pointer
    void* getMulticastPtr()
    {
        auto* data_ptr = reinterpret_cast<void*>(mMcPtr);
        tensorrt_llm::common::registerMcastDevMemBuffer(data_ptr, this);
        return data_ptr;
    }

    [[nodiscard]] size_t getRank() const
    {
        return mGroupRank;
    }

    [[nodiscard]] size_t getWorldSize() const
    {
        return mGroupSize;
    }

    ~McastDeviceMemory();

private:
    bool mIsMNNvlink;
    int mDeviceIdx;
    uint32_t mGroupSize, mGroupRank;
    size_t mBufSize;
    size_t mSignalPadOffset;
    size_t mAllocationSize;

    CUdeviceptr mMcPtr;
    std::vector<CUdeviceptr> mUcPtrs;
    std::vector<CUdeviceptr> mSignalPadsDev;
    CUmemGenericAllocationHandle mMcHandle;
    std::vector<CUmemGenericAllocationHandle> mUcHandles;

    // For intra-node mcast
    tensorrt_llm::runtime::IpcNvlsHandle* mNvlsHandle;

    void allocMnMcastMem(size_t bufSize);
    void allocNvlsMcastMem(size_t bufSize);
};

constexpr size_t kSIGNAL_PAD_SIZE = 2048;

//! \brief Wrapper class for McastDeviceMemory to facilitate PyTorch tensor creation.
//! It manages a buffer accessible via unicast or multicast for multi-node communication.
class McastGPUBuffer
{
public:
    // Disallow copy construction and assignment
    McastGPUBuffer(McastGPUBuffer const&) = delete;
    McastGPUBuffer& operator=(McastGPUBuffer const&) = delete;

    //! \brief Constructor for McastGpuBuffer.
    //! \param bufSize The total size of the buffer in bytes.
    //! \param groupSize The number of ranks in the communication group.
    //! \param groupRank The rank of the local process within the group.
    //! \param device The CUDA device for buffer allocation.
    //! \param mnNvlink Flag indicating if multi-node NVLink is used.
    McastGPUBuffer(size_t bufSize, uint32_t groupSize, uint32_t groupRank, at::Device device, bool mnNvlink)
        : mMcastDeviceMemory(bufSize, groupSize, groupRank, device.index(), mnNvlink)
        , mBufSize(bufSize)
        , mLocalDevice(device)
    {
    }

    //! \brief Returns a PyTorch tensor view of the unicast buffer portion for a specific rank.
    //! \param rank The target rank for the unicast pointer.
    //! \param sizes The desired shape (dimensions) of the tensor.
    //! \param dtype The data type of the tensor elements.
    //! \param storageOffset The offset in elements from the start of the buffer.
    //! \return An ATen tensor wrapping the unicast buffer section.
    at::Tensor getUCBuffer(uint32_t rank, c10::IntArrayRef sizes, c10::ScalarType dtype, int64_t storageOffset)
    {
        size_t const numel = std::accumulate(sizes.begin(), sizes.end(), 1UL, std::multiplies<size_t>());
        size_t const elementSize = c10::elementSize(dtype);
        size_t const reqSize = (numel + storageOffset) * elementSize;
        TORCH_CHECK(reqSize <= mBufSize, "McastGpuBuffer::getUcBuffer: the requested size (", reqSize,
            " bytes) exceeds the allocated size (", mBufSize, " bytes)");
        auto* dataPtr = static_cast<uint8_t*>(mMcastDeviceMemory.getUnicastPtr(rank)) + storageOffset * elementSize;

        auto options = at::TensorOptions().dtype(dtype).device(mLocalDevice);
        return at::for_blob(dataPtr, sizes).options(options).target_device(mLocalDevice).make_tensor();
    }

    //! \brief Returns a PyTorch tensor view of the multicast buffer portion.
    //! \param sizes The desired shape (dimensions) of the tensor.
    //! \param dtype The data type of the tensor elements.
    //! \param storageOffset The offset in elements from the start of the buffer.
    //! \return An ATen tensor wrapping the multicast buffer section.
    at::Tensor getMCBuffer(c10::IntArrayRef sizes, c10::ScalarType dtype, int64_t storageOffset)
    {
        size_t const numel = std::accumulate(sizes.begin(), sizes.end(), 1UL, std::multiplies<size_t>());
        size_t const elementSize = c10::elementSize(dtype);
        size_t const reqSize = (numel + storageOffset) * elementSize;
        TORCH_CHECK(reqSize <= mBufSize, "McastGpuBuffer::getMcBuffer: the requested size (", reqSize,
            " bytes) exceeds the allocated size (", mBufSize, " bytes)");
        auto* dataPtr = static_cast<uint8_t*>(mMcastDeviceMemory.getMulticastPtr()) + storageOffset * elementSize;

        auto options = at::TensorOptions().dtype(dtype).device(mLocalDevice);
        return at::for_blob(dataPtr, sizes).options(options).target_device(mLocalDevice).make_tensor();
    }

private:
    //!< Underlying memory manager for multi-node communication.
    tensorrt_llm::runtime::McastDeviceMemory mMcastDeviceMemory;
    size_t mBufSize;         //!< Total size of the managed buffer.
    at::Device mLocalDevice; //!< The local CUDA device.
};

} // namespace tensorrt_llm::runtime
