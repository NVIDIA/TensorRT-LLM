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
#pragma once

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace tensorrt_llm::runtime
{

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
    //! \param deviceIdx The CUDA device for buffer allocation.
    //! \param mnNvlink Flag indicating if multi-node NVLink is used.
    //! \param mpiCommFortranHandle The Fortran handle for the MPI communicator (from Python mpi4py).
    McastGPUBuffer(size_t bufSize, uint32_t groupSize, uint32_t groupRank, uint32_t deviceIdx, bool mnNvlink,
        int64_t mpiCommFortranHandle)
        : mMcastDeviceMemory(std::make_shared<McastDeviceMemory>(
            bufSize, groupSize, groupRank, deviceIdx, mnNvlink, mpiCommFortranHandle))
        , mBufSize(mMcastDeviceMemory->getBufferSize())
        , mLocalDevice(at::Device(at::DeviceType::CUDA, deviceIdx))
    {
        for (uint32_t rank = 0; rank < groupSize; ++rank)
        {
            tensorrt_llm::common::registerMcastDevMemBuffer(
                mMcastDeviceMemory->getUnicastPtr(rank), mMcastDeviceMemory);
        }
        tensorrt_llm::common::registerMcastDevMemBuffer(mMcastDeviceMemory->getMulticastPtr(), mMcastDeviceMemory);
    }

    //! \brief Returns the usable logical size after fabric-allocation rounding and signal-pad reservation.
    [[nodiscard]] size_t getBufferSize() const
    {
        return mBufSize;
    }

    //! \brief Returns a PyTorch tensor view of the unicast buffer portion for a specific rank.
    //! \param rank The target rank for the unicast pointer.
    //! \param sizes The desired shape (dimensions) of the tensor.
    //! \param dtype The data type of the tensor elements.
    //! \param storageOffset The offset in elements from the start of the buffer.
    //! \return An ATen tensor wrapping the unicast buffer section.
    at::Tensor getUCBuffer(uint32_t rank, std::vector<long int> sizes, torch::ScalarType dtype, int64_t storageOffset)
    {
        TORCH_CHECK(rank < mMcastDeviceMemory->getWorldSize(), "McastGPUBuffer::getUCBuffer: rank ", rank,
            " is outside world size ", mMcastDeviceMemory->getWorldSize());
        return makeTensor(
            mMcastDeviceMemory->getUnicastPtr(rank), sizes, dtype, storageOffset, "McastGPUBuffer::getUCBuffer");
    }

    //! \brief Returns a PyTorch tensor view of the multicast buffer portion.
    //! \param sizes The desired shape (dimensions) of the tensor.
    //! \param dtype The data type of the tensor elements.
    //! \param storageOffset The offset in elements from the start of the buffer.
    //! \return An ATen tensor wrapping the multicast buffer section.
    at::Tensor getMCBuffer(std::vector<long int> sizes, torch::ScalarType dtype, int64_t storageOffset)
    {
        return makeTensor(
            mMcastDeviceMemory->getMulticastPtr(), sizes, dtype, storageOffset, "McastGPUBuffer::getMCBuffer");
    }

private:
    using DeviceMemoryOwner = std::shared_ptr<tensorrt_llm::runtime::McastDeviceMemory>;

    static void deleteDeviceMemoryOwner(void* context)
    {
        delete static_cast<DeviceMemoryOwner*>(context);
    }

    at::Tensor makeTensor(void* basePtr, std::vector<long int> const& sizes, torch::ScalarType dtype,
        int64_t storageOffset, char const* caller)
    {
        TORCH_CHECK(storageOffset >= 0, caller, ": storage offset must be nonnegative, got ", storageOffset);

        size_t numel{1};
        for (long int const size : sizes)
        {
            TORCH_CHECK(size >= 0, caller, ": dimensions must be nonnegative, got ", size);
            size_t const unsignedSize = static_cast<size_t>(size);
            TORCH_CHECK(unsignedSize == 0 || numel <= std::numeric_limits<size_t>::max() / unsignedSize, caller,
                ": tensor element count overflows size_t");
            numel *= unsignedSize;
        }

        size_t const unsignedOffset = static_cast<size_t>(storageOffset);
        TORCH_CHECK(numel <= std::numeric_limits<size_t>::max() - unsignedOffset, caller,
            ": tensor storage extent overflows size_t");
        size_t const storageElements = numel + unsignedOffset;
        size_t const elementSize = c10::elementSize(dtype);
        TORCH_CHECK(elementSize == 0 || storageElements <= std::numeric_limits<size_t>::max() / elementSize, caller,
            ": tensor storage size overflows size_t");
        size_t const requiredSize = storageElements * elementSize;
        TORCH_CHECK(requiredSize <= mBufSize, caller, ": the requested size (", requiredSize,
            " bytes) exceeds the allocated size (", mBufSize, " bytes)");

        auto* dataPtr = static_cast<uint8_t*>(basePtr) + unsignedOffset * elementSize;
        auto* owner = new DeviceMemoryOwner(mMcastDeviceMemory);
        auto const options = at::TensorOptions().dtype(dtype).device(mLocalDevice);
        return at::for_blob(dataPtr, c10::IntArrayRef(sizes))
            .context(owner, deleteDeviceMemoryOwner)
            .options(options)
            .target_device(mLocalDevice)
            .make_tensor();
    }

    //!< Underlying memory manager for multi-node communication.
    DeviceMemoryOwner mMcastDeviceMemory;
    size_t mBufSize;         //!< Total size of the managed buffer.
    at::Device mLocalDevice; //!< The local CUDA device.
};

} // namespace tensorrt_llm::runtime
