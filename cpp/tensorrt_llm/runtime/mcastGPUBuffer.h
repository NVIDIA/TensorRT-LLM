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

#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"

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
    //! \param splitColor The color of the split for topology split.
    //! \param device The CUDA device for buffer allocation.
    //! \param mnNvlink Flag indicating if multi-node NVLink is used.
    McastGPUBuffer(
        size_t bufSize, uint32_t groupSize, uint32_t groupRank, uint32_t splitColor, uint32_t deviceIdx, bool mnNvlink)
        : mMcastDeviceMemory(bufSize, groupSize, groupRank, splitColor, deviceIdx, mnNvlink)
        , mBufSize(bufSize)
        , mLocalDevice(at::Device(at::DeviceType::CUDA, deviceIdx))
    {
    }

    //! \brief Returns a PyTorch tensor view of the unicast buffer portion for a specific rank.
    //! \param rank The target rank for the unicast pointer.
    //! \param sizes The desired shape (dimensions) of the tensor.
    //! \param dtype The data type of the tensor elements.
    //! \param storageOffset The offset in elements from the start of the buffer.
    //! \return An ATen tensor wrapping the unicast buffer section.
    at::Tensor getUCBuffer(uint32_t rank, std::vector<long int> sizes, torch::ScalarType dtype, int64_t storageOffset)
    {
        size_t const numel = std::accumulate(sizes.begin(), sizes.end(), 1UL, std::multiplies<size_t>());
        size_t const elementSize = c10::elementSize(dtype);
        size_t const reqSize = (numel + storageOffset) * elementSize;
        TORCH_CHECK(reqSize <= mBufSize, "McastGpuBuffer::getUcBuffer: the requested size (", reqSize,
            " bytes) exceeds the allocated size (", mBufSize, " bytes)");
        auto* dataPtr = static_cast<uint8_t*>(mMcastDeviceMemory.getUnicastPtr(rank)) + storageOffset * elementSize;

        auto options = at::TensorOptions().dtype(dtype).device(mLocalDevice);
        return at::for_blob(dataPtr, c10::IntArrayRef(sizes))
            .options(options)
            .target_device(mLocalDevice)
            .make_tensor();
    }

    //! \brief Returns a PyTorch tensor view of the multicast buffer portion.
    //! \param sizes The desired shape (dimensions) of the tensor.
    //! \param dtype The data type of the tensor elements.
    //! \param storageOffset The offset in elements from the start of the buffer.
    //! \return An ATen tensor wrapping the multicast buffer section.
    at::Tensor getMCBuffer(std::vector<long int> sizes, torch::ScalarType dtype, int64_t storageOffset)
    {
        size_t const numel = std::accumulate(sizes.begin(), sizes.end(), 1UL, std::multiplies<size_t>());
        size_t const elementSize = c10::elementSize(dtype);
        size_t const reqSize = (numel + storageOffset) * elementSize;
        TORCH_CHECK(reqSize <= mBufSize, "McastGpuBuffer::getMcBuffer: the requested size (", reqSize,
            " bytes) exceeds the allocated size (", mBufSize, " bytes)");
        auto* dataPtr = static_cast<uint8_t*>(mMcastDeviceMemory.getMulticastPtr()) + storageOffset * elementSize;

        auto options = at::TensorOptions().dtype(dtype).device(mLocalDevice);
        return at::for_blob(dataPtr, c10::IntArrayRef(sizes))
            .options(options)
            .target_device(mLocalDevice)
            .make_tensor();
    }

private:
    //!< Underlying memory manager for multi-node communication.
    tensorrt_llm::runtime::McastDeviceMemory mMcastDeviceMemory;
    size_t mBufSize;         //!< Total size of the managed buffer.
    at::Device mLocalDevice; //!< The local CUDA device.
};

} // namespace tensorrt_llm::runtime
