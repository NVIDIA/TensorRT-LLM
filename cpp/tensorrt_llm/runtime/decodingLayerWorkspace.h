/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory>

#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"

namespace tensorrt_llm::runtime
{

///@brief A collection of shared resources and data for the decoding layers.
class DecodingLayerWorkspace
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TensorUniquePtr = ITensor::UniquePtr;
    using TensorConstPtr = ITensor::SharedConstPtr;
    using BufferPtr = IBuffer::SharedPtr;

    DecodingLayerWorkspace(std::shared_ptr<BufferManager> bufferManager, layers::DecoderDomain const& decoderDomain,
        nvinfer1::DataType logitsType, size_t workspaceBufferSizeInBytes);

    DecodingLayerWorkspace() = delete;

    DecodingLayerWorkspace(DecodingLayerWorkspace const& decodingLayerWorkspace) = delete;

    ///@brief Gets a pointer to the start of the shared device workspace.
    [[nodiscard]] void* getRawWorkspaceDevicePtr() const;

    ///@brief Gets a pointer to the start of the shared device workspace, as a pointer to the given type.
    template <typename T>
    T* getWorkspaceDevicePtrAs() const
    {
        return reinterpret_cast<T*>(mWorkspaceDeviceBuffer->data());
    };

    ///@brief Gets a pointer to the buffer backing the device workspace.
    [[nodiscard]] BufferPtr getWorkspaceDeviceBuffer() const;

    ///@brief Sets the value of the device copy of the batch slots.
    void setDeviceBatchSlots(TensorConstPtr const& newBatchSlots);

    ///@brief Gets the pointer to the batch slots on device.
    [[nodiscard]] SizeType32 const* getDeviceBatchSlotsPtr() const;

    ///@brief Gets the device tensor containing the batch slots.
    [[nodiscard]] TensorConstPtr getDeviceBatchSlots() const;

    ///@brief Gets the device tensor containing the runtime logits.
    [[nodiscard]] TensorPtr getDeviceRuntimeLogits() const;

    ///@brief Gets a tensor with the given shape and type at the start of the device workspace.
    TensorPtr getWorkspaceAsDeviceTensor(ITensor::Shape shape, nvinfer1::DataType type);

    /// @brief A convenience function to copy the content of a standard vector to a device workspace.
    template <typename T, typename Alloc>
    static void copyToWorkspace(runtime::BufferManager const& bufferManager, std::vector<T, Alloc> const& src,
        runtime::IBuffer::SharedPtr workspace)
    {
        auto const sizeOfWorkspaceInBytes = workspace->getSizeInBytes();
        auto const sizeOfSrcInBytes = sizeof(T) * src.size();
        TLLM_CHECK_WITH_INFO(sizeOfSrcInBytes <= sizeOfWorkspaceInBytes,
            "The size of the workspace (%zu bytes) is insufficient for the data (%zu bytes)", sizeOfWorkspaceInBytes,
            sizeOfSrcInBytes);
        auto const sizePerWorkspaceElement = BufferDataType(workspace->getDataType()).getSize();
        TLLM_CHECK_WITH_INFO(sizePerWorkspaceElement == 1 || sizePerWorkspaceElement == sizeof(T),
            "Copy to typed workspace, but element size mismatched (src: %zu, workspace: %zu)", sizeof(T),
            sizePerWorkspaceElement);
        runtime::IBuffer::SharedPtr workspaceSlice
            = runtime::IBuffer::slice(workspace, 0, sizeOfSrcInBytes / sizePerWorkspaceElement);
        bufferManager.copy(src.data(), *workspaceSlice, runtime::MemoryType::kCPU);
    }

    /// @brief A convenience function to copy the content of a standard vector to the workspace.
    template <typename T>
    TensorPtr copyToWorkspace(std::vector<T> const& src)
    {
        copyToWorkspace(*mBufferManager, src, mWorkspaceDeviceBuffer);
        return getWorkspaceAsDeviceTensor(
            ITensor::makeShape({static_cast<SizeType32>(src.size())}), TRTDataType<T>::value);
    }

    ///@brief Ensures the workspace has at least the provided space in bytes. Does nothing if the workspace is already
    /// at least as large.
    void resize(size_t minSize);

    ///@brief Given a collection of tuples of tensor shapes and data types, returns the memory aligned size required to
    /// contain those tensors.
    template <typename... Args>
    size_t static calculateRequiredWorkspaceSize(Args&&... args)
    {
        size_t lastTensorOffset = 0;
        auto alignedSizeCalculator
            = [&lastTensorOffset](std::pair<ITensor::Shape, nvinfer1::DataType> const& tensorDescriptor)
        {
            auto const& [shape, type] = tensorDescriptor;
            auto const sizeInBytes = ITensor::volume(shape) * tensorrt_llm::common::getDTypeSize(type);
            auto const sliceEnd = lastTensorOffset + sizeInBytes;
            lastTensorOffset = tensorrt_llm::common::alignSize(sliceEnd, tensorrt_llm::common::kCudaMemAlign);
        };
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);
        forEach(alignedSizeCalculator, argTuple);
        return lastTensorOffset;
    }

    ///@brief Given a collection of tensors, creates tensors with the same shape and data types in the workspace and
    /// copies the data from the input tensors to their reflection on device.
    template <typename... Args>
    auto mirrorInWorkspace(Args&&... args)
    {
        auto* lastTensorEndPtr = reinterpret_cast<std::int8_t*>(mWorkspaceDeviceBuffer->data());
        auto tensorFactory = [&lastTensorEndPtr, this](auto const& tensor)
        {
            if (tensor == nullptr)
            {
                return std::unique_ptr<GenericTensor<BorrowingAllocator<MemoryType::kGPU>>>{};
            }
            auto const sizeInBytes = tensor->getSizeInBytes();
            auto const borrowingAllocator = BorrowingAllocator<MemoryType::kGPU>{lastTensorEndPtr, sizeInBytes};
            auto res = std::make_unique<GenericTensor<BorrowingAllocator<MemoryType::kGPU>>>(
                tensor->getShape(), tensor->getDataType(), borrowingAllocator);
            auto const sliceEnd = lastTensorEndPtr + sizeInBytes;
            lastTensorEndPtr = tensorrt_llm::common::alignPtr(sliceEnd, tensorrt_llm::common::kCudaMemAlign);
            mBufferManager->copy(*tensor, *res);
            return res;
        };
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);

        auto res = transform(tensorFactory, argTuple);
        std::size_t const numArgs = sizeof...(Args);
        std::size_t const sizeInBytes
            = lastTensorEndPtr - reinterpret_cast<std::int8_t*>(mWorkspaceDeviceBuffer->data());
        TLLM_LOG_DEBUG("Borrowing %lu bytes of the workspace for %i tensors.", sizeInBytes, numArgs);
        return res;
    }

    /// @brief A convenience function to initialize curand states from a provided seed.
    void initializeDeviceCurandStates(std::optional<std::vector<uint64_t>> const& randomSeed,
        runtime::SizeType32 batchSize, TensorConstPtr const& batchSlots, TensorPtr& statesDevice);

private:
    std::shared_ptr<BufferManager> mBufferManager;
    TensorPtr mBatchSlotsDevice;    // <! A copy of the batch slots on device ensure fast access when used in kernels.
    TensorPtr mRuntimeLogitsDevice; // <! The working state of the logits while decoding.
    TensorPtr
        mCurandStatesDevice; // <! The state information of the random number generators for sampling based decoding.
    BufferPtr mWorkspaceDeviceBuffer; // <! A buffer to be used as scratch space by the decoding layers.

    cudaStream_t getStream();

    ///@brief A helper template to apply a function to each element of a tuple and return a tuple of the results.
    template <typename Func, typename Tuple, std::size_t... I>
    auto static transformImpl(Func&& func, Tuple&& tuple, std::index_sequence<I...>)
    {
        return std::make_tuple(func(std::get<I>(tuple))...);
    }

    ///@brief A helper template to apply a function to each element of a tuple and return a tuple of the results.
    template <typename Func, typename... Args>
    auto static transform(Func&& func, std::tuple<Args...> const& tuple)
    {
        return transformImpl(std::forward<Func>(func), tuple, std::index_sequence_for<Args...>{});
    }

    ///@brief A helper template to apply a function to each element of a tuple.
    template <typename Func, typename Tuple, std::size_t... I>
    void static forEachImpl(Func&& func, Tuple&& tuple, std::index_sequence<I...>)
    {
        (func(std::get<I>(tuple)), ...);
    }

    ///@brief A helper template to apply a function to each element of a tuple.
    template <typename Func, typename... Args>
    void static forEach(Func&& func, std::tuple<Args...> const& tuple)
    {
        forEachImpl(std::forward<Func>(func), tuple, std::index_sequence_for<Args...>{});
    }
};

} // namespace tensorrt_llm::runtime
