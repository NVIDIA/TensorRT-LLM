/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class FabricMemory;
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::batch_manager
{

/// @brief Base class for cache transfer buffer management.
/// Handles buffer pool allocation, index assignment, and slicing.
/// Derived classes provide cache-specific size calculations.
class BaseTransBufferManager
{
public:
    virtual ~BaseTransBufferManager() = default;

    /// @brief Assign a buffer index for sending.
    /// @return Assigned buffer index, or nullopt if using dynamic buffers.
    std::optional<int> assignBufferIndexForSend();

    /// @brief Free a buffer index used for sending.
    /// @param bufferId The buffer index to free.
    void freeBufferIndexForSend(std::optional<int> bufferId);

    /// @brief Assign a buffer index for receiving.
    /// @return Assigned buffer index, or nullopt if using dynamic buffers.
    std::optional<int> assignBufferIndexForRecv();

    /// @brief Free a buffer index used for receiving.
    /// @param bufferId The buffer index to free.
    void freeBufferIndexForRecv(std::optional<int> bufferId);

    /// @brief Get or allocate send buffers for cache transfer.
    /// @param bufferId The assigned buffer ID.
    /// @param targetNum Number of target sequences.
    /// @param requestedNumberOfElements Sizes requested for each target.
    /// @param bufferManagerToUse Buffer manager for dynamic allocation.
    /// @return Tuple of (buffers, covered target count, is dynamic only).
    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateSendBuffers(
        std::optional<int> bufferId, int targetNum, std::vector<size_t> const& requestedNumberOfElements,
        runtime::BufferManager const& bufferManagerToUse);

    /// @brief Get or allocate receive buffers for cache transfer.
    /// @param bufferId The assigned buffer ID.
    /// @param targetNum Number of target sequences.
    /// @param requestedNumberOfElements Sizes requested for each target.
    /// @param bufferManagerToUse Buffer manager for dynamic allocation.
    /// @return Tuple of (buffers, covered target count, is dynamic only).
    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateRecvBuffers(
        std::optional<int> bufferId, int targetNum, std::vector<size_t> const& requestedNumberOfElements,
        runtime::BufferManager const& bufferManagerToUse);

    /// @brief Get the send buffer for a given buffer ID.
    runtime::ITensor::SharedPtr getSendBuffer(std::optional<int> bufferId);

    /// @brief Get the receive buffer for a given buffer ID.
    runtime::ITensor::SharedPtr getRecvBuffer(std::optional<int> bufferId);

    /// @brief Get the number of receive buffers.
    size_t getRecvBufferCount();

    /// @brief Get the number of send buffers.
    size_t getSendBufferCount();

    /// @brief Get the maximum number of tokens configured.
    std::optional<size_t> getMaxNumTokens()
    {
        return mMaxNumTokens;
    }

protected:
    /// @brief Constructor - derived classes call this after computing buffer sizes.
    /// @param transferBufferSize Size of each transfer buffer in bytes.
    /// @param dataType Data type for the buffers.
    /// @param maxNumTokens Optional max tokens for sizing.
    BaseTransBufferManager(
        size_t transferBufferSize, nvinfer1::DataType dataType, std::optional<size_t> maxNumTokens = std::nullopt);

    struct ConcurrenceResource
    {
        std::unordered_map<int, runtime::ITensor::SharedPtr> mBuffers;
        std::vector<int> mBufferIndexFlag;
        std::mutex mBuffersMutex;
        std::condition_variable mBuffersCV;
        std::atomic<int> mConcurrence{0};
    };

    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateBuffers(std::optional<int> bufferId,
        int targetNum, std::vector<size_t> const& requestedNumberOfElements,
        runtime::BufferManager const& bufferManagerToUse, ConcurrenceResource& concurrenceResource);

    void allocateBuffer();
    std::optional<int> assignBufferIndex(ConcurrenceResource& resource, size_t bufferCount, bool onlyUseDynamicBuffer);
    void freeBufferIndex(
        ConcurrenceResource& resource, std::optional<int> bufferId, size_t bufferCount, bool onlyUseDynamicBuffer);

    size_t mPreAllocBufferSize;
    size_t mRecvBufferCount;
    size_t mSendBufferCount;
    size_t mTransferBufferSize;
    bool mOnlyUseDynamicBuffer;
    bool mUseFabricMemory;
    size_t mNumberOfElements;
    nvinfer1::DataType mDataType;
    ConcurrenceResource mConcurrenceSendResource;
    ConcurrenceResource mConcurrenceRecvResource;
    runtime::BufferManager mBufferManager;
    std::vector<std::unique_ptr<kv_cache_manager::FabricMemory>> mFabricMemory;
    std::optional<size_t> mMaxNumTokens;
};

} // namespace tensorrt_llm::batch_manager
