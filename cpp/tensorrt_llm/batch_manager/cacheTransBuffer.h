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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class CacheTransBufferManager
{
public:
    CacheTransBufferManager(
        KVCacheManager::BaseKVCacheManager* cacheManager, std::optional<size_t> maxNumTokens = std::nullopt);

    static size_t preAllocBufferSize(
        std::optional<size_t> maxNumTokens = std::nullopt, std::optional<size_t> kvCacheSizePerToken = std::nullopt);

    std::optional<int> assignBufferIndexForSend();
    void freeBufferIndexForSend(std::optional<int> bufferId);
    std::optional<int> assignBufferIndexForRecv();
    void freeBufferIndexForRecv(std::optional<int> bufferId);

    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateSendBuffers(
        std::optional<int> bufferId, int targetNum, size_t targetBufferSize,
        runtime::BufferManager const& bufferManagerToUse);

    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateRecvBuffers(
        std::optional<int> bufferId, int targetNum, size_t targetBufferSize,
        runtime::BufferManager const& bufferManagerToUse);

    runtime::ITensor::SharedPtr getSendBuffer(std::optional<int> bufferId);
    runtime::ITensor::SharedPtr getRecvBuffer(std::optional<int> bufferId);
    size_t getRecvBufferCount();
    size_t getSendBufferCount();

private:
    struct ConcurrenceResource
    {
        std::unordered_map<int, runtime::ITensor::SharedPtr> mBuffers;
        std::vector<int> mBufferIndexFlag;
        std::mutex mBuffersMutex;
        std::condition_variable mBuffersCV;
        std::atomic<int> mConcurrence = 0;
    };

    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateBuffers(std::optional<int> bufferId,
        int targetNum, size_t targetBufferEleSize, runtime::BufferManager const& bufferManagerToUse,
        ConcurrenceResource& concurrenceResource);

    void allocateBuffer();
    std::optional<int> assignBufferIndex(ConcurrenceResource& resource, size_t bufferCount, bool onlyUseDynamicBuffer);
    void freeBufferIndex(
        ConcurrenceResource& resource, std::optional<int> bufferId, size_t bufferCount, bool onlyUseDynamicBuffer);

    size_t mPreAllocBufferSize;
    size_t mRecvBufferCount;
    size_t mSendBufferCount;
    size_t mTransferBufferSize;
    bool mOnlyUseDynamicBuffer;
    size_t mBufferEleSize;
    nvinfer1::DataType mDataType;
    ConcurrenceResource mConcurrenceSendResource;
    ConcurrenceResource mConcurrenceRecvResource;
    KVCacheManager::BaseKVCacheManager* mCacheManager;
    runtime::BufferManager mBufferManager;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
