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

#include "baseTransBuffer.h"
#include "cacheTransBuffer.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"

#include <mutex>

namespace tensorrt_llm::batch_manager
{

BaseTransBufferManager::BaseTransBufferManager(
    size_t transferBufferSize, nvinfer1::DataType dataType, std::optional<size_t> maxNumTokens)
    : mDataType{dataType}
    , mBufferManager{std::make_shared<runtime::CudaStream>()}
    , mMaxNumTokens{maxNumTokens}
{
    mTransferBufferSize = transferBufferSize;
    mOnlyUseDynamicBuffer = mTransferBufferSize == 0;
    mRecvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
    mSendBufferCount = common::getEnvKVCacheSendMaxConcurrenceNum();
    mUseFabricMemory = !(common::getEnvKVCacheTransferUseSyncBuffer() || common::getEnvKVCacheTransferUseAsyncBuffer())
        && kv_cache_manager::FabricMemory::supportFbaricMemory();
    if (mUseFabricMemory)
    {
        mTransferBufferSize = kv_cache_manager::FabricMemory::getAlignedSize(mTransferBufferSize);
    }
    mPreAllocBufferSize = mTransferBufferSize * (mRecvBufferCount + mSendBufferCount);

    TLLM_LOG_INFO(
        "BaseTransBufferManager: mMaxNumTokens:%ld, mRecvBufferCount:%ld, "
        "mSendBufferCount:%ld, mTransferBufferSize:%ld, mPreAllocBufferSize:%ld, mOnlyUseDynamicBuffer:%d, "
        "mUseFabricMemory:%d, mDataType:%d",
        maxNumTokens.has_value() ? maxNumTokens.value() : 0, mRecvBufferCount, mSendBufferCount, mTransferBufferSize,
        mPreAllocBufferSize, mOnlyUseDynamicBuffer, mUseFabricMemory, static_cast<int>(mDataType));

    allocateBuffer();
}

std::optional<int> BaseTransBufferManager::assignBufferIndexForSend()
{
    return assignBufferIndex(mConcurrenceSendResource, mSendBufferCount, mOnlyUseDynamicBuffer);
}

void BaseTransBufferManager::freeBufferIndexForSend(std::optional<int> bufferId)
{
    freeBufferIndex(mConcurrenceSendResource, bufferId, mSendBufferCount, mOnlyUseDynamicBuffer);
}

std::optional<int> BaseTransBufferManager::assignBufferIndexForRecv()
{
    return assignBufferIndex(mConcurrenceRecvResource, mRecvBufferCount, mOnlyUseDynamicBuffer);
}

void BaseTransBufferManager::freeBufferIndexForRecv(std::optional<int> bufferId)
{
    freeBufferIndex(mConcurrenceRecvResource, bufferId, mRecvBufferCount, mOnlyUseDynamicBuffer);
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> BaseTransBufferManager::getOrAllocateSendBuffers(
    std::optional<int> bufferId, int targetNum, std::vector<size_t> const& requestedNumberOfElements,
    runtime::BufferManager const& bufferManagerToUse)
{
    return getOrAllocateBuffers(
        bufferId, targetNum, requestedNumberOfElements, bufferManagerToUse, mConcurrenceSendResource);
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> BaseTransBufferManager::getOrAllocateRecvBuffers(
    std::optional<int> bufferId, int targetNum, std::vector<size_t> const& requestedNumberOfElements,
    runtime::BufferManager const& bufferManagerToUse)
{
    return getOrAllocateBuffers(
        bufferId, targetNum, requestedNumberOfElements, bufferManagerToUse, mConcurrenceRecvResource);
}

runtime::ITensor::SharedPtr BaseTransBufferManager::getSendBuffer(std::optional<int> bufferId)
{
    TLLM_CHECK(bufferId.has_value() || mOnlyUseDynamicBuffer);
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < mSendBufferCount);
        return mConcurrenceSendResource.mBuffers[bufferId.value()];
    }
    return nullptr;
}

runtime::ITensor::SharedPtr BaseTransBufferManager::getRecvBuffer(std::optional<int> bufferId)
{
    TLLM_CHECK(bufferId.has_value() || mOnlyUseDynamicBuffer);
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < mRecvBufferCount);
        // TLLM_CHECK(mConcurrenceRecvResource.mBufferIndexFlag[bufferId.value()] == 1);
        return mConcurrenceRecvResource.mBuffers[bufferId.value()];
    }
    return nullptr;
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> BaseTransBufferManager::getOrAllocateBuffers(
    std::optional<int> bufferId, int targetNum, std::vector<size_t> const& requestedNumberOfElements,
    runtime::BufferManager const& bufferManagerToUse, ConcurrenceResource& concurrenceResource)
{
    TLLM_CHECK(bufferId.has_value() || mOnlyUseDynamicBuffer);
    TLLM_CHECK(requestedNumberOfElements.size() >= static_cast<size_t>(targetNum));
    std::vector<runtime::ITensor::SharedPtr> retSplitCaches;

    size_t bufferCoverTargetNum = 0;

    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < concurrenceResource.mBuffers.size());
        TLLM_CHECK(concurrenceResource.mBufferIndexFlag[bufferId.value()] == 1);
        size_t preBufferEleSize = 0;
        for (int i = 0; i < targetNum; i++)
        {
            // Strict checking.
            if (preBufferEleSize + requestedNumberOfElements[i] <= mNumberOfElements)
            {
                auto slice = runtime::ITensor::slice(
                    concurrenceResource.mBuffers[bufferId.value()], preBufferEleSize, requestedNumberOfElements[i]);
                preBufferEleSize += requestedNumberOfElements[i];
                bufferCoverTargetNum++;
                retSplitCaches.push_back(std::move(slice));
            }
            else
            {
                retSplitCaches.push_back(bufferManagerToUse.gpu(
                    runtime::ITensor::makeShape({static_cast<int64_t>(requestedNumberOfElements[i])}), mDataType));
            }
        }
        TLLM_LOG_DEBUG("getOrAllocateBuffers bufferCoverTargetNum:%d", bufferCoverTargetNum);
        if (bufferCoverTargetNum < static_cast<size_t>(targetNum))
        {
            TLLM_LOG_WARNING(
                "CacheTransceiver getOrAllocateBuffers: bufferCoverTargetNum:%d < targetNum:%d, may use dynamic "
                "buffer which will fail with NIXL backend. It is recommended to set "
                "cacheTransceiverConfig.MaxTokensInBuffer (cache_transceiver_config.max_tokens_in_buffer in config "
                "YAML file) to a value greater than the maximum ISL of the processed requests. Otherwise, performance "
                "may be degraded or transfer may fail.  requestedNumberOfElements.size():%ld, "
                "mNumberOfElements:%ld, requestedNumberOfElements[0]:%ld",
                bufferCoverTargetNum, targetNum, requestedNumberOfElements.size(), mNumberOfElements,
                requestedNumberOfElements[0]);
        }
    }
    else
    {
        for (int i = 0; i < targetNum; i++)
        {
            retSplitCaches.push_back(bufferManagerToUse.gpu(
                runtime::ITensor::makeShape({static_cast<int64_t>(requestedNumberOfElements[i])}), mDataType));
        }
        bufferCoverTargetNum = targetNum;
    }

    return std::make_tuple(retSplitCaches, bufferCoverTargetNum, mOnlyUseDynamicBuffer);
}

void BaseTransBufferManager::allocateBuffer()
{
    if (mOnlyUseDynamicBuffer)
    {
        return;
    }
    mNumberOfElements = mTransferBufferSize / common::getDTypeSize(mDataType);
    mConcurrenceSendResource.mBufferIndexFlag.resize(mSendBufferCount, 0);
    mConcurrenceRecvResource.mBufferIndexFlag.resize(mRecvBufferCount, 0);
    if (mUseFabricMemory)
    {
        mFabricMemory.reserve(mSendBufferCount + mRecvBufferCount);
        for (size_t i = 0; i < mSendBufferCount; i++)
        {
            mFabricMemory.emplace_back(std::make_unique<kv_cache_manager::FabricMemory>(mTransferBufferSize));
            mConcurrenceSendResource.mBuffers[i] = runtime::ITensor::wrap(mFabricMemory.back()->getPtr(), mDataType,
                runtime::ITensor::makeShape({static_cast<int64_t>(mNumberOfElements)}), mNumberOfElements);
        }
        for (size_t i = 0; i < mRecvBufferCount; i++)
        {
            mFabricMemory.emplace_back(std::make_unique<kv_cache_manager::FabricMemory>(mTransferBufferSize));
            mConcurrenceRecvResource.mBuffers[i] = runtime::ITensor::wrap(mFabricMemory.back()->getPtr(), mDataType,
                runtime::ITensor::makeShape({static_cast<int64_t>(mNumberOfElements)}), mNumberOfElements);
        }
    }
    else if (common::getEnvKVCacheTransferUseAsyncBuffer())
    {
        for (size_t i = 0; i < mSendBufferCount; i++)
        {
            mConcurrenceSendResource.mBuffers[i]
                = mBufferManager.gpu(runtime::ITensor::makeShape({static_cast<int64_t>(mNumberOfElements)}), mDataType);
        }
        for (size_t i = 0; i < mRecvBufferCount; i++)
        {
            mConcurrenceRecvResource.mBuffers[i]
                = mBufferManager.gpu(runtime::ITensor::makeShape({static_cast<int64_t>(mNumberOfElements)}), mDataType);
        }
        mBufferManager.getStream().synchronize();
    }
    else
    {
        for (size_t i = 0; i < mSendBufferCount; i++)
        {
            mConcurrenceSendResource.mBuffers[i] = mBufferManager.gpuSync(
                runtime::ITensor::makeShape({static_cast<int64_t>(mNumberOfElements)}), mDataType);
        }
        for (size_t i = 0; i < mRecvBufferCount; i++)
        {
            mConcurrenceRecvResource.mBuffers[i] = mBufferManager.gpuSync(
                runtime::ITensor::makeShape({static_cast<int64_t>(mNumberOfElements)}), mDataType);
        }
    }
}

std::optional<int> BaseTransBufferManager::assignBufferIndex(
    ConcurrenceResource& resource, size_t bufferCount, bool onlyUseDynamicBuffer)
{
    if (onlyUseDynamicBuffer)
    {
        return std::nullopt;
    }
    std::unique_lock lk(resource.mBuffersMutex);
    resource.mBuffersCV.wait(
        lk, [&resource, bufferCount]() { return static_cast<size_t>(resource.mConcurrence) < bufferCount; });
    int bufferId = -1;
    for (size_t i = 0; i < bufferCount; i++)
    {
        if (resource.mBufferIndexFlag[i] == 0)
        {
            bufferId = i;
            resource.mBufferIndexFlag[bufferId] = 1;
            resource.mConcurrence++;
            break;
        }
    }
    TLLM_CHECK_WITH_INFO(bufferId >= 0 && static_cast<size_t>(bufferId) < bufferCount,
        " assignBufferIndex: Buffer index already assigned");

    return bufferId;
}

void BaseTransBufferManager::freeBufferIndex(
    ConcurrenceResource& resource, std::optional<int> bufferId, size_t bufferCount, bool onlyUseDynamicBuffer)
{
    if (onlyUseDynamicBuffer)
    {
        return;
    }
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < bufferCount);
        {
            std::scoped_lock lk(resource.mBuffersMutex);
            resource.mBufferIndexFlag[bufferId.value()] = 0;
        }
        resource.mConcurrence--;
        resource.mBuffersCV.notify_one();
    }
}

size_t BaseTransBufferManager::getRecvBufferCount()
{
    return mRecvBufferCount;
}

size_t BaseTransBufferManager::getSendBufferCount()
{
    return mSendBufferCount;
}

} // namespace tensorrt_llm::batch_manager
