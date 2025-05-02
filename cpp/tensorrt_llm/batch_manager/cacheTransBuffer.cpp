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

#include "cacheTransBuffer.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <NvInferRuntimeBase.h>
#include <mutex>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

CacheTransBufferManager::CacheTransBufferManager(
    KVCacheManager::BaseKVCacheManager* cacheManager, std::optional<size_t> maxNumTokens)
    : mCacheManager{cacheManager}
    , mBufferManager{std::make_shared<runtime::CudaStream>()}
{

    TLLM_CHECK(mCacheManager);
    mDataType = mCacheManager->getPrimaryPool(0)->getDataType();

    auto tokensPerBlock = mCacheManager->getBlockManager().getTokensPerBlock();
    if (maxNumTokens.has_value())
    {
        TLLM_CHECK(maxNumTokens.value() % tokensPerBlock == 0);
    }
    TLLM_LOG_INFO("maxNumTokens: %d", maxNumTokens.has_value() ? maxNumTokens.value() : 0);
    auto kvCachePerToken
        = (mCacheManager->getBlockManager().getBlockSize(0) * mCacheManager->getBlockManager().getNumLayers()
              * (mCacheManager->getCacheType() == CacheType::kSELFKONLY ? 1 : 2))
        / tokensPerBlock;
    mTransferBufferSize = maxNumTokens.has_value() ? maxNumTokens.value() * kvCachePerToken
                                                   : common::getEnvMemSizeForKVCacheTransferBuffer();
    monlyUseDynamicBuffer = mTransferBufferSize == 0;
    mRecvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
    mSendBufferCount = common::getEnvParallelCacheSend() ? common::getEnvKVCacheSendMaxConcurrenceNum() : 1;
    mPreAllocBufferSize = mTransferBufferSize * (mRecvBufferCount + mSendBufferCount);
    TLLM_LOG_INFO(
        "CacheTransBufferManager: mMaxNumTokens:%ld, mRecvBufferCount:%ld, "
        "mSendBufferCount:%ld,mTransferBufferSize:%ld, mPreAllocBufferSize:%ld",
        maxNumTokens.has_value() ? maxNumTokens.value() : 0, mRecvBufferCount, mSendBufferCount, mTransferBufferSize,
        mPreAllocBufferSize);
    bool to_allocate = common::getEnvUseMPIKvCache() || common::getEnvUseUCXKvCache();

    TLLM_CHECK_WITH_INFO(to_allocate, "CacheTransBufferManager: to_allocate is false");
    allocateBuffer();
}

size_t CacheTransBufferManager::preAllocBufferSize(
    std::optional<size_t> maxNumTokens, std::optional<size_t> kvCacheSizePerToken)
{
    bool to_allocate = common::getEnvUseMPIKvCache() || common::getEnvUseUCXKvCache();
    if (!to_allocate)
    {
        return 0;
    }
    if (maxNumTokens.has_value())
    {
        TLLM_CHECK(kvCacheSizePerToken.has_value());
    }
    size_t TransferBufferSize = common::getEnvMemSizeForKVCacheTransferBuffer();
    if (maxNumTokens.has_value())
    {
        TransferBufferSize = maxNumTokens.value() * kvCacheSizePerToken.value();
    }
    size_t RecvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
    size_t SendBufferCount = common::getEnvParallelCacheSend() ? common::getEnvKVCacheSendMaxConcurrenceNum() : 1;
    size_t PreAllocBufferSize = TransferBufferSize * (RecvBufferCount + SendBufferCount);
    return PreAllocBufferSize;
}

std::optional<int> CacheTransBufferManager::assignBufferIndexForSend()
{
    return assignBufferIndex(mConcurrenceSendResource, mSendBufferCount, monlyUseDynamicBuffer);
}

void CacheTransBufferManager::freeBufferIndexForSend(std::optional<int> bufferId)
{
    freeBufferIndex(mConcurrenceSendResource, bufferId, mSendBufferCount, monlyUseDynamicBuffer);
}

std::optional<int> CacheTransBufferManager::assignBufferIndexForRecv()
{
    return assignBufferIndex(mConcurrenceRecvResource, mRecvBufferCount, monlyUseDynamicBuffer);
}

void CacheTransBufferManager::freeBufferIndexForRecv(std::optional<int> bufferId)
{
    freeBufferIndex(mConcurrenceRecvResource, bufferId, mRecvBufferCount, monlyUseDynamicBuffer);
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> CacheTransBufferManager::getOrAllocateSendBuffers(
    std::optional<int> bufferId, int targetNum, size_t targetBufferSize,
    runtime::BufferManager const& bufferManagerToUse)
{
    return getOrAllocateBuffers(bufferId, targetNum, targetBufferSize, bufferManagerToUse, mConcurrenceSendResource);
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> CacheTransBufferManager::getOrAllocateRecvBuffers(
    std::optional<int> bufferId, int targetNum, size_t targetBufferSize,
    runtime::BufferManager const& bufferManagerToUse)
{
    return getOrAllocateBuffers(bufferId, targetNum, targetBufferSize, bufferManagerToUse, mConcurrenceRecvResource);
}

runtime::ITensor::SharedPtr CacheTransBufferManager::getSendBuffer(std::optional<int> bufferId)
{
    TLLM_CHECK(bufferId.has_value() || monlyUseDynamicBuffer);
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < mSendBufferCount);
        TLLM_CHECK(mConcurrenceSendResource.mBufferIndexFlag[bufferId.value()] == 1);
        return mConcurrenceSendResource.mBuffers[bufferId.value()];
    }
    return nullptr;
}

runtime::ITensor::SharedPtr CacheTransBufferManager::getRecvBuffer(std::optional<int> bufferId)
{
    TLLM_CHECK(bufferId.has_value() || monlyUseDynamicBuffer);
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < mRecvBufferCount);
        TLLM_CHECK(mConcurrenceRecvResource.mBufferIndexFlag[bufferId.value()] == 1);
        return mConcurrenceRecvResource.mBuffers[bufferId.value()];
    }
    return nullptr;
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> CacheTransBufferManager::getOrAllocateBuffers(
    std::optional<int> bufferId, int targetNum, size_t targetBufferEleSize,
    runtime::BufferManager const& bufferManagerToUse, ConcurrenceResource& concurrenceResource)
{
    TLLM_CHECK(bufferId.has_value() || monlyUseDynamicBuffer);
    std::vector<runtime::ITensor::SharedPtr> retSplitCaches;
    size_t bufferCoverTargetNum = std::min(
        static_cast<size_t>(targetNum), mTransferBufferSize / (targetBufferEleSize * common::getDTypeSize(mDataType)));
    TLLM_LOG_DEBUG("getOrAllocateBuffers bufferCoverTargetNum:%d", bufferCoverTargetNum);
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < mSendBufferCount);
        TLLM_CHECK(concurrenceResource.mBufferIndexFlag[bufferId.value()] == 1);

        for (int i = 0; i < targetNum; i++)
        {
            if (static_cast<size_t>(i) < bufferCoverTargetNum)
            {
                auto slice = runtime::ITensor::slice(
                    concurrenceResource.mBuffers[bufferId.value()], i * targetBufferEleSize, targetBufferEleSize);
                retSplitCaches.push_back(std::move(slice));
            }
            else
            {
                retSplitCaches.push_back(bufferManagerToUse.gpu(
                    runtime::ITensor::makeShape({static_cast<int64_t>(targetBufferEleSize)}), mDataType));
            }
        }
    }
    else
    {
        for (int i = 0; i < targetNum; i++)
        {
            retSplitCaches.push_back(bufferManagerToUse.gpu(
                runtime::ITensor::makeShape({static_cast<int64_t>(targetBufferEleSize)}), mDataType));
        }
    }
    if (monlyUseDynamicBuffer)
    {
        bufferCoverTargetNum = targetNum;
    }
    return std::make_tuple(retSplitCaches, bufferCoverTargetNum, monlyUseDynamicBuffer);
}

void CacheTransBufferManager::allocateBuffer()
{
    if (monlyUseDynamicBuffer)
    {
        TLLM_LOG_INFO("monlyUseDynamicBuffer: true");
        return;
    }
    mBufferEleSize = mTransferBufferSize / common::getDTypeSize(mDataType);
    mConcurrenceSendResource.mBufferIndexFlag.resize(mSendBufferCount, 0);
    mConcurrenceRecvResource.mBufferIndexFlag.resize(mRecvBufferCount, 0);
    if (common::getEnvKVCacheTransferUseAsyncBuffer())
    {
        for (size_t i = 0; i < mSendBufferCount; i++)
        {
            mConcurrenceSendResource.mBuffers[i]
                = mBufferManager.gpu(runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), mDataType);
        }
        for (size_t i = 0; i < mRecvBufferCount; i++)
        {
            mConcurrenceRecvResource.mBuffers[i]
                = mBufferManager.gpu(runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), mDataType);
        }
        mBufferManager.getStream().synchronize();
    }
    else
    {
        for (size_t i = 0; i < mSendBufferCount; i++)
        {
            mConcurrenceSendResource.mBuffers[i] = mBufferManager.gpuSync(
                runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), mDataType);
        }
        for (size_t i = 0; i < mRecvBufferCount; i++)
        {
            mConcurrenceRecvResource.mBuffers[i] = mBufferManager.gpuSync(
                runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), mDataType);
        }
    }
}

std::optional<int> CacheTransBufferManager::assignBufferIndex(
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

void CacheTransBufferManager::freeBufferIndex(
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

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
