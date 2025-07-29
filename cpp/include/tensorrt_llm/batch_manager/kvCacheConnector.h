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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/common.h"

#include <cstdint>
#include <vector>

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using RequestIdType = tensorrt_llm::batch_manager::LlmRequest::RequestIdType;

using namespace tensorrt_llm::batch_manager;

namespace tensorrt_llm::batch_manager::kv_connector
{

struct NewRequestData
{
    NewRequestData(RequestIdType requestId, std::vector<runtime::UniqueToken> const& newTokens,
        std::vector<SizeType32> const& blockIds, SizeType32 numComputedTokens)
        : requestId(requestId)
        , newTokens(newTokens)
        , blockIds(blockIds)
        , numComputedTokens(numComputedTokens)
    {
    }

    RequestIdType requestId;
    std::vector<runtime::UniqueToken> newTokens;
    std::vector<SizeType32> blockIds;
    SizeType32 numComputedTokens;
};

struct CachedRequestData
{
    CachedRequestData(RequestIdType requestId, std::vector<runtime::UniqueToken> const& newTokens,
        std::vector<SizeType32> const& newBlockIds, SizeType32 numComputedTokens)
        : requestId(requestId)
        , newTokens(newTokens)
        , newBlockIds(newBlockIds)
        , numComputedTokens(numComputedTokens)
    {
    }

    RequestIdType requestId;
    std::vector<runtime::UniqueToken> newTokens;
    std::vector<SizeType32> newBlockIds;
    SizeType32 numComputedTokens;
};

class SchedulerOutput
{
public:
    std::vector<NewRequestData> newRequests;
    std::vector<CachedRequestData> cachedRequests;
};

class KvCacheConnectorPoolData
{
public:
    KvCacheConnectorPoolData(runtime::ITensor::SharedPtr const& poolTensor, SizeType32 numBlocks)
        : mPoolTensor(poolTensor)
        , mNumBlocks(numBlocks)
    {
    }

    runtime::ITensor::SharedPtr const& getPoolTensor() const
    {
        return mPoolTensor;
    }

    SizeType32 getNumBlocks() const
    {
        return mNumBlocks;
    }

private:
    runtime::ITensor::SharedPtr mPoolTensor;
    SizeType32 mNumBlocks;
};

class KvCacheConnectorPoolsData
{
public:
    explicit KvCacheConnectorPoolsData(
        std::vector<KvCacheConnectorPoolData>& poolsData, std::vector<SizeType32>& layerToPoolMapping)
        : mPoolsData(poolsData)
        , mLayerToPoolMapping(layerToPoolMapping)
    {
    }

    std::vector<KvCacheConnectorPoolData>& getPoolsData()
    {
        return mPoolsData;
    }

    std::vector<SizeType32>& getLayerToPoolMapping()
    {
        return mLayerToPoolMapping;
    }

private:
    std::vector<KvCacheConnectorPoolData> mPoolsData;
    std::vector<SizeType32> mLayerToPoolMapping;
};

class KvCacheConnectorScheduler
{
public:
    explicit KvCacheConnectorScheduler() = default;
    virtual ~KvCacheConnectorScheduler() = default;

    virtual std::tuple<SizeType32, bool> getNumNewMatchedTokens(LlmRequest const& request, SizeType32 numComputedTokens)
        = 0;

    // TODO(jothomson): Need arguments here. Also, is this even needed?
    virtual void updateStateAfterAlloc();

    virtual bool requestFinished(LlmRequest const& request);
};

class KvCacheConnectorWorker
{
public:
    explicit KvCacheConnectorWorker() = default;
    virtual ~KvCacheConnectorWorker() = default;

    // TODO(jothomson): Need arguments here.
    virtual void registerKvCaches(KvCacheConnectorPoolsData const& kvCacheConnectorPoolsData);

    // TODO(jothomson): Need arguments here.
    virtual void startLoadKv() = 0;

    virtual void waitForLayerLoad(SizeType32 layer_idx) = 0;

    // TODO(jothomson): Need arguments here.
    virtual void saveKvLayer(SizeType32 layer_idx) = 0;

    virtual void waitForSave() = 0;

    virtual std::tuple<std::vector<RequestIdType>, std::vector<RequestIdType>> getFinished(
        std::vector<RequestIdType> const& finishedReqIds);
};

class KvCacheConnectorManager
{
public:
    KvCacheConnectorManager() = default;
    virtual ~KvCacheConnectorManager() = default;

    virtual SizeType32 getNumNewMatchedTokens(LlmRequest const& request, SizeType32 numComputedTokens) = 0;
};

} // namespace tensorrt_llm::batch_manager::kv_connector
