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

#include <utility>
#include <vector>

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using RequestIdType = tensorrt_llm::batch_manager::LlmRequest::RequestIdType;

/// See tensorrt_llm/_torch/pyexecutor/connector.py for details on the Connector API.

namespace tensorrt_llm::batch_manager::kv_connector
{

class KvCacheConnectorPoolData
{
public:
    KvCacheConnectorPoolData(runtime::ITensor::SharedPtr poolTensor, SizeType32 numBlocks)
        : mPoolTensor(std::move(poolTensor))
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

/// @brief Data used to provide the KV cache tensors to the connector worker for all the pools.
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

/// @brief The KV connector manager. This is passed into the C++ KV Cache Manager when adding sequences.
class KvCacheConnectorManager
{
public:
    KvCacheConnectorManager() = default;
    virtual ~KvCacheConnectorManager() = default;

    /// @brief Handle the getNumNewMatchedTokens call inside the C++ KV Cache Manager.
    /// @return The number of tokens that can be loaded from remote KV cache.
    virtual SizeType32 getNumNewMatchedTokens(LlmRequest const& request, SizeType32 numComputedTokens) = 0;
};

} // namespace tensorrt_llm::batch_manager::kv_connector
