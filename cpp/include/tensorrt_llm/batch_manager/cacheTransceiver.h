/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <future>
#include <map>
#include <memory>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager
{

class ContextProgress;
class BaseCacheTransceiver;
class DataResponder;
class DataRequester;

class CacheTransceiverFactory
{
public:
    static std::unique_ptr<BaseCacheTransceiver> createCacheTransceiver(
        kv_cache_manager::BaseKVCacheManager* cacheManager, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt);
};

class BaseCacheTransceiver
{
public:
    virtual ~BaseCacheTransceiver() = default;
    virtual void respondAndSendAsync(LlmRequest* llmRequest) = 0;
    virtual void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress)
        = 0;

    virtual void requestAndReceiveSync(LlmRequest* llmRequest) = 0;
    virtual void requestAndReceiveAsync(LlmRequest* llmRequest) = 0;

    virtual void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) = 0;

    virtual void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) = 0;

    [[nodiscard]] virtual bool checkGenTransferComplete() const = 0;
};

class CacheTransceiver : public BaseCacheTransceiver
{
public:
    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
        executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt);

    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, std::vector<SizeType32> numKvHeadsPerLayer,
        SizeType32 sizePerHead, SizeType32 tokensPerBlock, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt)
        : CacheTransceiver(cacheManager,
            executor::kv_cache::CacheState::ModelConfig{numKvHeadsPerLayer, sizePerHead, tokensPerBlock}, worldConfig,
            attentionLayerNumPerPP, dataType, attentionType, cacheTransceiverConfig)
    {
    }

    virtual ~CacheTransceiver();

    void respondAndSendAsync(LlmRequest* llmRequest) override;

    void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress) override;

    void requestAndReceiveSync(LlmRequest* llmRequest) override;
    void requestAndReceiveAsync(LlmRequest* llmRequest) override;

    void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;

    void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;

    [[nodiscard]] bool checkGenTransferComplete() const override;

private:
    void initializeCommState();

    void setContextState(LlmRequest* llmRequest);

    std::unique_ptr<DataResponder> mDataResponder;
    std::unique_ptr<DataRequester> mDataRequester;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mResponderFutures;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mRequesterFutures;
    mpi::MpiComm const *mMpiGroupComm{nullptr}, *mMpiWorldComm{nullptr};
    std::shared_ptr<mpi::MpiComm> mMpiGroupTensorParaComm, mMpiGroupPipeParaComm, mMpiGroupDataComm,
        mMpiGroupTPInDPComm;
    executor::kv_cache::CommState const* mCommState;
    std::unique_ptr<executor::kv_cache::CacheState> mCacheState;
    std::unique_ptr<executor::kv_cache::ConnectionManager> mManager;
    std::optional<executor::CacheTransceiverConfig> mCacheTransceiverConfig;
    std::unique_ptr<kv_cache_manager::CacheTransBufferManager> mCacheTransBufferManager;
    // library handle to the communicator related features,
    // this is used to defer dependency resolution until needed.
    static std::mutex mDllMutex;
    void* mWrapperLibHandle{nullptr};
};

} // namespace tensorrt_llm::batch_manager
