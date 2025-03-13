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
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT);
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

    virtual void checkContextTransferStatus(bool blocking = false) = 0;

    virtual void checkGenTransferStatus(int atLeastRequestNum = 0) = 0;

    [[nodiscard]] virtual bool checkGenTransferComplete() const = 0;
};

class CacheTransceiver : public BaseCacheTransceiver
{
public:
    enum class CommType : std::uint8_t
    {
        UNKNOWN = 0,
        MPI = 1,
        UCX = 2
    };

    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, CommType commType,
        executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
        nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT);

    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, CommType commType,
        std::vector<SizeType32> numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        runtime::WorldConfig const& worldConfig, nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT)
        : CacheTransceiver(cacheManager, commType,
            executor::kv_cache::CacheState::ModelConfig{numKvHeadsPerLayer, sizePerHead, tokensPerBlock}, worldConfig,
            dataType, attentionType)
    {
    }

    virtual ~CacheTransceiver();

    void respondAndSendAsync(LlmRequest* llmRequest) override;

    void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress) override;

    void requestAndReceiveSync(LlmRequest* llmRequest) override;
    void requestAndReceiveAsync(LlmRequest* llmRequest) override;

    void checkContextTransferStatus(bool blocking = false) override;

    void checkGenTransferStatus(int atLeastRequestNum = 0) override;

    [[nodiscard]] bool checkGenTransferComplete() const override;

private:
    void initializeCommState();

    void setContextState(LlmRequest* llmRequest);

    CommType mCommType;
    std::unique_ptr<DataResponder> mDataResponder;
    std::unique_ptr<DataRequester> mDataRequester;
    std::map<LlmRequest*, std::future<void>> mResponderFutures;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mRequesterFutures;
    mpi::MpiComm const *mMpiGroupComm{}, *mMpiWorldComm{};
    std::shared_ptr<mpi::MpiComm> mMpiGroupTensorParaComm, mMpiGroupPipeParaComm, mMpiGroupDataComm,
        mMpiGroupTPInDPComm;
    executor::kv_cache::CommState const* mCommState;
    std::unique_ptr<executor::kv_cache::CacheState> mCacheState;
    std::unique_ptr<executor::kv_cache::ConnectionManager> mManager;

    // library handle to the communicator related features,
    // this is used to defer dependency resolution until needed.
    static std::mutex mDllMutex;
    void* mWrapperLibHandle{nullptr};
};

} // namespace tensorrt_llm::batch_manager
