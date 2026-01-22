/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

#pragma once

#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager
{

namespace rnn_state_manager
{
class RnnStateManager;
class RnnCacheTransBufferManager;
} // namespace rnn_state_manager

/// @brief RNN cache sender - inherits from BaseCacheSenderImpl with RNN-specific logic.
class RnnCacheSender : public BaseCacheSenderImpl
{
public:
    /// @brief Constructor.
    RnnCacheSender(executor::kv_cache::ConnectionManager* manager, executor::rnn_cache::RnnCacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter);

    /// @brief Receive request information - RNN-specific implementation.
    [[nodiscard]] RequestInfo recvRequestInfo() override;

protected:
    [[nodiscard]] executor::DataTransceiverState& getSelfState() override;
    [[nodiscard]] executor::DataTransceiverState const& getSelfState() const override;

private:
    executor::DataTransceiverState mSelfState;
};

/// @brief RNN cache receiver - inherits from BaseCacheReceiverImpl with RNN-specific logic.
class RnnCacheReceiver : public BaseCacheReceiverImpl
{
public:
    /// @brief Constructor.
    RnnCacheReceiver(executor::kv_cache::ConnectionManager* manager, executor::rnn_cache::RnnCacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter);

    /// @brief Send request information - RNN-specific implementation.
    [[nodiscard]] TransferSession sendRequestInfo(LlmRequest const& llmRequest) override;

protected:
    [[nodiscard]] executor::DataTransceiverState& getSelfState() override;
    [[nodiscard]] executor::DataTransceiverState const& getSelfState() const override;

private:
    executor::DataTransceiverState mSelfState;
};

/// @brief RNN Cache Transceiver for disaggregated RNN/Mamba state transfer.
class RnnCacheTransceiver : public BaseCacheTransceiver
{
public:
    RnnCacheTransceiver(rnn_state_manager::RnnStateManager* rnnStateManager,
        executor::rnn_cache::RnnCacheState rnnCacheState, runtime::WorldConfig const& worldConfig,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt);

    ~RnnCacheTransceiver() override;

    void respondAndSendAsync(LlmRequest* llmRequest) override;

    void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress) override;

    void requestAndReceiveSync(LlmRequest* llmRequest) override;
    void requestAndReceiveAsync(LlmRequest* llmRequest) override;

    void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;

    void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;

    [[nodiscard]] bool checkGenTransferComplete() const override;

    bool cancelRequest(LlmRequest* llmRequest) override;

private:
    void initializeCommState();
    void setContextState(LlmRequest* llmRequest);

    rnn_state_manager::RnnStateManager* mRnnStateManager;
    std::unique_ptr<RnnCacheSender> mCacheSender;
    std::unique_ptr<RnnCacheReceiver> mCacheReceiver;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mSenderFutures;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mRequesterFutures;
    mpi::MpiComm const* mMpiWorldComm{nullptr};

    std::shared_ptr<CacheTransceiverComm> mGroupComm;
    std::shared_ptr<CacheTransceiverComm> mGroupTensorParaComm, mGroupPipeParaComm;

    executor::kv_cache::CommState const* mCommState;
    std::unique_ptr<executor::rnn_cache::RnnCacheState> mRnnCacheState;
    std::unique_ptr<executor::kv_cache::ConnectionManager> mManager;
    std::optional<executor::CacheTransceiverConfig> mCacheTransceiverConfig;
    std::unique_ptr<rnn_state_manager::RnnCacheTransBufferManager> mRnnCacheTransBufferManager;
    rnn_state_manager::RnnCacheTransBufferManager* mRnnCacheTransBufferManagerPtr{nullptr};

    static std::mutex mDllMutex;
    void* mWrapperLibHandle{nullptr};
};

} // namespace tensorrt_llm::batch_manager
