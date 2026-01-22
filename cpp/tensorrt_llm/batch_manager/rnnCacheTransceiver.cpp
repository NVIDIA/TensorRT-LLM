/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/types.h"
#include <cstdint>
#include <limits>
#include <sstream>
#define UCX_WRAPPER_LIB_NAME "tensorrt_llm_ucx_wrapper"
#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary(name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/batch_manager/rnnCacheTransceiver.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <numeric>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

std::mutex RnnCacheTransceiver::mDllMutex;

// ============================================================================
// RnnCacheSender Implementation
// ============================================================================

RnnCacheSender::RnnCacheSender(executor::kv_cache::ConnectionManager* manager,
    executor::rnn_cache::RnnCacheState selfCacheState, SizeType32 selfIndex,
    std::unique_ptr<BaseCacheFormatter> formatter)
    : BaseCacheSenderImpl(manager, std::move(formatter), selfIndex)
{
    mSelfState.setRnnCacheState(std::move(selfCacheState));
    mSelfState.setCommState(executor::kv_cache::CommState{mManager->getCommState()});
}

RequestInfo RnnCacheSender::recvRequestInfo()
{
    TLLM_THROW("RnnCacheSender::recvRequestInfo not yet implemented");
    return RequestInfo{};
}

executor::DataTransceiverState& RnnCacheSender::getSelfState()
{
    return mSelfState;
}

executor::DataTransceiverState const& RnnCacheSender::getSelfState() const
{
    return mSelfState;
}

// ============================================================================
// RnnCacheReceiver Implementation
// ============================================================================

RnnCacheReceiver::RnnCacheReceiver(executor::kv_cache::ConnectionManager* manager,
    executor::rnn_cache::RnnCacheState selfCacheState, SizeType32 selfIndex,
    std::unique_ptr<BaseCacheFormatter> formatter)
    : BaseCacheReceiverImpl(manager, std::move(formatter), selfIndex)
{
    // RNN-specific state initialization
    mSelfState.setRnnCacheState(std::move(selfCacheState));
    mSelfState.setCommState(executor::kv_cache::CommState{mManager->getCommState()});
}

TransferSession RnnCacheReceiver::sendRequestInfo(LlmRequest const& llmRequest)
{
    // TODO
    TLLM_THROW("RnnCacheReceiver::sendRequestInfo not yet implemented");
}

executor::DataTransceiverState& RnnCacheReceiver::getSelfState()
{
    return mSelfState;
}

executor::DataTransceiverState const& RnnCacheReceiver::getSelfState() const
{
    return mSelfState;
}

// ============================================================================
// RnnCacheTransceiver Implementation
// ============================================================================

// TODO: check if rnn cache state should be input or model config and inner vars as input from python
RnnCacheTransceiver::RnnCacheTransceiver(rnn_state_manager::RnnStateManager* rnnStateManager,
    executor::rnn_cache::RnnCacheState rnnCacheState, runtime::WorldConfig const& worldConfig,
    std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig)
    : mRnnStateManager{rnnStateManager}
    , mCacheTransceiverConfig{cacheTransceiverConfig}
{
    TLLM_CHECK(mRnnStateManager != nullptr);

    mRnnCacheState = std::make_unique<executor::rnn_cache::RnnCacheState>(std::move(rnnCacheState));

    if (useMPI())
    {
        mGroupComm = std::make_shared<CacheTransceiverComm>(std::addressof(tensorrt_llm::mpi::MpiComm::session()));
    }
    else
    {
        mGroupComm = std::make_shared<CacheTransceiverComm>(tensorrt_llm::pg_utils::get_world_pg());
    }

    if (worldConfig.isTensorParallel())
    {
        mGroupTensorParaComm = std::make_shared<CacheTransceiverComm>(
            mGroupComm->split(worldConfig.getPipelineParallelRank(), worldConfig.getTensorParallelRank()));
    }
    // TODO: attention DP

    auto backendType = mCacheTransceiverConfig.value().getBackendType();
    TLLM_CHECK_WITH_INFO(
        backendType.has_value() && (backendType.value() != executor::CacheTransceiverConfig::BackendType::DEFAULT),
        " CacheTransceiverConfig::BackendType is not set.");

    // std::optional<size_t> maxNumTokens = mCacheTransceiverConfig.value().getMaxTokensInBuffer();
    if (backendType.value() == executor::CacheTransceiverConfig::BackendType::MPI)
    {
        mMpiWorldComm = std::addressof(tensorrt_llm::mpi::MpiComm::world());
        mManager = std::make_unique<executor::kv_cache::MpiConnectionManager>(mMpiWorldComm);
        TLLM_LOG_INFO("RNN Cache: MPI Connection Manager created");
    }
    else if (backendType.value() == executor::CacheTransceiverConfig::BackendType::UCX)
    {
        std::lock_guard<std::mutex> lock(mDllMutex);
        mWrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
        TLLM_CHECK_WITH_INFO(
            mWrapperLibHandle != nullptr, "UCX wrapper library is not open correctly. error : %s", dlerror());
        auto load_sym = [](void* handle, char const* name)
        {
            void* ret = dllGetSym(handle, name);
            TLLM_CHECK_WITH_INFO(ret != nullptr, "Unable to load UCX wrapper library symbol");
            return ret;
        };
        std::unique_ptr<executor::kv_cache::ConnectionManager> (*makeUcxConnectionManager)();
        *(void**) (&makeUcxConnectionManager) = load_sym(mWrapperLibHandle, "makeUcxConnectionManager");
        mManager = makeUcxConnectionManager();
        TLLM_LOG_INFO("RNN Cache: UCX Connection Manager created");
    }
    else
    {
        TLLM_THROW("Unsupported cache transceiver backend type for RNN cache. Supported backends: MPI, UCX.");
    }

    // Create formatter - for now without buffer manager (MPI doesn't need it for basic transfer)
    auto makeFormatter = [this]() { return std::make_unique<RnnCacheFormatter>(mRnnStateManager); };

    mCacheSender
        = std::make_unique<RnnCacheSender>(mManager.get(), *mRnnCacheState, worldConfig.getRank(), makeFormatter());
    mCacheReceiver
        = std::make_unique<RnnCacheReceiver>(mManager.get(), *mRnnCacheState, worldConfig.getRank(), makeFormatter());

    initializeCommState();

    TLLM_LOG_INFO("RnnCacheTransceiver created successfully");
}

RnnCacheTransceiver::~RnnCacheTransceiver()
{
    if (mWrapperLibHandle)
    {
        std::lock_guard<std::mutex> lock(mDllMutex);
        dllClose(mWrapperLibHandle);
    }
}

void RnnCacheTransceiver::initializeCommState()
{
    mCommState = std::addressof(mCacheSender->getCommState());
}

void RnnCacheTransceiver::setContextState(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    auto contextState = std::make_unique<executor::DataTransceiverState>();
    contextState->setCommState(*mCommState);
    contextState->setRnnCacheState(*mRnnCacheState);
    if (!llmRequest->hasDraftTokens())
    {
        llmRequest->setContextPhaseParams(
            executor::ContextPhaseParams{{}, llmRequest->mRequestId, contextState.release(), std::nullopt});
    }
    else
    {
        llmRequest->setContextPhaseParams(executor::ContextPhaseParams{
            {}, llmRequest->mRequestId, contextState.release(), *llmRequest->getDraftTokens()});
    }
}

void RnnCacheTransceiver::respondAndSendAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    llmRequest->setState(LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS);

    if (llmRequest->getContextPhaseParams().has_value())
    {
        if (llmRequest->getContextProgress() == nullptr)
        {
            TLLM_LOG_WARNING("Request %ld is already responding", llmRequest->mRequestId);
        }
        return;
    }
    setContextState(llmRequest);
    auto future = mCacheSender->sendAsync(*llmRequest);
    mSenderFutures.emplace_back(llmRequest, std::move(future));
}

void RnnCacheTransceiver::respondAndSendLayerWise(
    RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress)
{
    TLLM_THROW("RnnCacheTransceiver::respondAndSendLayerWise not yet supported for RNN cache transfer");
}

void RnnCacheTransceiver::requestAndReceiveSync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    {
        auto future = mCacheReceiver->receiveAsync(*llmRequest);
        future.get();
    }
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
}

void RnnCacheTransceiver::requestAndReceiveAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());

    if (std::find_if(mRequesterFutures.begin(), mRequesterFutures.end(),
            [llmRequest](auto const& pair) { return pair.first->mRequestId == llmRequest->mRequestId; })
        != mRequesterFutures.end())
    {
        TLLM_LOG_WARNING("Request ID %zu is already in mRequestFutures.", llmRequest->mRequestId);
        return;
    }

    auto future = mCacheReceiver->receiveAsync(*llmRequest);
    mRequesterFutures.emplace_back(llmRequest, std::move(future));
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS);
}

void RnnCacheTransceiver::checkContextTransferStatus(std::optional<int> const& atLeastRequestNum)
{
    bool blockAll = !atLeastRequestNum.has_value();
    std::optional<int> senderFutureTimeoutMs = std::nullopt;
    if (!blockAll && mCacheTransceiverConfig.has_value())
    {
        senderFutureTimeoutMs = mCacheTransceiverConfig->getKvTransferSenderFutureTimeoutMs();
    }

    auto syncComm = mGroupTensorParaComm; // TODO: add attention DP

    // Collect request IDs that are ready locally
    std::vector<LlmRequest::RequestIdType> contextCompleteRequestIds;
    for (auto&& [request, future] : mSenderFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            contextCompleteRequestIds.push_back(request->mRequestId);
        }
    }

    // Synchronize across TP ranks (RNN doesn't use attention DP)
    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;
    if (syncComm && syncComm->getSize() > 1)
    {
        auto gatherRequestIdVec = gatherRequestIds(syncComm, contextCompleteRequestIds);
        for (auto&& requestId : gatherRequestIdVec)
        {
            frequencyMap[requestId]++;
        }
    }
    else
    {
        for (auto&& requestId : contextCompleteRequestIds)
        {
            frequencyMap[requestId]++;
        }
    }

    // Sort by frequency (descending) to prioritize requests complete on all ranks
    std::vector<std::pair<LlmRequest::RequestIdType, int>> freqVec(frequencyMap.begin(), frequencyMap.end());
    std::sort(
        freqVec.begin(), freqVec.end(), [](auto const& left, auto const& right) { return left.second > right.second; });

    // Collect requests that are complete on ALL ranks
    int syncSize = mGroupTensorParaComm ? mGroupTensorParaComm->getSize() : 1;
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    for (auto&& [requestId, freq] : freqVec)
    {
        if (freq == syncSize)
        {
            toCompleteIdSet.insert(requestId);
        }
    }

    // Ensure at least atLeastRequestNum requests are in the complete set
    for (auto it = mSenderFutures.begin();
         atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()) && it != mSenderFutures.end(); ++it)
    {
        auto& [request, future] = *it;
        toCompleteIdSet.insert(request->mRequestId);
    }

    // Complete all requests in toCompleteIdSet
    for (auto it = mSenderFutures.begin(); it != mSenderFutures.end();)
    {
        auto& [request, future] = *it;
        if (blockAll || (toCompleteIdSet.find(request->mRequestId) != toCompleteIdSet.end()))
        {
            try
            {
                auto status = future.wait_for(std::chrono::milliseconds(senderFutureTimeoutMs.value_or(0)));
                if (status == std::future_status::ready || !senderFutureTimeoutMs.has_value())
                {
                    future.get();
                    request->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
                    it = mSenderFutures.erase(it);
                }
                else if (status == std::future_status::timeout)
                {
                    TLLM_LOG_WARNING("Timed out waiting for context RNN cache transfer after %d milliseconds.",
                        senderFutureTimeoutMs.value());
                    ++it;
                }
                else
                {
                    TLLM_LOG_ERROR(
                        "Future returned unexpected status for request %ld. Marking as error", request->mRequestId);
                    request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                    it = mSenderFutures.erase(it);
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Error occurred during context RNN cache transfer for request %ld: %s",
                    request->mRequestId, e.what());
                request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                it = mSenderFutures.erase(it);
            }
        }
        else
        {
            ++it;
        }
    }
}

void RnnCacheTransceiver::checkGenTransferStatus(std::optional<int> const& atLeastRequestNum)
{
    bool blockAll = !atLeastRequestNum.has_value();

    // Collect request IDs that are ready locally
    std::vector<LlmRequest::RequestIdType> genTransferReadyRequestIds;
    for (auto&& [request, future] : mRequesterFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            genTransferReadyRequestIds.push_back(request->mRequestId);
        }
    }
    auto syncComm = mGroupComm; // TODO: add attention DP

    // Synchronize across all ranks
    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;
    if (syncComm && syncComm->getSize() > 1)
    {
        auto gatherRequestIdVec = gatherRequestIds(syncComm, genTransferReadyRequestIds);
        for (auto&& requestId : gatherRequestIdVec)
        {
            frequencyMap[requestId]++;
        }
    }
    else
    {
        for (auto&& requestId : genTransferReadyRequestIds)
        {
            frequencyMap[requestId]++;
        }
    }

    std::vector<std::pair<LlmRequest::RequestIdType, int>> freqVec(frequencyMap.begin(), frequencyMap.end());
    std::sort(
        freqVec.begin(), freqVec.end(), [](auto const& left, auto const& right) { return left.second > right.second; });

    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;

    // First, add requests from freqVec until we have atLeastRequestNum
    size_t idx = 0;
    while (atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()))
    {
        if (idx >= freqVec.size())
        {
            break;
        }
        toCompleteIdSet.insert(freqVec.at(idx).first);
        TLLM_LOG_DEBUG("checkGenTransferStatus at least from freqVec requestId: %zu", freqVec.at(idx).first);
        idx++;
    }

    // Then fill remaining from mRequesterFutures (in insertion order)
    idx = 0;
    while (atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()))
    {
        if (idx >= mRequesterFutures.size())
        {
            break;
        }
        if (toCompleteIdSet.find(mRequesterFutures.at(idx).first->mRequestId) == toCompleteIdSet.end())
        {
            toCompleteIdSet.insert(mRequesterFutures.at(idx).first->mRequestId);
            TLLM_LOG_DEBUG("checkGenTransferStatus at least from RequesterFuture requestId: %zu atLeastRequestNum:%d",
                mRequesterFutures.at(idx).first->mRequestId, atLeastRequestNum.value_or(0));
        }
        idx++;
    }

    for (auto&& [requestId, freq] : freqVec)
    {
        if (freq == ((syncComm != nullptr) ? syncComm->getSize() : 1))
        {
            toCompleteIdSet.insert(requestId);
        }
        TLLM_LOG_DEBUG("checkGenTransferStatus freqVec requestId: %zu, freq:%d", requestId, freq);
    }

    TLLM_LOG_DEBUG("checkGenTransferStatus toCompleteIdSet size: %zu, atLeastRequestNum: %d", toCompleteIdSet.size(),
        atLeastRequestNum.value_or(0));

    for (auto it = mRequesterFutures.begin(); it != mRequesterFutures.end();)
    {
        if (blockAll || toCompleteIdSet.find(it->first->mRequestId) != toCompleteIdSet.end())
        {
            // TODO: implement rnn cache transfer metrics. Needs request.rnn_cache_size().
            try
            {
                it->second.get();
                it->first->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
                TLLM_LOG_DEBUG("RNN cache transfer complete for request %ld, context request ID: %ld",
                    it->first->mRequestId, it->first->getContextPhaseParams().value().getReqId());
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Error occurred during generation RNN cache transfer for request %ld: %s",
                    it->first->mRequestId, e.what());
                it->first->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            }
            it = mRequesterFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool RnnCacheTransceiver::checkGenTransferComplete() const
{
    return mRequesterFutures.empty();
}

bool RnnCacheTransceiver::cancelRequest(LlmRequest* llmRequest)
{
    if (llmRequest->isContextOnlyRequest())
    {
        return mCacheSender->cancelRequest(*llmRequest);
    }
    else if (llmRequest->isGenerationOnlyRequest())
    {
        return mCacheReceiver->cancelRequest(*llmRequest);
    }
    return false;
}

} // namespace tensorrt_llm::batch_manager
