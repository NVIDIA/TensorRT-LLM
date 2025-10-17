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

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/dataTransceiverImpl.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/mlaCacheFormatter.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

std::mutex CacheTransceiver::mDllMutex;

std::unique_ptr<BaseCacheTransceiver> CacheTransceiverFactory::createCacheTransceiver(
    kv_cache_manager::BaseKVCacheManager* cacheManager, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, executor::kv_cache::CacheState::AttentionType attentionType,
    std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig)
{
    if (!cacheTransceiverConfig.has_value() || !cacheTransceiverConfig.value().getBackendType().has_value())
    {
        TLLM_LOG_INFO("CacheTransceiver is disabled.");
        return nullptr;
    }
    auto backendType = cacheTransceiverConfig.value().getBackendType();
    if (backendType.value() == executor::CacheTransceiverConfig::BackendType::DEFAULT)
    {
        if (common::getEnvUseUCXKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::UCX;
            TLLM_LOG_INFO("Enable UCX KV cache transport.");
        }
        else if (common::getEnvUseNixlKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::NIXL;
            TLLM_LOG_INFO("Enable NIXL KV cache transport.");
        }
        else if (common::getEnvUseMPIKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::MPI;
            TLLM_LOG_INFO("Enable MPI KV cache transport.");
            TLLM_LOG_WARNING("MPI KV cache transport is deprecated, please use UCX or NIXL instead.");
        }
        else
        {
            backendType = executor::CacheTransceiverConfig::BackendType::UCX;
        }
    }
    cacheTransceiverConfig.value().setBackendType(backendType);

    executor::kv_cache::CacheState::ModelConfig cacheStateCfg{
        modelConfig.getNumKvHeadsPerLayer(), modelConfig.getSizePerHead(), modelConfig.getTokensPerBlock()};

    return std::make_unique<CacheTransceiver>(
        cacheManager, cacheStateCfg, worldConfig, modelConfig.getKvDataType(), attentionType, cacheTransceiverConfig);
}

CacheTransceiver::CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
    executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
    nvinfer1::DataType dataType, executor::kv_cache::CacheState::AttentionType attentionType,
    std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig)
    : mMpiGroupComm(std::addressof(tensorrt_llm::mpi::MpiComm::session()))
    , mCacheTransceiverConfig{cacheTransceiverConfig}
{
    using tensorrt_llm::batch_manager::kv_cache_manager::CacheFormatter;
    if (worldConfig.isPipelineParallel())
    {
        mMpiGroupPipeParaComm = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            mMpiGroupComm->split(worldConfig.getTensorParallelRank(), worldConfig.getPipelineParallelRank()));
    }
    if (worldConfig.isTensorParallel())
    {
        mMpiGroupTensorParaComm = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            mMpiGroupComm->split(worldConfig.getPipelineParallelRank(), worldConfig.getTensorParallelRank()));
    }
    int kvFactor = 2;
    if (cacheManager->getCacheType() == kv_cache_manager::CacheType::kSELFKONLY)
    {
        kvFactor = 1;
    }
    mCacheState = std::make_unique<executor::kv_cache::CacheState>(
        cacheStateModelCfg, worldConfig, dataType, attentionType, kvFactor);

    if (mCacheState->getParallelConfig().mEnableAttentionDP)
    {
        int TPSizeInDPGroup
            = mCacheState->getParallelConfig().mTensorParallelism / mCacheState->getParallelConfig().mDPsize;
        int DPSize = mCacheState->getParallelConfig().mDPsize;
        int TPRankInDPGroup = worldConfig.getTensorParallelRank() % TPSizeInDPGroup;

        int DPRank = (worldConfig.getRank() - TPSizeInDPGroup * DPSize * worldConfig.getPipelineParallelRank()
                         - TPRankInDPGroup)
            / TPSizeInDPGroup;
        // <PP,DP,TP>
        mMpiGroupDataComm
            = std::make_shared<tensorrt_llm::mpi::MpiComm>(mMpiGroupComm->split(DPRank, worldConfig.getRank()));
        if (worldConfig.isTensorParallel())
        {
            mMpiGroupTPInDPComm = std::make_shared<tensorrt_llm::mpi::MpiComm>(
                mMpiGroupComm->split(worldConfig.getRank() / TPSizeInDPGroup, worldConfig.getRank()));
        }
    }
    bool isMLA = attentionType == executor::kv_cache::CacheState::AttentionType::kMLA;
    TLLM_CHECK_WITH_INFO(mCacheTransceiverConfig.has_value(), "CacheTransceiverConfig is not set.");
    auto backendType = mCacheTransceiverConfig.value().getBackendType();
    TLLM_CHECK_WITH_INFO(
        backendType.has_value() && (backendType.value() != executor::CacheTransceiverConfig::BackendType::DEFAULT),
        " CacheTransceiverConfig::BackendType is not set.");

    std::optional<size_t> maxNumTokens = mCacheTransceiverConfig.value().getMaxTokensInBuffer();

    mCacheTransBufferManager = std::make_unique<kv_cache_manager::CacheTransBufferManager>(cacheManager, maxNumTokens);
    if (backendType.value() == executor::CacheTransceiverConfig::BackendType::UCX)
    {
        std::lock_guard<std::mutex> lock(mDllMutex);
        mWrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
        TLLM_CHECK_WITH_INFO(
            mWrapperLibHandle != nullptr, "UCX wrapper library is not open correctly. error : %s", dlerror());
        auto load_sym = [](void* handle, char const* name)
        {
            void* ret = dllGetSym(handle, name);
            TLLM_CHECK_WITH_INFO(ret != nullptr,
                "Unable to load UCX wrapper library symbol, possible cause is that TensorRT-LLM library is not "
                "built with UCX support, please rebuild in UCX-enabled environment.");
            return ret;
        };
        std::unique_ptr<tensorrt_llm::executor::kv_cache::ConnectionManager> (*makeUcxConnectionManager)();
        *(void**) (&makeUcxConnectionManager) = load_sym(mWrapperLibHandle, "makeUcxConnectionManager");
        mManager = makeUcxConnectionManager();
        TLLM_LOG_INFO("UCX Connection Manager created");
    }
    else if (backendType.value() == executor::CacheTransceiverConfig::BackendType::NIXL)
    {
        mManager = std::make_unique<tensorrt_llm::executor::kv_cache::AgentConnectionManager>(
            mCacheTransBufferManager.get());
        TLLM_LOG_INFO("NIXL Connection Manager created");
    }
    else if (backendType.value() == executor::CacheTransceiverConfig::BackendType::MPI)
    {
        mMpiWorldComm = std::addressof(tensorrt_llm::mpi::MpiComm::world());
        mManager = std::make_unique<executor::kv_cache::MpiConnectionManager>(mMpiWorldComm);
        TLLM_LOG_INFO("MPI Connection Manager created");
    }
    else
    {
        TLLM_THROW("Unsupported cache transceiver backend type ");
    }

    using tensorrt_llm::batch_manager::kv_cache_manager::MLACacheFormatter;
    auto makeFormatter = [cacheManager, isMLA, this]()
    { return createCacheFormatter(cacheManager, mCacheTransBufferManager.get(), isMLA); };

    mDataResponder = std::make_unique<DataResponder>(
        std::make_unique<DataSenderImpl>(mManager.get(), *mCacheState, worldConfig.getRank(), makeFormatter()));
    mDataRequester = std::make_unique<DataRequester>(
        std::make_unique<DataReceiverImpl>(mManager.get(), *mCacheState, worldConfig.getRank(), makeFormatter()));

    initializeCommState();
}

CacheTransceiver::~CacheTransceiver()
{
    if (mWrapperLibHandle)
    {
        std::lock_guard<std::mutex> lock(mDllMutex);
        dllClose(mWrapperLibHandle);
    }
}

void CacheTransceiver::initializeCommState()
{
    mCommState = std::addressof(mDataResponder->getCommState());
}

void CacheTransceiver::setContextState(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    auto contextState = std::make_unique<executor::DataTransceiverState>();
    contextState->setCommState(*mCommState);
    contextState->setCacheState(*mCacheState);
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

void CacheTransceiver::respondAndSendAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    llmRequest->setState(LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS);
    // If context phase params is already set, it means that the KV cache
    // transfer is already in progress.
    if (llmRequest->getContextPhaseParams().has_value())
    {
        if (llmRequest->getContextProgress() == nullptr)
        {
            TLLM_LOG_WARNING("Request %ld is already responding", llmRequest->mRequestId);
        }
        return;
    }
    setContextState(llmRequest);
    auto future = mDataResponder->respondAndSendAsync(*llmRequest);
    mResponderFutures.emplace_back(llmRequest, std::move(future));
}

void CacheTransceiver::respondAndSendLayerWise(
    RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress)
{
    for (auto const& llmRequest : requests)
    {
        TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
        TLLM_CHECK(!llmRequest->getContextPhaseParams().has_value());
        llmRequest->setContextProgress(progress);
        TLLM_LOG_DEBUG("Request %ld is being sent layer-wise.", llmRequest->mRequestId);

        llmRequest->setState(LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS);
        setContextState(llmRequest.get());
        auto future = mDataResponder->respondAndSendAsync(*llmRequest);
        mResponderFutures.emplace_back(llmRequest.get(), std::move(future));
    }
}

void CacheTransceiver::requestAndReceiveSync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    {
        auto future = mDataRequester->requestAndReceiveAsync(*llmRequest);
        future.get();
    }
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
}

void CacheTransceiver::requestAndReceiveAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());

    if (std::find_if(mRequesterFutures.begin(), mRequesterFutures.end(),
            [llmRequest](auto const& pair) { return pair.first->mRequestId == llmRequest->mRequestId; })
        != mRequesterFutures.end())
    {
        TLLM_LOG_WARNING("Request ID %zu is already in mRequestFutures.", llmRequest->mRequestId);
        return;
    }

    auto future = mDataRequester->requestAndReceiveAsync(*llmRequest);
    mRequesterFutures.emplace_back(llmRequest, std::move(future));
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS);
}

std::vector<LlmRequest::RequestIdType> gatherRequestIds(
    mpi::MpiComm const& mpiComm, std::vector<LlmRequest::RequestIdType> const& requestIds)
{
    int localSize = static_cast<int>(requestIds.size());
    std::vector<int> sizes(mpiComm.getSize());
    mpiComm.allgather(&localSize, sizes.data(), 1, mpi::MpiType::kINT32);
    // std::vector<LlmRequest::RequestIdType> all_data(total_size);
    std::vector<int> displs(mpiComm.getSize());
    int totalSize = 0;
    for (int i = 0; i < mpiComm.getSize(); i++)
    {
        displs[i] = totalSize;
        totalSize += sizes[i];
    }
    std::vector<LlmRequest::RequestIdType> retData(totalSize);
    mpiComm.allgatherv(requestIds.data(), static_cast<int>(requestIds.size()), mpi::MpiType::kUINT64, retData.data(),
        sizes, displs, mpi::MpiType::kUINT64);
    return retData;
}

void updateKVCacheTransferBW(mpi::MpiComm const& mpiComm, LlmRequest* request)
{
    namespace su = executor::serialize_utils;
    int worldSize = mpiComm.getSize();

    std::ostringstream oStream;
    su::serialize(request->getKvCacheTransferStart(), oStream);
    su::serialize(request->getKvCacheTransferEnd(), oStream);

    auto str = oStream.str();
    std::vector<char> sendBuffer(str.begin(), str.end());
    auto sendBufferSize = sendBuffer.size();
    auto recvBufferSize = sendBufferSize * worldSize;
    std::vector<char> recvBuffer(recvBufferSize);

    mpiComm.allgather(sendBuffer.data(), recvBuffer.data(), sendBufferSize, mpi::MpiType::kCHAR);

    su::VectorWrapBuf<char> strbuf(recvBuffer);
    std::istream is(&strbuf);

    auto minStartTime = executor::RequestPerfMetrics::TimePoint::max();
    auto maxEndTime = executor::RequestPerfMetrics::TimePoint::min();

    for (int rank = 0; rank < worldSize; rank++)
    {
        minStartTime = std::min(su::deserialize<executor::RequestPerfMetrics::TimePoint>(is), minStartTime);
        maxEndTime = std::max(su::deserialize<executor::RequestPerfMetrics::TimePoint>(is), maxEndTime);
    }

    // Handle KV cache size separately - gather all sizes to the leader rank
    std::size_t localKVCacheSize = request->getKvCacheSize();
    std::vector<std::size_t> allKVCacheSizes(worldSize, 0);

    mpiComm.allgather(&localKVCacheSize, allKVCacheSizes.data(), 1, mpi::MpiType::kUINT64);

    std::size_t totalKVCacheSize = 0;
    for (int rank = 0; rank < worldSize; rank++)
    {
        totalKVCacheSize += allKVCacheSizes[rank];
    }

    // Update the latest KV cache transfer time for leader rank
    if (mpiComm.getRank() == 0)
    {
        request->setKvCacheTransferStart(minStartTime);
        request->setKvCacheTransferEnd(maxEndTime);
        request->setKvCacheSize(totalKVCacheSize);
    }
}

void CacheTransceiver::checkContextTransferStatus(std::optional<int> const& atLeastRequestNum)
{
    bool blockAll = !atLeastRequestNum.has_value();
    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mMpiGroupTPInDPComm : mMpiGroupTensorParaComm;
    std::vector<LlmRequest::RequestIdType> contextCompleteRequestIds;
    for (auto&& [request, future] : mResponderFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            contextCompleteRequestIds.push_back(request->mRequestId);
        }
    }

    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;
    if ((syncComm) && syncComm->getSize() > 1)
    {
        auto gatherRequestIdVec = gatherRequestIds(*syncComm, contextCompleteRequestIds);
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
    std::vector<std::pair<LlmRequest::RequestIdType, int>> freqVec(frequencyMap.begin(), frequencyMap.end());

    std::sort(freqVec.begin(), freqVec.end(),
        [](std::pair<LlmRequest::RequestIdType, int> const& left,
            std::pair<LlmRequest::RequestIdType, int> const& right) { return left.second > right.second; });
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    for (auto&& [requestId, freq] : freqVec)
    {
        if (freq == ((syncComm) ? syncComm->getSize() : 1))
        {
            toCompleteIdSet.insert(requestId);
        }
    }

    // Make sure there are at least atLeastRequestNum requests in toCompleteIdSet.
    // This will preserve the order of insertion for KVCache transfer requests.
    for (auto it = mResponderFutures.begin();
         atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()) && it != mResponderFutures.end();
         ++it)
    {
        auto& [request, future] = *it;
        toCompleteIdSet.insert(request->mRequestId);
    }

    // Complete all the requests in toCompleteIdSet
    for (auto it = mResponderFutures.begin(); it != mResponderFutures.end();)
    {
        auto& [request, future] = *it;
        if (blockAll || (toCompleteIdSet.find(request->mRequestId) != toCompleteIdSet.end()))
        {
            future.get();
            request->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
            it = mResponderFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void CacheTransceiver::checkGenTransferStatus(std::optional<int> const& atLeastRequestNum)
{
    bool blockAll = !atLeastRequestNum.has_value();
    std::vector<LlmRequest::RequestIdType> genTransferReadyRequestIds;
    for (auto&& [request, future] : mRequesterFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            genTransferReadyRequestIds.push_back(request->mRequestId);
        }
    }
    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;

    std::vector<LlmRequest::RequestIdType> toBlockRequestIds;
    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mMpiGroupDataComm.get() : mMpiGroupComm;
    if ((syncComm) && syncComm->getSize() > 1)
    {
        auto gatherRequestIdVec = gatherRequestIds(*syncComm, genTransferReadyRequestIds);
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

    std::sort(freqVec.begin(), freqVec.end(),
        [](std::pair<LlmRequest::RequestIdType, int> const& left,
            std::pair<LlmRequest::RequestIdType, int> const& right) { return left.second > right.second; });
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    size_t idx = 0;
    while (atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()))
    {
        if (idx >= freqVec.size())
        {
            break;
        }
        toCompleteIdSet.insert(freqVec.at(idx).first);
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " checkGenTransferStatus at least from freqVec requestId: %zu ",
            freqVec.at(idx).first);
        idx++;
    }
    idx = 0;

    // insert order
    while (atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()))
    {
        if (idx >= mRequesterFutures.size())
        {
            break;
        }
        if (toCompleteIdSet.find(mRequesterFutures.at(idx).first->mRequestId) == toCompleteIdSet.end())
        {
            toCompleteIdSet.insert(mRequesterFutures.at(idx).first->mRequestId);
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                " checkGenTransferStatus at least from RequesterFuture requestId: %zu atLeastRequestNum:%d",
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
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " checkGenTransferStatus freqVec requestId: %zu,freq:%d  ",
            requestId, freq);
    }
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        " checkGenTransferStatus toCompleteIdSet size: %zu, atLeastRequestNum: %d ", toCompleteIdSet.size(),
        atLeastRequestNum.value_or(0));
    for (auto it = mRequesterFutures.begin(); it != mRequesterFutures.end();)
    {
        if (blockAll || toCompleteIdSet.find(it->first->mRequestId) != toCompleteIdSet.end())
        {
            it->second.get();

            // Gather the kv cache transfer time from all workers and update to leader rank
            if (!common::getEnvKVCacheTransferOutputPath().empty())
            {
                auto syncComm
                    = mCacheState->getParallelConfig().mEnableAttentionDP ? mMpiGroupDataComm.get() : mMpiGroupComm;
                updateKVCacheTransferBW(*syncComm, it->first);
            }
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                "**** it->first->mRequestId: %ld, context request ID: %ld ******** get feature ***",
                it->first->mRequestId, it->first->getContextPhaseParams().value().getReqId());
            it->first->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
            it = mRequesterFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool CacheTransceiver::checkGenTransferComplete() const
{
    return mRequesterFutures.empty();
}

} // namespace tensorrt_llm::batch_manager
