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
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheType.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/mlaCacheFormatter.h"
#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/batch_manager/rnnCacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

namespace
{
/// @brief Source the rank used as the prefix of TLLM_LOG_DEBUG / TLLM_LOG_WARNING
///        calls. We call useMPI() at log time so the same call site works
///        regardless of whether the world communicator is initialized via
///        MPI or via the torch process group.
inline int currentRankForLog()
{
    return useMPI() ? mpi::MpiComm::world().getRank() : tensorrt_llm::pg_utils::get_world_pg()->getRank();
}
} // namespace

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
        else if (common::getEnvUseMooncakeKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::MOONCAKE;
            TLLM_LOG_INFO("Enable MOONCAKE KV cache transport.");
        }
        else if (common::getEnvUseMPIKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::MPI;
            TLLM_LOG_INFO("Enable MPI KV cache transport.");
            TLLM_LOG_WARNING("MPI KV cache transport is deprecated, please use UCX or NIXL instead.");
        }
        else
        {
            backendType = executor::CacheTransceiverConfig::BackendType::NIXL;
        }
    }
    cacheTransceiverConfig.value().setBackendType(backendType);

    executor::kv_cache::CacheState::ModelConfig cacheStateCfg{
        modelConfig.getNumKvHeadsPerLayer(), modelConfig.getSizePerHead(), modelConfig.getTokensPerBlock()};

    auto ppSize = worldConfig.getPipelineParallelism();

    std::vector<SizeType32> attentionLayerNumPerPP(ppSize, 0);
    for (int ppRank = 0; ppRank < ppSize; ppRank++)
    {
        attentionLayerNumPerPP[ppRank] = modelConfig.getNbAttentionLayers(ppSize, ppRank);
    }

    return std::make_unique<CacheTransceiver>(cacheManager, cacheStateCfg, worldConfig, attentionLayerNumPerPP,
        modelConfig.getKvDataType(), attentionType, cacheTransceiverConfig);
}

CacheTransceiver::CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
    executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
    std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
    executor::kv_cache::CacheState::AttentionType attentionType,
    std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig,
    rnn_state_manager::RnnStateManager* rnnStateManager, std::vector<SizeType32> const& rnnLayerNumPerPP)
    : mCacheTransceiverConfig{cacheTransceiverConfig}
    , mRnnStateManager{rnnStateManager}
{
    using tensorrt_llm::batch_manager::kv_cache_manager::CacheFormatter;
    if (useMPI())
    {
        mGroupComm = std::make_shared<CacheTransceiverComm>(std::addressof(tensorrt_llm::mpi::MpiComm::session()));
    }
    else
    {
        mGroupComm = std::make_shared<CacheTransceiverComm>(tensorrt_llm::pg_utils::get_world_pg());
    }

    if (worldConfig.isTensorParallel() || worldConfig.isContextParallel())
    {
        mGroupTensorParaComm = std::make_shared<CacheTransceiverComm>(
            mGroupComm->split(worldConfig.getPipelineParallelRank(), worldConfig.getRank()));
    }
    int kvFactor = 2;
    if (cacheManager->getCacheType() == kv_cache_manager::CacheType::kSELFKONLY)
    {
        kvFactor = 1;
    }
    mCacheState = std::make_unique<executor::kv_cache::CacheState>(cacheStateModelCfg, worldConfig,
        attentionLayerNumPerPP, dataType, attentionType, kvFactor, cacheManager->isEnableBlockReuse(),
        cacheManager->isEnablePartialReuse(), cacheManager->isEnableIndexerKCache(),
        cacheManager->getIndexerKCacheIndexHeadDim(), cacheManager->getIndexerKCacheQuantBlockSize());

    if (mCacheState->getParallelConfig().mEnableAttentionDP)
    {
        int dpSize = mCacheState->getParallelConfig().mDPsize;

        // dpRank is derived from the tensor parallel rank, which already accounts for CP.
        // Layout: rank = ppRank * (TP * CP) + tpRank * CP + cpRank.
        // getTensorParallelRank() correctly extracts tpRank regardless of CP.
        int dpRank = mCacheState->getParallelConfig().mDPrank;
        // <PP,DP,TP,CP>
        mGroupDataComm = std::make_shared<CacheTransceiverComm>(mGroupComm->split(dpRank, worldConfig.getRank()));
        if (worldConfig.isTensorParallel() || worldConfig.isContextParallel())
        {
            // Group ranks with same (ppRank, dpRank) accounting for CP.
            mGroupTPInDPComm = std::make_shared<CacheTransceiverComm>(
                mGroupComm->split(worldConfig.getPipelineParallelRank() * dpSize + dpRank, worldConfig.getRank()));
        }
    }
    bool isMLA = attentionType == executor::kv_cache::CacheState::AttentionType::kMLA;
    TLLM_CHECK_WITH_INFO(mCacheTransceiverConfig.has_value(), "CacheTransceiverConfig is not set.");
    auto backendType = mCacheTransceiverConfig.value().getBackendType();
    TLLM_CHECK_WITH_INFO(
        backendType.has_value() && (backendType.value() != executor::CacheTransceiverConfig::BackendType::DEFAULT),
        " CacheTransceiverConfig::BackendType is not set.");

    std::optional<size_t> maxNumTokens = mCacheTransceiverConfig.value().getMaxTokensInBuffer();

    mCacheTransBufferManagers.push_back(
        std::make_unique<kv_cache_manager::CacheTransBufferManager>(cacheManager, maxNumTokens));
    if (isMLA && cacheManager->isEnableIndexerKCache())
    {
        mCacheTransBufferManagers.push_back(
            std::make_unique<kv_cache_manager::CacheTransBufferManager>(cacheManager, maxNumTokens, true));
    }

    // RNN specific setup
    if (mRnnStateManager != nullptr)
    {
        TLLM_LOG_DEBUG("Setting up RNN cache transfer components.");
        TLLM_CHECK(!rnnLayerNumPerPP.empty());

        mRnnCacheTransBufferManager
            = std::make_unique<rnn_state_manager::RnnCacheTransBufferManager>(mRnnStateManager, maxNumTokens);

        auto rnnModelCfg = mRnnStateManager->getRnnCacheStateModelConfig();

        auto const convStateDataType = mRnnStateManager->getConvStateDataType();
        auto const ssmStateDataType = mRnnStateManager->getSsmStateDataType();

        mCacheState->setRnnConfig(rnnModelCfg, rnnLayerNumPerPP, convStateDataType, ssmStateDataType);

        TLLM_LOG_INFO("RNN cache transfer components initialized.");
    }

    mCacheTransBufferManagerPtrs.clear();
    mCacheTransBufferManagerPtrs.reserve(mCacheTransBufferManagers.size() + (mRnnCacheTransBufferManager ? 1 : 0));
    for (auto& manager : mCacheTransBufferManagers)
    {
        mCacheTransBufferManagerPtrs.push_back(manager.get());
    }
    if (mRnnCacheTransBufferManager)
    {
        mCacheTransBufferManagerPtrs.push_back(mRnnCacheTransBufferManager.get());
    }

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
                "Unable to load UCX wrapper library symbol, possible cause is that TensorRT LLM library is not "
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
        auto rnnState
            = mCacheState->hasRnnConfig() ? std::make_optional(mCacheState->getRnnCacheState()) : std::nullopt;
        mManager = std::make_unique<tensorrt_llm::executor::kv_cache::AgentConnectionManager>(
            mCacheTransBufferManagerPtrs, *mCacheState, "nixl", rnnState);
        TLLM_LOG_INFO("NIXL Connection Manager created");
    }
    else if (backendType.value() == executor::CacheTransceiverConfig::BackendType::MOONCAKE)
    {
        auto rnnState
            = mCacheState->hasRnnConfig() ? std::make_optional(mCacheState->getRnnCacheState()) : std::nullopt;
        mManager = std::make_unique<tensorrt_llm::executor::kv_cache::AgentConnectionManager>(
            mCacheTransBufferManagerPtrs, *mCacheState, "mooncake", rnnState);
        TLLM_LOG_INFO("MOONCAKE Connection Manager created");
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

    auto makeFormatter = [cacheManager, isMLA, this]()
    {
        std::vector<kv_cache_manager::CacheTransBufferManager*> kvBufferPtrs;
        kvBufferPtrs.reserve(mCacheTransBufferManagers.size());
        for (auto& mgr : mCacheTransBufferManagers)
        {
            kvBufferPtrs.push_back(mgr.get());
        }
        return createCacheFormatter(cacheManager, kvBufferPtrs, isMLA);
    };

    auto makeRnnFormatter = [this]() -> std::unique_ptr<RnnCacheFormatter>
    {
        if (mRnnStateManager != nullptr && mRnnCacheTransBufferManager != nullptr)
        {
            return std::make_unique<RnnCacheFormatter>(mRnnStateManager, mRnnCacheTransBufferManager.get());
        }
        return nullptr;
    };

    auto makeCacheTransferLayer
        = [&]() { return CacheTransferLayer(*mCacheState, makeFormatter(), makeRnnFormatter()); };

    mCacheSender = std::make_unique<CacheSender>(mManager.get(), worldConfig.getRank(), makeCacheTransferLayer());
    mCacheReceiver = std::make_unique<CacheReceiver>(mManager.get(), worldConfig.getRank(), makeCacheTransferLayer());

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
    mCommState = std::addressof(mCacheSender->getCommState());
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

void CacheTransceiver::respondAndSendAsync(std::shared_ptr<LlmRequest> llmRequest)
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
    setContextState(llmRequest.get());
    auto future = mCacheSender->sendAsync(llmRequest);
    mSenderFutures.emplace_back(std::move(llmRequest), std::move(future));
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
        auto future = mCacheSender->sendAsync(llmRequest);
        mSenderFutures.emplace_back(llmRequest, std::move(future));
    }
}

void CacheTransceiver::requestAndReceiveSync(std::shared_ptr<LlmRequest> llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    {
        auto future = mCacheReceiver->receiveAsync(llmRequest);
        future.get();
    }
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
}

void CacheTransceiver::requestAndReceiveAsync(std::shared_ptr<LlmRequest> llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());

    if (std::find_if(mRequesterFutures.begin(), mRequesterFutures.end(),
            [llmRequest](auto const& pair) { return pair.first->mRequestId == llmRequest->mRequestId; })
        != mRequesterFutures.end())
    {
        TLLM_LOG_WARNING("Request ID %zu is already in mRequestFutures.", llmRequest->mRequestId);
        return;
    }

    // Record the transfer start synchronously, before pushing the future.
    // CacheReceiver::receiveAsync() spawns a background thread that does the
    // same thing inside requestSync(), but until that thread runs,
    // getKvCacheTransferStart() would return the default (epoch) time point.
    // checkGenTransferStatus's elapsed-time deadline check would then see a
    // huge elapsed value and falsely evict the entry. A preemptive set
    // here keeps the invariant "every entry in mRequesterFutures has a
    // valid transfer start time" and is overwritten by requestSync() with a
    // slightly later time — harmless for the deadline check.
    llmRequest->setKvCacheTransferStart(LlmRequest::getSteadyClockNow());
    auto future = mCacheReceiver->receiveAsync(llmRequest);
    // Order: setKvCacheTransferStart -> receiveAsync -> setState -> emplace.
    // setState is intentionally moved AFTER receiveAsync (the original code
    // set it before). This is safe today because the async worker spawned
    // by receiveAsync does not read llmRequest state at entry; the new
    // ordering tightens the IN_PROGRESS<->mRequesterFutures invariant and
    // ensures a throw from receiveAsync leaves the request out of both
    // sets atomically. Do not revert without verifying the worker still
    // does not depend on the IN_PROGRESS state being set at entry.
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS);
    mRequesterFutures.emplace_back(std::move(llmRequest), std::move(future));
}

std::vector<LlmRequest::RequestIdType> gatherRequestIds(
    std::shared_ptr<CacheTransceiverComm> const& mComm, std::vector<LlmRequest::RequestIdType> const& requestIds)
{
    int localSize = static_cast<int>(requestIds.size());
    std::vector<int> sizes(mComm->getSize());
    std::vector<LlmRequest::RequestIdType> retData;
    if (useMPI())
    {
        mComm->allgather(&localSize, sizes.data(), 1, mpi::MpiType::kINT32);
        std::vector<int> displs(mComm->getSize());
        size_t totalSize = 0;
        for (int i = 0; i < mComm->getSize(); i++)
        {
            displs[i] = totalSize;
            totalSize += sizes[i];
        }
        retData.resize(totalSize);
        mComm->allgatherv(requestIds.data(), static_cast<int>(requestIds.size()), mpi::MpiType::kUINT64, retData.data(),
            sizes, displs, mpi::MpiType::kUINT64);
    }
    else
    {
        mComm->allgather(&localSize, std::ref(sizes), {});
        size_t totalSize = std::accumulate(sizes.begin(), sizes.end(), 0);
        retData.resize(totalSize);
        mComm->allgatherv(std::ref(requestIds), std::ref(retData), std::cref(sizes), {});
    }
    return retData;
}

void updateKVCacheTransferBW(std::shared_ptr<CacheTransceiverComm> const& mComm, LlmRequest* request)
{
    namespace su = executor::serialize_utils;
    int worldSize = mComm->getSize();

    std::ostringstream oStream;
    su::serialize(request->getKvCacheTransferStart(), oStream);
    su::serialize(request->getKvCacheTransferEnd(), oStream);

    auto str = oStream.str();
    std::vector<char> sendBuffer(str.begin(), str.end());
    auto sendBufferSize = sendBuffer.size();
    auto recvBufferSize = sendBufferSize * worldSize;
    std::vector<char> recvBuffer(recvBufferSize);

    if (useMPI())
    {
        mComm->allgather(sendBuffer.data(), recvBuffer.data(), sendBufferSize, mpi::MpiType::kCHAR);
    }
    else
    {
        mComm->allgather(std::ref(sendBuffer), std::ref(recvBuffer), {});
    }

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

    if (useMPI())
    {
        mComm->allgather(&localKVCacheSize, allKVCacheSizes.data(), 1, mpi::MpiType::kUINT64);
    }
    else
    {
        mComm->allgather(&localKVCacheSize, std::ref(allKVCacheSizes), {});
    }

    std::size_t totalKVCacheSize = 0;
    for (int rank = 0; rank < worldSize; rank++)
    {
        totalKVCacheSize += allKVCacheSizes[rank];
    }

    // Update the latest KV cache transfer time for leader rank
    if (mComm->getRank() == 0)
    {
        request->setKvCacheTransferStart(minStartTime);
        request->setKvCacheTransferEnd(maxEndTime);
        request->setKvCacheSize(totalKVCacheSize);
    }
}

RequestStatuses CacheTransceiver::checkContextTransferStatus(
    std::optional<int> const& atLeastRequestNum, bool markComplete)
{
    bool blockAll = !atLeastRequestNum.has_value();
    std::optional<int> senderFutureTimeoutMs = std::nullopt;
    std::optional<int> kvTransferTimeoutMs = std::nullopt;
    if (mCacheTransceiverConfig.has_value())
    {
        // The overall transfer deadline applies in both blockAll and polling modes; a
        // stuck sender must not hang indefinitely regardless of how callers invoke this.
        kvTransferTimeoutMs = mCacheTransceiverConfig->getKvTransferTimeoutMs();
        // The per-iteration poll timeout is only relevant when the caller wants to
        // return after checking at least `atLeastRequestNum` entries.
        if (!blockAll)
        {
            senderFutureTimeoutMs = mCacheTransceiverConfig->getKvTransferSenderFutureTimeoutMs();
        }
    }

    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupTPInDPComm : mGroupTensorParaComm;
    std::vector<LlmRequest::RequestIdType> contextCompleteRequestIds;
    for (auto&& [request, future] : mSenderFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            contextCompleteRequestIds.push_back(request->mRequestId);
        }
    }

    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;
    if ((syncComm) && syncComm->getSize() > 1)
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
    for (auto it = mSenderFutures.begin();
         atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()) && it != mSenderFutures.end(); ++it)
    {
        auto& [request, future] = *it;
        toCompleteIdSet.insert(request->mRequestId);
    }

    RequestStatuses requestsStatus{};

    // Complete all the requests in toCompleteIdSet
    for (auto it = mSenderFutures.begin(); it != mSenderFutures.end();)
    {
        auto& [request, future] = *it;
        // Enforce the overall deadline for every entry on every invocation,
        // independent of the readiness gate below. `toCompleteIdSet` is
        // populated from the consensus freqVec (entries that were ready on
        // every rank) plus up to `atLeastRequestNum` insertion-order entries.
        // A stuck sender whose future never becomes ready is not in the
        // consensus set, so with `atLeastRequestNum=0` it would otherwise
        // escape the deadline check entirely and pin KV blocks forever.
        // mSenderFutures holds shared_ptr<LlmRequest>, so the request is kept
        // alive for every C++ access here regardless of Python-side
        // termination timing.
        if (kvTransferTimeoutMs.has_value())
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                LlmRequest::getSteadyClockNow() - request->getKvCacheTransferStart());
            auto elapsedMs = static_cast<long>(elapsed.count());
            if (elapsedMs > kvTransferTimeoutMs.value())
            {
                TLLM_LOG_WARNING(
                    "Context KV cache transfer for request %ld exceeded total timeout: "
                    "elapsed %ld ms > limit %d ms. Marking as error.",
                    request->mRequestId, elapsedMs, kvTransferTimeoutMs.value());
                // Defense-in-depth sender-side cancel. Sender zombies empirically
                // unwind on peer teardown (decode-pod restart), but in general
                // CacheSender::cancelRequest clears mReadyResponses /
                // mCancelledRequests bookkeeping so a subsequent re-enqueue
                // or telemetry path doesn't see the stale request.
                mCacheSender->cancelRequest(*request);
                request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                requestsStatus.errorRequestIds.insert(request->mRequestId);
                it = mSenderFutures.erase(it);
                continue;
            }
        }
        if (blockAll || (toCompleteIdSet.find(request->mRequestId) != toCompleteIdSet.end()))
        {
            try
            {
                // Wait for up to a specified timeout (0 means a single non-blocking poll in blockAll mode).
                auto status = future.wait_for(std::chrono::milliseconds(senderFutureTimeoutMs.value_or(0)));
                if (status == std::future_status::ready)
                {
                    future.get();
                    requestsStatus.completedRequestIds.insert(request->mRequestId);
                    if (markComplete)
                    {
                        request->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
                    }
                    it = mSenderFutures.erase(it);
                }
                else if (status == std::future_status::timeout)
                {
                    // The overall deadline was already enforced unconditionally
                    // at the top of this loop, so if we reach here the deadline
                    // has not yet passed for this entry.
                    if (senderFutureTimeoutMs.has_value())
                    {
                        TLLM_LOG_WARNING("Timed out waiting for context KV cache transfer after %d milliseconds.",
                            senderFutureTimeoutMs.value());
                        ++it;
                    }
                    else
                    {
                        // blockAll mode with no deadline exceeded: block on get() as the
                        // caller intends, but the deadline will be re-checked on each entry
                        // of subsequent outer invocations.
                        future.get();
                        requestsStatus.completedRequestIds.insert(request->mRequestId);
                        if (markComplete)
                        {
                            request->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
                        }
                        it = mSenderFutures.erase(it);
                    }
                }
                else
                {
                    TLLM_LOG_ERROR(
                        "Future returned unexpected status for request %ld. Marking as error", request->mRequestId);

                    request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                    requestsStatus.errorRequestIds.insert(request->mRequestId);
                    it = mSenderFutures.erase(it);
                }
            }
            catch (std::exception const& e)
            {
                // mSenderFutures holds shared_ptr<LlmRequest>, so the request
                // is alive here regardless of Python-side termination timing.
                // Report as error so Python can call end_transfer to unpin
                // blocks.
                TLLM_LOG_WARNING("Error during context transfer for request %ld: %s. Marking as error.",
                    request->mRequestId, e.what());
                request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                requestsStatus.errorRequestIds.insert(request->mRequestId);
                it = mSenderFutures.erase(it);
            }
        }
        else
        {
            ++it;
        }
    }

    if (!requestsStatus.completedRequestIds.empty() || !requestsStatus.errorRequestIds.empty())
    {
        TLLM_LOG_DEBUG(
            "checkContextTransferStatus done: completed=%zu, errors=%zu, "
            "mSenderFutures.size()=%zu",
            requestsStatus.completedRequestIds.size(), requestsStatus.errorRequestIds.size(), mSenderFutures.size());
    }

    return requestsStatus;
}

void CacheTransceiver::checkGenTransferStatus(std::optional<int> const& atLeastRequestNum)
{
    bool blockAll = !atLeastRequestNum.has_value();
    std::optional<int> senderFutureTimeoutMs = std::nullopt;
    std::optional<int> kvTransferTimeoutMs = std::nullopt;
    if (mCacheTransceiverConfig.has_value())
    {
        // The overall transfer deadline applies in both blockAll and polling modes; a
        // stuck receiver must not hang indefinitely regardless of how callers invoke
        // this. Without this, mRequesterFutures would accumulate stuck entries,
        // pinning generation-side KV blocks and eventually exhausting the pool.
        kvTransferTimeoutMs = mCacheTransceiverConfig->getKvTransferTimeoutMs();
        // The per-iteration poll timeout is only relevant when the caller wants to
        // return after checking at least `atLeastRequestNum` entries.
        if (!blockAll)
        {
            senderFutureTimeoutMs = mCacheTransceiverConfig->getKvTransferSenderFutureTimeoutMs();
        }
    }

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
    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupDataComm : mGroupComm;
    if ((syncComm) && syncComm->getSize() > 1)
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
        TLLM_LOG_DEBUG(currentRankForLog(), " checkGenTransferStatus at least from freqVec requestId: %zu ",
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
            TLLM_LOG_DEBUG(currentRankForLog(),
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
        TLLM_LOG_DEBUG(
            currentRankForLog(), " checkGenTransferStatus freqVec requestId: %zu,freq:%d  ", requestId, freq);
    }
    TLLM_LOG_DEBUG(currentRankForLog(), " checkGenTransferStatus toCompleteIdSet size: %zu, atLeastRequestNum: %d ",
        toCompleteIdSet.size(), atLeastRequestNum.value_or(0));
    // Helper: finalize a generation-side transfer on the happy path.
    // Called in two places: status == ready, and blockAll fall-through
    // from the timeout branch.
    auto const completeEntry = [this](std::shared_ptr<LlmRequest> const& request)
    {
        request->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
        if (!common::getEnvKVCacheTimeOutputPath().empty())
        {
            auto transferSyncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupDataComm : mGroupComm;
            updateKVCacheTransferBW(transferSyncComm, request.get());
        }
        TLLM_LOG_DEBUG(currentRankForLog(),
            "**** it->first->mRequestId: %ld, context request ID: %ld ******** get feature ***", request->mRequestId,
            request->getContextPhaseParams().value().getReqId());
    };

    std::size_t numCompleted = 0;
    std::size_t numErrored = 0;
    for (auto it = mRequesterFutures.begin(); it != mRequesterFutures.end();)
    {
        auto& request = it->first; // std::shared_ptr<LlmRequest> const& — our own strong ref
        auto& future = it->second;
        // Enforce the overall deadline for every entry on every invocation,
        // independent of the readiness gate below. In polling mode (blockAll ==
        // false) `toCompleteIdSet` is populated from the initial
        // `wait_for(0)` sweep and only contains receivers whose futures were
        // already ready. A receiver whose peer sender evicted the transfer
        // (see checkContextTransferStatus above) never becomes ready, so
        // gating the deadline check behind `toCompleteIdSet` would let stuck
        // entries accumulate in `mRequesterFutures` and pin generation-side
        // KV blocks forever.
        //
        // Lifetime: mRequesterFutures holds shared_ptr<LlmRequest>, so the
        // request is kept alive for every C++ access here regardless of
        // whether Python has already dropped its active_requests reference.
        // This is the load-bearing invariant for the lifetime of all
        // accesses below; the older raw-pointer design relied on a Python
        // guard to delay _terminate_request and produced an orphan-induced
        // KV-block leak when the guard interacted with stuck transfers.
        if (kvTransferTimeoutMs.has_value())
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                LlmRequest::getSteadyClockNow() - request->getKvCacheTransferStart());
            auto elapsedMs = static_cast<long>(elapsed.count());
            if (elapsedMs > kvTransferTimeoutMs.value())
            {
                bool const firstTimeout = mTimedOutRequesterIds.insert(request->mRequestId).second;
                if (firstTimeout)
                {
                    TLLM_LOG_WARNING(
                        "Generation KV cache transfer for request %ld exceeded total timeout: "
                        "elapsed %ld ms > limit %d ms. Requesting cancellation.",
                        request->mRequestId, elapsedMs, kvTransferTimeoutMs.value());
                    // cancelRequest requests worker unwind, but it is not a
                    // quiescence proof. Keep the future tracked until the
                    // worker future becomes ready, otherwise Python could
                    // free KV resources while the worker/transport may still
                    // reference the advertised buffers.
                    mCacheReceiver->cancelRequest(*request);
                }
                if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    try
                    {
                        future.get();
                    }
                    catch (std::exception const& e)
                    {
                        TLLM_LOG_WARNING(
                            "Generation KV cache transfer for timed-out request %ld finished with error: %s",
                            request->mRequestId, e.what());
                    }
                    request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                    mTimedOutRequesterIds.erase(request->mRequestId);
                    it = mRequesterFutures.erase(it);
                    ++numErrored;
                }
                else
                {
                    if (firstTimeout)
                    {
                        ++numErrored;
                    }
                    ++it;
                }
                continue;
            }
        }
        if (blockAll || toCompleteIdSet.find(request->mRequestId) != toCompleteIdSet.end())
        {
            try
            {
                // Wait for up to a specified timeout (0 means a single non-blocking
                // poll in blockAll mode).
                auto status = future.wait_for(std::chrono::milliseconds(senderFutureTimeoutMs.value_or(0)));
                if (status == std::future_status::ready)
                {
                    future.get();
                    if (request->getState() == LlmRequestState::kDISAGG_TRANS_ERROR)
                    {
                        TLLM_LOG_WARNING(
                            "Generation KV cache transfer for request %ld finished after an error state was set.",
                            request->mRequestId);
                        ++numErrored;
                    }
                    else
                    {
                        completeEntry(request);
                        ++numCompleted;
                    }
                    mTimedOutRequesterIds.erase(request->mRequestId);
                    it = mRequesterFutures.erase(it);
                }
                else if (status == std::future_status::timeout)
                {
                    // The overall deadline was already enforced unconditionally at
                    // the top of this loop, so if we reach here the deadline has
                    // not yet passed for this entry.
                    if (senderFutureTimeoutMs.has_value())
                    {
                        TLLM_LOG_WARNING(
                            "Timed out waiting for generation KV cache transfer for request %ld "
                            "after %d milliseconds (per-iteration).",
                            request->mRequestId, senderFutureTimeoutMs.value());
                        ++it;
                    }
                    else
                    {
                        // blockAll mode with deadline not yet exceeded: block on get()
                        // as the caller requested. A hard cap across a stuck receiver
                        // only applies on subsequent outer invocations — callers that
                        // need an unconditional hard cap should pass atLeastRequestNum
                        // so the per-iteration poll timeout can drive the deadline
                        // check each tick.
                        future.get();
                        if (request->getState() == LlmRequestState::kDISAGG_TRANS_ERROR)
                        {
                            TLLM_LOG_WARNING(
                                "Generation KV cache transfer for request %ld finished after an error state was set.",
                                request->mRequestId);
                            ++numErrored;
                        }
                        else
                        {
                            completeEntry(request);
                            ++numCompleted;
                        }
                        mTimedOutRequesterIds.erase(request->mRequestId);
                        it = mRequesterFutures.erase(it);
                    }
                }
                else
                {
                    TLLM_LOG_ERROR("Future returned unexpected status for generation request %ld. Marking as error.",
                        request->mRequestId);
                    request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                    mTimedOutRequesterIds.erase(request->mRequestId);
                    it = mRequesterFutures.erase(it);
                    ++numErrored;
                }
            }
            catch (std::exception const& e)
            {
                // mRequesterFutures holds shared_ptr<LlmRequest>, so the
                // request is alive here regardless of Python-side
                // termination timing. Report as error so Python's
                // _check_cache_transfer_errors picks it up (via state ==
                // kDISAGG_TRANS_ERROR) and cleanly unwinds the request.
                TLLM_LOG_WARNING("Error during generation transfer for request %ld: %s. Marking as error.",
                    request->mRequestId, e.what());
                request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                mTimedOutRequesterIds.erase(request->mRequestId);
                it = mRequesterFutures.erase(it);
                ++numErrored;
            }
        }
        else
        {
            ++it;
        }
    }

    if (numCompleted > 0 || numErrored > 0)
    {
        TLLM_LOG_DEBUG("checkGenTransferStatus done: completed=%zu, errored=%zu, mRequesterFutures.size()=%zu",
            numCompleted, numErrored, mRequesterFutures.size());
    }
}

bool CacheTransceiver::checkGenTransferComplete() const
{
    return mRequesterFutures.empty();
}

bool CacheTransceiver::cancelRequest(std::shared_ptr<LlmRequest> llmRequest)
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
