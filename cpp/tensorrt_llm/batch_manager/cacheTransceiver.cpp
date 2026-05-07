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
    // Note: no admit-time deadline is captured. The per-request KV
    // transfer timeout (kvTransferTimeoutMs) is anchored to
    // LlmRequest::getKvCacheActualTransferStart(), set by the
    // formatters once the worker has acquired a transfer-buffer slot.
    mSenderFutures.push_back(TrackedFuture{std::move(llmRequest), std::move(future), false, false});
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
        mSenderFutures.push_back(TrackedFuture{llmRequest, std::move(future), false, false});
    }
}

void CacheTransceiver::requestAndReceiveSync(std::shared_ptr<LlmRequest> llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    auto const reqId = llmRequest->mRequestId;
    {
        auto future = mCacheReceiver->receiveAsync(llmRequest);

        // [wedge-trace] turn the previously unbounded future.get() into a
        // wait_for(interval) loop so the executor thread can log a
        // heartbeat when this synchronous gen-side path
        // (TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1) is blocked. The
        // worker thread's own heartbeats will tell us whether the wedge
        // is in NIXL/UCX or somewhere downstream. See
        // docs/source/features/disagg-kv-transfer-wedge-trace-logs.md.
        auto const heartbeatIntervalMs = common::getEnvDisaggWedgeTraceIntervalMs();
        if (heartbeatIntervalMs > 0)
        {
            auto const startTime = std::chrono::steady_clock::now();
            while (future.wait_for(std::chrono::milliseconds(heartbeatIntervalMs)) != std::future_status::ready)
            {
                auto const elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - startTime)
                                           .count();
                TLLM_LOG_WARNING(
                    "[wedge-trace] CacheTransceiver::requestAndReceiveSync executor thread blocked on "
                    "future.get for request %ld for %lld ms (TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP path). "
                    "Worker thread should be emitting its own [wedge-trace] heartbeat for the actual "
                    "wedge location.",
                    reqId, static_cast<long long>(elapsedMs));
            }
        }
        future.get();
    }
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
}

void CacheTransceiver::requestAndReceiveAsync(std::shared_ptr<LlmRequest> llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());

    auto const reqId = llmRequest->mRequestId;
    if (std::find_if(mRequesterFutures.begin(), mRequesterFutures.end(),
            [reqId](auto const& entry) { return entry.request->mRequestId == reqId; })
        != mRequesterFutures.end())
    {
        TLLM_LOG_WARNING("Request ID %zu is already in mRequestFutures.", reqId);
        return;
    }

    auto future = mCacheReceiver->receiveAsync(llmRequest);
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS);
    // Note: no admit-time deadline; see respondAndSendAsync.
    mRequesterFutures.push_back(TrackedFuture{std::move(llmRequest), std::move(future), false, false});
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

// ----- Internal helpers ----------------------------------------------------

void CacheTransceiver::updateHealthLocked()
{
    auto const now = std::chrono::steady_clock::now();
    auto const sinceProgress
        = std::chrono::duration_cast<std::chrono::milliseconds>(now - mLastProgressTime).count();
    bool wedged = sinceProgress > mGlobalProgressDeadlineMs.count() && (!mSenderFutures.empty() || !mRequesterFutures.empty());
    bool overBudget = mQuarantinedTransferCount > mQuarantineBudget;
    bool nextHealthy = !overBudget && !wedged;
    if (mIsHealthy && !nextHealthy)
    {
        TLLM_LOG_WARNING(
            "CacheTransceiver flipping to UNHEALTHY: quarantined=%zu budget=%zu sinceProgressMs=%lld deadlineMs=%lld",
            mQuarantinedTransferCount, mQuarantineBudget, static_cast<long long>(sinceProgress),
            static_cast<long long>(mGlobalProgressDeadlineMs.count()));
    }
    mIsHealthy = nextHealthy;
}

void CacheTransceiver::releaseTrackedFutureLocked(std::vector<TrackedFuture>& vec, size_t index)
{
    TLLM_CHECK(index < vec.size());
    {
        std::scoped_lock lk(mHealthMutex);
        if (vec[index].quarantined && mQuarantinedTransferCount > 0)
        {
            --mQuarantinedTransferCount;
        }
        mLastProgressTime = std::chrono::steady_clock::now();
        updateHealthLocked();
    }
    vec.erase(vec.begin() + index);
}

bool CacheTransceiver::maybeQuarantineLocked(TrackedFuture& entry, RequestStatuses* outStatus)
{
    if (entry.quarantined)
    {
        return false;
    }
    if (!mCacheTransceiverConfig.has_value())
    {
        return false;
    }
    auto const timeoutMs = mCacheTransceiverConfig->getKvTransferTimeoutMs();
    if (!timeoutMs.has_value())
    {
        return false;
    }
    // Defer the deadline check until the worker has actually started
    // the transfer (post slot acquisition). Requests still sitting in
    // the worker queue or waiting on a slot are not "stuck" in any
    // transport-relevant sense; quarantining them for queue starvation
    // turned every burst into a false-positive cascade — see
    // claude-swe/disagg-deadline-at-transfer-start-fix.md and the
    // d224/d227 findings (TCP at ~3 Gbps, 1-slot serialization, conc
    // 128) where this mis-accounting caused ~80% of requests to be
    // quarantined for queue wait rather than for any actual NIXL/UCX
    // wedge.
    if (!entry.request->hasKvCacheActualTransferStart())
    {
        return false;
    }
    auto const transferStart = entry.request->getKvCacheActualTransferStart();
    auto const deadline = transferStart + std::chrono::milliseconds(timeoutMs.value());
    auto const now = std::chrono::steady_clock::now();
    if (now < deadline)
    {
        return false;
    }
    entry.quarantined = true;
    entry.request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
    if (outStatus != nullptr)
    {
        outStatus->errorRequestIds.insert(entry.request->mRequestId);
    }
    {
        std::scoped_lock lk(mHealthMutex);
        ++mQuarantinedTransferCount;
        updateHealthLocked();
    }
    TLLM_LOG_WARNING(
        "Quarantining request %ld: per-request KV transfer deadline exceeded after slot acquisition "
        "(actual transfer time > %lld ms). Future stays pinned in C++ until worker reaches a final state.",
        entry.request->mRequestId, static_cast<long long>(timeoutMs.value()));
    return true;
}

// ----- Public API ----------------------------------------------------------

RequestStatuses CacheTransceiver::checkContextTransferStatus(
    std::optional<int> const& atLeastRequestNum, bool markComplete)
{
    // Non-blocking poll. The historical "blockAll" semantics (no
    // atLeastRequestNum, fall through to future.get on every entry) are
    // gone: a wedged worker thread must never freeze the executor event
    // loop. Use @ref drainContextTransferStatus for shutdown drain.
    return checkContextTransferStatusImpl(atLeastRequestNum, markComplete, /*allowBlocking=*/false);
}

RequestStatuses CacheTransceiver::drainContextTransferStatus(bool markComplete)
{
    // Blocking — only safe on shutdown/teardown paths.
    return checkContextTransferStatusImpl(std::nullopt, markComplete, /*allowBlocking=*/true);
}

RequestStatuses CacheTransceiver::checkContextTransferStatusImpl(
    std::optional<int> const& atLeastRequestNum, bool markComplete, bool allowBlocking)
{
    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupTPInDPComm : mGroupTensorParaComm;

    // Pass 1: which futures are READY right now?
    std::vector<LlmRequest::RequestIdType> contextCompleteRequestIds;
    for (auto const& entry : mSenderFutures)
    {
        if (entry.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            contextCompleteRequestIds.push_back(entry.request->mRequestId);
        }
    }

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

    std::vector<std::pair<LlmRequest::RequestIdType, int>> freqVec(frequencyMap.begin(), frequencyMap.end());
    std::sort(freqVec.begin(), freqVec.end(),
        [](auto const& left, auto const& right) { return left.second > right.second; });

    int const expectedFreq = syncComm ? syncComm->getSize() : 1;
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    for (auto const& [requestId, freq] : freqVec)
    {
        if (freq == expectedFreq)
        {
            toCompleteIdSet.insert(requestId);
        }
    }

    // For atLeastRequestNum > 0 we may need to admit additional requests.
    // CRITICAL: prefer entries whose future is already ready. Never select
    // an unready entry just to satisfy the count — that is the
    // "future.get on selected-but-unready entry" wedge the analysis doc
    // calls out. If we run out of ready entries we simply return what we
    // have; the caller polls again next iteration.
    if (atLeastRequestNum.has_value())
    {
        for (auto const& entry : mSenderFutures)
        {
            if (static_cast<int>(toCompleteIdSet.size()) >= atLeastRequestNum.value())
            {
                break;
            }
            if (entry.future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
            {
                continue;
            }
            toCompleteIdSet.insert(entry.request->mRequestId);
        }
    }

    RequestStatuses requestsStatus{};
    bool madeProgress = false;
    for (size_t i = 0; i < mSenderFutures.size();)
    {
        auto& entry = mSenderFutures[i];
        bool const selected = toCompleteIdSet.find(entry.request->mRequestId) != toCompleteIdSet.end();
        bool const isReady = entry.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;

        if (selected && isReady)
        {
            try
            {
                entry.future.get();
                requestsStatus.completedRequestIds.insert(entry.request->mRequestId);
                if (markComplete)
                {
                    entry.request->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Error occurred during context transfer for request %ld: %s",
                    entry.request->mRequestId, e.what());
                entry.request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                requestsStatus.errorRequestIds.insert(entry.request->mRequestId);
            }
            releaseTrackedFutureLocked(mSenderFutures, i);
            madeProgress = true;
            continue;
        }

        if (allowBlocking)
        {
            // Drain path: shutdown is happening. Wait for the worker to
            // finish so we can release resources cleanly. Note this still
            // never blocks on the executor thread — only shutdown calls
            // this codepath.
            try
            {
                entry.future.get();
                requestsStatus.completedRequestIds.insert(entry.request->mRequestId);
                if (markComplete)
                {
                    entry.request->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Error occurred during context transfer drain for request %ld: %s",
                    entry.request->mRequestId, e.what());
                entry.request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                requestsStatus.errorRequestIds.insert(entry.request->mRequestId);
            }
            releaseTrackedFutureLocked(mSenderFutures, i);
            madeProgress = true;
            continue;
        }

        // Per-entry timeout: surface error to caller, but keep the future
        // pinned. Worker may still be writing; releasing the entry now
        // would let Python free request resources that NIXL/UCX may still
        // touch. The future stays here until the worker finishes (or the
        // global progress deadline marks the transceiver unhealthy and
        // orchestration restarts).
        maybeQuarantineLocked(entry, &requestsStatus);
        ++i;
    }

    if (madeProgress)
    {
        std::scoped_lock lk(mHealthMutex);
        mLastProgressTime = std::chrono::steady_clock::now();
        updateHealthLocked();
    }
    else
    {
        std::scoped_lock lk(mHealthMutex);
        updateHealthLocked();
    }
    return requestsStatus;
}

void CacheTransceiver::checkGenTransferStatus(std::optional<int> const& atLeastRequestNum)
{
    checkGenTransferStatusImpl(atLeastRequestNum, /*allowBlocking=*/false);
}

void CacheTransceiver::drainGenTransferStatus()
{
    checkGenTransferStatusImpl(std::nullopt, /*allowBlocking=*/true);
}

void CacheTransceiver::checkGenTransferStatusImpl(std::optional<int> const& atLeastRequestNum, bool allowBlocking)
{
    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupDataComm : mGroupComm;

    std::vector<LlmRequest::RequestIdType> genTransferReadyRequestIds;
    for (auto const& entry : mRequesterFutures)
    {
        if (entry.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            genTransferReadyRequestIds.push_back(entry.request->mRequestId);
        }
    }

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
    std::sort(freqVec.begin(), freqVec.end(),
        [](auto const& left, auto const& right) { return left.second > right.second; });

    int const expectedFreq = syncComm ? syncComm->getSize() : 1;
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    for (auto const& [requestId, freq] : freqVec)
    {
        if (freq == expectedFreq)
        {
            toCompleteIdSet.insert(requestId);
        }
    }

    if (atLeastRequestNum.has_value())
    {
        for (auto const& entry : mRequesterFutures)
        {
            if (static_cast<int>(toCompleteIdSet.size()) >= atLeastRequestNum.value())
            {
                break;
            }
            if (entry.future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
            {
                continue;
            }
            toCompleteIdSet.insert(entry.request->mRequestId);
        }
    }

    bool madeProgress = false;
    for (size_t i = 0; i < mRequesterFutures.size();)
    {
        auto& entry = mRequesterFutures[i];
        bool const selected = toCompleteIdSet.find(entry.request->mRequestId) != toCompleteIdSet.end();
        bool const isReady = entry.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;

        if (selected && isReady)
        {
            try
            {
                entry.future.get();
                entry.request->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
                if (!common::getEnvKVCacheTimeOutputPath().empty())
                {
                    updateKVCacheTransferBW(syncComm, entry.request.get());
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Error occurred during generation transfer for request %ld: %s",
                    entry.request->mRequestId, e.what());
                entry.request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            }
            releaseTrackedFutureLocked(mRequesterFutures, i);
            madeProgress = true;
            continue;
        }

        if (allowBlocking)
        {
            try
            {
                entry.future.get();
                entry.request->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
                if (!common::getEnvKVCacheTimeOutputPath().empty())
                {
                    updateKVCacheTransferBW(syncComm, entry.request.get());
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Error occurred during generation transfer drain for request %ld: %s",
                    entry.request->mRequestId, e.what());
                entry.request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            }
            releaseTrackedFutureLocked(mRequesterFutures, i);
            madeProgress = true;
            continue;
        }

        maybeQuarantineLocked(entry, /*outStatus=*/nullptr);
        ++i;
    }

    {
        std::scoped_lock lk(mHealthMutex);
        if (madeProgress)
        {
            mLastProgressTime = std::chrono::steady_clock::now();
        }
        updateHealthLocked();
    }
}

bool CacheTransceiver::checkGenTransferComplete() const
{
    return mRequesterFutures.empty();
}

TransferCancelResult CacheTransceiver::cancelRequestStructured(std::shared_ptr<LlmRequest> llmRequest)
{
    TLLM_CHECK(llmRequest != nullptr);

    auto const reqId = llmRequest->mRequestId;
    bool const sender = llmRequest->isContextOnlyRequest();
    bool const receiver = llmRequest->isGenerationOnlyRequest();
    if (!sender && !receiver)
    {
        return TransferCancelResult::kNotCancellable;
    }

    // Pre-advertise check first: a request that is still queued in the
    // sender / receiver's pending list has not exposed any buffer to a
    // peer and is always safe to release — even when the transceiver
    // is otherwise unhealthy. Releasing pre-advertise requests during
    // an unhealthy window reduces the resource pressure on the wedged
    // backend rather than adding to it.
    if (sender && mCacheSender->cancelRequest(*llmRequest))
    {
        return TransferCancelResult::kCancelledBeforeAdvertise;
    }
    if (receiver && mCacheReceiver->cancelRequest(*llmRequest))
    {
        return TransferCancelResult::kCancelledBeforeAdvertise;
    }

    auto& futures = sender ? mSenderFutures : mRequesterFutures;
    for (size_t i = 0; i < futures.size(); ++i)
    {
        if (futures[i].request->mRequestId != reqId)
        {
            continue;
        }
        auto status = futures[i].future.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready)
        {
            return TransferCancelResult::kAlreadyComplete;
        }
        // Worker is mid-flight. If the transceiver is unhealthy the
        // caller must defer cleanup pending an orchestration restart;
        // otherwise it is just a normal in-flight cancel.
        return isHealthy() ? TransferCancelResult::kCancelRequestedInFlight
                           : TransferCancelResult::kBackendUnhealthy;
    }
    return TransferCancelResult::kNotFound;
}

bool CacheTransceiver::cancelRequest(std::shared_ptr<LlmRequest> llmRequest)
{
    auto result = cancelRequestStructured(std::move(llmRequest));
    return result == TransferCancelResult::kCancelledBeforeAdvertise || result == TransferCancelResult::kAlreadyComplete;
}

bool CacheTransceiver::isHealthy() const
{
    std::scoped_lock lk(mHealthMutex);
    return mIsHealthy;
}

TransceiverHealth CacheTransceiver::getHealth() const
{
    std::scoped_lock lk(mHealthMutex);
    auto const sinceProgress = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - mLastProgressTime)
                                   .count();
    return TransceiverHealth{
        mIsHealthy,
        mQuarantinedTransferCount,
        mQuarantineBudget,
        sinceProgress,
        std::chrono::duration_cast<std::chrono::duration<double>>(mGlobalProgressDeadlineMs).count(),
    };
}

} // namespace tensorrt_llm::batch_manager
