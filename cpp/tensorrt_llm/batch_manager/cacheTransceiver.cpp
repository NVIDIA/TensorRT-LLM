/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

namespace
{

/// Generate a UUID-like hex string (e.g. "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
/// to uniquely identify a CacheTransceiver instance across gen instances.
std::string generateInstanceId()
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    uint64_t a = dis(gen);
    uint64_t b = dis(gen);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(8) << (a >> 32) << "-" << std::setw(4)
        << ((a >> 16) & 0xFFFF) << "-" << std::setw(4) << (a & 0xFFFF) << "-" << std::setw(4) << (b >> 48) << "-"
        << std::setw(12) << (b & 0xFFFFFFFFFFFF);
    return oss.str();
}

} // anonymous namespace

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

    // Generate instance ID on rank 0 and broadcast to all ranks in the session
    // so every rank in the same gen/ctx instance shares the same ID.
    {
        if (mGroupComm->getRank() == 0)
        {
            mInstanceId = generateInstanceId();
        }
        if (useMPI())
        {
            int len = static_cast<int>(mInstanceId.size());
            tensorrt_llm::mpi::MpiComm::session().bcast(&len, 1, mpi::MpiType::kINT32, 0);
            mInstanceId.resize(len);
            tensorrt_llm::mpi::MpiComm::session().bcast(mInstanceId.data(), len, mpi::MpiType::kCHAR, 0);
        }
        else
        {
            // PG path: rank 0 sends via allgather, others receive.
            constexpr int kUuidLen = 36;
            std::vector<char> sendBuf(kUuidLen, '\0');
            if (mGroupComm->getRank() == 0)
            {
                std::copy_n(mInstanceId.begin(), std::min<size_t>(mInstanceId.size(), kUuidLen), sendBuf.begin());
            }
            std::vector<char> recvBuf(kUuidLen * mGroupComm->getSize(), '\0');
            mGroupComm->allgather(std::ref(sendBuf), std::ref(recvBuf), {});
            // Take rank 0's segment.
            mInstanceId = std::string(recvBuf.begin(), recvBuf.begin() + kUuidLen);
        }
    }

    // Calibrate steady_clock across ranks so that cross-node allgather
    // in batchUpdateKVCacheTransferBW can compare time points.
    // Python's _set_global_steady_clock_offset writes to the nanobind
    // module's copy of sGlobalSteadyClockOffset, which is invisible to
    // libtensorrt_llm.so (separate inline-static instances across .so
    // boundaries).  We redo the calibration here in the C++ library.
    if (!LlmRequest::sGlobalSteadyClockOffset.has_value())
    {
        using Duration = LlmRequest::Duration;
        using TimePoint = LlmRequest::TimePoint;
        // Barrier + take local timestamp
        if (useMPI())
        {
            tensorrt_llm::mpi::MpiComm::session().barrier();
        }
        auto localNow = std::chrono::steady_clock::now();
        auto localNs = std::chrono::duration_cast<std::chrono::nanoseconds>(localNow.time_since_epoch()).count();

        // Allgather timestamps from all ranks
        std::vector<int64_t> allNs(mGroupComm->getSize(), 0);
        if (useMPI())
        {
            tensorrt_llm::mpi::MpiComm::session().allgather(&localNs, allNs.data(), 1, mpi::MpiType::kINT64);
        }
        else
        {
            mGroupComm->allgather(localNs, std::ref(allNs), {});
        }

        // Offset = rank0's timestamp - my timestamp (same formula as Python)
        auto offsetNs = allNs[0] - localNs;
        LlmRequest::sGlobalSteadyClockOffset = Duration(offsetNs);

        TLLM_LOG_INFO(mGroupComm->getRank(),
            "CacheTransceiver: set sGlobalSteadyClockOffset = %.6f sec for rank %d",
            static_cast<double>(offsetNs) / 1e9, mGroupComm->getRank());
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
    mCacheState
        = std::make_unique<executor::kv_cache::CacheState>(cacheStateModelCfg, worldConfig, attentionLayerNumPerPP,
            dataType, attentionType, kvFactor, cacheManager->isEnableBlockReuse(), cacheManager->isEnablePartialReuse(),
            cacheManager->isEnableIndexerKCache(), cacheManager->getIndexerKCacheIndexHeadDim(),
            cacheManager->getIndexerKCacheQuantBlockSize(), cacheManager->getIndexerKCacheUseFp4());

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

    mCacheSender
        = std::make_unique<CacheSender>(mManager.get(), worldConfig.getRank(), makeCacheTransferLayer(), mInstanceId);
    mCacheReceiver
        = std::make_unique<CacheReceiver>(mManager.get(), worldConfig.getRank(), makeCacheTransferLayer(), mInstanceId);

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
    auto future = mCacheSender->sendAsync(*llmRequest);
    mSenderFutures.emplace_back(llmRequest, std::move(future));
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
        auto future = mCacheSender->sendAsync(*llmRequest);
        mSenderFutures.emplace_back(llmRequest.get(), std::move(future));
    }
}

void CacheTransceiver::requestAndReceiveSync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    {
        auto future = mCacheReceiver->receiveAsync(*llmRequest);
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

    auto future = mCacheReceiver->receiveAsync(*llmRequest);
    mRequesterFutures.emplace_back(llmRequest, std::move(future));
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS);
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

void batchUpdateKVCacheTransferBW(
    std::shared_ptr<CacheTransceiverComm> const& comm, std::vector<LlmRequest*> const& requests)
{
    // Key-based merge: each rank serializes (requestId, start, end, size)
    // tuples and we use allgatherv so ranks may have different request counts.
    // The merge matches by requestId, not by position — this tolerates
    // ordering differences and count mismatches across ranks.

    namespace su = executor::serialize_utils;
    int const worldSize = comm->getSize();

    // --- Serialize local entries keyed by requestId ---
    std::size_t const numReqs = requests.size();

    std::ostringstream oStream;
    su::serialize(numReqs, oStream);
    for (auto* req : requests)
    {
        su::serialize(req->getContextPhaseParams().value().getReqId(), oStream);
        su::serialize(req->getKvCacheTransferStart(), oStream);
        su::serialize(req->getKvCacheTransferEnd(), oStream);
        su::serialize(req->getKvCacheSize(), oStream);
    }

    auto str = oStream.str();
    std::vector<char> sendBuffer(str.begin(), str.end());
    int const sendSize = static_cast<int>(sendBuffer.size());

    // --- Step 1: allgather per-rank buffer sizes ---
    std::vector<int> recvCounts(worldSize, 0);
    if (useMPI())
    {
        comm->allgather(&sendSize, recvCounts.data(), 1, mpi::MpiType::kINT32);
    }
    else
    {
        comm->allgather(sendSize, std::ref(recvCounts), {});
    }

    // --- Step 2: allgatherv the serialized data ---
    std::vector<int> displs(worldSize, 0);
    int totalRecvSize = 0;
    for (int r = 0; r < worldSize; ++r)
    {
        displs[r] = totalRecvSize;
        totalRecvSize += recvCounts[r];
    }
    std::vector<char> recvBuffer(totalRecvSize, 0);

    if (useMPI())
    {
        comm->allgatherv(
            sendBuffer.data(), sendSize, mpi::MpiType::kCHAR, recvBuffer.data(), recvCounts, displs, mpi::MpiType::kCHAR);
    }
    else
    {
        comm->allgatherv(std::ref(sendBuffer), std::ref(recvBuffer), recvCounts, {});
    }

    // --- Step 3: Deserialize and merge by requestId ---
    using TimePoint = executor::RequestPerfMetrics::TimePoint;
    using ReqIdType = LlmRequest::RequestIdType;

    struct MergedEntry
    {
        TimePoint minStart = TimePoint::max();
        TimePoint maxEnd = TimePoint::min();
        std::size_t totalSize = 0;
    };
    std::unordered_map<ReqIdType, MergedEntry> merged;

    su::VectorWrapBuf<char> strbuf(recvBuffer);
    std::istream is(&strbuf);

    for (int rank = 0; rank < worldSize; ++rank)
    {
        auto rankNumReqs = su::deserialize<std::size_t>(is);
        for (std::size_t i = 0; i < rankNumReqs; ++i)
        {
            auto rid = su::deserialize<ReqIdType>(is);
            auto start = su::deserialize<TimePoint>(is);
            auto end = su::deserialize<TimePoint>(is);
            auto size = su::deserialize<std::size_t>(is);

            auto& entry = merged[rid];
            entry.minStart = std::min(entry.minStart, start);
            entry.maxEnd = std::max(entry.maxEnd, end);
            entry.totalSize += size;
        }
    }

    // --- Step 4: Update local requests ---
    for (auto* req : requests)
    {
        auto reqId = req->getContextPhaseParams().value().getReqId();
        auto it = merged.find(reqId);
        if (it != merged.end())
        {
            req->setKvCacheTransferStart(it->second.minStart);
            req->setKvCacheTransferEnd(it->second.maxEnd);
            req->setKvCacheSize(it->second.totalSize);
        }
    }
}

RequestStatuses CacheTransceiver::checkContextTransferStatus(
    std::optional<int> const& atLeastRequestNum, bool markComplete)
{
    bool blockAll = !atLeastRequestNum.has_value();
    std::optional<int> senderFutureTimeoutMs = std::nullopt;
    // If blockAll is true, we want to block and not use a timeout
    if (!blockAll && mCacheTransceiverConfig.has_value())
    {
        senderFutureTimeoutMs = mCacheTransceiverConfig->getKvTransferSenderFutureTimeoutMs();
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
        if (blockAll || (toCompleteIdSet.find(request->mRequestId) != toCompleteIdSet.end()))
        {
            try
            {
                // Wait for up to a specified timeout
                auto status = future.wait_for(std::chrono::milliseconds(senderFutureTimeoutMs.value_or(0)));
                if (status == std::future_status::ready || !senderFutureTimeoutMs.has_value())
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
                    TLLM_LOG_WARNING("Timed out waiting for context KV cache transfer after %d milliseconds.",
                        senderFutureTimeoutMs.value());
                    ++it;
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
                TLLM_LOG_ERROR(
                    "Error occurred during context transfer for request %ld: %s", request->mRequestId, e.what());
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

    return requestsStatus;
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
        if (useMPI())
        {
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                " checkGenTransferStatus at least from freqVec requestId: %zu ", freqVec.at(idx).first);
        }
        else
        {
            TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                " checkGenTransferStatus at least from freqVec requestId: %zu ", freqVec.at(idx).first);
        }
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
            if (useMPI())
            {
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    " checkGenTransferStatus at least from RequesterFuture requestId: %zu atLeastRequestNum:%d",
                    mRequesterFutures.at(idx).first->mRequestId, atLeastRequestNum.value_or(0));
            }
            else
            {
                TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                    " checkGenTransferStatus at least from RequesterFuture requestId: %zu atLeastRequestNum:%d",
                    mRequesterFutures.at(idx).first->mRequestId, atLeastRequestNum.value_or(0));
            }
        }
        idx++;
    }
    for (auto&& [requestId, freq] : freqVec)
    {
        if (freq == ((syncComm != nullptr) ? syncComm->getSize() : 1))
        {
            toCompleteIdSet.insert(requestId);
        }
        if (useMPI())
        {
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " checkGenTransferStatus freqVec requestId: %zu,freq:%d  ",
                requestId, freq);
        }
        else
        {
            TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                " checkGenTransferStatus freqVec requestId: %zu,freq:%d  ", requestId, freq);
        }
    }
    if (useMPI())
    {
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            " checkGenTransferStatus toCompleteIdSet size: %zu, atLeastRequestNum: %d ", toCompleteIdSet.size(),
            atLeastRequestNum.value_or(0));
    }
    else
    {
        TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
            " checkGenTransferStatus toCompleteIdSet size: %zu, atLeastRequestNum: %d ", toCompleteIdSet.size(),
            atLeastRequestNum.value_or(0));
    }
    // Phase 1: Wait on futures and collect completed requests.
    std::vector<LlmRequest*> completedRequests;
    for (auto it = mRequesterFutures.begin(); it != mRequesterFutures.end();)
    {
        if (blockAll || toCompleteIdSet.find(it->first->mRequestId) != toCompleteIdSet.end())
        {
            try
            {
                it->second.get();
                it->first->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
                completedRequests.push_back(it->first);
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR(
                    "Error occurred during generation transfer for request %ld: %s", it->first->mRequestId, e.what());
                it->first->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            }
            if (useMPI())
            {
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    "**** it->first->mRequestId: %ld, context request ID: %ld ******** get feature ***",
                    it->first->mRequestId, it->first->getContextPhaseParams().value().getReqId());
            }
            else
            {
                TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                    "**** it->first->mRequestId: %ld, context request ID: %ld ******** get feature ***",
                    it->first->mRequestId, it->first->getContextPhaseParams().value().getReqId());
            }
            it = mRequesterFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Phase 2: Batch-sync timing across ranks in one allgather (instead of per-request).
    if (!completedRequests.empty() && !common::getEnvKVCacheTimeOutputPath().empty())
    {
        auto bwSyncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupDataComm : mGroupComm;
        batchUpdateKVCacheTransferBW(bwSyncComm, completedRequests);

        // Write gen-side transfer summary CSV
        {
            std::lock_guard<std::mutex> lock(mGenTransferSummaryMutex);
            if (!mGenTransferSummaryFile.is_open())
            {
                namespace fs = std::filesystem;
                auto outputPath = fs::path(common::getEnvKVCacheTimeOutputPath());
                fs::create_directories(outputPath);
                int rank
                    = useMPI() ? mpi::MpiComm::world().getRank() : tensorrt_llm::pg_utils::get_world_pg()->getRank();
                auto filePath
                    = outputPath / (mInstanceId + "_" + std::to_string(rank) + "_gen_transfer_summary.csv");
                mGenTransferSummaryFile.open(filePath);
                mGenTransferSummaryFile << "RequestID,gen_side_transfer_time(ms),kv_cache_size" << '\n';
            }
            for (auto* req : completedRequests)
            {
                auto reqId = req->getContextPhaseParams().value().getReqId();
                mGenTransferSummaryFile << reqId << "," << req->getKvCacheTransferTimeMS() << ","
                                        << req->getKvCacheSize() << '\n';
            }
            mGenTransferSummaryFile << std::flush;
        }
    }
}

bool CacheTransceiver::checkGenTransferComplete() const
{
    return mRequesterFutures.empty();
}

bool CacheTransceiver::cancelRequest(LlmRequest* llmRequest)
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
