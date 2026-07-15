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
#include "tensorrt_llm/batch_manager/transferStatusConsensus.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

std::mutex CacheTransceiver::mDllMutex;

namespace
{

using RequestIdType = LlmRequest::RequestIdType;

constexpr int kTransferFuturePollIntervalMs = 10;
constexpr int kContextConsensusPollIntervalMs = 1;

// Finite status checks are scheduler polls, not terminal deadlines. Pure polls
// use short slices; calls that ask for at least one completion keep bounded
// backpressure by waiting up to the configured future timeout.
std::chrono::milliseconds getTransferFutureWaitInterval(
    std::optional<int> const& configuredTimeoutMs, bool const needsProgress)
{
    auto waitMs = kTransferFuturePollIntervalMs;
    if (configuredTimeoutMs.has_value())
    {
        waitMs = needsProgress ? configuredTimeoutMs.value()
                               : std::min(configuredTimeoutMs.value(), kTransferFuturePollIntervalMs);
    }
    return std::chrono::milliseconds(std::max(1, waitMs));
}

enum class TransferConsensusState : std::uint64_t
{
    kCompleted = 1,
    kFailed = 2,
    kTimedOut = 3,
};

struct TransferStateCounts
{
    int completedCount{0};
    int failedCount{0};
    int timedOutCount{0};
};

struct TransferConsensusOutcome
{
    std::unordered_set<RequestIdType> completedRequestIds;
    std::unordered_set<RequestIdType> failedRequestIds;
    std::unordered_set<RequestIdType> timedOutRequestIds;
};

template <typename CancelFn>
bool requestCancellationNoThrow(RequestIdType requestId, char const* transferKind, CancelFn&& cancelFn) noexcept
{
    try
    {
        return cancelFn();
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR(
            "%s cancellation for request %ld failed and will be retried: %s", transferKind, requestId, error.what());
    }
    catch (...)
    {
        TLLM_LOG_ERROR("%s cancellation for request %ld failed with an unknown error and will be retried", transferKind,
            requestId);
    }
    return false;
}

long getTransferElapsedMs(std::shared_ptr<LlmRequest> const& request, LlmRequest::TimePoint end)
{
    auto const elapsed
        = std::chrono::duration_cast<std::chrono::milliseconds>(end - request->getKvCacheTransferStart());
    return static_cast<long>(elapsed.count());
}

std::vector<RequestIdType> sortedRequestIds(std::unordered_set<RequestIdType> const& requestIds)
{
    std::vector<RequestIdType> result(requestIds.begin(), requestIds.end());
    std::sort(result.begin(), result.end());
    return result;
}

void appendPackedTransferState(
    std::vector<std::uint64_t>& packedStates, RequestIdType requestId, TransferConsensusState state)
{
    packedStates.push_back(requestId);
    packedStates.push_back(static_cast<std::uint64_t>(state));
}

std::vector<std::uint64_t> gatherPackedTransferStates(
    std::shared_ptr<CacheTransceiverComm> const& comm, std::vector<std::uint64_t> const& packedStates)
{
    int localSize = static_cast<int>(packedStates.size());
    std::vector<int> sizes(comm->getSize());
    std::vector<std::uint64_t> gatheredStates;
    if (useMPI())
    {
        comm->allgather(&localSize, sizes.data(), 1, mpi::MpiType::kINT32);
        std::vector<int> displs(comm->getSize());
        size_t totalSize = 0;
        for (int i = 0; i < comm->getSize(); i++)
        {
            displs[i] = static_cast<int>(totalSize);
            totalSize += sizes[i];
        }
        gatheredStates.resize(totalSize);
        comm->allgatherv(packedStates.data(), static_cast<int>(packedStates.size()), mpi::MpiType::kUINT64,
            gatheredStates.data(), sizes, displs, mpi::MpiType::kUINT64);
    }
    else
    {
        comm->allgather(&localSize, std::ref(sizes), {});
        size_t totalSize = std::accumulate(sizes.begin(), sizes.end(), 0);
        gatheredStates.resize(totalSize);
        comm->allgatherv(std::ref(packedStates), std::ref(gatheredStates), std::cref(sizes), {});
    }
    return gatheredStates;
}

TransferConsensusOutcome reduceTransferStates(std::shared_ptr<CacheTransceiverComm> const& comm,
    std::unordered_set<RequestIdType> const& completedRequestIds,
    std::unordered_set<RequestIdType> const& failedRequestIds,
    std::unordered_set<RequestIdType> const& timedOutRequestIds)
{
    std::vector<std::uint64_t> localStates;
    localStates.reserve((completedRequestIds.size() + failedRequestIds.size() + timedOutRequestIds.size()) * 2);
    for (auto const requestId : completedRequestIds)
    {
        if (failedRequestIds.find(requestId) == failedRequestIds.end())
        {
            appendPackedTransferState(localStates, requestId, TransferConsensusState::kCompleted);
        }
    }
    for (auto const requestId : failedRequestIds)
    {
        appendPackedTransferState(localStates, requestId, TransferConsensusState::kFailed);
    }
    for (auto const requestId : timedOutRequestIds)
    {
        appendPackedTransferState(localStates, requestId, TransferConsensusState::kTimedOut);
    }

    int const syncSize = (comm != nullptr) ? comm->getSize() : 1;
    auto const gatheredStates
        = ((comm != nullptr) && syncSize > 1) ? gatherPackedTransferStates(comm, localStates) : std::move(localStates);

    constexpr size_t kPackedStateFields = 2;
    TLLM_CHECK_WITH_INFO(gatheredStates.size() % kPackedStateFields == 0,
        "Packed transfer state consensus payload must contain request/state pairs.");

    std::unordered_map<RequestIdType, TransferStateCounts> stateCounts;
    for (size_t idx = 0; idx < gatheredStates.size(); idx += kPackedStateFields)
    {
        auto const requestId = gatheredStates.at(idx);
        auto const state = static_cast<TransferConsensusState>(gatheredStates.at(idx + 1));
        auto& counts = stateCounts[requestId];
        switch (state)
        {
        case TransferConsensusState::kCompleted: counts.completedCount++; break;
        case TransferConsensusState::kFailed: counts.failedCount++; break;
        case TransferConsensusState::kTimedOut: counts.timedOutCount++; break;
        }
    }

    TransferConsensusOutcome outcome;
    for (auto const& [requestId, counts] : stateCounts)
    {
        auto const terminalCount = counts.completedCount + counts.failedCount;
        if (counts.timedOutCount > 0)
        {
            outcome.timedOutRequestIds.insert(requestId);
        }
        if (terminalCount == syncSize && (counts.failedCount > 0 || counts.timedOutCount > 0))
        {
            outcome.failedRequestIds.insert(requestId);
        }
        else if (counts.completedCount == syncSize)
        {
            outcome.completedRequestIds.insert(requestId);
        }
    }
    return outcome;
}

TransferConsensusOutcome reduceTransferStates(std::shared_ptr<CacheTransceiverComm> const& firstComm,
    std::shared_ptr<CacheTransceiverComm> const& secondComm,
    std::unordered_set<RequestIdType> const& completedRequestIds,
    std::unordered_set<RequestIdType> const& failedRequestIds,
    std::unordered_set<RequestIdType> const& timedOutRequestIds)
{
    auto const firstOutcome
        = reduceTransferStates(firstComm, completedRequestIds, failedRequestIds, timedOutRequestIds);
    return reduceTransferStates(
        secondComm, firstOutcome.completedRequestIds, firstOutcome.failedRequestIds, firstOutcome.timedOutRequestIds);
}

void recordLocalTransferOutcome(RequestIdType requestId, std::shared_ptr<LlmRequest> request, bool failed,
    std::unordered_set<RequestIdType>& completedRequestIds, std::unordered_set<RequestIdType>& failedRequestIds,
    std::unordered_map<RequestIdType, std::shared_ptr<LlmRequest>>& requestsAwaitingConsensus)
{
    requestsAwaitingConsensus[requestId] = std::move(request);
    if (failed)
    {
        completedRequestIds.erase(requestId);
        failedRequestIds.insert(requestId);
    }
    else if (failedRequestIds.find(requestId) == failedRequestIds.end())
    {
        completedRequestIds.insert(requestId);
    }
}

void eraseLocalTransferOutcome(RequestIdType requestId, std::unordered_set<RequestIdType>& completedRequestIds,
    std::unordered_set<RequestIdType>& failedRequestIds,
    std::unordered_map<RequestIdType, std::shared_ptr<LlmRequest>>& requestsAwaitingConsensus)
{
    completedRequestIds.erase(requestId);
    failedRequestIds.erase(requestId);
    requestsAwaitingConsensus.erase(requestId);
}

} // namespace

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
    TLLM_CHECK_WITH_INFO(!common::getEnvDisaggEnableInflightCancel(),
        "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 is supported only by the PyExecutor C++ NIXL transceiver path; "
        "the legacy C++ executor does not provide the required deferred cleanup and poison escalation.");
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
    std::vector<SizeType32> const& rnnLayerNumPerPP)
    : CacheTransceiver(cacheManager, cacheStateModelCfg, worldConfig, attentionLayerNumPerPP, dataType, attentionType,
        cacheTransceiverConfig, rnnLayerNumPerPP, /*enableWorkerPublishedContextConsensus=*/false)
{
}

CacheTransceiver::CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
    executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
    std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
    executor::kv_cache::CacheState::AttentionType attentionType,
    std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig,
    std::vector<SizeType32> const& rnnLayerNumPerPP, bool const enableWorkerPublishedContextConsensus)
    : mCacheTransceiverConfig{cacheTransceiverConfig}
{
    using tensorrt_llm::batch_manager::kv_cache_manager::CacheFormatter;
    TLLM_CHECK_WITH_INFO(mCacheTransceiverConfig.has_value(), "CacheTransceiverConfig is not set.");
    auto const backendType = mCacheTransceiverConfig.value().getBackendType();
    TLLM_CHECK_WITH_INFO(
        backendType.has_value() && (backendType.value() != executor::CacheTransceiverConfig::BackendType::DEFAULT),
        " CacheTransceiverConfig::BackendType is not set.");
    if (common::getEnvDisaggEnableInflightCancel())
    {
        auto const nixlBackend = common::getEnvNixlBackend();
        TLLM_CHECK_WITH_INFO(
            backendType.value() == executor::CacheTransceiverConfig::BackendType::NIXL && nixlBackend == "UCX",
            "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 is experimental and currently supported only with the "
            "NIXL cache transceiver and the UCX NIXL backend.");
        TLLM_CHECK_WITH_INFO(mCacheTransceiverConfig->getKvTransferTimeoutMs().has_value(),
            "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 requires kv_transfer_timeout_ms to enforce a finite deadline.");
        TLLM_CHECK_WITH_INFO(!common::getEnvDisableKVCacheTransferOverlap(),
            "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 requires asynchronous KV cache transfer; "
            "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1 is not supported.");
        TLLM_CHECK_WITH_INFO(!common::getEnvDisaggLayerwise(),
            "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 does not support layer-wise KV cache transfer.");
        TLLM_CHECK_WITH_INFO(!common::getEnvTryZCopyForKVCacheTransfer(),
            "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 does not support zero-copy KV cache transfer because request "
            "blocks cannot be quarantined after an unquiesced cancellation.");
    }

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
    if (worldConfig.isPipelineParallel())
    {
        auto const ppGroupColor = worldConfig.getTensorParallelRank() * worldConfig.getContextParallelism()
            + worldConfig.getContextParallelRank();
        mGroupPipeParaComm
            = std::make_shared<CacheTransceiverComm>(mGroupComm->split(ppGroupColor, worldConfig.getRank()));
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

    if (useMPI() && worldConfig.getPipelineParallelism() > 1)
    {
        // This is a one-time startup agreement, not a scheduler-hot-path collective. Every MPI PP constructor
        // participates so a mixed binding/configuration selects the synchronous fallback on every rank.
        auto const threadSupport = mpi::getThreadSupport();
        auto const consensusConfig = [&]()
        {
            WorkerPublishedConsensusConfig config;
            config.enabledForCppBinding = enableWorkerPublishedContextConsensus;
            config.mpiControlPlane = true;
            config.nixlUcxBackend = backendType.value() == executor::CacheTransceiverConfig::BackendType::NIXL
                && common::getEnvNixlBackend() == "UCX";
            config.tensorParallelism = worldConfig.getTensorParallelism();
            config.contextParallelism = worldConfig.getContextParallelism();
            config.pipelineParallelism = worldConfig.getPipelineParallelism();
            config.attentionDp = mCacheState->getParallelConfig().mEnableAttentionDP;
            config.inflightCancellation = common::getEnvDisaggEnableInflightCancel();
            config.transferOverlap = !common::getEnvDisableKVCacheTransferOverlap();
            config.layerwiseTransfer = common::getEnvDisaggLayerwise();
            config.mpiThreadMultiple = threadSupport == mpi::MpiThreadSupport::THREAD_MULTIPLE;
            return config;
        }();
        constexpr std::uint64_t kWorkerPublishedConsensusVersion = 1;
        auto const localVersion
            = supportsWorkerPublishedConsensus(consensusConfig) ? kWorkerPublishedConsensusVersion : 0;
        TLLM_CHECK(mGroupPipeParaComm);
        std::vector<std::uint64_t> protocolVersions(static_cast<std::size_t>(mGroupPipeParaComm->getSize()));
        mGroupPipeParaComm->allgather(&localVersion, protocolVersions.data(), 1, mpi::MpiType::kUINT64);
        auto const useWorkerPublishedContextConsensus = selectWorkerPublishedConsensus(protocolVersions);
        if (useWorkerPublishedContextConsensus)
        {
            TLLM_LOG_INFO(
                "Enabled worker-published context-transfer consensus v%lu for the C++ NIXL/UCX TP1/CP1 "
                "pipeline-parallel path.",
                kWorkerPublishedConsensusVersion);
        }
        else if (std::any_of(protocolVersions.begin(), protocolVersions.end(),
                     [](std::uint64_t const version) { return version != 0; }))
        {
            TLLM_LOG_WARNING(
                "Pipeline ranks advertised different worker-published context-transfer consensus capabilities; "
                "retaining synchronous context-transfer consensus on every rank.");
        }

        // Construct the mailbox immediately after the all-rank agreement. If later transceiver initialization
        // throws, member unwinding still sends this rank's ordered close marker to its peers.
        if (useWorkerPublishedContextConsensus)
        {
            mContextTransferVoteMailbox = std::make_unique<ContextTransferVoteMailbox>(mGroupPipeParaComm);
        }
    }
    bool isMLA = attentionType == executor::kv_cache::CacheState::AttentionType::kMLA;
    std::optional<size_t> maxNumTokens = mCacheTransceiverConfig.value().getMaxTokensInBuffer();

    mCacheTransBufferManagers.push_back(
        std::make_unique<kv_cache_manager::CacheTransBufferManager>(cacheManager, maxNumTokens));
    if (isMLA && cacheManager->isEnableIndexerKCache())
    {
        mCacheTransBufferManagers.push_back(
            std::make_unique<kv_cache_manager::CacheTransBufferManager>(cacheManager, maxNumTokens, true));
    }

    // Unified pool path (CppMambaHybridCacheManager): build RnnModelConfig from
    // LinearAttentionMetadata. Detected by rnnLayerNumPerPP being non-empty.
    if (!rnnLayerNumPerPP.empty())
    {
        auto const& blockManager = cacheManager->getBlockManager();
        auto const& linearMeta = blockManager.getLinearAttentionMetadata();
        TLLM_CHECK_WITH_INFO(linearMeta.has_value(), "LinearAttentionMetadata not found for unified pool RNN config");

        executor::kv_cache::CacheState::RnnModelConfig rnnModelCfg{};
        rnnModelCfg.mNumHeads = linearMeta->rnnNumHeads;
        rnnModelCfg.mHeadDim = linearMeta->rnnHeadDim;
        rnnModelCfg.mDState = linearMeta->rnnDState;
        rnnModelCfg.mDConv = linearMeta->rnnDConv;
        rnnModelCfg.mNGroups = linearMeta->rnnNGroups;
        rnnModelCfg.mHiddenSize = linearMeta->rnnHeadDim * linearMeta->rnnNumHeads;
        rnnModelCfg.mConvSectionLayout = static_cast<executor::kv_cache::CacheState::RnnModelConfig::ConvSectionLayout>(
            linearMeta->rnnConvSectionLayout);

        // Derive actual SSM and conv dtypes from metadata byte sizes.
        // Pool dtype is UINT8 (raw byte storage), so we cannot use pool->getDataType().
        // Only the byte size matters for split/concat kernel stride calculations — the actual
        // dtype enum is not interpreted numerically, just used for getDTypeSize() dispatch.
        auto dtypeFromSize = [](SizeType32 size) -> nvinfer1::DataType
        {
            switch (size)
            {
            case 4: return nvinfer1::DataType::kFLOAT;
            case 2: return nvinfer1::DataType::kBF16;
            case 1: return nvinfer1::DataType::kFP8;
            default: TLLM_THROW("Unsupported RNN state dtype size: %d", size);
            }
        };
        TLLM_CHECK_WITH_INFO(linearMeta->rnnSsmDtypeSize > 0, "rnnSsmDtypeSize not set in LinearAttentionMetadata");
        TLLM_CHECK_WITH_INFO(linearMeta->rnnConvDtypeSize > 0, "rnnConvDtypeSize not set in LinearAttentionMetadata");
        nvinfer1::DataType ssmDtype = dtypeFromSize(linearMeta->rnnSsmDtypeSize);
        nvinfer1::DataType convDtype = dtypeFromSize(linearMeta->rnnConvDtypeSize);
        mCacheState->setRnnConfig(rnnModelCfg, rnnLayerNumPerPP, convDtype, ssmDtype);

        // Create RnnCacheTransBufferManager for unified pool path.
        mRnnCacheTransBufferManager
            = std::make_unique<rnn_state_manager::RnnCacheTransBufferManager>(cacheManager, *mCacheState, maxNumTokens);

        TLLM_LOG_INFO(
            "Unified pool RNN config: numHeads=%d, headDim=%d, dState=%d, dConv=%d, "
            "nGroups=%d, hiddenSize=%d, convSectionLayout=%d",
            rnnModelCfg.mNumHeads, rnnModelCfg.mHeadDim, rnnModelCfg.mDState, rnnModelCfg.mDConv, rnnModelCfg.mNGroups,
            rnnModelCfg.mHiddenSize, static_cast<int>(rnnModelCfg.mConvSectionLayout));
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

    auto makeRnnFormatter = [this, cacheManager]() -> std::unique_ptr<RnnCacheFormatter>
    {
        // Unified pool path (CppMambaHybridCacheManager)
        if (mCacheState->hasRnnConfig() && mRnnCacheTransBufferManager != nullptr)
        {
            return std::make_unique<RnnCacheFormatter>(cacheManager, mRnnCacheTransBufferManager.get());
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
    // Stop sender/receiver workers while the connection manager and transfer
    // plugin are still alive. The workers can access both during termination.
    mCacheSender.reset();
    if (mContextTransferVoteMailbox)
    {
        mContextTransferVoteMailbox->shutdown();
    }
    mCacheReceiver.reset();

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
    CacheSender::CompletionCallback completionCallback;
    if (mContextTransferVoteMailbox)
    {
        auto* mailbox = mContextTransferVoteMailbox.get();
        auto const requestId = llmRequest->mRequestId;
        completionCallback
            = [mailbox, requestId](bool const failed) { mailbox->publishOutcomeToPeers(requestId, failed); };
    }
    auto future = completionCallback ? mCacheSender->sendAsync(llmRequest, std::move(completionCallback))
                                     : mCacheSender->sendAsync(llmRequest);
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
    auto const requestId = llmRequest->mRequestId;
    auto const contextRequestId = llmRequest->getContextPhaseParams().value().getReqId();
    TLLM_LOG_DEBUG("Synchronous KV cache receive request %zu, context request %zu waiting for native completion.",
        requestId, contextRequestId);
    try
    {
        auto future = mCacheReceiver->receiveAsync(llmRequest);
        future.get();
    }
    catch (std::exception const& err)
    {
        llmRequest->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
        llmRequest->setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
        TLLM_LOG_ERROR("Synchronous KV cache receive request %zu, context request %zu failed: %s", requestId,
            contextRequestId, err.what());
        return;
    }
    catch (...)
    {
        llmRequest->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
        llmRequest->setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
        TLLM_LOG_ERROR("Synchronous KV cache receive request %zu, context request %zu failed with an unknown error",
            requestId, contextRequestId);
        return;
    }
    if (llmRequest->getState() == LlmRequestState::kDISAGG_TRANS_ERROR)
    {
        TLLM_LOG_ERROR("Synchronous KV cache receive request %zu, context request %zu completed with an error state.",
            requestId, contextRequestId);
        return;
    }
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
}

void CacheTransceiver::requestAndReceiveAsync(std::shared_ptr<LlmRequest> llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());

    auto const requestId = llmRequest->mRequestId;
    if (std::find_if(mRequesterFutures.begin(), mRequesterFutures.end(),
            [requestId](auto const& pair) { return pair.first->mRequestId == requestId; })
        != mRequesterFutures.end())
    {
        TLLM_LOG_WARNING("Request ID %zu is already in mRequestFutures.", requestId);
        return;
    }

    llmRequest->setKvCacheTransferStart(LlmRequest::getSteadyClockNow());
    auto future = mCacheReceiver->receiveAsync(llmRequest);
    auto* requestPtr = llmRequest.get();
    mRequesterFutures.emplace_back(std::move(llmRequest), std::move(future));
    requestPtr->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS);
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

namespace
{

class ContextTransferConsensusBackend
{
public:
    ContextTransferConsensusBackend(ContextTransferVoteMailbox* mailbox,
        std::shared_ptr<CacheTransceiverComm> const& intraStageComm,
        std::shared_ptr<CacheTransceiverComm> const& pipeStageComm)
        : mMailbox(mailbox)
        , mIntraStageComm(intraStageComm)
        , mPipeStageComm(pipeStageComm)
    {
    }

    [[nodiscard]] bool isWorkerPublished() const noexcept
    {
        return mMailbox != nullptr;
    }

    template <typename SenderFutures>
    [[nodiscard]] std::unordered_set<RequestIdType> selectReadyRequests(SenderFutures const& senderFutures) const
    {
        if (isWorkerPublished())
        {
            std::unordered_set<RequestIdType> readyRequestIds;
            for (auto const& [request, future] : senderFutures)
            {
                if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    readyRequestIds.insert(request->mRequestId);
                }
            }
            return readyRequestIds;
        }

        std::vector<RequestIdType> locallyReadyRequestIds;
        for (auto const& [request, future] : senderFutures)
        {
            if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
            {
                locallyReadyRequestIds.push_back(request->mRequestId);
            }
        }
        std::unordered_map<RequestIdType, int> frequencyMap;
        if (mIntraStageComm && mIntraStageComm->getSize() > 1)
        {
            for (auto const requestId : gatherRequestIds(mIntraStageComm, locallyReadyRequestIds))
            {
                ++frequencyMap[requestId];
            }
        }
        else
        {
            for (auto const requestId : locallyReadyRequestIds)
            {
                ++frequencyMap[requestId];
            }
        }

        auto const requiredReadyCount = mIntraStageComm ? mIntraStageComm->getSize() : 1;
        std::unordered_set<RequestIdType> readyRequestIds;
        for (auto const& [requestId, readyCount] : frequencyMap)
        {
            if (readyCount == requiredReadyCount)
            {
                readyRequestIds.insert(requestId);
            }
        }
        return readyRequestIds;
    }

    [[nodiscard]] bool resolveLocalFailure(RequestIdType const requestId, bool const observedFailure) const
    {
        if (!isWorkerPublished())
        {
            return observedFailure;
        }

        auto const publishedFailure = mMailbox->recordLocalOutcome(requestId);
        TLLM_CHECK_WITH_INFO(!observedFailure || publishedFailure,
            "Context-transfer future failed after its worker published a successful terminal vote.");
        return publishedFailure;
    }

    [[nodiscard]] TransferConsensusOutcome poll(std::unordered_set<RequestIdType> const& completedRequestIds,
        std::unordered_set<RequestIdType> const& failedRequestIds,
        std::unordered_set<RequestIdType> const& timedOutRequestIds) const
    {
        if (!isWorkerPublished())
        {
            return reduceTransferStates(
                mIntraStageComm, mPipeStageComm, completedRequestIds, failedRequestIds, timedOutRequestIds);
        }

        auto mailboxResult = mMailbox->poll();
        TransferConsensusOutcome outcome;
        outcome.completedRequestIds = std::move(mailboxResult.completedRequestIds);
        outcome.failedRequestIds = std::move(mailboxResult.failedRequestIds);
        return outcome;
    }

private:
    ContextTransferVoteMailbox* mMailbox;
    std::shared_ptr<CacheTransceiverComm> const& mIntraStageComm;
    std::shared_ptr<CacheTransceiverComm> const& mPipeStageComm;
};

std::unordered_set<RequestIdType> const& getEmptyRequestIdSet()
{
    static std::unordered_set<RequestIdType> const emptyRequestIds;
    return emptyRequestIds;
}

} // namespace

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
    std::optional<int> const& atLeastRequestNum, bool const markComplete)
{
    bool const blockAll = !atLeastRequestNum.has_value();
    bool const inflightCancelEnabled = common::getEnvDisaggEnableInflightCancel();
    TLLM_CHECK_WITH_INFO(!inflightCancelEnabled || !blockAll,
        "In-flight cancellation requires a finite context-transfer status poll; pass 0 for a nonblocking poll.");

    bool const needsProgress = atLeastRequestNum.value_or(0) > 0;
    std::optional<int> senderFutureTimeoutMs;
    std::optional<int> kvTransferTimeoutMs;
    if (mCacheTransceiverConfig.has_value())
    {
        senderFutureTimeoutMs = mCacheTransceiverConfig->getKvTransferSenderFutureTimeoutMs();
        kvTransferTimeoutMs = mCacheTransceiverConfig->getKvTransferTimeoutMs();
    }
    auto const futureWaitInterval = getTransferFutureWaitInterval(senderFutureTimeoutMs, needsProgress);

    auto const& intraStageComm
        = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupTPInDPComm : mGroupTensorParaComm;
    ContextTransferConsensusBackend consensusBackend{
        mContextTransferVoteMailbox.get(), intraStageComm, mGroupPipeParaComm};
    std::optional<std::chrono::steady_clock::time_point> progressDeadline;
    if (consensusBackend.isWorkerPublished() && needsProgress)
    {
        progressDeadline = std::chrono::steady_clock::now() + futureWaitInterval;
    }
    auto toProcess = consensusBackend.selectReadyRequests(mSenderFutures);

    // Preserve insertion order when a caller asks this poll to make progress on at least N requests.
    for (auto it = mSenderFutures.begin();
         atLeastRequestNum.value_or(0) > static_cast<int>(toProcess.size()) && it != mSenderFutures.end(); ++it)
    {
        toProcess.insert(it->first->mRequestId);
    }

    // Future polling, timeout accounting, and local outcome retention are shared by every consensus backend.
    for (auto it = mSenderFutures.begin(); it != mSenderFutures.end();)
    {
        auto& [request, future] = *it;
        auto const requestId = request->mRequestId;
        if (kvTransferTimeoutMs.has_value()
            && future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        {
            auto const elapsedMs = getTransferElapsedMs(request, LlmRequest::getSteadyClockNow());
            if (elapsedMs > kvTransferTimeoutMs.value() && mTimedOutSenderIds.insert(requestId).second)
            {
                TLLM_LOG_WARNING(
                    "Context KV cache transfer for request %ld exceeded configured timeout: "
                    "elapsed %ld ms > limit %d ms (%s).",
                    requestId, elapsedMs, kvTransferTimeoutMs.value(),
                    inflightCancelEnabled ? "requesting cancellation" : "observe-only");
            }
        }

        if (!blockAll && toProcess.find(requestId) == toProcess.end())
        {
            ++it;
            continue;
        }

        auto waitInterval = futureWaitInterval;
        if (progressDeadline.has_value())
        {
            // Worker-published status polling has one total budget across local future waits and mailbox progress.
            waitInterval = std::max(std::chrono::milliseconds(0),
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    progressDeadline.value() - std::chrono::steady_clock::now()));
        }
        auto const status = blockAll ? std::future_status::ready : future.wait_for(waitInterval);
        if (status == std::future_status::timeout)
        {
            TLLM_LOG_DEBUG(
                "Context KV cache transfer for request %ld is not ready after %ld ms wait slice; keeping it "
                "in progress.",
                requestId, static_cast<long>(waitInterval.count()));
            ++it;
            continue;
        }

        bool observedFailure = status != std::future_status::ready;
        if (!observedFailure)
        {
            try
            {
                future.get();
                if (kvTransferTimeoutMs.has_value())
                {
                    auto const elapsedMs = getTransferElapsedMs(request, request->getKvCacheTransferEnd());
                    if (elapsedMs > kvTransferTimeoutMs.value() && mTimedOutSenderIds.insert(requestId).second)
                    {
                        TLLM_LOG_WARNING(
                            "Context KV cache transfer for request %ld completed after its deadline: "
                            "elapsed %ld ms > limit %d ms (%s).",
                            requestId, elapsedMs, kvTransferTimeoutMs.value(),
                            inflightCancelEnabled ? "failing request" : "observe-only");
                    }
                }
            }
            catch (std::exception const& error)
            {
                TLLM_LOG_ERROR("Error occurred during context transfer for request %ld: %s", requestId, error.what());
                observedFailure = true;
            }
            catch (...)
            {
                TLLM_LOG_ERROR("Unknown error occurred during context transfer for request %ld", requestId);
                observedFailure = true;
            }
        }
        else
        {
            TLLM_LOG_ERROR("Future returned unexpected status for request %ld. Recording as failed.", requestId);
        }

        auto completedRequest = request;
        it = mSenderFutures.erase(it);
        // Erase the consumed future before a backend invariant can throw, so no no-state future remains pollable.
        auto const failed = consensusBackend.resolveLocalFailure(requestId, observedFailure);
        recordLocalTransferOutcome(requestId, std::move(completedRequest), failed, mCompletedSenderRequestIds,
            mFailedSenderRequestIds, mSenderRequestsAwaitingConsensus);
    }

    RequestStatuses requestsStatus;
    auto commitConsensusOutcome = [&](TransferConsensusOutcome const& outcome)
    {
        for (auto const requestId : sortedRequestIds(outcome.failedRequestIds))
        {
            auto const requestIt = mSenderRequestsAwaitingConsensus.find(requestId);
            if (requestIt == mSenderRequestsAwaitingConsensus.end())
            {
                TLLM_CHECK_WITH_INFO(!consensusBackend.isWorkerPublished(),
                    "Missing retained context request %zu for a globally failed worker-published transfer.",
                    static_cast<std::size_t>(requestId));
                continue;
            }
            requestIt->second->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            requestsStatus.errorRequestIds.insert(requestId);
            mTimedOutSenderIds.erase(requestId);
            mCancelRequestedSenderIds.erase(requestId);
            eraseLocalTransferOutcome(
                requestId, mCompletedSenderRequestIds, mFailedSenderRequestIds, mSenderRequestsAwaitingConsensus);
        }
        for (auto const requestId : sortedRequestIds(outcome.completedRequestIds))
        {
            auto const requestIt = mSenderRequestsAwaitingConsensus.find(requestId);
            if (requestIt == mSenderRequestsAwaitingConsensus.end())
            {
                TLLM_CHECK_WITH_INFO(!consensusBackend.isWorkerPublished(),
                    "Missing retained context request %zu for a globally completed worker-published transfer.",
                    static_cast<std::size_t>(requestId));
                continue;
            }
            requestsStatus.completedRequestIds.insert(requestId);
            if (markComplete)
            {
                requestIt->second->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
            }
            mTimedOutSenderIds.erase(requestId);
            mCancelRequestedSenderIds.erase(requestId);
            eraseLocalTransferOutcome(
                requestId, mCompletedSenderRequestIds, mFailedSenderRequestIds, mSenderRequestsAwaitingConsensus);
        }
    };

    auto requestTimedOutCancellations = [&](TransferConsensusOutcome const& outcome)
    {
        if (!inflightCancelEnabled)
        {
            return;
        }

        for (auto const requestId : outcome.timedOutRequestIds)
        {
            auto const futureIt = std::find_if(mSenderFutures.begin(), mSenderFutures.end(),
                [requestId](auto const& entry) { return entry.first->mRequestId == requestId; });
            if (futureIt == mSenderFutures.end()
                || futureIt->second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready
                || mCancelRequestedSenderIds.find(requestId) != mCancelRequestedSenderIds.end())
            {
                continue;
            }
            mTimedOutSenderIds.insert(requestId);
            if (requestCancellationNoThrow(
                    requestId, "Context", [&]() { return mCacheSender->cancelRequest(*futureIt->first); }))
            {
                mCancelRequestedSenderIds.insert(requestId);
            }
            else
            {
                TLLM_LOG_DEBUG("Context cancellation for request %ld was not accepted; will retry", requestId);
            }
        }
    };

    auto pollAndCommitConsensus = [&]()
    {
        auto const& timedOutRequestIds = inflightCancelEnabled ? mTimedOutSenderIds : getEmptyRequestIdSet();
        auto const outcome
            = consensusBackend.poll(mCompletedSenderRequestIds, mFailedSenderRequestIds, timedOutRequestIds);
        requestTimedOutCancellations(outcome);
        commitConsensusOutcome(outcome);
    };

    // Collective consensus must be entered exactly once per status call. The mailbox can make independent progress
    // and may be polled repeatedly without requiring a matching peer scheduler call.
    pollAndCommitConsensus();
    if (consensusBackend.isWorkerPublished())
    {
        auto const committedCount
            = [&]() { return requestsStatus.completedRequestIds.size() + requestsStatus.errorRequestIds.size(); };
        if (blockAll)
        {
            while (!mSenderRequestsAwaitingConsensus.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(kContextConsensusPollIntervalMs));
                pollAndCommitConsensus();
            }
        }
        else if (needsProgress)
        {
            while (committedCount() < static_cast<std::size_t>(atLeastRequestNum.value())
                && !mSenderRequestsAwaitingConsensus.empty()
                && std::chrono::steady_clock::now() < progressDeadline.value())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(kContextConsensusPollIntervalMs));
                pollAndCommitConsensus();
            }
        }
    }

    return requestsStatus;
}

void CacheTransceiver::checkGenTransferStatus(std::optional<int> const& atLeastRequestNum)
{
    bool const blockAll = !atLeastRequestNum.has_value();
    bool const inflightCancelEnabled = common::getEnvDisaggEnableInflightCancel();
    TLLM_CHECK_WITH_INFO(!inflightCancelEnabled || !blockAll,
        "In-flight cancellation requires a finite generation-transfer status poll; pass 0 for a nonblocking poll.");
    bool const needsProgress = atLeastRequestNum.value_or(0) > 0;
    std::optional<int> genTransferPollIntervalMs = std::nullopt;
    if (mCacheTransceiverConfig.has_value())
    {
        genTransferPollIntervalMs = mCacheTransceiverConfig->getKvTransferPollIntervalMs();
    }
    auto const futureWaitInterval = getTransferFutureWaitInterval(genTransferPollIntervalMs, needsProgress);

    std::vector<LlmRequest::RequestIdType> genTransferReadyRequestIds;
    auto collectReadyRequestIds = [&]()
    {
        genTransferReadyRequestIds.clear();
        for (auto&& [request, future] : mRequesterFutures)
        {
            if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
            {
                genTransferReadyRequestIds.push_back(request->mRequestId);
            }
        }
    };
    collectReadyRequestIds();
    if (needsProgress)
    {
        auto const deadline = std::chrono::steady_clock::now() + futureWaitInterval;
        while (static_cast<int>(genTransferReadyRequestIds.size()) < atLeastRequestNum.value()
            && std::chrono::steady_clock::now() < deadline)
        {
            auto const remaining
                = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            std::this_thread::sleep_for(std::min(std::chrono::milliseconds(kTransferFuturePollIntervalMs), remaining));
            collectReadyRequestIds();
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

    // Gen-side mirror of the context deadline/consensus path.
    std::optional<int> kvTransferTimeoutMs = std::nullopt;
    if (mCacheTransceiverConfig.has_value())
    {
        kvTransferTimeoutMs = mCacheTransceiverConfig->getKvTransferTimeoutMs();
    }
    for (auto it = mRequesterFutures.begin(); it != mRequesterFutures.end();)
    {
        auto& request = it->first;
        auto const requestId = request->mRequestId;
        if (kvTransferTimeoutMs.has_value()
            && it->second.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        {
            auto const elapsedMs = getTransferElapsedMs(request, LlmRequest::getSteadyClockNow());
            if (elapsedMs > kvTransferTimeoutMs.value())
            {
                if (mTimedOutRequesterIds.insert(requestId).second)
                {
                    TLLM_LOG_WARNING(
                        "Generation KV cache transfer for request %ld exceeded configured timeout: "
                        "elapsed %ld ms > limit %d ms (%s).",
                        requestId, elapsedMs, kvTransferTimeoutMs.value(),
                        inflightCancelEnabled ? "requesting cancellation" : "observe-only");
                }
            }
        }
        if (blockAll || toCompleteIdSet.find(requestId) != toCompleteIdSet.end())
        {
            try
            {
                auto const status = blockAll ? std::future_status::ready : it->second.wait_for(futureWaitInterval);
                if (status == std::future_status::ready)
                {
                    it->second.get();
                    if (kvTransferTimeoutMs.has_value())
                    {
                        auto const elapsedMs = getTransferElapsedMs(request, request->getKvCacheTransferEnd());
                        if (elapsedMs > kvTransferTimeoutMs.value() && mTimedOutRequesterIds.insert(requestId).second)
                        {
                            TLLM_LOG_WARNING(
                                "Generation KV cache transfer for request %ld completed after its deadline: "
                                "elapsed %ld ms > limit %d ms (%s).",
                                requestId, elapsedMs, kvTransferTimeoutMs.value(),
                                inflightCancelEnabled ? "failing request" : "observe-only");
                        }
                    }
                    recordLocalTransferOutcome(requestId, request, /*failed=*/false, mCompletedRequesterRequestIds,
                        mFailedRequesterRequestIds, mRequesterRequestsAwaitingConsensus);
                }
                else if (status == std::future_status::timeout)
                {
                    TLLM_LOG_DEBUG(
                        "Generation KV cache transfer for request %ld is not ready after %ld ms wait slice; keeping "
                        "it in progress.",
                        requestId, static_cast<long>(futureWaitInterval.count()));
                    ++it;
                    continue;
                }
                else
                {
                    TLLM_LOG_ERROR("Future returned unexpected status for request %ld. Marking as error.", requestId);
                    recordLocalTransferOutcome(requestId, request, /*failed=*/true, mCompletedRequesterRequestIds,
                        mFailedRequesterRequestIds, mRequesterRequestsAwaitingConsensus);
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Error occurred during generation transfer for request %ld: %s", requestId, e.what());
                recordLocalTransferOutcome(requestId, request, /*failed=*/true, mCompletedRequesterRequestIds,
                    mFailedRequesterRequestIds, mRequesterRequestsAwaitingConsensus);
            }
            catch (...)
            {
                TLLM_LOG_ERROR("Unknown error occurred during generation transfer for request %ld", requestId);
                recordLocalTransferOutcome(requestId, request, /*failed=*/true, mCompletedRequesterRequestIds,
                    mFailedRequesterRequestIds, mRequesterRequestsAwaitingConsensus);
            }
            if (useMPI())
            {
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    "**** it->first->mRequestId: %ld, context request ID: %ld ******** get feature ***", requestId,
                    request->getContextPhaseParams().value().getReqId());
            }
            else
            {
                TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                    "**** it->first->mRequestId: %ld, context request ID: %ld ******** get feature ***", requestId,
                    request->getContextPhaseParams().value().getReqId());
            }
            it = mRequesterFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }

    auto const consensusOutcome
        = reduceTransferStates(syncComm, mCompletedRequesterRequestIds, mFailedRequesterRequestIds,
            inflightCancelEnabled ? mTimedOutRequesterIds : std::unordered_set<RequestIdType>{});
    if (inflightCancelEnabled)
    {
        for (auto const requestId : consensusOutcome.timedOutRequestIds)
        {
            auto const futureIt = std::find_if(mRequesterFutures.begin(), mRequesterFutures.end(),
                [requestId](auto const& entry) { return entry.first->mRequestId == requestId; });
            if (futureIt == mRequesterFutures.end()
                || futureIt->second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready
                || mCancelRequestedRequesterIds.find(requestId) != mCancelRequestedRequesterIds.end())
            {
                continue;
            }
            mTimedOutRequesterIds.insert(requestId);
            if (requestCancellationNoThrow(
                    requestId, "Generation", [&]() { return mCacheReceiver->cancelRequest(*futureIt->first); }))
            {
                mCancelRequestedRequesterIds.insert(requestId);
            }
            else
            {
                TLLM_LOG_DEBUG("Generation cancellation for request %ld was not accepted; will retry", requestId);
            }
        }
    }
    for (auto const requestId : sortedRequestIds(consensusOutcome.failedRequestIds))
    {
        auto const requestIt = mRequesterRequestsAwaitingConsensus.find(requestId);
        if (requestIt == mRequesterRequestsAwaitingConsensus.end())
        {
            continue;
        }
        requestIt->second->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
        mTimedOutRequesterIds.erase(requestId);
        mCancelRequestedRequesterIds.erase(requestId);
        eraseLocalTransferOutcome(
            requestId, mCompletedRequesterRequestIds, mFailedRequesterRequestIds, mRequesterRequestsAwaitingConsensus);
    }
    for (auto const requestId : sortedRequestIds(consensusOutcome.completedRequestIds))
    {
        auto const requestIt = mRequesterRequestsAwaitingConsensus.find(requestId);
        if (requestIt == mRequesterRequestsAwaitingConsensus.end())
        {
            continue;
        }
        requestIt->second->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);

        // Gather the kv cache transfer time from all workers and update to leader rank.
        if (!common::getEnvKVCacheTimeOutputPath().empty())
        {
            updateKVCacheTransferBW(syncComm, requestIt->second.get());
        }
        mTimedOutRequesterIds.erase(requestId);
        mCancelRequestedRequesterIds.erase(requestId);
        eraseLocalTransferOutcome(
            requestId, mCompletedRequesterRequestIds, mFailedRequesterRequestIds, mRequesterRequestsAwaitingConsensus);
    }
}

bool CacheTransceiver::checkGenTransferComplete() const
{
    return mRequesterFutures.empty() && mCompletedRequesterRequestIds.empty() && mFailedRequesterRequestIds.empty();
}

bool CacheTransceiver::hasPoisonedTransferBuffer() const
{
    return std::any_of(mCacheTransBufferManagerPtrs.begin(), mCacheTransBufferManagerPtrs.end(),
        [](BaseTransBufferManager const* manager) { return manager != nullptr && manager->hasPoisonedBuffer(); });
}

bool CacheTransceiver::cancelRequest(std::shared_ptr<LlmRequest> llmRequest)
{
    if (llmRequest == nullptr)
    {
        TLLM_LOG_WARNING("Cannot cancel a null KV cache transfer request");
        return false;
    }
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
