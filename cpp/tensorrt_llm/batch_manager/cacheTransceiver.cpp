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
#include "tensorrt_llm/batch_manager/contextTransferCoordinator.h"
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
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

namespace
{

/// Generate a UUID-like hex string (e.g. "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
/// to uniquely identify a CacheTransceiver instance across gen instances.
std::string generateInstanceId()
{
    // The RNG state is comparatively expensive to construct/seed, so keep one
    // per thread instead of building it on every call.
    static thread_local std::mt19937_64 gen{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dis;
    uint64_t a = dis(gen);
    uint64_t b = dis(gen);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(8) << (a >> 32) << "-" << std::setw(4) << ((a >> 16) & 0xFFFF)
        << "-" << std::setw(4) << (a & 0xFFFF) << "-" << std::setw(4) << (b >> 48) << "-" << std::setw(12)
        << (b & 0xFFFFFFFFFFFF);
    return oss.str();
}

} // anonymous namespace

std::mutex CacheTransceiver::mDllMutex;

namespace
{

using RequestIdType = LlmRequest::RequestIdType;

constexpr int kTransferFuturePollIntervalMs = 10;

char const* cacheTransceiverBackendName(executor::CacheTransceiverConfig::BackendType backendType)
{
    using BackendType = executor::CacheTransceiverConfig::BackendType;
    switch (backendType)
    {
    case BackendType::DEFAULT: return "DEFAULT";
    case BackendType::MPI: return "MPI";
    case BackendType::UCX: return "UCX";
    case BackendType::NIXL: return "NIXL";
    case BackendType::MOONCAKE: return "MOONCAKE";
    }
    return "UNKNOWN";
}

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
    std::vector<SizeType32> const& attentionLayerNumPerPP, tensorrt_llm::DataType dataType,
    executor::kv_cache::CacheState::AttentionType attentionType,
    std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig,
    std::vector<SizeType32> const& rnnLayerNumPerPP)
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
    // globalSteadyClockOffset() reads a single process-global copy shared with
    // the nanobind module, so if the Python runtime already calibrated the offset
    // (PyExecutor::_set_global_steady_clock_offset) it is visible here and we skip;
    // the pure-C++ path performs the calibration below.
    // The check-and-set is guarded by a mutex so that CacheTransceiver instances
    // constructed concurrently in the same process (e.g. multi-engine serving) do
    // not race on the shared offset or issue mismatched collectives.
    {
        static std::mutex sSteadyClockCalibrationMutex;
        std::lock_guard<std::mutex> lock(sSteadyClockCalibrationMutex);
        if (!globalSteadyClockOffset().has_value())
        {
            using Duration = LlmRequest::Duration;
            // Synchronize all ranks immediately before sampling the local clock so
            // every rank measures from a consistent point.
            if (useMPI())
            {
                tensorrt_llm::mpi::MpiComm::session().barrier();
            }
            else
            {
                // CacheTransceiverComm exposes no barrier primitive, so use a cheap
                // allgather as a pseudo-barrier for the process-group path.
                int64_t const dummy = 0;
                std::vector<int64_t> dummyRecv(mGroupComm->getSize(), 0);
                mGroupComm->allgather(dummy, std::ref(dummyRecv), {});
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
            globalSteadyClockOffset() = Duration(offsetNs);

            TLLM_LOG_INFO(mGroupComm->getRank(),
                "CacheTransceiver: set global steady clock offset = %.6f sec for rank %d",
                static_cast<double>(offsetNs) / 1e9, mGroupComm->getRank());
        }
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
        auto dtypeFromSize = [](SizeType32 size) -> tensorrt_llm::DataType
        {
            switch (size)
            {
            case 4: return tensorrt_llm::DataType::kFLOAT;
            case 2: return tensorrt_llm::DataType::kBF16;
            case 1: return tensorrt_llm::DataType::kFP8;
            default: TLLM_THROW("Unsupported RNN state dtype size: %d", size);
            }
        };
        TLLM_CHECK_WITH_INFO(linearMeta->rnnSsmDtypeSize > 0, "rnnSsmDtypeSize not set in LinearAttentionMetadata");
        TLLM_CHECK_WITH_INFO(linearMeta->rnnConvDtypeSize > 0, "rnnConvDtypeSize not set in LinearAttentionMetadata");
        tensorrt_llm::DataType ssmDtype = dtypeFromSize(linearMeta->rnnSsmDtypeSize);
        tensorrt_llm::DataType convDtype = dtypeFromSize(linearMeta->rnnConvDtypeSize);
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

    mCacheSender
        = std::make_unique<CacheSender>(mManager.get(), worldConfig.getRank(), makeCacheTransferLayer(), mInstanceId);
    mCacheReceiver
        = std::make_unique<CacheReceiver>(mManager.get(), worldConfig.getRank(), makeCacheTransferLayer(), mInstanceId);

    // Keep automatic enablement within the currently qualified C++ NIXL/UCX TP1/CP1 pipeline topology.
    bool const coordinatorTopologyEligible = worldConfig.getPipelineParallelism() > 1 && useMPI()
        && backendType.value() == executor::CacheTransceiverConfig::BackendType::NIXL
        && common::getEnvNixlBackend() == "UCX" && worldConfig.getTensorParallelism() == 1
        && worldConfig.getContextParallelism() == 1 && !mCacheState->getParallelConfig().mEnableAttentionDP;
    if (worldConfig.getPipelineParallelism() > 1 && useMPI())
    {
        TLLM_CHECK(mGroupPipeParaComm != nullptr);
        constexpr std::uint64_t kCoordinatorProtocolVersion = 1;
        std::uint64_t const localVersion = coordinatorTopologyEligible ? kCoordinatorProtocolVersion : 0;
        bool const cancellationEnabled = common::getEnvDisaggEnableInflightCancel();
        std::uint64_t const localProtocolMode = (localVersion << 1) | static_cast<std::uint64_t>(cancellationEnabled);
        std::vector<std::uint64_t> protocolModes(static_cast<std::size_t>(mGroupPipeParaComm->getSize()));
        mGroupPipeParaComm->allgather(&localProtocolMode, protocolModes.data(), 1, mpi::MpiType::kUINT64);
        TLLM_CHECK_WITH_INFO(std::all_of(protocolModes.begin(), protocolModes.end(),
                                 [&](std::uint64_t const mode) { return mode == localProtocolMode; }),
            "Context-transfer consensus protocol version or cancellation mode differs across PP ranks.");
        if (localVersion != 0)
        {
            mContextTransferCoordinator = std::make_unique<ContextTransferCoordinator>(mGroupPipeParaComm);
            TLLM_LOG_INFO(
                "Enable asynchronous context-transfer consensus version %llu for PP group of size %d; in-flight "
                "cancellation=%s.",
                static_cast<unsigned long long>(kCoordinatorProtocolVersion), mGroupPipeParaComm->getSize(),
                cancellationEnabled ? "enabled" : "disabled");
        }
    }

    initializeCommState();
}

CacheTransceiver::~CacheTransceiver()
{
    // Stop sender/receiver workers while the connection manager and transfer
    // plugin are still alive. The workers can access both during termination.
    mCacheSender.reset();
    mCacheReceiver.reset();
    mContextTransferCoordinator.reset();

    if (mWrapperLibHandle)
    {
        std::lock_guard<std::mutex> lock(mDllMutex);
        dllClose(mWrapperLibHandle);
    }
}

std::string CacheTransceiver::getStatusDump() const
{
    auto const backendType = mCacheTransceiverConfig->getBackendType().value();
    StatusSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(mStatusSnapshotMutex);
        snapshot = mStatusSnapshot;
    }
    auto const requesterSyncActive = mSyncRequesterActive.load(std::memory_order_relaxed);
    std::ostringstream oss;
    oss << "KV cache transceiver | backend=" << cacheTransceiverBackendName(backendType)
        << " | TX(async_active=" << snapshot.senderAsyncActive << ", timed_out=" << snapshot.timedOutSenders
        << ", cancel_requested=" << snapshot.cancelingSenders << ", local_completed=" << snapshot.completedSenders
        << ", local_failed=" << snapshot.failedSenders << ", awaiting_consensus=" << snapshot.sendersAwaitingConsensus
        << ") | RX(async_active=" << snapshot.requesterAsyncActive << ", sync_active=" << requesterSyncActive
        << ", timed_out=" << snapshot.timedOutRequesters << ", cancel_requested=" << snapshot.cancelingRequesters
        << ", local_completed=" << snapshot.completedRequesters << ", local_failed=" << snapshot.failedRequesters
        << ", awaiting_consensus=" << snapshot.requestersAwaitingConsensus
        << ") | poisoned=" << (hasPoisonedTransferBuffer() ? "yes" : "no");
    return oss.str();
}

void CacheTransceiver::publishStatusSnapshot() noexcept
{
    StatusSnapshot snapshot;
    snapshot.senderAsyncActive = mSenderFutures.size();
    snapshot.requesterAsyncActive = mRequesterFutures.size();
    snapshot.timedOutSenders = mTimedOutSenderIds.size();
    snapshot.timedOutRequesters = mTimedOutRequesterIds.size();
    snapshot.cancelingSenders = mCancelRequestedSenderIds.size();
    snapshot.cancelingRequesters = mCancelRequestedRequesterIds.size();
    snapshot.completedSenders = mCompletedSenderRequestIds.size();
    snapshot.completedRequesters = mCompletedRequesterRequestIds.size();
    snapshot.failedSenders = mFailedSenderRequestIds.size();
    snapshot.failedRequesters = mFailedRequesterRequestIds.size();
    snapshot.sendersAwaitingConsensus = mSenderRequestsAwaitingConsensus.size();
    snapshot.requestersAwaitingConsensus = mRequesterRequestsAwaitingConsensus.size();
    try
    {
        std::lock_guard<std::mutex> lock(mStatusSnapshotMutex);
        mStatusSnapshot = snapshot;
    }
    catch (std::system_error const&)
    {
        // Status publication is best-effort and must never fail a transfer path.
    }
}

CacheTransceiver::SyncRequesterStatusGuard::SyncRequesterStatusGuard(CacheTransceiver& transceiver)
    : mTransceiver{transceiver}
{
    mTransceiver.mSyncRequesterActive.fetch_add(1, std::memory_order_relaxed);
}

CacheTransceiver::SyncRequesterStatusGuard::~SyncRequesterStatusGuard() noexcept
{
    mTransceiver.mSyncRequesterActive.fetch_sub(1, std::memory_order_relaxed);
}

void CacheTransceiver::initializeCommState()
{
    mCommState = std::addressof(mCacheSender->getCommState());
}

std::vector<char> CacheTransceiver::getSerializedDataTransceiverState() const
{
    TLLM_CHECK(mCommState != nullptr && mCacheState != nullptr);
    executor::DataTransceiverState state;
    state.setCommState(*mCommState);
    state.setCacheState(*mCacheState);
    // Only this API marks the state; context responses leave it unset.
    state.setIsArbitraryTransferState(true);
    return executor::Serialization::serialize(state);
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
    publishStatusSnapshot();
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
    publishStatusSnapshot();
}

void CacheTransceiver::requestAndReceiveSync(std::shared_ptr<LlmRequest> llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    auto const requestId = llmRequest->mRequestId;
    auto const contextRequestId = llmRequest->getContextPhaseParams().value().getReqId();
    TLLM_LOG_DEBUG("Synchronous KV cache receive request %zu, context request %zu waiting for native completion.",
        requestId, contextRequestId);
    SyncRequesterStatusGuard statusGuard{*this};
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
    publishStatusSnapshot();
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
        comm->allgatherv(sendBuffer.data(), sendSize, mpi::MpiType::kCHAR, recvBuffer.data(), recvCounts, displs,
            mpi::MpiType::kCHAR);
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
    bool const blockAll = !atLeastRequestNum.has_value();
    bool const inflightCancelEnabled = common::getEnvDisaggEnableInflightCancel();
    TLLM_CHECK_WITH_INFO(!inflightCancelEnabled || !blockAll,
        "In-flight cancellation requires a finite context-transfer status poll; pass 0 for a nonblocking poll.");
    std::optional<int> senderFutureTimeoutMs = std::nullopt;
    if (mCacheTransceiverConfig.has_value())
    {
        senderFutureTimeoutMs = mCacheTransceiverConfig->getKvTransferSenderFutureTimeoutMs();
    }
    bool const needsProgress = atLeastRequestNum.value_or(0) > 0;
    auto const futureWaitInterval = getTransferFutureWaitInterval(senderFutureTimeoutMs, needsProgress);
    // Without the opt-in flag, deadline checks remain observe-only. With the
    // flag, timeout IDs participate in the same topology consensus as terminal
    // outcomes and request cancellation is requested on every nonterminal rank.
    std::optional<int> kvTransferTimeoutMs = std::nullopt;
    if (mCacheTransceiverConfig.has_value())
    {
        kvTransferTimeoutMs = mCacheTransceiverConfig->getKvTransferTimeoutMs();
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
    publishStatusSnapshot();
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

    auto recordTimeout = [&](RequestIdType const requestId)
    {
        bool const inserted = mTimedOutSenderIds.insert(requestId).second;
        if (inserted && inflightCancelEnabled && mContextTransferCoordinator)
        {
            mContextTransferCoordinator->publishTimeout(requestId);
        }
        return inserted;
    };
    auto recordOutcome
        = [&](RequestIdType const requestId, std::shared_ptr<LlmRequest> const& request, bool const failed)
    {
        recordLocalTransferOutcome(requestId, request, failed, mCompletedSenderRequestIds, mFailedSenderRequestIds,
            mSenderRequestsAwaitingConsensus);
        if (mContextTransferCoordinator)
        {
            mContextTransferCoordinator->publishLocalOutcome(requestId, failed);
        }
    };

    // Record local terminal outcomes for requests selected this round. The
    // request is reported only after all ranks in the sync group agree that the
    // request reached a terminal state.
    for (auto it = mSenderFutures.begin(); it != mSenderFutures.end();)
    {
        auto& [request, future] = *it;
        auto const requestId = request->mRequestId;
        if (kvTransferTimeoutMs.has_value()
            && future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        {
            auto const elapsedMs = getTransferElapsedMs(request, LlmRequest::getSteadyClockNow());
            if (elapsedMs > kvTransferTimeoutMs.value())
            {
                if (recordTimeout(requestId))
                {
                    TLLM_LOG_WARNING(
                        "Context KV cache transfer for request %ld exceeded configured timeout: "
                        "elapsed %ld ms > limit %d ms (%s).",
                        requestId, elapsedMs, kvTransferTimeoutMs.value(),
                        inflightCancelEnabled ? "requesting cancellation" : "observe-only");
                }
            }
        }
        if (blockAll || (toCompleteIdSet.find(requestId) != toCompleteIdSet.end()))
        {
            bool terminal = false;
            bool failed = false;
            try
            {
                auto const status = blockAll ? std::future_status::ready : future.wait_for(futureWaitInterval);
                if (status == std::future_status::ready)
                {
                    future.get();
                    if (kvTransferTimeoutMs.has_value())
                    {
                        auto const elapsedMs = getTransferElapsedMs(request, request->getKvCacheTransferEnd());
                        if (elapsedMs > kvTransferTimeoutMs.value() && recordTimeout(requestId))
                        {
                            TLLM_LOG_WARNING(
                                "Context KV cache transfer for request %ld completed after its deadline: "
                                "elapsed %ld ms > limit %d ms (%s).",
                                requestId, elapsedMs, kvTransferTimeoutMs.value(),
                                inflightCancelEnabled ? "failing request" : "observe-only");
                        }
                    }
                    failed = request->getState() == LlmRequestState::kDISAGG_TRANS_ERROR;
                    terminal = true;
                }
                else if (status == std::future_status::timeout)
                {
                    TLLM_LOG_DEBUG(
                        "Context KV cache transfer for request %ld is not ready after %ld ms wait slice; keeping it "
                        "in progress.",
                        requestId, static_cast<long>(futureWaitInterval.count()));
                    ++it;
                }
                else
                {
                    TLLM_LOG_ERROR(
                        "Future returned unexpected status for request %ld. Recording as failed.", requestId);
                    failed = true;
                    terminal = true;
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Error occurred during context transfer for request %ld: %s", requestId, e.what());
                failed = true;
                terminal = true;
            }
            catch (...)
            {
                TLLM_LOG_ERROR("Unknown error occurred during context transfer for request %ld", requestId);
                failed = true;
                terminal = true;
            }
            if (terminal)
            {
                auto terminalRequest = request;
                it = mSenderFutures.erase(it);
                // Publish outside the transfer-future try/catch. A protocol error must not rewrite an immutable vote.
                recordOutcome(requestId, terminalRequest, failed);
            }
        }
        else
        {
            ++it;
        }
    }

    // Publish after local polling and before consensus, which may be the point at which a rank hangs.
    publishStatusSnapshot();
    RequestStatuses requestsStatus{};
    TransferConsensusOutcome consensusOutcome;
    if (mContextTransferCoordinator)
    {
        auto mergeCoordinatorOutcome = [&]()
        {
            auto coordinatorOutcome = mContextTransferCoordinator->poll();
            consensusOutcome.completedRequestIds.insert(
                coordinatorOutcome.completedRequestIds.begin(), coordinatorOutcome.completedRequestIds.end());
            consensusOutcome.failedRequestIds.insert(
                coordinatorOutcome.failedRequestIds.begin(), coordinatorOutcome.failedRequestIds.end());
            consensusOutcome.timedOutRequestIds.insert(
                coordinatorOutcome.timedOutRequestIds.begin(), coordinatorOutcome.timedOutRequestIds.end());
        };
        do
        {
            mergeCoordinatorOutcome();
            if (blockAll
                && consensusOutcome.completedRequestIds.size() + consensusOutcome.failedRequestIds.size()
                    < mSenderRequestsAwaitingConsensus.size())
            {
                std::this_thread::yield();
            }
        } while (blockAll
            && consensusOutcome.completedRequestIds.size() + consensusOutcome.failedRequestIds.size()
                < mSenderRequestsAwaitingConsensus.size());

        if (inflightCancelEnabled)
        {
            // A timeout update is transmitted once, but cancellation may be declined transiently. Keep the globally
            // observed timeout active so rank-local cancellation is retried on every poll until terminal commit.
            consensusOutcome.timedOutRequestIds.insert(mTimedOutSenderIds.begin(), mTimedOutSenderIds.end());
        }
    }
    else
    {
        consensusOutcome = reduceTransferStates(syncComm, mGroupPipeParaComm, mCompletedSenderRequestIds,
            mFailedSenderRequestIds, inflightCancelEnabled ? mTimedOutSenderIds : std::unordered_set<RequestIdType>{});
    }
    if (inflightCancelEnabled)
    {
        for (auto const requestId : consensusOutcome.timedOutRequestIds)
        {
            // Persist the global timeout even if this rank has not registered its local future yet. The one-shot
            // coordinator update must remain actionable when that future appears on a later scheduler poll.
            mTimedOutSenderIds.insert(requestId);
            auto const futureIt = std::find_if(mSenderFutures.begin(), mSenderFutures.end(),
                [requestId](auto const& entry) { return entry.first->mRequestId == requestId; });
            if (futureIt == mSenderFutures.end()
                || futureIt->second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready
                || mCancelRequestedSenderIds.find(requestId) != mCancelRequestedSenderIds.end())
            {
                continue;
            }
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
    }
    for (auto const requestId : sortedRequestIds(consensusOutcome.failedRequestIds))
    {
        auto const requestIt = mSenderRequestsAwaitingConsensus.find(requestId);
        if (requestIt == mSenderRequestsAwaitingConsensus.end())
        {
            continue;
        }
        requestIt->second->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
        requestsStatus.errorRequestIds.insert(requestId);
        mTimedOutSenderIds.erase(requestId);
        mCancelRequestedSenderIds.erase(requestId);
        eraseLocalTransferOutcome(
            requestId, mCompletedSenderRequestIds, mFailedSenderRequestIds, mSenderRequestsAwaitingConsensus);
    }
    for (auto const requestId : sortedRequestIds(consensusOutcome.completedRequestIds))
    {
        auto const requestIt = mSenderRequestsAwaitingConsensus.find(requestId);
        if (requestIt == mSenderRequestsAwaitingConsensus.end())
        {
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

    publishStatusSnapshot();
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
    publishStatusSnapshot();
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

    // Publish after local polling and before collectives, which may be the point at which a rank hangs.
    publishStatusSnapshot();
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

    // Collect consensus-completed requests so timing can be synced across ranks in a single
    // batched allgather (instead of one collective per request).
    std::vector<LlmRequest*> completedRequests;
    for (auto const requestId : sortedRequestIds(consensusOutcome.completedRequestIds))
    {
        auto const requestIt = mRequesterRequestsAwaitingConsensus.find(requestId);
        if (requestIt == mRequesterRequestsAwaitingConsensus.end())
        {
            continue;
        }
        requestIt->second->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
        completedRequests.push_back(requestIt->second.get());
        mTimedOutRequesterIds.erase(requestId);
        mCancelRequestedRequesterIds.erase(requestId);
        eraseLocalTransferOutcome(
            requestId, mCompletedRequesterRequestIds, mFailedRequesterRequestIds, mRequesterRequestsAwaitingConsensus);
    }
    publishStatusSnapshot();

    // Batch-sync timing across ranks in one allgather (instead of per-request), then write
    // the gen-side transfer summary CSV.
    if (!completedRequests.empty() && !common::getEnvKVCacheTimeOutputPath().empty())
    {
        batchUpdateKVCacheTransferBW(syncComm, completedRequests);
        writeGenTransferSummary(completedRequests);
    }
}

void CacheTransceiver::writeGenTransferSummary(std::vector<LlmRequest*> const& completedRequests)
{
    std::lock_guard<std::mutex> lock(mGenTransferSummaryMutex);
    if (!mGenTransferSummaryFile.is_open())
    {
        namespace fs = std::filesystem;
        auto outputPath = fs::path(common::getEnvKVCacheTimeOutputPath());
        fs::create_directories(outputPath);
        int rank = useMPI() ? mpi::MpiComm::world().getRank() : tensorrt_llm::pg_utils::get_world_pg()->getRank();
        auto filePath = outputPath / (mInstanceId + "_" + std::to_string(rank) + "_gen_transfer_summary.csv");
        mGenTransferSummaryFile.open(filePath);
        TLLM_CHECK_WITH_INFO(mGenTransferSummaryFile.is_open(), "Failed to open gen transfer summary file: %s",
            filePath.string().c_str());
        mGenTransferSummaryFile << "RequestID,gen_side_transfer_time(ms),kv_cache_size" << '\n';
    }
    for (auto* req : completedRequests)
    {
        auto reqId = req->getContextPhaseParams().value().getReqId();
        mGenTransferSummaryFile << reqId << "," << req->getKvCacheTransferTimeMS() << "," << req->getKvCacheSize()
                                << '\n';
    }
    mGenTransferSummaryFile << std::flush;
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
